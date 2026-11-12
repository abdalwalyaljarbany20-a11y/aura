import assert from 'node:assert/strict';
import { test } from 'node:test';
import { DEFAULT_CONFIG } from '../src/config/defaultConfig.js';
import { EventBus } from '../src/core/EventBus.js';
import { gestureEventError } from '../src/interaction/gestureEvents.js';
import { GRAB_ROTATE_GAIN, InteractionController } from '../src/interaction/InteractionController.js';

function make() {
  const bus = new EventBus();
  const log = [];
  bus.on('*', (e) => log.push(e));
  return { c: new InteractionController(DEFAULT_CONFIG.interaction, { bus }), log };
}

test('canonical gesture-event contract', () => {
  assert.equal(gestureEventError({ type: 'start', gesture: 'FIST', t: 1, confidence: 0.8 }), null);
  assert.equal(gestureEventError({ type: 'end', gesture: 'PINCH', durationMs: 300 }), null);
  assert.equal(gestureEventError({ type: 'event', gesture: 'SWIPE_LEFT', confidence: null }), null);
  for (const bad of [
    null, [], { type: 'begin', gesture: 'FIST' }, { type: 'start', gesture: 'SWIPE_LEFT' }, { type: 'event', gesture: 'FIST' },
    { type: 'start', gesture: 'UNKNOWN' }, { type: 'start', gesture: 'NONE' }, { type: 'start', gesture: 'MOVING' },
    { type: 'start', gesture: 'FIST', confidence: 2 }, { type: 'start', gesture: 'FIST', t: NaN },
  ]) assert.ok(gestureEventError(bad), JSON.stringify(bad));
});

test('invalid gesture events are rejected before they can change state', () => {
  const { c, log } = make();
  c.setHandPresent(true, 0);
  const r = c.handleGestureEvents([{ type: 'start', gesture: 'UNKNOWN' }, { type: 'start', gesture: 'SWIPE_LEFT' }, { gesture: 'FIST' }], 0);
  assert.deepEqual(r, []);
  assert.equal(c.state, 'TRACKING');
  assert.equal(c.counts.invalidEvents, 3);
  assert.equal(log.filter((e) => e.type === 'gesture_rejected').length, 3);
});

test('a hold start arriving after the hand is gone does not activate a hold', () => {
  const { c } = make();
  c.setHandPresent(true, 0);
  c.setHandPresent(false, 10); // newer tracking state: no hand
  const [r] = c.handleGestureEvents([{ type: 'start', gesture: 'PINCH' }], 20);
  assert.equal(r.reason, 'NO_HAND');
  assert.equal(c.state, 'IDLE');
});

test('every default binding maps a canonical gesture to an implemented command (deterministic)', () => {
  const { c } = make();
  const expected = {
    OPEN_PALM: 'WAVE', FIST: 'TOGGLE_LOCK', PINCH: 'GRAB_ROTATE', POINT: 'AIM', SWIPE_LEFT: 'SPIN_LEFT', SWIPE_RIGHT: 'SPIN_RIGHT',
    TWO_FINGERS: 'WALK', THREE_FINGERS: 'RUN', PINCH_TAP: 'SQUAT', FINGER_WALK: null, SPIN: 'SPIN', THUMB_UP: 'LAUGH', THUMB_DOWN: 'SAD',
  };
  assert.deepEqual(c.bindings, expected);
  for (const [gesture, command] of Object.entries(expected)) {
    if (command === null) continue; // finger stepping is detected but not bound
    const ctl = make().c;
    ctl.setHandPresent(true, 0);
    const type = gesture.startsWith('SWIPE') || gesture === 'PINCH_TAP' ? 'event' : 'start';
    const [r] = ctl.handleGestureEvents([{ type, gesture }], 0, { features: { valid: true, palmCenterNorm: { x: 0.5, y: 0.5 } } });
    assert.equal(r.name, command);
    assert.equal(r.accepted, true, gesture);
    const state = {
      WAVE: 'EXECUTING', SPIN_LEFT: 'EXECUTING', SPIN_RIGHT: 'EXECUTING', TOGGLE_LOCK: 'LOCKED', GRAB_ROTATE: 'CONTROL', AIM: 'CONTROL',
      WALK: 'CONTROL', RUN: 'CONTROL', SPIN: 'CONTROL', LAUGH: 'EXECUTING', SAD: 'EXECUTING', SQUAT: 'EXECUTING',
    }[command];
    assert.equal(ctl.state, state, gesture);
  }
});

test('GRAB_ROTATE changes the real robot pose; RESET_POSE returns it to 0', () => {
  const { c, log } = make();
  c.setHandPresent(true, 0);
  const f = (x) => ({ valid: true, palmCenterNorm: { x, y: 0.5 } });
  c.handleGestureEvents([{ type: 'start', gesture: 'PINCH' }], 0, { features: f(0.4) });
  c.updateHold(f(0.5));
  assert.ok(Math.abs(c.pose.yaw - 0.1 * GRAB_ROTATE_GAIN) < 1e-9);
  c.handleGestureEvents([{ type: 'end', gesture: 'PINCH' }], 100);
  c.updateHold(f(0.9)); // hold ended: no change
  const yaw = c.pose.yaw;
  assert.ok(Math.abs(yaw - 0.1 * GRAB_ROTATE_GAIN) < 1e-9);
  assert.equal(c.snapshot(100).lastResult.outcome, 'ended');
  assert.equal(c.submit({ name: 'RESET_POSE', source: 'keyboard' }, 200).accepted, true);
  c.tick(200 + 350);
  assert.ok(c.pose.yaw > 0 && c.pose.yaw < yaw, 'animating back');
  c.tick(200 + 700);
  assert.equal(c.pose.yaw, 0);
  assert.equal(c.state, 'TRACKING');
  const done = log.filter((e) => e.type === 'command' && e.status === 'completed').at(-1);
  assert.deepEqual(done.pose, { yaw: 0, x: 0, z: 0, lift: 0 });
  assert.equal(c.snapshot(900).lastResult.name, 'RESET_POSE');
});

test('pose does not change while locked', () => {
  const { c } = make();
  c.setHandPresent(true, 0);
  c.submit({ name: 'TOGGLE_LOCK', source: 'keyboard' }, 0);
  const [r] = c.handleGestureEvents([{ type: 'start', gesture: 'PINCH' }], 10, { features: { valid: true, palmCenterNorm: { x: 0.1, y: 0 } } });
  assert.equal(r.reason, 'LOCKED');
  c.updateHold({ valid: true, palmCenterNorm: { x: 0.9, y: 0 } });
  assert.equal(c.pose.yaw, 0);
});

test('error during a grab ends the hold and leaves no active interaction', () => {
  const { c } = make();
  c.setHandPresent(true, 0);
  c.handleGestureEvents([{ type: 'start', gesture: 'PINCH' }], 0, { features: { valid: true, palmCenterNorm: { x: 0.4, y: 0 } } });
  c.fail('engine disconnected', 50);
  const s = c.snapshot(50);
  assert.equal(s.state, 'ERROR');
  assert.equal(s.hold, null);
  assert.equal(s.action, null);
  c.updateHold({ valid: true, palmCenterNorm: { x: 0.9, y: 0 } });
  assert.equal(c.pose.yaw, 0);
});
