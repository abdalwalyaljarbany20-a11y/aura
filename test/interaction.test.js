import assert from 'node:assert/strict';
import { test } from 'node:test';
import { DEFAULT_CONFIG } from '../src/config/defaultConfig.js';
import { EventBus } from '../src/core/EventBus.js';
import { StateMachine } from '../src/core/StateMachine.js';
import { commandRequestError } from '../src/interaction/commands.js';
import { InteractionController } from '../src/interaction/InteractionController.js';
import { ROBOT_STATES, ROBOT_TRANSITIONS, createRobotStateMachine } from '../src/interaction/RobotStateMachine.js';

function make(overrides = {}) {
  const bus = new EventBus();
  const log = [];
  bus.on('*', (e) => log.push(e));
  const c = new InteractionController({ ...DEFAULT_CONFIG.interaction, ...overrides }, { bus });
  return { c, log };
}

// ------------------------------------------------------------ state machine

test('robot state machine accepts only declared transitions', () => {
  const fsm = createRobotStateMachine();
  assert.equal(fsm.state, 'IDLE');
  assert.equal(fsm.transition('CONTROL', 1), false, 'IDLE -> CONTROL needs a tracked hand first');
  assert.equal(fsm.state, 'IDLE');
  assert.equal(fsm.transition('TRACKING', 2), true);
  assert.equal(fsm.transition('CONTROL', 3), true);
  assert.equal(fsm.transition('EXECUTING', 4), false, 'CONTROL -> EXECUTING is a conflict');
  assert.equal(fsm.transition('ERROR', 5), true);
  assert.equal(fsm.transition('TRACKING', 6), false, 'ERROR only recovers to IDLE');
  assert.equal(fsm.transition('IDLE', 7), true);
});

test('every declared transition target is a declared state', () => {
  for (const targets of Object.values(ROBOT_TRANSITIONS)) for (const t of targets) assert.ok(ROBOT_STATES.includes(t));
  assert.throws(() => new StateMachine({ A: ['B'] }, 'A'), /undeclared/);
  assert.throws(() => new StateMachine({ A: [] }, 'Z'), /initial/);
});

// ------------------------------------------------------------ controller

test('hand presence drives IDLE <-> TRACKING', () => {
  const { c } = make();
  c.setHandPresent(true, 0);
  assert.equal(c.state, 'TRACKING');
  c.setHandPresent(false, 10);
  assert.equal(c.state, 'IDLE');
});

test('gesture start triggers its bound action; it completes after its duration', () => {
  const { c, log } = make();
  c.setHandPresent(true, 0);
  const [r] = c.handleGestureEvents([{ type: 'start', gesture: 'OPEN_PALM', t: 100 }], 100);
  assert.equal(r.accepted, true);
  assert.equal(r.name, 'WAVE');
  assert.equal(c.state, 'EXECUTING');
  assert.ok(c.snapshot(1000).action.progress > 0.4);
  c.tick(1000);
  assert.equal(c.state, 'EXECUTING');
  c.tick(100 + 1800);
  assert.equal(c.state, 'TRACKING');
  assert.ok(log.some((e) => e.type === 'command' && e.status === 'completed' && e.name === 'WAVE'));
});

test('action finishing after the hand left returns to IDLE', () => {
  const { c } = make();
  c.setHandPresent(true, 0);
  c.submit({ name: 'SPIN_LEFT', source: 'gesture' }, 0);
  c.setHandPresent(false, 100);
  assert.equal(c.state, 'EXECUTING');
  c.tick(2000);
  assert.equal(c.state, 'IDLE');
});

test('rejects unsupported, malformed and conflicting commands with explicit reasons', () => {
  const { c, log } = make();
  c.setHandPresent(true, 0);
  assert.equal(c.submit({ name: 'LAUNCH_ROCKET', source: 'agent' }, 0).reason, 'UNSUPPORTED');
  assert.equal(c.submit({ name: 'wave', source: 'gesture' }, 0).reason, 'INVALID');
  assert.equal(c.submit({ name: 'WAVE', source: 'internet' }, 0).reason, 'INVALID');
  assert.equal(c.submit({ name: 'WAVE', source: 'gesture', params: [] }, 0).reason, 'INVALID');
  assert.equal(c.submit(null, 0).reason, 'INVALID');
  assert.equal(c.submit({ name: 'WAVE', source: 'gesture' }, 0).accepted, true);
  assert.equal(c.submit({ name: 'SPIN_LEFT', source: 'gesture' }, 10).reason, 'BUSY');
  const rejected = log.filter((e) => e.type === 'command' && e.status === 'rejected');
  assert.equal(rejected.length, 6);
  assert.equal(c.state, 'EXECUTING', 'rejections do not change state');
});

test('unbound gestures are ignored; bindings to unknown commands are rejected as UNSUPPORTED', () => {
  const { c } = make({ bindings: { ...DEFAULT_CONFIG.interaction.bindings, OPEN_PALM: 'OPEN_BROWSER', FIST: null } });
  c.setHandPresent(true, 0);
  assert.deepEqual(c.handleGestureEvents([{ type: 'start', gesture: 'FIST' }], 0), []);
  const [r] = c.handleGestureEvents([{ type: 'start', gesture: 'OPEN_PALM' }], 0);
  assert.equal(r.reason, 'UNSUPPORTED');
  assert.equal(c.state, 'TRACKING');
});

test('repeated activation of the same command is suppressed by the cooldown', () => {
  const { c } = make({ cooldownMs: 900 });
  c.setHandPresent(true, 0);
  assert.equal(c.submit({ name: 'RESET_POSE', source: 'keyboard' }, 0).accepted, true);
  c.tick(700);
  assert.equal(c.submit({ name: 'RESET_POSE', source: 'keyboard' }, 750).reason, 'COOLDOWN');
  assert.equal(c.submit({ name: 'RESET_POSE', source: 'keyboard' }, 950).accepted, true);
});

test('hold commands: start on gesture start, end on gesture end or hand loss, block actions', () => {
  const { c } = make();
  c.setHandPresent(true, 0);
  const features = { valid: true, palmCenterNorm: { x: 0.4, y: 0.5 } };
  const [r] = c.handleGestureEvents([{ type: 'start', gesture: 'PINCH' }], 10, { features });
  assert.equal(r.name, 'GRAB_ROTATE');
  assert.equal(c.state, 'CONTROL');
  assert.deepEqual(c.snapshot(10).hold.params.anchor, { x: 0.4, y: 0.5 });
  assert.equal(c.submit({ name: 'WAVE', source: 'keyboard' }, 20).reason, 'CONFLICT');
  c.handleGestureEvents([{ type: 'end', gesture: 'PINCH' }], 30);
  assert.equal(c.state, 'TRACKING');
  assert.equal(c.snapshot(30).hold, null);

  c.handleGestureEvents([{ type: 'start', gesture: 'POINT' }], 40, { features });
  assert.equal(c.state, 'CONTROL');
  c.setHandPresent(false, 50);
  assert.equal(c.state, 'IDLE');
  assert.equal(c.snapshot(50).hold, null);
});

test('hold commands need a tracked hand', () => {
  const { c } = make();
  assert.equal(c.submit({ name: 'AIM', source: 'keyboard' }, 0).reason, 'NO_HAND');
});

test('lock: only TOGGLE_LOCK is accepted while locked; unlock restores the resting state', () => {
  const { c } = make();
  c.setHandPresent(true, 0);
  assert.equal(c.handleGestureEvents([{ type: 'start', gesture: 'FIST' }], 0)[0].accepted, true);
  assert.equal(c.state, 'LOCKED');
  assert.equal(c.submit({ name: 'WAVE', source: 'gesture' }, 10).reason, 'LOCKED');
  assert.equal(c.submit({ name: 'GRAB_ROTATE', source: 'gesture' }, 10).reason, 'LOCKED');
  c.setHandPresent(false, 20);
  assert.equal(c.state, 'LOCKED', 'lock persists without a hand');
  c.setHandPresent(true, 30);
  assert.equal(c.submit({ name: 'TOGGLE_LOCK', source: 'gesture' }, 2000).accepted, true);
  assert.equal(c.state, 'TRACKING');
});

test('ERROR rejects everything until recover()', () => {
  const { c } = make();
  c.setHandPresent(true, 0);
  c.submit({ name: 'AIM', source: 'gesture' }, 0);
  c.fail('camera lost', 10);
  assert.equal(c.state, 'ERROR');
  assert.equal(c.snapshot(10).hold, null);
  assert.equal(c.submit({ name: 'WAVE', source: 'keyboard' }, 20).reason, 'ERROR_STATE');
  c.setHandPresent(false, 25);
  assert.equal(c.state, 'ERROR', 'hand presence does not clear an error');
  assert.equal(c.recover(30), true);
  assert.equal(c.state, 'IDLE');
  assert.equal(c.submit({ name: 'WAVE', source: 'keyboard' }, 40).accepted, true);
});

test('commandRequestError validates untrusted requests', () => {
  assert.equal(commandRequestError({ name: 'WAVE', source: 'agent', params: { a: 1 } }), null);
  assert.ok(commandRequestError({ name: 'WAVE; rm -rf', source: 'agent' }));
  assert.ok(commandRequestError({ name: 'X'.repeat(100), source: 'agent' }));
  assert.ok(commandRequestError('WAVE'));
});
