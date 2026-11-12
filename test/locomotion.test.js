// Embodied interaction: face gaze, continuous spin, cadence walking, emotions.
// Synthetic landmark trajectories and synthetic face boxes only (no camera, no network, no API key).
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { AGENT_ROBOT_COMMANDS } from '../src/agent/agentTools.js';
import { DEFAULT_CONFIG } from '../src/config/defaultConfig.js';
import { EventBus } from '../src/core/EventBus.js';
import { FeatureExtractionEngine } from '../src/gestures/FeatureExtractionEngine.js';
import { GestureRecognitionEngine } from '../src/gestures/GestureRecognitionEngine.js';
import { SpinDetector } from '../src/gestures/SpinDetector.js';
import { StaticGestureClassifier } from '../src/gestures/StaticGestureClassifier.js';
import { AttentionController } from '../src/interaction/AttentionController.js';
import { COMMANDS } from '../src/interaction/commands.js';
import { InteractionController, spinRateFor, walkSpeedFor } from '../src/interaction/InteractionController.js';
import { resultError } from '../src/remote/PythonEngineClient.js';
import { FaceTrackingEngine, parseFaces } from '../src/tracking/FaceTrackingEngine.js';
import { makeHand, POSES, stepPose, THUMB_VIEW } from './fixtures/hands.js';

const ASPECT = 16 / 9;
const DEG = Math.PI / 180;
const gcfg = () => JSON.parse(JSON.stringify(DEFAULT_CONFIG.gestures));
const fx = new FeatureExtractionEngine({ mirror: true });
const feats = (pose, view = {}) => fx.extract(makeHand(pose, { aspect: ASPECT, ...view }), { aspect: ASPECT });
/** Display-space point (aspect-corrected, mirrored, y down) -> makeHand image centre. */
const viewAt = (xd, yd) => ({ cx: 1 - xd / ASPECT, cy: yd });
const circleView = (t, revPerS, dir = 1) => {
  const a = dir * 2 * Math.PI * revPerS * (t / 1000);
  return viewAt(0.9 + 0.12 * Math.cos(a), 0.65 + 0.12 * Math.sin(a));
};

function controller(icfg = DEFAULT_CONFIG.interaction) {
  const bus = new EventBus();
  const log = [];
  bus.on('*', (e) => log.push(e));
  const c = new InteractionController(icfg, { bus });
  return { c, log };
}
/** Optional cadence mode (config): finger stepping drives WALK, the static V does nothing. */
const CADENCE = { ...DEFAULT_CONFIG.interaction, bindings: { ...DEFAULT_CONFIG.interaction.bindings, FINGER_WALK: 'WALK', TWO_FINGERS: null, THREE_FINGERS: null } };
const cadenceCtl = () => controller(CADENCE);

/** Run the real engine + controller over generated frames. frame(t) -> features|null. */
function drive(frame, { from = 0, to = 3000, dt = 33, eng = new GestureRecognitionEngine(gcfg()), ctl = controller(), onFrame } = {}) {
  const events = [];
  const out = [];
  for (let t = from; t <= to; t += dt) {
    const f = frame(t);
    const r = eng.update(f, t);
    ctl.c.setHandPresent(!!f?.valid, t);
    ctl.c.handleGestureEvents(r.events, t, { features: f });
    ctl.c.updateHold(f, r.motion);
    ctl.c.tick(t);
    events.push(...r.events.map((e) => ({ ...e, t })));
    out.push({ t, motion: r.motion, rate: ctl.c.spinRate, speed: ctl.c.walkSpeed, state: ctl.c.state, mv: ctl.c.movement().state });
    onFrame?.(t, r, ctl);
  }
  return { events, out, eng, ctl };
}
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const kinds = (events, g) => events.filter((e) => e.gesture === g).map((e) => e.type);

// ================================================================ face gaze

test('gaze: a visible face becomes the attention target; angles point at it and stay bounded', () => {
  const a = new AttentionController(DEFAULT_CONFIG.interaction.attention);
  const head = { x: 0.75, y: 0.3 };
  let g;
  for (let t = 0; t <= 2000; t += 16) {
    if (t % 112 === 0) a.observeFaces(t, [{ x: 0.35, y: 0.32, w: 0.12, h: 0.2, score: 0.95 }]); // ~detector rate
    g = a.update(t, { head });
  }
  assert.equal(g.mode, 'USER');
  assert.ok(g.yaw < 0, 'face left of the robot on screen -> head turns to that side');
  assert.ok(Math.abs(g.yaw - (0.35 - 0.75) * 1.8) < 0.02, `converged yaw ${g.yaw}`);
  // an extreme position is clamped
  for (let t = 2016; t <= 5000; t += 16) {
    if (t % 112 === 0) a.observeFaces(t, [{ x: 0.0, y: 1.0, w: 0.1, h: 0.2, score: 0.9 }]); // a jump: a different face after the switch delay
    g = a.update(t, { head });
  }
  const cfg = DEFAULT_CONFIG.interaction.attention;
  assert.ok(Math.abs(g.yaw) <= cfg.maxYawDeg * DEG + 1e-9 && g.pitch <= cfg.maxPitchDownDeg * DEG + 1e-9);
});

test('gaze smoothing is time-based and never snaps; it is frame-rate independent', () => {
  const at = (fps) => {
    const a = new AttentionController(DEFAULT_CONFIG.interaction.attention);
    a.observeFaces(0, [{ x: 0.2, y: 0.3, w: 0.1, h: 0.2, score: 0.9 }]);
    let g;
    for (let t = 0; t <= 150; t += 1000 / fps) g = a.update(t, { head: { x: 0.75, y: 0.3 } });
    return g.yaw;
  };
  const a = new AttentionController(DEFAULT_CONFIG.interaction.attention);
  a.observeFaces(0, [{ x: 0.2, y: 0.3, w: 0.1, h: 0.2, score: 0.9 }]);
  a.update(0, { head: { x: 0.75, y: 0.3 } });
  const first = a.update(16, { head: { x: 0.75, y: 0.3 } }).yaw;
  assert.ok(Math.abs(first) <= DEFAULT_CONFIG.interaction.attention.maxSpeedDegS * DEG * 0.016 + 1e-9, 'one frame moves at most the speed cap');
  assert.ok(Math.abs(at(30) - at(144)) < 0.03, `${at(30)} vs ${at(144)}`);
});

test('the hand does not capture the gaze; aiming does, briefly, and the gaze returns to the face', () => {
  const a = new AttentionController(DEFAULT_CONFIG.interaction.attention);
  const head = { x: 0.75, y: 0.3 };
  let g;
  for (let t = 0; t <= 1000; t += 16) {
    a.observeFaces(t, [{ x: 0.5, y: 0.3, w: 0.1, h: 0.2, score: 0.9 }]);
    g = a.update(t, { head, hold: null, hand: { x: 0.1 + (t / 1000) * 0.3, y: 0.8 } }); // moving hand, no aiming
  }
  assert.equal(g.mode, 'USER');
  // default: the robot always looks at the user, even while aiming (the arm still aims)
  for (const hold of ['GRAB_ROTATE', 'SPIN', 'WALK', 'RUN', 'AIM']) assert.equal(a.update(1100, { head, hold, hand: { x: 0.1, y: 0.8 } }).mode, 'USER', hold);
  // configurable: interaction.attention.handAttentionHolds = ['AIM'] looks at the hand while aiming
  const b = new AttentionController({ ...DEFAULT_CONFIG.interaction.attention, handAttentionHolds: ['AIM'] });
  b.observeFaces(1000, [{ x: 0.5, y: 0.3, w: 0.1, h: 0.2, score: 0.9 }]);
  assert.equal(b.update(1200, { head, hold: 'AIM', hand: { x: 0.1, y: 0.8 } }).mode, 'HAND');
  assert.equal(b.update(1300, { head, hold: null, hand: { x: 0.1, y: 0.8 } }).mode, 'USER', 'back to the face when the gesture ends');
});

test('face loss: hold the last gaze briefly, then return smoothly to neutral; hand control is unaffected', () => {
  const a = new AttentionController(DEFAULT_CONFIG.interaction.attention);
  const head = { x: 0.75, y: 0.3 };
  for (let t = 0; t <= 1000; t += 16) {
    a.observeFaces(t, [{ x: 0.3, y: 0.3, w: 0.1, h: 0.2, score: 0.9 }]);
    a.update(t, { head });
  }
  for (let t = 1000; t <= 1500; t += 120) a.observeFaces(t, []); // detections keep running, no face
  assert.equal(a.update(1500, { head }).mode, 'USER', 'within faceHoldMs');
  let g;
  for (let t = 1516; t <= 3500; t += 16) g = a.update(t, { head });
  assert.equal(g.mode, 'NEUTRAL');
  assert.ok(Math.abs(g.yaw) < 0.01 && Math.abs(g.pitch) < 0.01, 'returned to straight ahead');
  // the controller knows nothing about faces: a walk is not affected by face loss
  const { c } = controller();
  c.setHandPresent(true, 0);
  c.handleGestureEvents([{ type: 'start', gesture: 'TWO_FINGERS' }], 0, { features: { valid: true, palmCenterNorm: { x: 0.5, y: 0.5 } } });
  assert.equal(c.state, 'CONTROL');
});

test('invalid face coordinates and scores are ignored; output is always finite and bounded', () => {
  const a = new AttentionController(DEFAULT_CONFIG.interaction.attention);
  const bad = [{ x: NaN, y: 0.3, w: 0.1, h: 0.1, score: 0.9 }, { x: 1e9, y: 0.3, w: 0.1, h: 0.1, score: 0.9 }, { x: 0.3, y: 0.3, w: 0.1, h: 0.1, score: 0.2 },
    { x: 0.3, y: 0.3, w: -1, h: 0.1, score: 0.9 }, null, 'x', { x: 0.4, y: -3, w: 0.1, h: 0.1, score: 0.9 }];
  a.observeFaces(0, bad);
  let g = a.update(0, { head: { x: 0.75, y: 0.3 } });
  for (let t = 16; t < 2000; t += 16) g = a.update(t, { head: { x: NaN, y: Infinity }, hold: 'AIM', hand: { x: NaN, y: 0 } });
  assert.equal(g.mode, 'NEUTRAL');
  assert.ok(Number.isFinite(g.yaw) && Number.isFinite(g.pitch) && Math.abs(g.yaw) < 1e-9);
  assert.equal(a.faceCount, 0);
});

test('multiple faces: a consistent primary; no switching every frame; switch only after the primary is gone', () => {
  const a = new AttentionController(DEFAULT_CONFIG.interaction.attention);
  const A = { x: 0.3, y: 0.3, w: 0.12, h: 0.2, score: 0.9 };
  const B = { x: 0.7, y: 0.35, w: 0.2, h: 0.3, score: 0.95 }; // larger
  a.observeFaces(0, [A]);
  for (let t = 120; t <= 2400; t += 120) {
    a.observeFaces(t, t % 240 ? [A, B] : [B, A]); // order alternates, B is bigger
    assert.ok(Math.abs(a.primary.x - 0.3) < 1e-9, `kept A at ${t}`);
  }
  a.observeFaces(2500, [B]); // A missing briefly: still A
  assert.ok(Math.abs(a.primary.x - 0.3) < 1e-9);
  a.observeFaces(3100, [B]); // A gone > switchAfterMs
  assert.ok(Math.abs(a.primary.x - 0.7) < 1e-9);
});

test('face detector: throttled on the existing frames, validated output, failure only disables gaze', async () => {
  let calls = 0;
  const backend = {
    async init() { return { delegate: 'CPU' }; },
    detect() {
      calls++;
      return { detections: [{ boundingBox: { originX: 100, originY: 50, width: 200, height: 250 }, categories: [{ score: 0.9 }] },
        { boundingBox: { originX: NaN, originY: 0, width: 1, height: 1 }, categories: [{ score: 0.9 }] }] };
    },
  };
  const ft = new FaceTrackingEngine({ backend, intervalMs: 120, minScore: 0.6 });
  await ft.init();
  const size = { width: 1000, height: 500 };
  const results = [];
  for (let t = 0; t <= 1000; t += 33) results.push(ft.detect({}, t, size));
  assert.equal(calls, results.filter(Boolean).length);
  assert.ok(calls >= 8 && calls <= 9, `${calls} detections in 1 s at 30 fps (throttled to ~8 Hz)`);
  assert.deepEqual(results.find(Boolean), [{ x: 0.2, y: 0.35, w: 0.2, h: 0.5, score: 0.9 }]);
  const broken = new FaceTrackingEngine({ backend: { async init() { throw new Error('face model not found'); } }, intervalMs: 120, minScore: 0.6 });
  assert.equal((await broken.init()).ok, false);
  assert.equal(broken.detect({}, 0, size), null);
  assert.deepEqual(parseFaces({ detections: [{ boundingBox: { originX: 0, originY: 0, width: 10, height: 10 }, categories: [{ score: 0.3 }] }] }, size, 0.6), []);
  // python results may carry faces: validated, hands unaffected
  const base = { status: 'NOT_DETECTED', session: 1, frameId: 1, hands: [], primary: null, gesture: { stable: 'NONE', label: 'NONE' }, events: [] };
  assert.equal(resultError({ ...base, faces: [{ x: 0.5, y: 0.4, w: 0.2, h: 0.3, score: 0.9 }] }), null);
  assert.match(resultError({ ...base, faces: [{ x: 'a' }] }), /bad face/);
});

// ================================================================ continuous spin

test('spin: slow / medium / fast circles give correctly ordered finger rates and robot rates', () => {
  const runAt = (rev) => drive((t) => feats(POSES.POINT, circleView(t, rev)), { to: 3000 }).out.filter((o) => o.t > 1800);
  const r = [0.5, 1, 2.5].map((rev) => ({ rev, finger: mean(runAt(rev).map((o) => o.motion.spin.omega)), robot: mean(runAt(rev).map((o) => o.rate)) }));
  for (const x of r) assert.ok(Math.abs(x.finger - 2 * Math.PI * x.rev) / (2 * Math.PI * x.rev) < 0.15, `finger rate ~ true rate at ${x.rev} rev/s: ${x.finger}`);
  assert.ok(r[0].robot > 0 && r[0].robot < r[1].robot && r[1].robot < r[2].robot, JSON.stringify(r));
  assert.ok(r[0].robot < 60 * DEG, 'slow circling -> slow rotation');
  assert.ok(r[2].robot > 400 * DEG, 'fast circling -> tornado');
});

test('spin: clockwise is +yaw (right), counter-clockwise is -yaw (left)', () => {
  const cw = drive((t) => feats(POSES.POINT, circleView(t, 1, 1)), { to: 2500 });
  const ccw = drive((t) => feats(POSES.POINT, circleView(t, 1, -1)), { to: 2500 });
  assert.ok(cw.out.at(-1).rate > 0 && cw.ctl.c.movement().direction === 'RIGHT');
  assert.ok(ccw.out.at(-1).rate < 0 && ccw.ctl.c.movement().direction === 'LEFT');
});

test('spin: the robot speeds up and slows down with the finger, and stops smoothly when circling stops', () => {
  // 0.6 rev/s, then 2 rev/s, then 0.6 rev/s, then the finger holds still
  const phaseAt = (t) => (t < 2000 ? 0.6 * t : t < 4000 ? 1200 + 2 * (t - 2000) : t < 6000 ? 5200 + 0.6 * (t - 4000) : 6400) / 1000;
  const view = (t) => {
    const a = 2 * Math.PI * phaseAt(t);
    return viewAt(0.9 + 0.12 * Math.cos(a), 0.65 + 0.12 * Math.sin(a));
  };
  const { out, events } = drive((t) => feats(POSES.POINT, view(t)), { to: 8000 });
  const avg = (a, b) => mean(out.filter((o) => o.t > a && o.t < b).map((o) => o.rate));
  const slow1 = avg(1500, 2000);
  const fast = avg(3500, 4000);
  const slow2 = avg(5500, 6000);
  assert.ok(fast > slow1 * 2 && fast > slow2 * 2, `${slow1} -> ${fast} -> ${slow2}`);
  assert.equal(out.at(-1).rate, 0, 'stopped after the finger stopped');
  assert.deepEqual(kinds(events, 'SPIN'), ['start', 'end'], 'one continuous rotation, one start and one end');
  // smooth: after the end the rate decays over frames (decel limit), it does not jump to 0
  const endT = events.find((e) => e.gesture === 'SPIN' && e.type === 'end').t;
  const after = out.filter((o) => o.t >= endT && o.t < endT + 300).map((o) => Math.abs(o.rate));
  assert.ok(after.length > 1 && after.every((v, i) => i === 0 || v <= after[i - 1] + 1e-9));
});

test('spin: lines, a still finger, jitter, tracking jumps and brief arcs never rotate', () => {
  let seed = 3;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) - 0.5;
  const cases = {
    line: (t) => viewAt(0.6 + 0.6 * ((t % 1500) / 1500), 0.6),
    still: () => viewAt(0.9, 0.6),
    jitter: () => viewAt(0.9 + rnd() * 0.01, 0.6 + rnd() * 0.01),
    jumps: (t) => viewAt(0.9 + (Math.floor(t / 100) % 2 ? 0.3 : -0.3), 0.6),
    noise: () => viewAt(0.9 + rnd() * 0.25, 0.6 + rnd() * 0.25),
  };
  for (const [name, view] of Object.entries(cases)) {
    const { events, out } = drive((t) => feats(POSES.POINT, view(t)), { to: 3000 });
    assert.deepEqual(kinds(events, 'SPIN'), [], name);
    assert.ok(out.every((o) => o.rate === 0), name);
  }
  // a brief accidental arc (a quarter turn) does not start rotation
  const arc = drive((t) => feats(POSES.POINT, t < 300 ? circleView(t, 0.8) : circleView(300, 0.8)), { to: 1500 });
  assert.deepEqual(kinds(arc.events, 'SPIN'), []);
  // circling with the wrong shape (open palm) never spins
  assert.deepEqual(kinds(drive((t) => feats(POSES.OPEN_PALM, circleView(t, 1))).events, 'SPIN'), []);
});

test('spin: hand loss and low confidence stop rotation at once; the release timeout tolerates a brief gap only', () => {
  const lost = drive((t) => (t < 2000 ? feats(POSES.POINT, circleView(t, 1.5)) : null), { to: 2300 });
  const atLoss = lost.out.find((o) => o.t >= 2000);
  assert.equal(atLoss.rate, 0, 'hand lost: immediate stop');
  assert.deepEqual(kinds(lost.events, 'SPIN'), ['start', 'end']);
  const low = drive((t) => feats(POSES.POINT, { ...circleView(t, 1.5), score: t < 2000 ? 0.95 : 0.3 }), { to: 2600 });
  assert.equal(low.out.at(-1).rate, 0, 'low tracking confidence: stopped');
  // a 100 ms pose glitch (< poseGraceMs) does not end the spin
  const glitch = drive((t) => feats(t > 1500 && t < 1600 ? POSES.CLAW : POSES.POINT, circleView(t, 1.5)), { to: 2500 });
  assert.deepEqual(kinds(glitch.events, 'SPIN'), ['start']);
});

test('spin: one continuous gesture, never discrete 90-degree turns; the lock cancels it immediately', () => {
  assert.ok(!('TURN_LEFT' in COMMANDS) && !('TURN_RIGHT' in COMMANDS), 'the discrete-turn commands are gone');
  const { events, ctl } = drive((t) => feats(POSES.POINT, circleView(t, 1)), { to: 4000 });
  assert.ok(!events.some((e) => e.type === 'event'), 'no discrete circle / swipe events');
  assert.deepEqual(kinds(events, 'SPIN'), ['start']);
  const c = ctl.c;
  assert.ok(c.spinRate !== 0);
  assert.equal(c.submit({ name: 'TOGGLE_LOCK', source: 'keyboard' }, 4001).accepted, true);
  assert.equal(c.spinRate, 0);
  assert.equal(c.state, 'LOCKED');
  const yaw = c.pose.yaw;
  c.updateHold(null, { spin: { omega: 20 }, walk: { cadence: 0 } });
  for (let t = 4002; t < 4500; t += 16) c.tick(t);
  assert.equal(c.pose.yaw, yaw, 'locked: no rotation even with a stale rate');
});

test('spin mapping: dead zone, gain and cap; integration is frame-rate independent', () => {
  const sc = DEFAULT_CONFIG.interaction.spin;
  assert.equal(spinRateFor(sc.deadZoneDegS * DEG * 0.9), 0);
  assert.ok(spinRateFor(10) > spinRateFor(5) && spinRateFor(-10) === -spinRateFor(10));
  assert.equal(spinRateFor(1000), sc.maxSpeedDegS * DEG);
  const yawAfter = (fps) => {
    const { c } = controller();
    c.setHandPresent(true, 0);
    c.handleGestureEvents([{ type: 'start', gesture: 'SPIN' }], 0);
    let yaw = 0;
    let prev = 0;
    for (let t = 0; t <= 1000; t += 1000 / fps) {
      c.updateHold(null, { spin: { omega: 8 }, walk: { cadence: 0 } });
      c.tick(t);
      let d = c.pose.yaw - prev;
      if (d < -Math.PI) d += 2 * Math.PI;
      if (d > Math.PI) d -= 2 * Math.PI;
      yaw += d;
      prev = c.pose.yaw;
    }
    return yaw;
  };
  assert.ok(Math.abs(yawAfter(15) - yawAfter(120)) < 0.15, `${yawAfter(15)} vs ${yawAfter(120)}`);
});

// ================================================================ finger walking

const walkRun = (freqAt, to = 5000) => drive((t) => feats(stepPose(t, { freq: typeof freqAt === 'function' ? freqAt(t) : freqAt })), { to, ctl: cadenceCtl() });

test('walking (default): raising index + middle (V) walks at once at the walk speed; lowering them stops', () => {
  const wc = DEFAULT_CONFIG.interaction.walk;
  const v = drive((t) => (t < 3000 ? feats(POSES.VICTORY) : feats(POSES.OPEN_PALM)), { to: 3600 });
  const startT = v.events.find((e) => e.type === 'start' && e.gesture === 'TWO_FINGERS').t;
  assert.ok(startT <= DEFAULT_CONFIG.gestures.definitions.TWO_FINGERS.holdMs + 40, `fast response: walking after ${startT} ms`);
  const walking = v.out.filter((o) => o.t > startT + 400 && o.t < 3000);
  assert.ok(walking.every((o) => o.mv === 'WALKING' && Math.abs(o.speed - wc.speed) < 1e-9), 'steady walk speed');
  const reach = v.out.find((o) => o.t >= startT && o.speed >= wc.speed - 1e-9).t - startT;
  assert.ok(reach <= 200, `reaches walking speed in ${reach} ms`);
  assert.ok(v.ctl.c.pose.z > 1.2, `walked ${v.ctl.c.pose.z} m`);
  assert.equal(v.out.at(-1).speed, 0, 'stopped after the fingers changed');
  assert.ok(!v.out.some((o) => o.mv === 'RUNNING'));
});

test('running (default): adding the ring finger runs; back to the V walks again without stopping', () => {
  const wc = DEFAULT_CONFIG.interaction.walk;
  const pose = (t) => (t < 1500 ? POSES.VICTORY : t < 3000 ? POSES.THREE_FINGERS : POSES.VICTORY);
  const r = drive((t) => feats(pose(t)), { to: 4500 });
  const at = (a, b) => r.out.filter((o) => o.t > a && o.t < b);
  assert.ok(at(800, 1500).every((o) => o.mv === 'WALKING'));
  assert.ok(at(2300, 3000).every((o) => o.mv === 'RUNNING' && Math.abs(o.speed - wc.runSpeed) < 1e-9), 'running at runSpeed');
  assert.ok(at(3800, 4500).every((o) => o.mv === 'WALKING' && Math.abs(o.speed - wc.speed) < 1e-9), 'walking again');
  // walk <-> run hand-over keeps moving: the speed never drops to 0 between them
  const switchOver = at(1500, 2300).concat(at(3000, 3800));
  assert.ok(switchOver.filter((o) => o.speed === 0).length <= 2, 'at most a frame or two at the gesture boundary');
  assert.ok(r.ctl.log.some((e) => e.type === 'command' && e.name === 'RUN' && e.status === 'accepted'));
  assert.ok(!AGENT_ROBOT_COMMANDS.includes('RUN'), 'the agent cannot run the robot');
  // the safety lock stops running at once
  const { c } = controller();
  c.setHandPresent(true, 0);
  c.handleGestureEvents([{ type: 'start', gesture: 'THREE_FINGERS' }], 0, { features: { valid: true, palmCenterNorm: { x: 0.5, y: 0.5 } } });
  for (let t = 0; t < 1000; t += 33) c.tick(t);
  assert.ok(c.walkSpeed > 1);
  assert.equal(c.submit({ name: 'TOGGLE_LOCK', source: 'keyboard' }, 1000).accepted, true);
  assert.equal(c.walkSpeed, 0);
  c.handleGestureEvents([{ type: 'start', gesture: 'THREE_FINGERS' }], 1100);
  assert.equal(c.state, 'LOCKED');
  // hand loss: no carry-over into a later walk
  const d = controller().c;
  d.setHandPresent(true, 0);
  d.handleGestureEvents([{ type: 'start', gesture: 'THREE_FINGERS' }], 0);
  for (let t = 0; t < 1000; t += 33) d.tick(t);
  d.setHandPresent(false, 1000);
  d.setHandPresent(true, 1100);
  d.handleGestureEvents([{ type: 'start', gesture: 'TWO_FINGERS' }], 1150);
  assert.equal(d.walkSpeed, 0, 'starts from rest after a hand loss');
});

test('walking (cadence mode): alternating finger steps start one walk; a static V does not walk', () => {
  const w = walkRun(1.25);
  assert.deepEqual(kinds(w.events, 'FINGER_WALK'), ['start']);
  assert.ok(w.ctl.c.pose.z > 0.3, `walked ${w.ctl.c.pose.z} m`);
  const v = drive((t) => feats(POSES.VICTORY), { to: 5000, ctl: cadenceCtl() });
  assert.deepEqual(kinds(v.events, 'FINGER_WALK'), []);
  assert.equal(v.ctl.c.pose.z, 0, 'holding a V does not move the robot in cadence mode');
  assert.ok(v.events.some((e) => e.type === 'start' && e.gesture === 'TWO_FINGERS'), 'the V is still recognised (shown as a pose)');
  assert.ok(!v.ctl.log.some((e) => e.type === 'command' && e.name === 'WALK'));
});

test('walking: slow cadence walks slowly, fast cadence faster; cadence follows the fingers', () => {
  const at = (f) => {
    const o = walkRun(f).out.filter((x) => x.t > 3500);
    return { cadence: mean(o.map((x) => x.motion.walk.cadence)), speed: mean(o.map((x) => x.speed)) };
  };
  const slow = at(0.65);
  const mid = at(1.25);
  const fast = at(2.2);
  for (const [r, f] of [[slow, 0.65], [mid, 1.25], [fast, 2.2]]) assert.ok(Math.abs(r.cadence - 2 * f) / (2 * f) < 0.15, `cadence ${r.cadence} vs ${2 * f}`);
  assert.ok(slow.speed > 0 && slow.speed < mid.speed && mid.speed < fast.speed, JSON.stringify({ slow, mid, fast }));
  assert.ok(Math.abs(fast.speed - walkSpeedFor(fast.cadence)) < 0.05, 'speed reaches the mapped target');
});

test('walking: slowing the steps slows the robot; stopping the steps stops it promptly', () => {
  const freq = (t) => (t < 3000 ? 2 : 0.7);
  // phase-continuous frequency change
  const phase = (t) => (t < 3000 ? 2 * t : 6000 + 0.7 * (t - 3000)) / 1000;
  const pose = (t) => {
    const s = Math.sin(2 * Math.PI * phase(t));
    const k = (a) => [40 * a, 48 * a, 28 * a];
    return { index: k(0.5 + 0.5 * s), middle: k(0.5 - 0.5 * s), ring: POSES.VICTORY.ring, pinky: POSES.VICTORY.pinky, thumb: 'tucked' };
  };
  const { out } = drive((t) => feats(pose(t)), { to: 6500, ctl: cadenceCtl() });
  const sp = (a, b) => mean(out.filter((o) => o.t > a && o.t < b).map((o) => o.speed));
  assert.ok(sp(2500, 3000) > sp(5500, 6500) * 1.5, `${sp(2500, 3000)} -> ${sp(5500, 6500)} (${freq(0)} -> ${freq(5000)} Hz)`);
  // stop stepping (fingers held still): speed 0 within ~1 s, walk ended
  const stop = drive((t) => feats(t < 3000 ? stepPose(t, { freq: 1.5 }) : stepPose(3000, { freq: 1.5 })), { to: 4600, ctl: cadenceCtl() });
  const endT = stop.events.find((e) => e.gesture === 'FINGER_WALK' && e.type === 'end')?.t;
  assert.ok(endT !== undefined && endT - 3000 <= DEFAULT_CONFIG.gestures.walk.stopMs + 40, `ended ${endT - 3000} ms after the last step`);
  const z = stop.ctl.c.pose.z;
  for (let t = 4700; t < 5500; t += 33) stop.ctl.c.tick(t);
  assert.equal(stop.ctl.c.pose.z, z, 'no coasting');
});

test('walking: shaking, in-phase bending, one moving finger and invalid landmarks are rejected', () => {
  const none = (frame, label) => assert.deepEqual(kinds(drive(frame, { to: 5000 }).events, 'FINGER_WALK'), [], label);
  none((t) => feats(POSES.VICTORY, viewAt(0.9 + 0.15 * Math.sin(t / 60), 0.6 + 0.05 * Math.cos(t / 45))), 'shaking a V');
  none((t) => feats(stepPose(t, { freq: 1.25, inPhase: true })), 'both fingers bend together');
  none((t) => feats(stepPose(t, { freq: 1.25, only: 'index' })), 'only the index finger moves');
  none((t) => feats(stepPose(t, { freq: 1.25, only: 'middle' })), 'only the middle finger moves');
  none((t) => feats(stepPose(t, { freq: 7 })), 'too fast (14 steps/s = shaking)');
  // a pointing hand: the index flexes a little while the CURLED middle finger wobbles in
  // anti-phase. Without the minReach rule this started a walk (a curled finger is not a leg).
  none((t) => {
    const s = Math.sin(2 * Math.PI * 1.25 * (t / 1000));
    return feats({ ...POSES.POINT, index: [20 + 20 * s, 24 + 24 * s, 14 + 14 * s], middle: [60 - 30 * s, 90, 55] });
  }, 'pointing hand with a wobbling curled middle finger');
  none((t) => (Math.floor(t / 66) % 2 ? feats(stepPose(t, { freq: 1.25 })) : { valid: false, reason: 'degenerate' }), 'invalid landmarks every other frame');
  // a valid walk ends at once when the hand is lost
  const lost = drive((t) => (t < 3500 ? feats(stepPose(t, { freq: 1.25 })) : null), { to: 3600 });
  assert.deepEqual(kinds(lost.events, 'FINGER_WALK'), ['start', 'end']);
  assert.equal(lost.out.at(-1).speed, 0);
});

test('walking: bounded area, lock stops it, agent cannot override it, gait phase follows distance', () => {
  const { c } = cadenceCtl();
  c.setHandPresent(true, 0);
  c.handleGestureEvents([{ type: 'start', gesture: 'FINGER_WALK' }], 0, { features: { valid: true, palmCenterNorm: { x: 0.5, y: 0.5 } } });
  let phaseSteps = 0;
  let prev = 0;
  for (let t = 0; t <= 20000; t += 33) {
    c.updateHold(null, { spin: { omega: 0 }, walk: { cadence: 4 } });
    c.tick(t);
    if (c.gaitPhase < prev) phaseSteps++;
    prev = c.gaitPhase;
  }
  const max = DEFAULT_CONFIG.interaction.walk.maxDistance;
  assert.ok(Math.abs(Math.hypot(c.pose.x, c.pose.z) - max) < 1e-9 && c.movement().atBoundary);
  assert.ok(phaseSteps >= Math.floor(max / DEFAULT_CONFIG.interaction.walk.strideLength / 2) - 1, 'gait cycles match the distance walked');
  assert.equal(c.submit({ name: 'TOGGLE_LOCK', source: 'agent' }, 20001).reason, 'CONFLICT');
  assert.equal(c.submit({ name: 'TOGGLE_LOCK', source: 'keyboard' }, 20002).accepted, true);
  assert.equal(c.walkSpeed, 0);
  assert.equal(c.state, 'LOCKED');
  assert.equal(c.handleGestureEvents([{ type: 'start', gesture: 'FINGER_WALK' }], 20100)[0].reason, 'LOCKED');
  // cadence-to-speed mapping: dead zone, minimum, cap
  assert.equal(walkSpeedFor(0.5), 0);
  assert.equal(walkSpeedFor(1), DEFAULT_CONFIG.interaction.walk.minSpeed);
  assert.equal(walkSpeedFor(100), DEFAULT_CONFIG.interaction.walk.maxSpeed);
});

test('priority: walking and spinning never run together; starting a walk stops a spin coast', () => {
  const { c } = cadenceCtl();
  c.setHandPresent(true, 0);
  c.handleGestureEvents([{ type: 'start', gesture: 'FINGER_WALK' }], 0);
  assert.equal(c.handleGestureEvents([{ type: 'start', gesture: 'SPIN' }], 10)[0].reason, 'CONFLICT');
  c.handleGestureEvents([{ type: 'end', gesture: 'FINGER_WALK' }], 20);
  c.handleGestureEvents([{ type: 'start', gesture: 'SPIN' }], 30);
  for (let t = 30; t < 600; t += 16) {
    c.updateHold(null, { spin: { omega: 12 }, walk: { cadence: 0 } });
    c.tick(t);
  }
  c.handleGestureEvents([{ type: 'end', gesture: 'SPIN' }], 600);
  assert.ok(c.spinRate !== 0, 'coasting down after release');
  c.handleGestureEvents([{ type: 'start', gesture: 'FINGER_WALK' }], 616);
  assert.equal(c.spinRate, 0, 'yaw is stable while walking');
  // in the engine, the stepping motion does not also produce a spin
  const w = walkRun(1.25);
  assert.deepEqual(kinds(w.events, 'SPIN'), []);
});

// ================================================================ emotions

test('emotions: thumb up laughs, thumb down is sad; sideways and half-extended thumbs are ambiguous', () => {
  const cls = new StaticGestureClassifier(gcfg());
  assert.equal(cls.classify(feats(POSES.THUMBS_UP, THUMB_VIEW.up)).gesture, 'THUMB_UP');
  assert.equal(cls.classify(feats(POSES.THUMBS_UP, THUMB_VIEW.down)).gesture, 'THUMB_DOWN');
  assert.equal(cls.classify(feats(POSES.THUMBS_UP)).gesture, 'UNKNOWN', 'sideways thumb');
  assert.equal(cls.classify(feats(POSES.THUMBS_UP, { ...THUMB_VIEW.up, score: 0.3 })).gesture, 'UNKNOWN', 'low confidence');
  assert.equal(cls.classify(feats(POSES.FIST)).gesture, 'FIST', 'the safety gesture keeps its meaning');
  const laugh = drive((t) => feats(POSES.THUMBS_UP, THUMB_VIEW.up), { to: 3000 });
  assert.deepEqual(laugh.ctl.log.filter((e) => e.type === 'command' && e.name === 'LAUGH').map((e) => e.status), ['accepted', 'completed'], 'held for 3 s: one laugh, not one per frame');
  const sad = drive((t) => feats(POSES.THUMBS_UP, THUMB_VIEW.down), { to: 1500 });
  assert.equal(sad.ctl.c.movement().state, 'SAD');
  assert.ok(!AGENT_ROBOT_COMMANDS.includes('LAUGH') && !AGENT_ROBOT_COMMANDS.includes('SAD'));
});

test('emotions: return to neutral; never move the robot; lock always wins; blocked while moving', () => {
  const { c, log } = controller();
  c.setHandPresent(true, 0);
  c.handleGestureEvents([{ type: 'start', gesture: 'THUMB_UP' }], 0);
  assert.equal(c.movement().state, 'LAUGHING');
  c.tick(DEFAULT_CONFIG.interaction.emotions.laughMs + 1);
  assert.equal(c.movement().state, 'IDLE');
  assert.equal(c.state, 'TRACKING');
  assert.deepEqual(c.pose, { yaw: 0, x: 0, z: 0, lift: 0 });
  // lock during an emotion: cancelled, then locked
  c.handleGestureEvents([{ type: 'start', gesture: 'THUMB_DOWN' }], 5000);
  assert.equal(c.submit({ name: 'TOGGLE_LOCK', source: 'agent' }, 5100).reason, 'BUSY', 'the agent cannot interrupt');
  assert.equal(c.submit({ name: 'TOGGLE_LOCK', source: 'keyboard' }, 5200).accepted, true);
  assert.equal(c.state, 'LOCKED');
  assert.ok(log.some((e) => e.type === 'command' && e.status === 'cancelled' && e.name === 'SAD'));
  assert.equal(c.handleGestureEvents([{ type: 'start', gesture: 'THUMB_UP' }], 5300)[0].reason, 'LOCKED');
  // while walking: rejected, the walk continues
  const w = controller().c;
  w.setHandPresent(true, 0);
  w.handleGestureEvents([{ type: 'start', gesture: 'TWO_FINGERS' }], 0);
  assert.equal(w.handleGestureEvents([{ type: 'start', gesture: 'THUMB_UP' }], 10)[0].reason, 'CONFLICT');
  assert.equal(w.movement().state, 'WALKING');
});

// ================================================================ regression

test('regression: existing gestures, bindings and the agent catalog are unchanged', () => {
  const eng = new GestureRecognitionEngine(gcfg());
  const starts = (pose, view) => {
    eng.reset();
    const out = [];
    for (let t = 0; t <= 1000; t += 33) out.push(...eng.update(feats(pose, view), t).events.filter((e) => e.type === 'start').map((e) => e.gesture));
    return out;
  };
  assert.deepEqual(starts(POSES.OPEN_PALM), ['OPEN_PALM']);
  assert.deepEqual(starts(POSES.FIST), ['FIST']);
  assert.deepEqual(starts(POSES.PINCH), ['PINCH']);
  assert.deepEqual(starts(POSES.POINT), ['POINT']);
  const e2 = new GestureRecognitionEngine(gcfg());
  const sw = [];
  for (let t = 0; t <= 300; t += 33) sw.push(...e2.update(feats(POSES.OPEN_PALM, viewAt(0.4 + (t / 300) * 0.9, 0.5)), t).events);
  assert.deepEqual(sw.filter((e) => e.type === 'event').map((e) => e.gesture), ['SWIPE_RIGHT']);
  const b = DEFAULT_CONFIG.interaction.bindings;
  assert.deepEqual([b.OPEN_PALM, b.FIST, b.PINCH, b.POINT, b.SWIPE_LEFT, b.SWIPE_RIGHT], ['WAVE', 'TOGGLE_LOCK', 'GRAB_ROTATE', 'AIM', 'SPIN_LEFT', 'SPIN_RIGHT']);
  assert.deepEqual(AGENT_ROBOT_COMMANDS, ['WAVE', 'SPIN_LEFT', 'SPIN_RIGHT', 'RESET_POSE', 'TOGGLE_LOCK']);
  // pinch-drag keeps its semantics and is not superseded by a spin
  const { c } = controller();
  c.setHandPresent(true, 0);
  c.handleGestureEvents([{ type: 'start', gesture: 'PINCH' }], 0, { features: { valid: true, palmCenterNorm: { x: 0.5, y: 0.5 } } });
  assert.equal(c.handleGestureEvents([{ type: 'start', gesture: 'SPIN' }], 10)[0].reason, 'CONFLICT');
  assert.equal(c.hold.name, 'GRAB_ROTATE');
  // SPIN supersedes AIM (same pointing shape)
  const d = controller().c;
  d.setHandPresent(true, 0);
  d.handleGestureEvents([{ type: 'start', gesture: 'POINT' }], 0, { features: { valid: true, palmCenterNorm: { x: 0.5, y: 0.5 } } });
  assert.equal(d.handleGestureEvents([{ type: 'start', gesture: 'SPIN' }], 500)[0].accepted, true);
  assert.equal(d.hold.name, 'SPIN');
  // RESET_POSE still returns home and stops a spin coast
  d.handleGestureEvents([{ type: 'end', gesture: 'SPIN' }], 600);
  d.spinRate = 3;
  assert.equal(d.submit({ name: 'RESET_POSE', source: 'keyboard' }, 700).accepted, true);
  assert.equal(d.spinRate, 0);
});

test('detector unit: SpinDetector requires the sustained signature before starting (enterMs)', () => {
  const det = new SpinDetector(gcfg().spin);
  let startT = null;
  for (let t = 0; t <= 2000 && startT === null; t += 33) {
    const a = 2 * Math.PI * (t / 1000);
    if (det.update(t, { x: 0.9 + 0.12 * Math.cos(a), y: 0.6 + 0.12 * Math.sin(a) }, 0.22, true).event === 'start') startT = t;
  }
  assert.ok(startT >= gcfg().spin.minWindowMs + gcfg().spin.enterMs, `started at ${startT} ms`);
});
