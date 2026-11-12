// Walk / run / squat, head imitation, and the project-trained hand-pose network.
// Synthetic landmarks and synthetic weights only (no camera, no network, no API key).
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { AGENT_ROBOT_COMMANDS } from '../src/agent/agentTools.js';
import { ApplicationCoordinator } from '../src/app/ApplicationCoordinator.js';
import { DEFAULT_CONFIG } from '../src/config/defaultConfig.js';
import { EventBus } from '../src/core/EventBus.js';
import { FeatureExtractionEngine } from '../src/gestures/FeatureExtractionEngine.js';
import { GestureRecognitionEngine } from '../src/gestures/GestureRecognitionEngine.js';
import { StaticGestureClassifier } from '../src/gestures/StaticGestureClassifier.js';
import { InteractionController } from '../src/interaction/InteractionController.js';
import { CUE_LANDMARKS, HeadCueCalibrator, headCueFromLandmarks, PITCH_NEUTRAL, PITCH_RANGE } from '../src/perception/faceAnalysis.js';
import { createHandPoseNet, POSE_DIM, poseVector } from '../src/perception/handPoseNet.js';
import { parseFaces } from '../src/tracking/FaceTrackingEngine.js';
import { makeHand, POSES } from './fixtures/hands.js';

const ASPECT = 16 / 9;
const gcfg = () => JSON.parse(JSON.stringify(DEFAULT_CONFIG.gestures));
const fx = new FeatureExtractionEngine({ mirror: true });
const feats = (pose, view = {}) => fx.extract(makeHand(pose, { aspect: ASPECT, ...view }), { aspect: ASPECT });

function drive(frame, to, eng = new GestureRecognitionEngine(gcfg())) {
  const bus = new EventBus();
  const log = [];
  bus.on('*', (e) => log.push(e));
  const c = new InteractionController(DEFAULT_CONFIG.interaction, { bus });
  const events = [];
  const states = [];
  for (let t = 0; t <= to; t += 33) {
    const f = frame(t);
    const r = eng.update(f, t);
    c.setHandPresent(!!f?.valid, t);
    c.handleGestureEvents(r.events, t, { features: f });
    c.updateHold(f, r.motion);
    c.tick(t);
    events.push(...r.events.map((e) => ({ ...e, t })));
    states.push({ t, mv: c.movement().state });
  }
  return { c, log, events, states };
}

// ================================================================ squat

test('squat: touching and releasing thumb + index (a quick pinch) does one squat, quickly', () => {
  const { events, log, states } = drive((t) => feats(t < 400 ? POSES.PINCH : POSES.OPEN_PALM), 2200);
  const tap = events.filter((e) => e.gesture === 'PINCH_TAP');
  assert.equal(tap.length, 1);
  const sq = log.filter((e) => e.type === 'command' && e.name === 'SQUAT').map((e) => e.status);
  assert.deepEqual(sq, ['accepted', 'completed']);
  const first = states.find((s) => s.mv === 'SQUATTING').t;
  assert.ok(first - 400 <= DEFAULT_CONFIG.gestures.exitMs + 70, `squat starts ${first - 400} ms after the release`);
  assert.ok(states.filter((s) => s.mv === 'SQUATTING').length * 33 <= DEFAULT_CONFIG.interaction.squatMs + 66);
  assert.ok(!AGENT_ROBOT_COMMANDS.includes('SQUAT'), 'gesture only');
});

test('squat: a long pinch (turn / lift) and a pinch lost with the hand never squat', () => {
  const long = drive((t) => feats(t < 1500 ? POSES.PINCH : POSES.OPEN_PALM), 2600);
  assert.equal(long.events.filter((e) => e.gesture === 'PINCH_TAP').length, 0);
  assert.ok(long.log.some((e) => e.type === 'command' && e.name === 'GRAB_ROTATE' && e.status === 'accepted'));
  const lost = drive((t) => (t < 400 ? feats(POSES.PINCH) : null), 1200);
  assert.equal(lost.events.filter((e) => e.gesture === 'PINCH_TAP').length, 0);
});

test('squat: repeated taps squat again (no cooldown); the safety lock cancels a squat; locked: no squat', () => {
  const c = new InteractionController(DEFAULT_CONFIG.interaction);
  c.setHandPresent(true, 0);
  assert.equal(c.handleGestureEvents([{ type: 'event', gesture: 'PINCH_TAP' }], 0)[0].accepted, true);
  c.tick(DEFAULT_CONFIG.interaction.squatMs + 1);
  assert.equal(c.handleGestureEvents([{ type: 'event', gesture: 'PINCH_TAP' }], DEFAULT_CONFIG.interaction.squatMs + 10)[0].accepted, true, 'again at once');
  assert.equal(c.submit({ name: 'TOGGLE_LOCK', source: 'agent' }, 1200).reason, 'BUSY');
  assert.equal(c.submit({ name: 'TOGGLE_LOCK', source: 'keyboard' }, 1200).accepted, true);
  assert.equal(c.state, 'LOCKED');
  assert.equal(c.handleGestureEvents([{ type: 'event', gesture: 'PINCH_TAP' }], 1300)[0].reason, 'LOCKED');
  assert.deepEqual(c.pose, { yaw: 0, x: 0, z: 0, lift: 0 }, 'a squat never moves the robot');
});

test('status plate: RUN and SQUAT have their own labels', () => {
  const st = (snap) => ApplicationCoordinator.prototype._robotStatus.call({}, { action: null, hold: null, movement: { state: 'IDLE' }, state: 'TRACKING', ...snap }, { features: {} }, true);
  assert.equal(st({ hold: { name: 'RUN' } }).label, 'SPRINTING');
  assert.equal(st({ state: 'EXECUTING', action: { name: 'SQUAT' } }).label, 'SQUATTING');
});

// ================================================================ head imitation

/** 478 synthetic face landmarks; only the cue points matter. */
function face({ yaw = 0, ratio = PITCH_NEUTRAL } = {}) {
  const lms = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  const { noseTip, cheekRight, cheekLeft, forehead, chin } = CUE_LANDMARKS;
  lms[cheekRight] = { x: 0.4, y: 0.5, z: 0 };
  lms[cheekLeft] = { x: 0.6, y: 0.5, z: 0 };
  lms[forehead] = { x: 0.5, y: 0.3, z: 0 };
  lms[chin] = { x: 0.5, y: 0.7, z: 0 };
  lms[noseTip] = { x: 0.5 + yaw * 0.1, y: 0.3 + ratio * 0.4, z: 0 };
  return lms;
}

test('head cues: frontal is 0; turning moves the yaw cue; looking up / down moves the pitch cue; bounded', () => {
  assert.deepEqual(headCueFromLandmarks(face()), { yaw: 0, pitch: 0 });
  assert.equal(headCueFromLandmarks(face({ yaw: 0.5 })).yaw, 0.5, 'nose toward image right');
  assert.equal(headCueFromLandmarks(face({ yaw: -0.5 })).yaw, -0.5);
  assert.ok(headCueFromLandmarks(face({ ratio: PITCH_NEUTRAL - PITCH_RANGE / 2 })).pitch < -0.49, 'nose closer to the forehead = looking up');
  assert.ok(headCueFromLandmarks(face({ ratio: PITCH_NEUTRAL + PITCH_RANGE })).pitch >= 0.999, 'looking down');
  assert.equal(headCueFromLandmarks(face({ yaw: 3 })).yaw, 1, 'clamped');
  assert.equal(headCueFromLandmarks(face().slice(0, 100)), null);
  const bad = face();
  bad[1] = { x: NaN, y: 0 };
  assert.equal(headCueFromLandmarks(bad), null);
  // FaceLandmarker parsing carries the cue
  const [f] = parseFaces({ faceLandmarks: [face({ yaw: 0.3 })], faceBlendshapes: [], facialTransformationMatrixes: [] }, { width: 640, height: 480 }, 0.5);
  assert.equal(f.cue.yaw, 0.3);
});

test('head cues: the mirrored view flips the turn direction; the smoothed cue decays when the face is lost', () => {
  const ctx = { config: { camera: { mirror: true } }, dom: { stage: { getBoundingClientRect: () => ({ width: 1000, height: 500 }) } },
    _mapping: () => ({ displayToScreen: (p) => ({ x: p.x * 1000, y: p.y * 500 }) }), attention: { observeFaces: (t, faces) => (ctx.seen = faces) } };
  ApplicationCoordinator.prototype._observeFaces.call(ctx, [{ x: 0.5, y: 0.5, w: 0.2, h: 0.3, score: 1, cue: { yaw: 0.4, pitch: -0.2 } }]);
  assert.deepEqual(ctx.seen[0].cue, { yaw: -0.4, pitch: -0.2 }, 'image right = screen left in a mirrored view; pitch unchanged');
  ctx.config.camera.mirror = false;
  ApplicationCoordinator.prototype._observeFaces.call(ctx, [{ x: 0.5, y: 0.5, w: 0.2, h: 0.3, score: 1, cue: { yaw: 0.4, pitch: -0.2 } }]);
  assert.equal(ctx.seen[0].cue.yaw, 0.4);
  const s = {};
  const smooth = (t, cue) => ApplicationCoordinator.prototype._smoothCue.call(s, t, cue);
  smooth(0, null);
  let c;
  for (let t = 16; t <= 400; t += 16) c = smooth(t, { yaw: 0.5, pitch: -0.5 });
  assert.ok(Math.abs(c.yaw - 0.5) < 0.01 && c.active, `follows within ~0.4 s: ${c.yaw}`);
  const early = smooth(416, { yaw: 0.5, pitch: -0.5 });
  assert.ok(early.yaw <= 0.5 + 1e-9, 'never overshoots');
  for (let t = 432; t <= 2000; t += 16) c = smooth(t, null);
  assert.ok(Math.abs(c.yaw) < 0.01 && !c.active, 'decays to neutral without a face');
});

// ================================================================ hand-pose network

function tinyNet(bias = {}) {
  const classes = ['OPEN_PALM', 'FIST', 'POINT', 'TWO_FINGERS', 'THREE_FINGERS', 'THUMB_UP', 'THUMB_DOWN', 'OTHER'];
  return {
    format: 'aura-handpose-mlp', version: 0, classes, inputMean: new Array(63).fill(0), inputStd: new Array(63).fill(1),
    layers: [
      { in: 63, out: 4, W: new Array(252).fill(0), b: [0, 0, 0, 0] },
      { in: 4, out: classes.length, W: new Array(4 * classes.length).fill(0), b: classes.map((c) => bias[c] ?? 0) },
    ],
  };
}

test('pose vector: wrist-relative, palm-size normalised (scale / translation invariant), 63 values', () => {
  const h = makeHand(POSES.OPEN_PALM, { aspect: ASPECT });
  const disp = (lms, k = 1, dx = 0) => lms.map((p) => ({ x: p.x * k + dx, y: p.y * k, z: (p.z ?? 0) * k }));
  const a = poseVector(disp(h.landmarks));
  const b = poseVector(disp(h.landmarks, 2.5, 0.3));
  assert.equal(a.length, POSE_DIM);
  assert.ok(a.every((v, i) => Math.abs(v - b[i]) < 1e-9));
  assert.deepEqual(a.slice(0, 3), [0, 0, 0]);
  assert.equal(poseVector(h.landmarks.slice(0, 20)), null);
  assert.ok(feats(POSES.OPEN_PALM).pose.length === POSE_DIM, 'features carry the pose vector');
});

test('network runtime: softmax output; malformed weight files are rejected', () => {
  const net = createHandPoseNet(tinyNet({ THREE_FINGERS: 3 }));
  const p = net.predict(new Array(63).fill(0.1));
  assert.equal(p.label, 'THREE_FINGERS');
  assert.ok(Math.abs(Object.values(p.scores).reduce((s, v) => s + v, 0) - 1) < 1e-9);
  assert.throws(() => createHandPoseNet({ format: 'other' }));
  const w = tinyNet();
  w.layers[0].W = w.layers[0].W.slice(1);
  assert.throws(() => createHandPoseNet(w), /inconsistent/);
  const w2 = tinyNet();
  w2.inputMean = [1];
  assert.throws(() => createHandPoseNet(w2), /63/);
});

test('fusion with the net: three fingers over Victory / None; OTHER vetoes rule assist; FIST never from the net', () => {
  const cls = new StaticGestureClassifier(gcfg());
  const withNet = (pose, model, bias, view) => {
    const f = feats(pose, view);
    f.model = model;
    f.net = createHandPoseNet(tinyNet(bias)).predict(f.pose);
    return cls.classify(f);
  };
  const r1 = withNet(POSES.THREE_FINGERS, { label: 'Victory', score: 0.85, scores: {} }, { THREE_FINGERS: 5 });
  assert.equal(r1.gesture, 'THREE_FINGERS');
  assert.match(r1.source, /aura-net/);
  // a canned Victory is not overridden when the net does not say three fingers
  assert.equal(withNet(POSES.VICTORY, { label: 'Victory', score: 0.85, scores: {} }, { TWO_FINGERS: 5 }).gesture, 'TWO_FINGERS');
  // canned None + sideways point: the rule would assist; a confident OTHER rejects it
  assert.equal(withNet(POSES.POINT, { label: 'None', score: 0.9, scores: {} }, { OTHER: 5 }, { rotationDeg: 90 }).gesture, 'UNKNOWN');
  assert.equal(withNet(POSES.POINT, { label: 'None', score: 0.9, scores: {} }, { POINT: 5 }, { rotationDeg: 90 }).gesture, 'POINT');
  // the safety lock stays with the canned model
  assert.equal(withNet(POSES.FIST, { label: 'None', score: 0.9, scores: {} }, { FIST: 5 }).gesture, 'UNKNOWN');
  // a confident canned class is not overridden by the net
  assert.equal(withNet(POSES.OPEN_PALM, { label: 'Open_Palm', score: 0.9, scores: {} }, { POINT: 5 }).gesture, 'OPEN_PALM');
  // the engine runs the net when it is given one
  const eng = new GestureRecognitionEngine(gcfg(), { net: createHandPoseNet(tinyNet({ OTHER: 2 })) });
  const f = feats(POSES.OPEN_PALM);
  eng.update(f, 0);
  assert.equal(f.net.label, 'OTHER');
});

test('head cue calibration: the user\'s resting pose becomes 0; held turns are kept; a new face relearns', () => {
  const cal = new HeadCueCalibrator();
  let r;
  let t = 0;
  // resting head reads pitch -1.6 / yaw 0.1 for this face + camera (as measured live)
  for (; t <= 2000; t += 33) r = cal.update(t, { yaw: 0.1, pitch: -1.6 });
  assert.ok(Math.abs(r.yaw) < 0.02 && Math.abs(r.pitch) < 0.05 && !r.calibrating, `rest -> 0: ${JSON.stringify(r)}`);
  // turning and holding for 5 s: most of the turn is kept
  for (const end = t + 5000; t <= end; t += 33) r = cal.update(t, { yaw: 0.6, pitch: -0.6 });
  assert.ok(r.yaw > 0.4 && r.pitch > 0.7, `held turn kept: ${JSON.stringify(r)}`);
  // back to rest, then the face leaves for 6 s; the next face is calibrated afresh
  for (const end = t + 1000; t <= end; t += 33) r = cal.update(t, { yaw: 0.1, pitch: -1.6 });
  for (const end = t + 6000; t <= end; t += 33) assert.equal(cal.update(t, null), null);
  for (const end = t + 2000; t <= end; t += 33) r = cal.update(t, { yaw: -0.2, pitch: 0.5 });
  assert.ok(Math.abs(r.yaw) < 0.05 && Math.abs(r.pitch) < 0.05, `new face relearned: ${JSON.stringify(r)}`);
  // bounded
  assert.ok(Math.abs(cal.update(t + 33, { yaw: 1, pitch: 4 }).pitch) <= 1.5);
  // the raw pitch cue keeps a wider range than before (calibration subtracts the rest)
  assert.ok(headCueFromLandmarks(face({ ratio: PITCH_NEUTRAL - 2 * PITCH_RANGE })).pitch <= -1.99);
});
