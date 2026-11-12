// Deep-learning perception layer: model/rule gesture fusion, face analysis (expressions,
// head pose, gaze), plus the robot behaviours they feed (lift, steering, strings, status, flag).
// Synthetic inputs only: model outputs are given as the MediaPipe result shapes.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as THREE from 'three';
import { ApplicationCoordinator } from '../src/app/ApplicationCoordinator.js';
import { DEFAULT_CONFIG } from '../src/config/defaultConfig.js';
import { fallingTips, pairTipsToAnchors, strandCurve } from '../src/fx/marionette.js';
import { FeatureExtractionEngine } from '../src/gestures/FeatureExtractionEngine.js';
import { GestureRecognitionEngine } from '../src/gestures/GestureRecognitionEngine.js';
import { MODEL_LABELS, StaticGestureClassifier } from '../src/gestures/StaticGestureClassifier.js';
import { InteractionController, steerRateFor } from '../src/interaction/InteractionController.js';
import { expressionScores, FaceExpressionEngine, gazeFromBlendshapes, headPoseFromMatrix, pickBlendshapes } from '../src/perception/faceAnalysis.js';
import { buildProceduralRobot, YEMEN_FLAG } from '../src/scene/Scene3D.js';
import { parseFaces } from '../src/tracking/FaceTrackingEngine.js';
import { toModelGesture } from '../src/tracking/HandTrackingEngine.js';
import { makeHand, POSES, THUMB_VIEW } from './fixtures/hands.js';

const A = 16 / 9;
const DEG = Math.PI / 180;
const fx = new FeatureExtractionEngine({ mirror: true });
const gcfg = () => JSON.parse(JSON.stringify(DEFAULT_CONFIG.gestures));
const cls = new StaticGestureClassifier(gcfg());
const model = (label, score, extra = {}) => ({ label, score, scores: { [label]: score, ...extra } });
const feats = (pose, m = null, view = {}) => fx.extract({ ...makeHand(pose, { aspect: A, ...view }), model: m }, { aspect: A });

// ================================================================ gesture fusion

test('model fusion: the learned classifier decides; its score and decision source are reported', () => {
  for (const [label, g] of Object.entries(MODEL_LABELS)) {
    // geometry agreeing with the class (thumbs need it): use the matching synthetic pose
    const pose = { OPEN_PALM: POSES.OPEN_PALM, FIST: POSES.FIST, POINT: POSES.POINT, THUMB_UP: POSES.THUMBS_UP, THUMB_DOWN: POSES.THUMBS_UP, TWO_FINGERS: POSES.VICTORY }[g];
    const view = g === 'THUMB_UP' ? THUMB_VIEW.up : g === 'THUMB_DOWN' ? THUMB_VIEW.down : {};
    const r = cls.classify(feats(pose, model(label, 0.91), view));
    assert.equal(r.gesture, g, label);
    assert.equal(r.source, 'model');
    assert.equal(r.confidence, 0.91);
  }
  const low = cls.classify(feats(POSES.OPEN_PALM, model('Open_Palm', 0.55)));
  assert.equal(low.gesture, 'UNKNOWN', 'below model.minScore: no gesture (no false activation)');
  assert.match(low.reason, /Open_Palm 0.55/);
  assert.equal(cls.classify(feats(POSES.OPEN_PALM, model('ILoveYou', 0.95))).gesture, 'UNKNOWN', 'unmapped class');
});

test('fusion safety: FIST (lock) only from the model; the rules assist only POINT / TWO_FINGERS', () => {
  // fist geometry but the model says None: never a rule-made FIST (the safety lock)
  assert.equal(cls.classify(feats(POSES.FIST, model('None', 0.9))).gesture, 'UNKNOWN');
  assert.equal(cls.classify(feats(POSES.OPEN_PALM, model('None', 0.9))).gesture, 'UNKNOWN');
  // a sideways point the canned classes miss: rules may report POINT, labelled as such
  const r = cls.classify(feats(POSES.POINT, model('None', 0.88), { rotationDeg: 90 }));
  assert.equal(r.gesture, 'POINT');
  assert.equal(r.source, 'rules (model: None)');
  // PINCH comes from the pinch distance (the model has no pinch class)
  const p = cls.classify(feats(POSES.PINCH, model('Open_Palm', 0.8)));
  assert.equal(p.gesture, 'PINCH');
  assert.equal(p.source, 'pinch-distance');
  // without a model result (model missing / Python tests): rule classifier, labelled
  assert.equal(cls.classify(feats(POSES.FIST)).source, 'rules');
  const off = new StaticGestureClassifier({ ...gcfg(), model: { enabled: false, minScore: 0.6 } });
  assert.equal(off.classify(feats(POSES.FIST, model('Open_Palm', 0.99))).gesture, 'FIST', 'model disabled by config');
});

test('fusion guard: a flat palm the model calls Thumb_Down (seen on the reference video) is rejected', () => {
  const r = cls.classify(feats(POSES.OPEN_PALM, model('Thumb_Down', 0.77), { rotationDeg: 90 }));
  assert.equal(r.gesture, 'UNKNOWN');
  assert.match(r.reason, /geometry disagrees/);
  // the same model output on real thumb-down geometry is accepted
  assert.equal(cls.classify(feats(POSES.THUMBS_UP, model('Thumb_Down', 0.77), THUMB_VIEW.down)).gesture, 'THUMB_DOWN');
  // end to end: held for 2 s it never reaches a command
  const eng = new GestureRecognitionEngine(gcfg());
  const starts = [];
  for (let t = 0; t <= 2000; t += 33) starts.push(...eng.update(feats(POSES.OPEN_PALM, model('Thumb_Down', 0.77), { rotationDeg: 90 }), t).events.filter((e) => e.type === 'start'));
  assert.deepEqual(starts, []);
});

test('model output parsing and temporal validation (one onset per held gesture)', () => {
  assert.deepEqual(toModelGesture([{ categoryName: 'None', score: 0.1 }, { categoryName: 'Pointing_Up', score: 0.8 }, { categoryName: 'x', score: NaN }]),
    { label: 'Pointing_Up', score: 0.8, scores: { None: 0.1, Pointing_Up: 0.8 } });
  assert.equal(toModelGesture([]), null);
  const eng = new GestureRecognitionEngine(gcfg());
  const ev = [];
  // a held, model-recognised open palm with one dropped frame: exactly one onset
  for (let t = 0; t <= 2000; t += 33) ev.push(...eng.update(feats(POSES.OPEN_PALM, t === 660 ? model('None', 0.7) : model('Open_Palm', 0.9)), t).events);
  assert.deepEqual(ev.filter((e) => e.type === 'start').map((e) => e.gesture), ['OPEN_PALM']);
  // a single-frame model spike never becomes a gesture
  const e2 = new GestureRecognitionEngine(gcfg());
  const s2 = [];
  for (let t = 0; t <= 1500; t += 33) s2.push(...e2.update(feats(POSES.OPEN_PALM, t === 330 ? model('Closed_Fist', 0.99) : model('None', 0.9)), t).events);
  assert.ok(!s2.some((e) => e.gesture === 'FIST'), 'no lock from a one-frame spike');
});

// ================================================================ face analysis

const bsOf = (o) => ({ ...o });

test('expressions: happy / sad / surprised need sustained evidence; speech and blinks stay neutral', () => {
  const run = (bs, ms = 1500) => {
    const e = new FaceExpressionEngine(DEFAULT_CONFIG.interaction.expression);
    let r;
    for (let t = 0; t <= ms; t += 33) r = e.update(t, typeof bs === 'function' ? bs(t) : bs);
    return r;
  };
  assert.equal(run(bsOf({ mouthSmileLeft: 0.8, mouthSmileRight: 0.75 })).label, 'HAPPY');
  assert.equal(run(bsOf({ mouthFrownLeft: 0.6, mouthFrownRight: 0.6, browInnerUp: 0.6 })).label, 'SAD');
  assert.equal(run(bsOf({ jawOpen: 0.8, browOuterUpLeft: 0.7, browOuterUpRight: 0.7, eyeWideLeft: 0.6, eyeWideRight: 0.6 })).label, 'SURPRISED');
  assert.equal(run(bsOf({ jawOpen: 0.7 })).label, 'NEUTRAL', 'an open mouth alone (speaking) is not surprise');
  assert.equal(run(bsOf({ eyeBlinkLeft: 1, eyeBlinkRight: 1 })).label, 'NEUTRAL');
  assert.equal(run(bsOf({ mouthSmileLeft: 0.9, mouthSmileRight: 0.9 }), 150).label, 'NEUTRAL', 'a brief (150 ms) smile does not switch yet (hold + smoothing)');
  // face lost: decays back to neutral; hysteresis keeps a fading smile briefly
  const e = new FaceExpressionEngine(DEFAULT_CONFIG.interaction.expression);
  for (let t = 0; t <= 1500; t += 33) e.update(t, { mouthSmileLeft: 0.8, mouthSmileRight: 0.8 });
  assert.equal(e.update(1533, null).label, 'HAPPY', 'not dropped on the first missing frame');
  let last;
  for (let t = 1566; t <= 3500; t += 33) last = e.update(t, null);
  assert.equal(last.label, 'NEUTRAL');
  assert.ok(last.scores.HAPPY < 0.05);
  const s = expressionScores({ mouthSmileLeft: 1, mouthSmileRight: 1, mouthFrownLeft: 1, mouthFrownRight: 1 });
  assert.equal(s.SAD, 0, 'a smile suppresses the sad reading');
});

test('head pose from the transformation matrix; gaze from the eye-look blendshapes', () => {
  const m4 = (yaw, pitch, roll) => {
    const m = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(pitch, yaw, roll, 'ZYX'));
    return Array.from(m.elements); // column-major, as MediaPipe gives it
  };
  for (const [y, p, r] of [[20, 0, 0], [0, -15, 0], [0, 0, 25], [-30, 10, 5]]) {
    const hp = headPoseFromMatrix(m4(y * DEG, p * DEG, r * DEG));
    assert.ok(Math.abs(hp.yawDeg - y) < 0.5 && Math.abs(hp.pitchDeg - p) < 0.5 && Math.abs(hp.rollDeg - r) < 0.5, JSON.stringify({ y, p, r, hp }));
  }
  assert.equal(headPoseFromMatrix([NaN]), null);
  assert.equal(headPoseFromMatrix('x'), null);
  const g = gazeFromBlendshapes({ eyeLookInLeft: 0.8, eyeLookOutRight: 0.8 });
  assert.ok(g.x > 0.7 && Math.abs(g.y) < 1e-9, 'looking to their right -> screen right in the mirrored view');
  assert.ok(gazeFromBlendshapes({ eyeLookDownLeft: 0.6, eyeLookDownRight: 0.6 }).y > 0.5);
  assert.deepEqual(gazeFromBlendshapes({}), { x: 0, y: 0 });
});

test('face landmarker results are parsed: box from landmarks, blendshape subset, pose matrix', () => {
  const lms = Array.from({ length: 478 }, (_, i) => ({ x: 0.4 + (i % 10) * 0.01, y: 0.3 + Math.floor(i / 10) * 0.004, z: 0 }));
  const faces = parseFaces({
    faceLandmarks: [lms, [{ x: NaN, y: 0 }]],
    faceBlendshapes: [{ categories: [{ categoryName: 'mouthSmileLeft', score: 0.7 }, { categoryName: 'tongueOut', score: 0.9 }] }],
    facialTransformationMatrixes: [{ data: new Float32Array(16).fill(0.5) }],
  }, { width: 640, height: 480 }, 0.6);
  assert.equal(faces.length, 1, 'malformed face dropped');
  assert.ok(Math.abs(faces[0].x - 0.445) < 1e-9 && faces[0].w > 0 && faces[0].h > 0);
  assert.deepEqual(faces[0].bs, { mouthSmileLeft: 0.7 }, 'only the categories the app uses');
  assert.equal(faces[0].matrix.length, 16);
  assert.deepEqual(pickBlendshapes([{ categoryName: 'jawOpen', score: 2 }]), { jawOpen: 1 });
});

// ================================================================ robot behaviours

const F = (x, y) => ({ valid: true, palmCenterNorm: { x, y } });

test('heavy lifting: raising the pinched hand lifts the robot (bounded); it descends smoothly afterwards', () => {
  const c = new InteractionController(DEFAULT_CONFIG.interaction);
  c.setHandPresent(true, 0);
  c.handleGestureEvents([{ type: 'start', gesture: 'PINCH' }], 0, { features: F(0.5, 0.7) });
  let t = 0;
  for (; t <= 1000; t += 16) {
    c.updateHold(F(0.5, 0.7 - 0.25 * (t / 1000)), null);
    c.tick(t);
  }
  assert.ok(c.pose.lift > 0.4 && c.pose.lift <= DEFAULT_CONFIG.interaction.lift.maxHeight, `lift ${c.pose.lift}`);
  assert.equal(c.movement().state, 'LIFTING');
  for (; t <= 3000; t += 16) {
    c.updateHold(F(0.5, -2), null); // an absurd hand position is clamped
    c.tick(t);
  }
  assert.ok(Math.abs(c.pose.lift - DEFAULT_CONFIG.interaction.lift.maxHeight) < 1e-6, 'capped at maxHeight');
  // the safety lock ends the lift hold immediately (safety before interaction) ...
  assert.equal(c.submit({ name: 'TOGGLE_LOCK', source: 'keyboard' }, t).accepted, true);
  assert.equal(c.state, 'LOCKED');
  assert.equal(c.hold, null);
  // ... and the agent could not have done it (it never overrides a held gesture)
  const g = new InteractionController(DEFAULT_CONFIG.interaction);
  g.setHandPresent(true, 0);
  g.handleGestureEvents([{ type: 'start', gesture: 'POINT' }], 0, { features: F(0.5, 0.5) });
  assert.equal(g.submit({ name: 'TOGGLE_LOCK', source: 'agent' }, 10).reason, 'CONFLICT');
  assert.equal(g.submit({ name: 'TOGGLE_LOCK', source: 'keyboard' }, 20).accepted, true, 'L ends an AIM hold and locks');
  const top = c.pose.lift;
  c.tick(t + 16);
  assert.ok(c.pose.lift < top && c.pose.lift > top * 0.9, 'descends smoothly, never drops instantly');
  for (let k = t + 32; k <= t + 4000; k += 16) c.tick(k);
  assert.equal(c.pose.lift, 0);
  // hand loss while lifting: the same smooth descent
  const d = new InteractionController(DEFAULT_CONFIG.interaction);
  d.setHandPresent(true, 0);
  d.handleGestureEvents([{ type: 'start', gesture: 'PINCH' }], 0, { features: F(0.5, 0.7) });
  for (let k = 0; k < 800; k += 16) {
    d.updateHold(F(0.5, 0.5), null);
    d.tick(k);
  }
  d.setHandPresent(false, 800);
  assert.equal(d.hold, null);
  for (let k = 816; k < 4000; k += 16) d.tick(k);
  assert.equal(d.pose.lift, 0);
});

test('steering while walking: hand offset turns the robot (dead zone, cap); only while walking (also while running)', () => {
  assert.equal(steerRateFor(0.03), 0);
  assert.ok(steerRateFor(0.2) > 0 && steerRateFor(-0.2) < 0);
  assert.equal(steerRateFor(10), DEFAULT_CONFIG.interaction.steer.maxDegS * DEG);
  const c = new InteractionController(DEFAULT_CONFIG.interaction);
  c.setHandPresent(true, 0);
  c.handleGestureEvents([{ type: 'start', gesture: 'TWO_FINGERS' }], 0, { features: F(0.5, 0.6) });
  for (let t = 0; t <= 2000; t += 16) {
    c.updateHold(F(0.75, 0.6), { spin: { omega: 0 }, walk: { cadence: 3 } });
    c.tick(t);
  }
  assert.ok(c.pose.yaw > 0.5, `turned right while walking: ${c.pose.yaw}`);
  assert.equal(c.movement().direction, 'RIGHT');
  const idle = new InteractionController(DEFAULT_CONFIG.interaction);
  idle.setHandPresent(true, 0);
  for (let t = 0; t <= 1000; t += 16) {
    idle.updateHold(F(0.9, 0.6), null);
    idle.tick(t);
  }
  assert.equal(idle.pose.yaw, 0, 'no steering without a walk');
});

test('marionette strings: tips paired left-to-right (never crossing), sag, and fall when the hand is lost', () => {
  const tips = [{ x: 500, y: 300 }, { x: 420, y: 250 }, { x: 460, y: 240 }, { x: 540, y: 330 }, { x: 380, y: 280 }];
  const anchors = [{ x: 900, y: 200 }, { x: 800, y: 400 }, { x: 860, y: 300 }, { x: 940, y: 420 }, { x: 830, y: 150 }];
  const pairs = pairTipsToAnchors(tips, anchors);
  assert.equal(pairs.length, 5);
  for (let i = 1; i < pairs.length; i++) {
    assert.ok(pairs[i].tip.x >= pairs[i - 1].tip.x && pairs[i].anchor.x >= pairs[i - 1].anchor.x, 'ordered: strings never cross');
  }
  const c = strandCurve({ x: 0, y: 0 }, { x: 400, y: 0 }, 1);
  assert.ok(c.ctrl.y > 50, 'strings sag under gravity');
  const f0 = fallingTips(tips, 0);
  const f1 = fallingTips(tips, 400);
  assert.equal(f0.alpha, 1);
  assert.ok(f1.alpha < 1 && f1.alpha > 0 && f1.tips.every((p, i) => p.y > tips[i].y && p.x < tips[i].x), 'falls down and to the left');
  assert.equal(fallingTips(tips, 900).alpha, 0);
  assert.deepEqual(pairTipsToAnchors([{ x: NaN, y: 0 }], anchors), []);
});

test('status plate text follows the robot behaviour (reference video vocabulary)', () => {
  const st = (snap, tracked = true) => ApplicationCoordinator.prototype._robotStatus.call({}, { action: null, hold: null, movement: { state: 'IDLE' }, state: 'TRACKING', ...snap }, tracked ? { features: {} } : null, true);
  assert.equal(st({}).label, 'IDLE');
  assert.equal(st({}).task, 'awaiting instruction');
  assert.equal(st({ hold: { name: 'WALK' } }).label, 'RUNNING IMPLEMENTATION');
  assert.equal(st({ hold: { name: 'SPIN' } }).label, 'SPINNING UP NEW IDEA');
  assert.equal(st({ hold: null, movement: { state: 'ROTATING', source: 'SPIN' } }).label, 'SPINNING UP NEW IDEA', 'coasting');
  assert.equal(st({ hold: { name: 'GRAB_ROTATE' } }).label, 'HEAVY LIFTING');
  assert.equal(st({ state: 'LOCKED' }).label, 'LOCKED');
  assert.equal(st({}, true).line, '[ok] hand tracked');
  assert.equal(st({}, false).line, '[--] hand lost');
});

test('Yemen flag: red / white / black horizontal strips at 2:3 on the torso, moving with the body', () => {
  const r = buildProceduralRobot(null);
  const flag = r.flag;
  assert.equal(flag.name, 'yemenFlag');
  assert.equal(flag.parent, r.torso, 'a real 3D surface on the body, not an overlay');
  const strips = flag.children.filter((c) => c.name.startsWith('yemenFlag_'));
  assert.deepEqual(strips.map((s) => `#${s.material.color.getHexString()}`), YEMEN_FLAG.colors);
  assert.deepEqual(YEMEN_FLAG.colors, ['#ce1126', '#ffffff', '#000000']);
  assert.ok(strips[0].position.y > strips[1].position.y && strips[1].position.y > strips[2].position.y, 'red on top, black at the bottom');
  const size = new THREE.Box3().setFromObject(new THREE.Group().add(...strips.map((s) => s.clone()))).getSize(new THREE.Vector3());
  assert.ok(Math.abs(size.x / size.y - 1.5) < 1e-6, `aspect ${size.x / size.y} (official 2:3)`);
  // follows the body: rotate the whole robot, the flag keeps its place on the torso front
  r.root.rotation.y = Math.PI / 2;
  r.root.updateMatrixWorld(true);
  const local = r.torso.worldToLocal(flag.getWorldPosition(new THREE.Vector3()));
  assert.ok(Math.abs(local.z - 0.28) < 1e-6 && Math.abs(local.x) < 1e-6, 'still on the torso front after a turn');
  // in front of the armour surface (no z-fighting), below the chest plate (no overlap)
  const front = flag.position.z + Math.min(...strips.map((s) => s.position.z)) - 0.003;
  assert.ok(front > 0.28, 'strip faces are in front of the torso surface');
  const edge = YEMEN_FLAG.height / 2 + YEMEN_FLAG.border;
  assert.ok(flag.position.y + edge < 0.31, 'frame top below the chest plate (no overlap)');
  assert.ok(flag.position.y - edge >= 0.15 - 1e-9, 'frame bottom on the flat front face (not floating over the rounded edge)');
});

test('expressions: per-user calibration (resting level + gain) catches a weak live smile, not the resting face', () => {
  const e = new FaceExpressionEngine(DEFAULT_CONFIG.interaction.expression);
  const face = (smile) => ({ mouthSmileLeft: smile, mouthSmileRight: smile, mouthFrownLeft: 0.03, mouthFrownRight: 0.03 });
  let r;
  let t = 0;
  // resting face with a little smile activation and noise (as measured live, VALIDATION.md §15)
  for (; t <= 10000; t += 33) r = e.update(t, face(0.05 + 0.08 * Math.max(0, Math.sin(t / 700))));
  assert.equal(r.label, 'NEUTRAL', 'the resting face is not a smile');
  assert.ok(r.baseline.HAPPY > 0.01 && r.baseline.HAPPY < 0.1, `resting level learned (${r.baseline.HAPPY})`);
  // a live smile whose raw score is only about 0.28
  for (const end = t + 1500; t <= end; t += 33) r = e.update(t, face(0.28));
  assert.equal(r.label, 'HAPPY');
  // a constant expression slowly becomes the new resting level (it is not reported forever)
  for (const end = t + 120000; t <= end; t += 100) r = e.update(t, face(0.28));
  assert.equal(r.label, 'NEUTRAL');
  // calibration off: raw thresholds as before
  const raw = new FaceExpressionEngine({ ...DEFAULT_CONFIG.interaction.expression, baseline: { enabled: false }, gain: { HAPPY: 1, SAD: 1, SURPRISED: 1 } });
  for (let u = 0; u <= 1500; u += 33) r = raw.update(u, face(0.28));
  assert.equal(r.label, 'NEUTRAL');
});
