import assert from 'node:assert/strict';
import { test } from 'node:test';
import { FramePipeline } from '../src/app/FramePipeline.js';
import { DEFAULT_CONFIG } from '../src/config/defaultConfig.js';
import { EventBus } from '../src/core/EventBus.js';
import { RateMeter, RollingStats } from '../src/core/metrics.js';
import { FeatureExtractionEngine } from '../src/gestures/FeatureExtractionEngine.js';
import { GestureRecognitionEngine } from '../src/gestures/GestureRecognitionEngine.js';
import { InteractionController } from '../src/interaction/InteractionController.js';
import { resolveRobotAsset } from '../src/scene/robotAsset.js';
import { HandTrackingEngine } from '../src/tracking/HandTrackingEngine.js';
import { makeHand, POSES } from './fixtures/hands.js';

const clone = (o) => JSON.parse(JSON.stringify(o));

/** Backend that replays a scripted list of MediaPipe-shaped results. */
function scriptedBackend(script) {
  let i = 0;
  const calls = { init: 0, detect: 0, close: 0, timestamps: [] };
  return {
    calls,
    async init() {
      calls.init++;
      return { delegate: 'CPU' };
    },
    detect(_src, ts) {
      calls.detect++;
      calls.timestamps.push(ts);
      const step = script[Math.min(i++, script.length - 1)];
      if (step instanceof Error) throw step;
      return step;
    },
    close() {
      calls.close++;
    },
  };
}

const toResult = (...hands) => ({
  landmarks: hands.map((h) => h.landmarks),
  worldLandmarks: hands.map((h) => h.worldLandmarks),
  handedness: hands.map((h) => [{ categoryName: h.handedness, score: h.score, index: 0, displayName: '' }]),
});
const EMPTY = { landmarks: [], worldLandmarks: [], handedness: [] };

// ------------------------------------------------------------ tracking engine

test('tracking engine: init failure is reported, detect refuses to run', async () => {
  const bus = new EventBus();
  const seen = [];
  bus.on('tracker', (e) => seen.push(e));
  const engine = new HandTrackingEngine({
    bus,
    backend: {
      init: async () => {
        throw new Error('hand model not found');
      },
    },
  });
  const r = await engine.init();
  assert.equal(r.ok, false);
  assert.equal(engine.status, 'error');
  assert.match(seen[0].error, /not found/);
  assert.equal(engine.detect({}, 0).ok, false);
});

test('tracking engine: parses results, drops malformed hands, handles no hands', async () => {
  const good = makeHand(POSES.OPEN_PALM);
  const backend = scriptedBackend([
    EMPTY,
    { landmarks: [good.landmarks, good.landmarks.slice(0, 5)], worldLandmarks: [good.worldLandmarks, []], handedness: [[{ categoryName: 'Right', score: 0.9 }], []] },
    null,
  ]);
  const engine = new HandTrackingEngine({ backend });
  await engine.init();
  assert.deepEqual(engine.detect({}, 1).hands, []);
  const r = engine.detect({}, 2);
  assert.equal(r.hands.length, 1);
  assert.equal(r.hands[0].score, 0.9);
  assert.equal(engine.stats.rejectedHands, 1);
  assert.deepEqual(engine.detect({}, 3).hands, [], 'null result is treated as no hands');
});

test('tracking engine: timestamps passed to the model are strictly increasing', async () => {
  const backend = scriptedBackend([EMPTY]);
  const engine = new HandTrackingEngine({ backend });
  await engine.init();
  engine.detect({}, 100);
  engine.detect({}, 100);
  engine.detect({}, 50);
  const ts = backend.calls.timestamps;
  assert.ok(ts[1] > ts[0] && ts[2] > ts[1]);
});

test('tracking engine: model exceptions are contained; close() is idempotent', async () => {
  const backend = scriptedBackend([new Error('gpu lost')]);
  const engine = new HandTrackingEngine({ backend });
  await engine.init();
  const r = engine.detect({}, 1);
  assert.equal(r.ok, false);
  assert.match(r.error, /gpu lost/);
  engine.close();
  engine.close();
  assert.equal(backend.calls.close, 1);
  assert.equal(engine.detect({}, 2).ok, false);
});

test('tracking engine: primary hand is sticky by handedness, else the largest', async () => {
  const small = makeHand(POSES.FIST, { size: 1.0, handedness: 'Left', cx: 0.3 });
  const big = makeHand(POSES.OPEN_PALM, { size: 2.5, handedness: 'Right', cx: 0.7 });
  const engine = new HandTrackingEngine({ backend: scriptedBackend([toResult(small, big)]) });
  await engine.init();
  assert.equal(engine.detect({}, 1).primary.handedness, 'Right');
  const bigLeft = makeHand(POSES.OPEN_PALM, { size: 3, handedness: 'Left', cx: 0.3 });
  const r = engine.selectPrimary(engine.parse(toResult(bigLeft, makeHand(POSES.FIST, { size: 1, handedness: 'Right' }))));
  assert.equal(r.handedness, 'Right', 'keeps following the same hand');
});

// ------------------------------------------------------------ full pipeline

async function makePipeline(script) {
  const bus = new EventBus();
  const log = [];
  bus.on('*', (e) => log.push(e));
  const cfg = clone(DEFAULT_CONFIG);
  const backend = scriptedBackend(script);
  const tracker = new HandTrackingEngine({ backend, bus });
  await tracker.init();
  let clock = 0;
  const controller = new InteractionController(cfg.interaction, { bus });
  const pipeline = new FramePipeline({
    tracker,
    extractor: new FeatureExtractionEngine({ mirror: true }),
    recognizer: new GestureRecognitionEngine(cfg.gestures),
    controller,
    bus,
    now: () => clock,
    maxConsecutiveFailures: 3,
  });
  const run = (t) => {
    clock = t + 5; // 5 ms processing time
    return pipeline.process({ video: {}, t, aspect: 16 / 9 });
  };
  return { pipeline, controller, run, log, backend };
}

test('pipeline: real gesture sequence drives commands and the robot state end to end', async () => {
  const palm = toResult(makeHand(POSES.OPEN_PALM));
  const script = [EMPTY, ...Array(40).fill(palm), ...Array(70).fill(EMPTY)];
  const { run, controller, log, pipeline } = await makePipeline(script);
  const states = [];
  for (let i = 0; i < 110; i++) {
    run(i * 33);
    states.push(controller.state);
  }
  assert.equal(states[0], 'IDLE');
  assert.equal(states[1], 'TRACKING');
  assert.ok(states.includes('EXECUTING'), 'WAVE was executed');
  assert.equal(states.at(-1), 'IDLE');
  const accepted = log.filter((e) => e.type === 'command' && e.status === 'accepted').map((e) => e.name);
  assert.deepEqual(accepted, ['WAVE'], 'held open palm triggers exactly one WAVE');
  assert.equal(pipeline.session.trackingLosses, 1);
  assert.equal(pipeline.session.framesWithHand, 40);
  assert.equal(pipeline.metrics.pipelineMs.last, 5);
  assert.ok(log.some((e) => e.type === 'tracking' && e.status === 'acquired'));
});

test('pipeline: repeated tracker failures enter ERROR; success recovers', async () => {
  const err = new Error('inference failed');
  const { run, controller, log } = await makePipeline([err, err, err, EMPTY]);
  run(0);
  run(33);
  assert.notEqual(controller.state, 'ERROR');
  run(66);
  assert.equal(controller.state, 'ERROR');
  run(99);
  assert.equal(controller.state, 'IDLE');
  assert.ok(log.some((e) => e.type === 'tracker' && e.status === 'recovered'));
});

test('pipeline: tracker failure while pinching releases the hold', async () => {
  const pinch = toResult(makeHand(POSES.PINCH));
  const { run, controller } = await makePipeline([...Array(10).fill(pinch), new Error('x')]);
  for (let i = 0; i < 10; i++) run(i * 33);
  assert.equal(controller.state, 'CONTROL');
  run(400);
  assert.equal(controller.state, 'IDLE');
});

// ------------------------------------------------------------ metrics & assets

test('RateMeter measures event rate from timestamps', () => {
  const m = new RateMeter(1000);
  assert.equal(m.rate(0), null);
  for (let t = 0; t <= 2000; t += 40) m.tick(t);
  assert.ok(Math.abs(m.rate(2000) - 25) <= 1, String(m.rate(2000)));
  assert.equal(m.rate(5000), 0, 'decays to zero when events stop');
});

test('RollingStats keeps a bounded window', () => {
  const s = new RollingStats(3);
  [1, 2, 3, 4, NaN].forEach((v) => s.push(v));
  assert.equal(s.mean, 3);
  assert.equal(s.last, 4);
  assert.equal(s.percentile(95), 4);
  assert.equal(s.summary().samples, 3);
});

test('robot asset: no URL or a failing/timeout load falls back to the procedural robot', async () => {
  assert.equal((await resolveRobotAsset(null, { loadGLTF: async () => ({}) })).kind, 'procedural');
  const failed = await resolveRobotAsset('/models/robot.glb', {
    loadGLTF: async () => {
      throw new Error('404');
    },
  });
  assert.equal(failed.kind, 'procedural');
  assert.match(failed.reason, /404/);
  const empty = await resolveRobotAsset('/x.glb', { loadGLTF: async () => ({}) });
  assert.match(empty.reason, /no scene/);
  const slow = await resolveRobotAsset('/x.glb', { loadGLTF: () => new Promise(() => {}), timeoutMs: 20 });
  assert.match(slow.reason, /timed out/);
  const ok = await resolveRobotAsset('/x.glb', { loadGLTF: async () => ({ scene: {} }) });
  assert.equal(ok.kind, 'gltf');
});

test('EventBus isolates failing listeners', () => {
  const bus = new EventBus();
  let got = 0;
  const orig = console.error;
  console.error = () => {};
  bus.on('x', () => {
    throw new Error('bad listener');
  });
  bus.on('*', () => got++);
  bus.emit('x', { a: 1 });
  console.error = orig;
  assert.equal(got, 1);
  assert.equal(bus.listenerErrors, 1);
});
