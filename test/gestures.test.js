import assert from 'node:assert/strict';
import { test } from 'node:test';
import { DEFAULT_CONFIG } from '../src/config/defaultConfig.js';
import { FeatureExtractionEngine } from '../src/gestures/FeatureExtractionEngine.js';
import { GestureRecognitionEngine } from '../src/gestures/GestureRecognitionEngine.js';
import { GestureStabilizer } from '../src/gestures/GestureStabilizer.js';
import { StaticGestureClassifier } from '../src/gestures/StaticGestureClassifier.js';
import { SwipeDetector } from '../src/gestures/SwipeDetector.js';
import { makeHand, POSES } from './fixtures/hands.js';

const cfg = () => JSON.parse(JSON.stringify(DEFAULT_CONFIG.gestures));
const fx = new FeatureExtractionEngine({ mirror: true });
const classifier = new StaticGestureClassifier(cfg());
const features = (pose, view) => fx.extract(makeHand(pose, view), { aspect: 16 / 9 });

// ---------------------------------------------------------------- static rules

for (const [pose, expected] of [
  ['OPEN_PALM', 'OPEN_PALM'],
  ['FIST', 'FIST'],
  ['POINT', 'POINT'],
  ['PINCH', 'PINCH'],
  ['PINCH_CURLED', 'PINCH'],
  ['FLAT_HAND', 'OPEN_PALM'],
  ['VICTORY', 'TWO_FINGERS'],
  ['THREE_FINGERS', 'THREE_FINGERS'],
]) {
  test(`classifies synthetic ${pose} as ${expected}`, () => {
    for (const view of [{}, { world: false }, { rotationDeg: 40, size: 1.5 }, { rotationDeg: -60, cx: 0.3 }]) {
      const r = classifier.classify(features(POSES[pose], view));
      assert.equal(r.gesture, expected, JSON.stringify({ view, r }));
      assert.ok(r.confidence >= 0.6);
    }
  });
}

for (const pose of ['CLAW', 'THUMBS_UP', 'VICTORY_THUMB_OUT', 'HALF_MIDDLE']) {
  test(`out-of-vocabulary / ambiguous shape ${pose} stays UNKNOWN`, () => {
    const r = classifier.classify(features(POSES[pose]));
    assert.equal(r.gesture, 'UNKNOWN');
    assert.ok(r.reason);
  });
}

test('no hand yields NONE; low tracking score and partial visibility yield UNKNOWN', () => {
  assert.equal(classifier.classify(null).gesture, 'NONE');
  assert.equal(classifier.classify({ valid: false, reason: 'x' }).gesture, 'NONE');
  const low = classifier.classify(features(POSES.FIST, { score: 0.3 }));
  assert.equal(low.gesture, 'UNKNOWN');
  assert.match(low.reason, /tracking confidence/);
  const edge = classifier.classify(features(POSES.FIST, { cx: 0.99 }));
  assert.equal(edge.gesture, 'UNKNOWN');
  assert.match(edge.reason, /outside/);
});

test('disabled gestures cannot win', () => {
  const r = classifier.classify(features(POSES.FIST), ['OPEN_PALM', 'POINT', 'PINCH']);
  assert.equal(r.gesture, 'UNKNOWN');
});

test('ambiguity margin: two near-equal scores yield UNKNOWN', () => {
  const c = new StaticGestureClassifier({ minScore: 0.5, minMargin: 0.2 });
  c.scores = () => ({ OPEN_PALM: 0.8, FIST: 0, POINT: 0.7, PINCH: 0 });
  const r = c.classify({ valid: true, score: 1, inFrameRatio: 1 });
  assert.equal(r.gesture, 'UNKNOWN');
  assert.match(r.reason, /ambiguous/);
});

// ---------------------------------------------------------------- stabilizer

test('stabilizer requires a continuous hold and emits exactly one start', () => {
  const s = new GestureStabilizer({ exitMs: 150, holdMs: () => 300 });
  const events = [];
  for (let t = 0; t <= 2000; t += 33) events.push(...s.update('FIST', 0.9, t));
  assert.deepEqual(events.map((e) => e.type), ['start']);
  assert.ok(events[0].t >= 300);
  assert.equal(s.stable, 'FIST');
});

test('stabilizer ignores short flicker and single-frame dropouts', () => {
  const s = new GestureStabilizer({ exitMs: 150, holdMs: () => 300 });
  const ev = [];
  // flicker shorter than holdMs never becomes stable
  for (let t = 0; t < 200; t += 33) ev.push(...s.update('POINT', 0.9, t));
  for (let t = 200; t < 400; t += 33) ev.push(...s.update('UNKNOWN', 0, t));
  assert.equal(ev.length, 0);
  // stable gesture survives a 1-frame misclassification
  for (let t = 400; t < 900; t += 33) ev.push(...s.update('POINT', 0.9, t));
  ev.push(...s.update('OPEN_PALM', 0.7, 900));
  for (let t = 933; t < 1200; t += 33) ev.push(...s.update('POINT', 0.9, t));
  assert.deepEqual(ev.map((e) => `${e.type}:${e.gesture}`), ['start:POINT']);
});

test('stabilizer releases after exitMs and re-triggers only after a new hold', () => {
  const s = new GestureStabilizer({ exitMs: 150, holdMs: () => 200 });
  const ev = [];
  for (let t = 0; t <= 500; t += 50) ev.push(...s.update('FIST', 0.9, t));
  for (let t = 550; t <= 800; t += 50) ev.push(...s.update('NONE', 0, t));
  for (let t = 850; t <= 1200; t += 50) ev.push(...s.update('FIST', 0.9, t));
  assert.deepEqual(ev.map((e) => e.type), ['start', 'end', 'start']);
});

test('stabilizer release() ends the gesture immediately', () => {
  const s = new GestureStabilizer({ exitMs: 150, holdMs: () => 0 });
  s.update('PINCH', 1, 0);
  assert.deepEqual(s.release(10).map((e) => e.type), ['end']);
  assert.equal(s.stable, 'NONE');
  assert.deepEqual(s.release(20), []);
});

// ---------------------------------------------------------------- swipes

function runSwipe(detector, { fromX, toX, durationMs, y0 = 0.5, y1 = 0.5, t0 = 0, palm = 0.2, step = 33 }) {
  const out = [];
  for (let t = t0; t <= t0 + durationMs; t += step) {
    const k = (t - t0) / durationMs;
    const r = detector.update(t, { x: fromX + (toX - fromX) * k, y: y0 + (y1 - y0) * k }, palm);
    if (r) out.push({ ...r, t });
  }
  return out;
}

test('fast horizontal motion produces one swipe in the right direction', () => {
  const d = new SwipeDetector(cfg().swipe);
  const left = runSwipe(d, { fromX: 1.2, toX: 0.5, durationMs: 300 });
  assert.equal(left.length, 1);
  assert.equal(left[0].gesture, 'SWIPE_LEFT');
  const d2 = new SwipeDetector(cfg().swipe);
  const right = runSwipe(d2, { fromX: 0.5, toX: 1.2, durationMs: 300 });
  assert.deepEqual(right.map((r) => r.gesture), ['SWIPE_RIGHT']);
});

test('slow, short or vertical motion does not produce a swipe', () => {
  assert.equal(runSwipe(new SwipeDetector(cfg().swipe), { fromX: 0.5, toX: 1.2, durationMs: 2000 }).length, 0);
  assert.equal(runSwipe(new SwipeDetector(cfg().swipe), { fromX: 0.5, toX: 0.6, durationMs: 100 }).length, 0);
  assert.equal(runSwipe(new SwipeDetector(cfg().swipe), { fromX: 0.5, toX: 0.9, y0: 0.1, y1: 0.9, durationMs: 300 }).length, 0);
});

test('distances are normalised by palm size (same motion, closer hand = smaller relative travel)', () => {
  // 0.5 image-heights of travel is 2.5 palms for palm 0.2 but only 1.25 palms for palm 0.4
  assert.equal(runSwipe(new SwipeDetector(cfg().swipe), { fromX: 0.5, toX: 1.0, durationMs: 250, palm: 0.2 }).length, 1);
  assert.equal(runSwipe(new SwipeDetector(cfg().swipe), { fromX: 0.5, toX: 1.0, durationMs: 250, palm: 0.4 }).length, 0);
});

test('return stroke in the opposite direction is suppressed; same direction after cooldown is allowed', () => {
  const d = new SwipeDetector(cfg().swipe);
  const a = runSwipe(d, { fromX: 0.5, toX: 1.2, durationMs: 300, t0: 0 });
  const back = runSwipe(d, { fromX: 1.2, toX: 0.5, durationMs: 300, t0: 400 });
  const again = runSwipe(d, { fromX: 0.5, toX: 1.2, durationMs: 300, t0: 1500 });
  assert.equal(a.length, 1);
  assert.equal(back.length, 0);
  assert.deepEqual(again.map((r) => r.gesture), ['SWIPE_RIGHT']);
});

test('a tracking gap resets the swipe history', () => {
  const d = new SwipeDetector(cfg().swipe);
  d.update(0, { x: 0.5, y: 0.5 }, 0.2);
  d.update(33, { x: 0.55, y: 0.5 }, 0.2);
  // 300 ms gap (> maxGapMs) then a jump: must not count as a swipe
  assert.equal(d.update(333, { x: 1.2, y: 0.5 }, 0.2), null);
  assert.equal(d.history.length, 1);
});

// ---------------------------------------------------------------- engine

function feed(engine, pose, t0, t1, view = {}, step = 33) {
  const events = [];
  let last;
  for (let t = t0; t <= t1; t += step) {
    last = engine.update(fx.extract(makeHand(pose, view), { aspect: 16 / 9 }), t);
    events.push(...last.events);
  }
  return { events, last };
}

test('engine: held gesture emits one start; hand loss emits end', () => {
  const e = new GestureRecognitionEngine(cfg());
  const { events, last } = feed(e, POSES.OPEN_PALM, 0, 3000);
  assert.deepEqual(events.map((x) => `${x.type}:${x.gesture}`), ['start:OPEN_PALM']);
  assert.equal(last.stable.gesture, 'OPEN_PALM');
  const lost = e.update(null, 3033);
  assert.equal(lost.hand, false);
  assert.deepEqual(lost.events.map((x) => x.type), ['end']);
  assert.equal(e.update(null, 3066).events.length, 0);
});

test('engine: requireStill gestures do not start while the hand moves fast', () => {
  const e = new GestureRecognitionEngine(cfg());
  const events = [];
  let moving = 0;
  let frames = 0;
  // open palm waving vertically (not a swipe) for 1.5 s, staying inside the image
  for (let t = 0; t <= 1500; t += 33) {
    const cy = 0.65 + 0.12 * Math.sin(t / 50);
    const r = e.update(fx.extract(makeHand(POSES.OPEN_PALM, { cy }), { aspect: 16 / 9 }), t);
    events.push(...r.events);
    if (t > 200) {
      frames++;
      if (r.raw.label === 'MOVING') moving++;
      assert.notEqual(r.raw.gesture, 'UNKNOWN');
    }
  }
  assert.ok(moving / frames > 0.8, `${moving}/${frames} frames flagged MOVING`);
  assert.equal(events.filter((x) => x.type === 'start').length, 0);
});

test('engine: swipe fires once and suppresses static onsets right after it', () => {
  const e = new GestureRecognitionEngine(cfg());
  const events = [];
  for (let t = 0; t <= 300; t += 33) {
    const cx = 0.75 - (t / 300) * 0.5; // image x decreasing = display x increasing (mirrored)
    events.push(...e.update(fx.extract(makeHand(POSES.OPEN_PALM, { cx }), { aspect: 16 / 9 }), t).events);
  }
  const swipes = events.filter((x) => x.type === 'event');
  assert.deepEqual(swipes.map((x) => x.gesture), ['SWIPE_RIGHT']);
  const t1 = 330;
  const after = feed(e, POSES.OPEN_PALM, t1, t1 + 600, { cx: 0.25 });
  assert.equal(after.events.filter((x) => x.type === 'start').length, 0, 'suppressed during postSwipeSuppressMs');
  const later = feed(e, POSES.OPEN_PALM, t1 + 633, t1 + 2000, { cx: 0.25 });
  assert.equal(later.events.filter((x) => x.type === 'start').length, 1, 'deliberate hold afterwards still works');
});

test('engine: swipes are not detected while pinching (pinch-drag)', () => {
  const e = new GestureRecognitionEngine(cfg());
  const events = [];
  for (let t = 0; t <= 600; t += 33) {
    const cx = t < 300 ? 0.5 : 0.5 - ((t - 300) / 300) * 0.5;
    events.push(...e.update(fx.extract(makeHand(POSES.PINCH, { cx }), { aspect: 16 / 9 }), t).events);
  }
  assert.equal(events.filter((x) => x.type === 'event').length, 0);
  assert.deepEqual(events.filter((x) => x.type === 'start').map((x) => x.gesture), ['PINCH']);
});

test('engine: gesture definitions are configurable (disable + hold time)', () => {
  const c = cfg();
  c.definitions.FIST.enabled = false;
  c.definitions.POINT.holdMs = 1000;
  const e = new GestureRecognitionEngine(c);
  assert.equal(feed(e, POSES.FIST, 0, 2000).events.length, 0);
  e.reset();
  const p = feed(e, POSES.POINT, 5000, 5900);
  assert.equal(p.events.length, 0, 'not yet held for 1000 ms');
  const p2 = feed(e, POSES.POINT, 5933, 6200);
  assert.deepEqual(p2.events.map((x) => x.gesture), ['POINT']);
});
