import assert from 'node:assert/strict';
import { test } from 'node:test';
import { FeatureExtractionEngine } from '../src/gestures/FeatureExtractionEngine.js';
import { landmarkSetError } from '../src/gestures/landmarks.js';
import { makeHand, POSES } from './fixtures/hands.js';

const fx = new FeatureExtractionEngine({ mirror: true });
const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

test('extracts finger extension from a valid open palm', () => {
  const f = fx.extract(makeHand(POSES.OPEN_PALM), { aspect: 16 / 9 });
  assert.equal(f.valid, true);
  assert.equal(f.source, 'world');
  for (const name of ['index', 'middle', 'ring', 'pinky', 'thumb']) {
    assert.ok(f.fingers[name].extension > 0.9, `${name} should be extended`);
  }
  assert.ok(f.pinchDistance > 1, 'thumb and index tips are far apart');
});

test('curled fingers have low extension', () => {
  const f = fx.extract(makeHand(POSES.FIST));
  for (const name of ['index', 'middle', 'ring', 'pinky']) {
    assert.ok(f.fingers[name].extension < 0.1, `${name} should be curled`);
    assert.ok(f.fingers[name].straightness < 0.5);
  }
});

test('shape features are invariant to hand position, size and in-plane rotation', () => {
  const a = fx.extract(makeHand(POSES.POINT, { world: false }), { aspect: 16 / 9 });
  const b = fx.extract(makeHand(POSES.POINT, { world: false, cx: 0.3, cy: 0.4, size: 1.2, rotationDeg: 35 }), { aspect: 16 / 9 });
  assert.equal(a.source, 'image');
  for (const name of ['index', 'middle', 'ring', 'pinky', 'thumb']) close(a.fingers[name].extension, b.fingers[name].extension, 1e-9);
  close(a.pinchDistance, b.pinchDistance, 1e-9);
  // palm size in image units scales with apparent size
  close(b.palmSizeImage / a.palmSizeImage, 1.2 / 2.5, 1e-9);
});

test('image-only and world landmarks yield the same shape features for an ideal hand', () => {
  const w = fx.extract(makeHand(POSES.PINCH, { world: true }), { aspect: 4 / 3 });
  const i = fx.extract(makeHand(POSES.PINCH, { world: false, aspect: 4 / 3 }), { aspect: 4 / 3 });
  close(w.pinchDistance, i.pinchDistance, 1e-9);
});

test('mirroring flips the display-space x coordinate', () => {
  const hand = makeHand(POSES.OPEN_PALM, { cx: 0.2 });
  const mirrored = new FeatureExtractionEngine({ mirror: true }).extract(hand, { aspect: 1 });
  const raw = new FeatureExtractionEngine({ mirror: false }).extract(hand, { aspect: 1 });
  close(mirrored.palmCenterNorm.x, 1 - raw.palmCenterNorm.x, 1e-9);
});

test('reports the fraction of landmarks inside the image', () => {
  const f = fx.extract(makeHand(POSES.OPEN_PALM, { cx: 0.98 }), { aspect: 16 / 9 });
  assert.ok(f.inFrameRatio < 1);
});

test('rejects missing, truncated and non-finite landmark sets', () => {
  assert.equal(fx.extract(null).valid, false);
  assert.equal(fx.extract({}).valid, false);
  const hand = makeHand(POSES.OPEN_PALM);
  assert.match(fx.extract({ ...hand, landmarks: hand.landmarks.slice(0, 20) }).reason, /expected 21/);
  const bad = hand.landmarks.map((p) => ({ ...p }));
  bad[7].x = NaN;
  assert.match(fx.extract({ ...hand, landmarks: bad, worldLandmarks: null }).reason, /landmark 7/);
  assert.equal(landmarkSetError('nope'), 'landmarks missing');
});

test('falls back to image landmarks when world landmarks are malformed', () => {
  const hand = makeHand(POSES.OPEN_PALM);
  const f = fx.extract({ ...hand, worldLandmarks: hand.worldLandmarks.slice(0, 3) });
  assert.equal(f.valid, true);
  assert.equal(f.source, 'image');
});

test('rejects a degenerate hand with all points identical', () => {
  const p = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  const f = fx.extract({ landmarks: p });
  assert.equal(f.valid, false);
  assert.match(f.reason, /degenerate/);
});
