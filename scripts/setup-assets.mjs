#!/usr/bin/env node
/**
 * Prepares runtime assets so the app can run fully offline afterwards:
 *   1. copies the MediaPipe Tasks Vision WASM runtime from node_modules
 *      into public/mediapipe/wasm
 *   2. downloads the MediaPipe hand landmarker model (Apache-2.0) into
 *      public/models/hand_landmarker.task, unless it is already present
 *   3. downloads the optional models (Apache-2.0): gesture_recognizer.task (DL gestures),
 *      face_landmarker.task (face landmarks, blendshapes, head pose) and the BlazeFace
 *      detector (gaze fallback) into public/models
 *
 * Usage:
 *   node scripts/setup-assets.mjs           copy + download if missing
 *   node scripts/setup-assets.mjs --check   copy if missing; only report a missing model
 *
 * This is the only network access the project performs, and it happens at
 * setup time, never while the camera is running.
 */
import { copyFile, mkdir, rename, stat, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const WASM_SRC = join(ROOT, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const WASM_DST = join(ROOT, 'public', 'mediapipe', 'wasm');
const WASM_FILES = [
  'vision_wasm_internal.js',
  'vision_wasm_internal.wasm',
  'vision_wasm_nosimd_internal.js',
  'vision_wasm_nosimd_internal.wasm',
];
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const MODEL_DST = join(ROOT, 'public', 'models', 'hand_landmarker.task');
// Optional: face detector for the robot's gaze (BlazeFace short range, Apache-2.0, ~230 kB).
const FACE_URL =
  'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite';
const FACE_DST = join(ROOT, 'public', 'models', 'blaze_face_short_range.tflite');

const checkOnly = process.argv.includes('--check');

async function size(path) {
  try {
    return (await stat(path)).size;
  } catch {
    return -1;
  }
}

async function copyWasm() {
  await mkdir(WASM_DST, { recursive: true });
  for (const f of WASM_FILES) {
    const src = join(WASM_SRC, f);
    const dst = join(WASM_DST, f);
    const s = await size(src);
    if (s < 0) throw new Error(`missing ${src}; run "npm install" first`);
    if ((await size(dst)) !== s) {
      await copyFile(src, dst);
      console.log(`[setup] copied ${f}`);
    }
  }
}

async function downloadModel() {
  if ((await size(MODEL_DST)) > 1_000_000) return 'present';
  if (checkOnly) return 'missing';
  console.log(`[setup] downloading hand landmarker model\n        ${MODEL_URL}`);
  const res = await fetch(MODEL_URL);
  if (!res.ok) throw new Error(`model download failed: HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  if (buf.length < 1_000_000) throw new Error(`model download looks truncated (${buf.length} bytes)`);
  await mkdir(dirname(MODEL_DST), { recursive: true });
  const tmp = `${MODEL_DST}.part`;
  await writeFile(tmp, buf);
  await rename(tmp, MODEL_DST);
  console.log(`[setup] saved model (${(buf.length / 1e6).toFixed(1)} MB)`);
  return 'downloaded';
}

/**
 * Optional models (each has a fallback, so a failed download never breaks the app):
 *   gesture_recognizer.task       DL hand landmarks + DL gesture classes (else rule fallback)
 *   face_landmarker.task          face landmarks + blendshapes + head pose (else BlazeFace gaze only)
 *   blaze_face_short_range.tflite face detector, the gaze fallback
 */
const OPTIONAL_MODELS = [
  { file: 'gesture_recognizer.task', url: 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task', min: 5_000_000, without: 'gestures use the rule classifier (fallback)' },
  { file: 'face_landmarker.task', url: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', min: 2_000_000, without: 'no facial expressions / head pose' },
  { file: 'blaze_face_short_range.tflite', url: FACE_URL, min: 100_000, without: 'no gaze fallback' },
];

async function downloadOptional({ file, url, min, without }) {
  const dst = join(ROOT, 'public', 'models', file);
  if ((await size(dst)) > min) return 'present';
  if (checkOnly) return 'missing';
  console.log(`[setup] downloading ${file}
        ${url}`);
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const buf = Buffer.from(await res.arrayBuffer());
    if (buf.length < min) throw new Error(`looks truncated (${buf.length} bytes)`);
    await mkdir(dirname(dst), { recursive: true });
    await writeFile(`${dst}.part`, buf);
    await rename(`${dst}.part`, dst);
    return 'downloaded';
  } catch (err) {
    console.warn(`[setup] ${file} download failed (${err.message}); ${without}.`);
    console.warn(`        manual: curl -L -o public/models/${file} ${url}`);
    return 'missing';
  }
}

try {
  await copyWasm();
  const model = await downloadModel();
  for (const m of OPTIONAL_MODELS) {
    if ((await downloadOptional(m)) === 'missing' && checkOnly) console.warn(`[setup] ${m.file} is missing: ${m.without} (run "npm run setup").`);
  }
  if (model === 'missing') {
    console.warn('[setup] hand landmarker model is missing. Run "npm run setup" once (requires internet).');
    console.warn('        The app will start and report the missing model in the HUD.');
  } else {
    console.log(`[setup] assets ready (model ${model})`);
  }
} catch (err) {
  console.error(`[setup] ${err.message}`);
  process.exitCode = checkOnly ? 0 : 1;
}
