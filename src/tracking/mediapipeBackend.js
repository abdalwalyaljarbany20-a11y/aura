import { FaceDetector, FaceLandmarker, FilesetResolver, GestureRecognizer, HandLandmarker } from '@mediapipe/tasks-vision';

/**
 * MediaPipe Tasks Vision backends (WASM, run locally). Models and runtime are served
 * from this app's own origin (see scripts/setup-assets.mjs); no frame leaves the browser.
 *
 * Hand: GestureRecognizer (deep-learning hand landmarker + learned gesture classifier in
 * one bundle, so there is still exactly one hand pipeline). If its model is missing, the
 * plain HandLandmarker is used and the result simply carries no `gestures` (the rule
 * classifier then decides, labelled as the fallback).
 * Face: FaceLandmarker (detector + 478 landmarks + 52 blendshapes + head-pose matrix), or
 * the BlazeFace detector alone if the landmarker model is missing (gaze only).
 * Each tries the configured delegate first and falls back to CPU if GPU init fails.
 */

async function modelPresent(url) {
  if (!url) return false;
  const head = await fetch(url, { method: 'HEAD' }).catch(() => null);
  const type = head?.headers.get('content-type') ?? '';
  return !!head && head.ok && !type.includes('text/html');
}

async function withDelegate(delegate, create) {
  try {
    return { task: await create(delegate), delegate };
  } catch (err) {
    if (delegate !== 'GPU') throw err;
    console.warn('[mediapipe] GPU delegate failed, falling back to CPU', err);
    return { task: await create('CPU'), delegate: 'CPU (GPU failed)' };
  }
}

export function createMediaPipeBackend(cfg) {
  let task = null;
  let kind = null; // 'gesture' | 'landmarks'

  const common = {
    runningMode: 'VIDEO',
    numHands: cfg.numHands,
    minHandDetectionConfidence: cfg.minHandDetectionConfidence,
    minHandPresenceConfidence: cfg.minHandPresenceConfidence,
    minTrackingConfidence: cfg.minTrackingConfidence,
  };

  return {
    async init() {
      const fileset = await FilesetResolver.forVisionTasks(cfg.wasmPath);
      if (await modelPresent(cfg.gestureModelAssetPath)) {
        const r = await withDelegate(cfg.delegate, (delegate) => GestureRecognizer.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: cfg.gestureModelAssetPath, delegate },
          ...common,
          // keep every class score: the fusion applies its own threshold and reports it
          cannedGesturesClassifierOptions: { scoreThreshold: 0 },
        }));
        task = r.task;
        kind = 'gesture';
        return { delegate: r.delegate, model: 'GestureRecognizer (DL landmarks + DL gestures)' };
      }
      if (!(await modelPresent(cfg.modelAssetPath))) {
        throw new Error(`hand model not found at ${cfg.gestureModelAssetPath ?? ''} or ${cfg.modelAssetPath} (run "npm run setup")`);
      }
      const r = await withDelegate(cfg.delegate, (delegate) => HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: cfg.modelAssetPath, delegate },
        ...common,
      }));
      task = r.task;
      kind = 'landmarks';
      return { delegate: r.delegate, model: 'HandLandmarker (gesture model missing: rule fallback)' };
    },
    detect(source, timestampMs) {
      return kind === 'gesture' ? task.recognizeForVideo(source, timestampMs) : task.detectForVideo(source, timestampMs);
    },
    close() {
      task?.close();
      task = null;
    },
  };
}

/** Face backend: FaceLandmarker when available, else the BlazeFace detector. */
export function createMediaPipeFaceBackend(cfg, tracking) {
  let task = null;
  let kind = null; // 'landmarker' | 'detector'
  return {
    get kind() {
      return kind;
    },
    async init() {
      const fileset = await FilesetResolver.forVisionTasks(tracking.wasmPath);
      if (await modelPresent(cfg.modelAssetPath)) {
        const r = await withDelegate(tracking.delegate, (delegate) => FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: cfg.modelAssetPath, delegate },
          runningMode: 'VIDEO',
          numFaces: cfg.numFaces,
          minFaceDetectionConfidence: cfg.minScore,
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: true,
        }));
        task = r.task;
        kind = 'landmarker';
        return { delegate: r.delegate, model: 'FaceLandmarker' };
      }
      if (!(await modelPresent(cfg.detectorAssetPath))) throw new Error(`face model not found at ${cfg.modelAssetPath} (run "npm run setup")`);
      const r = await withDelegate(tracking.delegate, (delegate) => FaceDetector.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: cfg.detectorAssetPath, delegate },
        runningMode: 'VIDEO',
        minDetectionConfidence: cfg.minScore,
      }));
      task = r.task;
      kind = 'detector';
      return { delegate: r.delegate, model: 'FaceDetector (landmarker missing: no expressions)' };
    },
    detect(source, timestampMs) {
      return task.detectForVideo(source, timestampMs);
    },
    close() {
      task?.close();
      task = null;
    },
  };
}
