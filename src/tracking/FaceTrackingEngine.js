/**
 * FaceTrackingEngine: throttled local face detection on the frame the camera
 * already delivers (no second stream). Used for the robot's gaze only; a missing
 * model or a failure disables gaze and never affects hand tracking.
 *
 * Output faces are image-normalised centres and sizes:
 *   { x, y, w, h, score }   (x, y = box centre in [0, 1], top-left image origin)
 * The coordinator mirrors / maps them to the screen for AttentionController.
 * Status: disabled | uninitialized | loading | ready | error | closed.
 */
import { pickBlendshapes, headCueFromLandmarks } from '../perception/faceAnalysis.js';

export class FaceTrackingEngine {
  /** @param {{ backend: any|null, intervalMs: number, minScore: number, now?: () => number }} deps */
  constructor(deps) {
    this.backend = deps.backend;
    this.intervalMs = deps.intervalMs;
    this.minScore = deps.minScore;
    this.now = deps.now ?? (() => performance.now());
    this.status = deps.backend ? 'uninitialized' : 'disabled';
    this.error = null;
    this.delegate = null;
    this.lastRun = -Infinity;
    this.lastTimestamp = -Infinity;
    this.inferenceMs = null;
  }

  async init() {
    if (!this.backend) return { ok: false };
    this.status = 'loading';
    try {
      const info = await this.backend.init();
      this.delegate = info?.delegate ?? null;
      this.model = info?.model ?? null;
      this.status = 'ready';
      return { ok: true };
    } catch (err) {
      this.status = 'error';
      this.error = err?.message ?? String(err);
      return { ok: false, error: this.error };
    }
  }

  /** @returns {boolean} a detection is due for a frame at time t */
  due(t) {
    return this.status === 'ready' && t - this.lastRun >= this.intervalMs;
  }

  /**
   * @param {any} source video element
   * @param {number} t frame time (ms)
   * @param {{ width: number, height: number }} size source size in pixels
   * @returns {null | { x: number, y: number, w: number, h: number, score: number }[]} null = did not run
   */
  detect(source, t, size) {
    if (!this.due(t) || !(size?.width > 0 && size?.height > 0)) return null;
    this.lastRun = t;
    const ts = t > this.lastTimestamp ? t : this.lastTimestamp + 0.001;
    this.lastTimestamp = ts;
    const t0 = this.now();
    let result;
    try {
      result = this.backend.detect(source, ts);
    } catch {
      return null;
    }
    this.inferenceMs = this.now() - t0;
    return parseFaces(result, size, this.minScore);
  }

  close() {
    try {
      this.backend?.close?.();
    } catch {
      /* ignore */
    }
    if (this.backend) this.status = 'closed';
  }
}

/**
 * MediaPipe result -> validated normalised faces.
 *   FaceLandmarker: box from the 478 normalised landmarks; `bs` (blendshape subset) and
 *                   `matrix` (head pose, 16 column-major) for expressions / head pose
 *   FaceDetector:   pixel boxes with a detection score (fallback: gaze only)
 */
export function parseFaces(result, size, minScore) {
  const out = [];
  if (Array.isArray(result?.faceLandmarks)) {
    result.faceLandmarks.forEach((lms, i) => {
      if (!Array.isArray(lms) || lms.length < 10) return;
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const p of lms) {
        if (!Number.isFinite(p?.x) || !Number.isFinite(p?.y)) return;
        x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y);
      }
      const w = x1 - x0;
      const h = y1 - y0;
      if (!(w > 0 && w <= 1.5 && h > 0 && h <= 1.5)) return;
      const m = result.facialTransformationMatrixes?.[i]?.data;
      out.push({
        x: (x0 + x1) / 2, y: (y0 + y1) / 2, w, h, score: 1,
        bs: pickBlendshapes(result.faceBlendshapes?.[i]?.categories),
        matrix: m && m.length === 16 ? Array.from(m, (v) => Math.round(v * 1e4) / 1e4) : null,
        cue: headCueFromLandmarks(lms), // head-turn cues, image space (faceAnalysis.js)
      });
    });
    return out;
  }
  for (const d of Array.isArray(result?.detections) ? result.detections : []) {
    const b = d?.boundingBox;
    const score = d?.categories?.[0]?.score;
    const vals = [b?.originX, b?.originY, b?.width, b?.height, score];
    if (!vals.every((v) => typeof v === 'number' && Number.isFinite(v))) continue;
    const w = b.width / size.width;
    const h = b.height / size.height;
    if (score < minScore || !(w > 0 && w <= 1.5 && h > 0 && h <= 1.5)) continue;
    out.push({ x: (b.originX + b.width / 2) / size.width, y: (b.originY + b.height / 2) / size.height, w, h, score });
  }
  return out;
}
