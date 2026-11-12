import { landmarkSetError } from '../gestures/landmarks.js';

/**
 * HandTrackingEngine
 *
 * Backend-agnostic wrapper around a hand-landmark model. A backend must
 * implement:
 *
 *   init(): Promise<{ delegate: string }>
 *   detect(source, timestampMs): { landmarks, worldLandmarks, handedness }  (MediaPipe result shape)
 *   close(): void
 *
 * The engine validates every result, converts it into plain observations
 * and selects a primary hand. It contains no rendering code.
 *
 * Observation: { landmarks: {x,y,z}[21], worldLandmarks: {x,y,z}[21]|null,
 *                handedness: 'Left'|'Right'|null, score: number|null }
 *   - landmarks are normalised to [0,1] image coordinates (unmirrored)
 *   - score is MediaPipe's handedness classification score, used as the
 *     per-hand tracking confidence (the JS API exposes no other per-hand score)
 */
export class HandTrackingEngine {
  /** @param {{ backend: any, bus?: any, now?: () => number }} deps */
  constructor(deps) {
    this.backend = deps.backend;
    this.bus = deps.bus ?? null;
    this.now = deps.now ?? (() => performance.now());
    /** @type {'uninitialized'|'loading'|'ready'|'error'|'closed'} */
    this.status = 'uninitialized';
    this.error = null;
    this.delegate = null;
    this.model = null;
    this.lastTimestamp = -Infinity;
    this.stats = { frames: 0, failures: 0, rejectedHands: 0 };
    this._primaryHandedness = null;
  }

  async init() {
    this.status = 'loading';
    this.error = null;
    try {
      const info = await this.backend.init();
      this.delegate = info?.delegate ?? null;
      this.model = info?.model ?? null;
      this.status = 'ready';
      this.bus?.emit('tracker', { status: 'ready', delegate: this.delegate });
      return { ok: true };
    } catch (err) {
      this.status = 'error';
      this.error = err?.message ?? String(err);
      this.bus?.emit('tracker', { status: 'error', error: this.error });
      return { ok: false, error: this.error };
    }
  }

  /**
   * Run the model on one frame.
   * @param {any} source video element / image
   * @param {number} timestampMs must increase monotonically (MediaPipe requirement)
   * @returns {{ ok: boolean, hands: object[], primary: object|null, inferenceMs: number, error?: string }}
   */
  detect(source, timestampMs) {
    if (this.status !== 'ready') return { ok: false, hands: [], primary: null, inferenceMs: 0, error: `tracker ${this.status}` };
    // MediaPipe rejects non-increasing timestamps; never feed it one.
    const ts = timestampMs > this.lastTimestamp ? timestampMs : this.lastTimestamp + 0.001;
    this.lastTimestamp = ts;

    const t0 = this.now();
    let result;
    try {
      result = this.backend.detect(source, ts);
    } catch (err) {
      this.stats.failures++;
      return { ok: false, hands: [], primary: null, inferenceMs: this.now() - t0, error: err?.message ?? String(err) };
    }
    const inferenceMs = this.now() - t0;
    this.stats.frames++;
    const hands = this.parse(result);
    return { ok: true, hands, primary: this.selectPrimary(hands), inferenceMs };
  }

  /**
   * Convert and validate a MediaPipe-shaped result. Malformed hands are
   * dropped (and counted), never passed downstream.
   * @param {any} result
   */
  parse(result) {
    const hands = [];
    const lms = Array.isArray(result?.landmarks) ? result.landmarks : [];
    for (let i = 0; i < lms.length; i++) {
      const landmarks = lms[i];
      if (landmarkSetError(landmarks)) {
        this.stats.rejectedHands++;
        continue;
      }
      const world = result.worldLandmarks?.[i];
      const cat = (result.handedness ?? result.handednesses)?.[i]?.[0];
      // GestureRecognizer: the learned classifier's categories for this hand (best first)
      const g = Array.isArray(result.gestures?.[i]) ? result.gestures[i] : null;
      const model = g ? toModelGesture(g) : null;
      hands.push({
        landmarks: landmarks.map((p) => ({ x: p.x, y: p.y, z: p.z ?? 0 })),
        worldLandmarks: world && !landmarkSetError(world) ? world.map((p) => ({ x: p.x, y: p.y, z: p.z ?? 0 })) : null,
        // MediaPipe labels handedness assuming a mirrored image; we pass it through unchanged.
        handedness: cat?.categoryName || null,
        score: typeof cat?.score === 'number' ? cat.score : null,
        model,
      });
    }
    return hands;
  }

  /**
   * Primary hand = the one that keeps the previous primary's handedness if
   * still present (avoids switching between two hands), else the hand with
   * the largest apparent size (closest to the camera).
   * @param {object[]} hands
   */
  selectPrimary(hands) {
    if (!hands.length) {
      this._primaryHandedness = null;
      return null;
    }
    let primary = null;
    if (this._primaryHandedness) primary = hands.find((h) => h.handedness === this._primaryHandedness) ?? null;
    if (!primary) {
      const size = (h) => {
        const xs = h.landmarks.map((p) => p.x);
        const ys = h.landmarks.map((p) => p.y);
        return (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
      };
      primary = hands.reduce((a, b) => (size(b) > size(a) ? b : a));
    }
    this._primaryHandedness = primary.handedness;
    return primary;
  }

  close() {
    if (this.status === 'closed') return;
    try {
      this.backend.close?.();
    } catch (err) {
      console.warn('[HandTrackingEngine] close failed', err);
    }
    this.status = 'closed';
  }
}

/** GestureRecognizer categories -> { label, score, scores } (validated; label of the best class). */
export function toModelGesture(categories) {
  const scores = {};
  let best = null;
  for (const c of categories) {
    if (typeof c?.categoryName !== 'string' || !(typeof c.score === 'number' && Number.isFinite(c.score))) continue;
    scores[c.categoryName] = c.score;
    if (!best || c.score > best.score) best = { label: c.categoryName, score: c.score };
  }
  return best ? { ...best, scores } : null;
}
