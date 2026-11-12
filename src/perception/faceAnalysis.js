/**
 * Face analysis from MediaPipe FaceLandmarker outputs (shared by both engines: the Python
 * engine forwards the same model outputs, the browser runs the model itself).
 *
 * Inputs per face:
 *   bs       blendshape scores (a subset of the 52 ARKit-style categories) from the learned
 *            blendshape model, each in [0, 1]
 *   matrix   facial transformation matrix (16 numbers, column-major), head pose
 * Outputs:
 *   headPose  { yawDeg, pitchDeg, rollDeg } from the rotation part of the matrix
 *   gaze      { x, y } in [-1, 1] from the eye-look blendshapes (+x = toward screen right in
 *             the mirrored view, +y = down)
 *   expression (FaceExpressionEngine): HAPPY | SAD | SURPRISED | NEUTRAL with a confidence.
 *
 * Honesty note: the blendshape scores come from a deep-learning model; the expression label
 * is a fixed, documented mapping of those scores (below), smoothed over time. It is an
 * estimate of visible facial movement, not a measurement of anyone's emotion.
 */

/** Blendshape categories the app uses (the rest are not transferred). */
export const BLENDSHAPE_KEYS = Object.freeze([
  'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft', 'mouthFrownRight', 'jawOpen',
  'browInnerUp', 'browDownLeft', 'browDownRight', 'browOuterUpLeft', 'browOuterUpRight',
  'eyeWideLeft', 'eyeWideRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'cheekSquintLeft', 'cheekSquintRight',
  'mouthPressLeft', 'mouthPressRight',
  'eyeLookInLeft', 'eyeLookOutLeft', 'eyeLookInRight', 'eyeLookOutRight',
  'eyeLookUpLeft', 'eyeLookUpRight', 'eyeLookDownLeft', 'eyeLookDownRight',
]);

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : 0);
const DEG = 180 / Math.PI;

/** Category list ([{ categoryName, score }]) -> { key: score } for BLENDSHAPE_KEYS. */
export function pickBlendshapes(categories) {
  const out = {};
  for (const c of Array.isArray(categories) ? categories : []) {
    if (BLENDSHAPE_KEYS.includes(c?.categoryName) && Number.isFinite(c.score)) out[c.categoryName] = Math.round(clamp01(c.score) * 1000) / 1000;
  }
  return out;
}

/** Column-major 4x4 -> head pose in degrees (yaw about y, pitch about x, roll about z). */
export function headPoseFromMatrix(m) {
  if (!Array.isArray(m) && !(m instanceof Float32Array) && !(m instanceof Float64Array)) return null;
  if (m.length < 11 || ![0, 1, 2, 4, 5, 6, 8, 9, 10].every((i) => Number.isFinite(m[i]))) return null;
  const r00 = m[0], r10 = m[1], r20 = m[2], r21 = m[6], r22 = m[10];
  const yaw = Math.asin(Math.max(-1, Math.min(1, -r20)));
  const pitch = Math.atan2(r21, r22);
  const roll = Math.atan2(r10, r00);
  return { yawDeg: yaw * DEG, pitchDeg: pitch * DEG, rollDeg: roll * DEG };
}

/** FaceLandmarker indices used for the head-turn cues (canonical 478-point mesh). */
export const CUE_LANDMARKS = Object.freeze({ noseTip: 1, cheekRight: 234, cheekLeft: 454, forehead: 10, chin: 152 });
/**
 * Calibrated on real FaceLandmarker output (docs/VALIDATION.md section 14): the nose tip
 * sits at this fraction of the forehead -> chin distance for a frontal face.
 */
export const PITCH_NEUTRAL = 0.565;
/** Half-range of the pitch ratio mapped to +-1 (yaw cue measured 0.021 per deg of head yaw). */
export const PITCH_RANGE = 0.045; // about +-25 deg of head pitch (measured 0.0018 per deg)

/**
 * Head-turn cues from face landmarks, in IMAGE space (not mirrored), each in [-1, 1]:
 *   yaw   = nose tip offset from the cheek midpoint / half the cheek distance
 *           (+ = nose toward image right)
 *   pitch = (nose position between forehead and chin - PITCH_NEUTRAL) / PITCH_RANGE
 *           (+ = looking down, - = looking up)
 * Ratios along one image axis only, so they do not depend on the image aspect or the
 * distance to the camera. Identical arithmetic in aura_cv.faces.head_cue (parity-tested).
 * @param {{x:number,y:number}[]} lms
 * @returns {{ yaw: number, pitch: number } | null}
 */
export function headCueFromLandmarks(lms) {
  if (!Array.isArray(lms) || lms.length < 455) return null;
  const { noseTip, cheekRight, cheekLeft, forehead, chin } = CUE_LANDMARKS;
  const p = [noseTip, cheekRight, cheekLeft, forehead, chin].map((i) => lms[i]);
  if (!p.every((q) => Number.isFinite(q?.x) && Number.isFinite(q?.y))) return null;
  const [n, cr, cl, fh, ch] = p;
  const half = Math.abs(cl.x - cr.x) / 2;
  const span = ch.y - fh.y;
  if (!(half > 1e-4) || !(span > 1e-4)) return null;
  const c = (v, m) => Math.max(-m, Math.min(m, v));
  const yaw = c((n.x - (cl.x + cr.x) / 2) / half, 1);
  // pitch keeps a wider raw range: the neutral ratio differs per face and camera height, and
  // HeadCueCalibrator subtracts the user's own resting value before the robot uses it
  const pitch = c(((n.y - fh.y) / span - PITCH_NEUTRAL) / PITCH_RANGE, PITCH_RAW_MAX);
  return { yaw: Math.round(yaw * 1e4) / 1e4, pitch: Math.round(pitch * 1e4) / 1e4 };
}

export const PITCH_RAW_MAX = 4;

/**
 * Per-user resting head pose. Live (VALIDATION.md §15) a neutral head gave a pitch cue of -1
 * (saturated "looking up"), because the neutral ratio depends on the face and the camera height.
 * The first `calibrateMs` of visible face learn the resting cue quickly; afterwards it adapts
 * slowly (`baselineMs`), so a held head turn is mostly kept. A face absent for `relearnMs` starts
 * a new quick calibration (a new user may have sat down). Output = raw - rest, clamped to ±1.5.
 */
export class HeadCueCalibrator {
  constructor(cfg = {}) {
    this.cfg = { calibrateMs: 1500, baselineMs: 30000, relearnMs: 5000, enabled: true, ...cfg };
    this.rest = null;
    this.seenMs = 0;
    this.lostMs = 0;
    this._t = null;
  }

  update(t, cue) {
    const dt = this._t === null ? 33 : Math.max(0, Math.min(200, t - this._t));
    this._t = t;
    if (!cue || !Number.isFinite(cue.yaw) || !Number.isFinite(cue.pitch)) {
      this.lostMs += dt;
      if (this.lostMs >= this.cfg.relearnMs) this.seenMs = 0;
      return null;
    }
    this.lostMs = 0;
    if (!this.cfg.enabled) return { yaw: cue.yaw, pitch: Math.max(-1.5, Math.min(1.5, cue.pitch)) };
    if (!this.rest) this.rest = { yaw: cue.yaw, pitch: cue.pitch };
    const tau = this.seenMs < this.cfg.calibrateMs ? 250 : this.cfg.baselineMs;
    const k = 1 - Math.exp(-dt / tau);
    this.rest.yaw += (cue.yaw - this.rest.yaw) * k;
    this.rest.pitch += (cue.pitch - this.rest.pitch) * k;
    this.seenMs += dt;
    const c = (v) => Math.max(-1.5, Math.min(1.5, v));
    return { yaw: c(cue.yaw - this.rest.yaw), pitch: c(cue.pitch - this.rest.pitch), calibrating: this.seenMs < this.cfg.calibrateMs };
  }
}

/** Eye-look blendshapes -> gaze direction in [-1, 1] (screen space of the mirrored view). */
export function gazeFromBlendshapes(bs = {}) {
  // user looks to their right: eyeLookOutRight + eyeLookInLeft rise; a mirrored view shows
  // that as looking toward screen right
  const right = (num(bs.eyeLookInLeft) + num(bs.eyeLookOutRight)) / 2;
  const left = (num(bs.eyeLookOutLeft) + num(bs.eyeLookInRight)) / 2;
  const down = (num(bs.eyeLookDownLeft) + num(bs.eyeLookDownRight)) / 2;
  const up = (num(bs.eyeLookUpLeft) + num(bs.eyeLookUpRight)) / 2;
  return { x: Math.max(-1, Math.min(1, right - left)), y: Math.max(-1, Math.min(1, down - up)) };
}

/**
 * Expression scores from blendshapes (each in [0, 1]):
 *   HAPPY     = avg(mouthSmile)                                   (a smile)
 *   SAD       = clamp(0.7*avg(mouthFrown) + 0.5*browInnerUp + 0.2*avg(mouthPress)) * (1 - HAPPY)
 *   SURPRISED = clamp(0.5*jawOpen + 0.3*avg(browOuterUp) + 0.4*avg(eyeWide)) * (1 - HAPPY)
 *               (all three together; an open mouth alone, e.g. speaking, stays below the threshold)
 */
export function expressionScores(bs = {}) {
  const avg = (a, b) => (num(bs[a]) + num(bs[b])) / 2;
  const happy = clamp01(avg('mouthSmileLeft', 'mouthSmileRight'));
  const sad = clamp01(0.7 * avg('mouthFrownLeft', 'mouthFrownRight') + 0.5 * num(bs.browInnerUp) + 0.2 * avg('mouthPressLeft', 'mouthPressRight')) * (1 - happy);
  const surprised = clamp01(0.5 * num(bs.jawOpen) + 0.3 * avg('browOuterUpLeft', 'browOuterUpRight') + 0.4 * avg('eyeWideLeft', 'eyeWideRight')) * (1 - happy);
  return { HAPPY: happy, SAD: sad, SURPRISED: surprised };
}

export const EXPRESSION_DEFAULTS = Object.freeze({
  enabled: true, smoothingMs: 250, enter: 0.5, exit: 0.3, holdMs: 400,
  // per-user calibration: score = gain * (raw - the user's resting level). Live sessions (VALIDATION.md
  // §13, §15) showed raw smile scores of only ~0.2 for a visible smile; surprise keeps gain 1 so that
  // an open mouth while speaking stays neutral.
  gain: Object.freeze({ HAPPY: 2.5, SAD: 2, SURPRISED: 1 }),
  baseline: Object.freeze({ enabled: true, upMs: 30000, downMs: 1500 }),
});
const ZERO = Object.freeze({ HAPPY: 0, SAD: 0, SURPRISED: 0 });

/**
 * Temporal expression state: exponential smoothing (time constant smoothingMs) of the
 * scores; an expression is reported after its smoothed score stayed >= enter for holdMs
 * and released when it falls below exit (hysteresis); NEUTRAL otherwise. When no face is
 * observed the scores decay toward 0, so face loss always falls back to NEUTRAL.
 */
export class FaceExpressionEngine {
  constructor(cfg = {}) {
    this.cfg = { ...EXPRESSION_DEFAULTS, ...cfg, gain: { ...EXPRESSION_DEFAULTS.gain, ...cfg.gain }, baseline: { ...EXPRESSION_DEFAULTS.baseline, ...cfg.baseline } };
    this.reset();
  }

  reset() {
    this.smoothed = { HAPPY: 0, SAD: 0, SURPRISED: 0 };
    // resting level of the current user's face: follows the raw score down fast, up slowly
    this.baseline = { HAPPY: 0, SAD: 0, SURPRISED: 0 };
    this.label = 'NEUTRAL';
    this.candidate = null;
    this.candidateSince = 0;
    this._lastT = null;
  }

  /**
   * @param {number} t ms
   * @param {object|null} bs blendshape scores of the primary face, or null (no face)
   * @returns {{ label: 'HAPPY'|'SAD'|'SURPRISED'|'NEUTRAL', confidence: number, scores: object }}
   */
  update(t, bs) {
    const c = this.cfg;
    // first call: one nominal 33 ms step; afterwards the real elapsed time (capped at 1 s)
    const first = this._lastT === null;
    const dt = first ? 33 : t > this._lastT ? Math.min(1000, t - this._lastT) : 0;
    if (first || t > this._lastT) this._lastT = t;
    const raw = c.enabled && bs ? expressionScores(bs) : null;
    const target = { ...ZERO };
    if (raw) {
      for (const key of Object.keys(target)) {
        if (c.baseline.enabled && dt > 0) {
          const tau = raw[key] < this.baseline[key] ? c.baseline.downMs : c.baseline.upMs;
          this.baseline[key] += (raw[key] - this.baseline[key]) * (1 - Math.exp(-dt / tau));
        }
        const rest = c.baseline.enabled ? this.baseline[key] : 0;
        target[key] = Math.max(0, Math.min(1, (raw[key] - rest) * (c.gain[key] ?? 1)));
      }
    }
    const k = dt > 0 ? 1 - Math.exp(-dt / c.smoothingMs) : 0;
    for (const key of Object.keys(this.smoothed)) this.smoothed[key] += (target[key] - this.smoothed[key]) * k;
    const [best, score] = Object.entries(this.smoothed).sort((a, b) => b[1] - a[1])[0];
    if (this.label !== 'NEUTRAL' && this.smoothed[this.label] < c.exit) this.label = 'NEUTRAL';
    if (this.label === 'NEUTRAL' || (best !== this.label && score >= c.enter && score > this.smoothed[this.label] + 0.1)) {
      if (score >= c.enter) {
        if (this.candidate !== best) {
          this.candidate = best;
          this.candidateSince = t;
        }
        if (t - this.candidateSince >= c.holdMs) {
          this.label = best;
          this.candidate = null;
        }
      } else this.candidate = null;
    }
    const confidence = this.label === 'NEUTRAL' ? 1 - Math.min(1, score / c.enter) * 0.5 : this.smoothed[this.label];
    return { label: this.label, confidence, scores: { ...this.smoothed }, baseline: { ...this.baseline } };
  }
}
