/**
 * AttentionController: where the robot looks (face-oriented gaze).
 *
 * Inputs are in normalised *screen* coordinates (0..1 of the stage, y down), so
 * they line up with where the robot is drawn:
 *   observeFaces(t, faces)  each time a face detection ran (faces may be [])
 *   update(t, ctx)          once per render frame -> smoothed, bounded head angles
 *
 * Policy (deterministic):
 *   HAND     a hold listed in `handAttentionHolds` (default AIM: the robot aims at
 *            the fingertip, which the model represents convincingly) is active and
 *            the hand point is valid
 *   USER     else, the primary face was seen within `faceHoldMs`
 *   NEUTRAL  else (no face, face lost for longer, or invalid input): straight ahead
 * Hand tracking keeps running in every mode; only the gaze target changes.
 *
 * Primary face: the largest face when none is tracked; afterwards the detection
 * nearest to the primary within `maxJump` keeps being the primary. Another face is
 * taken only after the primary was missing for `switchAfterMs`, so two people in
 * view never make the robot flick between them.
 *
 * Angles: yaw = clamp((face.x - head.x) * yawGain, +-maxYaw), pitch = clamp((face.y -
 * head.y) * pitchGain, -maxPitchUp .. maxPitchDown) (+ = looking down), relative to
 * the robot's default facing. They follow the target with an exponential time
 * constant `smoothingMs` and an angular speed cap `maxSpeedDegS`: frame-rate
 * independent, and a jump of the target never snaps the head. Non-finite or
 * out-of-range input is ignored, so the output is always finite and bounded.
 */
const DEG = Math.PI / 180;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const isPoint = (p) => p && Number.isFinite(p.x) && Number.isFinite(p.y) && p.x >= -0.25 && p.x <= 1.25 && p.y >= -0.25 && p.y <= 1.25;

export const ATTENTION_DEFAULTS = Object.freeze({
  handAttentionHolds: ['AIM'],
  minFaceScore: 0.6,
  faceHoldMs: 700,
  switchAfterMs: 500,
  maxJump: 0.25,
  yawGain: 1.8,
  pitchGain: 1.4,
  maxYawDeg: 46,
  maxPitchUpDeg: 22,
  maxPitchDownDeg: 26,
  smoothingMs: 140,
  maxSpeedDegS: 300,
});

export class AttentionController {
  /** @param {object} [cfg] interaction.attention */
  constructor(cfg = {}) {
    this.cfg = { ...ATTENTION_DEFAULTS, ...cfg };
    this.primary = null; // { x, y, w, h, score, lastSeen }
    this.faceCount = 0;
    this.lastObserved = null;
    this.yaw = 0;
    this.pitch = 0;
    this.mode = 'NEUTRAL';
    this._lastT = null;
  }

  /**
   * @param {number} t ms
   * @param {{ x: number, y: number, w: number, h: number, score: number }[]} faces screen-normalised centres
   */
  observeFaces(t, faces) {
    const c = this.cfg;
    const valid = (Array.isArray(faces) ? faces : []).filter(
      (f) => isPoint(f) && f.x >= 0 && f.x <= 1 && f.y >= 0 && f.y <= 1 && Number.isFinite(f.w) && Number.isFinite(f.h) && f.w > 0 && f.h > 0 && f.w <= 1.5 && f.h <= 1.5 && Number.isFinite(f.score) && f.score >= c.minFaceScore,
    );
    this.faceCount = valid.length;
    this.lastObserved = t;
    if (!valid.length) return;
    const p = this.primary;
    if (p) {
      let best = null;
      let bestD = Infinity;
      for (const f of valid) {
        const d = Math.hypot(f.x - p.x, f.y - p.y);
        if (d < bestD) {
          bestD = d;
          best = f;
        }
      }
      if (best && bestD <= c.maxJump) {
        this.primary = { ...best, lastSeen: t };
        return;
      }
      if (t - p.lastSeen < c.switchAfterMs) return; // keep the primary for a while: no flicking
    }
    const largest = valid.reduce((a, b) => (b.w * b.h > a.w * a.h ? b : a));
    this.primary = { ...largest, lastSeen: t };
  }

  /** @returns {boolean} the primary face was seen within faceHoldMs */
  faceVisible(t) {
    return !!this.primary && t - this.primary.lastSeen <= this.cfg.faceHoldMs;
  }

  /**
   * @param {number} t ms
   * @param {{ head: {x:number,y:number}|null, hold?: string|null, hand?: {x:number,y:number}|null }} ctx
   * @returns {{ mode: 'USER'|'HAND'|'NEUTRAL', yaw: number, pitch: number, target: {x:number,y:number}|null, faceVisible: boolean, faceCount: number }}
   */
  update(t, ctx = {}) {
    const c = this.cfg;
    const head = isPoint(ctx.head) ? ctx.head : { x: 0.5, y: 0.4 };
    let target = null;
    let mode = 'NEUTRAL';
    if (ctx.hold && c.handAttentionHolds.includes(ctx.hold) && isPoint(ctx.hand)) {
      mode = 'HAND';
      target = ctx.hand;
    } else if (this.faceVisible(t)) {
      mode = 'USER';
      target = this.primary;
    }
    this.mode = mode;
    const maxYaw = c.maxYawDeg * DEG;
    const ty = target ? clamp((target.x - head.x) * c.yawGain, -maxYaw, maxYaw) : 0;
    const tp = target ? clamp((target.y - head.y) * c.pitchGain, -c.maxPitchUpDeg * DEG, c.maxPitchDownDeg * DEG) : 0;

    const dt = this._lastT === null || !(t > this._lastT) ? 0 : Math.min(0.25, (t - this._lastT) / 1000);
    if (this._lastT === null || t > this._lastT) this._lastT = t;
    if (dt > 0) {
      const k = 1 - Math.exp(-(dt * 1000) / c.smoothingMs);
      const maxStep = c.maxSpeedDegS * DEG * dt;
      this.yaw += clamp((ty - this.yaw) * k, -maxStep, maxStep);
      this.pitch += clamp((tp - this.pitch) * k, -maxStep, maxStep);
    }
    return {
      mode,
      yaw: this.yaw,
      pitch: this.pitch,
      target: target ? { x: target.x, y: target.y } : null,
      faceVisible: this.faceVisible(t),
      faceCount: this.faceCount,
    };
  }

  reset() {
    this.primary = null;
    this.faceCount = 0;
  }
}
