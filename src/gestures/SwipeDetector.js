/**
 * Dynamic gesture detector for horizontal swipes, based on a timestamped
 * history of the palm centre in display space (mirrored if the view is).
 *
 * A swipe fires when, within the last `windowMs`, the palm travelled at
 * least `minDistance` palm lengths horizontally, at an average speed of at
 * least `minSpeed` palm lengths/s, with |dy| <= maxVerticalRatio * |dx|.
 *
 * Re-trigger protection:
 *  - history is cleared after a swipe, and no swipe can fire for `cooldownMs`;
 *  - a swipe in the opposite direction is suppressed for `oppositeSuppressMs`
 *    (the natural return stroke after a swipe);
 *  - a tracking gap longer than `maxGapMs` clears the history.
 *
 * Distances are divided by the hand's apparent palm size, so the same
 * physical motion is recognised at different distances from the camera.
 */
export class SwipeDetector {
  /** @param {object} cfg gestures.swipe configuration */
  constructor(cfg) {
    this.cfg = { ...cfg };
    /** @type {{t:number,x:number,y:number,palm:number}[]} */
    this.history = [];
    this.blockedUntil = -Infinity;
    this.last = null; // { direction, t }
  }

  reset() {
    this.history = [];
  }

  /**
   * @param {number} t timestamp (ms)
   * @param {{x:number,y:number}|null} palmCenter display-space position (units of image height)
   * @param {number} palmSize apparent palm size in the same units
   * @param {{ suppressed?: boolean }} [opts] suppressed=true records motion but never fires
   * @returns {null | { gesture: 'SWIPE_LEFT'|'SWIPE_RIGHT', distance: number, speed: number, durationMs: number }}
   */
  update(t, palmCenter, palmSize, opts = {}) {
    const c = this.cfg;
    if (!c.enabled || !palmCenter || !(palmSize > 0)) {
      this.reset();
      return null;
    }
    const prev = this.history[this.history.length - 1];
    if (prev && t - prev.t > c.maxGapMs) this.history = [];
    this.history.push({ t, x: palmCenter.x, y: palmCenter.y, palm: palmSize });
    while (this.history.length && this.history[0].t < t - c.windowMs) this.history.shift();

    if (opts.suppressed || t < this.blockedUntil || this.history.length < 3) return null;

    const now = this.history[this.history.length - 1];
    const palm = this.history.reduce((s, h) => s + h.palm, 0) / this.history.length;
    // Try every start sample inside the window; keep the strongest stroke.
    let best = null;
    for (let i = 0; i < this.history.length - 2; i++) {
      const s = this.history[i];
      const dt = now.t - s.t;
      if (dt <= 0) continue;
      const dx = (now.x - s.x) / palm;
      const dy = (now.y - s.y) / palm;
      const distance = Math.abs(dx);
      const speed = distance / (dt / 1000);
      if (distance < c.minDistance || speed < c.minSpeed) continue;
      if (Math.abs(dy) > c.maxVerticalRatio * distance) continue;
      if (!best || distance > best.distance) best = { dx, distance, speed, durationMs: dt };
    }
    if (!best) return null;

    const gesture = best.dx < 0 ? 'SWIPE_LEFT' : 'SWIPE_RIGHT';
    if (this.last && this.last.gesture !== gesture && t - this.last.t < c.oppositeSuppressMs) {
      this.history = [];
      return null;
    }
    this.last = { gesture, t };
    this.blockedUntil = t + c.cooldownMs;
    this.history = [];
    return { gesture, distance: best.distance, speed: best.speed, durationMs: best.durationMs };
  }

  /** Current palm speed (palm lengths / s) over the last ~150 ms, or 0. */
  speed() {
    const h = this.history;
    if (h.length < 2) return 0;
    const now = h[h.length - 1];
    let i = h.length - 2;
    while (i > 0 && now.t - h[i].t < 150) i--;
    const s = h[i];
    const dt = (now.t - s.t) / 1000;
    if (dt <= 0) return 0;
    return Math.hypot(now.x - s.x, now.y - s.y) / now.palm / dt;
  }
}
