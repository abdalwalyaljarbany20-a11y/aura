/**
 * Continuous index-finger rotation ("tornado"): a signed angular velocity of the
 * index fingertip while the hand holds the pointing shape.
 *
 * Coordinates are display space (aspect-corrected, mirrored like the selfie view,
 * y down), so a positive rate is clockwise as the user sees it on screen. Radii are
 * in palm lengths (palmSizeImage), so the rule does not depend on camera distance.
 *
 * Each frame, the last `windowMs` of the trajectory (at least `minWindowMs`) is
 * checked for a rotational signature around its own centroid:
 *   - mean radius >= minRadius                       (rejects a still finger, jitter)
 *   - radius std / mean <= maxRadiusCv               (rejects lines and zig-zags)
 *   - no single step larger than maxStepDeg           (rejects tracking jumps)
 *   - >= minConsistency of the steps turn the same way (rejects noise)
 *   - |swept angle| >= minSweepDeg                    (rejects brief accidental arcs)
 * The rate is the swept angle of the newest `rateWindowMs` divided by its duration,
 * so it follows speed changes within a fraction of a turn; a full circle is never
 * required per update.
 *
 * Hysteresis: rotation becomes active after the signature persisted `enterMs`
 * (emits start), and ends when it has been absent for `releaseMs`, the pointing
 * shape was lost for `poseGraceMs`, or the hand was lost (immediately) (emits end).
 * While active without a fresh signature (inside releaseMs) the last rate is held.
 * Mirrored exactly by aura_cv.gestures.SpinDetector (same arithmetic order).
 */
const MIN_POINTS = 6;
const DEG = Math.PI / 180;

export class SpinDetector {
  /** @param {object} cfg gestures.spin configuration */
  constructor(cfg) {
    this.cfg = { ...cfg };
    this.history = []; // [t, x, y, palm]
    this.active = false;
    this.omega = 0; // rad/s, + = clockwise on screen
    this.signatureSince = null;
    this.lastSignature = -Infinity;
    this.inactiveSince = null;
  }

  reset() {
    this.history = [];
    this.active = false;
    this.omega = 0;
    this.signatureSince = null;
    this.lastSignature = -Infinity;
    this.inactiveSince = null;
  }

  /**
   * @param {number} t ms
   * @param {{x:number,y:number}|null} tip index fingertip, display space (null = no hand)
   * @param {number} palm palm length, same units
   * @param {boolean} pose the hand currently has the pointing shape
   * @param {boolean} [blocked] another motion gesture owns the hand: do not start
   * @returns {{ active: boolean, omega: number, event: 'start'|'end'|null }}
   */
  update(t, tip, palm, pose, blocked = false) {
    const c = this.cfg;
    if (!c.enabled || !tip || !(palm > 0)) return this._end(true);
    if (!pose) {
      if (this.inactiveSince === null) this.inactiveSince = t;
      if (t - this.inactiveSince > c.poseGraceMs) return this._end(true);
      return { active: this.active, omega: this.active ? this.omega : 0, event: null };
    }
    this.inactiveSince = null;
    const h = this.history;
    if (h.length && t - h[h.length - 1][0] > c.maxGapMs) h.length = 0;
    h.push([t, tip.x, tip.y, palm]);
    while (h.length && h[0][0] < t - c.windowMs) h.shift();

    const m = this._measure(h);
    let event = null;
    if (m) {
      this.lastSignature = t;
      if (this.signatureSince === null) this.signatureSince = t;
      if (this.active) this.omega = m.rate;
      else if (!blocked && t - this.signatureSince >= c.enterMs) {
        this.active = true;
        this.omega = m.rate;
        event = 'start';
      }
    } else {
      this.signatureSince = null;
      if (this.active && t - this.lastSignature > c.releaseMs) {
        this.active = false;
        this.omega = 0;
        event = 'end';
      }
    }
    return { active: this.active, omega: this.active ? this.omega : 0, event };
  }

  _end(clear) {
    const was = this.active;
    if (clear) this.reset();
    return { active: false, omega: 0, event: was ? 'end' : null };
  }

  /** Signature metrics for the whole history; null unless every criterion holds. */
  _measure(h) {
    const c = this.cfg;
    const n = h.length;
    if (n < MIN_POINTS) return null;
    const last = h[n - 1][0];
    if (last - h[0][0] < c.minWindowMs) return null;
    let cx = 0;
    let cy = 0;
    let palm = 0;
    for (let k = 0; k < n; k++) {
      cx += h[k][1];
      cy += h[k][2];
      palm += h[k][3];
    }
    cx /= n;
    cy /= n;
    palm /= n;
    let rSum = 0;
    let r2Sum = 0;
    for (let k = 0; k < n; k++) {
      const dx = h[k][1] - cx;
      const dy = h[k][2] - cy;
      const r = Math.sqrt(dx * dx + dy * dy) / palm;
      rSum += r;
      r2Sum += r * r;
    }
    const radius = rSum / n;
    if (radius < c.minRadius) return null;
    const variance = Math.max(0, r2Sum / n - radius * radius);
    if (Math.sqrt(variance) / radius > c.maxRadiusCv) return null;

    let total = 0;
    let pos = 0;
    let neg = 0;
    let recent = 0;
    let recentFrom = null;
    let prev = Math.atan2(h[0][2] - cy, h[0][1] - cx);
    for (let k = 1; k < n; k++) {
      const a = Math.atan2(h[k][2] - cy, h[k][1] - cx);
      let d = a - prev;
      if (d > Math.PI) d -= 2 * Math.PI;
      else if (d <= -Math.PI) d += 2 * Math.PI;
      if (Math.abs(d) > c.maxStepDeg * DEG) return null;
      total += d;
      if (d > 0) pos++;
      else if (d < 0) neg++;
      if (h[k - 1][0] >= last - c.rateWindowMs) {
        if (recentFrom === null) recentFrom = h[k - 1][0];
        recent += d;
      }
      prev = a;
    }
    if (Math.abs(total) < c.minSweepDeg * DEG) return null;
    const steps = pos + neg;
    const consistency = steps ? (total > 0 ? pos : neg) / steps : 0;
    if (consistency < c.minConsistency) return null;
    const rate = recentFrom !== null && last > recentFrom ? recent / ((last - recentFrom) / 1000) : total / ((last - h[0][0]) / 1000);
    return { total, radius, consistency, rate };
  }
}
