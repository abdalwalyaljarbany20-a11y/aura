/**
 * Finger walking: the index and middle fingers act as two legs. Walking is the
 * *alternating* motion of the two fingers, and its cadence (steps per second)
 * controls the robot's speed. A static V sign, a single moving finger and a
 * shaking hand do not walk.
 *
 * Input per frame (features.legs, from joint geometry): each finger's length
 * projected on the palm axis (wrist -> middle MCP), in palm lengths. A finger that
 * bends or swings shortens its projection, so the two "leg" signals move in
 * anti-phase while stepping. Both are translation- and scale-invariant, and the
 * palm axis makes them invariant to in-plane hand rotation.
 *
 * Over the last `windowMs`:
 *   - each finger must move at least `minAmplitude` peak-to-peak  (static V, one finger: rejected)
 *     and reach `minReach` (nearly straight) at some point      (a curled finger is not a leg)
 *   - the two signals must be anti-correlated (<= maxCorrelation) (whole-hand shaking: rejected)
 *   - a step is a crossing of the difference (index - middle) through a band of
 *     +-hysteresis around its window mean (jitter inside the band never counts), and
 *     only while the difference itself moved >= hysteresis within the last 200 ms
 *   - cadence = steps per second over the last few steps, lowered to the
 *     time since the last step when the user slows down
 *   - steps faster than maxStepRate are treated as shaking
 * Walking starts after `minSteps` valid alternating steps within the window (false-start filter) and
 * ends when no step happened for `stopMs`, the legs pose was lost for
 * `poseGraceMs`, the signal stops qualifying, or the hand was lost (immediately).
 * Mirrored exactly by aura_cv.gestures.WalkCadenceDetector.
 */
export class WalkCadenceDetector {
  /** @param {object} cfg gestures.walk configuration */
  constructor(cfg) {
    this.cfg = { ...cfg };
    this.reset();
  }

  reset() {
    this.history = []; // [t, index, middle]
    this.steps = []; // step times (ms)
    this.side = 0;
    this.active = false;
    this.cadence = 0;
    this.inactiveSince = null;
  }

  /**
   * @param {number} t ms
   * @param {{index:number, middle:number}|null} legs leg signals (null = no hand)
   * @param {boolean} pose the hand has the legs pose (ring, pinky, thumb folded)
   * @param {boolean} [blocked] another motion gesture owns the hand: do not start
   * @returns {{ active: boolean, cadence: number, event: 'start'|'end'|null }}
   */
  update(t, legs, pose, blocked = false) {
    const c = this.cfg;
    if (!c.enabled || !legs) return this._end();
    if (!pose) {
      if (this.inactiveSince === null) this.inactiveSince = t;
      if (t - this.inactiveSince > c.poseGraceMs) return this._end();
      return { active: this.active, cadence: this.active ? this.cadence : 0, event: null };
    }
    this.inactiveSince = null;
    const h = this.history;
    h.push([t, legs.index, legs.middle]);
    while (h.length && h[0][0] < t - c.windowMs) h.shift();
    if (this.steps.length > 6) this.steps.splice(0, this.steps.length - 6); // the last steps, for the cadence

    const n = h.length;
    let mi = 0;
    let mm = 0;
    let lo = [Infinity, Infinity];
    let hi = [-Infinity, -Infinity];
    for (let k = 0; k < n; k++) {
      mi += h[k][1];
      mm += h[k][2];
      lo = [Math.min(lo[0], h[k][1]), Math.min(lo[1], h[k][2])];
      hi = [Math.max(hi[0], h[k][1]), Math.max(hi[1], h[k][2])];
    }
    mi /= n;
    mm /= n;
    let sii = 0;
    let smm = 0;
    let sim = 0;
    for (let k = 0; k < n; k++) {
      const a = h[k][1] - mi;
      const b = h[k][2] - mm;
      sii += a * a;
      smm += b * b;
      sim += a * b;
    }
    const corr = sii > 0 && smm > 0 ? sim / Math.sqrt(sii * smm) : 0;
    // each leg must also be (nearly) straight at some point: a pointing hand's curled middle
    // finger never is, so circling while pointing is not walking
    const moving = hi[0] - lo[0] >= c.minAmplitude && hi[1] - lo[1] >= c.minAmplitude && hi[0] >= c.minReach && hi[1] >= c.minReach;
    const alternating = moving && corr <= c.maxCorrelation;

    // step counting on the difference signal, with hysteresis around the window mean
    const dev = legs.index - legs.middle - (mi - mm);
    const side = dev > c.hysteresis ? 1 : dev < -c.hysteresis ? -1 : 0;
    // a crossing counts as a step only if the fingers really moved recently (not the
    // window mean drifting past a still hand)
    let dLo = Infinity;
    let dHi = -Infinity;
    for (let k = n - 1; k >= 0 && h[k][0] >= t - 200; k--) {
      const d = h[k][1] - h[k][2];
      dLo = Math.min(dLo, d);
      dHi = Math.max(dHi, d);
    }
    const stepping = dHi - dLo >= c.hysteresis;
    if (side !== 0 && side !== this.side) {
      if (this.side !== 0 && alternating && stepping) this.steps.push(t);
      this.side = side;
    }

    const s = this.steps;
    let cadence = 0;
    if (s.length >= 2) {
      const k0 = Math.max(0, s.length - 5);
      const avg = ((s.length - 1 - k0) * 1000) / (s[s.length - 1] - s[k0]);
      const sinceLast = t - s[s.length - 1];
      cadence = sinceLast > 1000 / avg ? Math.min(avg, 1000 / sinceLast) : avg;
    }
    const recent = s.length > 0 && t - s[s.length - 1] <= c.stopMs;
    // minSteps (within the window) is a start filter; continuing needs a recent step only
    const inWindow = s.filter((x) => x >= t - c.windowMs).length;
    const valid = alternating && recent && (this.active || inWindow >= c.minSteps) && cadence <= c.maxStepRate;

    let event = null;
    if (valid) {
      this.cadence = cadence;
      if (!this.active && !blocked) {
        this.active = true;
        event = 'start';
      }
    } else if (this.active) {
      this.active = false;
      this.cadence = 0;
      event = 'end';
    }
    return { active: this.active, cadence: this.active ? this.cadence : 0, event };
  }

  _end() {
    const was = this.active;
    this.reset();
    return { active: false, cadence: 0, event: was ? 'end' : null };
  }
}
