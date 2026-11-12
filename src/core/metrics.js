/**
 * Measurement utilities. Every number shown in the HUD comes from one of
 * these, fed with real timestamps (performance.now() in the browser).
 */

/** Counts events and reports their rate (events/second) over a sliding window. */
export class RateMeter {
  /** @param {number} windowMs */
  constructor(windowMs = 1000) {
    this.windowMs = windowMs;
    /** @type {number[]} */
    this._times = [];
    this.total = 0;
    this._start = null;
  }

  /** @param {number} t timestamp in ms */
  tick(t, count = 1) {
    if (this._start === null) this._start = t;
    for (let i = 0; i < count; i++) this._times.push(t);
    this.total += count;
    this._trim(t);
  }

  /**
   * Events per second: events inside the window ending at `now`, divided by
   * the window length (or by the time since the first event while the meter
   * is younger than one window). Decays to 0 when events stop.
   * Returns null until the meter has observed at least 100 ms.
   * @param {number} now
   */
  rate(now) {
    if (this._start === null) return null;
    this._trim(now);
    const span = Math.min(this.windowMs, now - this._start);
    if (span < 100) return null;
    return this._times.length / (span / 1000);
  }

  _trim(now) {
    const cutoff = now - this.windowMs;
    let i = 0;
    while (i < this._times.length && this._times[i] < cutoff) i++;
    if (i) this._times.splice(0, i);
  }

  reset() {
    this._times = [];
    this.total = 0;
    this._start = null;
  }
}

/** Rolling statistics over the last N samples (e.g. latency in ms). */
export class RollingStats {
  /** @param {number} size */
  constructor(size = 120) {
    this.size = size;
    /** @type {number[]} */
    this._samples = [];
    this.count = 0;
  }

  /** @param {number} value */
  push(value) {
    if (!Number.isFinite(value)) return;
    this._samples.push(value);
    if (this._samples.length > this.size) this._samples.shift();
    this.count++;
  }

  get last() {
    return this._samples.length ? this._samples[this._samples.length - 1] : null;
  }

  get mean() {
    const n = this._samples.length;
    if (!n) return null;
    let s = 0;
    for (const v of this._samples) s += v;
    return s / n;
  }

  /** @param {number} p percentile in [0, 100] */
  percentile(p) {
    const n = this._samples.length;
    if (!n) return null;
    const sorted = [...this._samples].sort((a, b) => a - b);
    const idx = Math.min(n - 1, Math.max(0, Math.ceil((p / 100) * n) - 1));
    return sorted[idx];
  }

  summary() {
    return { last: this.last, mean: this.mean, p95: this.percentile(95), samples: this._samples.length };
  }

  reset() {
    this._samples = [];
    this.count = 0;
  }
}
