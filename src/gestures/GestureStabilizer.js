/**
 * Temporal stabiliser with hysteresis for static gestures.
 *
 *  - Enter: a label must be observed continuously for its `holdMs` before it
 *    becomes the stable gesture (any interruption restarts the timer).
 *  - Exit: the stable gesture is released only after it has been absent for
 *    `exitMs`, so single-frame misclassifications do not end it.
 *  - Exactly one `start` event is emitted per continuous gesture and one
 *    `end` event when it is released. Commands are bound to `start`, so a
 *    held gesture can never trigger its command repeatedly.
 *
 * Neutral labels (NONE, UNKNOWN, MOVING) never become stable.
 */
export const NEUTRAL = Object.freeze(new Set(['NONE', 'UNKNOWN', 'MOVING']));

export class GestureStabilizer {
  /**
   * @param {{ exitMs: number, holdMs: (gesture: string) => number }} opts
   */
  constructor(opts) {
    this.exitMs = opts.exitMs;
    this.holdMs = opts.holdMs;
    this.reset();
  }

  reset() {
    this.stable = 'NONE';
    this.stableSince = 0;
    this.lastSeenStable = 0;
    this.candidate = null;
    this.candidateSince = 0;
    this.stableConfidence = 0;
  }

  /**
   * @param {string} label raw per-frame label
   * @param {number} confidence raw per-frame confidence
   * @param {number} t timestamp (ms)
   * @returns {{ type: 'start'|'end', gesture: string, t: number, confidence?: number, durationMs?: number }[]}
   */
  update(label, confidence, t) {
    const events = [];

    if (label === this.stable && !NEUTRAL.has(label)) {
      this.lastSeenStable = t;
      this.stableConfidence = confidence;
      this.candidate = null;
      return events;
    }

    if (!NEUTRAL.has(this.stable) && t - this.lastSeenStable >= this.exitMs) {
      events.push({ type: 'end', gesture: this.stable, t, durationMs: this.lastSeenStable - this.stableSince });
      this.stable = 'NONE';
      this.stableConfidence = 0;
    }

    if (NEUTRAL.has(label)) {
      this.candidate = null;
      return events;
    }

    if (this.candidate !== label) {
      this.candidate = label;
      this.candidateSince = t;
    }
    if (NEUTRAL.has(this.stable) && t - this.candidateSince >= this.holdMs(label)) {
      this.stable = label;
      this.stableSince = t;
      this.lastSeenStable = t;
      this.stableConfidence = confidence;
      this.candidate = null;
      events.push({ type: 'start', gesture: label, t, confidence });
    }
    return events;
  }

  /**
   * Immediately release the stable gesture (e.g. tracking lost).
   * @param {number} t
   */
  release(t) {
    const events = [];
    if (!NEUTRAL.has(this.stable)) {
      events.push({ type: 'end', gesture: this.stable, t, durationMs: this.lastSeenStable - this.stableSince });
    }
    this.reset();
    return events;
  }
}
