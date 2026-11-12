/**
 * Small explicit finite-state machine with a declared transition table.
 * Invalid transitions are rejected (returned as false), never applied.
 */
export class StateMachine {
  /**
   * @param {Record<string, string[]>} transitions state -> allowed next states
   * @param {string} initial
   * @param {(from: string, to: string, reason: string) => void} [onChange]
   */
  constructor(transitions, initial, onChange) {
    if (!transitions[initial]) throw new Error(`Unknown initial state "${initial}"`);
    for (const [from, targets] of Object.entries(transitions)) {
      for (const to of targets) {
        if (!transitions[to]) throw new Error(`Transition ${from} -> ${to} targets an undeclared state`);
      }
    }
    this.transitions = transitions;
    this.state = initial;
    this.enteredAt = 0;
    this.onChange = onChange;
  }

  /** @param {string} to */
  can(to) {
    return to === this.state || (this.transitions[this.state]?.includes(to) ?? false);
  }

  /**
   * @param {string} to
   * @param {number} t timestamp (ms)
   * @param {string} [reason]
   * @returns {boolean} whether the machine is now in `to`
   */
  transition(to, t, reason = '') {
    if (to === this.state) return true;
    if (!this.can(to)) return false;
    const from = this.state;
    this.state = to;
    this.enteredAt = t;
    this.onChange?.(from, to, reason);
    return true;
  }
}
