/**
 * Minimal synchronous publish/subscribe bus.
 *
 * Used to decouple producers of application events (camera, tracker,
 * controller) from consumers (HUD event log, session recorder, future
 * integrations). Listener exceptions are isolated so one faulty consumer
 * cannot break the frame pipeline.
 */
export class EventBus {
  constructor() {
    /** @type {Map<string, Set<Function>>} */
    this._listeners = new Map();
    this.listenerErrors = 0;
  }

  /**
   * @param {string} type event type, or '*' for every event
   * @param {(event: object) => void} fn
   * @returns {() => void} unsubscribe function
   */
  on(type, fn) {
    if (!this._listeners.has(type)) this._listeners.set(type, new Set());
    this._listeners.get(type).add(fn);
    return () => this._listeners.get(type)?.delete(fn);
  }

  /**
   * @param {string} type
   * @param {object} [payload]
   */
  emit(type, payload = {}) {
    const event = { type, ...payload };
    for (const key of [type, '*']) {
      const set = this._listeners.get(key);
      if (!set) continue;
      for (const fn of [...set]) {
        try {
          fn(event);
        } catch (err) {
          this.listenerErrors++;
          console.error(`[EventBus] listener for "${type}" failed`, err);
        }
      }
    }
    return event;
  }

  clear() {
    this._listeners.clear();
  }
}
