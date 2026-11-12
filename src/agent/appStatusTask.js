/**
 * `app.status`: read-only snapshot of the live application state, so the agent
 * answers questions about the app from the app itself, not from conversation
 * memory. Takes no arguments; changes nothing.
 *
 * @param {import('./TaskBroker.js').TaskBroker} broker
 * @param {() => object} getStatus returns a plain, JSON-safe object (built by the coordinator)
 */
export function registerAppStatusTask(broker, getStatus) {
  broker.register('app.status', {
    description: 'Read the current application state (read-only)',
    timeoutMs: 1000,
    validateArgs(args) {
      return Object.keys(args).length ? 'app.status takes no arguments' : null;
    },
    execute() {
      return structuredClone(getStatus());
    },
  });
}
