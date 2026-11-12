/**
 * `app.diagnostics`: read-only troubleshooting snapshot (see
 * ApplicationCoordinator.agentDiagnostics for the data sources). Takes no
 * arguments and changes nothing; unknown values are null, never guessed.
 *
 * @param {import('./TaskBroker.js').TaskBroker} broker
 * @param {() => Promise<object>|object} getDiagnostics
 */
export function registerDiagnosticsTask(broker, getDiagnostics) {
  broker.register('app.diagnostics', {
    description: 'Read application diagnostics (read-only)',
    timeoutMs: 4000, // includes one local /api/agent/status request
    validateArgs(args) {
      return Object.keys(args).length ? 'app.diagnostics takes no arguments' : null;
    },
    async execute() {
      return structuredClone(await getDiagnostics());
    },
  });
}
