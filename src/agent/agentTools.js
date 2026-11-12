import { COMMANDS } from '../interaction/commands.js';

/**
 * Agent tool catalog: the ONLY actions an AI model may propose.
 *
 * Shared by the server (validates model output before it leaves the server and
 * builds the provider's tool definitions) and the browser (re-validates before
 * execution and applies the confirmation policy). Every tool maps to a task
 * type registered in TaskBroker; nothing here executes anything by itself.
 *
 * Policy:
 *   'auto'    read-only, or a low-risk reversible robot animation
 *   'confirm' changes how the user controls the app (gesture lock) - requires an
 *             explicit click in the agent panel on the exact proposed action
 */

/** Robot commands the agent may request (holds need a live hand; gesture-only commands are not offered). */
export const AGENT_ROBOT_COMMANDS = Object.freeze(
  Object.entries(COMMANDS).filter(([, d]) => d.kind !== 'hold' && !d.gestureOnly).map(([name]) => name),
);

/** Commands that need an explicit user confirmation before they run. */
export const CONFIRM_COMMANDS = Object.freeze(['TOGGLE_LOCK']);

const LABELS = {
  WAVE: 'Make the robot wave',
  SPIN_LEFT: 'Spin the robot a full turn to the left',
  SPIN_RIGHT: 'Spin the robot a full turn to the right',
  RESET_POSE: "Reset the robot's orientation to face forward",
  TOGGLE_LOCK: 'Toggle the gesture lock (locked: all gesture commands are ignored until unlocked)',
};

/** Fixed, deterministic workflows the model may choose (implemented in workflows.js). */
export const AGENT_WORKFLOWS = Object.freeze(['recover_robot', 'checked_command']);

export const AGENT_TOOLS = Object.freeze({
  robot_command: {
    taskType: 'robot.command',
    description:
      'Run one robot command through the application\'s validated command path. ' +
      'WAVE, SPIN_LEFT, SPIN_RIGHT and RESET_POSE are short animations; RESET_POSE also turns the robot back to face forward. ' +
      'TOGGLE_LOCK locks or unlocks gesture commands and always requires the user to confirm in the app. ' +
      'The result reports whether the command really completed or was rejected (e.g. LOCKED, BUSY, COOLDOWN).',
    inputSchema: {
      type: 'object',
      properties: {
        command: { type: 'string', enum: [...AGENT_ROBOT_COMMANDS], description: 'The robot command to run.' },
      },
      required: ['command'],
      additionalProperties: false,
    },
  },
  app_diagnostics: {
    taskType: 'app.diagnostics',
    description:
      'Troubleshooting snapshot of the live application (read-only): vision engine and its connection, camera status, ' +
      'hand-tracking status (unknown while the camera is not live), robot state, current blockers that would make robot ' +
      'commands fail (with what each one affects), AI provider health (configured / last call outcome, never secrets), and ' +
      'recent operational issues such as rejected commands. Use it to explain why something failed.',
    inputSchema: { type: 'object', properties: {}, required: [], additionalProperties: false },
  },
  run_workflow: {
    taskType: null, // runs several validated tasks; see workflows.js
    description:
      'Run a fixed, checked multi-step workflow; every step is a validated app task and consequential steps ask the user to confirm. ' +
      'recover_robot: diagnose, wait for a running action, unlock gestures if locked (asks the user), reset orientation if turned, then verify the result; ' +
      'it stops without acting if a held gesture controls the robot or the robot is in ERROR. ' +
      'checked_command: diagnose blockers first, run the given command only if it would not be rejected, then verify the resulting state. ' +
      'The result reports every step, whether the workflow completed, and whether the outcome was verified.',
    inputSchema: {
      type: 'object',
      properties: {
        workflow: { type: 'string', enum: [...AGENT_WORKFLOWS], description: 'Which workflow to run.' },
        command: { type: 'string', enum: [...AGENT_ROBOT_COMMANDS], description: 'Required for checked_command; not used by recover_robot.' },
      },
      required: ['workflow'],
      additionalProperties: false,
    },
  },
  app_status: {
    taskType: 'app.status',
    description:
      'Read the current live application state: active vision engine, camera status, hand-tracking status, ' +
      'current stable gesture, robot state, robot orientation, gesture lock, and the last finished command. Read-only.',
    inputSchema: { type: 'object', properties: {}, required: [], additionalProperties: false },
  },
});

export const AGENT_TOOL_NAMES = Object.freeze(Object.keys(AGENT_TOOLS));

const isPlainObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v) && Object.getPrototypeOf(v) === Object.prototype;

/**
 * Validate a model-proposed tool call against the allowlist and its schema.
 * @returns {{ ok: true, tool: string, taskType: string, args: object, policy: 'auto'|'confirm', label: string }
 *         | { ok: false, code: 'UNSUPPORTED_TOOL'|'INVALID_ARGS', message: string }}
 */
export function validateToolCall(name, input) {
  if (typeof name !== 'string' || !Object.hasOwn(AGENT_TOOLS, name)) {
    return { ok: false, code: 'UNSUPPORTED_TOOL', message: `tool ${JSON.stringify(String(name).slice(0, 64))} is not available` };
  }
  if (!isPlainObject(input)) return { ok: false, code: 'INVALID_ARGS', message: 'arguments must be an object' };
  if (name === 'app_status' || name === 'app_diagnostics') {
    if (Object.keys(input).length) return { ok: false, code: 'INVALID_ARGS', message: name + ' takes no arguments' };
    return name === 'app_status'
      ? { ok: true, tool: name, taskType: 'app.status', args: {}, policy: 'auto', label: 'Read the current application state' }
      : { ok: true, tool: name, taskType: 'app.diagnostics', args: {}, policy: 'auto', label: 'Read application diagnostics' };
  }
  if (name === 'run_workflow') {
    if (Object.keys(input).some((k) => k !== 'workflow' && k !== 'command')) return { ok: false, code: 'INVALID_ARGS', message: 'run_workflow takes { workflow, command? }' };
    if (!AGENT_WORKFLOWS.includes(input.workflow)) return { ok: false, code: 'INVALID_ARGS', message: 'workflow must be one of ' + AGENT_WORKFLOWS.join(', ') };
    if (input.workflow === 'checked_command') {
      if (!AGENT_ROBOT_COMMANDS.includes(input.command)) return { ok: false, code: 'INVALID_ARGS', message: 'checked_command needs command: one of ' + AGENT_ROBOT_COMMANDS.join(', ') };
      return { ok: true, tool: name, taskType: null, workflow: 'checked_command', args: { workflow: 'checked_command', command: input.command }, policy: 'auto', label: 'Check, run and verify ' + input.command };
    }
    if ('command' in input) return { ok: false, code: 'INVALID_ARGS', message: 'recover_robot takes no command' };
    return { ok: true, tool: name, taskType: null, workflow: 'recover_robot', args: { workflow: 'recover_robot' }, policy: 'auto', label: 'Recover the robot (diagnose, unlock if needed, reset, verify)' };
  }
  const keys = Object.keys(input);
  if (keys.length !== 1 || keys[0] !== 'command') return { ok: false, code: 'INVALID_ARGS', message: 'robot_command takes exactly { command }' };
  if (!AGENT_ROBOT_COMMANDS.includes(input.command)) {
    return { ok: false, code: 'INVALID_ARGS', message: `command must be one of ${AGENT_ROBOT_COMMANDS.join(', ')}` };
  }
  return {
    ok: true,
    tool: name,
    taskType: 'robot.command',
    args: { command: input.command },
    policy: CONFIRM_COMMANDS.includes(input.command) ? 'confirm' : 'auto',
    label: LABELS[input.command],
  };
}

/** Tool definitions in the Claude Messages API format (strict schemas). */
export function providerToolDefinitions() {
  return AGENT_TOOL_NAMES.map((name) => ({
    name,
    description: AGENT_TOOLS[name].description,
    input_schema: AGENT_TOOLS[name].inputSchema,
    // strict schema enforcement where there is something to enforce; parameter-less tools
    // skip it (strict support for an empty object schema is not documented). Server and
    // browser validate every call either way.
    ...(Object.keys(AGENT_TOOLS[name].inputSchema.properties).length ? { strict: true } : {}),
  }));
}

/** Outcome statuses the browser may report back for a tool call. */
/** Tools whose result is a read of application state (no side effects). */
export const READ_ONLY_TOOLS = Object.freeze(['app_status', 'app_diagnostics']);

export const TOOL_OUTCOME_STATUSES = Object.freeze(['succeeded', 'failed', 'rejected', 'declined', 'expired']);

/**
 * Validate a tool outcome sent from the browser to the server.
 * @returns {string|null} error
 */
export function toolOutcomeError(o) {
  if (!isPlainObject(o)) return 'outcome must be an object';
  const allowed = new Set(['status', 'result', 'error', 'taskId']);
  if (Object.keys(o).some((k) => !allowed.has(k))) return 'unknown outcome fields';
  if (!TOOL_OUTCOME_STATUSES.includes(o.status)) return `status must be one of ${TOOL_OUTCOME_STATUSES.join(', ')}`;
  if (o.result !== undefined && o.result !== null && !isPlainObject(o.result)) return 'result must be an object or null';
  if (o.error !== undefined && o.error !== null && !(isPlainObject(o.error) && typeof o.error.code === 'string' && typeof o.error.message === 'string')) {
    return 'error must be { code, message } or null';
  }
  if (o.taskId !== undefined && o.taskId !== null && typeof o.taskId !== 'string') return 'taskId must be a string';
  if (o.status === 'succeeded' && !isPlainObject(o.result)) return 'a succeeded outcome needs a result';
  if (o.status !== 'succeeded' && !o.error) return 'a non-succeeded outcome needs an error';
  try {
    if (JSON.stringify(o).length > 8192) return 'outcome too large';
  } catch {
    return 'outcome must be JSON-serialisable';
  }
  return null;
}
