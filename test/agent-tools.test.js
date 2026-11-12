import assert from 'node:assert/strict';
import { test } from 'node:test';
import { AGENT_ROBOT_COMMANDS, AGENT_TOOLS, CONFIRM_COMMANDS, providerToolDefinitions, toolOutcomeError, validateToolCall } from '../src/agent/agentTools.js';
import { registerAppStatusTask } from '../src/agent/appStatusTask.js';
import { registerDiagnosticsTask } from '../src/agent/diagnosticsTask.js';
import { WORKFLOW_NAMES } from '../src/agent/workflows.js';
import { AGENT_WORKFLOWS } from '../src/agent/agentTools.js';
import { registerRobotTasks } from '../src/agent/robotTasks.js';
import { TaskBroker } from '../src/agent/TaskBroker.js';
import { DEFAULT_CONFIG } from '../src/config/defaultConfig.js';
import { EventBus } from '../src/core/EventBus.js';
import { COMMANDS } from '../src/interaction/commands.js';
import { InteractionController } from '../src/interaction/InteractionController.js';

test('every agent tool maps to a registered task type; commands match the real registry', () => {
  const bus = new EventBus();
  const broker = new TaskBroker({ bus });
  registerRobotTasks(broker, { controller: new InteractionController(DEFAULT_CONFIG.interaction, { bus }), bus });
  registerAppStatusTask(broker, () => ({}));
  registerDiagnosticsTask(broker, async () => ({}));
  const types = broker.describe().map((d) => d.type);
  for (const t of Object.values(AGENT_TOOLS)) if (t.taskType !== null) assert.ok(types.includes(t.taskType), t.taskType);
  assert.equal(AGENT_TOOLS.run_workflow.taskType, null, 'workflows run several tasks, each through the broker');
  assert.deepEqual(AGENT_WORKFLOWS, WORKFLOW_NAMES, 'catalog and implementation list the same workflows');
  assert.deepEqual(AGENT_ROBOT_COMMANDS, Object.keys(COMMANDS).filter((n) => COMMANDS[n].kind !== 'hold' && !COMMANDS[n].gestureOnly));
  assert.ok(!AGENT_ROBOT_COMMANDS.includes('GRAB_ROTATE') && !AGENT_ROBOT_COMMANDS.includes('AIM'));
  // locomotion stays gesture-only: the agent catalog is unchanged by it
  assert.deepEqual(AGENT_ROBOT_COMMANDS, ['WAVE', 'SPIN_LEFT', 'SPIN_RIGHT', 'RESET_POSE', 'TOGGLE_LOCK']);
  const defs = providerToolDefinitions();
  assert.deepEqual(defs.map((d) => d.name), ['robot_command', 'app_diagnostics', 'run_workflow', 'app_status']);
  assert.ok(defs.every((d) => /^[a-zA-Z0-9_-]{1,64}$/.test(d.name)));
  assert.ok(defs.filter((d) => Object.keys(d.input_schema.properties).length).every((d) => d.strict === true));
});

test('allowlist and schema validation', () => {
  assert.deepEqual(validateToolCall('robot_command', { command: 'WAVE' }).args, { command: 'WAVE' });
  assert.equal(validateToolCall('app_status', {}).taskType, 'app.status');
  for (const [name, input, code] of [
    ['run_shell', { cmd: 'x' }, 'UNSUPPORTED_TOOL'],
    ['__proto__', {}, 'UNSUPPORTED_TOOL'],
    ['robot.command', { command: 'WAVE' }, 'UNSUPPORTED_TOOL'],
    [null, {}, 'UNSUPPORTED_TOOL'],
    ['robot_command', { command: 'DANCE' }, 'INVALID_ARGS'],
    ['robot_command', { command: 'AIM' }, 'INVALID_ARGS'],
    ['robot_command', { command: 'WAVE', speed: 3 }, 'INVALID_ARGS'],
    ['robot_command', 'WAVE', 'INVALID_ARGS'],
    ['robot_command', [], 'INVALID_ARGS'],
    ['app_status', { verbose: true }, 'INVALID_ARGS'],
  ]) {
    const v = validateToolCall(name, input);
    assert.equal(v.ok, false, `${name} ${JSON.stringify(input)}`);
    assert.equal(v.code, code);
  }
});

test('confirmation policy: TOGGLE_LOCK confirm; visual commands and status auto', () => {
  assert.deepEqual(CONFIRM_COMMANDS, ['TOGGLE_LOCK']);
  assert.equal(validateToolCall('robot_command', { command: 'TOGGLE_LOCK' }).policy, 'confirm');
  for (const c of ['WAVE', 'SPIN_LEFT', 'SPIN_RIGHT', 'RESET_POSE']) assert.equal(validateToolCall('robot_command', { command: c }).policy, 'auto');
  assert.equal(validateToolCall('app_status', {}).policy, 'auto');
});

test('tool outcome validation (browser -> server)', () => {
  assert.equal(toolOutcomeError({ status: 'succeeded', result: { command: 'WAVE' }, taskId: 't1' }), null);
  assert.equal(toolOutcomeError({ status: 'failed', error: { code: 'LOCKED', message: 'locked' } }), null);
  for (const bad of [
    null, { status: 'done' }, { status: 'succeeded' }, { status: 'failed' }, { status: 'succeeded', result: {}, extra: 1 },
    { status: 'failed', error: 'x' }, { status: 'succeeded', result: { big: 'x'.repeat(9000) } },
  ]) assert.ok(toolOutcomeError(bad), JSON.stringify(bad)?.slice(0, 60));
});

test('app.status is read-only and returns a copy of live state', async () => {
  let state = { robot: { state: 'IDLE' } };
  const broker = new TaskBroker();
  registerAppStatusTask(broker, () => state);
  const r = await broker.submit({ type: 'app.status', origin: 'agent' });
  assert.equal(r.status, 'succeeded');
  assert.deepEqual(r.result, { robot: { state: 'IDLE' } });
  r.result.robot.state = 'HACKED';
  assert.equal(state.robot.state, 'IDLE');
  assert.equal((await broker.submit({ type: 'app.status', origin: 'agent', args: { x: 1 } })).error.code, 'INVALID_ARGS');
});
