import assert from 'node:assert/strict';
import { test } from 'node:test';
import { registerRobotTasks } from '../src/agent/robotTasks.js';
import { TaskBroker, TaskError, taskRequestError } from '../src/agent/TaskBroker.js';
import { DEFAULT_CONFIG } from '../src/config/defaultConfig.js';
import { EventBus } from '../src/core/EventBus.js';
import { InteractionController } from '../src/interaction/InteractionController.js';

function setup() {
  let clock = 0;
  const bus = new EventBus();
  const tasks = [];
  bus.on('task', (e) => tasks.push(e.record.status));
  const controller = new InteractionController(DEFAULT_CONFIG.interaction, { bus });
  let n = 0;
  const broker = new TaskBroker({ bus, now: () => clock, newId: () => `t${++n}`, timeoutMs: 200 });
  registerRobotTasks(broker, { controller, bus, now: () => clock });
  return { bus, controller, broker, tasks, advance: (ms) => (clock += ms), clock: () => clock };
}

test('task request contract validation', () => {
  assert.equal(taskRequestError({ type: 'robot.command', args: { command: 'WAVE' }, origin: 'agent' }), null);
  assert.equal(taskRequestError({ id: 'abc-1', type: 'robot.command', origin: 'test' }), null);
  for (const bad of [
    null, [], 'x', { type: 'robot.command', origin: 'gesture' }, { type: 'robot.command', origin: 'internet' },
    { type: 'Robot', origin: 'agent' }, { type: 'robot.command', origin: 'agent', args: [] },
    { type: 'robot.command', origin: 'agent', id: 'bad id!' }, { type: 'robot.command', origin: 'agent', extra: 1 },
    { type: 'robot.command', origin: 'agent', args: { blob: 'x'.repeat(5000) } },
  ]) assert.ok(taskRequestError(bad), JSON.stringify(bad)?.slice(0, 60));
});

test('a robot.command task succeeds only after the command really completed', async () => {
  const s = setup();
  const p = s.broker.submit({ type: 'robot.command', args: { command: 'WAVE' }, origin: 'agent' });
  await Promise.resolve();
  assert.equal(s.controller.state, 'EXECUTING');
  assert.equal(s.broker.get('t1').status, 'running');
  s.advance(1800);
  s.controller.tick(s.clock());
  const rec = await p;
  assert.equal(rec.status, 'succeeded');
  assert.deepEqual(rec.result, { command: 'WAVE', state: 'IDLE', pose: { yaw: 0, x: 0, z: 0, lift: 0 }, completedAt: 1800 });
  assert.deepEqual(s.tasks, ['running', 'succeeded']);
});

test('controller rejections become structured task failures (no fake success)', async () => {
  const s = setup();
  s.controller.submit({ name: 'TOGGLE_LOCK', source: 'keyboard' }, 0);
  const rec = await s.broker.submit({ type: 'robot.command', args: { command: 'SPIN_LEFT' }, origin: 'agent' });
  assert.equal(rec.status, 'failed');
  assert.equal(rec.error.code, 'LOCKED');
  assert.equal(rec.result, null);
  const tooSoon = await s.broker.submit({ type: 'robot.command', args: { command: 'TOGGLE_LOCK' }, origin: 'agent' });
  assert.equal(tooSoon.error.code, 'COOLDOWN', 'same validation as every other command source');
  s.advance(1000);
  const unlock = await s.broker.submit({ type: 'robot.command', args: { command: 'TOGGLE_LOCK' }, origin: 'agent' });
  assert.equal(unlock.status, 'succeeded');
  assert.equal(unlock.result.state, 'IDLE');
});

test('invalid, unsupported, duplicate and hold-command tasks are rejected', async () => {
  const s = setup();
  assert.equal((await s.broker.submit({ type: 'robot.command', origin: 'gesture', args: { command: 'WAVE' } })).error.code, 'INVALID_REQUEST');
  assert.equal((await s.broker.submit({ type: 'files.delete', origin: 'agent' })).error.code, 'UNSUPPORTED_TASK');
  assert.equal((await s.broker.submit({ type: 'robot.command', origin: 'agent', args: { command: 'AIM' } })).error.code, 'INVALID_ARGS');
  assert.equal((await s.broker.submit({ type: 'robot.command', origin: 'agent', args: { command: 'WAVE', speed: 9 } })).error.code, 'INVALID_ARGS');
  const first = s.broker.submit({ id: 'job-1', type: 'robot.command', origin: 'agent', args: { command: 'RESET_POSE' } });
  const dup = await s.broker.submit({ id: 'job-1', type: 'robot.command', origin: 'agent', args: { command: 'RESET_POSE' } });
  assert.equal(dup.error.code, 'DUPLICATE_ID');
  s.advance(700);
  s.controller.tick(s.clock());
  assert.equal((await first).status, 'succeeded');
  assert.equal(s.controller.counts.accepted, 1, 'rejected tasks never reached the controller');
});

test('robot error during execution fails the task with INTERRUPTED; no completion -> TIMEOUT', async () => {
  const s = setup();
  const p = s.broker.submit({ type: 'robot.command', args: { command: 'WAVE' }, origin: 'agent' });
  await Promise.resolve();
  s.controller.fail('camera interrupted', 10);
  const rec = await p;
  assert.equal(rec.status, 'failed');
  assert.equal(rec.error.code, 'INTERRUPTED');
  s.controller.recover(20);
  const stuck = await s.broker.submit({ type: 'robot.command', args: { command: 'SPIN_RIGHT' }, origin: 'agent' }); // never ticked
  assert.equal(stuck.status, 'failed');
  assert.equal(stuck.error.code, 'TIMEOUT');
});

test('executor exceptions are reported without internals; records are bounded', async () => {
  const broker = new TaskBroker({ maxRecords: 3 });
  broker.register('test.boom', { validateArgs: () => null, execute: () => { throw new Error('secret stack detail'); } });
  broker.register('test.coded', { validateArgs: () => null, execute: () => { throw new TaskError('NOPE', 'not allowed'); } });
  const orig = console.error;
  console.error = () => {};
  const r = await broker.submit({ type: 'test.boom', origin: 'test' });
  console.error = orig;
  assert.deepEqual(r.error, { code: 'INTERNAL', message: 'task executor failed' });
  assert.deepEqual((await broker.submit({ type: 'test.coded', origin: 'test' })).error, { code: 'NOPE', message: 'not allowed' });
  for (let i = 0; i < 5; i++) await broker.submit({ type: 'test.coded', origin: 'test' });
  assert.equal(broker.records.size, 3);
  assert.throws(() => broker.register('test.coded', { validateArgs() {}, execute() {} }), /already/);
  assert.deepEqual(broker.describe().map((d) => d.type), ['test.boom', 'test.coded']);
});
