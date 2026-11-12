// Deterministic workflows over the REAL TaskBroker, robot tasks and InteractionController.
// No provider is involved here: workflows are trusted application code.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { registerAppStatusTask } from '../src/agent/appStatusTask.js';
import { deriveBlockers, IssueLog, sanitizeMessage } from '../src/agent/diagnostics.js';
import { registerDiagnosticsTask } from '../src/agent/diagnosticsTask.js';
import { registerRobotTasks } from '../src/agent/robotTasks.js';
import { TaskBroker, TaskError } from '../src/agent/TaskBroker.js';
import { WorkflowRunner } from '../src/agent/workflows.js';
import { DEFAULT_CONFIG } from '../src/config/defaultConfig.js';
import { EventBus } from '../src/core/EventBus.js';
import { InteractionController } from '../src/interaction/InteractionController.js';

function world({ camera = 'live' } = {}) {
  let clock = 0;
  const bus = new EventBus();
  const controller = new InteractionController(DEFAULT_CONFIG.interaction, { bus });
  const tasks = new TaskBroker({ bus, now: () => clock, timeoutMs: 5000 });
  registerRobotTasks(tasks, { controller, bus, now: () => clock });
  const status = () => {
    const s = controller.snapshot(clock);
    return {
      robot: {
        state: s.state,
        gestureLock: s.locked ? 'locked' : 'unlocked',
        yawDegrees: Math.round((s.pose.yaw * 180) / Math.PI),
        runningAction: s.action?.name ?? null,
        activeControl: s.hold?.name ?? null,
        lastFinishedCommand: s.lastResult ? { name: s.lastResult.name, outcome: s.lastResult.outcome } : null,
        error: s.error,
      },
    };
  };
  registerAppStatusTask(tasks, status);
  registerDiagnosticsTask(tasks, async () => ({ ...status(), blockers: deriveBlockers(controller.snapshot(clock), camera) }));
  const dispatched = [];
  bus.on('task', ({ record }) => record.status === 'running' && dispatched.push(`${record.type}${record.args.command ? `:${record.args.command}` : ''}`));
  // Advance the robot clock while a command task is pending, so animations complete.
  const tick = (ms) => {
    clock += ms;
    controller.tick(clock);
  };
  const pump = () => {
    const id = setInterval(() => tick(100), 1);
    return () => clearInterval(id);
  };
  return { bus, controller, tasks, dispatched, tick, pump, now: () => clock };
}

function runner(w, answers = [], extra = {}) {
  const asked = [];
  const r = new WorkflowRunner({
    tasks: w.tasks,
    now: w.now,
    sleep: async () => w.tick(250),
    requestConfirmation: async (label) => {
      asked.push(label);
      return answers.shift() ?? 'declined';
    },
    ...extra,
  });
  return { r, asked };
}

const ids = (report) => report.steps.map((s) => `${s.id}:${s.status}`);

test('recover_robot: locked and turned -> confirm unlock -> reset -> verified (fixed order, each step once)', async () => {
  const w = world();
  w.controller.setHandPresent(true, 0);
  w.controller.pose.yaw = Math.PI / 2; // turned by an earlier pinch-drag
  w.controller.submit({ name: 'TOGGLE_LOCK', source: 'keyboard' }, 0);
  w.tick(1000);
  const stop = w.pump();
  const { r, asked } = runner(w, ['confirmed']);
  const report = await r.run('recover_robot');
  stop();
  assert.deepEqual(ids(report), ['diagnose:succeeded', 'unlock:succeeded', 'reset:succeeded', 'verify:succeeded']);
  assert.equal(report.outcome, 'completed');
  assert.equal(report.verified, true);
  assert.deepEqual(asked, ['Unlock gesture commands'], 'the consequential step asked for confirmation');
  assert.deepEqual(w.dispatched, ['app.diagnostics', 'robot.command:TOGGLE_LOCK', 'robot.command:RESET_POSE', 'app.status']);
  assert.equal(w.controller.locked, false);
  assert.equal(w.controller.pose.yaw, 0);
});

test('recover_robot: declined confirmation stops safely; nothing changes', async () => {
  const w = world();
  w.controller.pose.yaw = 1;
  w.controller.submit({ name: 'TOGGLE_LOCK', source: 'keyboard' }, 0);
  const { r } = runner(w, ['declined']);
  const report = await r.run('recover_robot');
  assert.equal(report.outcome, 'stopped');
  assert.equal(report.stoppedReason.code, 'USER_DECLINED');
  assert.deepEqual(ids(report), ['diagnose:succeeded', 'unlock:declined']);
  assert.deepEqual(w.dispatched, ['app.diagnostics'], 'no robot command was dispatched');
  assert.equal(w.controller.locked, true);
  assert.equal(w.controller.pose.yaw, 1);
});

test('recover_robot never overrides a held gesture and reports why', async () => {
  const w = world();
  w.controller.setHandPresent(true, 0);
  w.controller.handleGestureEvents([{ type: 'start', gesture: 'PINCH' }], 0, { features: { valid: true, palmCenterNorm: { x: 0.5, y: 0.5 } } });
  const { r, asked } = runner(w, ['confirmed']);
  const report = await r.run('recover_robot');
  assert.equal(report.stoppedReason.code, 'HOLD_ACTIVE');
  assert.match(report.stoppedReason.message, /GRAB_ROTATE is held by a gesture/);
  assert.deepEqual(w.dispatched, ['app.diagnostics']);
  assert.deepEqual(asked, []);
  assert.equal(w.controller.state, 'CONTROL');
  assert.equal(w.controller.hold.name, 'GRAB_ROTATE', 'the hold is untouched');
});

test('recover_robot stops on robot ERROR (no pretend repair)', async () => {
  const w = world({ camera: 'error' });
  w.controller.fail('camera interrupted', 0);
  const report = await runner(w).r.run('recover_robot');
  assert.equal(report.stoppedReason.code, 'ROBOT_ERROR');
  assert.match(report.stoppedReason.message, /restarted from the app, which the agent cannot do/);
  assert.deepEqual(w.dispatched, ['app.diagnostics']);
});

test('recover_robot when nothing is wrong: reset skipped, verified', async () => {
  const w = world();
  const report = await runner(w).r.run('recover_robot');
  assert.deepEqual(ids(report), ['diagnose:succeeded', 'reset:skipped', 'verify:succeeded']);
  assert.equal(report.verified, true);
  assert.deepEqual(w.dispatched, ['app.diagnostics', 'app.status']);
});

test('a running action is waited for (bounded), never interrupted', async () => {
  const w = world();
  w.controller.submit({ name: 'WAVE', source: 'keyboard' }, 0); // 1.8 s animation
  w.controller.pose.yaw = 0.5;
  const stop = w.pump();
  const report = await runner(w, [], { waitMs: 3000 }).r.run('recover_robot');
  stop();
  assert.deepEqual(ids(report).slice(0, 2), ['diagnose:succeeded', 'wait:succeeded']);
  assert.equal(report.verified, true);
  const w2 = world();
  w2.controller.submit({ name: 'WAVE', source: 'keyboard' }, 0);
  const r2 = await runner(w2, [], { waitMs: 500, sleep: async () => {} , now: (() => { let t = 0; return () => (t += 200); })() }).r.run('recover_robot');
  assert.equal(r2.stoppedReason.code, 'STILL_EXECUTING');
  assert.ok(!w2.dispatched.some((d) => d.startsWith('robot.command')));
});

test('blockers are re-checked after waiting: a hold that starts during the wait stops it before dispatch', async () => {
  for (const [name, args] of [['recover_robot', {}], ['checked_command', { command: 'WAVE' }]]) {
    const w = world();
    w.controller.setHandPresent(true, 0);
    w.controller.pose.yaw = 0.5; // recover_robot would otherwise reset
    w.controller.submit({ name: 'WAVE', source: 'keyboard' }, 0);
    let started = false;
    const sleep = async () => {
      w.tick(2000); // the running action finishes ...
      if (!started) {
        started = true; // ... and the user starts a pinch hold before the next poll
        w.controller.handleGestureEvents([{ type: 'start', gesture: 'PINCH' }], w.now(), { features: { valid: true, palmCenterNorm: { x: 0.5, y: 0.5 } } });
      }
    };
    const report = await runner(w, ['confirmed'], { sleep }).r.run(name, args);
    assert.equal(report.stoppedReason?.code, 'HOLD_ACTIVE', name);
    assert.ok(!w.dispatched.some((d) => d.startsWith('robot.command')), `${name}: nothing dispatched into the hold`);
    assert.equal(w.controller.hold?.name, 'GRAB_ROTATE');
  }
});

test('checked_command: blockers stop it before dispatch; success is verified from observed state', async () => {
  const w = world();
  w.controller.submit({ name: 'TOGGLE_LOCK', source: 'keyboard' }, 0);
  const locked = await runner(w).r.run('checked_command', { command: 'WAVE' });
  assert.equal(locked.stoppedReason.code, 'GESTURE_LOCK_ON');
  assert.deepEqual(w.dispatched, ['app.diagnostics']);

  const ok = world();
  const stop = ok.pump();
  const report = await runner(ok).r.run('checked_command', { command: 'WAVE' });
  stop();
  assert.deepEqual(ids(report), ['diagnose:succeeded', 'execute:succeeded', 'verify:succeeded']);
  assert.equal(report.verified, true);
  assert.match(report.summary, /WAVE reported completed by the robot controller/);
});

test('checked_command TOGGLE_LOCK asks first and verifies the flipped lock', async () => {
  const w = world();
  const { r, asked } = runner(w, ['confirmed']);
  const report = await r.run('checked_command', { command: 'TOGGLE_LOCK' });
  assert.equal(asked.length, 1);
  assert.match(asked[0], /^Lock gesture commands/, 'the confirmation names the actual effect');
  assert.equal(report.verified, true);
  assert.match(report.summary, /gesture lock is now locked/);
  assert.equal((await runner(world()).r.run('checked_command', { command: 'AIM' })).stoppedReason.code, 'INVALID_ARGS');
});

test('a rejected command is reported as such and changes nothing', async () => {
  const w = world();
  w.controller.submit({ name: 'RESET_POSE', source: 'keyboard' }, 0); // cooldown for RESET_POSE
  w.tick(800); // finished, but still within its 900 ms cooldown
  const yaw = (w.controller.pose.yaw = 0.7);
  const report = await runner(w).r.run('checked_command', { command: 'RESET_POSE' });
  assert.equal(report.stoppedReason.code, 'STEP_FAILED');
  assert.match(report.stoppedReason.message, /COOLDOWN/);
  assert.equal(w.controller.pose.yaw, yaw);
});

test('unknown outcome: not retried, state read once, uncertainty reported', async () => {
  const bus = new EventBus();
  const tasks = new TaskBroker({ bus });
  let calls = 0;
  tasks.register('robot.command', { validateArgs: () => null, execute: async () => { calls++; throw new TaskError('TIMEOUT', 'no completion seen'); } });
  registerAppStatusTask(tasks, () => ({ robot: { state: 'EXECUTING', gestureLock: 'unlocked', yawDegrees: 0, lastFinishedCommand: null } }));
  registerDiagnosticsTask(tasks, async () => ({ robot: { state: 'IDLE', gestureLock: 'unlocked', yawDegrees: 0 }, blockers: [] }));
  const report = await new WorkflowRunner({ tasks, requestConfirmation: async () => 'confirmed' }).run('checked_command', { command: 'WAVE' });
  assert.equal(report.stoppedReason.code, 'OUTCOME_UNKNOWN');
  assert.equal(calls, 1, 'never retried');
  assert.equal(report.observedAfterUnknown.state, 'EXECUTING');
});

test('bounds: cancellation, deadline, single use, unknown workflow', async () => {
  const w = world();
  let n = 0;
  const cancelled = await runner(w, [], { isCancelled: () => n++ > 0 }).r.run('recover_robot');
  assert.equal(cancelled.stoppedReason.code, 'CANCELLED');
  assert.equal(w.dispatched.length, 1);
  const late = await runner(world(), [], { deadlineMs: -1 }).r.run('recover_robot');
  assert.equal(late.stoppedReason.code, 'DEADLINE');
  const { r } = runner(world());
  await r.run('recover_robot');
  await assert.rejects(r.run('recover_robot'), /exactly one workflow/);
  assert.equal((await runner(world()).r.run('format_disk')).stoppedReason.code, 'UNKNOWN_WORKFLOW');
});

test('diagnostics helpers: blockers from real state, sanitised bounded issue log', () => {
  const c = new InteractionController(DEFAULT_CONFIG.interaction);
  assert.deepEqual(deriveBlockers(c.snapshot(0), 'live'), []);
  assert.deepEqual(deriveBlockers(c.snapshot(0), 'stopped').map((b) => b.code), ['CAMERA_NOT_LIVE']);
  assert.match(deriveBlockers(c.snapshot(0), 'stopped')[0].affects, /agent still work/);
  c.setHandPresent(true, 0);
  c.handleGestureEvents([{ type: 'start', gesture: 'POINT' }], 0, {});
  assert.match(deriveBlockers(c.snapshot(0), 'live')[0].detail, /AIM is active \(held POINT gesture\)/);
  assert.equal(sanitizeMessage('model at D:\\Ai_Cv\\public\\models\\x.task missing; key sk-ant-abc123'), 'model at <path> missing; key [redacted]');
  assert.equal(sanitizeMessage('x'.repeat(500)).length, 140);
  let t = 0;
  const log = new IssueLog(3, () => t);
  for (let i = 0; i < 5; i++) {
    t = i * 1000;
    log.record('command', 'LOCKED', `rejected ${i}`);
  }
  const recent = log.recent(5);
  assert.equal(recent.length, 3);
  assert.deepEqual(recent.map((x) => x.message), ['rejected 4', 'rejected 3', 'rejected 2']);
  assert.equal(recent[2].secondsAgo, 2);
});
