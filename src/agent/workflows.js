import { validateToolCall } from './agentTools.js';

/**
 * Deterministic, bounded workflows. The model may only *choose* a workflow (and
 * its validated argument); the sequence of operations and every decision are
 * fixed here, made from real task outcomes. Each operation goes through
 * TaskBroker, is dispatched at most once, and consequential operations wait for
 * an explicit user confirmation.
 *
 * recover_robot
 *   1 diagnose  app.diagnostics
 *     ERROR          -> stop ROBOT_ERROR      (needs a camera/engine restart: not an agent capability)
 *     CONTROL        -> stop HOLD_ACTIVE      (a held gesture owns the robot: never overridden)
 *     EXECUTING      -> 2 wait (app.status polls, max 3 s) or stop STILL_EXECUTING
 *     LOCKED         -> 3 unlock TOGGLE_LOCK  (confirmation required; declined/expired -> stop)
 *   4 reset     RESET_POSE if |yaw| >= 1 deg, else skipped
 *   5 verify    app.status: unlocked, not ERROR/CONTROL, yaw 0 -> verified
 *
 * checked_command(command)
 *   1 diagnose  app.diagnostics; a blocker that would reject `command` -> stop (nothing dispatched)
 *     EXECUTING -> wait as above
 *   2 execute   robot.command (TOGGLE_LOCK asks for confirmation)
 *   3 verify    app.status: the expected resulting state is observed -> verified, else
 *               "executed, but the resulting state could not be confirmed"
 *
 * Termination: a failed prerequisite stops the workflow. A dispatched step whose
 * outcome is unknown (TIMEOUT / INTERRUPTED) is never repeated: the state is read
 * once and the uncertainty reported. Bounded by maxSteps and deadlineMs.
 */
export const WORKFLOWS = Object.freeze({
  recover_robot: { label: 'Recover the robot: diagnose, unlock if needed, reset orientation, verify', maxSteps: 6 },
  checked_command: { label: 'Check, run and verify a robot command', maxSteps: 5 },
});
export const WORKFLOW_NAMES = Object.freeze(Object.keys(WORKFLOWS));

const UNKNOWN_OUTCOME = new Set(['TIMEOUT', 'INTERRUPTED']);

export class WorkflowRunner {
  /**
   * @param {{ tasks: import('./TaskBroker.js').TaskBroker,
   *           requestConfirmation: (label: string) => Promise<'confirmed'|'declined'|'expired'>,
   *           isCancelled?: () => boolean, onUpdate?: (report: object) => void,
   *           now?: () => number, sleep?: (ms: number) => Promise<void>,
   *           deadlineMs?: number, waitMs?: number, pollMs?: number }} deps
   */
  constructor(deps) {
    this.tasks = deps.tasks;
    this.requestConfirmation = deps.requestConfirmation;
    this.isCancelled = deps.isCancelled ?? (() => false);
    this.onUpdate = deps.onUpdate ?? (() => {});
    this.now = deps.now ?? (() => performance.now());
    this.sleep = deps.sleep ?? ((ms) => new Promise((r) => setTimeout(r, ms)));
    this.deadlineMs = deps.deadlineMs ?? 60000; // includes up to 30 s for a confirmation
    this.waitMs = deps.waitMs ?? 3000;
    this.pollMs = deps.pollMs ?? 250;
    this._used = false;
  }

  /** @returns {Promise<object>} compact report (sent to the model, shown in the UI) */
  async run(name, args = {}) {
    if (this._used) throw new Error('a WorkflowRunner runs exactly one workflow');
    this._used = true;
    const def = WORKFLOWS[name];
    this.report = { workflow: name, args: { ...args }, outcome: 'running', verified: null, stoppedReason: null, steps: [], summary: '' };
    this._start = this.now();
    this._max = def?.maxSteps ?? 0;
    try {
      if (!def) return this._stop('UNKNOWN_WORKFLOW', `workflow ${name} is not available`);
      return name === 'recover_robot' ? await this._recover() : await this._checked(args.command);
    } catch (err) {
      if (err instanceof Halt) return this.report;
      return this._stop('INTERNAL', 'the workflow stopped because of an internal error');
    }
  }

  // ------------------------------------------------------------------ workflows

  async _recover() {
    const d = await this._read('diagnose', 'Read diagnostics', 'app.diagnostics');
    const blocked = (robot) => {
      if (robot.state === 'ERROR') return this._stop('ROBOT_ERROR', `the robot is in ERROR (${robot.error ?? 'unknown cause'}); recovering needs the camera/engine restarted from the app, which the agent cannot do`);
      if (robot.state === 'CONTROL') return this._stop('HOLD_ACTIVE', `${robot.activeControl ?? 'a control'} is held by a gesture; release the gesture first (the agent never overrides it)`);
      return null;
    };
    let state = d.robot;
    if (blocked(state)) return this.report;
    if (state.state === 'EXECUTING') {
      state = await this._waitIdle();
      // the state may have changed while waiting (a new hold, an error): check again before dispatching
      if (blocked(state)) return this.report;
    }
    if (state.gestureLock === 'locked') {
      await this._action('unlock', 'Unlock gesture commands', 'TOGGLE_LOCK', { confirm: true });
    }
    const yaw = state.yawDegrees;
    if (Math.abs(yaw ?? 0) >= 1) await this._action('reset', 'Reset robot orientation', 'RESET_POSE');
    else this._skip('reset', 'Reset robot orientation', yaw === null || yaw === undefined ? 'orientation unknown' : 'already facing forward');
    const v = await this._read('verify', 'Read resulting state', 'app.status');
    const ok = v.robot.gestureLock === 'unlocked' && !['ERROR', 'CONTROL'].includes(v.robot.state) && Math.abs(v.robot.yawDegrees) < 1;
    return this._finish(ok, ok ? 'robot unlocked, idle-ready and facing forward' : `resulting state: ${v.robot.state}, lock ${v.robot.gestureLock}, yaw ${v.robot.yawDegrees} deg`);
  }

  async _checked(command) {
    const v = validateToolCall('robot_command', { command });
    if (!v.ok) return this._stop('INVALID_ARGS', v.message);
    const d = await this._read('diagnose', 'Read diagnostics', 'app.diagnostics');
    const blocked = (robot) => {
      if (robot.state === 'ERROR') return this._stop('ROBOT_ERROR', `the robot is in ERROR (${robot.error ?? 'unknown cause'}); ${command} would be rejected`);
      if (robot.state === 'CONTROL') return this._stop('HOLD_ACTIVE', `${robot.activeControl ?? 'a control'} is held by a gesture; ${command} would be rejected (CONFLICT) until it is released`);
      if (robot.state === 'LOCKED' && command !== 'TOGGLE_LOCK') return this._stop('GESTURE_LOCK_ON', `the gesture lock is on; ${command} would be rejected (LOCKED). Unlock first`);
      return null;
    };
    let robot = d.robot;
    if (blocked(robot)) return this.report;
    if (robot.state === 'EXECUTING') {
      robot = await this._waitIdle();
      if (blocked(robot)) return this.report; // re-checked: nothing is dispatched into a new blocker
    }
    const before = robot;
    const label = command === 'TOGGLE_LOCK'
      ? (before.gestureLock === 'locked' ? 'Unlock gesture commands' : 'Lock gesture commands (gestures are ignored until unlocked)')
      : v.label;
    await this._action('execute', label, command, { confirm: v.policy === 'confirm' });
    const after = (await this._read('verify', 'Read resulting state', 'app.status')).robot;
    let ok;
    let what;
    if (command === 'TOGGLE_LOCK') {
      ok = after.gestureLock !== before.gestureLock;
      what = `gesture lock is now ${after.gestureLock}`;
    } else if (command === 'RESET_POSE') {
      ok = Math.abs(after.yawDegrees) < 1 && after.lastFinishedCommand?.name === 'RESET_POSE';
      what = `orientation ${after.yawDegrees} deg`;
    } else {
      ok = after.lastFinishedCommand?.name === command && after.lastFinishedCommand?.outcome === 'completed';
      what = ok ? `${command} reported completed by the robot controller` : `last finished command: ${after.lastFinishedCommand?.name ?? 'none'}`;
    }
    return this._finish(ok, ok ? what : `${command} executed, but the resulting state could not be confirmed (${what})`);
  }

  // ------------------------------------------------------------------ steps

  _step(id, label, task, args) {
    if (this.report.steps.some((s) => s.id === id)) throw new Error(`step ${id} would be dispatched twice`);
    if (this.report.steps.length >= this._max) throw this._halt('STEP_LIMIT', 'workflow step limit reached');
    if (this.now() - this._start > this.deadlineMs) throw this._halt('DEADLINE', 'workflow time limit reached');
    if (this.isCancelled()) throw this._halt('CANCELLED', 'the workflow was cancelled');
    const step = { id, label, task, args, status: 'executing' };
    this.report.steps.push(step);
    this._emit();
    return step;
  }

  async _submit(step) {
    const rec = await this.tasks.submit({ type: step.task, args: step.args, origin: 'agent' });
    step.taskId = rec.id;
    step.status = rec.status === 'succeeded' ? 'succeeded' : rec.status === 'rejected' ? 'rejected' : 'failed';
    if (rec.error) step.error = { code: rec.error.code, message: rec.error.message };
    this._emit();
    return rec;
  }

  async _read(id, label, task) {
    const step = this._step(id, label, task, {});
    const rec = await this._submit(step);
    if (rec.status !== 'succeeded') throw this._halt('PREREQUISITE_FAILED', `${label} failed (${rec.error?.code ?? rec.status})`);
    return rec.result;
  }

  async _action(id, label, command, { confirm = false } = {}) {
    const v = validateToolCall('robot_command', { command }); // never outside the allowlist
    if (!v.ok) throw this._halt('INVALID_STEP', v.message);
    const step = this._step(id, label, 'robot.command', { command });
    if (confirm) {
      step.status = 'awaiting_confirmation';
      this._emit();
      const answer = await this.requestConfirmation(label);
      if (answer !== 'confirmed') {
        step.status = answer === 'declined' ? 'declined' : 'expired';
        step.error = { code: answer === 'declined' ? 'USER_DECLINED' : 'CONFIRMATION_EXPIRED', message: 'not executed' };
        throw this._halt(step.error.code, `${label} was not confirmed; nothing was executed`);
      }
      if (this.isCancelled()) throw this._halt('CANCELLED', 'the workflow was cancelled');
      step.status = 'executing';
      this._emit();
    }
    const rec = await this._submit(step);
    if (rec.status === 'succeeded') return rec.result;
    if (UNKNOWN_OUTCOME.has(rec.error?.code)) {
      // Outcome unknown: do not repeat; read the state once and report the uncertainty.
      let observed = null;
      try {
        observed = (await this.tasks.submit({ type: 'app.status', origin: 'agent' })).result?.robot ?? null;
      } catch {
        /* reported as unknown */
      }
      this.report.observedAfterUnknown = observed;
      throw this._halt('OUTCOME_UNKNOWN', `${label}: outcome unknown (${rec.error.code}); it was not retried`);
    }
    throw this._halt('STEP_FAILED', `${label} was rejected: ${rec.error?.code ?? rec.status}${rec.error?.message ? ` (${rec.error.message})` : ''}`);
  }

  async _waitIdle() {
    const step = this._step('wait', 'Wait for the running action to finish', 'app.status', {});
    const end = this.now() + this.waitMs;
    let robot = null;
    while (this.now() <= end) {
      const rec = await this.tasks.submit({ type: 'app.status', origin: 'agent' });
      if (rec.status !== 'succeeded') break;
      robot = rec.result.robot;
      if (robot.state !== 'EXECUTING') {
        step.status = 'succeeded';
        this._emit();
        return robot;
      }
      await this.sleep(this.pollMs);
    }
    step.status = 'failed';
    step.error = { code: 'STILL_EXECUTING', message: 'the running action did not finish in time' };
    throw this._halt('STILL_EXECUTING', 'a robot action was still running; nothing else was dispatched');
  }

  _skip(id, label, why) {
    this.report.steps.push({ id, label, task: null, args: {}, status: 'skipped', note: why });
    this._emit();
  }

  // ------------------------------------------------------------------ outcome

  _halt(code, message) {
    this._stop(code, message);
    return new Halt();
  }

  _stop(code, message) {
    this.report.outcome = 'stopped';
    this.report.verified = false;
    this.report.stoppedReason = { code, message };
    this.report.summary = `Stopped: ${message}.`;
    this._emit();
    return this.report;
  }

  _finish(verified, what) {
    this.report.outcome = 'completed';
    this.report.verified = verified;
    this.report.summary = verified ? `Completed and verified: ${what}.` : `Completed, not verified: ${what}.`;
    this._emit();
    return this.report;
  }

  _emit() {
    try {
      this.onUpdate(this.report);
    } catch {
      /* UI listener errors never affect the workflow */
    }
  }
}

class Halt extends Error {}
