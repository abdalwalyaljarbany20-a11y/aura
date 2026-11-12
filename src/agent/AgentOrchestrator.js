import { validateToolCall } from './agentTools.js';
import { WorkflowRunner } from './workflows.js';

/**
 * AgentOrchestrator: the browser half of the agent loop.
 *
 *   user text -> server (model plans) -> step
 *     reply      -> show it
 *     tool_call  -> re-validate against the allowlist
 *                   policy 'auto'    -> execute through TaskBroker (origin 'agent')
 *                   policy 'confirm' -> wait for an explicit click on this exact proposal
 *                -> report the TaskBroker record's real outcome -> server (model explains) -> step
 *
 * Phases: unavailable | idle | planning | awaiting_confirmation | executing.
 * One turn at a time (duplicate submissions are refused). Every request carries
 * a fresh requestId and the current epoch; clearing or cancelling bumps the
 * epoch, so a late response can never change the UI or trigger an action.
 * Action messages show the task record's own status, independent of whatever
 * the model later says about it.
 */
const STATUS_FROM_RECORD = { succeeded: 'succeeded', failed: 'failed', rejected: 'rejected' };

export class AgentOrchestrator {
  /**
   * @param {{ client: import('./AgentClient.js').AgentClient, tasks: import('./TaskBroker.js').TaskBroker,
   *           isLocked: () => boolean, now?: () => number, newId?: () => string,
   *           confirmTtlMs?: number, setTimer?: Function, clearTimer?: Function }} deps
   */
  constructor(deps) {
    this.client = deps.client;
    this.tasks = deps.tasks;
    this.isLocked = deps.isLocked;
    this.now = deps.now ?? (() => Date.now());
    this.newId = deps.newId ?? (() => globalThis.crypto.randomUUID());
    this.confirmTtlMs = deps.confirmTtlMs ?? 30000;
    this.setTimer = deps.setTimer ?? ((fn, ms) => setTimeout(fn, ms));
    this.clearTimer = deps.clearTimer ?? ((id) => clearTimeout(id));
    this.phase = 'unavailable';
    this.availability = { available: false, reason: 'checking…', model: null };
    this.messages = []; // { id, role: 'user'|'agent'|'action'|'notice'|'error', text, ... }
    this.pending = null; // awaiting confirmation
    this.conversationId = null;
    this.epoch = 0;
    this._abort = null;
    this._timer = null;
    this._listeners = new Set();
    this.stats = { staleResponsesIgnored: 0, duplicateSubmitsRefused: 0, actionsExecuted: 0, workflowsRun: 0 };
    // Latency, separated: model (server-measured) vs task execution (measured here).
    this.timing = { lastModelMs: null, lastModelCalls: null, lastTaskMs: null, lastWorkflowMs: null };
    this._workflowCancel = false;
    this._inFlight = 0; // dispatched robot actions / workflows that have not finished yet
  }

  onChange(fn) {
    this._listeners.add(fn);
    return () => this._listeners.delete(fn);
  }

  _emit() {
    for (const fn of this._listeners) {
      try {
        fn(this);
      } catch (err) {
        console.error('[agent] listener failed', err);
      }
    }
  }

  _push(msg) {
    const m = { id: this.newId(), at: this.now(), ...msg };
    this.messages.push(m);
    if (this.messages.length > 100) this.messages.splice(0, this.messages.length - 100);
    return m;
  }

  async init() {
    try {
      const s = await this.client.status();
      this.availability = { available: !!s.available, reason: s.reason ?? null, model: s.model ?? null, provider: s.provider ?? null };
    } catch (err) {
      this.availability = { available: false, reason: err.message, model: null };
    }
    this.phase = this.availability.available ? 'idle' : 'unavailable';
    this._emit();
    return this.availability;
  }

  // ------------------------------------------------------------------ user actions

  /** @returns {{ accepted: boolean, reason?: string }} */
  send(text) {
    const t = typeof text === 'string' ? text.trim() : '';
    if (this.phase === 'awaiting_confirmation' && this.pending?.workflow) {
      this.stats.duplicateSubmitsRefused++;
      return { accepted: false, reason: 'BUSY' };
    }
    if (this.phase === 'awaiting_confirmation' && t && t.length <= 2000) {
      // A new request invalidates the pending proposal; the server records it as
      // not executed ("superseded") when the new message arrives.
      const p = this.pending;
      this._clearTimer();
      this.pending = null;
      p.action.status = 'expired';
      p.action.error = 'superseded by a new message; nothing was executed';
      this.phase = 'idle';
    }
    if (this.phase !== 'idle') {
      this.stats.duplicateSubmitsRefused++;
      return { accepted: false, reason: this.phase === 'unavailable' ? 'UNAVAILABLE' : 'BUSY' };
    }
    if (!t) return { accepted: false, reason: 'EMPTY' };
    if (t.length > 2000) return { accepted: false, reason: 'TOO_LONG' };
    this._push({ role: 'user', text: t });
    this._turn((conv, requestId, signal) => this.client.sendMessage(conv, { requestId, text: t }, signal));
    return { accepted: true };
  }

  /** Explicit user confirmation of the pending proposal (from a click in the panel). */
  confirm(proposalId) {
    const p = this.pending;
    if (this.phase !== 'awaiting_confirmation' || !p || p.proposalId !== proposalId) return { ok: false, reason: 'STALE_CONFIRMATION' };
    if (this.now() > p.expiresAt) {
      this._resolvePending('expired', { code: 'CONFIRMATION_EXPIRED', message: 'the confirmation expired before it was given' });
      return { ok: false, reason: 'EXPIRED' };
    }
    if (this.isLocked() !== p.lockedAtProposal) {
      // The toggle would now do the opposite of what the user was shown.
      this._resolvePending('expired', { code: 'STATE_CHANGED', message: 'the gesture lock changed since this was proposed; nothing was executed' });
      return { ok: false, reason: 'STATE_CHANGED' };
    }
    this._clearTimer();
    this.pending = null;
    if (p.workflow) {
      this.phase = 'executing';
      this._emit();
      p.resolve('confirmed');
      return { ok: true };
    }
    this._execute(p.action, p.toolCallId, p.validated, this.epoch);
    return { ok: true };
  }

  decline(proposalId) {
    const p = this.pending;
    if (this.phase !== 'awaiting_confirmation' || !p || p.proposalId !== proposalId) return { ok: false, reason: 'STALE_CONFIRMATION' };
    this._resolvePending('declined', { code: 'USER_DECLINED', message: 'the user declined this action; nothing was executed' });
    return { ok: true };
  }

  /** Abandon the current request where that is safe. */
  cancel() {
    if (this.phase === 'planning') {
      this._abort?.abort();
      this.epoch++;
      this.phase = 'idle';
      this._push({ role: 'notice', text: 'Request cancelled. Nothing was executed.' });
      this._emit();
      return { ok: true };
    }
    if (this.phase === 'awaiting_confirmation') return this.decline(this.pending.proposalId);
    if (this.phase === 'executing' && this._workflowRunning) {
      this._workflowCancel = true;
      return { ok: true, reason: 'The workflow stops before its next step; a step already running finishes on its own.' };
    }
    if (this.phase === 'executing') return { ok: false, reason: 'A running robot action cannot be cancelled; it finishes on its own.' };
    return { ok: false, reason: 'NOTHING_TO_CANCEL' };
  }

  /** Forget the conversation (client and server); late responses are ignored. */
  clear() {
    this.epoch++;
    this._abort?.abort();
    this._clearTimer();
    if (this.conversationId) this.client.deleteConversation(this.conversationId);
    this.conversationId = null;
    if (this.pending?.workflow) this.pending.resolve('expired');
    this._workflowCancel = true;
    this.pending = null;
    this.messages = [];
    // A dispatched robot action / workflow step cannot be recalled: stay 'executing'
    // until it finishes, so no new request can run alongside it.
    this.phase = this._inFlight > 0 ? 'executing' : this._restingPhase();
    this._emit();
  }

  _restingPhase() {
    return this.availability.available ? 'idle' : 'unavailable';
  }

  /** A stale (cleared) execution finished: release the phase once nothing is in flight. */
  _staleExecutionDone() {
    this._stale();
    if (this._inFlight === 0 && this.phase === 'executing' && !this.pending) this.phase = this._restingPhase();
    this._emit();
  }

  dispose() {
    this.epoch++;
    this._abort?.abort();
    this._clearTimer();
    this._listeners.clear();
  }

  // ------------------------------------------------------------------ turn handling

  async _turn(call) {
    const epoch = this.epoch;
    const requestId = this.newId().replace(/[^A-Za-z0-9_-]/g, '');
    const abort = new AbortController();
    this._abort = abort;
    this.phase = 'planning';
    this._emit();
    let step;
    try {
      if (!this.conversationId) this.conversationId = (await this.client.createConversation(abort.signal)).conversationId;
      if (epoch !== this.epoch) return this._stale();
      step = await call(this.conversationId, requestId, abort.signal);
    } catch (err) {
      if (epoch !== this.epoch) return this._stale();
      if (err.code === 'UNKNOWN_CONVERSATION') this.conversationId = null;
      this._push({ role: 'error', code: err.code, text: this._errorText(err) });
      this.phase = 'idle';
      this._emit();
      return;
    } finally {
      if (this._abort === abort) this._abort = null;
    }
    if (epoch !== this.epoch || step?.requestId !== requestId) return this._stale();
    this._handleStep(step, epoch);
  }

  _stale() {
    this.stats.staleResponsesIgnored++;
  }

  _errorText(err) {
    const retry = err.retryAfterS ? ` (retry in about ${Math.ceil(err.retryAfterS)} s)` : '';
    return `${err.message}${retry}`;
  }

  _handleStep(step, epoch) {
    if (step.timing) {
      this.timing.lastModelMs = step.timing.modelMs;
      this.timing.lastModelCalls = step.timing.modelCalls;
    }
    for (const r of step.rejectedToolCalls ?? []) {
      this._push({ role: 'notice', text: `The model proposed an unsupported action (${r.tool}: ${r.code}); it was not executed.` });
    }
    if (step.kind === 'reply') {
      this._push({ role: 'agent', text: step.text, refusal: !!step.refusal, truncated: !!step.truncated });
      this.phase = 'idle';
      this._emit();
      return;
    }
    if (step.kind !== 'tool_call') {
      this._push({ role: 'error', text: 'The agent server sent an unknown response; nothing was executed.' });
      this.phase = 'idle';
      this._emit();
      return;
    }
    if (step.text) this._push({ role: 'agent', text: step.text });
    // Defence in depth: the browser validates again before anything can run.
    const v = validateToolCall(step.tool, step.args);
    if (!v.ok) {
      const action = this._push({ role: 'action', label: `${step.tool}`, status: 'rejected', error: v.message });
      this._report(step.toolCallId, { status: 'rejected', error: { code: v.code, message: v.message } }, action, epoch);
      return;
    }
    if (v.tool === 'run_workflow') {
      this._runWorkflow(step.toolCallId, v, epoch);
      return;
    }
    const locked = this.isLocked();
    const label = v.args.command === 'TOGGLE_LOCK' ? (locked ? 'Unlock gesture commands' : 'Lock gesture commands (gestures are ignored until unlocked)') : v.label;
    const action = this._push({ role: 'action', label, tool: v.tool, args: v.args, policy: v.policy, status: 'proposed' });
    if (v.policy === 'confirm') {
      action.status = 'awaiting_confirmation';
      this.pending = {
        proposalId: this.newId(),
        toolCallId: step.toolCallId,
        validated: v,
        action,
        label,
        lockedAtProposal: locked,
        expiresAt: this.now() + this.confirmTtlMs,
      };
      this.phase = 'awaiting_confirmation';
      this._timer = this.setTimer(() => {
        if (this.pending && this.epoch === epoch) {
          this._resolvePending('expired', { code: 'CONFIRMATION_EXPIRED', message: 'the confirmation request timed out; nothing was executed' });
        }
      }, this.confirmTtlMs);
      this._emit();
      return;
    }
    this._execute(action, step.toolCallId, v, epoch);
  }

  async _execute(action, toolCallId, v, epoch) {
    this.phase = 'executing';
    action.status = 'executing';
    this._emit();
    this.stats.actionsExecuted++;
    // The only execution path: the existing task boundary (same validation as gestures and keys).
    const t0 = this.now();
    this._inFlight++;
    let record;
    try {
      record = await this.tasks.submit({ type: v.taskType, args: v.args, origin: 'agent' });
    } finally {
      this._inFlight--;
    }
    this.timing.lastTaskMs = this.now() - t0;
    action.status = STATUS_FROM_RECORD[record.status] ?? 'failed';
    action.taskId = record.id;
    action.error = record.error ? `${record.error.code}: ${record.error.message}` : null;
    if (epoch !== this.epoch) {
      // Conversation cleared while the action ran: the result is real (HUD log) but is not reported.
      this._staleExecutionDone();
      return;
    }
    const outcome = { status: action.status, taskId: record.id, result: record.result ?? null, error: record.error ?? null };
    this._report(toolCallId, outcome, action, epoch);
  }

  _resolvePending(status, error) {
    const p = this.pending;
    this._clearTimer();
    this.pending = null;
    if (p.workflow) {
      this.phase = 'executing';
      this._emit();
      p.resolve(status === 'declined' ? 'declined' : 'expired');
      return;
    }
    p.action.status = status;
    p.action.error = error.message;
    this._report(p.toolCallId, { status, error }, p.action, this.epoch);
  }

  /**
   * Run a fixed workflow (workflows.js). Every step is a TaskBroker task; the
   * card shows each step's real status; the model receives the compact report.
   */
  async _runWorkflow(toolCallId, v, epoch) {
    const card = this._push({ role: 'workflow', label: v.label, status: 'executing', steps: [], summary: null });
    this.phase = 'executing';
    this._workflowRunning = true;
    this._workflowCancel = false;
    this.stats.workflowsRun++;
    this._emit();
    const t0 = this.now();
    const runner = new WorkflowRunner({
      tasks: this.tasks,
      isCancelled: () => this._workflowCancel || epoch !== this.epoch,
      onUpdate: (r) => {
        card.steps = r.steps.map((s) => ({ id: s.id, label: s.label, status: s.status, error: s.error?.code ?? null, note: s.note ?? null }));
        this._emit();
      },
      requestConfirmation: (label) => this._awaitWorkflowConfirmation(label, epoch),
    });
    let report;
    this._inFlight++;
    try {
      report = await runner.run(v.workflow, v.args);
    } finally {
      this._inFlight--;
      this._workflowRunning = false;
    }
    this.timing.lastWorkflowMs = this.now() - t0;
    card.status = report.outcome === 'completed' ? (report.verified ? 'succeeded' : 'unverified') : 'failed';
    card.summary = report.summary;
    if (epoch !== this.epoch) {
      this._staleExecutionDone();
      return;
    }
    const compact = {
      workflow: report.workflow,
      outcome: report.outcome,
      verified: report.verified,
      summary: report.summary,
      steps: report.steps.map((s) => ({ id: s.id, status: s.status, ...(s.error ? { error: s.error.code } : {}), ...(s.note ? { note: s.note } : {}) })),
      ...(report.observedAfterUnknown ? { observedAfterUnknown: report.observedAfterUnknown } : {}),
    };
    const outcome = report.outcome === 'completed'
      ? { status: 'succeeded', result: compact }
      : { status: 'failed', result: compact, error: { code: report.stoppedReason.code, message: report.stoppedReason.message } };
    this._report(toolCallId, outcome, card, epoch);
  }

  /** A confirmation requested by a running workflow step. Resolves confirmed | declined | expired. */
  _awaitWorkflowConfirmation(label, epoch) {
    return new Promise((resolve) => {
      if (epoch !== this.epoch) return resolve('expired');
      this.pending = {
        proposalId: this.newId(),
        workflow: true,
        label,
        lockedAtProposal: this.isLocked(),
        expiresAt: this.now() + this.confirmTtlMs,
        resolve,
      };
      this.phase = 'awaiting_confirmation';
      this._timer = this.setTimer(() => {
        if (this.pending?.resolve === resolve) this._resolvePending('expired', { code: 'CONFIRMATION_EXPIRED', message: 'timed out' });
      }, this.confirmTtlMs);
      this._emit();
    });
  }

  /** Provider health as seen by the agent server (never secrets). */
  async providerHealth() {
    try {
      const s = await this.client.status();
      return {
        agentServerReachable: true,
        configured: !!s.available,
        model: s.model ?? null,
        unavailableReason: s.available ? null : s.reason ?? null,
        lastCall: s.lastCall ? { ok: s.lastCall.ok, code: s.lastCall.code, at: s.lastCall.at } : null,
        refusalFallback: s.refusalFallback ?? null,
      };
    } catch (err) {
      return { agentServerReachable: false, configured: null, model: null, unavailableReason: err.message, lastCall: null, refusalFallback: null };
    }
  }

  _report(toolCallId, outcome, action, epoch) {
    if (epoch !== this.epoch) return this._stale();
    this._turn((conv, requestId, signal) => this.client.sendToolResult(conv, { requestId, toolCallId, outcome }, signal));
  }

  _clearTimer() {
    if (this._timer !== null) this.clearTimer(this._timer);
    this._timer = null;
  }
}
