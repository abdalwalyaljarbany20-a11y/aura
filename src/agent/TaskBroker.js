/**
 * TaskBroker: the integration boundary for a future AI agent (Phase 3).
 *
 * Vocabulary (kept distinct on purpose):
 *   gesture event     what the CV engine saw            -> gestureEvents.js (never creates tasks)
 *   command request   a validated robot command          -> InteractionController.submit
 *   task request      a user/agent asks for work         -> TaskBroker.submit(request)
 *   task record       request + lifecycle + outcome      -> returned by submit, emitted as 'task' events
 *   task result       structured success payload         -> record.result
 *   execution failure structured error                   -> record.error { code, message }
 *
 * Lifecycle: received -> rejected (invalid request / args)
 *                     -> running -> succeeded | failed (executor error, TIMEOUT)
 *
 * Only registered task types can run, and each validates its own arguments.
 * Gestures are not a valid task origin: a raw gesture can never trigger an
 * agent action. This module makes no network calls and holds no credentials;
 * an agent adapter (Phase 3) will call `submit` with origin 'agent'.
 */

export const TASK_ORIGINS = Object.freeze(['agent', 'voice', 'keyboard', 'test']);
export const TASK_STATUSES = Object.freeze(['received', 'rejected', 'running', 'succeeded', 'failed']);
const ID_RE = /^[A-Za-z0-9_-]{1,64}$/;
const TYPE_RE = /^[a-z][a-z0-9]*(\.[a-z][a-z0-9]*)+$/; // e.g. robot.command

const isPlainObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v) && Object.getPrototypeOf(v) === Object.prototype;

/** Failure raised by an executor; `code` becomes record.error.code. */
export class TaskError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

/** @returns {string|null} why a task request is malformed (type registration is checked separately) */
export function taskRequestError(req) {
  if (!isPlainObject(req)) return 'task request must be a plain object';
  const allowed = new Set(['id', 'type', 'args', 'origin']);
  const extra = Object.keys(req).filter((k) => !allowed.has(k));
  if (extra.length) return `unknown request fields: ${extra.join(', ')}`;
  if (req.id !== undefined && !(typeof req.id === 'string' && ID_RE.test(req.id))) return 'id must match [A-Za-z0-9_-]{1,64}';
  if (typeof req.type !== 'string' || !TYPE_RE.test(req.type)) return 'type must look like "domain.action"';
  if (req.args !== undefined && !isPlainObject(req.args)) return 'args must be a plain object';
  if (!TASK_ORIGINS.includes(req.origin)) return `origin must be one of ${TASK_ORIGINS.join(', ')}`;
  try {
    if (JSON.stringify(req.args ?? {}).length > 4096) return 'args too large';
  } catch {
    return 'args must be JSON-serialisable';
  }
  return null;
}

export class TaskBroker {
  /**
   * @param {{ bus?: any, now?: () => number, newId?: () => string, timeoutMs?: number, maxRecords?: number }} [deps]
   */
  constructor(deps = {}) {
    this.bus = deps.bus ?? null;
    this.now = deps.now ?? (() => performance.now());
    this.newId = deps.newId ?? (() => globalThis.crypto.randomUUID());
    this.timeoutMs = deps.timeoutMs ?? 10000;
    this.maxRecords = deps.maxRecords ?? 50;
    /** @type {Map<string, { validateArgs: (args: object) => string|null, execute: (args: object, ctx: object) => Promise<object>, timeoutMs?: number, description?: string }>} */
    this.types = new Map();
    /** @type {Map<string, object>} */
    this.records = new Map();
  }

  register(type, handler) {
    if (!TYPE_RE.test(type)) throw new Error(`invalid task type name ${type}`);
    if (this.types.has(type)) throw new Error(`task type ${type} already registered`);
    if (typeof handler?.validateArgs !== 'function' || typeof handler?.execute !== 'function') throw new Error('handler needs validateArgs and execute');
    this.types.set(type, handler);
  }

  /** Registered task types, for an agent to discover what it may request. */
  describe() {
    return [...this.types.entries()].map(([type, h]) => ({ type, description: h.description ?? '' }));
  }

  get(id) {
    const r = this.records.get(id);
    return r ? structuredClone(r) : null;
  }

  /**
   * Validate and run a task. Never throws: every outcome is a record.
   * @param {{ id?: string, type: string, args?: object, origin: string }} req
   */
  async submit(req) {
    const t = this.now();
    const malformed = taskRequestError(req);
    const id = (!malformed && req.id) || this.newId();
    const record = {
      id,
      type: typeof req?.type === 'string' ? req.type : null,
      args: isPlainObject(req?.args) ? structuredClone(req.args) : {},
      origin: TASK_ORIGINS.includes(req?.origin) ? req.origin : null,
      status: 'received',
      createdAt: t,
      finishedAt: null,
      result: null,
      error: null,
    };
    if (malformed) return this._finish(record, 'rejected', null, { code: 'INVALID_REQUEST', message: malformed });
    if (this.records.has(id)) return this._finish({ ...record, id: this.newId() }, 'rejected', null, { code: 'DUPLICATE_ID', message: `task id ${id} already used` });
    const handler = this.types.get(req.type);
    if (!handler) return this._finish(record, 'rejected', null, { code: 'UNSUPPORTED_TASK', message: `task type ${req.type} is not available` });
    const argError = handler.validateArgs(record.args);
    if (argError) return this._finish(record, 'rejected', null, { code: 'INVALID_ARGS', message: argError });

    this._store(record);
    record.status = 'running';
    this._emit(record);
    let timer;
    try {
      const timeoutMs = handler.timeoutMs ?? this.timeoutMs;
      const result = await Promise.race([
        Promise.resolve().then(() => handler.execute(structuredClone(record.args), { id, origin: record.origin })),
        new Promise((_, reject) => {
          timer = setTimeout(() => reject(new TaskError('TIMEOUT', `task did not finish within ${timeoutMs} ms`)), timeoutMs);
        }),
      ]);
      if (!isPlainObject(result)) throw new TaskError('INTERNAL', 'executor returned no structured result');
      return this._finish(record, 'succeeded', result, null);
    } catch (err) {
      // Structured error only: no stack traces leave the broker.
      const code = err instanceof TaskError ? err.code : 'INTERNAL';
      const message = err instanceof TaskError ? err.message : 'task executor failed';
      if (!(err instanceof TaskError)) console.error(`[TaskBroker] ${record.type} failed`, err);
      return this._finish(record, 'failed', null, { code, message });
    } finally {
      clearTimeout(timer);
    }
  }

  _finish(record, status, result, error) {
    record.status = status;
    record.result = result;
    record.error = error;
    record.finishedAt = this.now();
    this._store(record);
    this._emit(record);
    return structuredClone(record);
  }

  _store(record) {
    this.records.set(record.id, record);
    while (this.records.size > this.maxRecords) this.records.delete(this.records.keys().next().value);
  }

  _emit(record) {
    this.bus?.emit('task', { record: structuredClone(record) });
  }
}
