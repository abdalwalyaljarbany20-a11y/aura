/**
 * PythonEngineClient: WebSocket client for the Python CV engine (protocol v2,
 * see backend/aura_cv/protocol.py).
 *
 * - validates every incoming message; malformed ones are counted and dropped
 *   (never turned into detections)
 * - parses binary preview frames ("AVF2" + uint32 session + uint32 frameId + JPEG)
 * - reconnects with capped exponential backoff while the app runs
 *
 * The WebSocket constructor is injectable for tests.
 */
import { gestureEventError } from '../interaction/gestureEvents.js';

export const PROTOCOL_VERSION = 2;
const PREVIEW_MAGIC = [0x41, 0x56, 0x46, 0x32]; // "AVF2"
export const TRACKING_STATUSES = Object.freeze(['DETECTED', 'NOT_DETECTED', 'TRACKING_LOST', 'CAMERA_UNAVAILABLE', 'PROCESSING_ERROR']);

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

/** @returns {string|null} why a landmark list is unusable */
function landmarksError(lms) {
  if (!Array.isArray(lms) || lms.length !== 21) return 'expected 21 landmarks';
  for (const p of lms) if (!Array.isArray(p) || p.length !== 3 || !p.every(isNum)) return 'landmark is not [x, y, z]';
  return null;
}

/**
 * Validate a `result` message from the engine.
 * @returns {string|null} error or null
 */
export function resultError(m) {
  if (!TRACKING_STATUSES.includes(m.status)) return `unknown status ${m.status}`;
  if (!Number.isInteger(m.session) || m.session < 1) return 'session must be a positive integer';
  if (m.frameId !== null && !(Number.isInteger(m.frameId) && m.frameId > 0)) return 'frameId must be a positive integer or null';
  if (!Array.isArray(m.hands)) return 'hands must be an array';
  for (const h of m.hands) {
    const e = landmarksError(h?.landmarks);
    if (e) return e;
    if (h.worldLandmarks !== null && h.worldLandmarks !== undefined && landmarksError(h.worldLandmarks)) return 'bad world landmarks';
    if (h.score !== null && h.score !== undefined && !isNum(h.score)) return 'bad score';
  }
  if (m.status === 'DETECTED' && !(Number.isInteger(m.primary) && m.primary >= 0 && m.primary < m.hands.length)) return 'DETECTED without a valid primary hand';
  if (m.status !== 'DETECTED' && m.primary !== null) return 'primary hand reported without detection';
  if (!m.gesture || typeof m.gesture.stable !== 'string' || typeof m.gesture.label !== 'string') return 'gesture missing';
  if (!Array.isArray(m.events)) return 'events must be an array';
  if (m.faces !== undefined) {
    if (!Array.isArray(m.faces) || m.faces.length > 16) return 'faces must be an array (max 16)';
    for (const f of m.faces) if (!f || ![f.x, f.y, f.w, f.h, f.score].every(isNum)) return 'bad face';
  }
  const mo = m.gesture.motion;
  if (mo !== undefined && mo !== null && !(isNum(mo?.spin?.omega) && isNum(mo?.walk?.cadence))) return 'bad gesture.motion';
  for (const e of m.events) {
    const err = gestureEventError(e);
    if (err) return `bad event: ${err}`;
  }
  return null;
}

/** @param {ArrayBuffer} buf */
export function parsePreview(buf) {
  const b = new Uint8Array(buf);
  if (b.length < 12 || PREVIEW_MAGIC.some((v, i) => b[i] !== v)) return null;
  const dv = new DataView(buf);
  return { session: dv.getUint32(4, false), frameId: dv.getUint32(8, false), jpeg: buf.slice(12) };
}

export class PythonEngineClient {
  /**
   * @param {string} url
   * @param {{ bus?: any, WebSocketImpl?: any, connectTimeoutMs?: number, now?: () => number,
   *           setTimeout?: Function, clearTimeout?: Function }} [deps]
   */
  constructor(url, deps = {}) {
    this.url = url;
    this.bus = deps.bus ?? null;
    this.WS = deps.WebSocketImpl ?? globalThis.WebSocket;
    this.connectTimeoutMs = deps.connectTimeoutMs ?? 4000;
    this.now = deps.now ?? (() => performance.now());
    this._setTimeout = deps.setTimeout ?? setTimeout.bind(globalThis);
    this._clearTimeout = deps.clearTimeout ?? clearTimeout.bind(globalThis);
    this._random = deps.random ?? Math.random;
    /** @type {'idle'|'connecting'|'connected'|'disconnected'} */
    this.status = 'idle';
    this.hello = null;
    this.engineStatus = null; // { camera, tracker }
    this.metrics = null;
    this.lastError = null;
    this.stats = { results: 0, previews: 0, malformed: 0, reconnects: 0, interruptions: 0 };
    // Outage timing (ms, monotonic): from losing a working connection to the next valid hello.
    this.outage = { since: null, lastDurationMs: null };
    this.nextRetryMs = null;
    this.handlers = { result: [], preview: [], status: [], hello: [], connection: [], metrics: [] };
    this.ws = null;
    this._retryMs = 500;
    this._retryTimer = null;
    this._closed = false;
    this._pendingStart = null;
  }

  on(type, fn) {
    this.handlers[type].push(fn);
    return () => {
      this.handlers[type] = this.handlers[type].filter((f) => f !== fn);
    };
  }

  _emit(type, payload) {
    for (const fn of this.handlers[type]) {
      try {
        fn(payload);
      } catch (err) {
        console.error(`[PythonEngineClient] ${type} handler failed`, err);
      }
    }
  }

  /** Connect (and keep reconnecting). Resolves with the hello message or rejects on first failure. */
  connect() {
    this._closed = false;
    return new Promise((resolve, reject) => {
      let settled = false;
      const timer = this._setTimeout(() => {
        if (!settled) {
          settled = true;
          reject(new Error(`Python engine not reachable at ${this.url} (start it with "npm run backend")`));
        }
      }, this.connectTimeoutMs);
      const once = this.on('hello', (h) => {
        once();
        if (!settled) {
          settled = true;
          this._clearTimeout(timer);
          resolve(h);
        }
      });
      this._open();
    });
  }

  _open() {
    if (this._closed) return;
    this._setStatus('connecting');
    let ws;
    try {
      ws = new this.WS(this.url);
    } catch (err) {
      this.lastError = err?.message ?? String(err);
      this._scheduleReconnect();
      return;
    }
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    ws.onopen = () => {
      // Backoff is reset only after a valid hello: an engine that accepts and
      // immediately closes (e.g. "busy") must not be retried every 500 ms.
      ws.send(JSON.stringify({ type: 'hello', client: 'aura-frontend', protocol: PROTOCOL_VERSION }));
    };
    ws.onmessage = (ev) => this._onMessage(ev.data);
    ws.onerror = () => {
      this.lastError = `WebSocket error (${this.url})`;
    };
    ws.onclose = (ev) => {
      if (this.ws !== ws) return;
      this.ws = null;
      this.hello = null;
      const reason = ev?.reason || (ev?.code === 1013 ? 'engine busy' : 'connection closed');
      this.lastError = reason;
      if (this.status === 'connected') {
        this.stats.interruptions++;
        this.outage.since = this.now();
      }
      this._setStatus('disconnected', reason);
      this._resolveStart({ ok: false, error: { code: 'ENGINE_DISCONNECTED', message: `Python engine disconnected: ${reason}` } });
      this._scheduleReconnect();
    };
  }

  /** Bounded exponential backoff with ±20 % jitter: ~0.5 s, 1 s, 2 s, 4 s, then 5 s. */
  _scheduleReconnect() {
    if (this._closed) return;
    this.stats.reconnects++;
    const delay = Math.round(this._retryMs * (0.8 + 0.4 * this._random()));
    this.nextRetryMs = delay;
    this._retryTimer = this._setTimeout(() => this._open(), delay);
    this._retryMs = Math.min(5000, this._retryMs * 2);
  }

  _setStatus(status, reason) {
    this.status = status;
    this._emit('connection', { status, reason: reason ?? null });
  }

  _onMessage(data) {
    if (data instanceof ArrayBuffer) {
      const p = parsePreview(data);
      if (!p) {
        this.stats.malformed++;
        return;
      }
      this.stats.previews++;
      this._emit('preview', p);
      return;
    }
    let m;
    try {
      m = JSON.parse(data);
    } catch {
      this.stats.malformed++;
      return;
    }
    if (!m || typeof m !== 'object') {
      this.stats.malformed++;
      return;
    }
    switch (m.type) {
      case 'hello':
        if (m.protocol !== PROTOCOL_VERSION) {
          this.lastError = `engine protocol ${m.protocol} is not supported (expected ${PROTOCOL_VERSION})`;
          this.close();
          return;
        }
        this.hello = m;
        this._retryMs = 500;
        if (this.outage.since !== null) {
          this.outage.lastDurationMs = this.now() - this.outage.since;
          this.outage.since = null;
        }
        this.engineStatus = { camera: m.camera, tracker: m.tracker, session: m.session ?? null };
        this._setStatus('connected');
        this._emit('hello', m);
        this._emit('status', this.engineStatus);
        return;
      case 'status':
        this.engineStatus = { camera: m.camera, tracker: m.tracker, session: m.session ?? null };
        this._emit('status', this.engineStatus);
        if (m.startError) this._resolveStart({ ok: false, error: m.startError });
        else if (m.camera?.status === 'live') this._resolveStart({ ok: true, session: m.session ?? null });
        else if (m.camera?.status === 'error') this._resolveStart({ ok: false, error: m.camera.error });
        return;
      case 'result': {
        const err = resultError(m);
        if (err) {
          this.stats.malformed++;
          this.lastError = `malformed result dropped: ${err}`;
          return;
        }
        m.receivedAt = this.now();
        this.stats.results++;
        this._emit('result', m);
        return;
      }
      case 'metrics':
        this.metrics = m;
        this._emit('metrics', m);
        return;
      case 'error':
        this.lastError = String(m.message);
        this.bus?.emit('pipeline_error', { message: `engine: ${m.message}` });
        return;
      case 'pong':
        return;
      default:
        this.stats.malformed++;
    }
  }

  /** Ask the engine to open its camera. Resolves { ok, error }. */
  startCamera(timeoutMs = 15000) {
    if (this.status !== 'connected') {
      return Promise.resolve({ ok: false, error: { code: 'ENGINE_UNREACHABLE', message: `Python engine not connected (${this.lastError ?? this.status})` } });
    }
    this._resolveStart({ ok: false, error: { code: 'ABORTED', message: 'superseded' } });
    return new Promise((resolve) => {
      const timer = this._setTimeout(() => this._resolveStart({ ok: false, error: { code: 'TIMEOUT', message: 'Python engine did not open the camera in time' } }), timeoutMs);
      this._pendingStart = (r) => {
        this._clearTimeout(timer);
        resolve(r);
      };
      this.ws.send(JSON.stringify({ type: 'start' }));
    });
  }

  stopCamera() {
    if (this.status === 'connected') this.ws.send(JSON.stringify({ type: 'stop' }));
  }

  _resolveStart(r) {
    const p = this._pendingStart;
    this._pendingStart = null;
    p?.(r);
  }

  close() {
    this._closed = true;
    if (this._retryTimer) this._clearTimeout(this._retryTimer);
    const ws = this.ws;
    this.ws = null;
    try {
      ws?.close();
    } catch {
      /* ignore */
    }
    this._setStatus('disconnected', 'closed by client');
  }
}
