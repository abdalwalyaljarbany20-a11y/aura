import { AgentError, AgentService } from './agentService.js';
import { AnthropicProvider, ProviderError } from './anthropicProvider.js';
import { agentAvailability, loadAgentConfig } from './config.js';

/**
 * HTTP surface of the agent (Connect-style middleware, mounted in the Vite dev
 * and preview servers - the same local server that serves the page in both
 * vision-engine modes).
 *
 *   GET    /api/agent/status                          availability, provider, model (never the key)
 *   POST   /api/agent/conversations                   -> { conversationId }
 *   POST   /api/agent/conversations/:id/messages      { requestId, text }                 -> step
 *   POST   /api/agent/conversations/:id/tool-results  { requestId, toolCallId, outcome }  -> step
 *   DELETE /api/agent/conversations/:id
 *
 * Request hygiene (the endpoint spends the user's provider credit, so other
 * websites must not be able to drive it):
 *   - Host must be localhost / 127.0.0.1 / [::1] (blocks DNS rebinding)
 *   - a present Origin must equal this server's own origin
 *   - mutating requests need `X-Aura-Agent: 1` and `Content-Type: application/json`
 *     (both force a CORS preflight, which this server never approves)
 *   - bodies are limited to 16 KB
 */
const PREFIX = '/api/agent';
const MAX_BODY = 16 * 1024;
const LOCAL_HOST = /^(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/i;

function send(res, status, body) {
  const json = JSON.stringify(body);
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.end(json);
}

function sendError(res, err) {
  if (err instanceof AgentError || err instanceof ProviderError) {
    const status = err.httpStatus ?? 500;
    return send(res, status, { error: { code: err.code, message: err.message, ...(err.retryAfterS ? { retryAfterS: err.retryAfterS } : {}) } });
  }
  return send(res, 500, { error: { code: 'INTERNAL', message: 'internal agent error' } });
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on('data', (c) => {
      size += c.length;
      if (size > MAX_BODY) {
        reject(new AgentError('PAYLOAD_TOO_LARGE', 'request body too large', 413));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on('end', () => {
      try {
        const v = JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}');
        if (!v || typeof v !== 'object' || Array.isArray(v)) throw new Error('not an object');
        resolve(v);
      } catch {
        reject(new AgentError('INVALID_JSON', 'request body must be a JSON object', 400));
      }
    });
    req.on('error', () => reject(new AgentError('INVALID_REQUEST', 'could not read request', 400)));
  });
}

/** @returns {string|null} why the request is not allowed */
export function requestGuardError(req) {
  const host = req.headers.host ?? '';
  if (!LOCAL_HOST.test(host)) return 'the agent API only answers on localhost';
  const origin = req.headers.origin;
  if (origin !== undefined && origin !== `http://${host}` && origin !== `https://${host}`) return 'cross-origin requests are not allowed';
  if (req.method !== 'GET') {
    if (req.headers['x-aura-agent'] !== '1') return 'missing X-Aura-Agent header';
    if (!/^application\/json\b/i.test(req.headers['content-type'] ?? '')) return 'Content-Type must be application/json';
  }
  return null;
}

/**
 * @param {{ env?: Record<string, string|undefined>, service?: AgentService, logger?: Console }} [opts]
 */
export function createAgentMiddleware(opts = {}) {
  const log = opts.logger ?? console;
  const cfg = loadAgentConfig(opts.env ?? process.env);
  const availability = agentAvailability(cfg);
  // Outcome of the most recent provider call (no content): lets diagnostics tell
  // "configured" apart from "actually reachable".
  const health = { lastCall: null };
  let service = opts.service;
  if (!service) {
    let provider = null;
    if (availability.available) provider = trackProvider(new AnthropicProvider(cfg, { logger: log }), health);
    service = new AgentService({ provider, unavailableReason: availability.reason, logger: log });
  }
  log.info?.(`[agent] ${availability.available ? `ready: ${cfg.provider} ${cfg.model} (effort ${cfg.effort})` : `unavailable: ${availability.reason}`}`);

  return async function agentMiddleware(req, res, next) {
    const url = new URL(req.url ?? '/', 'http://localhost');
    if (!url.pathname.startsWith(PREFIX)) return next?.();
    const guard = requestGuardError(req);
    if (guard) return send(res, 403, { error: { code: 'FORBIDDEN', message: guard } });

    const path = url.pathname.slice(PREFIX.length);
    const abort = new AbortController();
    res.on('close', () => {
      if (!res.writableFinished) abort.abort(); // client went away: cancel the provider call
    });
    const t0 = performance.now();
    try {
      if (req.method === 'GET' && path === '/status') {
        return send(res, 200, {
          ...availability, provider: cfg.provider, model: cfg.model, effort: cfg.effort,
          refusalFallback: cfg.refusalFallback, lastCall: health.lastCall,
        });
      }
      if (req.method === 'POST' && path === '/conversations') {
        await readJson(req);
        return send(res, 201, service.createConversation());
      }
      const m = /^\/conversations\/([A-Za-z0-9-]{8,64})(\/messages|\/tool-results)?$/.exec(path);
      if (m && req.method === 'DELETE' && !m[2]) return send(res, 200, service.deleteConversation(m[1]));
      if (m && req.method === 'POST' && m[2]) {
        const body = await readJson(req);
        const step = m[2] === '/messages'
          ? await service.userMessage(m[1], body, abort.signal)
          : await service.toolResult(m[1], body, abort.signal);
        log.info?.(`[agent] ${m[2].slice(1)} req=${String(body.requestId).slice(0, 12)} -> ${step.kind} ms=${Math.round(performance.now() - t0)}`);
        return send(res, 200, step);
      }
      return send(res, 404, { error: { code: 'NOT_FOUND', message: 'unknown agent endpoint' } });
    } catch (err) {
      if (!(err instanceof AgentError) && !(err instanceof ProviderError)) log.warn?.(`[agent] unexpected ${err?.name ?? 'error'}`);
      if (!res.writableEnded) sendError(res, err);
    }
  };
}

/** Wrap a provider so each call's outcome (not its content) is recorded. */
export function trackProvider(provider, health, now = () => Date.now()) {
  return {
    async complete(req) {
      const t0 = now();
      try {
        const out = await provider.complete(req);
        health.lastCall = { ok: true, code: null, at: new Date(now()).toISOString(), ms: now() - t0 };
        return out;
      } catch (err) {
        if (err?.code !== 'CANCELLED') health.lastCall = { ok: false, code: err?.code ?? 'INTERNAL', at: new Date(now()).toISOString(), ms: now() - t0 };
        throw err;
      }
    },
  };
}

/** Vite plugin: mounts the agent API in `vite` (dev) and `vite preview`. */
export function auraAgentPlugin(env) {
  let middleware = null;
  const get = () => (middleware ??= createAgentMiddleware({ env }));
  return {
    name: 'aura-agent-api',
    configureServer(server) {
      server.middlewares.use((req, res, next) => get()(req, res, next));
    },
    configurePreviewServer(server) {
      server.middlewares.use((req, res, next) => get()(req, res, next));
    },
  };
}
