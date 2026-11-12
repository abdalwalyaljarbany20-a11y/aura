import Anthropic from '@anthropic-ai/sdk';

/**
 * Claude Messages API adapter (official SDK). The only place that talks to the
 * provider; everything else sees the normalised result below, so another
 * provider can be added by implementing the same `complete()` contract.
 *
 * complete({ system, tools, messages, signal }) ->
 *   { stopReason: 'end_turn'|'tool_use'|'max_tokens'|'refusal'|..., text, toolCalls: [{ id, name, input }],
 *     content (raw assistant content, replayed verbatim), refusalCategory, usage }
 * throws ProviderError { code, httpStatus, retryAfterS, message } - messages are safe to show.
 */

export class ProviderError extends Error {
  constructor(code, message, { httpStatus = 502, retryAfterS = null } = {}) {
    super(message);
    this.code = code;
    this.httpStatus = httpStatus;
    this.retryAfterS = retryAfterS;
  }
}

// Request features per model family (see the Claude API reference):
// adaptive thinking is always on for these; effort is the depth control;
// forced tool_choice is not used (auto + strict tools instead).
const MODERN = /^claude-(opus-5-5|opus-5|sonnet-5-5|haiku-5-5|fable-5-1|fable-5)$/;
// Server-side refusal fallback ("default" routing): opt-in (AURA_AGENT_REFUSAL_FALLBACK=default),
// because a fallback run is an additional billed request. Not offered for Haiku 5.5.
const FALLBACKS = /^claude-(opus-5-5|opus-5|sonnet-5-5|fable-5-1)$/;

const MAX_TEXT_CHARS = 8000;

/** Build the request body (exported for tests). */
export function buildRequest(cfg, { system, tools, messages }) {
  const req = {
    model: cfg.model,
    max_tokens: cfg.maxTokens,
    system,
    tools,
    tool_choice: { type: 'auto', disable_parallel_tool_use: true },
    messages,
  };
  const betas = [];
  if (MODERN.test(cfg.model)) {
    req.output_config = { effort: cfg.effort };
    // History is append-only (see conversationStore); if it was ever trimmed,
    // drop stale thinking blocks instead of failing the request.
    req.thinking = { type: 'adaptive', block_binding: { prefix_mismatch_behavior: 'drop_block' } };
    betas.push('thinking-binding-controls-2026-08-01');
  }
  if (cfg.refusalFallback === 'default' && FALLBACKS.test(cfg.model)) {
    req.fallbacks = 'default';
    betas.push('server-side-fallback-2026-07-01');
  }
  if (betas.length) req.betas = betas;
  return req;
}

function retryAfter(headers) {
  const v = headers?.get?.('retry-after');
  const n = v === null || v === undefined ? NaN : Number(v);
  return Number.isFinite(n) && n >= 0 ? Math.min(n, 3600) : null;
}

/** Provider error text for display: key-like strings redacted, length bounded. */
function safeDetail(err) {
  const raw = String(err?.error?.error?.message ?? err?.message ?? 'invalid request');
  return raw.replace(/sk-ant-[A-Za-z0-9_-]+/g, '[redacted]').replace(/\s+/g, ' ').slice(0, 200);
}

/** Map SDK errors to safe, typed provider errors (most specific first). */
export function mapProviderError(err) {
  if (err instanceof ProviderError) return err;
  if (err instanceof Anthropic.APIUserAbortError) return new ProviderError('CANCELLED', 'request cancelled', { httpStatus: 499 });
  if (err instanceof Anthropic.APIConnectionTimeoutError) return new ProviderError('TIMEOUT', 'the AI provider did not answer in time', { httpStatus: 504 });
  if (err instanceof Anthropic.APIConnectionError) return new ProviderError('NETWORK', 'could not reach the AI provider (network error)', { httpStatus: 502 });
  if (err instanceof Anthropic.AuthenticationError) return new ProviderError('PROVIDER_AUTH', 'the AI provider rejected the credential (check ANTHROPIC_API_KEY)', { httpStatus: 502 });
  if (err instanceof Anthropic.PermissionDeniedError) return new ProviderError('PROVIDER_PERMISSION', 'the credential is not allowed to use this model', { httpStatus: 502 });
  if (err instanceof Anthropic.NotFoundError) return new ProviderError('MODEL_NOT_FOUND', 'the configured model was not found (check AURA_AGENT_MODEL)', { httpStatus: 502 });
  if (err instanceof Anthropic.RateLimitError) {
    return new ProviderError('RATE_LIMITED', 'the AI provider is rate-limiting requests; try again shortly', { httpStatus: 429, retryAfterS: retryAfter(err.headers) });
  }
  if (err instanceof Anthropic.BadRequestError) {
    // e.g. a parameter the configured model does not support - say which (redacted, bounded)
    return new ProviderError('PROVIDER_BAD_REQUEST', `the AI provider rejected the request: ${safeDetail(err)}`, { httpStatus: 502 });
  }
  if (err instanceof Anthropic.InternalServerError) {
    const code = err.status === 529 ? 'PROVIDER_OVERLOADED' : 'PROVIDER_ERROR';
    return new ProviderError(code, code === 'PROVIDER_OVERLOADED' ? 'the AI provider is overloaded; try again shortly' : 'the AI provider returned a server error', { httpStatus: 502, retryAfterS: retryAfter(err.headers) });
  }
  if (err instanceof Anthropic.APIError && err.status === 402) {
    return new ProviderError('PROVIDER_BILLING', 'the provider account cannot make requests (billing or credit); check the account in the Claude Console', { httpStatus: 502 });
  }
  if (err instanceof Anthropic.APIError) return new ProviderError('PROVIDER_ERROR', `the AI provider returned an error (${err.status ?? 'unknown'})`, { httpStatus: 502 });
  return new ProviderError('INTERNAL', 'unexpected error while calling the AI provider', { httpStatus: 500 });
}

/** Validate and normalise a Messages API response. Throws MALFORMED_RESPONSE. */
export function normaliseResponse(resp) {
  const bad = (why) => new ProviderError('MALFORMED_RESPONSE', `the AI provider returned an unexpected response (${why})`);
  if (!resp || typeof resp !== 'object') throw bad('not an object');
  if (!Array.isArray(resp.content)) throw bad('content missing');
  if (typeof resp.stop_reason !== 'string') throw bad('stop_reason missing');
  const texts = [];
  const toolCalls = [];
  for (const b of resp.content) {
    if (!b || typeof b.type !== 'string') throw bad('content block without type');
    if (b.type === 'text') {
      if (typeof b.text !== 'string') throw bad('text block without text');
      texts.push(b.text);
    } else if (b.type === 'tool_use') {
      if (typeof b.id !== 'string' || typeof b.name !== 'string' || !b.input || typeof b.input !== 'object' || Array.isArray(b.input)) {
        throw bad('malformed tool_use block');
      }
      toolCalls.push({ id: b.id, name: b.name, input: b.input });
    }
    // thinking / redacted_thinking / fallback blocks: kept in `content` for replay, not shown
  }
  if (resp.stop_reason === 'tool_use' && !toolCalls.length) throw bad('tool_use stop without a tool call');
  let text = texts.join('\n').trim();
  let truncated = false;
  if (text.length > MAX_TEXT_CHARS) {
    text = `${text.slice(0, MAX_TEXT_CHARS)}…`;
    truncated = true;
  }
  return {
    stopReason: resp.stop_reason,
    text,
    truncated,
    toolCalls,
    content: resp.content,
    refusalCategory: resp.stop_reason === 'refusal' ? resp.stop_details?.category ?? null : null,
    model: typeof resp.model === 'string' ? resp.model : null,
    usage: resp.usage ? { input: resp.usage.input_tokens ?? null, output: resp.usage.output_tokens ?? null } : null,
  };
}

export class AnthropicProvider {
  /**
   * @param {ReturnType<import('./config.js').loadAgentConfig>} cfg
   * @param {{ ClientClass?: any, logger?: { info: Function, warn: Function } }} [deps]
   */
  constructor(cfg, deps = {}) {
    if (!cfg.apiKey) throw new ProviderError('NOT_CONFIGURED', 'no provider credential configured', { httpStatus: 503 });
    this.cfg = cfg;
    const ClientClass = deps.ClientClass ?? Anthropic;
    this.client = new ClientClass({
      apiKey: cfg.apiKey,
      ...(cfg.baseURL ? { baseURL: cfg.baseURL } : {}),
      timeout: cfg.timeoutMs,
      maxRetries: 1, // model calls have no side effects; keep wall-clock bounded
      logLevel: 'off', // the SDK must not log request details
    });
    this.log = deps.logger ?? { info() {}, warn() {} };
  }

  async complete({ system, tools, messages, signal, requestId }) {
    const body = buildRequest(this.cfg, { system, tools, messages });
    const t0 = performance.now();
    try {
      const resp = await this.client.beta.messages.create(body, { signal, timeout: this.cfg.timeoutMs });
      const out = normaliseResponse(resp);
      // Metadata only: never the key, the prompt or the reply text.
      this.log.info(`[agent] provider ok req=${requestId ?? '-'} stop=${out.stopReason} ms=${Math.round(performance.now() - t0)}` +
        ` in=${out.usage?.input ?? '?'} out=${out.usage?.output ?? '?'}`);
      return out;
    } catch (err) {
      const e = mapProviderError(err);
      this.log.warn(`[agent] provider error req=${requestId ?? '-'} code=${e.code} status=${err?.status ?? '-'} ms=${Math.round(performance.now() - t0)}`);
      throw e;
    }
  }
}
