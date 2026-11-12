/**
 * Agent server configuration, from environment variables only (server side).
 * Vite exposes only VITE_-prefixed variables to browser code, and none of these
 * use that prefix, so none of them can reach the page.
 *
 *   AURA_AGENT_PROVIDER     anthropic (default; the only implemented provider)
 *   ANTHROPIC_API_KEY       provider credential (required)
 *   AURA_AGENT_MODEL        model id (default claude-opus-5-5)
 *   AURA_AGENT_BASE_URL     optional API base URL (e.g. a gateway); default: SDK default
 *   AURA_AGENT_EFFORT       low | medium | high | xhigh | max (default low: short tool-selection turns)
 *   AURA_AGENT_TIMEOUT_MS   per provider request (default 30000, 1000-120000)
 *   AURA_AGENT_MAX_TOKENS   per response (default 16000, 256-64000)
 *   AURA_AGENT_REFUSAL_FALLBACK  off (default) | default
 *                           "default" opts into Anthropic's server-side refusal fallback: when
 *                           a safety classifier declines a request, the API re-runs it on a
 *                           recommended fallback model inside the same call. That second run is
 *                           billed at the fallback model's rates (and some declines are billed
 *                           too). It only happens on a refusal; it is off unless you enable it.
 */
export const DEFAULT_MODEL = 'claude-opus-5-5';
const EFFORTS = ['low', 'medium', 'high', 'xhigh', 'max'];

function intIn(raw, def, lo, hi, name, errors) {
  if (raw === undefined || raw === '') return def;
  const n = Number(raw);
  if (!Number.isInteger(n) || n < lo || n > hi) {
    errors.push(`${name} must be an integer between ${lo} and ${hi}`);
    return def;
  }
  return n;
}

/**
 * @param {Record<string, string|undefined>} env
 * @returns {{ provider: string, apiKey: string|null, model: string, baseURL: string|null, effort: string,
 *             timeoutMs: number, maxTokens: number, errors: string[] }}
 */
export function loadAgentConfig(env) {
  const errors = [];
  const provider = (env.AURA_AGENT_PROVIDER || 'anthropic').trim();
  if (provider !== 'anthropic') errors.push(`AURA_AGENT_PROVIDER "${provider}" is not supported (supported: anthropic)`);
  const apiKey = (env.ANTHROPIC_API_KEY || '').trim() || null;
  const model = (env.AURA_AGENT_MODEL || DEFAULT_MODEL).trim();
  if (!/^[a-z0-9][a-z0-9.\-]{2,80}$/i.test(model)) errors.push('AURA_AGENT_MODEL is not a valid model id');
  let baseURL = (env.AURA_AGENT_BASE_URL || '').trim() || null;
  if (baseURL && !/^https?:\/\/[^\s]+$/.test(baseURL)) {
    errors.push('AURA_AGENT_BASE_URL must be an http(s) URL');
    baseURL = null;
  }
  let effort = (env.AURA_AGENT_EFFORT || 'low').trim();
  if (!EFFORTS.includes(effort)) {
    errors.push(`AURA_AGENT_EFFORT must be one of ${EFFORTS.join(', ')}`);
    effort = 'low';
  }
  return {
    provider,
    apiKey,
    model,
    baseURL,
    effort,
    timeoutMs: intIn(env.AURA_AGENT_TIMEOUT_MS, 30000, 1000, 120000, 'AURA_AGENT_TIMEOUT_MS', errors),
    maxTokens: intIn(env.AURA_AGENT_MAX_TOKENS, 16000, 256, 64000, 'AURA_AGENT_MAX_TOKENS', errors),
    refusalFallback: (() => {
      const v = (env.AURA_AGENT_REFUSAL_FALLBACK || 'off').trim();
      if (v === 'off' || v === 'default') return v;
      errors.push('AURA_AGENT_REFUSAL_FALLBACK must be "off" or "default"');
      return 'off';
    })(),
    errors,
  };
}

/** Public availability info (never includes the key). */
export function agentAvailability(cfg) {
  if (cfg.errors.length) return { available: false, reason: `agent configuration invalid: ${cfg.errors.join('; ')}` };
  if (!cfg.apiKey) return { available: false, reason: 'no provider credential: set ANTHROPIC_API_KEY in .env.local (see docs/AGENT_SETUP.md)' };
  return { available: true, reason: null };
}
