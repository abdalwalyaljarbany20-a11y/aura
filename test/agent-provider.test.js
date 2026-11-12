// MOCKED PROVIDER: the real Anthropic SDK talks to a local fake Messages API
// (test/fixtures/fakeAnthropic.mjs). No real AI inference happens here.
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { AnthropicProvider, ProviderError, buildRequest, normaliseResponse } from '../server/agent/anthropicProvider.js';
import { agentAvailability, loadAgentConfig } from '../server/agent/config.js';
import { providerToolDefinitions } from '../src/agent/agentTools.js';
import { apiError, message, startFakeAnthropic, text, thinking, toolUse } from './fixtures/fakeAnthropic.mjs';

const SECRET = 'sk-ant-test-SECRET-do-not-log-1234';
const PROMPT = 'TOPSECRET-PROMPT-CONTENT';
let fake;
let next = () => ({ body: message([text('ok')]) });
before(async () => {
  fake = await startFakeAnthropic({ handler: (body, req) => next(body, req) });
});
after(() => fake.close());

function capturingLogger() {
  const lines = [];
  return { lines, info: (l) => lines.push(l), warn: (l) => lines.push(l) };
}

function provider(over = {}, logger = capturingLogger()) {
  const cfg = loadAgentConfig({ ANTHROPIC_API_KEY: SECRET, AURA_AGENT_BASE_URL: fake.url, AURA_AGENT_TIMEOUT_MS: '2000', ...over });
  return { p: new AnthropicProvider(cfg, { logger }), logger, cfg };
}

const call = (p, extra = {}) => p.complete({
  system: 'sys', tools: providerToolDefinitions(), messages: [{ role: 'user', content: [{ type: 'text', text: PROMPT }] }], requestId: 'req12345678', ...extra,
});

test('configuration: defaults, validation and availability never expose the key', () => {
  const cfg = loadAgentConfig({ ANTHROPIC_API_KEY: SECRET });
  assert.equal(cfg.model, 'claude-opus-5-5');
  assert.equal(cfg.effort, 'low');
  assert.deepEqual(agentAvailability(cfg), { available: true, reason: null });
  assert.match(agentAvailability(loadAgentConfig({})).reason, /ANTHROPIC_API_KEY/);
  const bad = loadAgentConfig({ ANTHROPIC_API_KEY: 'k', AURA_AGENT_PROVIDER: 'other', AURA_AGENT_EFFORT: 'huge', AURA_AGENT_TIMEOUT_MS: '5', AURA_AGENT_BASE_URL: 'ftp://x' });
  assert.equal(bad.errors.length, 4);
  assert.equal(agentAvailability(bad).available, false);
  assert.ok(!JSON.stringify(agentAvailability(cfg)).includes(SECRET));
  assert.throws(() => new AnthropicProvider(loadAgentConfig({})), (e) => e instanceof ProviderError && e.code === 'NOT_CONFIGURED');
});

test('request construction: model, auto tool choice, strict tools, adaptive thinking, effort, betas; no fallback by default', async () => {
  next = () => ({ body: message([text('hello')]) });
  const { p } = provider();
  await call(p);
  const r = fake.requests.at(-1);
  assert.equal(r.url, '/v1/messages?beta=true');
  assert.equal(r.headers['x-api-key'], SECRET);
  assert.ok(r.headers['anthropic-version']);
  assert.deepEqual(r.headers['anthropic-beta'].split(','), ['thinking-binding-controls-2026-08-01']);
  const b = r.body;
  assert.equal(b.model, 'claude-opus-5-5');
  assert.equal(b.max_tokens, 16000);
  assert.deepEqual(b.tool_choice, { type: 'auto', disable_parallel_tool_use: true });
  assert.deepEqual(b.thinking, { type: 'adaptive', block_binding: { prefix_mismatch_behavior: 'drop_block' } });
  assert.deepEqual(b.output_config, { effort: 'low' });
  assert.equal(b.fallbacks, undefined, 'refusal fallback is opt-in (extra billed request)');
  assert.ok(b.tools.every((t) => t.input_schema.additionalProperties === false));
  for (const t of b.tools) assert.equal(t.strict === true, Object.keys(t.input_schema.properties).length > 0, t.name);
});

test('refusal fallback only when opted in, and only for models that support it', async () => {
  next = () => ({ body: message([text('hello')]) });
  const { p } = provider({ AURA_AGENT_REFUSAL_FALLBACK: 'default' });
  await call(p);
  const r = fake.requests.at(-1);
  assert.equal(r.body.fallbacks, 'default');
  assert.deepEqual(r.headers['anthropic-beta'].split(',').sort(), ['server-side-fallback-2026-07-01', 'thinking-binding-controls-2026-08-01']);
  const haiku = buildRequest(loadAgentConfig({ ANTHROPIC_API_KEY: 'k', AURA_AGENT_MODEL: 'claude-haiku-5-5', AURA_AGENT_REFUSAL_FALLBACK: 'default' }), { system: 's', tools: [], messages: [] });
  assert.equal(haiku.fallbacks, undefined);
  assert.ok(loadAgentConfig({ ANTHROPIC_API_KEY: 'k', AURA_AGENT_REFUSAL_FALLBACK: 'always' }).errors.length === 1);
  assert.equal(r.body.system, 'sys');
});

test('unknown / older models get a minimal request (no thinking, effort or fallbacks)', () => {
  const body = buildRequest(loadAgentConfig({ ANTHROPIC_API_KEY: 'k', AURA_AGENT_MODEL: 'claude-haiku-4-5' }), { system: 's', tools: [], messages: [] });
  assert.equal(body.thinking, undefined);
  assert.equal(body.output_config, undefined);
  assert.equal(body.fallbacks, undefined);
  assert.equal(body.betas, undefined);
  const haiku = buildRequest(loadAgentConfig({ ANTHROPIC_API_KEY: 'k', AURA_AGENT_MODEL: 'claude-haiku-5-5' }), { system: 's', tools: [], messages: [] });
  assert.ok(haiku.thinking && !haiku.fallbacks, 'Haiku 5.5: adaptive thinking, no server-side fallback');
});

test('parses text replies and tool calls; keeps raw content (thinking) for verbatim replay', async () => {
  next = () => ({ body: message([thinking(), text('Waving now.'), toolUse('robot_command', { command: 'WAVE' }, 'toolu_1')], 'tool_use') });
  const { p } = provider();
  const out = await call(p);
  assert.equal(out.stopReason, 'tool_use');
  assert.equal(out.text, 'Waving now.');
  assert.deepEqual(out.toolCalls, [{ id: 'toolu_1', name: 'robot_command', input: { command: 'WAVE' } }]);
  assert.equal(out.content[0].type, 'thinking');
  next = () => ({ body: message([text('x')], 'refusal', { stop_details: { type: 'refusal', category: 'cyber', explanation: null } }) });
  const ref = await call(p);
  assert.equal(ref.stopReason, 'refusal');
  assert.equal(ref.refusalCategory, 'cyber');
});

test('malformed provider responses are rejected', async () => {
  const { p } = provider();
  for (const body of [
    { not: 'a message' },
    message('oops'),
    { ...message([text('x')]), stop_reason: undefined },
    message([{ type: 'tool_use', id: 't', name: 'robot_command', input: 'WAVE' }], 'tool_use'),
    message([text('no tool')], 'tool_use'),
  ]) {
    next = () => ({ body });
    await assert.rejects(call(p), (e) => e instanceof ProviderError && e.code === 'MALFORMED_RESPONSE', JSON.stringify(body).slice(0, 80));
  }
  assert.throws(() => normaliseResponse(null), /unexpected response/);
  next = () => ({ body: message([text('x'.repeat(9000))]) });
  const long = await call(p);
  assert.equal(long.truncated, true);
  assert.ok(long.text.length <= 8001);
});

test('authentication, permission, model, rate-limit, overload and server errors map to safe codes', async () => {
  const { p } = provider();
  const cases = [
    [apiError(401, 'authentication_error', 'invalid x-api-key'), 'PROVIDER_AUTH'],
    [apiError(403, 'permission_error', 'nope'), 'PROVIDER_PERMISSION'],
    [apiError(404, 'not_found_error', 'model: x'), 'MODEL_NOT_FOUND'],
    [apiError(400, 'invalid_request_error', 'bad'), 'PROVIDER_BAD_REQUEST'],
    [apiError(402, 'billing_error', 'credit balance too low'), 'PROVIDER_BILLING'],
    [apiError(429, 'rate_limit_error', 'slow down', { 'retry-after': '7' }), 'RATE_LIMITED'],
    [apiError(529, 'overloaded_error', 'overloaded'), 'PROVIDER_OVERLOADED'],
    [apiError(500, 'api_error', 'boom'), 'PROVIDER_ERROR'],
  ];
  for (const [resp, code] of cases) {
    next = () => resp;
    const err = await call(p).catch((e) => e);
    assert.ok(err instanceof ProviderError, code);
    assert.equal(err.code, code);
    assert.ok(!err.message.includes(SECRET));
    if (code === 'RATE_LIMITED') {
      assert.equal(err.retryAfterS, 7);
      assert.equal(err.httpStatus, 429);
    }
  }
});

test('unsupported-parameter errors say which parameter, with key-like text redacted', async () => {
  const { p } = provider();
  next = () => apiError(400, 'invalid_request_error', `output_config.effort: not supported for this model (key ${SECRET})`);
  const err = await call(p).catch((e) => e);
  assert.equal(err.code, 'PROVIDER_BAD_REQUEST');
  assert.match(err.message, /output_config.effort: not supported/);
  assert.ok(!err.message.includes(SECRET) && err.message.includes('[redacted]'));
  assert.ok(err.message.length < 300);
});

test('timeouts, network failures and cancellation are bounded and typed', async () => {
  next = () => ({ body: message([text('late')]), delayMs: 5000 });
  const { p } = provider({ AURA_AGENT_TIMEOUT_MS: '1000' });
  const t0 = Date.now();
  const err = await call(p).catch((e) => e);
  assert.equal(err.code, 'TIMEOUT');
  assert.ok(Date.now() - t0 < 4500, 'one retry at most, still bounded');

  const dead = new AnthropicProvider(loadAgentConfig({ ANTHROPIC_API_KEY: SECRET, AURA_AGENT_BASE_URL: 'http://127.0.0.1:1', AURA_AGENT_TIMEOUT_MS: '1000' }));
  assert.equal((await call(dead).catch((e) => e)).code, 'NETWORK');

  next = () => ({ body: message([text('late')]), delayMs: 3000 });
  const ac = new AbortController();
  setTimeout(() => ac.abort(), 100);
  assert.equal((await call(p, { signal: ac.signal }).catch((e) => e)).code, 'CANCELLED');
});

test('logs contain metadata only: never the key, the prompt or the reply', async () => {
  const logger = capturingLogger();
  const { p } = provider({}, logger);
  next = () => ({ body: message([text('REPLY-TEXT-PRIVATE')]) });
  await call(p);
  next = () => apiError(401, 'authentication_error', 'bad key');
  await call(p).catch(() => {});
  const all = logger.lines.join('\n');
  assert.ok(logger.lines.length >= 2);
  for (const secret of [SECRET, PROMPT, 'REPLY-TEXT-PRIVATE']) assert.ok(!all.includes(secret), `log leaked ${secret}`);
  assert.match(all, /provider ok req=req12345678/);
  assert.match(all, /code=PROVIDER_AUTH/);
});
