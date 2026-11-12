#!/usr/bin/env node
/**
 * LIVE provider smoke test: ONE real request to the configured model.
 * It costs a small amount of API credit and needs ANTHROPIC_API_KEY (shell
 * environment, .env or .env.local). Nothing is executed in the app; this only
 * checks that the real provider answers and proposes an allowlisted tool.
 *
 *   npm run agent:smoke
 *
 * Prints metadata only (model, stop reason, proposed tool, latency) - never the key.
 */
import { loadEnv } from 'vite';
import { AnthropicProvider, ProviderError } from '../server/agent/anthropicProvider.js';
import { SYSTEM_PROMPT } from '../server/agent/agentService.js';
import { agentAvailability, loadAgentConfig } from '../server/agent/config.js';
import { providerToolDefinitions, validateToolCall } from '../src/agent/agentTools.js';

const env = { ...loadEnv('development', process.cwd(), ['ANTHROPIC_', 'AURA_AGENT_']), ...process.env };
const cfg = loadAgentConfig(env);
const avail = agentAvailability(cfg);
if (!avail.available) {
  console.error(`[smoke] LIVE test not run: ${avail.reason}`);
  process.exit(2);
}
const provider = new AnthropicProvider(cfg, { logger: console });
const t0 = performance.now();
try {
  const out = await provider.complete({
    system: SYSTEM_PROMPT,
    tools: providerToolDefinitions(),
    messages: [{ role: 'user', content: [{ type: 'text', text: 'Please make the robot wave.' }] }],
    requestId: 'smoke-live-0001',
  });
  const call = out.toolCalls[0];
  const v = call ? validateToolCall(call.name, call.input) : null;
  console.log(JSON.stringify({
    live: true,
    model: out.model ?? cfg.model,
    stopReason: out.stopReason,
    proposedTool: call ? { name: call.name, input: call.input, allowlisted: v.ok } : null,
    replyChars: out.text.length,
    latencyMs: Math.round(performance.now() - t0),
    usage: out.usage,
  }, null, 2));
  process.exit(call && v.ok ? 0 : 1);
} catch (err) {
  const e = err instanceof ProviderError ? err : { code: 'INTERNAL', message: 'unexpected error' };
  console.error(`[smoke] LIVE request failed: ${e.code}: ${e.message}`);
  process.exit(1);
}
