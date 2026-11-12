// Server-side agent service with a SCRIPTED provider (no AI inference).
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { AgentError, AgentService, LIMITS } from '../server/agent/agentService.js';
import { ProviderError, normaliseResponse } from '../server/agent/anthropicProvider.js';
import { message, text, thinking, toolUse } from './fixtures/fakeAnthropic.mjs';

/** Provider that returns scripted responses and records what it was sent. */
function scripted(steps) {
  const calls = [];
  return {
    calls,
    async complete(req) {
      calls.push(structuredClone({ messages: req.messages, system: req.system, tools: req.tools }));
      const s = steps.shift();
      if (!s) throw new Error('no scripted step left');
      if (s instanceof Error) throw s;
      if (s.delayMs) await new Promise((r, j) => {
        const t = setTimeout(r, s.delayMs);
        req.signal?.addEventListener('abort', () => {
          clearTimeout(t);
          j(new ProviderError('CANCELLED', 'request cancelled', { httpStatus: 499 }));
        });
      });
      return normaliseResponse(s.body ?? s);
    },
  };
}
let n = 0;
const rid = () => `req${String(++n).padStart(8, '0')}`;

function service(steps) {
  const provider = scripted(steps);
  const svc = new AgentService({ provider });
  const { conversationId } = svc.createConversation();
  return { svc, provider, conv: conversationId };
}

test('natural-language request -> validated tool call -> real outcome -> explanation', async () => {
  const { svc, provider, conv } = service([
    message([thinking(), text('Sure.'), toolUse('robot_command', { command: 'WAVE' }, 'tu1')], 'tool_use'),
    message([thinking(), text('Done, it waved.')]),
  ]);
  const step = await svc.userMessage(conv, { requestId: rid(), text: 'please wave' });
  assert.equal(step.kind, 'tool_call');
  assert.deepEqual([step.tool, step.args, step.taskType, step.policy], ['robot_command', { command: 'WAVE' }, 'robot.command', 'auto']);
  const reply = await svc.toolResult(conv, { requestId: rid(), toolCallId: 'tu1', outcome: { status: 'succeeded', result: { command: 'WAVE' } } });
  assert.equal(reply.kind, 'reply');
  assert.equal(reply.text, 'Done, it waved.');
  // append-only history: assistant content (with thinking) replayed verbatim, tool_result attached to the right id
  const sent = provider.calls[1].messages;
  assert.deepEqual(sent[1].content[0], thinking());
  assert.equal(sent[2].content[0].type, 'tool_result');
  assert.equal(sent[2].content[0].tool_use_id, 'tu1');
  assert.deepEqual(provider.calls[0].system, provider.calls[1].system);
  assert.deepEqual(provider.calls[0].tools, provider.calls[1].tools);
});

test('unsupported tools and invalid arguments never leave the server; the model is told and explains', async () => {
  const { svc, provider, conv } = service([
    message([toolUse('run_shell', { cmd: 'del /s *' }, 'bad1')], 'tool_use'),
    message([toolUse('robot_command', { command: 'DANCE' }, 'bad2')], 'tool_use'),
    message([text('I cannot run commands or dance; I can wave or spin the robot.')]),
  ]);
  const step = await svc.userMessage(conv, { requestId: rid(), text: 'run a shell command and dance' });
  assert.equal(step.kind, 'reply');
  assert.deepEqual(step.rejectedToolCalls.map((r) => [r.tool, r.code]), [['run_shell', 'UNSUPPORTED_TOOL'], ['robot_command', 'INVALID_ARGS']]);
  const errResult = provider.calls[1].messages.at(-1).content[0];
  assert.equal(errResult.is_error, true);
  assert.match(errResult.content, /UNSUPPORTED_TOOL/);
});

test('repeated invalid proposals stop at the loop bound without executing anything', async () => {
  const steps = Array.from({ length: LIMITS.maxModelCallsPerRequest }, (_, i) => message([toolUse('run_shell', {}, `x${i}`)], 'tool_use'));
  const { svc, conv } = service(steps);
  const step = await svc.userMessage(conv, { requestId: rid(), text: 'keep trying' });
  assert.equal(step.kind, 'reply');
  assert.equal(step.stopped, true);
  assert.equal(step.rejectedToolCalls.length, LIMITS.maxModelCallsPerRequest);
});

test('a second parallel tool call is answered as an error, not executed', async () => {
  const { svc, provider, conv } = service([
    message([toolUse('robot_command', { command: 'WAVE' }, 'a'), toolUse('robot_command', { command: 'SPIN_LEFT' }, 'b')], 'tool_use'),
    message([text('ok')]),
  ]);
  const step = await svc.userMessage(conv, { requestId: rid(), text: 'wave and spin' });
  assert.equal(step.toolCallId, 'a');
  await svc.toolResult(conv, { requestId: rid(), toolCallId: 'a', outcome: { status: 'succeeded', result: {} } });
  const results = provider.calls[1].messages.at(-1).content;
  assert.deepEqual(results.map((r) => [r.tool_use_id, !!r.is_error]), [['a', false], ['b', true]]);
});

test('duplicate requestId is answered from cache without a second model call; concurrent requests are BUSY', async () => {
  const { svc, provider, conv } = service([message([text('hi')]), { body: message([text('slow')]), delayMs: 200 }]);
  const id = rid();
  const a = await svc.userMessage(conv, { requestId: id, text: 'hello' });
  const b = await svc.userMessage(conv, { requestId: id, text: 'hello' });
  assert.deepEqual(a, b);
  assert.equal(provider.calls.length, 1);
  const slow = svc.userMessage(conv, { requestId: rid(), text: 'again' });
  await assert.rejects(svc.userMessage(conv, { requestId: rid(), text: 'third' }), (e) => e.code === 'BUSY');
  await slow;
});

test('stale or mismatched tool results are rejected; outcomes are validated', async () => {
  const { svc, conv } = service([message([toolUse('robot_command', { command: 'WAVE' }, 'tu1')], 'tool_use'), message([text('ok')])]);
  await svc.userMessage(conv, { requestId: rid(), text: 'wave' });
  await assert.rejects(svc.toolResult(conv, { requestId: rid(), toolCallId: 'other', outcome: { status: 'succeeded', result: {} } }), (e) => e.code === 'STALE_TOOL_RESULT');
  await assert.rejects(svc.toolResult(conv, { requestId: rid(), toolCallId: 'tu1', outcome: { status: 'succeeded' } }), (e) => e.code === 'INVALID_OUTCOME');
  await svc.toolResult(conv, { requestId: rid(), toolCallId: 'tu1', outcome: { status: 'succeeded', result: {} } });
  await assert.rejects(svc.toolResult(conv, { requestId: rid(), toolCallId: 'tu1', outcome: { status: 'succeeded', result: {} } }), (e) => e.code === 'STALE_TOOL_RESULT');
});

test('a new message supersedes an unconfirmed proposal: recorded as not executed', async () => {
  const { svc, provider, conv } = service([message([toolUse('robot_command', { command: 'TOGGLE_LOCK' }, 'lock1')], 'tool_use'), message([text('ok, never mind')])]);
  const step = await svc.userMessage(conv, { requestId: rid(), text: 'lock gestures' });
  assert.equal(step.policy, 'confirm');
  await svc.userMessage(conv, { requestId: rid(), text: 'actually no' });
  const blocks = provider.calls[1].messages.at(-1).content;
  assert.equal(blocks[0].type, 'tool_result');
  assert.equal(blocks[0].is_error, true);
  assert.match(blocks[0].content, /SUPERSEDED/);
  assert.equal(blocks[1].text, 'actually no');
});

test('provider failure rolls the conversation back; the next request works', async () => {
  const { svc, provider, conv } = service([new ProviderError('RATE_LIMITED', 'slow down', { httpStatus: 429, retryAfterS: 3 }), message([text('fine now')])]);
  await assert.rejects(svc.userMessage(conv, { requestId: rid(), text: 'first' }), (e) => e.code === 'RATE_LIMITED' && e.retryAfterS === 3);
  const ok = await svc.userMessage(conv, { requestId: rid(), text: 'second' });
  assert.equal(ok.text, 'fine now');
  assert.equal(provider.calls[1].messages.length, 1, 'the failed message was rolled back');
  assert.equal(provider.calls[1].messages[0].content[0].text, 'second');
});

test('input limits, unknown conversations, not-configured service', async () => {
  const { svc, conv } = service([]);
  await assert.rejects(svc.userMessage(conv, { requestId: rid(), text: '   ' }), (e) => e.code === 'INVALID_REQUEST');
  await assert.rejects(svc.userMessage(conv, { requestId: rid(), text: 'x'.repeat(2001) }), (e) => e.code === 'MESSAGE_TOO_LONG');
  await assert.rejects(svc.userMessage(conv, { requestId: 'short', text: 'hi' }), (e) => e.code === 'INVALID_REQUEST');
  await assert.rejects(svc.userMessage('nope-nope-nope', { requestId: rid(), text: 'hi' }), (e) => e.code === 'UNKNOWN_CONVERSATION');
  const off = new AgentService({ provider: null, unavailableReason: 'no key' });
  const c = off.createConversation().conversationId;
  await assert.rejects(off.userMessage(c, { requestId: rid(), text: 'hi' }), (e) => e instanceof AgentError && e.code === 'NOT_CONFIGURED' && e.httpStatus === 503);
});

test('history trimming cuts at a user turn and strips thinking from what is kept', async () => {
  const replies = Array.from({ length: 40 }, () => message([thinking(), text('ok')]));
  const { svc, provider, conv } = service(replies);
  for (let i = 0; i < 35; i++) await svc.userMessage(conv, { requestId: rid(), text: `msg ${i}` });
  const lengths = provider.calls.map((c) => c.messages.length);
  const trimAt = lengths.findIndex((len, i) => i > 0 && len < lengths[i - 1]);
  assert.ok(trimAt > 0, 'a trim happened');
  const afterTrim = provider.calls[trimAt].messages;
  assert.ok(Math.max(...lengths) <= LIMITS.maxMessages + 1);
  assert.equal(afterTrim[0].role, 'user');
  assert.ok(afterTrim[0].content.every((b) => b.type === 'text'), 'cut at a user text turn, never inside a tool pair');
  // Every assistant turn retained across the trim had its thinking removed (their
  // prefix changed); blocks created after the trim are valid and are kept.
  assert.ok(afterTrim.filter((m) => m.role === 'assistant').every((m) => m.content.every((b) => b.type !== 'thinking')));
  assert.ok(provider.calls.at(-1).messages.some((m) => m.role === 'assistant' && m.content.some((b) => b.type === 'thinking')), 'post-trim thinking replayed');
  assert.equal(svc.conversations.get(conv).trimmed, true);
});

test('global concurrency is bounded', async () => {
  const provider = scripted([{ body: message([text('a')]), delayMs: 150 }, { body: message([text('b')]), delayMs: 150 }]);
  const svc = new AgentService({ provider });
  const ids = [0, 1, 2].map(() => svc.createConversation().conversationId);
  const p1 = svc.userMessage(ids[0], { requestId: rid(), text: 'a' });
  const p2 = svc.userMessage(ids[1], { requestId: rid(), text: 'b' });
  await assert.rejects(svc.userMessage(ids[2], { requestId: rid(), text: 'c' }), (e) => e.code === 'SERVER_BUSY');
  await Promise.all([p1, p2]);
});
