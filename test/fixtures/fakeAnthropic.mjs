/**
 * TEST FIXTURE - a local fake of the Claude Messages API (POST /v1/messages).
 *
 * It is NOT an AI model and is never used by the application at runtime. Tests
 * and the mocked UI integration run point the real SDK at it through
 * AURA_AGENT_BASE_URL to check request construction, response parsing, error
 * handling and the full agent workflow deterministically.
 *
 *   const fake = await startFakeAnthropic({ handler });   // handler(body, req) -> { status, body, headers, delayMs }
 *   fake.url, fake.requests, await fake.close()
 *
 * `mockModel` is a keyword-rule responder used for workflow tests (labelled MOCK).
 */
import http from 'node:http';

export function message(content, stopReason = 'end_turn', extra = {}) {
  return {
    id: `msg_mock_${Math.random().toString(36).slice(2, 10)}`,
    type: 'message',
    role: 'assistant',
    model: 'mock-model',
    content,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: { input_tokens: 10, output_tokens: 5 },
    ...extra,
  };
}

export const thinking = () => ({ type: 'thinking', thinking: '', signature: 'mock-signature' });
export const text = (t) => ({ type: 'text', text: t });
export const toolUse = (name, input, id = `toolu_mock_${Math.random().toString(36).slice(2, 10)}`) => ({ type: 'tool_use', id, name, input });
export const apiError = (status, type, msg, headers = {}) => ({ status, headers, body: { type: 'error', error: { type, message: msg } } });

function lastUser(body) {
  const msgs = body.messages ?? [];
  for (let i = msgs.length - 1; i >= 0; i--) if (msgs[i].role === 'user') return msgs[i];
  return null;
}

/** MOCK model: deterministic keyword rules (not AI). */
export function mockModel(body) {
  const u = lastUser(body);
  const blocks = Array.isArray(u?.content) ? u.content : [{ type: 'text', text: String(u?.content ?? '') }];
  const results = blocks.filter((b) => b.type === 'tool_result');
  const userText = blocks.filter((b) => b.type === 'text').map((b) => b.text).join(' ').toLowerCase();
  if (results.length && !userText) {
    const r = JSON.parse(results[0].content);
    if (r.result?.workflow) {
      const steps = r.result.steps.map((s) => `${s.id}:${s.status}`).join(', ');
      return { body: message([thinking(), text(`[mock] Workflow ${r.result.workflow} ${r.result.outcome}, verified=${r.result.verified} (${steps}).${r.error ? ` Reason: ${r.error.code}.` : ''}`)]) };
    }
    if (r.status === 'succeeded' && r.result?.blockers) {
      const b = r.result.blockers.map((x) => x.code).join(', ') || 'none';
      return { body: message([thinking(), text(`[mock] Diagnosis: robot ${r.result.robot.state}, camera ${r.result.camera.status}, hand ${r.result.handTracking.handDetected ?? 'unknown'}, blockers: ${b}.`)]) };
    }
    if (r.status === 'succeeded' && r.result?.robot) {
      return { body: message([thinking(), text(`[mock] The robot is ${r.result.robot.state} and the gesture lock is ${r.result.robot.gestureLock}.`)]) };
    }
    if (r.status === 'succeeded') return { body: message([thinking(), text(`[mock] Done: ${r.result.command} completed.`)]) };
    return { body: message([thinking(), text(`[mock] That did not happen (${r.error.code}): ${r.error.message}.`)]) };
  }
  const delayMs = userText.includes('slow') ? 6000 : 0;
  const call = (name, input) => ({ delayMs, body: message([thinking(), text('[mock] On it.'), toolUse(name, input)], 'tool_use') });
  if (userText.includes('shell')) return call('run_shell', { cmd: 'del /s *' });
  if (userText.includes('recover') || userText.includes('fix')) return call('run_workflow', { workflow: 'recover_robot' });
  if (userText.includes('make sure') && userText.includes('wave')) return call('run_workflow', { workflow: 'checked_command', command: 'WAVE' });
  if (userText.includes('bogus workflow')) return call('run_workflow', { workflow: 'format_disk' });
  if (userText.includes('diagnos') || userText.includes('why')) return call('app_diagnostics', {});
  if (userText.includes('dance')) return call('robot_command', { command: 'DANCE' });
  if (userText.includes('unlock') || userText.includes('lock')) return call('robot_command', { command: 'TOGGLE_LOCK' });
  if (userText.includes('spin left')) return call('robot_command', { command: 'SPIN_LEFT' });
  if (userText.includes('spin right')) return call('robot_command', { command: 'SPIN_RIGHT' });
  if (userText.includes('reset')) return call('robot_command', { command: 'RESET_POSE' });
  if (userText.includes('wave')) return call('robot_command', { command: 'WAVE' });
  if (userText.includes('status') || userText.includes('doing')) return call('app_status', {});
  return { delayMs, body: message([thinking(), text('[mock] I can make the robot wave, spin, reset its orientation, toggle the gesture lock, or report the app status.')]) };
}

export function startFakeAnthropic({ handler = mockModel, port = 0 } = {}) {
  const requests = [];
  const server = http.createServer((req, res) => {
    let raw = '';
    req.on('data', (c) => (raw += c));
    req.on('end', async () => {
      let body = null;
      try {
        body = JSON.parse(raw || '{}');
      } catch {
        /* leave null */
      }
      requests.push({ method: req.method, url: req.url, headers: req.headers, body });
      if (req.method !== 'POST' || !req.url.startsWith('/v1/messages')) {
        res.writeHead(404, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ type: 'error', error: { type: 'not_found_error', message: 'not found' } }));
        return;
      }
      const out = (await handler(body, req)) ?? {};
      if (out.hangUp) {
        req.socket.destroy();
        return;
      }
      if (out.delayMs) await new Promise((r) => setTimeout(r, out.delayMs));
      if (res.destroyed) return;
      res.writeHead(out.status ?? 200, { 'content-type': 'application/json', 'request-id': 'req_mock', ...(out.headers ?? {}) });
      res.end(typeof out.body === 'string' ? out.body : JSON.stringify(out.body));
    });
  });
  return new Promise((resolve) => {
    server.listen(port, '127.0.0.1', () => {
      const addr = server.address();
      resolve({
        url: `http://127.0.0.1:${addr.port}`,
        port: addr.port,
        requests,
        close: () => new Promise((r) => {
          server.closeAllConnections?.();
          server.close(() => r());
        }),
      });
    });
  });
}
