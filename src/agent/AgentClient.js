/**
 * Browser client for the local agent API (/api/agent, served by the same local
 * server as the page). Holds no credentials: the provider key exists only on
 * the server. Every call has a timeout and can be cancelled.
 */
export class AgentClientError extends Error {
  constructor(code, message, { status = null, retryAfterS = null } = {}) {
    super(message);
    this.code = code;
    this.status = status;
    this.retryAfterS = retryAfterS;
  }
}

export class AgentClient {
  /** @param {{ fetchImpl?: typeof fetch, base?: string, timeoutMs?: number }} [opts] */
  constructor(opts = {}) {
    this.fetch = opts.fetchImpl ?? globalThis.fetch.bind(globalThis);
    this.base = opts.base ?? '/api/agent';
    this.timeoutMs = opts.timeoutMs ?? 100000;
  }

  status(signal) {
    return this._req('GET', '/status', undefined, signal);
  }

  createConversation(signal) {
    return this._req('POST', '/conversations', {}, signal);
  }

  sendMessage(conversationId, body, signal) {
    return this._req('POST', `/conversations/${encodeURIComponent(conversationId)}/messages`, body, signal);
  }

  sendToolResult(conversationId, body, signal) {
    return this._req('POST', `/conversations/${encodeURIComponent(conversationId)}/tool-results`, body, signal);
  }

  deleteConversation(conversationId) {
    return this._req('DELETE', `/conversations/${encodeURIComponent(conversationId)}`, {}, undefined).catch(() => null);
  }

  async _req(method, path, body, signal) {
    const timeout = AbortSignal.timeout(this.timeoutMs);
    const combined = signal ? AbortSignal.any([signal, timeout]) : timeout;
    let res;
    try {
      res = await this.fetch(`${this.base}${path}`, {
        method,
        headers: method === 'GET' ? {} : { 'Content-Type': 'application/json', 'X-Aura-Agent': '1' },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: combined,
        cache: 'no-store',
      });
    } catch (err) {
      if (signal?.aborted) throw new AgentClientError('CANCELLED', 'request cancelled');
      if (timeout.aborted) throw new AgentClientError('TIMEOUT', 'the agent did not answer in time');
      throw new AgentClientError('BACKEND_UNREACHABLE', 'the agent server is not reachable (is the app server running?)');
    }
    let data = null;
    try {
      data = await res.json();
    } catch {
      throw new AgentClientError('BAD_RESPONSE', `the agent server returned an unreadable response (HTTP ${res.status})`, { status: res.status });
    }
    if (!res.ok) {
      const e = data?.error ?? {};
      throw new AgentClientError(e.code ?? 'HTTP_ERROR', e.message ?? `HTTP ${res.status}`, { status: res.status, retryAfterS: e.retryAfterS ?? null });
    }
    return data;
  }
}
