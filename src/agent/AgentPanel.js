/**
 * AgentPanel: DOM view of the AgentOrchestrator (no agent logic here).
 *
 * - All model and user text is rendered with textContent (never as HTML).
 * - Confirm / Cancel act on the specific proposal id shown, and only for real
 *   user input events (event.isTrusted), so no script or model text can confirm.
 * - Enter sends, Shift+Enter inserts a newline, Escape cancels a pending request.
 *   Keys typed in the panel never reach the app's global shortcuts.
 */
const PHASE_TEXT = {
  unavailable: 'Unavailable',
  idle: 'Ready',
  planning: 'Planning…',
  awaiting_confirmation: 'Awaiting your confirmation',
  executing: 'Executing…',
};
const ACTION_TEXT = {
  proposed: 'proposed',
  awaiting_confirmation: 'awaiting confirmation',
  executing: 'executing',
  succeeded: 'succeeded',
  failed: 'failed',
  rejected: 'rejected',
  declined: 'declined',
  expired: 'expired',
  skipped: 'skipped',
  unverified: 'done, not verified',
};

export class AgentPanel {
  /**
   * @param {HTMLElement} root  #agent element
   * @param {import('./AgentOrchestrator.js').AgentOrchestrator} agent
   * @param {{ toggleButton?: HTMLElement|null }} [opts]
   */
  constructor(root, agent, opts = {}) {
    this.root = root;
    this.agent = agent;
    this.toggleButton = opts.toggleButton ?? null;
    this.$ = (sel) => root.querySelector(sel);
    this.log = this.$('[data-agent="log"]');
    this.phase = this.$('[data-agent="phase"]');
    this.status = this.$('[data-agent="status"]');
    this.input = this.$('#agent-input');
    this.sendBtn = this.$('#agent-send');
    this.stopBtn = this.$('#agent-stop');
    this.confirmBox = this.$('[data-agent="confirm"]');
    this.confirmText = this.$('[data-agent="confirm-text"]');
    this.confirmTimer = this.$('[data-agent="confirm-timer"]');
    this.yesBtn = this.$('#agent-confirm-yes');
    this.noBtn = this.$('#agent-confirm-no');
    this._renderedCount = -1;
    this._wire();
    agent.onChange(() => this.render());
    this._tick = setInterval(() => this._renderCountdown(), 500);
    this.render();
  }

  _wire() {
    this.$('#agent-form').addEventListener('submit', (e) => {
      e.preventDefault();
      this._send();
    });
    this.input.addEventListener('keydown', (e) => {
      e.stopPropagation(); // never trigger global shortcuts (L, W, R, H, E, Esc) while typing
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        this._send();
      } else if (e.key === 'Escape') {
        e.preventDefault();
        this.agent.cancel();
      }
    });
    this.root.addEventListener('keydown', (e) => e.stopPropagation());
    this.stopBtn.addEventListener('click', () => this.agent.cancel());
    this.$('#agent-clear').addEventListener('click', () => {
      this.agent.clear();
      this.input.focus();
    });
    this.$('#agent-close').addEventListener('click', () => this.setOpen(false));
    this.toggleButton?.addEventListener('click', () => this.setOpen(this.root.classList.contains('hidden')));
    const onDecision = (yes) => (e) => {
      if (!e.isTrusted) return; // only a real user action confirms
      const p = this.agent.pending;
      if (!p) return;
      const r = yes ? this.agent.confirm(p.proposalId) : this.agent.decline(p.proposalId);
      if (!r.ok && r.reason !== 'STALE_CONFIRMATION') this.render();
      this.input.focus();
    };
    this.yesBtn.addEventListener('click', onDecision(true));
    this.noBtn.addEventListener('click', onDecision(false));
  }

  setOpen(open) {
    this.root.classList.toggle('hidden', !open);
    this.toggleButton?.setAttribute('aria-expanded', String(open));
    document.querySelector('.legend')?.classList.toggle('agent-open', open);
    if (open) this.input.focus();
  }

  _send() {
    const text = this.input.value;
    const r = this.agent.send(text);
    if (r.accepted) this.input.value = '';
  }

  render() {
    const a = this.agent;
    const phase = a.phase;
    this.root.dataset.phase = phase;
    const wf = [...a.messages].reverse().find((m) => m.role === 'workflow' && (m.status === 'executing'));
    const current = wf?.steps?.find((st) => st.status === 'executing' || st.status === 'awaiting_confirmation');
    this.phase.textContent = phase === 'unavailable'
      ? `Unavailable — ${a.availability.reason ?? 'agent not configured'}`
      : wf && current && (phase === 'executing' || phase === 'awaiting_confirmation')
        ? `${PHASE_TEXT[phase]} — workflow step: ${current.label}`
        : PHASE_TEXT[phase];
    this.status.textContent = a.availability.available ? `${a.availability.model ?? ''}` : 'offline';
    this.status.dataset.level = a.availability.available ? 'ok' : 'bad';
    const busy = phase === 'planning' || phase === 'executing';
    const wasBusy = this.input.disabled && this._lastPhase !== 'unavailable';
    this._lastPhase = phase;
    this.input.disabled = phase === 'unavailable' || busy;
    // Disabling the field drops focus to <body>; give it back when input is possible
    // again (never to the Confirm button: an accidental Enter must not confirm).
    if (wasBusy && !this.input.disabled && !this.root.classList.contains('hidden')
      && (document.activeElement === document.body || document.activeElement === null)) {
      queueMicrotask(() => this.input.focus());
    }
    this.sendBtn.disabled = phase === 'unavailable' || busy;
    this.stopBtn.hidden = !(phase === 'planning');
    this._renderLog();
    const p = a.pending;
    this.confirmBox.hidden = !p;
    if (p) {
      this.confirmText.textContent = `Confirm action: ${p.label}`;
      this.confirmBox.dataset.proposal = p.proposalId;
      this._renderCountdown();
    }
  }

  _renderCountdown() {
    const p = this.agent.pending;
    if (!p || !this.confirmTimer) return;
    const s = Math.max(0, Math.ceil((p.expiresAt - this.agent.now()) / 1000));
    this.confirmTimer.textContent = `expires in ${s} s`;
  }

  _renderLog() {
    const msgs = this.agent.messages;
    const atBottom = this.log.scrollHeight - this.log.scrollTop - this.log.clientHeight < 24;
    this.log.replaceChildren(
      ...msgs.map((m) => {
        const li = document.createElement('li');
        li.dataset.role = m.role;
        if (m.role === 'workflow') {
          li.dataset.status = m.status;
          const head = document.createElement('span');
          head.className = 'agent-action-label';
          head.textContent = `App workflow · ${m.label}`;
          const chip = document.createElement('b');
          chip.className = 'agent-chip';
          chip.textContent = ACTION_TEXT[m.status] ?? m.status;
          const steps = document.createElement('ol');
          steps.className = 'agent-steps';
          for (const st of m.steps ?? []) {
            const s = document.createElement('li');
            s.dataset.status = st.status;
            s.textContent = `${st.label} — ${ACTION_TEXT[st.status] ?? st.status}${st.error ? ` (${st.error})` : ''}${st.note ? ` (${st.note})` : ''}`;
            steps.append(s);
          }
          li.append(head, chip, steps);
          if (m.summary) {
            const sum = document.createElement('small');
            sum.textContent = m.summary; // trusted application summary, not model text
            li.append(sum);
          }
          return li;
        }
        if (m.role === 'action') {
          li.dataset.status = m.status;
          const label = document.createElement('span');
          label.className = 'agent-action-label';
          label.textContent = `App · ${m.label}`;
          const chip = document.createElement('b');
          chip.className = 'agent-chip';
          chip.textContent = ACTION_TEXT[m.status] ?? m.status;
          li.append(label, chip);
          if (m.error) {
            const err = document.createElement('small');
            err.textContent = m.error;
            li.append(err);
          }
        } else {
          const who = document.createElement('span');
          who.className = 'agent-who';
          who.textContent = { user: 'You', agent: 'AURA · model', notice: 'App', error: 'Error' }[m.role] ?? m.role;
          const text = document.createElement('span');
          text.className = 'agent-text';
          text.textContent = m.text + (m.truncated ? ' (truncated)' : '');
          li.append(who, text);
        }
        return li;
      }),
    );
    // Follow new content (incl. workflow step updates) unless the user scrolled up to read.
    if (msgs.length !== this._renderedCount || atBottom) {
      this._renderedCount = msgs.length;
      this.log.scrollTop = this.log.scrollHeight;
    }
  }

  dispose() {
    clearInterval(this._tick);
  }
}
