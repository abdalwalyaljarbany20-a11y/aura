/**
 * HUDController: renders application state into the DOM panel.
 *
 * It only displays values passed in `update(view)`; it computes nothing
 * itself, so every number is traceable to a measurement in the
 * coordinator/pipeline. Text is refreshed at most every `refreshMs`.
 * All text is set with textContent (no HTML injection).
 */
const fmt = (v, digits = 1, unit = '') => (v === null || v === undefined || !Number.isFinite(v) ? '—' : `${v.toFixed(digits)}${unit}`);

const LEVEL_BY_STATE = { IDLE: '', TRACKING: 'ok', CONTROL: 'active', EXECUTING: 'active', LOCKED: 'warn', ERROR: 'bad' };

export class HUDController {
  /**
   * @param {HTMLElement} root element containing [data-field] nodes
   * @param {object} cfg hud configuration
   */
  constructor(root, cfg) {
    this.root = root;
    this.cfg = cfg;
    this.fields = new Map();
    for (const el of root.querySelectorAll('[data-field]')) this.fields.set(el.dataset.field, el);
    this.events = [];
    this.lastRefresh = -Infinity;
    this._eventsDirty = false;
  }

  set(field, text, level) {
    const el = this.fields.get(field);
    if (!el) return;
    const s = String(text);
    if (el.textContent !== s) el.textContent = s;
    if (level !== undefined) {
      if (level) el.dataset.level = level;
      else delete el.dataset.level;
    }
  }

  pill(name, text, level) {
    this.set(`pill.${name}`, text);
    const pill = this.root.querySelector(`[data-pill="${name}"]`);
    if (pill) pill.dataset.level = level ?? '';
  }

  /**
   * Append a real application event to the log.
   * @param {string} text
   * @param {'ok'|'warn'|'bad'|'active'|'info'|''} level
   * @param {number} [wallTime] Date.now()
   */
  log(text, level = '', wallTime = Date.now()) {
    this.events.unshift({ text, level, time: new Date(wallTime) });
    if (this.events.length > 200) this.events.length = 200;
    this._eventsDirty = true;
  }

  renderLegend(bindings, commands, activeCommand) {
    const ul = this.fields.get('legend');
    if (!ul) return;
    const key = JSON.stringify([bindings, activeCommand]);
    if (ul.dataset.key === key) return;
    ul.dataset.key = key;
    ul.replaceChildren(
      ...Object.entries(bindings).map(([gesture, command]) => {
        const li = document.createElement('li');
        const b = document.createElement('b');
        b.textContent = gesture.replace('_', ' ');
        const span = document.createElement('span');
        span.textContent = command ?? 'unbound';
        if (command && !commands[command]) {
          li.dataset.unsupported = 'true';
          span.textContent = `${command} (unsupported)`;
        }
        if (command && command === activeCommand) li.dataset.active = 'true';
        li.append(b, span);
        return li;
      }),
    );
  }

  /** @param {number} now @param {any} v view model built by the coordinator */
  update(now, v) {
    if (now - this.lastRefresh < this.cfg.refreshMs) return;
    this.lastRefresh = now;

    // top pills
    this.pill('camera', v.camera.pill, v.camera.level);
    this.pill('tracker', v.tracker.pill, v.tracker.level);
    this.pill('state', v.robot.state, LEVEL_BY_STATE[v.robot.state]);

    // gesture
    const g = v.gesture;
    const nameEl = this.fields.get('gesture.stable');
    this.set('gesture.stable', g.stable.replace('_', ' '));
    if (nameEl) nameEl.dataset.neutral = String(g.stable === 'NONE');
    this.set('gesture.raw', g.raw);
    this.set('gesture.confidence', g.confidence === null ? '—' : g.confidence.toFixed(2));
    this.set('gesture.reason', g.reason ?? '');
    const meter = this.fields.get('gesture.meter');
    if (meter) meter.style.width = `${Math.round((g.confidence ?? 0) * 100)}%`;

    // interaction
    this.set('robot.state', v.robot.state, LEVEL_BY_STATE[v.robot.state]);
    this.set('robot.active', v.robot.active, v.robot.active === '—' ? '' : 'active');
    this.set('robot.movement', v.robot.movement.text, v.robot.movement.level);
    this.set('robot.position', v.robot.position);
    this.set('robot.face', v.robot.face.text, v.robot.face.level);
    this.set('robot.attention', v.robot.attention.text, v.robot.attention.level);
    const ts = v.tracking.status;
    this.set('tracking.status', ts, ts === 'DETECTED' ? 'ok' : ts === 'TRACKING_LOST' ? 'warn' : ts === 'NOT_DETECTED' ? '' : 'bad');
    this.set('tracking.hands', v.tracking.hands);
    this.set('tracking.primary', v.tracking.primary);

    // performance (all measured)
    const p = v.perf;
    this.set('perf.camera', fmt(p.cameraFps, 1));
    this.set('perf.processed', fmt(p.processedFps, 1));
    this.set('perf.render', fmt(p.renderFps, 0));
    this.set('perf.inference', p.inference.mean === null ? '—' : `${fmt(p.inference.mean)} / p95 ${fmt(p.inference.p95)} ms`);
    this.set('perf.pipeline', p.pipeline.mean === null ? '—' : `${fmt(p.pipeline.mean)} ms`);
    this.set('perf.overlay', p.overlay.mean === null ? '—' : `${fmt(p.overlay.mean)} ms`);
    this.set('perf.dropped', p.dropped === null ? '—' : String(p.dropped));
    this.set('perf.preview', p.preview);

    // system
    this.set('sys.engine', v.engine.detail, v.engine.level);
    if (v.agent) {
      this.set('sys.agent', v.agent.system, v.agent.systemLevel);
      this.set('robot.agent', v.agent.phase, v.agent.phaseLevel);
    }
    this.set('sys.camera', v.camera.detail, v.camera.level);
    this.set('sys.tracker', v.tracker.detail, v.tracker.level);
    this.set('sys.asset', v.asset.detail, v.asset.level);
    this.set('sys.config', v.config.detail, v.config.level);
    this.set('sys.session', v.session);

    if (this._eventsDirty) this._renderEvents();
  }

  _renderEvents() {
    this._eventsDirty = false;
    const ol = this.fields.get('events');
    if (!ol) return;
    ol.replaceChildren(
      ...this.events.slice(0, this.cfg.maxEvents).map((e) => {
        const li = document.createElement('li');
        if (e.level) li.dataset.level = e.level;
        const time = document.createElement('time');
        time.textContent = e.time.toTimeString().slice(0, 8);
        const span = document.createElement('span');
        span.textContent = e.text;
        li.append(time, span);
        return li;
      }),
    );
  }
}
