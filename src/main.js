import { ApplicationCoordinator } from './app/ApplicationCoordinator.js';
import { fetchConfigOverrides, loadConfig } from './config/config.js';
import { COMMAND_NAMES } from './interaction/commands.js';
import { AgentPanel } from './agent/AgentPanel.js';

const $ = (sel) => document.querySelector(sel);

async function main() {
  const fetched = await fetchConfigOverrides('/aura.config.json');
  // URL overrides (?engine=python, ?delegate=CPU&numHands=1), validated like the file.
  const q = new URLSearchParams(location.search);
  const overrides = structuredClone(fetched.overrides ?? {});
  if (q.has('engine') || q.has('engineUrl')) {
    overrides.engine = { ...(overrides.engine ?? {}) };
    if (q.has('engine')) overrides.engine.mode = q.get('engine');
    if (q.has('engineUrl')) overrides.engine.url = q.get('engineUrl');
    fetched.source += ' + URL';
  }
  if (q.has('delegate') || q.has('numHands')) {
    overrides.tracking = { ...(overrides.tracking ?? {}) };
    if (q.has('delegate')) overrides.tracking.delegate = q.get('delegate').toUpperCase();
    if (q.has('numHands')) overrides.tracking.numHands = Number(q.get('numHands'));
    fetched.source += ' + URL';
  }
  const { config, errors, warnings } = loadConfig(overrides, { knownCommands: COMMAND_NAMES });
  if (fetched.error) errors.unshift(fetched.error);

  const dom = {
    stage: $('#stage'),
    video: $('#camera'),
    preview: $('#preview'),
    fxCanvas: $('#fx'),
    sceneCanvas: $('#scene'),
    hudRoot: document.body,
    overlay: $('#overlay'),
  };
  const app = new ApplicationCoordinator(dom, { config, configInfo: { source: fetched.source, errors, warnings } });
  // Read-only debugging hook (used by scripts/integration-check.mjs).
  window.__aura = { snapshot: () => app.snapshot() };

  const overlay = $('#overlay');
  const overlayError = overlay.querySelector('[data-field="overlay.error"]');
  const select = $('#camera-select');
  const toggleBtn = $('#btn-toggle');
  const startBtn = $('#btn-start');

  const showOverlay = (error, { info = false } = {}) => {
    overlay.classList.remove('hidden');
    overlayError.hidden = !error;
    overlayError.textContent = error ?? '';
    overlayError.classList.toggle('overlay-info', info);
    startBtn.textContent = error && !info ? 'Retry' : 'Start camera';
  };

  // Explicit engine switch (full reload, so the current camera owner is shut down
  // first). The app never switches engines on its own when one fails.
  const otherMode = config.engine.mode === 'python' ? 'browser' : 'python';
  overlay.querySelector('[data-field="overlay.engine"]').textContent = `Engine: ${config.engine.mode}.`;
  const engineBtn = $('#btn-engine');
  engineBtn.textContent = `Switch to ${otherMode} engine`;
  engineBtn.addEventListener('click', () => {
    const url = new URL(location.href);
    url.searchParams.set('engine', otherMode);
    location.assign(url);
  });
  // Python engine came (back) online while the dialog shows: say so, do not auto-start the camera.
  app.remote?.on('hello', () => {
    if (!overlay.classList.contains('hidden') && app.camera.status !== 'live' && !overlayError.hidden) {
      showOverlay('Python engine is available. Click Start camera.', { info: true });
    }
  });
  const refreshDevices = async () => {
    const devices = await app.camera.listDevices().catch(() => []);
    select.replaceChildren(...devices.map((d) => Object.assign(document.createElement('option'), { value: d.deviceId, textContent: d.label })));
    if (app.camera.info?.deviceId) select.value = app.camera.info.deviceId;
    select.parentElement.hidden = devices.length < 2;
  };
  const start = async () => {
    startBtn.disabled = true;
    const r = await app.startCamera(select.value || undefined);
    startBtn.disabled = false;
    if (r.ok) {
      overlay.classList.add('hidden');
      toggleBtn.textContent = 'Stop camera';
      refreshDevices();
    } else if (r.error) {
      // The browser's raw message only adds information for unmapped failures.
      const detail = r.error.detail && ['UNKNOWN', 'PLAYBACK_FAILED'].includes(r.error.code) ? ` (${r.error.detail})` : '';
      showOverlay(`${r.error.message}${detail}`);
    }
  };
  const stop = () => {
    app.stopCamera();
    toggleBtn.textContent = 'Start camera';
    showOverlay(null);
  };
  app.onCameraInterrupted = (err) => {
    toggleBtn.textContent = 'Start camera';
    showOverlay(err.message);
  };

  startBtn.addEventListener('click', start);
  toggleBtn.addEventListener('click', () => (app.camera.status === 'live' ? stop() : start()));
  const toggleHud = () => {
    $('#hud').classList.toggle('hidden');
    $('.legend').classList.toggle('hidden');
  };
  $('#btn-hud').addEventListener('click', toggleHud);
  $('#btn-export').addEventListener('click', () => app.exportSession());

  window.addEventListener('keydown', (e) => {
    // Never treat typing in a field (e.g. the agent panel) as an app shortcut.
    const t = e.target;
    if (e.repeat || t instanceof HTMLSelectElement || t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement || t?.isContentEditable) return;
    switch (e.key.toLowerCase()) {
      case 'l': app.submitKeyboard('TOGGLE_LOCK'); break;
      case 'w': app.submitKeyboard('WAVE'); break;
      case 'r': app.submitKeyboard('RESET_POSE'); break;
      case 'h': toggleHud(); break;
      case 'e': app.exportSession(); break;
      case 'escape': if (app.camera.status === 'live') stop(); break;
      default: return;
    }
    e.preventDefault();
  });
  window.addEventListener('pagehide', () => app.shutdown());

  // AI agent panel (talks only to the local /api/agent server; executes only via app.tasks).
  const agentPanel = new AgentPanel($('#agent'), app.agent, { toggleButton: $('#btn-agent') });
  window.addEventListener('pagehide', () => agentPanel.dispose());
  app.agent.init();

  await app.init();
  if (app.tracker.status === 'error') {
    showOverlay(app.remote
      ? `${app.tracker.error}. The app keeps retrying; click Start camera once the engine is running.`
      : `Hand model unavailable: ${app.tracker.error}. The camera can start, but no tracking will run.`);
  }
}

main().catch((err) => {
  console.error(err);
  const el = document.querySelector('[data-field="overlay.error"]');
  if (el) {
    el.hidden = false;
    el.textContent = `Startup failed: ${err?.message ?? err}`;
  }
});
