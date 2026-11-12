import { createHandPoseNet } from '../perception/handPoseNet.js';
import { CameraManager } from '../camera/CameraManager.js';
import { EventBus } from '../core/EventBus.js';
import { RateMeter, RollingStats } from '../core/metrics.js';
import { createCoverMapping } from '../fx/screenMapping.js';
import { VisualEffectsEngine } from '../fx/VisualEffectsEngine.js';
import { FeatureExtractionEngine } from '../gestures/FeatureExtractionEngine.js';
import { GestureRecognitionEngine } from '../gestures/GestureRecognitionEngine.js';
import { HUDController } from '../hud/HUDController.js';
import { COMMANDS } from '../interaction/commands.js';
import { InteractionController } from '../interaction/InteractionController.js';
import { Scene3D } from '../scene/Scene3D.js';
import { HandTrackingEngine } from '../tracking/HandTrackingEngine.js';
import { createMediaPipeBackend, createMediaPipeFaceBackend } from '../tracking/mediapipeBackend.js';
import { FaceTrackingEngine } from '../tracking/FaceTrackingEngine.js';
import { AttentionController } from '../interaction/AttentionController.js';
import { FaceExpressionEngine, gazeFromBlendshapes, HeadCueCalibrator, headPoseFromMatrix } from '../perception/faceAnalysis.js';
import { PythonEngineClient } from '../remote/PythonEngineClient.js';
import { registerRobotTasks } from '../agent/robotTasks.js';
import { TaskBroker } from '../agent/TaskBroker.js';
import { AgentClient } from '../agent/AgentClient.js';
import { AgentOrchestrator } from '../agent/AgentOrchestrator.js';
import { registerAppStatusTask } from '../agent/appStatusTask.js';
import { deriveBlockers, IssueLog, sanitizeMessage } from '../agent/diagnostics.js';
import { registerDiagnosticsTask } from '../agent/diagnosticsTask.js';
import { RemoteCamera, RemoteFramePipeline, RemoteTracker } from '../remote/RemoteEngine.js';
import { FramePipeline } from './FramePipeline.js';

/**
 * ApplicationCoordinator
 *
 * Initialisation order:
 *   1. event bus, HUD (so every later step can report status)
 *   2. 3D scene + effects (render loop starts; robot idles without camera)
 *   3. hand model load (async, in parallel with the user's "Start camera")
 *   4. optional GLB robot load (async, never blocks tracking)
 *   5. camera on user request -> frame loop -> FramePipeline
 *
 * Shutdown (stop button, Esc, page hide): frame loop, camera tracks,
 * tracker, WebGL resources, listeners, in that order.
 *
 * Engine modes (config.engine.mode), exactly one camera owner each:
 *   browser  CameraManager (getUserMedia) + MediaPipe WASM + JS gesture engine
 *   python   the Python engine owns the camera; RemoteCamera / RemoteTracker /
 *            RemoteFramePipeline adapt its results to the same interfaces, so the
 *            controller, effects, scene and HUD are shared by both modes.
 */
export class ApplicationCoordinator {
  /**
   * @param {{ stage: HTMLElement, video: HTMLVideoElement, fxCanvas: HTMLCanvasElement,
   *           sceneCanvas: HTMLCanvasElement, hudRoot: HTMLElement, overlay: HTMLElement }} dom
   * @param {{ config: any, configInfo: { source: string, errors: string[], warnings: string[] } }} opts
   */
  constructor(dom, { config, configInfo }) {
    this.dom = dom;
    this.config = config;
    this.configInfo = configInfo;
    this.bus = new EventBus();
    this.hud = new HUDController(dom.hudRoot, config.hud);
    this.controller = new InteractionController(config.interaction, { bus: this.bus });
    // Integration boundary for the future agent (Phase 3). Not reachable from the
    // page or the network yet; an agent adapter will call this.tasks.submit().
    this.tasks = new TaskBroker({ bus: this.bus });
    registerRobotTasks(this.tasks, { controller: this.controller, bus: this.bus });
    registerAppStatusTask(this.tasks, () => this.agentStatus());
    this.issues = new IssueLog(8);
    registerDiagnosticsTask(this.tasks, () => this.agentDiagnostics());
    // AI agent (Phase 3): proposals from the local agent server run only through this.tasks.
    // Independent of the vision engine: provider failures never touch camera or tracking.
    this.agent = new AgentOrchestrator({ client: new AgentClient(), tasks: this.tasks, isLocked: () => this.controller.locked });
    this.mode = config.engine.mode;
    // Gaze: faces come from the active engine's own frames (browser: FaceTrackingEngine on
    // the same <video>; python: result.faces). Never a second camera; never sent remotely.
    this.attention = new AttentionController(config.interaction.attention);
    // facial expression / head pose / gaze of the primary face (both engines: same JS logic)
    this.expression = new FaceExpressionEngine(config.interaction.expression);
    this.cueCalibrator = new HeadCueCalibrator(config.interaction.mimic?.calibration);
    this.user = null;
    this.status = null;
    this.gaze = null;
    this.faceTracker = null;
    const extractor = new FeatureExtractionEngine({ mirror: config.camera.mirror });
    if (this.mode === 'python') {
      this.remote = new PythonEngineClient(config.engine.url, { bus: this.bus, connectTimeoutMs: config.engine.connectTimeoutMs });
      this.camera = new RemoteCamera(this.remote, { preview: dom.preview, bus: this.bus });
      this.tracker = new RemoteTracker(this.remote, { bus: this.bus });
      this.pipeline = new RemoteFramePipeline({ extractor, controller: this.controller, bus: this.bus });
    } else {
      this.remote = null;
      this.camera = new CameraManager(config.camera, { video: dom.video, bus: this.bus });
      this.tracker = new HandTrackingEngine({ backend: createMediaPipeBackend(config.tracking), bus: this.bus });
      this.faceTracker = new FaceTrackingEngine({
        backend: config.face.enabled ? createMediaPipeFaceBackend(config.face, config.tracking) : null,
        intervalMs: config.face.intervalMs,
        minScore: config.face.minScore,
      });
      this.pipeline = new FramePipeline({
        tracker: this.tracker,
        extractor,
        recognizer: (this.recognizer = new GestureRecognitionEngine(config.gestures)),
        controller: this.controller,
        bus: this.bus,
      });
    }
    this.scene = null;
    this.fx = null;
    this.frameTimeSource = null;
    this.sceneError = null;
    this.metrics = {
      camera: new RateMeter(1000),
      render: new RateMeter(1000),
      overlayMs: new RollingStats(120),
      skippedWhileLoading: 0,
    };
    this._lastPresented = 0;
    this._lastOverlayT = null;
    this._raf = null;
    this._lastNow = null;
    this._disposers = [];
    this.running = false;
  }

  async init() {
    this._wireEvents();
    const { errors, warnings, source } = this.configInfo;
    this.hud.log(`Config: ${source}`, 'info');
    for (const e of errors) this.hud.log(`Config error: ${e}`, 'bad');
    for (const w of warnings) this.hud.log(`Config: ${w}`, 'warn');
    this.dom.stage.classList.toggle('mirror', this.config.camera.mirror);

    try {
      this.scene = new Scene3D(this.dom.sceneCanvas, this.config.scene);
      this.scene.mimicCfg = { ...this.scene.mimicCfg, ...(this.config.interaction.mimic ?? {}) };
      this.hud.log('3D scene ready (WebGL)', 'ok');
    } catch (err) {
      this.sceneError = `WebGL unavailable: ${err?.message ?? err}`;
      this.hud.log(this.sceneError, 'bad');
    }
    this.fx = new VisualEffectsEngine(this.dom.fxCanvas, this.config.effects);
    this._onResize = () => this._resize();
    window.addEventListener('resize', this._onResize);
    this._disposers.push(() => window.removeEventListener('resize', this._onResize));
    this._resize();
    this._startRenderLoop();

    this.running = true;
    if (this.remote) {
      this.dom.video.hidden = true;
      this.hud.log(`Engine: Python (${this.config.engine.url})`, 'info');
      this._disposers.push(this.remote.on('hello', (h) => {
        const e = h.engine;
        this.hud.log(`Python engine ${e.version}: mediapipe ${e.mediapipe}, OpenCV ${e.opencv}, Python ${e.python}`, 'ok');
        for (const err of h.configErrors ?? []) this.hud.log(`Engine config error: ${err}`, 'bad');
        const pn = h.poseNet;
        if (pn) this.hud.log(pn.status === 'ready' ? `Hand-pose net ready (${pn.params} parameters, trained on ${pn.dataset?.name ?? 'HaGRIDv2'})` : `Hand-pose net ${pn.status}${pn.error ? `: ${pn.error}` : ''} (canned model + rules)`, pn.status === 'ready' ? 'ok' : 'warn');
      }));
      let down = false; // log each outage once, not every reconnect attempt
      this._disposers.push(this.remote.on('connection', (c) => {
        if (c.status === 'connected') {
          if (down) this.hud.log('Python engine reconnected', 'ok');
          down = false;
        } else if (c.status === 'disconnected' && c.reason !== 'closed by client' && !down) {
          down = true;
          this.hud.log(`Python engine disconnected: ${c.reason} (retrying)`, 'warn');
        }
      }));
    }
    this.hud.log(this.remote ? 'Connecting to Python engine…' : 'Loading hand landmark model…', 'info');
    const trackerReady = this.tracker.init();
    if (!this.remote) this._loadPoseNet();
    // Optional and independent: a missing face model only disables gaze.
    this.faceTracker?.init().then((r) => this.hud.log(r.ok ? `Face detector ready (${this.faceTracker.delegate})` : `Face detector unavailable: ${this.faceTracker.error ?? 'disabled'} (robot looks straight ahead)`, r.ok ? 'ok' : 'warn'));
    if (this.scene) {
      this.scene.loadRobot().then((a) => this.hud.log(`Robot: ${a.reason}`, a.status === 'fallback' ? 'warn' : 'ok'));
    }
    await trackerReady;
  }

  /** Optional project-trained hand-pose net (browser engine); missing / malformed = model + rules only. */
  async _loadPoseNet() {
    const path = this.config.tracking.handPoseNetPath;
    if (!path || this.config.gestures.net?.enabled === false) return;
    try {
      const res = await fetch(path);
      if (!res.ok) throw new Error(`HTTP ${res.status} (run training/train_handpose.py)`);
      const net = createHandPoseNet(await res.json());
      this.recognizer.net = net;
      this.poseNetInfo = { status: 'ready', ...net.info };
      this.hud.log(`Hand-pose net ready (${net.info.params} parameters, trained on ${net.info.dataset?.name ?? 'HaGRIDv2'})`, 'ok');
    } catch (err) {
      this.poseNetInfo = { status: 'missing', error: String(err?.message ?? err) };
      this.hud.log(`Hand-pose net unavailable: ${this.poseNetInfo.error} (canned model + rules)`, 'warn');
    }
  }

  async startCamera(deviceId) {
    if (this.camera.status === 'starting') return { ok: false };
    this.hud.log('Requesting camera…', 'info');
    const r = await this.camera.start(deviceId);
    if (!r.ok) {
      if (r.error.code !== 'ABORTED') this.controller.fail(r.error.message, performance.now());
      return r;
    }
    // A camera restart clears camera errors, but not a failed hand model.
    if (this.tracker.status !== 'error') this.controller.recover(performance.now());
    this._lastPresented = this.camera.stats.presented;
    // Python mode: results drive state immediately; drawn previews select the overlay pair.
    this.camera.startFrameLoop((frame) => this._onFrame(frame), this.remote ? (v) => this.pipeline.commitVisual(v.frameId, v.t) : undefined);
    return r;
  }

  stopCamera() {
    this.camera.stop();
    this.attention.reset();
    const now = performance.now();
    // Release gestures/holds exactly as if the hand disappeared.
    this.pipeline.recognizer.reset();
    this.controller.setHandPresent(false, now);
    if (this.remote) this.pipeline.reset();
    else this.pipeline.latest = null;
    this.fx?.clear();
  }

  _onFrame(frame) {
    if (this.remote) {
      this.pipeline.process(frame.result);
      if (Array.isArray(frame.result.faces)) this._observeFaces(frame.result.faces);
      return;
    }
    const delta = this.camera.stats.presented - this._lastPresented;
    this._lastPresented = this.camera.stats.presented;
    this.metrics.camera.tick(frame.t, Math.max(1, delta));
    if (this.tracker.status !== 'ready') {
      this.metrics.skippedWhileLoading++;
      return;
    }
    this.frameTimeSource = frame.timeSource;
    this.pipeline.process({ video: frame.video, t: frame.t, capturedAt: frame.capturedAt, aspect: this.camera.aspect });
    // same frame, throttled (face.intervalMs): gaze needs far fewer updates than the hand
    const v = frame.video;
    const faces = this.faceTracker?.detect(v, frame.t, { width: v.videoWidth, height: v.videoHeight });
    if (faces) this._observeFaces(faces);
  }

  /** Image-normalised faces -> screen-normalised (mirrored like the view) -> attention. */
  _observeFaces(faces) {
    const rect = this.dom.stage.getBoundingClientRect();
    if (!(rect.width > 0 && rect.height > 0)) return;
    const map = this._mapping();
    const mirror = this.config.camera.mirror;
    const toScreen = (x, y) => {
      const p = map.displayToScreen({ x: mirror ? 1 - x : x, y });
      return { x: p.x / rect.width, y: p.y / rect.height };
    };
    const screen = faces
      .filter((f) => [f?.x, f?.y, f?.w, f?.h, f?.score].every(Number.isFinite))
      .map((f) => {
        const c = toScreen(f.x, f.y);
        const a = toScreen(f.x - f.w / 2, f.y - f.h / 2);
        const b = toScreen(f.x + f.w / 2, f.y + f.h / 2);
        const out = { x: c.x, y: c.y, w: Math.abs(b.x - a.x), h: Math.abs(b.y - a.y), score: f.score };
        // face-model outputs travel with the face (validated where they are used)
        if (f.bs && typeof f.bs === 'object') out.bs = f.bs;
        if (Array.isArray(f.matrix) && f.matrix.length === 16) out.matrix = f.matrix;
        // head-turn cue in screen space: image yaw flips with the mirrored view
        if (Number.isFinite(f.cue?.yaw) && Number.isFinite(f.cue?.pitch)) out.cue = { yaw: mirror ? -f.cue.yaw : f.cue.yaw, pitch: f.cue.pitch };
        return out;
      });
    this.attention.observeFaces(performance.now(), screen);
  }

  /** Exponential smoothing of the head-turn cue (time constant 90 ms; 300 ms decay when lost). */
  _smoothCue(now, cue) {
    const s = (this._cue ??= { yaw: 0, pitch: 0, t: now });
    const dt = Math.max(0, Math.min(100, now - s.t));
    s.t = now;
    const k = 1 - Math.exp(-dt / (cue ? 90 : 300));
    s.yaw += ((cue?.yaw ?? 0) - s.yaw) * k;
    s.pitch += ((cue?.pitch ?? 0) - s.pitch) * k;
    return { yaw: s.yaw, pitch: s.pitch, active: !!cue };
  }

  /** Submit a command from the keyboard (same validation as gestures). */
  submitKeyboard(name) {
    return this.controller.submit({ name, source: 'keyboard' }, performance.now());
  }

  _startRenderLoop() {
    const step = (now) => {
      this._raf = requestAnimationFrame(step);
      try {
        this._renderFrame(now);
      } catch (err) {
        console.error('[render] frame failed', err);
      }
    };
    this._raf = requestAnimationFrame(step);
  }

  _renderFrame(now) {
    const dt = this._lastNow === null ? 0 : Math.min(0.1, (now - this._lastNow) / 1000);
    this._lastNow = now;
    this.metrics.render.tick(now);
    this.controller.tick(now);

    const live = this.camera.status === 'live';
    const latest = live ? this.pipeline.latest : null; // newest state (HUD, gestures, commands)
    // What may be drawn over the frame on screen (Python mode: frame-matched pairs only).
    const visual = !live ? null : this.remote ? this.pipeline.visualLatest(now) : latest;
    const map = this._mapping();
    const snapshot = this.controller.snapshot(now);
    const fresh = visual && now - visual.processedAt < 500;
    const f = fresh ? visual.features : null;
    const rect = this.dom.stage.getBoundingClientRect();
    const handScreen = f && rect.width > 0 ? (() => {
      const p = map.displayToScreen(f.indexTipNorm);
      return { x: p.x / rect.width, y: p.y / rect.height };
    })() : null;
    this.gaze = this.attention.update(now, { head: live ? this.scene?.headScreen() ?? null : null, hold: snapshot.hold?.name ?? null, hand: handScreen });
    if (!live) this.attention.reset();
    // perception -> decision inputs for the animation layer (no command is issued here)
    const pf = this.gaze?.faceVisible ? this.attention.primary : null;
    const bs = pf?.bs && Object.keys(pf.bs).length ? pf.bs : null;
    const expr = this.expression.update(now, bs);
    this.user = {
      faceModel: !!bs,
      expression: this.config.interaction.expression.react && bs ? expr : null,
      expressionRaw: expr,
      headPose: pf?.matrix ? headPoseFromMatrix(pf.matrix) : null,
      gaze: bs ? gazeFromBlendshapes(bs) : null,
      // head imitation (smoothed; decays to 0 without a face): + yaw = turned toward screen right
      headCue: this._smoothCue(now, this.cueCalibrator.update(now, this.config.interaction.mimic?.enabled !== false ? pf?.cue ?? null : null)),
    };
    this.status = this._robotStatus(snapshot, latest, live);

    this.fx.draw(visual, map, now, {
      stable: latest?.gesture?.stable?.gesture,
      controlling: snapshot.state === 'CONTROL',
      // marionette strings: fingertips -> robot anchors (video reference); they fall when the hand is lost
      anchors: live && this.scene ? this.scene.anchorPoints() : null,
    });
    if (visual && visual.hands.length && visual.t !== this._lastOverlayT) {
      // performance.now(), not the rAF timestamp: the rAF time marks the
      // start of this frame, which can precede the processing that just ran.
      this._lastOverlayT = visual.t;
      if (!visual.warmup) this.metrics.overlayMs.push(performance.now() - visual.capturedAt);
    }
    if (this.scene) {
      this.scene.update(dt, now, { snapshot, hand: f ? { palm: f.palmCenterNorm, indexTip: f.indexTipNorm, roll: f.roll } : null, map, gaze: this.gaze, user: this.user, status: this.status });
      this.scene.render();
    }
    this.hud.renderLegend(this.config.interaction.bindings, COMMANDS, snapshot.action?.name ?? snapshot.hold?.name ?? null);
    this.hud.update(now, this._view(now, snapshot, latest));
  }

  _mapping() {
    const rect = this.dom.stage.getBoundingClientRect();
    const size = this.remote ? this.camera.frameSize : null;
    return createCoverMapping({
      videoWidth: size ? size.width : this.dom.video.videoWidth,
      videoHeight: size ? size.height : this.dom.video.videoHeight,
      width: rect.width,
      height: rect.height,
      mirror: this.config.camera.mirror,
    });
  }

  _resize() {
    const rect = this.dom.stage.getBoundingClientRect();
    this.fx.resize(rect.width, rect.height, this.config.scene.pixelRatioCap);
    this.scene?.resize(rect.width, rect.height);
  }

  /** Builds the HUD view model from real state only. */
  _view(now, snapshot, latest) {
    const cam = this.camera;
    const camLevel = cam.status === 'live' ? 'ok' : cam.status === 'error' ? 'bad' : cam.status === 'starting' ? 'warn' : '';
    const camDetail =
      cam.status === 'live' ? `live ${cam.info.width ?? '?'}×${cam.info.height ?? '?'}${this.remote ? (cam.info.label.includes('TEST-FILE') ? ' (python · TEST-FILE)' : ' (python)') : ''}`
        : cam.status === 'error' ? cam.error.code
          : cam.status === 'starting' ? 'waiting for permission' : 'stopped';

    const tr = this.tracker;
    const trLevel = tr.status === 'ready' ? 'ok' : tr.status === 'error' ? 'bad' : 'warn';
    const trDetail = tr.status === 'ready' ? `ready · ${tr.delegate}` : tr.status === 'error' ? tr.error : tr.status;

    const g = latest?.gesture;
    const hands = latest?.hands ?? [];
    const f = latest?.features;
    const s = this.pipeline.session;
    const pm = this.pipeline.metrics;
    const assetLevel = !this.scene ? 'bad' : this.scene.asset.status === 'fallback' ? 'warn' : this.scene.asset.status === 'loading' ? 'warn' : 'ok';

    return {
      camera: { pill: cam.status === 'live' ? 'LIVE' : cam.status.toUpperCase(), level: camLevel, detail: camDetail },
      tracker: { pill: tr.status === 'ready' ? (tr.delegate ?? 'READY') : tr.status.toUpperCase(), level: trLevel, detail: trDetail },
      robot: {
        state: snapshot.state,
        active: snapshot.action ? `${snapshot.action.name} ${Math.round(snapshot.action.progress * 100)}%` : snapshot.hold ? snapshot.hold.name : snapshot.state === 'ERROR' ? snapshot.error : '—',
        movement: this._movementView(snapshot),
        face: this._faceView(),
        attention: this._attentionView(),
        position: `x ${snapshot.pose.x.toFixed(2)} · z ${snapshot.pose.z.toFixed(2)} m · yaw ${Math.round((snapshot.pose.yaw * 180) / Math.PI)}°`,
      },
      gesture: {
        stable: g?.stable?.gesture ?? 'NONE',
        raw: g ? `${g.raw.label}${g.raw.source ? ` · ${g.raw.source}` : ''}` : '—',
        confidence: g && g.hand ? g.raw.confidence : null,
        reason: g?.raw?.reason && g.raw.reason !== 'no hand' ? g.raw.reason : g?.raw?.label === 'MOVING' ? 'hand moving (hold still for static gestures)' : '',
      },
      tracking: {
        status: this._trackingStatus(latest),
        hands: String(hands.length),
        primary: f ? `${f.handedness ?? '?'} · ${f.score === null ? '—' : f.score.toFixed(2)}` : '—',
      },
      perf: {
        // Python mode: camera/processed FPS are measured by the engine itself.
        cameraFps: cam.status !== 'live' ? null : this.remote ? this.remote.metrics?.cameraFps ?? null : this.metrics.camera.rate(now),
        processedFps: cam.status !== 'live' ? null : this.remote ? this.remote.metrics?.processedFps ?? null : pm.processed.rate(now),
        renderFps: this.metrics.render.rate(now),
        inference: pm.inferenceMs.summary(),
        pipeline: pm.pipelineMs.summary(),
        overlay: this.metrics.overlayMs.summary(),
        dropped: cam.status !== 'live' ? null : this.remote ? this.remote.metrics?.droppedFrames ?? null : cam.stats.dropped,
        preview: this._previewView(cam),
      },
      engine: this._engineView(),
      agent: this._agentView(),
      asset: { detail: this.sceneError ?? this.scene.asset.reason, level: assetLevel },
      config: {
        detail: `${this.configInfo.source}${this.configInfo.errors.length ? ` · ${this.configInfo.errors.length} errors` : ''}`,
        level: this.configInfo.errors.length ? 'bad' : this.configInfo.warnings.length ? 'warn' : 'ok',
      },
      session: s.frames ? `${s.frames} fr · hand ${Math.round((100 * s.framesWithHand) / s.frames)}% · lost ${s.trackingLosses}` : '—',
    };
  }

  /** Movement line: WALKING (speed, cadence) / ROTATING (direction, rate) / LAUGHING / SAD / RETURNING / IDLE. */
  _movementView(snapshot) {
    const m = snapshot.movement;
    if (m.state === 'WALKING') {
      return { text: `WALKING · ${m.speed.toFixed(2)} m/s · ${m.cadence.toFixed(1)} steps/s${m.atBoundary ? ' · at boundary (R returns home)' : ''}`, level: m.atBoundary ? 'warn' : 'active' };
    }
    if (m.state === 'ROTATING') {
      const rate = m.rateDegS === null || m.rateDegS === undefined ? '' : ` · ${Math.abs(m.rateDegS)}°/s`;
      return { text: `ROTATING · ${(m.direction ?? 'starting').toLowerCase()}${rate}${m.source === 'SPIN' && snapshot.hold?.name !== 'SPIN' ? ' · slowing' : ''}`, level: 'active' };
    }
    if (m.state === 'LAUGHING') return { text: 'LAUGHING', level: 'ok' };
    if (m.state === 'SAD') return { text: 'SAD', level: 'warn' };
    if (m.state === 'RETURNING') return { text: 'RETURNING · home', level: 'active' };
    if (snapshot.state === 'LOCKED') return { text: 'IDLE · locked (FIST or L to unlock)', level: 'warn' };
    if (snapshot.state === 'ERROR') return { text: 'IDLE · stopped (error)', level: 'bad' };
    return { text: 'IDLE', level: '' };
  }

  /** Face detector status and whether a face is currently seen. */
  /**
   * Status shown on the robot's plate, in the reference video's vocabulary:
   * IDLE [awaiting instruction] / RUNNING IMPLEMENTATION [executing] (walking) /
   * SPINNING UP NEW IDEA [generating] (spin) / HEAVY LIFTING [crunching the task] (pinch lift),
   * plus the app's own states; second line "[ok] hand tracked" / "[--] hand lost".
   */
  _robotStatus(snapshot, latest, live) {
    const m = snapshot.movement;
    const a = snapshot.action?.name;
    const h = snapshot.hold?.name;
    let label = 'IDLE';
    let task = 'awaiting instruction';
    if (snapshot.state === 'ERROR') [label, task] = ['FAULT', 'check camera / engine'];
    else if (snapshot.state === 'LOCKED') [label, task] = ['LOCKED', 'safety lock (L / FIST)'];
    else if (h === 'WALK') [label, task] = ['RUNNING IMPLEMENTATION', 'executing'];
    else if (h === 'RUN') [label, task] = ['SPRINTING', 'running fast'];
    else if (a === 'SQUAT') [label, task] = ['SQUATTING', 'exercise'];
    else if (h === 'SPIN' || m.state === 'ROTATING' && m.source === 'SPIN') [label, task] = ['SPINNING UP NEW IDEA', 'generating'];
    else if (h === 'GRAB_ROTATE') [label, task] = ['HEAVY LIFTING', 'crunching the task'];
    else if (h === 'AIM') [label, task] = ['TARGETING', 'aiming'];
    else if (a === 'WAVE') [label, task] = ['GREETING', 'waving'];
    else if (a === 'LAUGH') [label, task] = ['LAUGHING', 'happy'];
    else if (a === 'SAD') [label, task] = ['FEELING DOWN', 'sad'];
    else if (a === 'SPIN_LEFT' || a === 'SPIN_RIGHT') [label, task] = ['SPINNING', 'turning'];
    else if (a === 'RESET_POSE') [label, task] = ['RETURNING', 'going home'];
    const tracked = live && !!latest?.features;
    return { label, task, line: tracked ? '[ok] hand tracked' : '[--] hand lost', handTracked: tracked };
  }

  _faceView() {
    const status = this.remote ? this.remote.hello?.face?.status ?? this.remote.status?.face?.status ?? null : this.faceTracker?.status ?? 'disabled';
    if (this.camera.status !== 'live') return { text: status === 'ready' ? 'ready (camera off)' : status ?? '—', level: '' };
    if (status && status !== 'ready') return { text: `unavailable (${status})`, level: 'warn' };
    const g = this.gaze;
    if (!g) return { text: '—', level: '' };
    if (!g.faceVisible) return { text: 'lost', level: 'warn' };
    const u = this.user;
    const e = u?.expressionRaw;
    const expr = u?.faceModel && e ? ` · ${e.label.toLowerCase()} ${e.label === 'NEUTRAL' ? '' : e.confidence.toFixed(2)}`.trimEnd() : '';
    const pose = u?.headPose ? ` · head ${Math.round(u.headPose.yawDeg)}°/${Math.round(u.headPose.pitchDeg)}°` : '';
    return { text: `detected${g.faceCount > 1 ? ` · ${g.faceCount} faces` : ''}${expr}${pose}`, level: 'ok' };
  }

  _attentionView() {
    const g = this.gaze;
    if (!g) return { text: '—', level: '' };
    const label = { USER: 'user (face)', HAND: 'hand (aiming)', NEUTRAL: 'neutral' }[g.mode];
    return { text: `${label} · yaw ${Math.round((g.yaw * 180) / Math.PI)}° pitch ${Math.round((g.pitch * 180) / Math.PI)}°`, level: g.mode === 'USER' ? 'ok' : '' };
  }

  /** Live, JSON-safe state for the read-only app.status task (what the agent may read). */
  agentStatus() {
    const now = performance.now();
    const s = this.controller.snapshot(now);
    const latest = this.camera.status === 'live' ? this.pipeline.latest : null;
    return {
      visionEngine: this.mode,
      camera: this.camera.status,
      tracking: this._trackingStatus(latest),
      handsVisible: latest?.hands?.length ?? 0,
      stableGesture: latest?.gesture?.stable?.gesture ?? 'NONE',
      robot: {
        state: s.state,
        gestureLock: s.locked ? 'locked' : 'unlocked',
        yawDegrees: Math.round((s.pose.yaw * 180) / Math.PI),
        runningAction: s.action ? s.action.name : null,
        activeControl: s.hold ? s.hold.name : null,
        lastFinishedCommand: s.lastResult ? { name: s.lastResult.name, outcome: s.lastResult.outcome } : null,
        error: s.error,
      },
    };
  }

  /**
   * Troubleshooting snapshot for app.diagnostics. Every field comes from live
   * state or recorded events; values that cannot be known are null, with a note.
   */
  async agentDiagnostics() {
    const now = performance.now();
    const s = this.controller.snapshot(now);
    const live = this.camera.status === 'live';
    const latest = live ? this.pipeline.latest : null;
    const tracking = this._trackingStatus(latest);
    const known = live && !!latest;
    return {
      visionEngine: this.remote
        ? { mode: 'python', connection: this.remote.status, engineVersion: this.remote.hello?.engine?.version ?? null }
        : { mode: 'browser', connection: 'in-page' },
      camera: { status: this.camera.status, errorCode: this.camera.error?.code ?? null },
      handTracking: {
        status: tracking,
        handDetected: known ? tracking === 'DETECTED' : null,
        stableGesture: known ? latest.gesture?.stable?.gesture ?? null : null,
        note: known ? null : live ? 'no frame processed yet' : 'camera is not live, so hand detection is unknown (this is not a hand-tracking failure)',
      },
      handModel: { status: this.tracker.status, error: sanitizeMessage(this.tracker.error) },
      robot: this.agentStatus().robot,
      blockers: deriveBlockers(s, this.camera.status),
      agentProvider: await this.agent.providerHealth(),
      recentIssues: this.issues.recent(5),
    };
  }

  /** DETECTED / NOT_DETECTED / TRACKING_LOST / CAMERA_UNAVAILABLE / PROCESSING_ERROR */
  _trackingStatus(latest) {
    if (this.camera.status !== 'live') return 'CAMERA_UNAVAILABLE';
    if (!latest) return 'NOT_DETECTED';
    if (this.remote) return latest.status;
    return latest.features ? 'DETECTED' : 'NOT_DETECTED';
  }

  /** Agent state for the HUD: phase / workflow step, and availability (no confidence or scores). */
  _agentView() {
    const a = this.agent;
    const avail = a.availability;
    const wf = [...a.messages].reverse().find((m) => m.role === 'workflow' && m.status === 'executing');
    const step = wf?.steps?.find((s) => s.status === 'executing' || s.status === 'awaiting_confirmation');
    const phaseText = { unavailable: 'unavailable', idle: 'idle', planning: 'planning (model)', awaiting_confirmation: 'awaiting your confirmation', executing: 'executing' }[a.phase] ?? a.phase;
    return {
      phase: step ? `${phaseText}: ${step.id}` : phaseText,
      phaseLevel: a.phase === 'awaiting_confirmation' ? 'warn' : a.phase === 'executing' || a.phase === 'planning' ? 'active' : '',
      system: avail.available ? `ready · ${avail.model ?? ''}` : `unavailable${avail.reason ? ` · ${avail.reason}` : ''}`,
      systemLevel: avail.available ? 'ok' : 'warn',
    };
  }

  _engineView() {
    if (!this.remote) return { detail: 'browser · MediaPipe WASM', level: 'ok' };
    const r = this.remote;
    const h = r.hello;
    if (r.status !== 'connected' || !h) {
      const retry = r.nextRetryMs ? ` · retry ≤${(r.nextRetryMs / 1000).toFixed(1)}s` : '';
      return { detail: `python · ${r.status}${retry}`, level: 'bad' };
    }
    const n = r.stats.interruptions;
    const rec = n && r.outage.lastDurationMs !== null ? ` · ${n} outage(s), back in ${(r.outage.lastDurationMs / 1000).toFixed(1)}s` : '';
    return { detail: `python ${h.engine.version} · mediapipe ${h.engine.mediapipe}${rec}`, level: n ? 'warn' : 'ok' };
  }

  /** Python mode: engine JPEG encode / browser decode (means) and size. Browser mode: not applicable. */
  _previewView(cam) {
    if (!this.remote) return 'n/a (browser engine)';
    if (cam.status !== 'live') return '—';
    const enc = this.pipeline.metrics.encodeMs.summary().mean;
    const dec = cam.decodeMs.summary().mean;
    const kb = cam.previewBytes.summary().mean;
    if (enc === null && dec === null) return '—';
    const f = (v) => (v === null ? '—' : v.toFixed(1));
    return `${f(enc)} / ${f(dec)} ms · ${kb === null ? '—' : (kb / 1024).toFixed(0)} kB`;
  }

  _wireEvents() {
    const log = (t, l) => this.hud.log(t, l);
    const on = (type, fn) => this._disposers.push(this.bus.on(type, fn));
    on('camera', (e) => {
      if (e.status === 'live') {
        const label = e.info.label.length > 64 ? `…${e.info.label.slice(-63)}` : e.info.label;
        log(`Camera live: ${label} ${e.info.width}×${e.info.height}`, 'ok');
      }
      else if (e.status === 'stopped') log('Camera stopped and released', 'info');
      else if (e.status === 'error') {
        log(`Camera error: ${e.error.message}`, 'bad');
        if (e.error.code === 'INTERRUPTED' || e.error.interrupted) {
          this.controller.fail(e.error.message, performance.now());
          this.onCameraInterrupted?.(e.error);
        }
      }
    });
    on('tracker', (e) => {
      if (e.status === 'ready') log(`Hand model ready (${e.delegate})`, 'ok');
      else if (e.status === 'error') {
        log(`Hand model failed: ${e.error}`, 'bad');
        this.controller.fail(`hand model unavailable: ${e.error}`, performance.now());
      } else if (e.status === 'detect_error') log(`Inference error: ${e.error}`, 'warn');
      else if (e.status === 'recovered') log('Inference recovered', 'ok');
    });
    on('tracking', (e) => {
      if (e.status === 'acquired') log(`Hand acquired${e.handedness ? ` (${e.handedness})` : ''}`, 'info');
      else log('Hand tracking lost', 'warn');
    });
    on('gesture', (e) => {
      if (e.phase === 'start') log(`Gesture ${e.gesture}${typeof e.confidence === 'number' ? ` (score ${e.confidence.toFixed(2)})` : ''}`, 'info');
      else if (e.phase === 'event') log(`Gesture ${e.gesture}`, 'info');
    });
    on('command', (e) => {
      if (e.status === 'accepted') log(`▶ ${e.name} accepted (${e.source})`, 'active');
      else if (e.status === 'completed') log(`✓ ${e.name} animation complete`, 'ok');
      else if (e.status === 'ended') log(`■ ${e.name} ended (${e.reason})`, 'info');
      else if (e.status === 'rejected') log(`✕ ${e.name} rejected: ${e.reason}${e.detail ? ` — ${e.detail}` : ''}`, e.reason === 'UNSUPPORTED' || e.reason === 'INVALID' ? 'bad' : 'warn');
    });
    on('state', (e) => {
      if (e.to === 'LOCKED') log('Commands locked (FIST again or L to unlock)', 'warn');
      else if (e.from === 'LOCKED') log('Commands unlocked', 'ok');
      else if (e.to === 'ERROR') log(`Robot entered ERROR: ${e.reason}`, 'bad');
      else if (e.from === 'ERROR') log('Recovered from ERROR', 'ok');
    });
    on('pipeline_error', (e) => log(`Pipeline error: ${e.message}`, 'bad'));
    // Recent operational issues for app.diagnostics (real events only).
    on('camera', (e) => e.status === 'error' && this.issues.record('camera', e.error.code, e.error.message));
    on('tracker', (e) => (e.status === 'error' || e.status === 'detect_error') && this.issues.record('hand_model', e.status, e.error));
    on('command', (e) => e.status === 'rejected' && this.issues.record('command', e.reason, `${e.name} from ${e.source} rejected: ${e.detail ?? e.reason}`));
    on('task', ({ record: r }) => (r.status === 'failed' || r.status === 'rejected') && this.issues.record('task', r.error?.code, `${r.type ?? '?'} ${r.status}: ${r.error?.message ?? ''}`));
    on('gesture_rejected', (e) => this.issues.record('gesture', 'INVALID_EVENT', e.reason));
    on('pipeline_error', (e) => this.issues.record('pipeline', 'PIPELINE_ERROR', e.message));
    on('gesture_rejected', (e) => log(`Invalid gesture event ignored: ${e.reason}`, 'warn'));
    on('task', ({ record: r }) => {
      if (r.status === 'succeeded') log(`Task ${r.type} succeeded (${r.origin})`, 'ok');
      else if (r.status === 'failed' || r.status === 'rejected') log(`Task ${r.type ?? '?'} ${r.status}: ${r.error.code}`, 'warn');
    });
  }

  /** Read-only state for debugging and automated integration checks. */
  snapshot() {
    const now = performance.now();
    const latest = this.pipeline.latest;
    return {
      engine: this.remote
        ? { mode: 'python', connection: this.remote.status, hello: this.remote.hello?.engine ?? null, stats: { ...this.remote.stats },
          outage: { ...this.remote.outage }, metrics: this.remote.metrics, trackingStatus: latest?.status ?? null, lastError: this.remote.lastError,
          preview: { ...this.camera.stats, encodeMs: this.pipeline.metrics.encodeMs.summary(), decodeMs: this.camera.decodeMs.summary(),
            bytes: this.camera.previewBytes.summary() } }
        : { mode: 'browser' },
      camera: { status: this.camera.status, error: this.camera.error, info: this.camera.info, stats: { ...this.camera.stats } },
      tracker: { status: this.tracker.status, delegate: this.tracker.delegate, error: this.tracker.error, stats: { ...this.tracker.stats } },
      robot: this.controller.snapshot(now),
      user: this.user ? { ...this.user } : null,
      status: this.status ? { ...this.status } : null,
      gaze: this.gaze ? { ...this.gaze, yawDeg: Math.round((this.gaze.yaw * 180) / Math.PI), pitchDeg: Math.round((this.gaze.pitch * 180) / Math.PI) } : null,
      poseNet: this.remote ? this.remote.hello?.poseNet?.status ?? null : this.poseNetInfo?.status ?? null,
      face: this.remote ? { engine: 'python', status: this.remote.hello?.face?.status ?? null } : { engine: 'browser', status: this.faceTracker?.status ?? 'disabled', error: this.faceTracker?.error ?? null, inferenceMs: this.faceTracker?.inferenceMs ?? null },
      asset: this.scene ? { ...this.scene.asset } : { error: this.sceneError },
      scene: this.scene ? this.scene.debugInfo() : null,
      handPx: latest?.features
        ? (() => {
          const m = this._mapping();
          return { palm: m.displayToScreen(latest.features.palmCenterNorm), indexTip: m.displayToScreen(latest.features.indexTipNorm) };
        })()
        : null,
      latest: latest
        ? {
          t: latest.t,
          hands: latest.hands.length,
          stable: latest.gesture.stable.gesture,
          raw: latest.gesture.raw.label,
          source: latest.gesture.raw.source ?? null,
          model: latest.primary?.obs?.model ? { label: latest.primary.obs.model.label, score: latest.primary.obs.model.score } : null,
          confidence: latest.gesture.raw.confidence,
          reason: latest.gesture.raw.reason,
          scores: latest.gesture.raw.scores,
          features: latest.features
            ? {
              extension: Object.fromEntries(Object.entries(latest.features.fingers).map(([k, v]) => [k, +v.extension.toFixed(3)])),
              straightness: Object.fromEntries(Object.entries(latest.features.fingers).map(([k, v]) => [k, +v.straightness.toFixed(3)])),
              thumbAbduction: +latest.features.fingers.thumb.abduction.toFixed(3),
              pinchDistance: +latest.features.pinchDistance.toFixed(3),
              source: latest.features.source,
            }
            : null,
        }
        : null,
      session: { ...this.pipeline.session, skippedWhileLoading: this.metrics.skippedWhileLoading },
      perf: {
        // Python mode: measured by the engine (same source as the HUD).
        cameraFps: this.remote ? this.remote.metrics?.cameraFps ?? null : this.metrics.camera.rate(now),
        processedFps: this.remote ? this.remote.metrics?.processedFps ?? null : this.pipeline.metrics.processed.rate(now),
        resultsReceivedFps: this.remote ? this.pipeline.metrics.processed.rate(now) : undefined,
        renderFps: this.metrics.render.rate(now),
        inferenceMs: this.pipeline.metrics.inferenceMs.summary(),
        warmupInferenceMs: this.pipeline.metrics.warmupInferenceMs,
        frameTimeSource: this.frameTimeSource ?? null,
        pipelineMs: this.pipeline.metrics.pipelineMs.summary(),
        overlayMs: this.metrics.overlayMs.summary(),
      },
      agent: {
        phase: this.agent.phase,
        available: this.agent.availability.available,
        reason: this.agent.availability.reason,
        model: this.agent.availability.model,
        pending: this.agent.pending ? { label: this.agent.pending.label } : null,
        messages: this.agent.messages.map((m) => ({ role: m.role, text: m.text ?? m.label, status: m.status ?? null, error: m.error ?? null })),
        stats: { ...this.agent.stats },
      },
      tasks: { types: this.tasks.describe(), recent: [...this.tasks.records.values()].slice(-10).map((r) => ({ id: r.id, type: r.type, status: r.status, error: r.error })) },
      events: this.hud.events.slice(0, 50).map((e) => ({ time: e.time.toISOString(), level: e.level, text: e.text })),
      config: this.configInfo,
    };
  }

  /** Download metrics + event log (no images, no landmarks). */
  exportSession() {
    const data = { generatedAt: new Date().toISOString(), note: 'AURA VISION session metrics. Contains no images or landmark data.', ...this.snapshot(), effectiveConfig: this.config };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `aura-session-${Date.now()}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    this.hud.log('Session metrics exported', 'info');
  }

  shutdown() {
    if (!this.running) return;
    this.running = false;
    this.agent.dispose();
    this.camera.stop();
    this.tracker.close();
    this.faceTracker?.close();
    if (this._raf !== null) cancelAnimationFrame(this._raf);
    this.scene?.dispose();
    for (const d of this._disposers.splice(0)) d();
    this.bus.clear();
  }
}
