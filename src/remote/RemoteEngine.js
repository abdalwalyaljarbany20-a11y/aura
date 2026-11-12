import { RateMeter, RollingStats } from '../core/metrics.js';

/**
 * Adapters that let the existing coordinator, effects, 3D scene and HUD run on
 * results from the Python engine. In this mode the browser never calls
 * getUserMedia: the Python engine is the only camera owner, and gesture
 * decisions come only from Python (the JS gesture engine is not used).
 */

const CAMERA_MESSAGES = {
  ENGINE_UNREACHABLE: 'Python engine is not reachable',
  ENGINE_DISCONNECTED: 'Python engine disconnected',
};

/**
 * Camera interface (same shape as CameraManager) backed by the Python engine.
 *
 * Two delivery paths with different guarantees (docs/PYTHON_ENGINE.md):
 *   state/events  every accepted result is handed to the pipeline immediately and
 *                 in order, so gesture events are never delayed or lost because a
 *                 preview frame is late or missing.
 *   visual        a preview frame is drawn only if it belongs to the current
 *                 session and is the frame of the newest accepted result; the
 *                 pipeline then pairs it with exactly that result. Late, foreign or
 *                 superseded previews are dropped and counted.
 * Results from another session, or whose frameId is not newer than the last
 * accepted one, are rejected as stale.
 */
export class RemoteCamera {
  /**
   * @param {import('./PythonEngineClient.js').PythonEngineClient} client
   * @param {{ preview?: HTMLCanvasElement|null, bus?: any, decode?: (blob: Blob) => Promise<any>, now?: () => number }} deps
   */
  constructor(client, deps = {}) {
    this.client = client;
    this.preview = deps.preview ?? null;
    this.ctx = this.preview?.getContext?.('2d') ?? null;
    this.bus = deps.bus ?? null;
    this.decode = deps.decode ?? ((blob) => createImageBitmap(blob));
    this.now = deps.now ?? (() => performance.now());
    /** @type {'stopped'|'starting'|'live'|'error'} */
    this.status = 'stopped';
    this.error = null;
    this.info = null;
    this.session = null;
    this.stats = { presented: 0, dropped: 0, staleResults: 0, previewsDrawn: 0, previewsStale: 0, previewsMissed: 0, previewErrors: 0 };
    this.decodeMs = new RollingStats(120);
    this.previewBytes = new RollingStats(120);
    this.frameSize = null;
    this._onFrame = null;
    this._onVisual = null;
    this._resetSession(null);
    client.on('result', (r) => this._onResult(r));
    client.on('preview', (p) => this._onPreview(p));
    client.on('status', (s) => this._onEngineStatus(s));
    client.on('connection', (c) => {
      if (c.status === 'disconnected' && (this.status === 'live' || this.status === 'starting')) {
        this._fail({ code: 'ENGINE_DISCONNECTED', message: `${CAMERA_MESSAGES.ENGINE_DISCONNECTED}: ${c.reason}`, interrupted: true });
      }
    });
  }

  get aspect() {
    return this.frameSize ? this.frameSize.width / this.frameSize.height : 16 / 9;
  }

  async start() {
    this.status = 'starting';
    this.error = null;
    this._resetSession(null);
    const r = await this.client.startCamera();
    if (!r.ok) {
      this._fail({ ...r.error, message: r.error?.message ?? CAMERA_MESSAGES.ENGINE_UNREACHABLE });
      return { ok: false, error: this.error };
    }
    this._resetSession(r.session ?? this.client.engineStatus?.session ?? null);
    const cam = this.client.engineStatus?.camera;
    this.info = {
      // A TEST-FILE source (automated checks) is labelled as such, never as a camera.
      label: cam?.info?.source && cam.info.source !== 'camera'
        ? `Python engine · ${cam.info.source}`
        : `Python engine · camera ${cam?.info?.index ?? '?'} (${cam?.info?.backend ?? 'OpenCV'})`,
      deviceId: null,
      width: cam?.info?.width ?? null,
      height: cam?.info?.height ?? null,
    };
    this.status = 'live';
    if (this.preview) this.preview.hidden = false;
    this.bus?.emit('camera', { status: 'live', info: this.info });
    return { ok: true };
  }

  stop() {
    const wasActive = this.status === 'live' || this.status === 'starting';
    this.client.stopCamera();
    this.stopFrameLoop();
    this._resetSession(null);
    if (this.ctx) this.ctx.clearRect(0, 0, this.preview.width, this.preview.height);
    if (wasActive) {
      this.status = 'stopped';
      this.bus?.emit('camera', { status: 'stopped' });
    }
  }

  async listDevices() {
    return []; // camera selection is done in the Python engine (--camera / backend.cameraIndex)
  }

  /**
   * @param {(f: { result: any, t: number }) => void} onFrame  every accepted result, immediately
   * @param {(f: { session: number, frameId: number, t: number }) => void} [onVisual]  its preview was drawn
   */
  startFrameLoop(onFrame, onVisual) {
    this._onFrame = onFrame;
    this._onVisual = onVisual ?? null;
  }

  stopFrameLoop() {
    this._onFrame = null;
    this._onVisual = null;
  }

  _resetSession(session) {
    this.session = session;
    this._lastFrameId = 0;
    this._awaitingPreview = null; // frameId of the newest accepted result
    this._lastDrawn = 0;
  }

  _onEngineStatus(s) {
    const cam = s.camera;
    if (cam?.status === 'error' && this.status === 'live') {
      this._fail({ ...cam.error, interrupted: true });
    }
  }

  _onResult(r) {
    if (this.status !== 'live') return;
    if (this.session !== null && r.session !== this.session) {
      this.stats.staleResults++; // from an earlier camera session
      return;
    }
    if (r.frameId !== null && r.frameId !== undefined) {
      if (r.frameId <= this._lastFrameId) {
        this.stats.staleResults++; // an older frame never overrides a newer one
        return;
      }
      if (this._awaitingPreview !== null) this.stats.previewsMissed++; // previous result's preview never came
      this._lastFrameId = r.frameId;
      this._awaitingPreview = r.frameId;
    }
    if (r.image?.width) this.frameSize = { width: r.image.width, height: r.image.height };
    this.stats.presented++;
    this._onFrame?.({ result: r, t: this.now() });
  }

  async _onPreview({ session, frameId, jpeg }) {
    if (this.status !== 'live' || session !== this.session || frameId !== this._awaitingPreview) {
      this.stats.previewsStale++;
      return;
    }
    this._awaitingPreview = null;
    this.previewBytes.push(jpeg.byteLength);
    const t0 = this.now();
    let bmp;
    try {
      bmp = await this.decode(new Blob([jpeg], { type: 'image/jpeg' }));
    } catch {
      this.stats.previewErrors++;
      return;
    }
    this.decodeMs.push(this.now() - t0);
    // A newer frame may have been drawn while this one decoded; never go backwards.
    if (this.status !== 'live' || session !== this.session || frameId <= this._lastDrawn) {
      this.stats.previewsStale++;
      bmp.close?.();
      return;
    }
    if (this.ctx) {
      if (this.preview.width !== bmp.width || this.preview.height !== bmp.height) {
        this.preview.width = bmp.width;
        this.preview.height = bmp.height;
      }
      this.ctx.drawImage(bmp, 0, 0);
    }
    bmp.close?.();
    this._lastDrawn = frameId;
    this.stats.previewsDrawn++;
    this._onVisual?.({ session, frameId, t: this.now() });
  }

  _fail(error) {
    this.error = { code: error.code ?? 'UNKNOWN', message: error.message ?? 'camera error', detail: error.detail ?? null };
    this.status = 'error';
    this._resetSession(null);
    this.bus?.emit('camera', { status: 'error', error: { ...this.error, code: error.interrupted ? 'INTERRUPTED' : this.error.code } });
  }
}

/** Tracker-like status view of the Python engine (the model runs in Python). */
export class RemoteTracker {
  constructor(client, deps = {}) {
    this.client = client;
    this.bus = deps.bus ?? null;
    this.status = 'loading';
    this.delegate = null;
    this.error = null;
    this.stats = {};
    client.on('status', (s) => this._apply(s.tracker));
  }

  async init() {
    try {
      const hello = await this.client.connect();
      this._apply(hello.tracker);
      return { ok: this.status === 'ready' };
    } catch (err) {
      this.status = 'error';
      this.error = err.message;
      this.bus?.emit('tracker', { status: 'error', error: this.error });
      return { ok: false, error: this.error };
    }
  }

  _apply(t) {
    if (!t) return;
    const prev = this.status;
    if (t.status === 'ready') {
      this.status = 'ready';
      this.delegate = `Python · ${t.delegate ?? '?'}`;
      this.error = null;
      if (prev !== 'ready') this.bus?.emit('tracker', { status: 'ready', delegate: this.delegate });
    } else if (t.status === 'error') {
      this.status = 'error';
      this.error = t.error;
      if (prev !== 'error') this.bus?.emit('tracker', { status: 'error', error: t.error });
    } else {
      this.status = t.status === 'uninitialized' ? 'loading' : t.status;
    }
  }

  close() {
    this.client.close();
    this.status = 'closed';
  }
}

/**
 * Turns Python `result` messages into the `latest` structure the renderer uses
 * and drives the InteractionController with Python's gesture events.
 * The JS FeatureExtractionEngine is used only for *render geometry* (palm
 * centre, arcs, aim target), never for gesture decisions.
 *
 * `latest` is the newest state (HUD, gestures, commands); `visualLatest(now)`
 * is what may be drawn over the preview currently on screen.
 */
export class RemoteFramePipeline {
  /** @param {{ extractor: any, controller: any, bus?: any, now?: () => number, maxVisualLagMs?: number }} deps */
  constructor(deps) {
    this.extractor = deps.extractor;
    this.controller = deps.controller;
    this.bus = deps.bus ?? null;
    this.now = deps.now ?? (() => performance.now());
    // How long the overlay may keep showing the last frame-matched pair while the
    // newest result's preview has not been drawn yet; after that it is hidden.
    this.maxVisualLagMs = deps.maxVisualLagMs ?? 250;
    this.latest = null;
    this.visual = null; // { entry, shownAt }
    this._recent = new Map(); // frameId -> entry (bounded)
    this.recognizer = { reset: () => {} }; // gesture state lives in Python
    this.metrics = {
      processed: new RateMeter(1000), // results received by this page
      inferenceMs: new RollingStats(120),
      pipelineMs: new RollingStats(120),
      encodeMs: new RollingStats(120),
      warmupInferenceMs: null,
    };
    this.session = { frames: 0, framesWithHand: 0, trackingLosses: 0, detectFailures: 0, gestureStarts: 0, swipes: 0, malformed: 0 };
    this.trackingStatus = 'CAMERA_UNAVAILABLE';
    this._hadHand = false;
  }

  /** @param {any} r validated result message (receivedAt set by the client) */
  process(r) {
    const t = this.now(); // controller timestamps use the browser clock only
    const aspect = r.image?.width ? r.image.width / r.image.height : 16 / 9;
    const hands = r.hands.map((h) => {
      const obs = {
        landmarks: h.landmarks.map(([x, y, z]) => ({ x, y, z })),
        worldLandmarks: h.worldLandmarks ? h.worldLandmarks.map(([x, y, z]) => ({ x, y, z })) : null,
        handedness: h.handedness ?? null,
        score: h.score ?? null,
      };
      return { obs, features: this.extractor.extract(obs, { aspect }) };
    });
    const primary = r.primary !== null && r.primary !== undefined ? hands[r.primary] ?? null : null;
    const features = r.status === 'DETECTED' && primary?.features?.valid ? primary.features : null;
    const g = r.gesture;
    const gesture = {
      hand: r.status === 'DETECTED',
      raw: { gesture: g.raw, label: g.label, confidence: g.confidence ?? 0, scores: g.scores ?? {}, reason: g.reason ?? (r.error ? r.error : null), source: typeof g.source === 'string' ? g.source : null },
      stable: { gesture: g.stable, confidence: g.stableConfidence ?? 0 },
      events: r.events,
      motion: g.motion ?? null,
    };

    const hasHand = !!features;
    this.controller.setHandPresent(hasHand, t);
    const commandResults = this.controller.handleGestureEvents(r.events, t, { features });
    this.controller.updateHold(features, gesture.motion);
    this.controller.tick(t);

    if (r.status === 'PROCESSING_ERROR') {
      this.session.detectFailures++;
      this.bus?.emit('tracker', { status: 'detect_error', error: r.error });
    }
    this.trackingStatus = r.status;
    this.session.frames++;
    if (hasHand) this.session.framesWithHand++;
    if (this._hadHand && !hasHand) {
      this.session.trackingLosses++;
      this.bus?.emit('tracking', { status: 'lost' });
    } else if (!this._hadHand && hasHand) {
      this.bus?.emit('tracking', { status: 'acquired', handedness: features.handedness });
    }
    this._hadHand = hasHand;
    for (const ev of r.events) {
      if (ev.type === 'start') this.session.gestureStarts++;
      if (ev.type === 'event') this.session.swipes++;
      this.bus?.emit('gesture', { phase: ev.type, gesture: ev.gesture, confidence: ev.confidence ?? null });
    }

    const tm = r.timing;
    const warmup = !!tm?.warmup;
    if (tm && !warmup && r.status !== 'PROCESSING_ERROR') {
      this.metrics.inferenceMs.push(tm.inferenceMs);
      this.metrics.pipelineMs.push(tm.totalMs);
    } else if (warmup) this.metrics.warmupInferenceMs = tm.inferenceMs;
    if (tm?.previewEncodeMs !== undefined && tm?.previewEncodeMs !== null) this.metrics.encodeMs.push(tm.previewEncodeMs);
    this.metrics.processed.tick(t);

    // Capture time expressed on the browser clock: receive time minus the engine's
    // own capture->send age. Localhost WebSocket transit is not included.
    const capturedAt = r.receivedAt !== undefined && r.ageAtSendMs !== undefined ? r.receivedAt - r.ageAtSendMs : t;
    this.latest = {
      t,
      capturedAt,
      processedAt: t,
      warmup,
      aspect,
      hands: r.status === 'DETECTED' ? hands : [],
      primary: features ? primary : null,
      features,
      gesture,
      commandResults,
      inferenceMs: tm?.inferenceMs ?? null,
      status: r.status,
      error: r.error ?? null,
      frameId: r.frameId,
      session: r.session,
    };
    if (r.frameId !== null && r.frameId !== undefined) {
      this._recent.set(r.frameId, this.latest);
      while (this._recent.size > 8) this._recent.delete(this._recent.keys().next().value);
    }
    return this.latest;
  }

  /** The preview of `frameId` is on screen: its own result becomes the visual pair. */
  commitVisual(frameId, t = this.now()) {
    const entry = this._recent.get(frameId);
    if (entry && entry.session === this.latest?.session) this.visual = { entry, shownAt: t };
  }

  /**
   * What the overlay may draw now: only landmarks computed from the frame on
   * screen, and none at all when the newest result has no hand.
   */
  visualLatest(now = this.now()) {
    const latest = this.latest;
    if (!latest) return null;
    const empty = { ...latest, hands: [], primary: null, features: null };
    const v = this.visual;
    if (latest.status !== 'DETECTED' || !v || v.entry.session !== latest.session || v.entry.status !== 'DETECTED') return empty;
    if (v.entry.frameId === latest.frameId || now - v.shownAt <= this.maxVisualLagMs) return v.entry;
    return empty;
  }

  reset() {
    this.latest = null;
    this.visual = null;
    this._recent.clear();
    this._hadHand = false;
  }
}
