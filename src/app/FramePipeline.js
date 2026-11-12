import { RateMeter, RollingStats } from '../core/metrics.js';

/**
 * FramePipeline: one call per new camera frame.
 *
 *   frame -> HandTrackingEngine.detect -> landmark validation (in engine)
 *         -> FeatureExtractionEngine (all hands) -> GestureRecognitionEngine (primary)
 *         -> InteractionController (command validation + state machine)
 *
 * Runs synchronously on the frame callback; rendering happens separately
 * in the render loop using `latest`. Because the camera hands out only the
 * newest frame, a slow model lowers the processed FPS but never builds up a
 * backlog.
 */
export class FramePipeline {
  /**
   * @param {{ tracker: any, extractor: any, recognizer: any, controller: any, bus?: any,
   *           now?: () => number, maxConsecutiveFailures?: number }} deps
   */
  constructor(deps) {
    Object.assign(this, {
      tracker: deps.tracker,
      extractor: deps.extractor,
      recognizer: deps.recognizer,
      controller: deps.controller,
      bus: deps.bus ?? null,
    });
    this.now = deps.now ?? (() => performance.now());
    this.maxConsecutiveFailures = deps.maxConsecutiveFailures ?? 30;
    this.latest = null;
    this.metrics = {
      processed: new RateMeter(1000),
      inferenceMs: new RollingStats(120), // steady state; the first call (warm-up) is kept separately
      warmupInferenceMs: null,
      pipelineMs: new RollingStats(120),
    };
    this.session = { frames: 0, framesWithHand: 0, trackingLosses: 0, detectFailures: 0, gestureStarts: 0, swipes: 0 };
    this._consecutiveFailures = 0;
    this._trackerFailed = false;
    this._hadHand = false;
  }

  /**
   * @param {{ video: any, t: number, capturedAt?: number, aspect?: number }} frame
   *        t = frame callback time (ms, performance.now clock; model timestamp)
   *        capturedAt = earliest known frame time, used for latency (defaults to t)
   */
  process(frame) {
    const det = this.tracker.detect(frame.video, frame.t);
    const done = this.now();
    if (!det.ok) {
      this.session.detectFailures++;
      this._consecutiveFailures++;
      if (this._consecutiveFailures === 1) this.bus?.emit('tracker', { status: 'detect_error', error: det.error });
      if (this._consecutiveFailures === this.maxConsecutiveFailures) {
        this.controller.fail(`hand tracking failed ${this.maxConsecutiveFailures} times in a row: ${det.error}`, done);
        this._trackerFailed = true;
      }
      // Treat as "no hand" so gestures and holds are released safely.
      this._consume([], null, frame, det.inferenceMs, false);
      return this.latest;
    }
    this._consecutiveFailures = 0;
    if (this._trackerFailed) {
      this._trackerFailed = false;
      this.controller.recover(done);
      this.bus?.emit('tracker', { status: 'recovered' });
    }
    // The first inference includes GPU/WASM initialisation; it is reported
    // separately and excluded from steady-state latency statistics.
    const warmup = this.metrics.warmupInferenceMs === null;
    if (warmup) this.metrics.warmupInferenceMs = det.inferenceMs;
    else this.metrics.inferenceMs.push(det.inferenceMs);
    return this._consume(det.hands, det.primary, frame, det.inferenceMs, warmup);
  }

  _consume(hands, primary, frame, inferenceMs, warmup) {
    const t = frame.t;
    const aspect = frame.aspect ?? 1;
    const analysed = hands.map((obs) => ({ obs, features: this.extractor.extract(obs, { aspect }) }));
    const primaryEntry = analysed.find((h) => h.obs === primary) ?? null;
    const features = primaryEntry?.features?.valid ? primaryEntry.features : null;

    const gesture = this.recognizer.update(features, t);
    const hasHand = !!features;
    this.controller.setHandPresent(hasHand, t);
    const commandResults = this.controller.handleGestureEvents(gesture.events, t, { features });
    this.controller.updateHold(features, gesture.motion);
    this.controller.tick(t);

    this.session.frames++;
    if (hasHand) this.session.framesWithHand++;
    if (this._hadHand && !hasHand) {
      this.session.trackingLosses++;
      this.bus?.emit('tracking', { status: 'lost' });
    } else if (!this._hadHand && hasHand) {
      this.bus?.emit('tracking', { status: 'acquired', handedness: features.handedness });
    }
    this._hadHand = hasHand;
    for (const ev of gesture.events) {
      if (ev.type === 'start') this.session.gestureStarts++;
      if (ev.type === 'event') this.session.swipes++;
      this.bus?.emit('gesture', { phase: ev.type, gesture: ev.gesture, confidence: ev.confidence ?? null });
    }

    const end = this.now();
    this.metrics.processed.tick(end);
    const capturedAt = frame.capturedAt ?? t;
    if (!warmup) this.metrics.pipelineMs.push(end - capturedAt);
    this.latest = {
      t,
      capturedAt,
      processedAt: end,
      warmup,
      aspect,
      hands: analysed,
      primary: primaryEntry,
      features,
      gesture,
      commandResults,
      inferenceMs,
    };
    return this.latest;
  }
}
