import { STATIC_GESTURES } from '../config/config.js';
import { GestureStabilizer } from './GestureStabilizer.js';
import { legsPoseScore, StaticGestureClassifier } from './StaticGestureClassifier.js';
import { SpinDetector } from './SpinDetector.js';
import { SwipeDetector } from './SwipeDetector.js';
import { WalkCadenceDetector } from './WalkCadenceDetector.js';

/**
 * GestureRecognitionEngine
 *
 * PINCH_TAP (dynamic event): thumb and index tips touch and part again within
 * pinchTap.maxMs of a stable pinch (a release, never a hand loss). A longer pinch is the
 * GRAB_ROTATE hold (turn / lift) only.
 *
 * Combines, for the primary hand:
 *   per-frame static classification  (StaticGestureClassifier)
 *   motion gating                    (requireStill gestures -> MOVING when fast)
 *   temporal stabilisation           (GestureStabilizer; one start per hold)
 *   dynamic gestures                 (SwipeDetector: palm history)
 *   continuous motion gestures       (WalkCadenceDetector: alternating index/middle
 *                                     "legs"; SpinDetector: index-tip rotation while
 *                                     pointing) -> start/end events + gesture.motion
 *
 * Priority between the two motion gestures is fixed: walking is evaluated first;
 * while one is active the other cannot start (a tie in one frame goes to walking).
 * Swipes are suppressed while pinching, pointing, holding the V, walking or
 * spinning, so those motions never also spin the robot.
 *
 * Output events:
 *   { type: 'start', gesture }  a static gesture became stable
 *   { type: 'end', gesture }    a stable static gesture was released
 *   { type: 'event', gesture }  a dynamic gesture (swipe) occurred
 *   { type: 'start'|'end', gesture: 'SPIN'|'FINGER_WALK' }  a motion gesture began / ended
 * plus `motion: { spin: { active, omega rad/s (+ = clockwise on screen) }, walk: { active, cadence steps/s } }`
 *
 * The engine knows nothing about commands; InteractionController maps
 * events to commands.
 */
export class GestureRecognitionEngine {
  /** @param {object} cfg gestures configuration (validated) */
  constructor(cfg, deps = {}) {
    this.cfg = cfg;
    /** optional project-trained hand-pose net (handPoseNet.js createHandPoseNet) */
    this.net = deps.net ?? null;
    this.classifier = new StaticGestureClassifier(cfg);
    this.swipe = new SwipeDetector(cfg.swipe);
    this.spin = new SpinDetector(cfg.spin);
    this.walk = new WalkCadenceDetector(cfg.walk);
    this.stabilizer = new GestureStabilizer({
      exitMs: cfg.exitMs,
      holdMs: (g) => cfg.definitions[g]?.holdMs ?? 300,
    });
    this.enabled = STATIC_GESTURES.filter((g) => cfg.definitions[g]?.enabled !== false);
    this.suppressStaticUntil = -Infinity;
    this.hadHand = false;
  }

  /**
   * @param {any|null} features FeatureExtractionEngine output for the primary hand, or null
   * @param {number} t frame timestamp (ms)
   */
  update(features, t) {
    if (!features || !features.valid) {
      const events = this.hadHand ? this.stabilizer.release(t) : [];
      const w = this.walk.update(t, null, false);
      const sp = this.spin.update(t, null, 0, false);
      if (w.event) events.push({ type: w.event, gesture: 'FINGER_WALK', t, confidence: null });
      if (sp.event) events.push({ type: sp.event, gesture: 'SPIN', t, confidence: null });
      this.swipe.reset();
      this.hadHand = false;
      return {
        motion: { spin: { active: false, omega: 0 }, walk: { active: false, cadence: 0 } },
        hand: false,
        raw: { gesture: 'NONE', label: 'NONE', confidence: 0, scores: {}, reason: features?.reason ?? 'no hand' },
        stable: { gesture: 'NONE', confidence: 0 },
        speed: 0,
        events,
      };
    }
    this.hadHand = true;

    if (this.net && features.pose) features.net = this.net.predict(features.pose);
    const raw = this.classifier.classify(features, this.enabled);
    const continuous = (g) => g === 'PINCH' || g === 'POINT' || g === 'TWO_FINGERS' || g === 'THREE_FINGERS';
    const noSwipe = continuous(raw.gesture) || continuous(this.stabilizer.stable) || this.walk.active || this.spin.active;
    const swipe = this.swipe.update(t, features.palmCenter, features.palmSizeImage, { suppressed: noSwipe });
    const speed = this.swipe.speed();
    const w = this.walk.update(t, features.legs, legsPoseScore(features) >= 0.5, this.spin.active);
    const sp = this.spin.update(t, features.indexTip, features.palmSizeImage, raw.gesture === 'POINT', w.active);

    let label = raw.gesture;
    if (this.cfg.definitions[label]?.requireStill && speed > this.cfg.maxStillSpeed) label = 'MOVING';
    if (t < this.suppressStaticUntil) label = 'NONE';

    const events = [];
    if (swipe) {
      events.push({ type: 'event', gesture: swipe.gesture, t, confidence: null, detail: swipe });
      this.suppressStaticUntil = t + this.cfg.postSwipeSuppressMs;
      label = 'NONE';
    }
    if (w.event) events.push({ type: w.event, gesture: 'FINGER_WALK', t, confidence: null });
    if (sp.event) events.push({ type: sp.event, gesture: 'SPIN', t, confidence: null });
    const stab = this.stabilizer.update(label, raw.confidence, t);
    events.push(...stab);
    // pinch tap: a stable pinch released (not lost: hand loss goes through release()) quickly
    const tap = this.cfg.pinchTap ?? {};
    for (const e of stab) {
      if (e.type === 'end' && e.gesture === 'PINCH' && tap.enabled !== false && e.durationMs <= (tap.maxMs ?? 600)) {
        events.push({ type: 'event', gesture: 'PINCH_TAP', t, confidence: null, detail: { durationMs: e.durationMs } });
      }
    }

    return {
      motion: { spin: { active: sp.active, omega: sp.omega }, walk: { active: w.active, cadence: w.cadence } },
      hand: true,
      raw: { ...raw, label },
      stable: { gesture: this.stabilizer.stable, confidence: this.stabilizer.stableConfidence },
      speed,
      events,
    };
  }

  reset() {
    this.stabilizer.reset();
    this.swipe.reset();
    this.spin.reset();
    this.walk.reset();
    this.hadHand = false;
    this.suppressStaticUntil = -Infinity;
  }
}
