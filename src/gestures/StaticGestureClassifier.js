import { ramp } from './FeatureExtractionEngine.js';

/**
 * Deterministic baseline classifier for static hand shapes.
 *
 * Each gesture has a rule score in [0, 1] built from fuzzy memberships
 * (linear ramps) over normalised features. Scores are *rule memberships*,
 * not calibrated probabilities. The winner must reach `minScore` and beat
 * the runner-up by `minMargin`; otherwise the result is UNKNOWN.
 *
 * Rules (e_x = extension of finger x in [0,1], pc = pinch closeness):
 *   pc        = ramp(pinchDistance, 0.38 -> 0.22)          (1 = tips touching)
 *   OPEN_PALM = ramp(min(e_index, e_middle, e_ring, e_pinky), 0.6 -> 0.9) * (1 - pc)
 *   FIST      = min(1-e_index, 1-e_middle, 1-e_ring, 1-e_pinky) * (1 - ramp(e_thumb, 0.7 -> 0.95))
 *   POINT     = min(e_index, 1-e_middle, 1-e_ring, 1-e_pinky) * (1 - pc)
 *   PINCH     = pc * ramp(e_index, 0.1 -> 0.35)
 *   TWO_FINGERS = ramp(min(e_index, e_middle), 0.6 -> 0.9) * ramp(max(e_ring, e_pinky), 0.9 -> 0.7)
 *                 * (1 - pc) * (1 - ramp(e_thumb, 0.75 -> 1.0))
 *   THREE_FINGERS = ramp(min(e_index, e_middle, e_ring), 0.6 -> 0.9) * ramp(e_pinky, 0.9 -> 0.7)
 *                 * (1 - pc) * (1 - ramp(e_thumb, 0.75 -> 1.0))      (run; excludes the V and the open palm)
 *
 * TWO_FINGERS (walk) is decided from joint geometry (each finger's straightness),
 * not from the fingertip spread: index and middle clearly straight (the same 0.6 -> 0.9
 * membership as OPEN_PALM, because this gesture moves the robot), ring and pinky folded,
 * thumb not clearly extended (so a "three" with the thumb out is not a walk).
 * Calibration (real MediaPipe output, docs/VALIDATION.md): in a frontal V sign the
 * folded ring and pinky are foreshortened and measured extension 0.43 / 0.66, so
 * "folded" means "not clearly straight" (max <= 0.7 full, >= 0.9 none) rather than
 * 1 - e; a straight ring or pinky (three fingers, open palm) still scores 0.
 * POINT needs a curled middle finger and TWO_FINGERS a straight one, so the two
 * shapes exclude each other; a half-bent middle finger is ambiguous (UNKNOWN).
 *
 * PINCH requires the index finger to be at least partly extended so that a
 * fist (where thumb and index tips may also be close) is not read as PINCH.
 *
 * Calibration notes (from real MediaPipe output on sample photos, see
 * docs/VALIDATION.md): OPEN_PALM does not require an abducted thumb, because
 * a flat hand with the thumb alongside the index finger is a common open-palm
 * posture (thumb extension measured 0.0); instead all four fingers must be
 * clearly straight (real open hand: min extension 0.965), which keeps a
 * half-curled "claw" (~0.67) out. FIST rejects a clearly extended
 * thumb (thumbs-up measured 1.0; a tucked thumb measured 0.45).
 *
 *   THUMB_UP    = fold4 * ramp(e_thumb, 0.7 -> 0.95) * ramp(-thumbDir.y, 0.5 -> 0.8) * (1 - pc)
 *   THUMB_DOWN  = fold4 * ramp(e_thumb, 0.7 -> 0.95) * ramp( thumbDir.y, 0.5 -> 0.8) * (1 - pc)
 *   fold4       = ramp(max(e_index, e_middle, e_ring, e_pinky), 0.95 -> 0.75)  (calibrated: a real
 *                 side-on thumbs-up photo measured curled ring 0.82, index 0.58)
 *
 * The thumb gestures (emotions) need a clearly extended thumb, so they exclude FIST
 * (which needs a folded thumb: the safety lock keeps its meaning); a half-extended
 * thumb scores in between and is UNKNOWN. thumbDir is the thumb's MCP -> tip direction
 * in display space (y down), so "up" means up on screen. A sideways thumb is UNKNOWN.
 *
 * This class is stateless; it can be replaced by a learned classifier that
 * implements the same `classify(features, enabled)` contract.
 */

export const PINCH_RANGE = Object.freeze({ open: 0.38, closed: 0.22 });

/**
 * Deep-learning gesture classes (MediaPipe GestureRecognizer canned model) -> app gestures.
 * ILoveYou and None have no app gesture.
 */
export const MODEL_LABELS = Object.freeze({
  Open_Palm: 'OPEN_PALM',
  Closed_Fist: 'FIST',
  Pointing_Up: 'POINT',
  Thumb_Up: 'THUMB_UP',
  Thumb_Down: 'THUMB_DOWN',
  Victory: 'TWO_FINGERS',
});
/** Shapes the rules may still report when the model's top class is None / below threshold. */
const RULE_ASSIST = new Set(['POINT', 'TWO_FINGERS', 'THREE_FINGERS']);
/**
 * Shapes the project-trained hand-pose network (handPoseNet.js) may report when the canned
 * model says None / is below threshold. FIST (the safety lock) and the emotion thumbs stay
 * with the canned model + geometric agreement.
 */
export const NET_ASSIST = Object.freeze(new Set(['POINT', 'TWO_FINGERS', 'THREE_FINGERS', 'OPEN_PALM']));
/**
 * The canned model has no three-finger class: a three-finger hand is reported by it as Victory,
 * Open_Palm or None. Those labels may be overridden by THREE_FINGERS (net or rule).
 */
const THREE_OVERRIDABLE = new Set([undefined, 'TWO_FINGERS', 'OPEN_PALM']);
/**
 * Geometric agreement required before a model class is accepted (false-activation guard):
 * the emotion thumbs need folded fingers + an extended thumb (a flat palm-down hand was
 * measured as Thumb_Down 0.77 on the reference video; its fingers are straight, rule 0).
 */
export const MODEL_AGREEMENT = Object.freeze({ THUMB_UP: 0.2, THUMB_DOWN: 0.2 });

export class StaticGestureClassifier {
  /**
   * @param {{ minScore?: number, minMargin?: number, minTrackingScore?: number, minInFrameRatio?: number }} [opts]
   */
  constructor(opts = {}) {
    this.model = { enabled: opts.model?.enabled ?? true, minScore: opts.model?.minScore ?? 0.6 };
    this.net = { enabled: opts.net?.enabled ?? true, minScore: opts.net?.minScore ?? 0.7 };
    this.minScore = opts.minScore ?? 0.6;
    this.minMargin = opts.minMargin ?? 0.2;
    this.minTrackingScore = opts.minTrackingScore ?? 0.6;
    this.minInFrameRatio = opts.minInFrameRatio ?? 0.9;
  }

  /** @param {any} f features from FeatureExtractionEngine */
  scores(f) {
    const e = (name) => f.fingers[name].extension;
    const pc = ramp(f.pinchDistance, PINCH_RANGE.open, PINCH_RANGE.closed);
    const fold4 = ramp(Math.max(e('index'), e('middle'), e('ring'), e('pinky')), 0.95, 0.75);
    const thumb = fold4 * ramp(e('thumb'), 0.7, 0.95) * (1 - pc);
    return {
      OPEN_PALM: ramp(Math.min(e('index'), e('middle'), e('ring'), e('pinky')), 0.6, 0.9) * (1 - pc),
      FIST: Math.min(1 - e('index'), 1 - e('middle'), 1 - e('ring'), 1 - e('pinky')) * (1 - ramp(e('thumb'), 0.7, 0.95)),
      POINT: Math.min(e('index'), 1 - e('middle'), 1 - e('ring'), 1 - e('pinky')) * (1 - pc),
      PINCH: pc * ramp(e('index'), 0.1, 0.35),
      TWO_FINGERS: ramp(Math.min(e('index'), e('middle')), 0.6, 0.9) * ramp(Math.max(e('ring'), e('pinky')), 0.9, 0.7) * (1 - pc) * (1 - ramp(e('thumb'), 0.75, 1.0)),
      THREE_FINGERS: ramp(Math.min(e('index'), e('middle'), e('ring')), 0.6, 0.9) * ramp(e('pinky'), 0.9, 0.7) * (1 - pc) * (1 - ramp(e('thumb'), 0.75, 1.0)),
      THUMB_UP: thumb * ramp(-f.thumbDir.y, 0.5, 0.8),
      THUMB_DOWN: thumb * ramp(f.thumbDir.y, 0.5, 0.8),
    };
  }

  /**
   * @param {any} f features (may be invalid)
   * @param {Set<string>|string[]} [enabled] gestures allowed to win
   * @returns {{ gesture: string, confidence: number, scores: Record<string, number>, reason: string|null }}
   */
  classify(f, enabled) {
    if (!f || !f.valid) return { gesture: 'NONE', confidence: 0, scores: {}, reason: f?.reason ?? 'no hand' };
    if (f.score !== null && f.score !== undefined && f.score < this.minTrackingScore) {
      return { gesture: 'UNKNOWN', confidence: 0, scores: {}, reason: 'low tracking confidence' };
    }
    if (f.inFrameRatio < this.minInFrameRatio) {
      return { gesture: 'UNKNOWN', confidence: 0, scores: {}, reason: 'hand partly outside the image' };
    }

    const all = this.scores(f);
    const allowed = enabled ? new Set(enabled) : null;
    const net = this.net.enabled && f.net && typeof f.net.label === 'string' ? f.net : null;
    if (this.model.enabled && f.model && typeof f.model.label === 'string') return this._fuse(f.model, all, allowed, net);
    const ranked = Object.entries(all)
      .filter(([g]) => !allowed || allowed.has(g))
      .sort((a, b) => b[1] - a[1]);
    if (!ranked.length) return { gesture: 'UNKNOWN', confidence: 0, scores: all, reason: 'no gestures enabled' };

    const [best, bestScore] = ranked[0];
    const second = ranked[1]?.[1] ?? 0;
    if (bestScore < this.minScore) {
      return { gesture: 'UNKNOWN', confidence: bestScore, scores: all, reason: 'no rule matched strongly' };
    }
    if (bestScore - second < this.minMargin) {
      return { gesture: 'UNKNOWN', confidence: bestScore, scores: all, reason: 'ambiguous between gestures' };
    }
    return { gesture: best, confidence: bestScore, scores: all, reason: null, source: 'rules' };
  }

  /**
   * Fusion when the learned gesture model ran for this hand:
   *   1. PINCH from the pinch distance (the model has no pinch class)
   *   2. the model's top class, if it maps to an enabled gesture and scores >= model.minScore
   *      (this includes FIST, the safety lock: only the model decides it); the emotion thumbs
   *      additionally need geometric agreement (MODEL_AGREEMENT)
   *   3. THREE_FINGERS (run), which the canned model has no class for: from the trained
   *      hand-pose net (score >= net.minScore) or, without the net, the rule (>= 0.8), when
   *      the canned model says Victory / Open_Palm / None or is below threshold
   *   4. if the model's top class is None / unmapped / below threshold: the trained net may
   *      report POINT / V / three / open palm (NET_ASSIST), then the rules POINT / V / three
   *      (orientations the canned classes do not cover)
   *   5. otherwise UNKNOWN. The `source` field says which path decided.
   */
  _fuse(model, rules, allowed, net = null) {
    const ok = (g) => !allowed || allowed.has(g);
    const scores = { ...rules };
    for (const [cls, g] of Object.entries(MODEL_LABELS)) if (model.scores?.[cls] !== undefined) scores[g] = model.scores[cls];
    if (ok('PINCH') && rules.PINCH >= this.minScore) return { gesture: 'PINCH', confidence: rules.PINCH, scores, reason: null, source: 'pinch-distance', model, net };
    const g = MODEL_LABELS[model.label];
    const weak = !g || model.score < this.model.minScore;
    if (ok('THREE_FINGERS') && (weak || THREE_OVERRIDABLE.has(g))) {
      if (net && net.label === 'THREE_FINGERS' && net.score >= this.net.minScore) {
        return { gesture: 'THREE_FINGERS', confidence: net.score, scores, reason: null, source: `aura-net (model: ${model.label})`, model, net };
      }
      if (!net && rules.THREE_FINGERS >= Math.max(this.minScore, 0.8)) {
        return { gesture: 'THREE_FINGERS', confidence: rules.THREE_FINGERS, scores, reason: null, source: `rules (model: ${model.label})`, model, net };
      }
    }
    const agrees = !g || !(g in MODEL_AGREEMENT) || (rules.THUMB_UP + rules.THUMB_DOWN) >= MODEL_AGREEMENT[g];
    if (g && ok(g) && model.score >= this.model.minScore && agrees) return { gesture: g, confidence: model.score, scores, reason: null, source: 'model', model, net };
    if (g && !agrees) return { gesture: 'UNKNOWN', confidence: model.score, scores, reason: `model: ${model.label} ${model.score.toFixed(2)} rejected (hand geometry disagrees)`, source: 'model', model, net };
    if (weak) {
      if (net && NET_ASSIST.has(net.label) && ok(net.label) && net.score >= this.net.minScore) {
        return { gesture: net.label, confidence: net.score, scores, reason: null, source: `aura-net (model: ${model.label})`, model, net };
      }
      // rules assist only when the net is absent or unsure (a confident OTHER rejects the pose)
      const netSure = net && net.score >= this.net.minScore;
      const ranked = netSure ? [] : Object.entries(rules).filter(([k]) => RULE_ASSIST.has(k) && ok(k)).sort((x, y) => y[1] - x[1]);
      if (ranked.length && ranked[0][1] >= Math.max(this.minScore, 0.8)) {
        return { gesture: ranked[0][0], confidence: ranked[0][1], scores, reason: null, source: `rules (model: ${model.label})`, model, net };
      }
    }
    return { gesture: 'UNKNOWN', confidence: model.score, scores, reason: `model: ${model.label} ${model.score.toFixed(2)}`, source: 'model', model, net };
  }
}

/**
 * Legs pose for finger walking: ring, pinky and thumb folded, at least one of the
 * two "legs" (index, middle) raised. Bending legs are expected, so their extension
 * is not required to stay high (the static TWO_FINGERS rule would flicker).
 * @returns {number} membership in [0, 1]; >= 0.5 counts as the legs pose
 */
export function legsPoseScore(f) {
  if (!f?.valid) return 0;
  const e = (name) => f.fingers[name].extension;
  return ramp(Math.max(e('ring'), e('pinky')), 0.9, 0.7) * (1 - ramp(e('thumb'), 0.75, 1.0)) * ramp(Math.max(e('index'), e('middle')), 0.3, 0.6);
}
