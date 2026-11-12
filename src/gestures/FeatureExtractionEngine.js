import { FINGER_CHAINS, LM, landmarkSetError } from './landmarks.js';
import { poseVector } from '../perception/handPoseNet.js';

/**
 * FeatureExtractionEngine
 *
 * Converts one hand's landmarks into scale-, translation- and
 * rotation-tolerant geometric features:
 *
 *  - palmSize: mean wrist->MCP distance (index, middle, pinky). All shape
 *    distances are divided by it, so features do not depend on how far the
 *    hand is from the camera or on image resolution.
 *  - fingers[f].straightness: |base->tip| / (sum of segment lengths) of the
 *    finger chain; ~1 for a straight finger, ~0.3-0.5 for a curled one.
 *    This ratio is invariant to hand position, scale and in-plane rotation.
 *  - fingers[f].extension: straightness mapped linearly to [0, 1].
 *  - thumb: additionally uses abduction (thumb tip distance from the index
 *    MCP, in palm sizes) because a tucked thumb can still be fairly straight.
 *  - pinchDistance: thumb tip <-> index tip distance in palm sizes.
 *  - legs: index / middle finger length projected on the palm axis (wrist ->
 *    middle MCP), in palm sizes: the two "leg" signals for finger walking.
 *  - thumbDir: unit thumb MCP -> tip direction in display space (y down).
 *  - palmCenter / palmSizeImage: in display image units (aspect-corrected,
 *    optionally mirrored) for motion features such as swipes.
 *
 * Shape features use MediaPipe's metric world landmarks when present (more
 * robust to perspective), otherwise aspect-corrected image landmarks.
 */

const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y, z: (a.z ?? 0) - (b.z ?? 0) });
const len = (v) => Math.hypot(v.x, v.y, v.z);
export const dist = (a, b) => len(sub(a, b));
export const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
/** Linear ramp: 0 at `from`, 1 at `to` (works for descending ramps too). */
export const ramp = (v, from, to) => clamp01((v - from) / (to - from));

export const EXTENSION_RANGE = Object.freeze({ curled: 0.55, straight: 0.92 });
export const THUMB_ABDUCTION_RANGE = Object.freeze({ tucked: 0.45, out: 0.8 });

/**
 * @typedef {{x:number,y:number,z?:number}} Point
 * @typedef {{ landmarks: Point[], worldLandmarks?: Point[]|null, score?: number, handedness?: string }} HandObservation
 */

export class FeatureExtractionEngine {
  /**
   * @param {{ mirror?: boolean }} [opts]
   */
  constructor(opts = {}) {
    this.mirror = opts.mirror ?? true;
  }

  /**
   * @param {HandObservation} hand
   * @param {{ aspect?: number }} [frame] aspect = imageWidth / imageHeight
   * @returns {{ valid: true, [k: string]: any } | { valid: false, reason: string }}
   */
  extract(hand, frame = {}) {
    if (!hand || typeof hand !== 'object') return { valid: false, reason: 'no hand' };
    const err = landmarkSetError(hand.landmarks);
    if (err) return { valid: false, reason: err };
    const aspect = frame.aspect && Number.isFinite(frame.aspect) && frame.aspect > 0 ? frame.aspect : 1;

    const image = hand.landmarks;
    // Display-space image points: x scaled by aspect so distances are
    // isotropic (units of image height); mirrored for a selfie view.
    const disp = image.map((p) => ({ x: (this.mirror ? 1 - p.x : p.x) * aspect, y: p.y, z: (p.z ?? 0) * aspect }));

    const useWorld = !landmarkSetError(hand.worldLandmarks ?? null);
    const shape = useWorld ? hand.worldLandmarks : disp;

    const palmSize =
      (dist(shape[LM.WRIST], shape[LM.INDEX_MCP]) +
        dist(shape[LM.WRIST], shape[LM.MIDDLE_MCP]) +
        dist(shape[LM.WRIST], shape[LM.PINKY_MCP])) / 3;
    if (!(palmSize > 1e-6)) return { valid: false, reason: 'degenerate hand (zero palm size)' };

    const fingers = {};
    for (const [name, chain] of Object.entries(FINGER_CHAINS)) {
      let path = 0;
      for (let i = 1; i < chain.length; i++) path += dist(shape[chain[i - 1]], shape[chain[i]]);
      const straightness = path > 0 ? dist(shape[chain[0]], shape[chain[chain.length - 1]]) / path : 0;
      fingers[name] = {
        straightness,
        extension: ramp(straightness, EXTENSION_RANGE.curled, EXTENSION_RANGE.straight),
      };
    }
    const thumbAbduction = dist(shape[LM.THUMB_TIP], shape[LM.INDEX_MCP]) / palmSize;
    fingers.thumb.abduction = thumbAbduction;
    fingers.thumb.extension = Math.min(
      fingers.thumb.extension,
      ramp(thumbAbduction, THUMB_ABDUCTION_RANGE.tucked, THUMB_ABDUCTION_RANGE.out),
    );

    const pinchDistance = dist(shape[LM.THUMB_TIP], shape[LM.INDEX_TIP]) / palmSize;

    const axis = sub(shape[LM.MIDDLE_MCP], shape[LM.WRIST]);
    const axisLen = len(axis);
    const along = (tip, mcp) => {
      const v = sub(shape[tip], shape[mcp]);
      return axisLen > 0 ? (v.x * axis.x + v.y * axis.y + v.z * axis.z) / axisLen / palmSize : 0;
    };
    const legs = { index: along(LM.INDEX_TIP, LM.INDEX_MCP), middle: along(LM.MIDDLE_TIP, LM.MIDDLE_MCP) };
    const tv = { x: disp[LM.THUMB_TIP].x - disp[LM.THUMB_MCP].x, y: disp[LM.THUMB_TIP].y - disp[LM.THUMB_MCP].y };
    const tl = Math.sqrt(tv.x * tv.x + tv.y * tv.y);
    const thumbDir = tl > 0 ? { x: tv.x / tl, y: tv.y / tl } : { x: 0, y: 0 };

    const palmIdx = [LM.WRIST, LM.INDEX_MCP, LM.MIDDLE_MCP, LM.RING_MCP, LM.PINKY_MCP];
    const palmCenter = { x: 0, y: 0 };
    for (const i of palmIdx) {
      palmCenter.x += disp[i].x / palmIdx.length;
      palmCenter.y += disp[i].y / palmIdx.length;
    }
    const palmSizeImage =
      (dist(disp[LM.WRIST], disp[LM.INDEX_MCP]) +
        dist(disp[LM.WRIST], disp[LM.MIDDLE_MCP]) +
        dist(disp[LM.WRIST], disp[LM.PINKY_MCP])) / 3;

    // Roll: angle of wrist -> middle MCP in the display plane (0 = upright).
    const up = sub(disp[LM.MIDDLE_MCP], disp[LM.WRIST]);
    const roll = Math.atan2(up.x, -up.y);

    let inFrame = 0;
    for (const p of image) if (p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1) inFrame++;

    return {
      valid: true,
      source: useWorld ? 'world' : 'image',
      palmSize,
      fingers,
      pinchDistance,
      legs,
      thumbDir,
      palmCenter,
      // Normalised display coordinates (0..1) of the palm and index tip,
      // convenient for consumers that map onto the screen.
      palmCenterNorm: { x: palmCenter.x / aspect, y: palmCenter.y },
      indexTipNorm: { x: disp[LM.INDEX_TIP].x / aspect, y: disp[LM.INDEX_TIP].y },
      // Index fingertip in the same display units as palmCenter (spin gesture).
      indexTip: { x: disp[LM.INDEX_TIP].x, y: disp[LM.INDEX_TIP].y },
      palmSizeImage,
      roll,
      inFrameRatio: inFrame / image.length,
      score: typeof hand.score === 'number' ? hand.score : null,
      handedness: hand.handedness ?? null,
      // 63-value pose vector for the project-trained hand-pose net (handPoseNet.js)
      pose: poseVector(disp),
      // learned gesture classifier output for this hand ({ label, score, scores }) or null
      model: hand.model && typeof hand.model.label === 'string' && Number.isFinite(hand.model.score) ? hand.model : null,
    };
  }
}
