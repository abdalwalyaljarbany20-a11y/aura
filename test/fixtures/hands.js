/**
 * Synthetic hand generator for deterministic tests.
 *
 * Builds 21 MediaPipe-ordered landmarks from a simple kinematic model
 * (metres, x right, y down, z negative toward the camera, palm facing the
 * camera). Each finger has three segments; flexion angles (degrees) rotate
 * the segments toward the camera/palm. The output mimics a HandLandmarker
 * observation: normalised image landmarks + metric world landmarks.
 *
 * These are geometric idealisations, not recordings of real hands: they
 * verify the rules' logic, not their real-world accuracy.
 */

const FINGERS = {
  index: { mcp: [-0.03, -0.085, 0], seg: [0.04, 0.025, 0.02] },
  middle: { mcp: [-0.008, -0.09, 0], seg: [0.045, 0.028, 0.022] },
  ring: { mcp: [0.012, -0.085, 0], seg: [0.042, 0.026, 0.02] },
  pinky: { mcp: [0.03, -0.075, 0], seg: [0.032, 0.02, 0.018] },
};
const THUMB_CMC = [-0.025, -0.025, 0];

export const FLEX = {
  straight: [0, 4, 4],
  curled: [80, 100, 60],
  pinchIndex: [35, 45, 25],
  half: [40, 55, 35],
};

const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const rad = (d) => (d * Math.PI) / 180;

function fingerChain(base, seg, flex) {
  const pts = [base];
  let theta = 0;
  let p = base;
  for (let i = 0; i < 3; i++) {
    theta += rad(flex[i]);
    p = add(p, scale([0, -Math.cos(theta), -Math.sin(theta)], seg[i]));
    pts.push(p);
  }
  return pts;
}

const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

/**
 * @param {{ index?: number[], middle?: number[], ring?: number[], pinky?: number[],
 *           thumb?: 'out'|'tucked'|'pinch' }} pose
 * @param {{ cx?: number, cy?: number, size?: number, rotationDeg?: number, aspect?: number,
 *           score?: number, handedness?: string, world?: boolean }} [view]
 */
export function makeHand(pose, view = {}) {
  const { cx = 0.5, cy = 0.6, size = 2.5, rotationDeg = 0, aspect = 16 / 9, score = 0.95, handedness = 'Right', world = true } = view;
  const w = new Array(21);
  w[0] = [0, 0, 0];
  const chains = {};
  for (const [name, f] of Object.entries(FINGERS)) {
    chains[name] = fingerChain(f.mcp, f.seg, pose[name] ?? FLEX.straight);
  }
  [5, 9, 13, 17].forEach((start, k) => {
    const c = chains[['index', 'middle', 'ring', 'pinky'][k]];
    for (let j = 0; j < 4; j++) w[start + j] = c[j];
  });

  const thumb = pose.thumb ?? 'out';
  w[1] = THUMB_CMC;
  if (thumb === 'out') {
    const dir = [-0.85, -0.53, 0];
    w[2] = add(THUMB_CMC, scale(dir, 0.035));
    w[3] = add(w[2], scale(dir, 0.03));
    w[4] = add(w[3], scale(dir, 0.025));
  } else if (thumb === 'tucked') {
    w[2] = [-0.03, -0.045, -0.012];
    w[3] = [-0.02, -0.06, -0.028];
    w[4] = [-0.008, -0.07, -0.038];
  } else {
    // pinch: thumb tip ~8 mm from the index tip
    const tip = add(w[8], [-0.008, 0.002, 0.0]);
    w[4] = tip;
    w[2] = add(lerp(THUMB_CMC, tip, 0.4), [-0.012, 0, -0.005]);
    w[3] = add(lerp(THUMB_CMC, tip, 0.72), [-0.008, 0, -0.008]);
  }

  const r = rad(rotationDeg);
  const rot = ([x, y, z]) => [x * Math.cos(r) - y * Math.sin(r), x * Math.sin(r) + y * Math.cos(r), z];
  const worldLandmarks = w.map((p) => {
    const [x, y, z] = rot(p);
    return { x, y, z };
  });
  // Image: units of image height scaled by `size`, x divided by aspect
  // because normalised x is a fraction of the image width.
  const landmarks = worldLandmarks.map((p) => ({ x: cx + (p.x * size) / aspect, y: cy + p.y * size, z: (p.z * size) / aspect }));
  return { landmarks, worldLandmarks: world ? worldLandmarks : null, handedness, score };
}

export const POSES = {
  OPEN_PALM: { thumb: 'out' },
  FIST: { index: FLEX.curled, middle: FLEX.curled, ring: FLEX.curled, pinky: FLEX.curled, thumb: 'tucked' },
  POINT: { middle: FLEX.curled, ring: FLEX.curled, pinky: FLEX.curled, thumb: 'tucked' },
  PINCH: { index: FLEX.pinchIndex, middle: FLEX.straight, ring: FLEX.straight, pinky: FLEX.straight, thumb: 'pinch' },
  PINCH_CURLED: { index: FLEX.pinchIndex, middle: FLEX.curled, ring: FLEX.curled, pinky: FLEX.curled, thumb: 'pinch' },
  // Index + middle extended, ring + pinky curled, thumb folded: the walk gesture.
  VICTORY: { ring: FLEX.curled, pinky: FLEX.curled, thumb: 'tucked' },
  // Ambiguous / out-of-vocabulary shapes that must stay UNKNOWN:
  THREE_FINGERS: { pinky: FLEX.curled, thumb: 'tucked' },
  VICTORY_THUMB_OUT: { ring: FLEX.curled, pinky: FLEX.curled, thumb: 'out' },
  HALF_MIDDLE: { middle: FLEX.half, ring: FLEX.curled, pinky: FLEX.curled, thumb: 'tucked' },
  CLAW: { index: FLEX.half, middle: FLEX.half, ring: FLEX.half, pinky: FLEX.half, thumb: 'tucked' },
  THUMBS_UP: { index: FLEX.curled, middle: FLEX.curled, ring: FLEX.curled, pinky: FLEX.curled, thumb: 'out' }, // sideways thumb: UNKNOWN

  // Four fingers extended, thumb alongside the palm: accepted as OPEN_PALM.
  FLAT_HAND: { thumb: 'tucked' },
};

/**
 * Finger-walking pose at time t: index and middle flex in anti-phase (two legs),
 * ring + pinky curled, thumb tucked. `freq` = gait cycles per second (2 steps per
 * cycle), `amp` = peak flexion (deg). inPhase: both fingers bend together (not walking).
 */
/** Thumbs-up / thumbs-down: the sideways-thumb fist turned so the thumb points up / down in the image. */
export const THUMB_VIEW = Object.freeze({ up: { rotationDeg: 58 }, down: { rotationDeg: 238 } });

export function stepPose(t, { freq = 1, amp = 40, inPhase = false, only = null } = {}) {
  const s = Math.sin(2 * Math.PI * freq * (t / 1000));
  const flex = (k) => [amp * k, amp * k * 1.2, amp * k * 0.7];
  const a = 0.5 + 0.5 * s;
  const b = inPhase ? a : 0.5 - 0.5 * s;
  return {
    index: only === 'middle' ? FLEX.straight : flex(a),
    middle: only === 'index' ? FLEX.straight : flex(b),
    ring: FLEX.curled,
    pinky: FLEX.curled,
    thumb: 'tucked',
  };
}
