/**
 * MediaPipe hand topology (21 landmarks). Any alternative tracking backend
 * must produce landmarks in this order to reuse the feature extractor.
 */
export const LM = Object.freeze({
  WRIST: 0,
  THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
});

export const NUM_LANDMARKS = 21;

/** Joint chains used for extension measurements (base -> tip). */
export const FINGER_CHAINS = Object.freeze({
  thumb: [1, 2, 3, 4],
  index: [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring: [13, 14, 15, 16],
  pinky: [17, 18, 19, 20],
});

/** Skeleton segments for rendering. */
export const HAND_CONNECTIONS = Object.freeze([
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
]);

export const FINGERTIPS = Object.freeze([4, 8, 12, 16, 20]);

const finite = (v) => typeof v === 'number' && Number.isFinite(v);

/**
 * @param {unknown} points
 * @returns {string|null} reason the landmark set is unusable, or null if valid
 */
export function landmarkSetError(points) {
  if (!Array.isArray(points)) return 'landmarks missing';
  if (points.length !== NUM_LANDMARKS) return `expected ${NUM_LANDMARKS} landmarks, got ${points.length}`;
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    if (!p || !finite(p.x) || !finite(p.y)) return `landmark ${i} is not a finite point`;
    if (p.z !== undefined && !finite(p.z)) return `landmark ${i} has a non-finite z`;
  }
  return null;
}
