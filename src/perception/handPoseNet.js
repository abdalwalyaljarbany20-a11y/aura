/**
 * AURA hand-pose network: a small MLP trained in this project (training/train_handpose.py)
 * on hand landmarks that MediaPipe extracted from the HaGRIDv2 dataset.
 *
 * Input: poseVector(display landmarks) = 21 points relative to the wrist, divided by the
 * image palm size (mean wrist -> index / middle / pinky MCP distance), flattened x,y,z
 * (63 values). Mirror invariant by training (random horizontal flips), so it works for
 * either hand and either camera mirroring.
 * Output: softmax over the app's static vocabulary + OTHER (any pose the app does not use).
 *
 * The weights file (public/models/aura_handpose.json) is optional: without it the
 * classifier fuses only the MediaPipe model and the geometric rules. Inference is
 * dependency-free and identical to backend/aura_cv/posenet.py (parity-tested).
 */

export const POSE_DIM = 63;
const WRIST = 0;
const PALM_MCPS = [5, 9, 17];

/**
 * @param {{x:number,y:number,z?:number}[]} disp display-space landmarks (aspect-corrected, optionally mirrored)
 * @returns {number[]|null}
 */
export function poseVector(disp) {
  if (!Array.isArray(disp) || disp.length !== 21) return null;
  const w = disp[WRIST];
  let size = 0;
  for (const i of PALM_MCPS) size += Math.hypot(disp[i].x - w.x, disp[i].y - w.y, (disp[i].z ?? 0) - (w.z ?? 0)) / 3;
  if (!(size > 1e-6)) return null;
  const v = new Array(POSE_DIM);
  for (let i = 0; i < 21; i++) {
    v[3 * i] = (disp[i].x - w.x) / size;
    v[3 * i + 1] = (disp[i].y - w.y) / size;
    v[3 * i + 2] = ((disp[i].z ?? 0) - (w.z ?? 0)) / size;
  }
  return v.every(Number.isFinite) ? v : null;
}

/**
 * Validate and wrap a weights object.
 * Format: { format: 'aura-handpose-mlp', version, classes: string[], inputMean: number[63],
 *           inputStd: number[63], layers: [{ in, out, W: number[out*in] (row-major), b: number[out] }] }
 * Hidden layers use ReLU, the last layer softmax.
 * @returns {{ classes: string[], predict(v: number[]): { label: string, score: number, scores: Record<string, number> } , info: object }}
 */
export function createHandPoseNet(json) {
  if (!json || json.format !== 'aura-handpose-mlp') throw new Error('not an aura-handpose-mlp weights file');
  const { classes, inputMean, inputStd, layers } = json;
  if (!Array.isArray(classes) || !classes.length) throw new Error('weights: classes missing');
  if (inputMean?.length !== POSE_DIM || inputStd?.length !== POSE_DIM) throw new Error('weights: input normalisation must have 63 values');
  let dim = POSE_DIM;
  for (const [k, l] of layers.entries()) {
    if (l.in !== dim || l.W?.length !== l.in * l.out || l.b?.length !== l.out) throw new Error(`weights: layer ${k} has inconsistent shapes`);
    dim = l.out;
  }
  if (dim !== classes.length) throw new Error('weights: output size does not match classes');
  const W = layers.map((l) => Float64Array.from(l.W));
  const B = layers.map((l) => Float64Array.from(l.b));
  return {
    classes,
    info: { version: json.version ?? null, dataset: json.dataset ?? null, metrics: json.metrics ?? null, params: W.reduce((s, w, i) => s + w.length + B[i].length, 0) },
    predict(v) {
      let x = new Float64Array(POSE_DIM);
      for (let i = 0; i < POSE_DIM; i++) x[i] = (v[i] - inputMean[i]) / inputStd[i];
      for (let k = 0; k < layers.length; k++) {
        const { in: n, out: m } = layers[k];
        const y = new Float64Array(m);
        for (let o = 0; o < m; o++) {
          let s = B[k][o];
          const row = o * n;
          for (let i = 0; i < n; i++) s += W[k][row + i] * x[i];
          y[o] = k < layers.length - 1 && s < 0 ? 0 : s;
        }
        x = y;
      }
      let max = -Infinity;
      for (const s of x) if (s > max) max = s;
      let sum = 0;
      const e = Array.from(x, (s) => ((sum += Math.exp(s - max)), Math.exp(s - max)));
      const scores = {};
      let best = 0;
      for (let i = 0; i < classes.length; i++) {
        scores[classes[i]] = e[i] / sum;
        if (e[i] > e[best]) best = i;
      }
      return { label: classes[best], score: e[best] / sum, scores };
    },
  };
}
