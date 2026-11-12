/**
 * Marionette strings (reference video): each of the five fingertips is tied to a robot
 * anchor (left hand, head, chest, right hand, hover base) by a small bundle of glowing,
 * sagging strings. Pure geometry here (screen px); VisualEffectsEngine draws it.
 *
 *   pairTipsToAnchors  both lists sorted left -> right and paired in order, so strings
 *                      never cross however the hand is turned
 *   strandCurve        quadratic curve per strand: the control point sags below the
 *                      midpoint (sag grows with the span) and strands fan out sideways
 *   fallingTips        after the hand is lost, the fingertip ends fall under gravity and
 *                      drift toward the lower left, fading out over `fallMs`
 */
export const STRANDS = 4;

export function pairTipsToAnchors(tips, anchors) {
  if (!Array.isArray(tips) || !Array.isArray(anchors) || !tips.length || !anchors.length) return [];
  const ok = (p) => p && Number.isFinite(p.x) && Number.isFinite(p.y);
  const t = tips.filter(ok).slice().sort((a, b) => a.x - b.x);
  const a = anchors.filter(ok).slice().sort((p, q) => p.x - q.x);
  const n = Math.min(t.length, a.length);
  const pairs = [];
  for (let i = 0; i < n; i++) pairs.push({ tip: t[i], anchor: a[Math.round((i * (a.length - 1)) / Math.max(1, n - 1))] });
  return pairs;
}

/** @returns {{ from, ctrl, to }} control polygon of strand k of a pair */
export function strandCurve(tip, anchor, k, strands = STRANDS) {
  const dx = anchor.x - tip.x;
  const dy = anchor.y - tip.y;
  const len = Math.hypot(dx, dy) || 1;
  const spread = (k - (strands - 1) / 2) * Math.min(14, len * 0.04);
  const nx = -dy / len;
  const ny = dx / len;
  const sag = len * 0.22;
  return {
    from: tip,
    ctrl: { x: (tip.x + anchor.x) / 2 + nx * spread, y: (tip.y + anchor.y) / 2 + sag + ny * spread },
    to: { x: anchor.x + nx * spread * 0.2, y: anchor.y + ny * spread * 0.2 },
  };
}

/** Fingertip ends after the hand was lost `elapsedMs` ago; alpha 1 -> 0 over fallMs. */
export function fallingTips(lastTips, elapsedMs, { fallMs = 800, gravity = 2600, drift = 220, floorY = Infinity } = {}) {
  const s = Math.max(0, elapsedMs) / 1000;
  const alpha = Math.max(0, 1 - elapsedMs / fallMs);
  const tips = (lastTips ?? []).map((p) => ({ x: p.x - drift * s, y: Math.min(floorY, p.y + 0.5 * gravity * s * s) }));
  return { tips, alpha };
}
