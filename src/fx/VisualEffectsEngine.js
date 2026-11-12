import { FINGERTIPS, HAND_CONNECTIONS, LM } from '../gestures/landmarks.js';
import { fallingTips, pairTipsToAnchors, STRANDS, strandCurve } from './marionette.js';

/**
 * VisualEffectsEngine (Canvas2D overlay)
 *
 * Everything drawn here is derived from the latest tracking result:
 *   - skeleton segments and glowing landmark points for every tracked hand
 *   - fingertip trails: a timestamped history of the primary hand's
 *     fingertips (one sample per *processed* frame), aged out after
 *     `trailMs` and capped at `maxTrailPoints`
 *   - palm arcs whose centre, radius and orientation follow the palm
 *   - pinch link between thumb and index tips, with its normalised distance
 *   - marionette strings from the five fingertips to the robot (hints.anchors), which
 *     fall and fade when the hand is lost (see marionette.js)
 * When tracking is lost, the last overlay fades out over `fadeOutMs` and
 * trails are cleared. Nothing is drawn without tracking data.
 */
const ORANGE = '255,138,42';
const CYAN = '56,225,255';

function makeGlowSprite(rgb, size = 64) {
  const c = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(size, size) : Object.assign(document.createElement('canvas'), { width: size, height: size });
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, `rgba(${rgb},1)`);
  grad.addColorStop(0.25, `rgba(${rgb},0.55)`);
  grad.addColorStop(1, `rgba(${rgb},0)`);
  g.fillStyle = grad;
  g.fillRect(0, 0, size, size);
  return c;
}

export class VisualEffectsEngine {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} cfg effects configuration
   */
  constructor(canvas, cfg) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.cfg = cfg;
    this.dpr = 1;
    /** @type {Map<number, {x:number,y:number,t:number}[]>} */
    this.trails = new Map(cfg.trailFingertips.map((i) => [i, []]));
    this.lastResultT = null;
    this.lostAt = null;
    this.lastDrawn = null; // last result that had hands, for fade-out
    this.strings = { tips: null, lostAt: null };
    this.glowOrange = makeGlowSprite(ORANGE);
    this.glowCyan = makeGlowSprite(CYAN);
  }

  resize(width, height, pixelRatioCap = 2) {
    this.dpr = Math.min(globalThis.devicePixelRatio || 1, pixelRatioCap);
    this.canvas.width = Math.round(width * this.dpr);
    this.canvas.height = Math.round(height * this.dpr);
    this.width = width;
    this.height = height;
  }

  clear() {
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    for (const t of this.trails.values()) t.length = 0;
    this.lastDrawn = null;
    this.lostAt = null;
    this.lastResultT = null;
  }

  /**
   * @param {any} latest FramePipeline.latest (or null)
   * @param {ReturnType<import('./screenMapping.js').createCoverMapping>} map
   * @param {number} now ms
   * @param {{ stable?: string, controlling?: boolean }} [hints]
   */
  draw(latest, map, now, hints = {}) {
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.width, this.height);
    if (!latest) {
      this.strings = { tips: null, lostAt: null };
      return;
    }

    const hasHands = latest.hands.length > 0;
    const isNew = latest.t !== this.lastResultT;
    this.lastResultT = latest.t;

    if (hasHands) {
      this.lostAt = null;
      this.lastDrawn = latest;
      if (isNew) this._sampleTrails(latest, map);
    } else if (this.lastDrawn) {
      if (this.lostAt === null) this.lostAt = now;
      for (const t of this.trails.values()) t.length = 0;
    }

    this._drawStrings(hasHands ? latest.primary : null, map, now, hints.anchors);
    const source = hasHands ? latest : this.lastDrawn;
    if (!source) return;
    const alpha = hasHands ? 1 : Math.max(0, 1 - (now - this.lostAt) / this.cfg.fadeOutMs);
    if (alpha <= 0) {
      this.lastDrawn = null;
      return;
    }

    this._ageTrails(now);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.globalCompositeOperation = 'lighter';
    if (hasHands) this._drawTrails(now);
    for (const h of source.hands) {
      const primary = h === source.primary;
      const pts = h.obs.landmarks.map((p) => map.toScreen(p));
      this._drawSkeleton(pts, primary);
      if (primary && h.features?.valid) this._drawPalmArcs(pts, h.features, map, now, hints);
      this._drawPoints(pts, primary);
      if (primary && h.features?.valid) this._drawPinch(pts, h.features);
    }
    ctx.restore();
    if (source.primary?.features?.valid && hints.stable) this._drawLabel(source.primary, map, hints, alpha);
  }

  /** Fingertips -> robot anchors; on hand loss the tips fall and fade (reference video). */
  _drawStrings(primary, map, now, anchors) {
    if (!anchors) return;
    let tips;
    let alpha = 1;
    if (primary) {
      tips = FINGERTIPS.map((i) => map.toScreen(primary.obs.landmarks[i]));
      this.strings = { tips, lostAt: null };
    } else if (this.strings.tips) {
      if (this.strings.lostAt === null) this.strings.lostAt = now;
      const f = fallingTips(this.strings.tips, now - this.strings.lostAt, { floorY: this.height - 4 });
      if (f.alpha <= 0) {
        this.strings = { tips: null, lostAt: null };
        return;
      }
      tips = f.tips;
      alpha = f.alpha;
    } else return;
    const ctx = this.ctx;
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.lineCap = 'round';
    for (const { tip, anchor } of pairTipsToAnchors(tips, anchors)) {
      for (let k = 0; k < STRANDS; k++) {
        const c = strandCurve(tip, anchor, k);
        ctx.strokeStyle = `rgba(${ORANGE},${(0.55 - k * 0.07) * alpha})`;
        ctx.lineWidth = k === 0 ? 1.6 : 1.0;
        ctx.beginPath();
        ctx.moveTo(c.from.x, c.from.y);
        ctx.quadraticCurveTo(c.ctrl.x, c.ctrl.y, c.to.x, c.to.y);
        ctx.stroke();
      }
      // glowing white fingertip knot
      ctx.globalAlpha = alpha;
      ctx.drawImage(this.glowCyan, tip.x - 14, tip.y - 14, 28, 28);
      ctx.fillStyle = 'rgba(255,255,255,0.95)';
      ctx.beginPath();
      ctx.arc(tip.x, tip.y, 4.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    ctx.restore();
  }

  _sampleTrails(latest, map) {
    const prim = latest.primary;
    if (!prim) return;
    for (const [idx, trail] of this.trails) {
      const p = map.toScreen(prim.obs.landmarks[idx]);
      trail.push({ x: p.x, y: p.y, t: latest.t });
      if (trail.length > this.cfg.maxTrailPoints) trail.shift();
    }
  }

  _ageTrails(now) {
    for (const trail of this.trails.values()) {
      while (trail.length && now - trail[0].t > this.cfg.trailMs) trail.shift();
    }
  }

  _drawTrails(now) {
    const ctx = this.ctx;
    ctx.lineCap = 'round';
    for (const [idx, trail] of this.trails) {
      if (trail.length < 3) continue;
      const rgb = idx === LM.INDEX_TIP || idx === LM.THUMB_TIP ? ORANGE : CYAN;
      for (let i = 1; i < trail.length - 1; i++) {
        const a = trail[i - 1];
        const b = trail[i];
        const c = trail[i + 1];
        const age = 1 - (now - b.t) / this.cfg.trailMs;
        if (age <= 0) continue;
        ctx.strokeStyle = `rgba(${rgb},${0.55 * age})`;
        ctx.lineWidth = 0.5 + 3.5 * age;
        ctx.beginPath();
        ctx.moveTo((a.x + b.x) / 2, (a.y + b.y) / 2);
        ctx.quadraticCurveTo(b.x, b.y, (b.x + c.x) / 2, (b.y + c.y) / 2);
        ctx.stroke();
      }
    }
  }

  _drawSkeleton(pts, primary) {
    const ctx = this.ctx;
    ctx.strokeStyle = primary ? `rgba(${CYAN},0.55)` : `rgba(${CYAN},0.25)`;
    ctx.lineWidth = primary ? 1.4 : 1;
    ctx.beginPath();
    for (const [a, b] of HAND_CONNECTIONS) {
      ctx.moveTo(pts[a].x, pts[a].y);
      ctx.lineTo(pts[b].x, pts[b].y);
    }
    ctx.stroke();
  }

  _drawPoints(pts, primary) {
    const ctx = this.ctx;
    const glow = this.cfg.glow;
    for (let i = 0; i < pts.length; i++) {
      const tip = FINGERTIPS.includes(i);
      const r = tip ? (primary ? 16 : 10) : primary ? 9 : 6;
      if (glow) ctx.drawImage(tip ? this.glowOrange : this.glowCyan, pts[i].x - r, pts[i].y - r, r * 2, r * 2);
      ctx.fillStyle = tip ? '#fff4e6' : '#e6fbff';
      ctx.beginPath();
      ctx.arc(pts[i].x, pts[i].y, tip ? 2.6 : 1.8, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  _drawPalmArcs(pts, f, map, now, hints) {
    const ctx = this.ctx;
    const c = map.displayToScreen(f.palmCenterNorm);
    const r = f.palmSizeImage * map.unit * 0.95;
    const base = f.roll - Math.PI / 2;
    const spin = (now / 1000) * (hints.controlling ? 1.6 : 0.5);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = `rgba(${ORANGE},${hints.controlling ? 0.85 : 0.5})`;
    ctx.beginPath();
    ctx.arc(c.x, c.y, r, base + spin, base + spin + Math.PI * 0.55);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(c.x, c.y, r, base + spin + Math.PI, base + spin + Math.PI * 1.35);
    ctx.stroke();
    ctx.strokeStyle = `rgba(${CYAN},0.35)`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(c.x, c.y, r * 1.18, base - spin * 0.7, base - spin * 0.7 + Math.PI * 1.2);
    ctx.stroke();
    // tick marks at the wrist direction
    const w = pts[LM.WRIST];
    ctx.strokeStyle = `rgba(${CYAN},0.4)`;
    ctx.beginPath();
    ctx.moveTo(c.x, c.y);
    ctx.lineTo(c.x + (w.x - c.x) * 0.25, c.y + (w.y - c.y) * 0.25);
    ctx.stroke();
  }

  _drawPinch(pts, f) {
    const closeness = Math.max(0, Math.min(1, (0.6 - f.pinchDistance) / 0.4));
    if (closeness <= 0) return;
    const ctx = this.ctx;
    const a = pts[LM.THUMB_TIP];
    const b = pts[LM.INDEX_TIP];
    ctx.strokeStyle = `rgba(${ORANGE},${0.3 + 0.6 * closeness})`;
    ctx.lineWidth = 1 + 2 * closeness;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  _drawLabel(prim, map, hints, alpha) {
    const ctx = this.ctx;
    const f = prim.features;
    const pts = prim.obs.landmarks.map((p) => map.toScreen(p));
    let minY = Infinity;
    let cx = 0;
    for (const p of pts) {
      minY = Math.min(minY, p.y);
      cx += p.x / pts.length;
    }
    const text = hints.stable === 'NONE' ? '' : hints.stable;
    const sub = f.handedness ? `${f.handedness.toUpperCase()} · ${(f.score ?? 0).toFixed(2)}` : '';
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.font = '600 13px ui-monospace, Consolas, monospace';
    ctx.textAlign = 'center';
    ctx.fillStyle = hints.stable === 'NONE' ? 'rgba(200,210,220,0.6)' : `rgb(${ORANGE})`;
    ctx.fillText(text, cx, minY - 22);
    ctx.font = '10px ui-monospace, Consolas, monospace';
    ctx.fillStyle = 'rgba(160,170,180,0.8)';
    ctx.fillText(sub, cx, minY - 8);
    ctx.restore();
  }
}
