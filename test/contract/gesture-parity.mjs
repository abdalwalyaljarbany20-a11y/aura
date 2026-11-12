// Used by backend/tests/test_parity.py: runs the JS gesture engine on the same
// timestamped landmark sequences as the Python engine and prints its events.
// Optional argv[3]: hand-pose net weights; every sequence then also runs with the net
// (output key "<name>@net"), including the per-frame net prediction.
import { readFileSync } from 'node:fs';
import { DEFAULT_CONFIG } from '../../src/config/defaultConfig.js';
import { FeatureExtractionEngine } from '../../src/gestures/FeatureExtractionEngine.js';
import { GestureRecognitionEngine } from '../../src/gestures/GestureRecognitionEngine.js';
import { createHandPoseNet } from '../../src/perception/handPoseNet.js';

const sequences = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const net = process.argv[3] ? createHandPoseNet(JSON.parse(readFileSync(process.argv[3], 'utf8'))) : null;
const r4 = (v) => Math.round(v * 1e4) / 1e4;
const out = {};
for (const withNet of net ? [false, true] : [false]) {
  for (const [name, frames] of Object.entries(sequences)) {
    const fx = new FeatureExtractionEngine({ mirror: true });
    const engine = new GestureRecognitionEngine(JSON.parse(JSON.stringify(DEFAULT_CONFIG.gestures)), { net: withNet ? net : null });
    const events = [];
    const stable = [];
    const motion = [];
    const nets = [];
    for (const f of frames) {
      const obs = f.hand && {
        landmarks: f.hand.landmarks.map(([x, y, z]) => ({ x, y, z })),
        worldLandmarks: f.hand.world ? f.hand.world.map(([x, y, z]) => ({ x, y, z })) : null,
        handedness: 'Right',
        score: f.hand.score,
        model: f.hand.model ?? null, // learned gesture classifier output (synthetic in these sequences)
      };
      const feats = obs ? fx.extract(obs, { aspect: f.aspect }) : null;
      const r = engine.update(feats, f.t);
      for (const e of r.events) events.push([e.type, e.gesture, Math.round(e.t)]);
      stable.push(r.stable.gesture);
      motion.push([Math.round(r.motion.spin.omega * 1000) / 1000, Math.round(r.motion.walk.cadence * 1000) / 1000]);
      if (withNet) nets.push(feats?.net ? [feats.net.label, r4(feats.net.score)] : null);
    }
    out[withNet ? `${name}@net` : name] = withNet ? { events, stable, motion, nets } : { events, stable, motion };
  }
}
console.log(JSON.stringify(out));
