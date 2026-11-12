// Contract check used by backend/tests/test_contract.py.
// Reads Python engine `result` messages (JSON lines), validates them with the
// frontend's validator and drives the real RemoteFramePipeline + InteractionController.
// Prints a JSON summary to stdout.
import { readFileSync } from 'node:fs';
import { DEFAULT_CONFIG } from '../../src/config/defaultConfig.js';
import { EventBus } from '../../src/core/EventBus.js';
import { FeatureExtractionEngine } from '../../src/gestures/FeatureExtractionEngine.js';
import { InteractionController } from '../../src/interaction/InteractionController.js';
import { resultError } from '../../src/remote/PythonEngineClient.js';
import { RemoteFramePipeline } from '../../src/remote/RemoteEngine.js';

const lines = readFileSync(process.argv[2], 'utf8').split('\n').filter(Boolean);
const bus = new EventBus();
const accepted = [];
bus.on('command', (e) => e.status === 'accepted' && accepted.push(e.name));
const controller = new InteractionController(DEFAULT_CONFIG.interaction, { bus });
// Simulated clock: the frames' own capture timestamps (so the run is deterministic).
let clock = 0;
const pipeline = new RemoteFramePipeline({ extractor: new FeatureExtractionEngine({ mirror: true }), controller, bus, now: () => clock });
const errors = [];
const statuses = [];
for (const line of lines) {
  const msg = JSON.parse(line);
  const err = resultError(msg);
  if (err) {
    errors.push(err);
    continue;
  }
  clock = msg.capturedAt;
  const latest = pipeline.process({ ...msg, receivedAt: clock + 2 });
  statuses.push(`${latest.status}:${latest.hands.length}:${latest.gesture.stable.gesture}`);
}
console.log(JSON.stringify({ count: lines.length, errors, accepted, statuses, finalState: controller.state }));
