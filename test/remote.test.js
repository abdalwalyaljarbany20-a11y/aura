import assert from 'node:assert/strict';
import { test } from 'node:test';
import { DEFAULT_CONFIG } from '../src/config/defaultConfig.js';
import { loadConfig } from '../src/config/config.js';
import { EventBus } from '../src/core/EventBus.js';
import { FeatureExtractionEngine } from '../src/gestures/FeatureExtractionEngine.js';
import { InteractionController } from '../src/interaction/InteractionController.js';
import { parsePreview, PythonEngineClient, resultError } from '../src/remote/PythonEngineClient.js';
import { RemoteCamera, RemoteFramePipeline, RemoteTracker } from '../src/remote/RemoteEngine.js';
import { makeHand, POSES } from './fixtures/hands.js';

/** Minimal fake of the browser WebSocket used by PythonEngineClient. */
class FakeWS {
  static instances = [];
  constructor(url) {
    this.url = url;
    this.sent = [];
    FakeWS.instances.push(this);
    queueMicrotask(() => this.onopen?.());
  }
  send(data) {
    this.sent.push(JSON.parse(data));
  }
  close() {
    this.onclose?.({ code: 1000, reason: 'closed' });
  }
  serverSend(obj) {
    this.onmessage?.({ data: typeof obj === 'string' || obj instanceof ArrayBuffer ? obj : JSON.stringify(obj) });
  }
}

const HELLO = {
  type: 'hello', protocol: 2, engine: { name: 'aura_cv', version: '0.1.0', mediapipe: '1.1.1', opencv: '5.0.0', python: '3.14.7' },
  tracker: { status: 'ready', delegate: 'CPU (GPU unavailable)' }, camera: { status: 'stopped', error: null, info: null }, configErrors: [],
};
const LIVE = (session = 1) => ({ type: 'status', session, camera: { status: 'live', info: { index: 0, width: 1280, height: 720, backend: 'DSHOW', source: 'camera' } }, tracker: HELLO.tracker });

const toPy = (hand) => ({
  landmarks: hand.landmarks.map((p) => [p.x, p.y, p.z]),
  worldLandmarks: hand.worldLandmarks.map((p) => [p.x, p.y, p.z]),
  handedness: hand.handedness,
  score: hand.score,
});

function result(frameId, { status = 'DETECTED', hand = makeHand(POSES.OPEN_PALM), events = [], stable = 'NONE', session = 1 } = {}) {
  const detected = status === 'DETECTED';
  return {
    type: 'result', session, frameId, capturedAt: 1000, status, image: { width: 1280, height: 720 },
    hands: detected ? [toPy(hand)] : [], primary: detected ? 0 : null, features: null,
    gesture: { raw: detected ? 'OPEN_PALM' : 'NONE', label: detected ? 'OPEN_PALM' : 'NONE', confidence: detected ? 1 : 0, reason: null, scores: {}, stable, stableConfidence: 1 },
    events, timing: { inferenceMs: 10, totalMs: 14, warmup: false, previewEncodeMs: 2 }, error: null, sentAt: 1015, ageAtSendMs: 15,
  };
}

function preview(frameId, session = 1) {
  const buf = new ArrayBuffer(16);
  new Uint8Array(buf).set([0x41, 0x56, 0x46, 0x32]);
  const dv = new DataView(buf);
  dv.setUint32(4, session, false);
  dv.setUint32(8, frameId, false);
  return buf;
}

const tick = () => new Promise((r) => setTimeout(r, 0));

async function connected() {
  FakeWS.instances = [];
  const bus = new EventBus();
  const client = new PythonEngineClient('ws://127.0.0.1:8765', { bus, WebSocketImpl: FakeWS, connectTimeoutMs: 1000 });
  const p = client.connect();
  await tick();
  const ws = FakeWS.instances[0];
  ws.serverSend(HELLO);
  await p;
  return { client, ws, bus };
}

/** Camera + pipeline + controller wired like the coordinator does in Python mode. */
async function liveStack({ decode, clock } = {}) {
  const { client, ws, bus } = await connected();
  let now = clock ?? 0;
  const nowFn = () => now;
  const camera = new RemoteCamera(client, { bus, now: nowFn, decode: decode ?? (async () => ({ width: 640, height: 360, close() {} })) });
  const controller = new InteractionController(DEFAULT_CONFIG.interaction, { bus });
  const pipeline = new RemoteFramePipeline({ extractor: new FeatureExtractionEngine({ mirror: true }), controller, bus, now: nowFn });
  const states = [];
  camera.startFrameLoop((f) => states.push(pipeline.process(f.result)), (v) => pipeline.commitVisual(v.frameId, v.t));
  const live = camera.start();
  ws.serverSend(LIVE(1));
  assert.equal((await live).ok, true);
  return { client, ws, bus, camera, controller, pipeline, states, advance: (ms) => (now += ms) };
}

// ------------------------------------------------------------------ config & client

test('engine config section validates and the backend section is left to Python', () => {
  const { config, errors } = loadConfig({ engine: { mode: 'python', url: 'ws://127.0.0.1:9000' }, backend: { cameraIndex: 2 } });
  assert.deepEqual(errors, []);
  assert.equal(config.engine.mode, 'python');
  assert.equal(config.backend, undefined);
  assert.ok(loadConfig({ engine: { mode: 'cloud', url: 'http://x' } }).errors.length === 2);
  assert.equal(DEFAULT_CONFIG.engine.mode, 'browser', 'browser engine stays the default');
});

test('client handshake sends a protocol v2 hello and exposes engine info', async () => {
  const { client, ws } = await connected();
  assert.deepEqual(ws.sent[0], { type: 'hello', client: 'aura-frontend', protocol: 2 });
  assert.equal(client.status, 'connected');
  assert.equal(client.hello.engine.mediapipe, '1.1.1');
});

test('client rejects an unsupported protocol version', async () => {
  FakeWS.instances = [];
  const client = new PythonEngineClient('ws://x', { WebSocketImpl: FakeWS, connectTimeoutMs: 50 });
  const p = client.connect();
  await tick();
  FakeWS.instances[0].serverSend({ ...HELLO, protocol: 1 });
  await assert.rejects(p, /not reachable/);
  assert.match(client.lastError, /protocol 1/);
});

test('connect fails clearly when no engine answers', async () => {
  class DeadWS {
    constructor() {
      queueMicrotask(() => this.onclose?.({ code: 1006, reason: '' }));
    }
    close() {}
  }
  const client = new PythonEngineClient('ws://127.0.0.1:1', { WebSocketImpl: DeadWS, connectTimeoutMs: 30 });
  await assert.rejects(client.connect(), /not reachable.*npm run backend/);
  client.close();
});

test('malformed results (including invalid gesture events) are dropped, never delivered', async () => {
  const { client, ws } = await connected();
  const got = [];
  client.on('result', (r) => got.push(r));
  const bad = result(1);
  bad.hands[0].landmarks = bad.hands[0].landmarks.slice(0, 5);
  ws.serverSend(bad);
  ws.serverSend({ ...result(2), status: 'MAYBE' });
  ws.serverSend({ ...result(3), primary: null });
  ws.serverSend({ ...result(4, { status: 'NOT_DETECTED' }), primary: 0 });
  ws.serverSend({ ...result(5), session: undefined });
  ws.serverSend(result(6, { events: [{ type: 'start', gesture: 'SWIPE_LEFT' }] })); // swipes are 'event', not 'start'
  ws.serverSend(result(7, { events: [{ type: 'start', gesture: 'UNKNOWN' }] }));
  ws.serverSend('{not json');
  ws.serverSend(new ArrayBuffer(3));
  ws.serverSend(result(8));
  assert.deepEqual(got.map((r) => r.frameId), [8]);
  assert.equal(client.stats.malformed, 9);
  assert.equal(resultError(result(9)), null);
  assert.deepEqual({ ...parsePreview(preview(77, 3)), jpeg: undefined }, { session: 3, frameId: 77, jpeg: undefined });
  assert.equal(parsePreview(new Uint8Array([0x41, 0x56, 0x46, 0x31, 0, 0, 0, 1]).buffer), null, 'v1 previews are rejected');
});

test('startCamera resolves with the session id and reports engine errors', async () => {
  const { client, ws } = await connected();
  const p = client.startCamera(1000);
  assert.equal(ws.sent.at(-1).type, 'start');
  ws.serverSend(LIVE(4));
  assert.deepEqual(await p, { ok: true, session: 4 });
  const p2 = client.startCamera(1000);
  ws.serverSend({ type: 'status', session: 4, camera: { status: 'error', error: { code: 'UNAVAILABLE', message: 'Camera could not be opened' } }, tracker: HELLO.tracker,
    startError: { code: 'UNAVAILABLE', message: 'Camera could not be opened' } });
  const r = await p2;
  assert.equal(r.ok, false);
  assert.equal(r.error.code, 'UNAVAILABLE');
});

// ------------------------------------------------------------------ delivery & synchronisation

test('events are applied immediately; the overlay waits for the frame-matched preview', async () => {
  const s = await liveStack();
  s.ws.serverSend(result(1, { events: [{ type: 'start', gesture: 'OPEN_PALM', confidence: 1 }], stable: 'OPEN_PALM' }));
  assert.equal(s.controller.state, 'EXECUTING', 'command executed without waiting for the preview');
  assert.equal(s.controller.snapshot(0).action.name, 'WAVE');
  assert.equal(s.pipeline.visualLatest(0).hands.length, 0, 'no overlay before its frame is on screen');
  s.ws.serverSend(preview(1));
  await tick();
  const v = s.pipeline.visualLatest(0);
  assert.equal(v.frameId, 1);
  assert.equal(v.hands.length, 1);
  assert.ok(v.features.valid, 'render geometry derived from Python landmarks');
});

test('a missing preview never blocks or loses gesture events', async () => {
  const s = await liveStack();
  s.ws.serverSend(result(1, { events: [{ type: 'start', gesture: 'PINCH', confidence: 1 }] }));
  assert.equal(s.controller.state, 'CONTROL');
  // preview for 1 never arrives; the release in result 2 must still apply
  s.ws.serverSend(result(2, { status: 'TRACKING_LOST', events: [{ type: 'end', gesture: 'PINCH' }] }));
  assert.equal(s.controller.state, 'IDLE');
  assert.equal(s.camera.stats.previewsMissed, 1);
});

test('stale results are rejected: older frameId and foreign session', async () => {
  const s = await liveStack();
  s.ws.serverSend(result(5));
  s.ws.serverSend(result(4, { events: [{ type: 'start', gesture: 'FIST' }] }));
  s.ws.serverSend(result(5));
  s.ws.serverSend(result(9, { session: 2, events: [{ type: 'start', gesture: 'FIST' }] }));
  assert.deepEqual(s.states.map((x) => x.frameId), [5]);
  assert.equal(s.camera.stats.staleResults, 3);
  assert.equal(s.controller.state, 'TRACKING', 'stale FIST did not lock');
});

test('previews of other sessions or superseded frames are not drawn', async () => {
  const s = await liveStack();
  s.ws.serverSend(result(1));
  s.ws.serverSend(preview(1, 2)); // wrong session
  s.ws.serverSend(result(2));
  s.ws.serverSend(preview(1)); // superseded by frame 2
  await tick();
  assert.equal(s.camera.stats.previewsDrawn, 0);
  assert.equal(s.camera.stats.previewsStale, 2);
  s.ws.serverSend(preview(2));
  await tick();
  assert.equal(s.camera.stats.previewsDrawn, 1);
  assert.equal(s.pipeline.visualLatest(0).frameId, 2);
});

test('slow decodes never draw an older frame over a newer one', async () => {
  const resolvers = [];
  const s = await liveStack({ decode: () => new Promise((r) => resolvers.push(() => r({ width: 640, height: 360, close() {} }))) });
  s.ws.serverSend(result(1));
  s.ws.serverSend(preview(1));
  s.ws.serverSend(result(2));
  s.ws.serverSend(preview(2));
  resolvers[1](); // frame 2 decodes first
  await tick();
  resolvers[0]();
  await tick();
  assert.equal(s.camera.stats.previewsDrawn, 1);
  assert.equal(s.pipeline.visualLatest(0).frameId, 2);
});

test('overlay is hidden when the newest frame has no hand or its preview lags too long', async () => {
  const s = await liveStack();
  s.ws.serverSend(result(1));
  s.ws.serverSend(preview(1));
  await tick();
  assert.equal(s.pipeline.visualLatest(0).hands.length, 1);
  s.ws.serverSend(result(2)); // newer DETECTED result, preview pending
  s.advance(100);
  assert.equal(s.pipeline.visualLatest(100).frameId, 1, 'last matched pair shown briefly');
  assert.equal(s.pipeline.visualLatest(400).hands.length, 0, 'hidden after maxVisualLagMs');
  s.ws.serverSend(result(3, { status: 'NOT_DETECTED' }));
  assert.equal(s.pipeline.visualLatest(400).hands.length, 0, 'no stale landmarks without a hand');
});

test('tracking loss / camera unavailable show no stale landmarks and release holds', async () => {
  const s = await liveStack();
  s.ws.serverSend(result(1, { events: [{ type: 'start', gesture: 'POINT', confidence: 1 }] }));
  s.ws.serverSend(preview(1));
  await tick();
  assert.equal(s.controller.state, 'CONTROL');
  s.ws.serverSend(result(2, { status: 'TRACKING_LOST', events: [{ type: 'end', gesture: 'POINT' }] }));
  assert.equal(s.states.at(-1).hands.length, 0);
  assert.equal(s.pipeline.visualLatest(0).hands.length, 0);
  assert.equal(s.controller.state, 'IDLE');
  s.ws.serverSend({ ...result(null, { status: 'CAMERA_UNAVAILABLE' }), frameId: null, error: 'Camera capture was interrupted' });
  assert.equal(s.states.at(-1).status, 'CAMERA_UNAVAILABLE');
});

test('a held gesture triggers its one-shot command once; re-trigger needs release', async () => {
  const s = await liveStack();
  const start = { type: 'start', gesture: 'OPEN_PALM', confidence: 1 };
  s.ws.serverSend(result(1, { events: [start] }));
  for (let i = 2; i <= 40; i++) s.ws.serverSend(result(i, { stable: 'OPEN_PALM' })); // held: no new events
  const accepted = s.controller.counts.accepted;
  assert.equal(accepted, 1);
  s.advance(2000);
  s.ws.serverSend(result(41, { events: [{ type: 'end', gesture: 'OPEN_PALM' }] }));
  s.ws.serverSend(result(42, { events: [start] }));
  assert.equal(s.controller.counts.accepted, 2);
});

test('engine disconnect while live is reported as an interrupted camera and clears the session', async () => {
  const s = await liveStack();
  const events = [];
  s.bus.on('camera', (e) => events.push(e));
  s.ws.onclose({ code: 1006, reason: 'engine crashed' });
  assert.equal(s.camera.status, 'error');
  assert.equal(events.at(-1).error.code, 'INTERRUPTED');
  assert.equal(s.camera.session, null);
  s.ws.serverSend(result(99)); // late message after loss
  assert.equal(s.states.length, 0);
  s.client.close();
});

// ------------------------------------------------------------------ reconnection policy

/** Manual timer queue so reconnect timing is deterministic. */
function fakeTimers() {
  const timers = [];
  return {
    timers,
    setTimeout: (fn, ms) => (timers.push({ fn, ms }), timers.length),
    clearTimeout: () => {},
    runNext() {
      const t = timers.shift();
      t?.fn();
      return t?.ms;
    },
  };
}

test('reconnect backoff is bounded (0.5 s doubling to a 5 s cap, ±20 % jitter)', async () => {
  class DeadWS {
    constructor() {
      queueMicrotask(() => this.onclose?.({ code: 1006, reason: '' }));
    }
    close() {}
  }
  const ft = fakeTimers();
  const client = new PythonEngineClient('ws://x', { WebSocketImpl: DeadWS, setTimeout: ft.setTimeout, clearTimeout: ft.clearTimeout, random: () => 0.5, connectTimeoutMs: 99999 });
  client.connect().catch(() => {});
  const delays = [];
  for (let i = 0; i < 7; i++) {
    await tick();
    const retry = ft.timers.find((t) => t.ms !== 99999);
    ft.timers.splice(ft.timers.indexOf(retry), 1);
    delays.push(retry.ms);
    retry.fn();
  }
  assert.deepEqual(delays, [500, 1000, 2000, 4000, 5000, 5000, 5000]);
  const jittered = new PythonEngineClient('ws://x', { WebSocketImpl: DeadWS, setTimeout: ft.setTimeout, random: () => 1 });
  jittered._scheduleReconnect();
  assert.equal(jittered.nextRetryMs, 600);
  client.close();
  jittered.close();
});

test('an engine that accepts then closes ("busy") does not reset the backoff', async () => {
  class BusyWS {
    constructor() {
      this.sent = [];
      queueMicrotask(() => {
        this.onopen?.();
        this.onclose?.({ code: 1013, reason: '' });
      });
    }
    send() {}
    close() {}
  }
  const ft = fakeTimers();
  const client = new PythonEngineClient('ws://x', { WebSocketImpl: BusyWS, setTimeout: ft.setTimeout, clearTimeout: ft.clearTimeout, random: () => 0.5, connectTimeoutMs: 99999 });
  client.connect().catch(() => {});
  const delays = [];
  for (let i = 0; i < 3; i++) {
    await tick();
    const retry = ft.timers.find((t) => t.ms !== 99999);
    ft.timers.splice(ft.timers.indexOf(retry), 1);
    delays.push(retry.ms);
    retry.fn();
  }
  assert.deepEqual(delays, [500, 1000, 2000]);
  assert.equal(client.lastError, 'engine busy');
  client.close();
});

test('engine restart: outage is measured, stale data is not revived, camera is not auto-started', async () => {
  let now = 1000;
  FakeWS.instances = [];
  const ft = fakeTimers();
  const bus = new EventBus();
  const client = new PythonEngineClient('ws://x', { bus, WebSocketImpl: FakeWS, setTimeout: ft.setTimeout, clearTimeout: ft.clearTimeout, random: () => 0.5, now: () => now });
  const p = client.connect();
  await tick();
  FakeWS.instances[0].serverSend(HELLO);
  await p;
  const camera = new RemoteCamera(client, { bus, now: () => now, decode: async () => ({ width: 1, height: 1, close() {} }) });
  const got = [];
  camera.startFrameLoop((f) => got.push(f.result.frameId));
  const live = camera.start();
  FakeWS.instances[0].serverSend(LIVE(1));
  await live;
  FakeWS.instances[0].serverSend(result(1));
  FakeWS.instances[0].onclose({ code: 1006, reason: 'engine exited' }); // engine process died
  assert.equal(camera.status, 'error');
  assert.equal(client.stats.interruptions, 1);
  now += 2500;
  ft.timers.find((t) => t.ms === 500).fn(); // reconnect attempt
  await tick();
  const ws2 = FakeWS.instances[1];
  ws2.serverSend({ ...HELLO, session: 0 }); // restarted engine
  assert.equal(client.status, 'connected');
  assert.equal(client.outage.lastDurationMs, 2500);
  ws2.serverSend(result(2)); // must not be shown: camera is not live
  assert.deepEqual(got, [1]);
  assert.ok(!ws2.sent.some((m) => m.type === 'start'), 'no automatic camera start after reconnect');
  client.close();
});

test('RemoteTracker reflects the engine tracker status', async () => {
  FakeWS.instances = [];
  const bus = new EventBus();
  const seen = [];
  bus.on('tracker', (e) => seen.push(e));
  const client = new PythonEngineClient('ws://x', { WebSocketImpl: FakeWS, bus, connectTimeoutMs: 1000 });
  const tracker = new RemoteTracker(client, { bus });
  const p = tracker.init();
  await tick();
  FakeWS.instances[0].serverSend({ ...HELLO, tracker: { status: 'error', error: 'hand model not found' } });
  await p;
  assert.equal(tracker.status, 'error');
  assert.match(seen.at(-1).error, /not found/);
  tracker.close();
});
