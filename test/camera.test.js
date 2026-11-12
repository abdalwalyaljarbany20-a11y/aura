import assert from 'node:assert/strict';
import { test } from 'node:test';
import { CameraManager, mapCameraError } from '../src/camera/CameraManager.js';
import { DEFAULT_CONFIG } from '../src/config/defaultConfig.js';
import { EventBus } from '../src/core/EventBus.js';

/** Fake MediaStreamTrack with an "ended" event. */
class FakeTrack {
  constructor() {
    this.stopped = false;
    this.listeners = {};
    this.label = 'Fake Cam';
  }
  stop() {
    this.stopped = true;
  }
  addEventListener(type, fn) {
    this.listeners[type] = fn;
  }
  removeEventListener(type) {
    delete this.listeners[type];
  }
  getSettings() {
    return { width: 640, height: 480, frameRate: 30, deviceId: 'cam1' };
  }
  fireEnded() {
    this.listeners.ended?.();
  }
}

function fakeStream() {
  const track = new FakeTrack();
  return { track, stream: { getTracks: () => [track], getVideoTracks: () => [track] } };
}

function fakeVideo({ playFails = false } = {}) {
  const v = {
    srcObject: null,
    videoWidth: 640,
    videoHeight: 480,
    readyState: 4,
    currentTime: 0,
    paused: true,
    async play() {
      if (playFails) throw new Error('autoplay blocked');
      this.paused = false;
    },
    pause() {
      this.paused = true;
    },
    _cbs: new Map(),
    _next: 1,
    requestVideoFrameCallback(cb) {
      const id = this._next++;
      this._cbs.set(id, cb);
      return id;
    },
    cancelVideoFrameCallback(id) {
      this._cbs.delete(id);
    },
    /** Simulate the browser presenting frame number `n` at time `t`. */
    present(t, n) {
      const cbs = [...this._cbs.values()];
      this._cbs.clear();
      for (const cb of cbs) cb(t, { mediaTime: t / 1000, presentedFrames: n });
    },
  };
  return v;
}

const cfg = DEFAULT_CONFIG.camera;

test('starts, reports info, and stop() releases every track and the video element', async () => {
  const { stream, track } = fakeStream();
  let constraints;
  const video = fakeVideo();
  const cam = new CameraManager(cfg, { video, mediaDevices: { getUserMedia: async (c) => ((constraints = c), stream) } });
  const r = await cam.start();
  assert.equal(r.ok, true);
  assert.equal(cam.status, 'live');
  assert.equal(video.srcObject, stream);
  assert.equal(constraints.audio, false, 'never requests the microphone');
  assert.equal(cam.info.width, 640);
  assert.equal(cam.aspect, 640 / 480);
  cam.stop();
  assert.equal(track.stopped, true);
  assert.equal(video.srcObject, null);
  assert.equal(cam.status, 'stopped');
  cam.stop(); // idempotent
});

for (const [name, code] of [
  ['NotAllowedError', 'PERMISSION_DENIED'],
  ['NotFoundError', 'NO_CAMERA'],
  ['NotReadableError', 'CAMERA_BUSY'],
  ['OverconstrainedError', 'CONSTRAINTS'],
  ['WeirdError', 'UNKNOWN'],
]) {
  test(`getUserMedia ${name} maps to ${code}`, async () => {
    const bus = new EventBus();
    const seen = [];
    bus.on('camera', (e) => seen.push(e));
    const err = Object.assign(new Error('x'), { name });
    const cam = new CameraManager(cfg, { video: fakeVideo(), bus, mediaDevices: { getUserMedia: async () => Promise.reject(err) } });
    const r = await cam.start();
    assert.equal(r.ok, false);
    assert.equal(r.error.code, code);
    assert.equal(cam.status, 'error');
    assert.equal(seen.at(-1).status, 'error');
    assert.equal(mapCameraError(err), code);
  });
}

test('missing mediaDevices (insecure context) is reported as UNSUPPORTED', async () => {
  const cam = new CameraManager(cfg, { video: fakeVideo(), mediaDevices: undefined });
  const r = await cam.start();
  assert.equal(r.error.code, 'UNSUPPORTED');
});

test('playback failure releases the stream', async () => {
  const { stream, track } = fakeStream();
  const cam = new CameraManager(cfg, { video: fakeVideo({ playFails: true }), mediaDevices: { getUserMedia: async () => stream } });
  const r = await cam.start();
  assert.equal(r.error.code, 'PLAYBACK_FAILED');
  assert.equal(track.stopped, true);
});

test('stop() during the permission prompt cancels and releases the late stream', async () => {
  const { stream, track } = fakeStream();
  let resolve;
  const cam = new CameraManager(cfg, { video: fakeVideo(), mediaDevices: { getUserMedia: () => new Promise((r) => (resolve = r)) } });
  const p = cam.start();
  cam.stop();
  resolve(stream);
  const r = await p;
  assert.equal(r.ok, false);
  assert.equal(track.stopped, true);
});

test('track "ended" (device unplugged) is reported as INTERRUPTED and stops the frame loop', async () => {
  const { stream, track } = fakeStream();
  const video = fakeVideo();
  const cam = new CameraManager(cfg, { video, mediaDevices: { getUserMedia: async () => stream } });
  await cam.start();
  let frames = 0;
  cam.startFrameLoop(() => frames++);
  video.present(10, 1);
  track.fireEnded();
  assert.equal(cam.status, 'error');
  assert.equal(cam.error.code, 'INTERRUPTED');
  video.present(20, 2);
  assert.equal(frames, 1);
});

test('frame loop delivers only the newest frame and counts dropped frames', async () => {
  const { stream } = fakeStream();
  const video = fakeVideo();
  const cam = new CameraManager(cfg, { video, mediaDevices: { getUserMedia: async () => stream } });
  await cam.start();
  const got = [];
  cam.startFrameLoop((f) => got.push(f.presentedFrames));
  video.present(0, 1);
  video.present(33, 2);
  video.present(100, 5); // frames 3 and 4 were never delivered (processing was busy)
  assert.deepEqual(got, [1, 2, 5]);
  assert.equal(cam.stats.dropped, 2);
  cam.stopFrameLoop();
  video.present(133, 6);
  assert.equal(got.length, 3);
  assert.equal(video._cbs.size, 0, 'no pending callbacks after stop');
});

test('frame loop falls back to rAF polling when requestVideoFrameCallback is unavailable', async () => {
  const { stream } = fakeStream();
  const video = fakeVideo();
  delete video.requestVideoFrameCallback;
  const queue = [];
  const cam = new CameraManager(cfg, {
    video,
    mediaDevices: { getUserMedia: async () => stream },
    requestAnimationFrame: (cb) => (queue.push(cb), queue.length),
    cancelAnimationFrame: () => {},
    now: () => 0,
  });
  await cam.start();
  let n = 0;
  cam.startFrameLoop(() => n++);
  const runRaf = () => queue.splice(0).forEach((cb) => cb());
  video.currentTime = 0.033;
  runRaf();
  runRaf(); // same currentTime: no new frame
  video.currentTime = 0.066;
  runRaf();
  assert.equal(n, 2);
});

test('a throwing frame callback is contained and reported', async () => {
  const { stream } = fakeStream();
  const video = fakeVideo();
  const bus = new EventBus();
  const errs = [];
  bus.on('pipeline_error', (e) => errs.push(e));
  const cam = new CameraManager(cfg, { video, bus, mediaDevices: { getUserMedia: async () => stream } });
  await cam.start();
  const orig = console.error;
  console.error = () => {};
  try {
    cam.startFrameLoop(() => {
      throw new Error('boom');
    });
    video.present(0, 1);
    video.present(33, 2);
  } finally {
    console.error = orig;
  }
  assert.equal(errs.length, 2, 'loop keeps running after an exception');
});

test('listDevices returns only video inputs', async () => {
  const cam = new CameraManager(cfg, {
    video: fakeVideo(),
    mediaDevices: {
      enumerateDevices: async () => [
        { kind: 'audioinput', deviceId: 'a' },
        { kind: 'videoinput', deviceId: 'v1', label: '' },
      ],
    },
  });
  assert.deepEqual(await cam.listDevices(), [{ deviceId: 'v1', label: 'Camera 1' }]);
});
