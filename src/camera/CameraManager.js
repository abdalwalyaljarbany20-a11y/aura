/**
 * CameraManager
 *
 * Owns the MediaStream and the <video> element it feeds.
 *  - start(): requests the camera, maps failures to explicit error codes
 *  - stop(): stops every track and detaches the stream (idempotent)
 *  - frame loop: invokes a callback once per *new* presented video frame,
 *    using requestVideoFrameCallback when available (falls back to
 *    requestAnimationFrame + currentTime polling). Only the most recent
 *    frame is ever handed out; there is no queue, so slow processing drops
 *    stale frames instead of accumulating latency. Drops are counted.
 *  - detects interrupted capture (track "ended", e.g. device unplugged).
 *
 * All browser objects are injectable so the class is unit-testable.
 */

export const CAMERA_ERRORS = Object.freeze({
  UNSUPPORTED: 'Camera API unavailable (requires a secure context: https or http://localhost)',
  PERMISSION_DENIED: 'Camera permission was denied',
  NO_CAMERA: 'No camera device was found',
  CAMERA_BUSY: 'The camera is in use by another application or could not be started',
  CONSTRAINTS: 'The requested camera settings are not supported by this device',
  PLAYBACK_FAILED: 'The camera stream could not be played',
  INTERRUPTED: 'Camera capture was interrupted (device disconnected or disabled)',
  UNKNOWN: 'Unknown camera error',
});

/** @param {any} err */
export function mapCameraError(err) {
  switch (err?.name) {
    case 'NotAllowedError':
    case 'SecurityError':
    case 'PermissionDeniedError':
      return 'PERMISSION_DENIED';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return 'NO_CAMERA';
    case 'NotReadableError':
    case 'TrackStartError':
    case 'AbortError':
      return 'CAMERA_BUSY';
    case 'OverconstrainedError':
    case 'ConstraintNotSatisfiedError':
      return 'CONSTRAINTS';
    default:
      return 'UNKNOWN';
  }
}

export class CameraManager {
  /**
   * @param {object} cfg camera configuration
   * @param {{ video: any, mediaDevices?: any, bus?: any, now?: () => number,
   *           requestAnimationFrame?: Function, cancelAnimationFrame?: Function }} deps
   */
  constructor(cfg, deps) {
    this.cfg = cfg;
    this.video = deps.video;
    this.mediaDevices = 'mediaDevices' in deps ? deps.mediaDevices : globalThis.navigator?.mediaDevices;
    this.bus = deps.bus ?? null;
    this.now = deps.now ?? (() => performance.now());
    this.raf = deps.requestAnimationFrame ?? globalThis.requestAnimationFrame?.bind(globalThis);
    this.caf = deps.cancelAnimationFrame ?? globalThis.cancelAnimationFrame?.bind(globalThis);
    /** @type {'stopped'|'starting'|'live'|'error'} */
    this.status = 'stopped';
    this.error = null; // { code, message, detail }
    this.stream = null;
    this.info = null;
    this._loop = null;
    this._onEnded = () => this._interrupted();
    this.stats = { presented: 0, delivered: 0, dropped: 0 };
  }

  /**
   * @param {string|null} [deviceId] overrides cfg.deviceId
   * @returns {Promise<{ ok: boolean, error?: { code: string, message: string } }>}
   */
  async start(deviceId = this.cfg.deviceId) {
    if (this.status === 'live' || this.status === 'starting') this.stop();
    this.error = null;
    if (!this.mediaDevices?.getUserMedia) return this._fail('UNSUPPORTED');

    this._setStatus('starting');
    const video = {
      width: { ideal: this.cfg.width },
      height: { ideal: this.cfg.height },
      frameRate: { ideal: this.cfg.frameRate },
    };
    if (deviceId) video.deviceId = { exact: deviceId };
    else video.facingMode = this.cfg.facingMode;

    let stream;
    try {
      stream = await this.mediaDevices.getUserMedia({ video, audio: false });
    } catch (err) {
      return this._fail(mapCameraError(err), err?.message);
    }
    if (this.status !== 'starting') {
      // stop() was called while the permission prompt was open.
      stream.getTracks().forEach((tr) => tr.stop());
      return { ok: false, error: { code: 'ABORTED', message: 'Camera start was cancelled' } };
    }
    this.stream = stream;
    const track = stream.getVideoTracks()[0];
    if (!track) {
      this.stop();
      return this._fail('NO_CAMERA', 'stream has no video track');
    }
    track.addEventListener?.('ended', this._onEnded);

    try {
      this.video.srcObject = stream;
      this.video.muted = true;
      this.video.playsInline = true;
      await this.video.play();
    } catch (err) {
      this._releaseStream();
      return this._fail('PLAYBACK_FAILED', err?.message);
    }

    const s = track.getSettings?.() ?? {};
    this.info = {
      label: track.label || 'camera',
      deviceId: s.deviceId ?? deviceId ?? null,
      width: this.video.videoWidth || s.width || null,
      height: this.video.videoHeight || s.height || null,
      settingsFrameRate: s.frameRate ?? null, // reported by the driver, not measured
    };
    this._setStatus('live');
    this.bus?.emit('camera', { status: 'live', info: this.info });
    return { ok: true };
  }

  /** Stop capture and release the device. Safe to call repeatedly. */
  stop() {
    this.stopFrameLoop();
    const wasActive = this.status === 'live' || this.status === 'starting';
    this._releaseStream();
    if (wasActive) {
      this._setStatus('stopped');
      this.bus?.emit('camera', { status: 'stopped' });
    }
  }

  get aspect() {
    const w = this.video?.videoWidth || this.info?.width;
    const h = this.video?.videoHeight || this.info?.height;
    return w && h ? w / h : 16 / 9;
  }

  /** @returns {Promise<{ deviceId: string, label: string }[]>} */
  async listDevices() {
    if (!this.mediaDevices?.enumerateDevices) return [];
    const all = await this.mediaDevices.enumerateDevices();
    return all.filter((d) => d.kind === 'videoinput').map((d, i) => ({ deviceId: d.deviceId, label: d.label || `Camera ${i + 1}` }));
  }

  /**
   * Call `onFrame({ video, t, capturedAt, timeSource, mediaTime, presentedFrames })`
   * once per new frame. All times use the performance.now() clock:
   *   t           when the frame callback ran (monotonic; used as model timestamp)
   *   capturedAt  earliest known frame time: captureTime > presentationTime > t
   *   timeSource  which of those `capturedAt` is ('capture'|'presentation'|'callback')
   * @param {(frame: object) => void} onFrame
   */
  startFrameLoop(onFrame) {
    this.stopFrameLoop();
    const v = this.video;
    const token = { active: true, id: null, rvfc: typeof v.requestVideoFrameCallback === 'function' };
    this._loop = token;
    let lastPresented = null;
    let lastMediaTime = -1;

    const deliver = (t, mediaTime, presentedFrames, capturedAt = t, timeSource = 'callback') => {
      if (lastPresented !== null && presentedFrames !== null) {
        const gap = presentedFrames - lastPresented;
        if (gap > 1) this.stats.dropped += gap - 1;
        this.stats.presented += Math.max(0, gap);
      } else {
        this.stats.presented++;
      }
      lastPresented = presentedFrames;
      this.stats.delivered++;
      try {
        onFrame({ video: v, t, capturedAt, timeSource, mediaTime, presentedFrames });
      } catch (err) {
        console.error('[CameraManager] frame callback failed', err);
        this.bus?.emit('pipeline_error', { message: err?.message ?? String(err) });
      }
    };

    if (token.rvfc) {
      const step = (now, meta) => {
        if (!token.active) return;
        // Earliest known timestamp of this frame: capture time (if the
        // browser reports it), else the time it was handed to the compositor.
        const capturedAt = meta?.captureTime ?? meta?.presentationTime ?? now;
        const source = meta?.captureTime ? 'capture' : meta?.presentationTime ? 'presentation' : 'callback';
        deliver(now, meta?.mediaTime ?? null, meta?.presentedFrames ?? null, capturedAt, source);
        if (token.active) token.id = v.requestVideoFrameCallback(step);
      };
      token.id = v.requestVideoFrameCallback(step);
    } else {
      const step = () => {
        if (!token.active) return;
        if (v.readyState >= 2 && v.currentTime !== lastMediaTime) {
          lastMediaTime = v.currentTime;
          deliver(this.now(), v.currentTime, null);
        }
        if (token.active) token.id = this.raf(step);
      };
      token.id = this.raf(step);
    }
  }

  stopFrameLoop() {
    const token = this._loop;
    if (!token) return;
    token.active = false;
    if (token.id !== null) {
      if (token.rvfc) this.video.cancelVideoFrameCallback?.(token.id);
      else this.caf?.(token.id);
    }
    this._loop = null;
  }

  _interrupted() {
    if (this.status !== 'live') return;
    this.stopFrameLoop();
    this._releaseStream();
    this._fail('INTERRUPTED');
  }

  _releaseStream() {
    if (this.stream) {
      for (const tr of this.stream.getTracks()) {
        tr.removeEventListener?.('ended', this._onEnded);
        tr.stop();
      }
    }
    this.stream = null;
    if (this.video) {
      try {
        this.video.pause?.();
      } catch {
        /* ignore */
      }
      this.video.srcObject = null;
    }
  }

  _fail(code, detail) {
    this.error = { code, message: CAMERA_ERRORS[code] ?? CAMERA_ERRORS.UNKNOWN, detail: detail ?? null };
    this._setStatus('error');
    this.bus?.emit('camera', { status: 'error', error: this.error });
    return { ok: false, error: this.error };
  }

  _setStatus(s) {
    this.status = s;
  }
}
