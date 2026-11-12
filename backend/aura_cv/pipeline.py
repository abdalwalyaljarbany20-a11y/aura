"""Processing pipeline: frame -> validation -> landmarks -> features -> gestures -> result.

``FrameProcessor.process`` is a pure step (given its tracker), unit-tested with fake
backends. ``ProcessingLoop`` runs it on a dedicated thread against the camera's
latest-frame slot and hands each result to a callback; it never queues frames.

Result schema (protocol version 1, JSON-safe; see docs/PYTHON_ENGINE.md):

  type: "result", session, frameId (both strictly identify the frame), capturedAt (ms, engine clock),
  status: DETECTED | NOT_DETECTED | TRACKING_LOST | CAMERA_UNAVAILABLE | PROCESSING_ERROR,
  image: {width, height}, hands: [HandObservation.to_dict()], primary: index | null,
  features: primary-hand feature summary | null,
  gesture: {raw, label, confidence, reason, scores, stable, stableConfidence},
  events: [{type: start|end|event, gesture, confidence, ...}],
  timing: {acquireMs, queueMs, inferenceMs, featuresMs, gestureMs, totalMs, warmup},
  error: string | null
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

import numpy as np

from .camera import CameraManager, Frame, validate_frame
from .features import extract_features
from .gestures import GestureRecognitionEngine
from .metrics import RateMeter, RollingStats, now_ms
from .tracking import HandTracker, TrackingStateMachine

log = logging.getLogger("aura_cv.pipeline")


def _bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _r(v, n: int = 3):
    return None if v is None else round(float(v), n)


class FrameProcessor:
    def __init__(self, tracker: HandTracker, gestures_cfg: dict, *, mirror: bool = True,
                 lost_hold_ms: float = 500.0, to_rgb: Callable[[np.ndarray], np.ndarray] = _bgr_to_rgb,
                 session: int = 1, face_tracker=None, pose_net=None) -> None:
        self.tracker = tracker
        self.face_tracker = face_tracker  # optional FaceTracker (gaze); runs on the same frame, throttled
        self.session_id = session
        self.engine = GestureRecognitionEngine(gestures_cfg, pose_net)
        self.mirror = mirror
        self.to_rgb = to_rgb
        self.state = TrackingStateMachine(lost_hold_ms)
        self.metrics = {
            "processed": RateMeter(1000),
            "inferenceMs": RollingStats(120),
            "totalMs": RollingStats(120),
            "warmupInferenceMs": None,
        }
        self.session = {"frames": 0, "framesWithHand": 0, "processingErrors": 0, "invalidFrames": 0,
                        "gestureStarts": 0, "swipes": 0}

    def process(self, frame: Frame) -> dict:
        t_start = now_ms()
        t = frame.captured_at
        h, w = (frame.image.shape[:2] if isinstance(frame.image, np.ndarray) and frame.image.ndim >= 2 else (0, 0))
        base = {"type": "result", "session": self.session_id, "frameId": frame.id, "capturedAt": _r(t, 2), "image": {"width": int(w), "height": int(h)},
                "hands": [], "primary": None, "features": None, "events": [], "error": None}

        invalid = validate_frame(frame.image)
        if invalid:
            self.session["invalidFrames"] += 1
            return self._error(base, f"invalid frame: {invalid}", t, t_start, frame)

        rgb = self.to_rgb(frame.image)
        det = self.tracker.detect(rgb, t)
        # faces only on frames where detection ran (throttled); absent otherwise
        faces = self.face_tracker.detect(rgb, t) if self.face_tracker is not None else None
        if not det.ok:
            self.session["processingErrors"] += 1
            return self._error(base, det.error or "inference failed", t, t_start, frame, det.inference_ms)

        warmup = self.metrics["warmupInferenceMs"] is None
        if warmup:
            self.metrics["warmupInferenceMs"] = det.inference_ms
        else:
            self.metrics["inferenceMs"].push(det.inference_ms)

        t_feat0 = now_ms()
        aspect = w / h
        feats = [extract_features(o.landmarks, o.world, aspect=aspect, mirror=self.mirror,
                                  score=o.score, handedness=o.handedness, model=o.model) for o in det.hands]
        primary = det.primary
        pf = feats[primary] if primary is not None and feats[primary].valid else None
        t_gest0 = now_ms()
        g = self.engine.update(pf, t)
        t_end = now_ms()

        status = self.state.update(pf is not None, t)
        self.session["frames"] += 1
        if pf is not None:
            self.session["framesWithHand"] += 1
        for ev in g["events"]:
            if ev["type"] == "start":
                self.session["gestureStarts"] += 1
            elif ev["type"] == "event":
                self.session["swipes"] += 1

        total = t_end - t
        if not warmup:
            self.metrics["totalMs"].push(total)
        self.metrics["processed"].tick(t_end)

        base.update({
            "status": status,
            # Landmarks are only reported for hands detected in *this* frame.
            "hands": [o.to_dict() for o in det.hands],
            "primary": primary if pf is not None else None,
            "features": pf.to_dict() if pf is not None else None,
            "gesture": self._gesture_dict(g),
            "events": [self._event_dict(e) for e in g["events"]],
            "timing": {"acquireMs": _r(frame.acquire_ms), "queueMs": _r(t_start - t), "inferenceMs": _r(det.inference_ms),
                       "featuresMs": _r(t_gest0 - t_feat0), "gestureMs": _r(t_end - t_gest0), "totalMs": _r(total),
                       "warmup": warmup},
            "rejectedHands": det.rejected,
        })
        if faces is not None:
            base["faces"] = faces
        return base

    def camera_unavailable(self, error: dict | None, t: float) -> dict:
        """Result emitted when the camera stops delivering frames (releases gestures)."""
        g = self.engine.update(None, t)
        return {"type": "result", "session": self.session_id, "frameId": None, "capturedAt": _r(t, 2), "status": self.state.set("CAMERA_UNAVAILABLE"),
                "image": None, "hands": [], "primary": None, "features": None, "gesture": self._gesture_dict(g),
                "events": [self._event_dict(e) for e in g["events"]], "timing": None,
                "error": (error or {}).get("message", "camera unavailable")}

    def _error(self, base: dict, message: str, t: float, t_start: float, frame: Frame, inference_ms: float = 0.0) -> dict:
        # Release gestures/holds exactly as if the hand disappeared; report no landmarks.
        g = self.engine.update(None, t)
        base.update({
            "status": self.state.set("PROCESSING_ERROR"),
            "gesture": self._gesture_dict(g),
            "events": [self._event_dict(e) for e in g["events"]],
            "timing": {"acquireMs": _r(frame.acquire_ms), "queueMs": _r(t_start - t), "inferenceMs": _r(inference_ms),
                       "featuresMs": None, "gestureMs": None, "totalMs": _r(now_ms() - t), "warmup": False},
            "error": message,
        })
        return base

    @staticmethod
    def _gesture_dict(g: dict) -> dict:
        raw = g["raw"]
        return {"raw": raw["gesture"], "label": raw["label"], "confidence": _r(raw["confidence"], 4),
                "reason": raw["reason"], "source": raw.get("source", "rules"), "scores": {k: _r(v, 4) for k, v in raw["scores"].items()},
                "stable": g["stable"]["gesture"], "stableConfidence": _r(g["stable"]["confidence"], 4),
                "speed": _r(g["speed"], 3),
                "motion": {"spin": {"active": g["motion"]["spin"]["active"], "omega": _r(g["motion"]["spin"]["omega"], 4)},
                           "walk": {"active": g["motion"]["walk"]["active"], "cadence": _r(g["motion"]["walk"]["cadence"], 4)}}}

    @staticmethod
    def _event_dict(e: dict) -> dict:
        out = {"type": e["type"], "gesture": e["gesture"], "t": _r(e["t"], 2), "confidence": _r(e.get("confidence"), 4)}
        if "detail" in e:
            out["detail"] = {k: _r(v, 3) for k, v in e["detail"].items() if k != "gesture"}
        if "durationMs" in e:
            out["durationMs"] = _r(e["durationMs"], 1)
        return out

    def metrics_snapshot(self, camera: CameraManager | None, now: float) -> dict:
        return {
            "cameraFps": _r(camera.capture_rate.rate(now), 2) if camera else None,
            "processedFps": _r(self.metrics["processed"].rate(now), 2),
            "inferenceMs": {k: _r(v) if isinstance(v, float) else v for k, v in self.metrics["inferenceMs"].summary().items()},
            "totalMs": {k: _r(v) if isinstance(v, float) else v for k, v in self.metrics["totalMs"].summary().items()},
            "warmupInferenceMs": _r(self.metrics["warmupInferenceMs"]),
            "droppedFrames": camera.dropped if camera else None,
            "trackingLosses": self.state.losses,
            "session": dict(self.session),
        }


class ProcessingLoop:
    """Runs FrameProcessor on the newest camera frame until stopped."""

    def __init__(self, camera: CameraManager, processor: FrameProcessor,
                 on_result: Callable[[dict, Frame | None], None]) -> None:
        self.camera = camera
        self.processor = processor
        self.on_result = on_result
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.errors = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="aura-processing", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout_s)
        self._thread = None

    def _run(self) -> None:
        last_id = 0
        while not self._stop.is_set():
            frame = self.camera.slot.get(last_id, timeout_s=0.5)
            if frame is None:
                if self.camera.status == "error":
                    self._emit(self.processor.camera_unavailable(self.camera.error, now_ms()), None)
                    return
                continue
            last_id = frame.id
            try:
                result = self.processor.process(frame)
            except Exception as exc:  # never fabricate a result; report the failure
                self.errors += 1
                log.exception("frame processing failed")
                result = self.processor._error(
                    {"type": "result", "session": self.processor.session_id, "frameId": frame.id, "capturedAt": _r(frame.captured_at, 2), "image": None,
                     "hands": [], "primary": None, "features": None, "events": [], "error": None},
                    f"pipeline exception: {exc!r}", frame.captured_at, now_ms(), frame)
            self._emit(result, frame)

    def _emit(self, result: dict, frame: Frame | None) -> None:
        try:
            self.on_result(result, frame)
        except Exception:  # pragma: no cover - consumer bug must not kill the loop
            log.exception("result consumer failed")
