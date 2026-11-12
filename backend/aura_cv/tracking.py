"""Hand landmark inference and tracking-state management.

``HandBackend`` protocol (swap in another landmark model by implementing it):

    init() -> dict                      # {"delegate": ..., "model": ...}; raises on failure
    detect(rgb: ndarray, ts_ms: int)    # MediaPipe-shaped result object
    close() -> None

``MediaPipeHandBackend`` uses the MediaPipe Tasks API (``mediapipe.tasks.python.vision
.HandLandmarker``, VIDEO running mode). The legacy ``mediapipe.solutions.hands`` API is
not present in mediapipe 1.x.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .landmarks import as_landmark_array, landmark_set_error
from .metrics import now_ms

log = logging.getLogger("aura_cv.tracking")


class HandBackend(Protocol):
    def init(self) -> dict: ...
    def detect(self, rgb: np.ndarray, ts_ms: int) -> Any: ...
    def close(self) -> None: ...


class MediaPipeHandBackend:
    """GestureRecognizer (DL landmarks + DL gesture classes) if its model exists, else HandLandmarker."""

    def __init__(self, model_path: Path, tracking_cfg: dict, gesture_model_path: Path | None = None) -> None:
        self.model_path = Path(model_path)
        self.gesture_model_path = Path(gesture_model_path) if gesture_model_path else None
        self.cfg = tracking_cfg
        self._landmarker = None
        self._mp = None
        self.kind = None  # "gesture" | "landmarks"

    def init(self) -> dict:
        use_gesture = self.gesture_model_path is not None and self.gesture_model_path.is_file()
        if not use_gesture and not self.model_path.is_file():
            raise FileNotFoundError(f"hand model not found at {self.model_path} (run \"npm run setup\")")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._mp = mp
        requested = self.cfg.get("delegate", "CPU")
        delegates = ["GPU", "CPU"] if requested == "GPU" else ["CPU"]
        last_exc: Exception | None = None
        for d in delegates:
            try:
                common = dict(
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=self.cfg["numHands"],
                    min_hand_detection_confidence=self.cfg["minHandDetectionConfidence"],
                    min_hand_presence_confidence=self.cfg["minHandPresenceConfidence"],
                    min_tracking_confidence=self.cfg["minTrackingConfidence"],
                )
                if use_gesture:
                    from mediapipe.tasks.python.components.processors import ClassifierOptions

                    options = vision.GestureRecognizerOptions(
                        base_options=BaseOptions(model_asset_path=str(self.gesture_model_path), delegate=getattr(BaseOptions.Delegate, d)),
                        canned_gesture_classifier_options=ClassifierOptions(score_threshold=0.0),
                        **common,
                    )
                    self._landmarker = vision.GestureRecognizer.create_from_options(options)
                    self.kind = "gesture"
                    model = f"GestureRecognizer ({self.gesture_model_path.name})"
                else:
                    options = vision.HandLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=str(self.model_path), delegate=getattr(BaseOptions.Delegate, d)),
                        **common,
                    )
                    self._landmarker = vision.HandLandmarker.create_from_options(options)
                    self.kind = "landmarks"
                    model = f"HandLandmarker ({self.model_path.name}; gesture model missing: rule fallback)"
                delegate = d if d == requested else f"{d} ({requested} unavailable)"
                return {"delegate": delegate, "model": model, "platform": platform.system()}
            except Exception as exc:  # GPU is not supported by the Windows Python package
                last_exc = exc
                log.warning("HandLandmarker init with %s delegate failed: %r", d, exc)
        raise RuntimeError(f"HandLandmarker initialisation failed: {last_exc!r}")

    def detect(self, rgb: np.ndarray, ts_ms: int):
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.kind == "gesture":
            return self._landmarker.recognize_for_video(image, ts_ms)
        return self._landmarker.detect_for_video(image, ts_ms)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


def to_model_gesture(categories) -> dict | None:
    """GestureRecognizer categories -> {label, score, scores} (port of toModelGesture)."""
    scores, best = {}, None
    for c in categories:
        name = getattr(c, "category_name", None)
        sc = getattr(c, "score", None)
        if not isinstance(name, str) or not isinstance(sc, (int, float)) or not np.isfinite(sc):
            continue
        scores[name] = float(sc)
        if best is None or sc > best[1]:
            best = (name, float(sc))
    return {"label": best[0], "score": best[1], "scores": scores} if best else None


@dataclass
class HandObservation:
    landmarks: np.ndarray  # (21, 3) normalised image coordinates (unmirrored)
    world: np.ndarray | None  # (21, 3) metres, or None
    handedness: str | None
    score: float | None  # handedness classification score (MediaPipe's per-hand confidence)
    model: dict | None = None  # learned gesture classifier output {label, score, scores} or None

    def to_dict(self) -> dict:
        r = lambda a: np.round(a, 5).tolist()  # noqa: E731
        return {
            "landmarks": r(self.landmarks),
            "worldLandmarks": r(self.world) if self.world is not None else None,
            "handedness": self.handedness,
            "score": None if self.score is None else round(float(self.score), 4),
            "model": None if self.model is None else {"label": self.model["label"], "score": round(self.model["score"], 4)},
        }


@dataclass
class DetectionResult:
    ok: bool
    hands: list[HandObservation] = field(default_factory=list)
    primary: int | None = None
    inference_ms: float = 0.0
    rejected: int = 0
    error: str | None = None


class HandTracker:
    """Backend wrapper: monotonic timestamps, result validation, primary-hand selection."""

    def __init__(self, backend: HandBackend) -> None:
        self.backend = backend
        self.status = "uninitialized"  # uninitialized | ready | error | closed
        self.error: str | None = None
        self.info: dict = {}
        self._last_ts = -1
        self._primary_handedness: str | None = None
        self.stats = {"frames": 0, "failures": 0, "rejectedHands": 0}

    def init(self) -> bool:
        try:
            self.info = self.backend.init() or {}
            self.status = "ready"
            return True
        except Exception as exc:
            self.status = "error"
            self.error = str(exc)
            log.error("tracker init failed: %s", exc)
            return False

    def detect(self, rgb: np.ndarray, t_ms: float) -> DetectionResult:
        if self.status != "ready":
            return DetectionResult(ok=False, error=f"tracker {self.status}")
        ts = int(t_ms)
        if ts <= self._last_ts:  # MediaPipe requires strictly increasing timestamps
            ts = self._last_ts + 1
        self._last_ts = ts
        t0 = now_ms()
        try:
            raw = self.backend.detect(rgb, ts)
        except Exception as exc:
            self.stats["failures"] += 1
            return DetectionResult(ok=False, inference_ms=now_ms() - t0, error=repr(exc))
        inference_ms = now_ms() - t0
        self.stats["frames"] += 1
        hands, rejected = self.parse(raw)
        self.stats["rejectedHands"] += rejected
        return DetectionResult(ok=True, hands=hands, primary=self.select_primary(hands),
                               inference_ms=inference_ms, rejected=rejected)

    @staticmethod
    def parse(raw: Any) -> tuple[list[HandObservation], int]:
        hands: list[HandObservation] = []
        rejected = 0
        lms = getattr(raw, "hand_landmarks", None) or []
        worlds = getattr(raw, "hand_world_landmarks", None) or []
        handed = getattr(raw, "handedness", None) or []
        gestures = getattr(raw, "gestures", None) or []
        for i, pts in enumerate(lms):
            arr = as_landmark_array(pts)
            if landmark_set_error(arr, normalized=True):
                rejected += 1
                continue
            world = as_landmark_array(worlds[i]) if i < len(worlds) else None
            if world is not None and landmark_set_error(world):
                world = None
            cat = handed[i][0] if i < len(handed) and handed[i] else None
            score = getattr(cat, "score", None)
            hands.append(HandObservation(
                landmarks=arr,
                world=world,
                handedness=getattr(cat, "category_name", None) or None,
                score=float(score) if isinstance(score, (int, float)) and np.isfinite(score) else None,
                model=to_model_gesture(gestures[i]) if i < len(gestures) and gestures[i] else None,
            ))
        return hands, rejected

    def select_primary(self, hands: list[HandObservation]) -> int | None:
        """Keep following the previous primary's handedness; else the largest hand."""
        if not hands:
            self._primary_handedness = None
            return None
        idx = None
        if self._primary_handedness:
            idx = next((i for i, h in enumerate(hands) if h.handedness == self._primary_handedness), None)
        if idx is None:
            areas = [float(np.ptp(h.landmarks[:, 0]) * np.ptp(h.landmarks[:, 1])) for h in hands]
            idx = int(np.argmax(areas))
        self._primary_handedness = hands[idx].handedness
        return idx

    def close(self) -> None:
        if self.status == "closed":
            return
        try:
            self.backend.close()
        except Exception as exc:  # pragma: no cover
            log.warning("tracker close failed: %r", exc)
        self.status = "closed"


class TrackingStateMachine:
    """Per-frame tracking status.

    DETECTED            a valid primary hand in this frame
    TRACKING_LOST       a hand was detected recently (within ``lost_hold_ms``) but not now;
                        no landmarks are reported for this frame
    NOT_DETECTED        no hand, and none recently
    CAMERA_UNAVAILABLE  no frames (camera not open / interrupted)
    PROCESSING_ERROR    inference failed for this frame
    """

    def __init__(self, lost_hold_ms: float = 500.0) -> None:
        self.lost_hold_ms = lost_hold_ms
        self.status = "NOT_DETECTED"
        self._last_detected: float | None = None
        self.losses = 0

    def update(self, detected: bool, t: float) -> str:
        if detected:
            self._last_detected = t
            self.status = "DETECTED"
        elif self._last_detected is not None and t - self._last_detected <= self.lost_hold_ms:
            if self.status == "DETECTED":
                self.losses += 1
            self.status = "TRACKING_LOST"
        else:
            if self.status == "DETECTED":
                self.losses += 1
            self._last_detected = None
            self.status = "NOT_DETECTED"
        return self.status

    def set(self, status: str) -> str:
        """CAMERA_UNAVAILABLE / PROCESSING_ERROR; clears recent-detection memory."""
        if self.status == "DETECTED":
            self.losses += 1
        self._last_detected = None
        self.status = status
        return status
