"""Local face detection for the robot's gaze (MediaPipe Tasks FaceDetector, BlazeFace short range).

Runs on the same frames the engine already owns (no second camera), throttled to
``face.intervalMs``. Results are image-normalised boxes; the browser mirrors them for the
selfie view and decides where the robot looks (src/interaction/AttentionController.js).
Faces never leave the local machine: they are sent only to the local browser page, like
hand landmarks. A missing model or a failed initialisation disables gaze, never tracking.

``FaceBackend`` protocol (swap in another detector by implementing it):
    init() -> dict                      raises on failure
    detect(rgb, ts_ms) -> list[{box: (x, y, w, h, score), bs?: {name: score}, matrix?: [16]}]
    close() -> None
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

log = logging.getLogger("aura_cv.faces")


# head-turn cues from face landmarks (port of faceAnalysis.js headCueFromLandmarks; parity-tested)
CUE_LANDMARKS = {"noseTip": 1, "cheekRight": 234, "cheekLeft": 454, "forehead": 10, "chin": 152}
PITCH_NEUTRAL = 0.565
PITCH_RAW_MAX = 4.0
PITCH_RANGE = 0.045


def head_cue(lms) -> dict | None:
    """Image-space cues (yaw in [-1, 1], pitch in [-4, 4]): yaw + = nose toward image right; pitch + = looking down."""
    if lms is None or len(lms) < 455:
        return None
    try:
        pts = [(float(lms[i][0]), float(lms[i][1])) for i in (1, 234, 454, 10, 152)]
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(v) for q in pts for v in q):
        return None
    (nx, ny), (crx, _), (clx, _), (_, fy), (_, cy) = pts
    half = abs(clx - crx) / 2
    span = cy - fy
    if not (half > 1e-4 and span > 1e-4):
        return None
    c = lambda v, m: max(-m, min(m, v))  # noqa: E731
    yaw = c((nx - (clx + crx) / 2) / half, 1.0)
    # wider raw pitch range: the browser's HeadCueCalibrator subtracts the user's resting value
    pitch = c(((ny - fy) / span - PITCH_NEUTRAL) / PITCH_RANGE, PITCH_RAW_MAX)
    return {"yaw": round(yaw * 1e4) / 1e4, "pitch": round(pitch * 1e4) / 1e4}


class FaceBackend(Protocol):
    def init(self) -> dict: ...
    def detect(self, rgb: np.ndarray, ts_ms: int) -> list: ...
    def close(self) -> None: ...


# Blendshape categories forwarded to the browser (port of BLENDSHAPE_KEYS in faceAnalysis.js)
BLENDSHAPE_KEYS = (
    "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight", "jawOpen",
    "browInnerUp", "browDownLeft", "browDownRight", "browOuterUpLeft", "browOuterUpRight",
    "eyeWideLeft", "eyeWideRight", "eyeBlinkLeft", "eyeBlinkRight", "cheekSquintLeft", "cheekSquintRight",
    "mouthPressLeft", "mouthPressRight",
    "eyeLookInLeft", "eyeLookOutLeft", "eyeLookInRight", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeLookDownLeft", "eyeLookDownRight",
)


class MediaPipeFaceBackend:
    """FaceLandmarker (landmarks + blendshapes + head-pose matrix) or, if its model is missing,
    the BlazeFace detector (boxes only: gaze without expressions)."""

    def __init__(self, model_path: Path, min_score: float, detector_path: Path | None = None, num_faces: int = 2) -> None:
        self.model_path = Path(model_path)
        self.detector_path = Path(detector_path) if detector_path else None
        self.min_score = min_score
        self.num_faces = num_faces
        self._task = None
        self._mp = None
        self.kind = None  # "landmarker" | "detector"

    def init(self) -> dict:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._mp = mp
        if self.model_path.is_file():
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path), delegate=BaseOptions.Delegate.CPU),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=self.num_faces,
                min_face_detection_confidence=self.min_score,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            self._task = vision.FaceLandmarker.create_from_options(options)
            self.kind = "landmarker"
            return {"delegate": "CPU", "model": f"FaceLandmarker ({self.model_path.name})"}
        if self.detector_path is not None and self.detector_path.is_file():
            options = vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(self.detector_path), delegate=BaseOptions.Delegate.CPU),
                running_mode=vision.RunningMode.VIDEO,
                min_detection_confidence=self.min_score,
            )
            self._task = vision.FaceDetector.create_from_options(options)
            self.kind = "detector"
            return {"delegate": "CPU", "model": f"FaceDetector ({self.detector_path.name}; landmarker missing: no expressions)"}
        raise FileNotFoundError(f"face model not found at {self.model_path} (run npm run setup)")

    def detect(self, rgb: np.ndarray, ts_ms: int) -> list:
        h, w = rgb.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        out = []
        if self.kind == "landmarker":
            res = self._task.detect_for_video(image, ts_ms)
            for i, lms in enumerate(res.face_landmarks or []):
                xs = [p.x for p in lms]
                ys = [p.y for p in lms]
                bs = {}
                if res.face_blendshapes and i < len(res.face_blendshapes):
                    for c in res.face_blendshapes[i]:
                        if c.category_name in BLENDSHAPE_KEYS:
                            bs[c.category_name] = round(float(c.score), 3)
                matrix = None
                if res.facial_transformation_matrixes and i < len(res.facial_transformation_matrixes):
                    m = np.asarray(res.facial_transformation_matrixes[i], dtype=float).reshape(4, 4)
                    matrix = [round(float(v), 4) for v in m.flatten(order="F")]  # column-major, like the JS API
                out.append({"box": (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), 1.0), "bs": bs, "matrix": matrix,
                            "cue": head_cue([(p.x, p.y) for p in lms])})
            return out
        for d in self._task.detect_for_video(image, ts_ms).detections:
            b = d.bounding_box
            score = d.categories[0].score if d.categories else 0.0
            out.append({"box": (b.origin_x / w, b.origin_y / h, b.width / w, b.height / h, float(score))})
        return out

    def close(self) -> None:
        if self._task is not None:
            self._task.close()
            self._task = None


class FaceTracker:
    """Throttled, validated face detection. Status: disabled | uninitialized | ready | error | closed."""

    def __init__(self, backend: FaceBackend | None, interval_ms: float, min_score: float) -> None:
        self.backend = backend
        self.interval_ms = interval_ms
        self.min_score = min_score
        self.status = "disabled" if backend is None else "uninitialized"
        self.error: str | None = None
        self.last_t = float("-inf")
        self.last_ts = -1
        self.inference_ms: float | None = None

    def init(self) -> bool:
        if self.backend is None:
            return False
        try:
            self.backend.init()
            self.status = "ready"
            return True
        except Exception as exc:  # missing model / unsupported platform: gaze disabled only
            self.status = "error"
            self.error = str(exc)
            log.warning("face detection unavailable: %s", exc)
            return False

    def due(self, t: float) -> bool:
        return self.status == "ready" and t - self.last_t >= self.interval_ms

    def detect(self, rgb: np.ndarray, t: float) -> list[dict] | None:
        """Faces in this frame, or None if detection did not run (not due / failed)."""
        if not self.due(t):
            return None
        self.last_t = t
        ts = max(int(t), self.last_ts + 1)
        self.last_ts = ts
        t0 = time.perf_counter()
        try:
            raw = self.backend.detect(rgb, ts)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("face detection failed: %r", exc)
            return None
        self.inference_ms = (time.perf_counter() - t0) * 1000
        faces = []
        for item in raw:
            # backends return {"box": (x, y, w, h, score), "bs"?, "matrix"?}; a bare tuple is a box
            box = item.get("box") if isinstance(item, dict) else item
            try:
                x, y, w, h, score = box
            except (TypeError, ValueError):
                continue
            vals = (x, y, w, h, score)
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
                continue
            if score < self.min_score or not (0 < w <= 1.5 and 0 < h <= 1.5):
                continue
            face = {"x": round(x + w / 2, 4), "y": round(y + h / 2, 4), "w": round(w, 4), "h": round(h, 4), "score": round(score, 3)}
            if isinstance(item, dict):
                bs = item.get("bs")
                if isinstance(bs, dict):
                    face["bs"] = {k: v for k, v in bs.items() if isinstance(v, (int, float)) and math.isfinite(v)}
                m = item.get("matrix")
                if isinstance(m, list) and len(m) == 16 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in m):
                    face["matrix"] = m
                cue = item.get("cue")
                if isinstance(cue, dict) and all(isinstance(cue.get(k), (int, float)) and math.isfinite(cue[k]) for k in ("yaw", "pitch")):
                    face["cue"] = {"yaw": cue["yaw"], "pitch": cue["pitch"]}
            faces.append(face)
        return faces

    def info(self) -> dict[str, Any]:
        return {"status": self.status, "error": self.error, "intervalMs": self.interval_ms}

    def close(self) -> None:
        if self.backend is not None and self.status == "ready":
            try:
                self.backend.close()
            except Exception:  # pragma: no cover
                pass
        self.status = "closed" if self.backend is not None else self.status
