"""Configuration for the Python engine.

The Python engine reads the *same* optional file as the frontend
(``public/aura.config.json``). It uses these sections:

  camera    width, height, frameRate, mirror          (shared with the browser engine)
  tracking  numHands, delegate, min*Confidence        (shared)
  gestures  every key                                  (shared; same rules as the JS engine)
  backend   Python-only settings (see DEFAULT_CONFIG["backend"])

Defaults for the shared sections mirror ``src/config/defaultConfig.js``;
``tests/test_config.py`` checks that they stay identical. Invalid values are
rejected individually (the default is kept) and reported.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Callable

STATIC_GESTURES = ("OPEN_PALM", "FIST", "PINCH", "POINT", "TWO_FINGERS", "THREE_FINGERS", "THUMB_UP", "THUMB_DOWN")

DEFAULT_CONFIG: dict[str, Any] = {
    "camera": {"width": 1280, "height": 720, "frameRate": 30, "mirror": True},
    "tracking": {
        "delegate": "GPU",
        "numHands": 1,
        "minHandDetectionConfidence": 0.5,
        "minHandPresenceConfidence": 0.5,
        "minTrackingConfidence": 0.5,
    },
    "gestures": {
        "model": {"enabled": True, "minScore": 0.6},
        # project-trained hand-pose net (training/; weights in backend.handPoseNetPath)
        "net": {"enabled": True, "minScore": 0.7},
        "minTrackingScore": 0.6,
        "minInFrameRatio": 0.9,
        "minScore": 0.6,
        "minMargin": 0.2,
        "exitMs": 180,
        "maxStillSpeed": 1.5,
        "postSwipeSuppressMs": 700,
        "definitions": {
            "OPEN_PALM": {"enabled": True, "holdMs": 500, "requireStill": True},
            "FIST": {"enabled": True, "holdMs": 600, "requireStill": True},
            "PINCH": {"enabled": True, "holdMs": 150, "requireStill": False},
            "POINT": {"enabled": True, "holdMs": 120, "requireStill": False},
            "TWO_FINGERS": {"enabled": True, "holdMs": 150, "requireStill": False},
            "THREE_FINGERS": {"enabled": True, "holdMs": 150, "requireStill": False},
            "THUMB_UP": {"enabled": True, "holdMs": 500, "requireStill": True},
            "THUMB_DOWN": {"enabled": True, "holdMs": 500, "requireStill": True},
        },
        "swipe": {
            "enabled": True,
            "windowMs": 450,
            "minDistance": 2.2,
            "minSpeed": 4.0,
            "maxVerticalRatio": 0.6,
            "maxGapMs": 150,
            "cooldownMs": 600,
            "oppositeSuppressMs": 1100,
        },
        "pinchTap": {"enabled": True, "maxMs": 600},
        "spin": {
            "enabled": True,
            "windowMs": 700,
            "rateWindowMs": 200,
            "minWindowMs": 160,
            "minSweepDeg": 100,
            "minRadius": 0.25,
            "maxRadiusCv": 0.5,
            "minConsistency": 0.75,
            "maxStepDeg": 150,
            "maxGapMs": 200,
            "poseGraceMs": 150,
            "enterMs": 100,
            "releaseMs": 200,
        },
        "walk": {
            "enabled": True,
            "windowMs": 2000,
            "minAmplitude": 0.12,
            "minReach": 0.6,
            "hysteresis": 0.04,
            "minSteps": 3,
            "maxCorrelation": -0.3,
            "stopMs": 1100,
            "maxStepRate": 8,
            "poseGraceMs": 200,
        },
    },
    # Face detection for the robot's gaze (shared with the browser engine; modelAssetPath is browser-only).
    "face": {"enabled": True, "numFaces": 2, "minScore": 0.6, "intervalMs": 100},
    "backend": {
        "host": "127.0.0.1",  # never bind to a public interface by default
        "port": 8765,
        "cameraIndex": 0,
        "cameraApi": "auto",  # auto | dshow | msmf
        "cameraFourcc": "auto",  # auto (driver default) or a 4-char pixel format, e.g. MJPG
        "modelPath": "public/models/hand_landmarker.task",  # relative to the repository root (fallback)
        "gestureModelPath": "public/models/gesture_recognizer.task",  # DL landmarks + DL gestures (preferred)
        "faceModelPath": "public/models/face_landmarker.task",  # landmarks + blendshapes + head pose (preferred)
        "faceDetectorPath": "public/models/blaze_face_short_range.tflite",  # fallback: gaze only
        "handPoseNetPath": "public/models/aura_handpose.json",  # project-trained hand-pose net (optional)
        "previewMaxWidth": 960,  # preview JPEG width sent to the browser (px)
        "previewJpegQuality": 70,
        "trackingLostHoldMs": 500,  # how long TRACKING_LOST is reported before NOT_DETECTED
        "maxConsecutiveReadFailures": 15,  # then the camera is reported as interrupted
        "allowedOrigins": ["http://127.0.0.1", "http://localhost"],  # scheme+host; any port
    },
}

Rule = Callable[[Any], "str | None"]


def _num(lo: float, hi: float) -> Rule:
    def rule(v: Any) -> str | None:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return "must be a finite number"
        return None if lo <= v <= hi else f"must be between {lo} and {hi}"

    return rule


def _int(lo: int, hi: int) -> Rule:
    def rule(v: Any) -> str | None:
        if isinstance(v, bool) or not isinstance(v, int):
            return "must be an integer"
        return _num(lo, hi)(v)

    return rule


def _bool(v: Any) -> str | None:
    return None if isinstance(v, bool) else "must be a boolean"


def _one_of(*values: str) -> Rule:
    return lambda v: None if v in values else f"must be one of {', '.join(values)}"


def _str(v: Any) -> str | None:
    return None if isinstance(v, str) and v else "must be a non-empty string"


def _origins(v: Any) -> str | None:
    if isinstance(v, list) and all(isinstance(o, str) and o.startswith(("http://", "https://")) for o in v):
        return None
    return "must be a list of http(s) origins"


RULES: dict[str, Rule] = {
    "camera.width": _int(160, 3840),
    "camera.height": _int(120, 2160),
    "camera.frameRate": _num(1, 120),
    "camera.mirror": _bool,
    "tracking.delegate": _one_of("GPU", "CPU"),
    "tracking.numHands": _int(1, 4),
    "tracking.minHandDetectionConfidence": _num(0, 1),
    "tracking.minHandPresenceConfidence": _num(0, 1),
    "tracking.minTrackingConfidence": _num(0, 1),
    "gestures.minTrackingScore": _num(0, 1),
    "gestures.minInFrameRatio": _num(0, 1),
    "gestures.minScore": _num(0, 1),
    "gestures.minMargin": _num(0, 1),
    "gestures.exitMs": _num(0, 5000),
    "gestures.maxStillSpeed": _num(0, 100),
    "gestures.postSwipeSuppressMs": _num(0, 10000),
    "gestures.swipe.enabled": _bool,
    "gestures.swipe.windowMs": _num(50, 5000),
    "gestures.swipe.minDistance": _num(0.1, 50),
    "gestures.swipe.minSpeed": _num(0, 100),
    "gestures.swipe.maxVerticalRatio": _num(0, 10),
    "gestures.swipe.maxGapMs": _num(1, 5000),
    "gestures.swipe.cooldownMs": _num(0, 10000),
    "gestures.swipe.oppositeSuppressMs": _num(0, 10000),
    "gestures.net.enabled": _bool,
    "gestures.net.minScore": _num(0, 1),
    "gestures.pinchTap.enabled": _bool,
    "gestures.pinchTap.maxMs": _num(50, 3000),
    "gestures.spin.enabled": _bool,
    "gestures.spin.windowMs": _num(200, 3000),
    "gestures.spin.rateWindowMs": _num(80, 2000),
    "gestures.spin.minWindowMs": _num(50, 2000),
    "gestures.spin.minSweepDeg": _num(30, 720),
    "gestures.spin.minRadius": _num(0.01, 20),
    "gestures.spin.maxRadiusCv": _num(0.01, 2),
    "gestures.spin.minConsistency": _num(0.5, 1),
    "gestures.spin.maxStepDeg": _num(5, 180),
    "gestures.spin.maxGapMs": _num(1, 2000),
    "gestures.spin.poseGraceMs": _num(0, 1000),
    "gestures.spin.enterMs": _num(0, 2000),
    "gestures.spin.releaseMs": _num(0, 1000),
    "gestures.walk.enabled": _bool,
    "gestures.walk.windowMs": _num(500, 5000),
    "gestures.walk.minAmplitude": _num(0.01, 2),
    "gestures.walk.minReach": _num(-1, 2),
    "gestures.walk.hysteresis": _num(0, 1),
    "gestures.walk.minSteps": _int(2, 20),
    "gestures.walk.maxCorrelation": _num(-1, 0),
    "gestures.walk.stopMs": _num(100, 3000),
    "gestures.walk.maxStepRate": _num(1, 30),
    "gestures.walk.poseGraceMs": _num(0, 1000),
    "face.enabled": _bool,
    "face.numFaces": _int(1, 4),
    "gestures.model.enabled": _bool,
    "gestures.model.minScore": _num(0, 1),
    "backend.gestureModelPath": _str,
    "backend.faceDetectorPath": _str,
    "backend.handPoseNetPath": _str,
    "face.minScore": _num(0, 1),
    "face.intervalMs": _num(16, 5000),
    "backend.faceModelPath": _str,
    "backend.host": _str,
    "backend.port": _int(1, 65535),
    "backend.cameraIndex": _int(0, 63),
    "backend.cameraApi": _one_of("auto", "dshow", "msmf"),
    "backend.cameraFourcc": lambda v: None if v == "auto" or (isinstance(v, str) and len(v) == 4 and v.isalnum() and v.isascii()) else 'must be "auto" or a 4-character code such as "MJPG"',
    "backend.modelPath": _str,
    "backend.previewMaxWidth": _int(160, 3840),
    "backend.previewJpegQuality": _int(10, 100),
    "backend.trackingLostHoldMs": _num(0, 10000),
    "backend.maxConsecutiveReadFailures": _int(1, 1000),
    "backend.allowedOrigins": _origins,
}

# Keys owned by the browser engine: accepted in the shared file, ignored here.
BROWSER_ONLY = {
    "camera.facingMode", "camera.deviceId", "tracking.modelAssetPath", "tracking.wasmPath", "face.modelAssetPath",
    "tracking.gestureModelAssetPath", "face.detectorAssetPath",
}
IGNORED_SECTIONS = {"interaction", "scene", "effects", "hud", "engine"}


def load_config(overrides: Any = None) -> tuple[dict[str, Any], list[str]]:
    """Merge overrides onto the defaults; return (config, errors)."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    errors: list[str] = []
    if overrides is None:
        return config, errors
    if not isinstance(overrides, dict):
        return config, ["configuration root must be a JSON object"]

    for section, value in overrides.items():
        if section in IGNORED_SECTIONS:
            continue
        if section not in config:
            errors.append(f"{section}: unknown setting")
            continue
        if not isinstance(value, dict):
            errors.append(f"{section}: must be an object")
            continue
        _merge(value, config[section], section, errors)
    return config, errors


def _merge(src: dict, dst: dict, path: str, errors: list[str]) -> None:
    for key, value in src.items():
        p = f"{path}.{key}"
        if p in BROWSER_ONLY:
            continue
        if p == "gestures.definitions":
            _merge_definitions(value, dst["definitions"], errors)
            continue
        if key not in dst:
            errors.append(f"{p}: unknown setting")
            continue
        if isinstance(dst[key], dict):
            if isinstance(value, dict):
                _merge(value, dst[key], p, errors)
            else:
                errors.append(f"{p}: must be an object")
            continue
        rule = RULES.get(p)
        err = rule(value) if rule else "no validation rule"
        if err:
            errors.append(f"{p}: {err}")
        else:
            dst[key] = value


def _merge_definitions(value: Any, dst: dict, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append("gestures.definitions: must be an object")
        return
    for gesture, d in value.items():
        p = f"gestures.definitions.{gesture}"
        if gesture not in STATIC_GESTURES:
            errors.append(f"{p}: unknown static gesture")
            continue
        if not isinstance(d, dict):
            errors.append(f"{p}: must be an object")
            continue
        for k, v in d.items():
            err = _bool(v) if k in ("enabled", "requireStill") else _num(0, 10000)(v) if k == "holdMs" else "unknown setting"
            if err:
                errors.append(f"{p}.{k}: {err}")
            else:
                dst[gesture][k] = v


def load_config_file(path: Path | None) -> tuple[dict[str, Any], list[str], str]:
    """Load the optional JSON file. A missing file is normal (defaults)."""
    if path is None or not path.exists():
        config, errors = load_config(None)
        return config, errors, "defaults"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        config, errors = load_config(None)
        return config, [f"{path}: {exc}"], "defaults"
    config, errors = load_config(data)
    return config, errors, str(path)
