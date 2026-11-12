"""Geometric feature extraction (Python port of src/gestures/FeatureExtractionEngine.js).

Definitions (identical to the browser engine, see docs/ARCHITECTURE.md):

  palm_size        mean(|wrist-indexMCP|, |wrist-middleMCP|, |wrist-pinkyMCP|); all
                   shape distances are divided by it
  straightness(f)  |base->tip| / sum(segment lengths) of the finger's 4-joint chain;
                   ~1 straight, ~0.3-0.55 curled; invariant to translation, scale and
                   in-plane rotation
  extension(f)     ramp(straightness, 0.55 -> 0.92)
  thumb extension  additionally min(.., ramp(abduction, 0.45 -> 0.8)), where abduction is
                   |thumbTip - indexMCP| / palm_size
  pinch_distance   |thumbTip - indexTip| / palm_size
  palm_center      display-space palm centre (x scaled by image aspect, mirrored if the
                   view is mirrored), in units of image height
  palm_size_image  apparent palm size in the same units (normalises motion)
  in_frame_ratio   fraction of image landmarks inside [0, 1]^2

Shape features use world landmarks when present, else aspect-corrected image landmarks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .landmarks import FINGER_CHAINS, LM, landmark_set_error
from .posenet import pose_vector

EXTENSION_RANGE = (0.55, 0.92)
THUMB_ABDUCTION_RANGE = (0.45, 0.8)
_EPS = 1e-9


def ramp(v: float, a: float, b: float) -> float:
    """Linear ramp: 0 at a, 1 at b, clamped (works for descending ramps too)."""
    t = (v - a) / (b - a)
    return 0.0 if t < 0 else 1.0 if t > 1 else float(t)


@dataclass
class HandFeatures:
    valid: bool
    reason: str | None = None
    source: str = ""
    palm_size: float = 0.0
    fingers: dict = field(default_factory=dict)  # name -> {"straightness", "extension", ["abduction"]}
    pinch_distance: float = 0.0
    palm_center: tuple[float, float] = (0.0, 0.0)
    palm_center_norm: tuple[float, float] = (0.0, 0.0)
    index_tip_norm: tuple[float, float] = (0.0, 0.0)
    index_tip: tuple[float, float] = (0.0, 0.0)  # display units, like palm_center (spin gesture)
    legs: tuple[float, float] = (0.0, 0.0)  # index / middle length along the palm axis, palm sizes (walking)
    thumb_dir: tuple[float, float] = (0.0, 0.0)  # unit thumb MCP -> tip direction, display space (y down)
    palm_size_image: float = 0.0
    roll: float = 0.0
    in_frame_ratio: float = 0.0
    score: float | None = None
    handedness: str | None = None
    model: dict | None = None  # learned gesture classifier output {label, score, scores}
    pose: np.ndarray | None = None  # 63-value pose vector for the hand-pose net (posenet.pose_vector)
    net: dict | None = None  # hand-pose net output {label, score, scores} (set by the gesture engine)

    def extension(self, name: str) -> float:
        return self.fingers[name]["extension"]

    def to_dict(self) -> dict:
        if not self.valid:
            return {"valid": False, "reason": self.reason}
        r = lambda v: round(float(v), 4)  # noqa: E731
        return {
            "valid": True,
            "source": self.source,
            "extension": {k: r(v["extension"]) for k, v in self.fingers.items()},
            "straightness": {k: r(v["straightness"]) for k, v in self.fingers.items()},
            "thumbAbduction": r(self.fingers["thumb"]["abduction"]),
            "pinchDistance": r(self.pinch_distance),
            "palmSizeImage": r(self.palm_size_image),
            "inFrameRatio": r(self.in_frame_ratio),
        }


def invalid(reason: str) -> HandFeatures:
    return HandFeatures(valid=False, reason=reason)


def extract_features(
    landmarks: np.ndarray | None,
    world: np.ndarray | None = None,
    *,
    aspect: float = 1.0,
    mirror: bool = True,
    score: float | None = None,
    handedness: str | None = None,
    model: dict | None = None,
) -> HandFeatures:
    err = landmark_set_error(landmarks)
    if err:
        return invalid(err)
    if not (isinstance(aspect, (int, float)) and np.isfinite(aspect) and aspect > 0):
        aspect = 1.0

    image = landmarks
    disp = image.copy()
    if mirror:
        disp[:, 0] = 1.0 - disp[:, 0]
    disp[:, 0] *= aspect
    disp[:, 2] *= aspect

    use_world = world is not None and landmark_set_error(world) is None
    shape = world if use_world else disp

    def d(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    palm_size = (d(shape[LM.WRIST], shape[LM.INDEX_MCP]) + d(shape[LM.WRIST], shape[LM.MIDDLE_MCP])
                 + d(shape[LM.WRIST], shape[LM.PINKY_MCP])) / 3.0
    if not palm_size > _EPS:
        return invalid("degenerate hand (zero palm size)")

    fingers: dict[str, dict] = {}
    for name, chain in FINGER_CHAINS.items():
        pts = shape[list(chain)]
        path = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        straightness = d(pts[0], pts[-1]) / path if path > _EPS else 0.0
        fingers[name] = {"straightness": straightness, "extension": ramp(straightness, *EXTENSION_RANGE)}
    abduction = d(shape[LM.THUMB_TIP], shape[LM.INDEX_MCP]) / palm_size
    fingers["thumb"]["abduction"] = abduction
    fingers["thumb"]["extension"] = min(fingers["thumb"]["extension"], ramp(abduction, *THUMB_ABDUCTION_RANGE))

    pinch = d(shape[LM.THUMB_TIP], shape[LM.INDEX_TIP]) / palm_size

    axis = shape[LM.MIDDLE_MCP] - shape[LM.WRIST]
    axis_len = float(np.linalg.norm(axis))

    def along(tip: int, mcp: int) -> float:
        v = shape[tip] - shape[mcp]
        return float((v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2]) / axis_len / palm_size) if axis_len > 0 else 0.0

    legs = (along(LM.INDEX_TIP, LM.INDEX_MCP), along(LM.MIDDLE_TIP, LM.MIDDLE_MCP))
    tvx = float(disp[LM.THUMB_TIP, 0] - disp[LM.THUMB_MCP, 0])
    tvy = float(disp[LM.THUMB_TIP, 1] - disp[LM.THUMB_MCP, 1])
    tl = (tvx * tvx + tvy * tvy) ** 0.5
    thumb_dir = (tvx / tl, tvy / tl) if tl > 0 else (0.0, 0.0)

    palm_idx = [LM.WRIST, LM.INDEX_MCP, LM.MIDDLE_MCP, LM.RING_MCP, LM.PINKY_MCP]
    center = disp[palm_idx, :2].mean(axis=0)
    palm_size_image = (d(disp[LM.WRIST], disp[LM.INDEX_MCP]) + d(disp[LM.WRIST], disp[LM.MIDDLE_MCP])
                       + d(disp[LM.WRIST], disp[LM.PINKY_MCP])) / 3.0
    up = disp[LM.MIDDLE_MCP] - disp[LM.WRIST]
    roll = float(np.arctan2(up[0], -up[1]))
    inside = np.all((image[:, :2] >= 0) & (image[:, :2] <= 1), axis=1)

    return HandFeatures(
        valid=True,
        source="world" if use_world else "image",
        palm_size=palm_size,
        fingers=fingers,
        pinch_distance=pinch,
        palm_center=(float(center[0]), float(center[1])),
        palm_center_norm=(float(center[0] / aspect), float(center[1])),
        index_tip_norm=(float(disp[LM.INDEX_TIP, 0] / aspect), float(disp[LM.INDEX_TIP, 1])),
        index_tip=(float(disp[LM.INDEX_TIP, 0]), float(disp[LM.INDEX_TIP, 1])),
        legs=legs,
        thumb_dir=thumb_dir,
        palm_size_image=palm_size_image,
        roll=roll,
        in_frame_ratio=float(inside.mean()),
        score=score,
        handedness=handedness,
        pose=pose_vector(disp),
        model=model if isinstance(model, dict) and isinstance(model.get("label"), str) and isinstance(model.get("score"), (int, float)) else None,
    )
