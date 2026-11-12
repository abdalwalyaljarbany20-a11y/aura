"""MediaPipe hand topology and landmark validation.

Input format used throughout the engine: a NumPy array of shape (21, 3),
dtype float64, in MediaPipe order (see LM). Image landmarks are normalised
(x, y in [0, 1] of the image width/height, z relative depth); world landmarks
are metric (metres, hand-centred).
"""

from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21


class LM:
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


FINGER_CHAINS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

# MediaPipe may place landmarks of a partly visible hand slightly outside the
# image; values far outside this band indicate a malformed result.
NORMALIZED_TOLERANCE = 0.5


def as_landmark_array(points) -> np.ndarray | None:
    """Convert a sequence of (x, y, z) or objects with .x/.y/.z into a (21, 3) array."""
    if points is None:
        return None
    try:
        if isinstance(points, np.ndarray):
            arr = points.astype(np.float64, copy=False)
        else:
            rows = []
            for p in points:
                if hasattr(p, "x"):
                    rows.append((p.x, p.y, getattr(p, "z", 0.0) or 0.0))
                else:
                    rows.append(tuple(p))
            arr = np.asarray(rows, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 2 and arr.shape[1] == 2:
        arr = np.hstack([arr, np.zeros((arr.shape[0], 1))])
    return arr


def landmark_set_error(arr: np.ndarray | None, normalized: bool = False) -> str | None:
    """Reason a landmark array is unusable, or None if valid."""
    if arr is None:
        return "landmarks missing"
    if arr.ndim != 2 or arr.shape[1] != 3:
        return f"landmarks must have shape (21, 3), got {arr.shape}"
    if arr.shape[0] != NUM_LANDMARKS:
        return f"expected {NUM_LANDMARKS} landmarks, got {arr.shape[0]}"
    if not np.all(np.isfinite(arr)):
        bad = int(np.argwhere(~np.isfinite(arr))[0][0])
        return f"landmark {bad} is not finite"
    if normalized:
        lo, hi = -NORMALIZED_TOLERANCE, 1 + NORMALIZED_TOLERANCE
        xy = arr[:, :2]
        if np.any(xy < lo) or np.any(xy > hi):
            return "normalised coordinates far outside the image"
    return None
