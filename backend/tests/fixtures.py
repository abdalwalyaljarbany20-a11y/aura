"""Synthetic hands and fakes for tests (port of test/fixtures/hands.js).

These are geometric idealisations used ONLY in tests; they verify rule logic, never
real-world accuracy, and are never presented as live detections.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

FINGERS = {
    "index": ((-0.03, -0.085, 0.0), (0.04, 0.025, 0.02)),
    "middle": ((-0.008, -0.09, 0.0), (0.045, 0.028, 0.022)),
    "ring": ((0.012, -0.085, 0.0), (0.042, 0.026, 0.02)),
    "pinky": ((0.03, -0.075, 0.0), (0.032, 0.02, 0.018)),
}
THUMB_CMC = np.array([-0.025, -0.025, 0.0])
FLEX = {"straight": (0, 4, 4), "curled": (80, 100, 60), "pinchIndex": (35, 45, 25), "half": (40, 55, 35)}


def _chain(base, seg, flex):
    pts = [np.array(base, dtype=float)]
    theta = 0.0
    p = pts[0]
    for i in range(3):
        theta += math.radians(flex[i])
        p = p + seg[i] * np.array([0.0, -math.cos(theta), -math.sin(theta)])
        pts.append(p)
    return pts


def make_hand(pose: dict, *, cx=0.5, cy=0.6, size=2.5, rotation_deg=0.0, aspect=16 / 9,
              score=0.95, handedness="Right", world=True):
    """Returns (image_landmarks (21,3), world_landmarks (21,3) | None, handedness, score)."""
    w = np.zeros((21, 3))
    for start, name in zip((5, 9, 13, 17), ("index", "middle", "ring", "pinky")):
        base, seg = FINGERS[name]
        for j, p in enumerate(_chain(base, seg, pose.get(name, FLEX["straight"]))):
            w[start + j] = p
    thumb = pose.get("thumb", "out")
    w[1] = THUMB_CMC
    if thumb == "out":
        d = np.array([-0.85, -0.53, 0.0])
        w[2] = THUMB_CMC + 0.035 * d
        w[3] = w[2] + 0.03 * d
        w[4] = w[3] + 0.025 * d
    elif thumb == "tucked":
        w[2], w[3], w[4] = (-0.03, -0.045, -0.012), (-0.02, -0.06, -0.028), (-0.008, -0.07, -0.038)
    else:  # pinch
        tip = w[8] + np.array([-0.008, 0.002, 0.0])
        w[4] = tip
        w[2] = THUMB_CMC + (tip - THUMB_CMC) * 0.4 + np.array([-0.012, 0, -0.005])
        w[3] = THUMB_CMC + (tip - THUMB_CMC) * 0.72 + np.array([-0.008, 0, -0.008])
    r = math.radians(rotation_deg)
    rot = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    w = w @ rot.T
    img = np.stack([cx + w[:, 0] * size / aspect, cy + w[:, 1] * size, w[:, 2] * size / aspect], axis=1)
    return img, (w.copy() if world else None), handedness, score


POSES = {
    "OPEN_PALM": {"thumb": "out"},
    "FIST": {"index": FLEX["curled"], "middle": FLEX["curled"], "ring": FLEX["curled"], "pinky": FLEX["curled"], "thumb": "tucked"},
    "POINT": {"middle": FLEX["curled"], "ring": FLEX["curled"], "pinky": FLEX["curled"], "thumb": "tucked"},
    "PINCH": {"index": FLEX["pinchIndex"], "thumb": "pinch"},
    "PINCH_CURLED": {"index": FLEX["pinchIndex"], "middle": FLEX["curled"], "ring": FLEX["curled"], "pinky": FLEX["curled"], "thumb": "pinch"},
    "FLAT_HAND": {"thumb": "tucked"},
    "VICTORY": {"ring": FLEX["curled"], "pinky": FLEX["curled"], "thumb": "tucked"},  # the walk gesture
    "THREE_FINGERS": {"pinky": FLEX["curled"], "thumb": "tucked"},
    "VICTORY_THUMB_OUT": {"ring": FLEX["curled"], "pinky": FLEX["curled"], "thumb": "out"},
    "HALF_MIDDLE": {"middle": FLEX["half"], "ring": FLEX["curled"], "pinky": FLEX["curled"], "thumb": "tucked"},
    "CLAW": {"index": FLEX["half"], "middle": FLEX["half"], "ring": FLEX["half"], "pinky": FLEX["half"], "thumb": "tucked"},
    "THUMBS_UP": {"index": FLEX["curled"], "middle": FLEX["curled"], "ring": FLEX["curled"], "pinky": FLEX["curled"], "thumb": "out"},
}


THUMB_VIEW = {"up": {"rotation_deg": 58}, "down": {"rotation_deg": 238}}  # sideways-thumb fist turned up / down


def step_pose(t, freq=1.0, amp=40.0, in_phase=False, only=None):
    """Finger-walking pose at time t (port of stepPose in test/fixtures/hands.js)."""
    s = math.sin(2 * math.pi * freq * (t / 1000))

    def flex(k):
        return (amp * k, amp * k * 1.2, amp * k * 0.7)

    a = 0.5 + 0.5 * s
    b = a if in_phase else 0.5 - 0.5 * s
    return {"index": FLEX["straight"] if only == "middle" else flex(a),
            "middle": FLEX["straight"] if only == "index" else flex(b),
            "ring": FLEX["curled"], "pinky": FLEX["curled"], "thumb": "tucked"}


def mp_result(*hands):
    """Build an object shaped like mediapipe's HandLandmarkerResult from make_hand() tuples."""
    lm = lambda arr: [SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in arr]  # noqa: E731
    return SimpleNamespace(
        hand_landmarks=[lm(h[0]) for h in hands],
        hand_world_landmarks=[lm(h[1]) if h[1] is not None else [] for h in hands],
        handedness=[[SimpleNamespace(category_name=h[2], score=h[3])] for h in hands],
    )


class ScriptedBackend:
    """HandBackend replaying a script of results (or exceptions). Test-only."""

    def __init__(self, script, fail_init: Exception | None = None):
        self.script = list(script)
        self.fail_init = fail_init
        self.calls = 0
        self.timestamps: list[int] = []
        self.closed = 0

    def init(self):
        if self.fail_init:
            raise self.fail_init
        return {"delegate": "FAKE"}

    def detect(self, rgb, ts_ms):
        self.timestamps.append(ts_ms)
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(step, Exception):
            raise step
        return step

    def close(self):
        self.closed += 1


EMPTY = SimpleNamespace(hand_landmarks=[], hand_world_landmarks=[], handedness=[])


class FakeCapture:
    """Stands in for cv2.VideoCapture. ``frames`` items: ndarray, None (read fails) or Exception."""

    def __init__(self, frames, loop_last=True, delay_s=0.0):
        self.frames = list(frames)
        self.i = 0
        self.loop_last = loop_last
        self.delay_s = delay_s
        self.released = 0

    def isOpened(self):  # noqa: N802 - OpenCV API
        return True

    def set(self, *_):
        return True

    def get(self, *_):
        return 30.0

    def read(self):
        import time

        if self.delay_s:
            time.sleep(self.delay_s)
        if self.i >= len(self.frames):
            if not self.loop_last or not self.frames:
                return False, None
            item = self.frames[-1]
        else:
            item = self.frames[self.i]
            self.i += 1
        if isinstance(item, Exception):
            raise item
        if item is None:
            return False, None
        return True, item

    def release(self):
        self.released += 1


def blank_frame(w=320, h=240):
    return np.full((h, w, 3), 40, dtype=np.uint8)
