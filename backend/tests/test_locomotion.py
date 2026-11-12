"""Embodied motion in the Python engine: continuous SPIN, cadence FINGER_WALK, thumb emotions, faces.

Synthetic landmark trajectories and scripted face detections only (test fixtures, never
camera data). Parity with the JS engine on identical input is checked in test_parity.py.
"""

import copy
import math

import numpy as np

from aura_cv.camera import Frame
from aura_cv.config import DEFAULT_CONFIG, load_config
from aura_cv.faces import FaceTracker
from aura_cv.features import extract_features
from aura_cv.gestures import GestureRecognitionEngine, SpinDetector, StaticGestureClassifier
from aura_cv.pipeline import FrameProcessor
from aura_cv.tracking import HandTracker

from fixtures import EMPTY, POSES, THUMB_VIEW, ScriptedBackend, make_hand, mp_result, step_pose

ASPECT = 16 / 9
CFG = copy.deepcopy(DEFAULT_CONFIG["gestures"])


def feats(pose, **view):
    img, world, hand, score = make_hand(POSES[pose] if isinstance(pose, str) else pose, aspect=ASPECT, **view)
    return extract_features(img, world, aspect=ASPECT, mirror=True, score=score, handedness=hand)


def view_at(xd, yd):
    return {"cx": 1 - xd / ASPECT, "cy": yd}


def circle_view(t, rev, direction=1):
    a = direction * 2 * math.pi * rev * t / 1000
    return view_at(0.9 + 0.12 * math.cos(a), 0.65 + 0.12 * math.sin(a))


def run(frame, to=3000):
    eng = GestureRecognitionEngine(copy.deepcopy(CFG))
    events, motion = [], []
    for t in range(0, to + 1, 33):
        r = eng.update(frame(t), t)
        events += [(e["type"], e["gesture"], t) for e in r["events"]]
        motion.append((t, r["motion"]))
    return events, motion


def kinds(events, g):
    return [e[0] for e in events if e[1] == g]


def test_spin_rates_ordered_and_signed():
    means = {}
    for rev in (0.5, 1.0, 2.5):
        _, motion = run(lambda t, rev=rev: feats("POINT", **circle_view(t, rev)))
        om = [m["spin"]["omega"] for t, m in motion if t > 1800]
        means[rev] = sum(om) / len(om)
        assert abs(means[rev] - 2 * math.pi * rev) / (2 * math.pi * rev) < 0.15, (rev, means[rev])
    assert 0 < means[0.5] < means[1.0] < means[2.5]
    _, ccw = run(lambda t: feats("POINT", **circle_view(t, 1.0, -1)))
    assert ccw[-1][1]["spin"]["omega"] < 0


def test_spin_rejections_and_stops():
    for name, frame in {
        "still": lambda t: feats("POINT", **view_at(0.9, 0.6)),
        "line": lambda t: feats("POINT", **view_at(0.6 + 0.6 * ((t % 1500) / 1500), 0.6)),
        "jumps": lambda t: feats("POINT", **view_at(0.9 + (0.3 if (t // 100) % 2 else -0.3), 0.6)),
        "open palm circling": lambda t: feats("OPEN_PALM", **circle_view(t, 1.0)),
    }.items():
        events, motion = run(frame)
        assert kinds(events, "SPIN") == [], name
        assert all(m["spin"]["omega"] == 0 for _, m in motion), name
    events, _ = run(lambda t: feats("POINT", **circle_view(t, 1.5)) if t < 2000 else None, to=2200)
    assert kinds(events, "SPIN") == ["start", "end"]
    det = SpinDetector(CFG["spin"])
    started = None
    for t in range(0, 2001, 33):
        a = 2 * math.pi * t / 1000
        if det.update(t, (0.9 + 0.12 * math.cos(a), 0.6 + 0.12 * math.sin(a)), 0.22, True)["event"] == "start":
            started = t
            break
    assert started >= CFG["spin"]["minWindowMs"] + CFG["spin"]["enterMs"]


def test_finger_walk_cadence_and_rejections():
    cad = {}
    for f in (0.65, 1.25, 2.2):
        events, motion = run(lambda t, f=f: feats(step_pose(t, freq=f)), to=5000)
        assert kinds(events, "FINGER_WALK") == ["start"], f
        c = [m["walk"]["cadence"] for t, m in motion if t > 3500]
        cad[f] = sum(c) / len(c)
        assert abs(cad[f] - 2 * f) / (2 * f) < 0.15, (f, cad[f])
    assert cad[0.65] < cad[1.25] < cad[2.2]
    for name, frame in {
        "static V": lambda t: feats("VICTORY"),
        "in phase": lambda t: feats(step_pose(t, freq=1.25, in_phase=True)),
        "one finger": lambda t: feats(step_pose(t, freq=1.25, only="index")),
        "shaking V": lambda t: feats("VICTORY", **view_at(0.9 + 0.15 * math.sin(t / 60), 0.6)),
    }.items():
        assert kinds(run(frame, to=5000)[0], "FINGER_WALK") == [], name
    events, _ = run(lambda t: feats(step_pose(t if t < 3000 else 3000, freq=1.5)), to=4600)
    end = [e[2] for e in events if e[1] == "FINGER_WALK" and e[0] == "end"]
    assert end and end[0] - 3000 <= CFG["walk"]["stopMs"] + 40, end


def test_thumb_emotions_and_fist_kept():
    cls = StaticGestureClassifier(CFG)
    assert cls.classify(feats("THUMBS_UP", **THUMB_VIEW["up"])).gesture == "THUMB_UP"
    assert cls.classify(feats("THUMBS_UP", **THUMB_VIEW["down"])).gesture == "THUMB_DOWN"
    assert cls.classify(feats("THUMBS_UP")).gesture == "UNKNOWN"
    assert cls.classify(feats("FIST")).gesture == "FIST"
    events, _ = run(lambda t: feats("THUMBS_UP", **THUMB_VIEW["up"]), to=3000)
    assert [e[0] for e in events if e[1] == "THUMB_UP"] == ["start"], "one onset per hold"


class ScriptedFaces:
    def __init__(self, faces):
        self.faces = faces
        self.calls = 0

    def init(self):
        return {"delegate": "FAKE"}

    def detect(self, rgb, ts):
        self.calls += 1
        return self.faces

    def close(self):
        pass


def test_faces_and_hands_coexist_in_one_result_throttled_and_validated():
    img, world, hand, score = make_hand(POSES["OPEN_PALM"], aspect=4 / 3)
    tracker = HandTracker(ScriptedBackend([mp_result((img, world, hand, score))]))
    tracker.init()
    backend = ScriptedFaces([(0.3, 0.2, 0.2, 0.3, 0.95), (float("nan"), 0, 0.1, 0.1, 0.9), (0.1, 0.1, 0.1, 0.1, 0.2)])
    faces = FaceTracker(backend, interval_ms=120, min_score=0.6)
    assert faces.init()
    proc = FrameProcessor(tracker, copy.deepcopy(CFG), to_rgb=lambda x: x, face_tracker=faces)
    frame_img = np.zeros((240, 320, 3), dtype=np.uint8)
    results = [proc.process(Frame(i + 1, frame_img, 1000.0 + i * 33, 1.0)) for i in range(30)]
    with_faces = [r for r in results if "faces" in r]
    assert all(r["status"] == "DETECTED" and len(r["hands"]) == 1 for r in results), "hands on every frame"
    assert 7 <= len(with_faces) <= 9 == backend.calls or 7 <= backend.calls <= 9
    assert len(with_faces) == backend.calls
    assert with_faces[0]["faces"] == [{"x": 0.4, "y": 0.35, "w": 0.2, "h": 0.3, "score": 0.95}]
    # no face model: gaze disabled, hand tracking unaffected
    off = FaceTracker(None, 120, 0.6)
    assert not off.init() and off.status == "disabled"
    proc2 = FrameProcessor(tracker, copy.deepcopy(CFG), to_rgb=lambda x: x, face_tracker=off)
    r = proc2.process(Frame(1, frame_img, 5000.0, 1.0))
    assert "faces" not in r and r["status"] == "DETECTED"
    assert "motion" in r["gesture"] and r["gesture"]["motion"]["spin"]["omega"] == 0


def test_face_backend_failure_only_disables_gaze():
    class Broken(ScriptedFaces):
        def init(self):
            raise FileNotFoundError("face model not found")

    ft = FaceTracker(Broken([]), 120, 0.6)
    assert not ft.init() and ft.status == "error" and "not found" in ft.error
    assert ft.detect(np.zeros((4, 4, 3)), 0) is None
    _ = EMPTY


def test_embodied_config_validated():
    cfg, errors = load_config({"gestures": {"spin": {"minSweepDeg": 5, "releaseMs": 150}, "walk": {"minSteps": 1, "stopMs": 900}},
                               "face": {"intervalMs": 200, "minScore": 2}})
    assert any("gestures.spin.minSweepDeg" in e for e in errors) and any("gestures.walk.minSteps" in e for e in errors)
    assert any("face.minScore" in e for e in errors)
    assert cfg["gestures"]["spin"]["releaseMs"] == 150 and cfg["gestures"]["walk"]["stopMs"] == 900
    assert cfg["face"]["intervalMs"] == 200 and cfg["face"]["minScore"] == 0.6
