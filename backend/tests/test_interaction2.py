"""Head-turn cues, the hand-pose net runtime and pinch tap (ports of test/interaction2.test.js)."""

import json

import numpy as np
import pytest

from aura_cv.config import DEFAULT_CONFIG
from aura_cv.faces import PITCH_NEUTRAL, PITCH_RANGE, head_cue
from aura_cv.features import extract_features
from aura_cv.gestures import GestureRecognitionEngine, StaticGestureClassifier
from aura_cv.posenet import POSE_DIM, HandPoseNet, pose_vector

from fixtures import POSES, make_hand

ASPECT = 16 / 9


def face(yaw=0.0, ratio=PITCH_NEUTRAL):
    lms = [(0.5, 0.5)] * 478
    lms[234], lms[454], lms[10], lms[152] = (0.4, 0.5), (0.6, 0.5), (0.5, 0.3), (0.5, 0.7)
    lms[1] = (0.5 + yaw * 0.1, 0.3 + ratio * 0.4)
    return lms


def test_head_cue_matches_the_browser_definition():
    assert head_cue(face()) == {"yaw": 0.0, "pitch": 0.0}
    assert head_cue(face(yaw=0.5))["yaw"] == 0.5
    assert head_cue(face(yaw=-0.5))["yaw"] == -0.5
    assert head_cue(face(ratio=PITCH_NEUTRAL - PITCH_RANGE / 2))["pitch"] < -0.49
    assert head_cue(face(ratio=PITCH_NEUTRAL + PITCH_RANGE))["pitch"] >= 0.999
    assert head_cue(face(yaw=3))["yaw"] == 1.0
    assert head_cue(face()[:100]) is None
    bad = face()
    bad[1] = (float("nan"), 0.0)
    assert head_cue(bad) is None


def tiny(bias):
    classes = ["OPEN_PALM", "FIST", "POINT", "TWO_FINGERS", "THREE_FINGERS", "THUMB_UP", "THUMB_DOWN", "OTHER"]
    return {"format": "aura-handpose-mlp", "version": 0, "classes": classes, "inputMean": [0.0] * 63, "inputStd": [1.0] * 63,
            "layers": [{"in": 63, "out": 4, "W": [0.0] * 252, "b": [0.0] * 4},
                       {"in": 4, "out": 8, "W": [0.0] * 32, "b": [bias.get(c, 0.0) for c in classes]}]}


def feats(pose, model=None, **view):
    img, world, _, score = make_hand(POSES[pose], aspect=ASPECT, **view)
    return extract_features(img, world, aspect=ASPECT, mirror=True, score=score, model=model)


def test_pose_vector_and_runtime():
    f = feats("OPEN_PALM")
    assert f.pose.shape == (POSE_DIM,) and np.allclose(f.pose[:3], 0)
    img, _, _, _ = make_hand(POSES["OPEN_PALM"], aspect=ASPECT)
    d = img.copy()
    assert np.allclose(pose_vector(d), pose_vector(d * 2.5 + 0.3), atol=1e-9)
    assert pose_vector(d[:20]) is None
    p = HandPoseNet(tiny({"THREE_FINGERS": 3})).predict(np.full(63, 0.1))
    assert p["label"] == "THREE_FINGERS" and abs(sum(p["scores"].values()) - 1) < 1e-9
    with pytest.raises(ValueError):
        HandPoseNet({"format": "other"})
    w = tiny({})
    w["layers"][0]["W"] = w["layers"][0]["W"][1:]
    with pytest.raises(ValueError, match="inconsistent"):
        HandPoseNet(w)


def test_fusion_with_the_net():
    cls = StaticGestureClassifier(DEFAULT_CONFIG["gestures"])

    def run(pose, model, bias, **view):
        f = feats(pose, model, **view)
        f.net = HandPoseNet(tiny(bias)).predict(f.pose)
        return cls.classify(f)

    v = {"label": "Victory", "score": 0.85, "scores": {}}
    none = {"label": "None", "score": 0.9, "scores": {}}
    r = run("THREE_FINGERS", v, {"THREE_FINGERS": 5})
    assert r.gesture == "THREE_FINGERS" and r.source.startswith("aura-net")
    assert run("VICTORY", v, {"TWO_FINGERS": 5}).gesture == "TWO_FINGERS"
    assert run("POINT", none, {"OTHER": 5}, rotation_deg=90).gesture == "UNKNOWN"
    assert run("POINT", none, {"POINT": 5}, rotation_deg=90).gesture == "POINT"
    assert run("FIST", none, {"FIST": 5}).gesture == "UNKNOWN", "the safety lock stays with the canned model"


def test_pinch_tap_only_for_a_quick_release():
    def taps(frames):
        eng = GestureRecognitionEngine(DEFAULT_CONFIG["gestures"])
        out = []
        for t, pose in frames:
            r = eng.update(feats(pose) if pose else None, t)
            out += [e for e in r["events"] if e["gesture"] == "PINCH_TAP"]
        return out

    quick = [(t, "PINCH" if t < 400 else "OPEN_PALM") for t in range(0, 1200, 33)]
    long = [(t, "PINCH" if t < 1500 else "OPEN_PALM") for t in range(0, 2200, 33)]
    lost = [(t, "PINCH" if t < 400 else None) for t in range(0, 900, 33)]
    assert len(taps(quick)) == 1 and taps(long) == [] and taps(lost) == []


def test_engine_reports_a_missing_or_malformed_net(tmp_path):
    from aura_cv.engine import Engine

    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg["backend"]["handPoseNetPath"] = str(tmp_path / "none.json")
    eng = Engine(cfg, repo_root=tmp_path, face_backend=None, capture_factory=lambda *a, **k: None)
    assert eng.status()["poseNet"]["status"] == "missing"
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": "aura-handpose-mlp"}', encoding="utf-8")
    cfg["backend"]["handPoseNetPath"] = str(bad)
    eng = Engine(cfg, repo_root=tmp_path, face_backend=None, capture_factory=lambda *a, **k: None)
    assert eng.status()["poseNet"]["status"] == "error"
    good = tmp_path / "good.json"
    good.write_text(json.dumps(tiny({})), encoding="utf-8")
    cfg["backend"]["handPoseNetPath"] = str(good)
    eng = Engine(cfg, repo_root=tmp_path, face_backend=None, capture_factory=lambda *a, **k: None)
    assert eng.status()["poseNet"]["status"] == "ready" and eng.pose_net is not None
