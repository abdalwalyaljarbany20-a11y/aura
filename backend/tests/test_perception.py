"""Deep-learning perception parsing in the Python engine: GestureRecognizer categories, the
model/rule fusion, and FaceLandmarker outputs (blendshapes, pose matrix). Synthetic results only."""

import copy
from types import SimpleNamespace

import numpy as np

from aura_cv.config import DEFAULT_CONFIG
from aura_cv.faces import FaceTracker
from aura_cv.features import extract_features
from aura_cv.gestures import StaticGestureClassifier
from aura_cv.tracking import HandTracker, to_model_gesture

from fixtures import POSES, THUMB_VIEW, make_hand

CFG = copy.deepcopy(DEFAULT_CONFIG["gestures"])
cat = lambda n, s: SimpleNamespace(category_name=n, score=s)  # noqa: E731


def feats(pose, model=None, **view):
    img, world, hand, score = make_hand(POSES[pose], aspect=16 / 9, **view)
    return extract_features(img, world, aspect=16 / 9, mirror=True, score=score, handedness=hand, model=model)


def test_gesture_categories_parsed_per_hand():
    assert to_model_gesture([cat("None", 0.1), cat("Pointing_Up", 0.8), cat("bad", float("nan"))]) == {
        "label": "Pointing_Up", "score": 0.8, "scores": {"None": 0.1, "Pointing_Up": 0.8}}
    img, world, hand, score = make_hand(POSES["POINT"], aspect=4 / 3)
    lm = lambda a: [SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in a]  # noqa: E731
    raw = SimpleNamespace(hand_landmarks=[lm(img)], hand_world_landmarks=[lm(world)], handedness=[[cat("Right", 0.97)]],
                          gestures=[[cat("Pointing_Up", 0.91), cat("None", 0.05)]])
    hands, rejected = HandTracker.parse(raw)
    assert rejected == 0 and hands[0].model["label"] == "Pointing_Up"
    assert hands[0].to_dict()["model"] == {"label": "Pointing_Up", "score": 0.91}
    # HandLandmarker results (no gestures attribute): model stays None -> rule fallback
    hands2, _ = HandTracker.parse(SimpleNamespace(hand_landmarks=[lm(img)], hand_world_landmarks=[], handedness=[[cat("Right", 0.9)]]))
    assert hands2[0].model is None


def test_fusion_matches_policy():
    c = StaticGestureClassifier(CFG)
    m = lambda label, s: {"label": label, "score": s, "scores": {label: s}}  # noqa: E731
    r = c.classify(feats("OPEN_PALM", m("Open_Palm", 0.9)))
    assert (r.gesture, r.source) == ("OPEN_PALM", "model")
    assert c.classify(feats("OPEN_PALM", m("Open_Palm", 0.55))).gesture == "UNKNOWN"
    assert c.classify(feats("FIST", m("None", 0.9))).gesture == "UNKNOWN", "FIST only from the model"
    assert c.classify(feats("POINT", m("None", 0.88), rotation_deg=90)).source == "rules (model: None)"
    assert c.classify(feats("PINCH", m("Open_Palm", 0.8))).source == "pinch-distance"
    g = c.classify(feats("OPEN_PALM", m("Thumb_Down", 0.77), rotation_deg=90))
    assert g.gesture == "UNKNOWN" and "geometry disagrees" in g.reason
    assert c.classify(feats("THUMBS_UP", m("Thumb_Down", 0.77), **THUMB_VIEW["down"])).gesture == "THUMB_DOWN"
    assert c.classify(feats("FIST")).source == "rules"


class Faces:
    def __init__(self, items):
        self.items = items

    def init(self):
        return {}

    def detect(self, rgb, ts):
        return self.items

    def close(self):
        pass


def test_face_landmarker_outputs_validated():
    good = {"box": (0.4, 0.3, 0.2, 0.3, 1.0), "bs": {"mouthSmileLeft": 0.7, "jawOpen": float("nan")}, "matrix": [0.5] * 16}
    bad_matrix = {"box": (0.1, 0.1, 0.1, 0.1, 1.0), "matrix": [1.0] * 15}
    junk = {"box": "nope"}
    ft = FaceTracker(Faces([good, bad_matrix, junk, (0.5, 0.5, 0.1, 0.1, 0.9)]), 100, 0.6)
    assert ft.init()
    faces = ft.detect(np.zeros((4, 4, 3), dtype=np.uint8), 0)
    assert len(faces) == 3
    assert faces[0]["bs"] == {"mouthSmileLeft": 0.7} and len(faces[0]["matrix"]) == 16
    assert "matrix" not in faces[1] and "bs" not in faces[2]
