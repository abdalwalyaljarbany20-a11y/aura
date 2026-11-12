import copy

import numpy as np
import pytest

from aura_cv.config import DEFAULT_CONFIG
from aura_cv.features import extract_features
from aura_cv.gestures import GestureRecognitionEngine, GestureStabilizer, StaticGestureClassifier, SwipeDetector
from aura_cv.landmarks import as_landmark_array, landmark_set_error

from fixtures import POSES, make_hand

CFG = copy.deepcopy(DEFAULT_CONFIG["gestures"])


def feats(pose, aspect=16 / 9, **view):
    img, world, hd, score = make_hand(POSES[pose], aspect=aspect, **view)
    return extract_features(img, world, aspect=aspect, mirror=True, score=score, handedness=hd)


# ---------------------------------------------------------------- features

def test_open_palm_fingers_extended():
    f = feats("OPEN_PALM")
    assert f.valid and f.source == "world"
    for name in ("thumb", "index", "middle", "ring", "pinky"):
        assert f.extension(name) > 0.9
    assert f.pinch_distance > 1


def test_curled_fingers_low_extension():
    f = feats("FIST")
    for name in ("index", "middle", "ring", "pinky"):
        assert f.extension(name) < 0.1
        assert f.fingers[name]["straightness"] < 0.5


def test_invariance_to_position_scale_rotation():
    a = feats("POINT", world=False)
    b = feats("POINT", world=False, cx=0.3, cy=0.4, size=1.2, rotation_deg=35)
    assert a.source == "image"
    for name in ("index", "middle", "ring", "pinky", "thumb"):
        assert a.extension(name) == pytest.approx(b.extension(name), abs=1e-9)
    assert a.pinch_distance == pytest.approx(b.pinch_distance, abs=1e-9)
    assert b.palm_size_image / a.palm_size_image == pytest.approx(1.2 / 2.5)


def test_mirroring_flips_display_x():
    img, world, *_ = make_hand(POSES["OPEN_PALM"], cx=0.2)
    m = extract_features(img, world, aspect=1, mirror=True)
    r = extract_features(img, world, aspect=1, mirror=False)
    assert m.palm_center_norm[0] == pytest.approx(1 - r.palm_center_norm[0])


def test_matches_browser_engine_reference_values():
    # Same synthetic hand as test/fixtures/hands.js; the JS FeatureExtractionEngine
    # printed these values for it (FIST index straightness 0.42, pinch distance 0.35).
    f = feats("FIST")
    assert f.fingers["index"]["straightness"] == pytest.approx(0.42, abs=0.01)
    assert f.pinch_distance == pytest.approx(0.35, abs=0.01)


@pytest.mark.parametrize("bad, fragment", [
    (None, "missing"),
    (np.zeros((20, 3)), "expected 21"),
    (np.zeros((21, 4)), "shape"),
])
def test_invalid_landmark_sets(bad, fragment):
    f = extract_features(bad)
    assert not f.valid and fragment in f.reason


def test_non_finite_landmark_rejected():
    img, *_ = make_hand(POSES["OPEN_PALM"])
    img[7, 0] = np.nan
    f = extract_features(img, None)
    assert not f.valid and "landmark 7" in f.reason


def test_degenerate_hand_rejected():
    f = extract_features(np.full((21, 3), 0.5))
    assert not f.valid and "degenerate" in f.reason


def test_malformed_world_falls_back_to_image():
    img, world, *_ = make_hand(POSES["OPEN_PALM"])
    f = extract_features(img, world[:5])
    assert f.valid and f.source == "image"


def test_normalized_range_validation_and_conversion():
    img, *_ = make_hand(POSES["OPEN_PALM"])
    assert landmark_set_error(img, normalized=True) is None
    far = img.copy()
    far[0, 0] = 3.0
    assert "outside" in landmark_set_error(far, normalized=True)
    assert as_landmark_array([(0.1, 0.2)] * 21).shape == (21, 3)
    assert as_landmark_array("garbage") is None or landmark_set_error(as_landmark_array("garbage")) is not None


# ---------------------------------------------------------------- static rules

classifier = StaticGestureClassifier(CFG)


@pytest.mark.parametrize("pose, expected", [
    ("OPEN_PALM", "OPEN_PALM"), ("FIST", "FIST"), ("POINT", "POINT"), ("PINCH", "PINCH"),
    ("PINCH_CURLED", "PINCH"), ("FLAT_HAND", "OPEN_PALM"), ("VICTORY", "TWO_FINGERS"), ("THREE_FINGERS", "THREE_FINGERS"),
])
@pytest.mark.parametrize("view", [{}, {"world": False}, {"rotation_deg": 40, "size": 1.5}, {"rotation_deg": -60, "cx": 0.3}])
def test_classifies_representative_poses(pose, expected, view):
    r = classifier.classify(feats(pose, **view))
    assert r.gesture == expected, (r, view)
    assert r.confidence >= 0.6


@pytest.mark.parametrize("pose", ["CLAW", "THUMBS_UP", "VICTORY_THUMB_OUT", "HALF_MIDDLE"])
def test_similar_or_out_of_vocabulary_shapes_unknown(pose):
    r = classifier.classify(feats(pose))
    assert r.gesture == "UNKNOWN" and r.reason


def test_neutral_states():
    assert classifier.classify(None).gesture == "NONE"
    low = classifier.classify(feats("FIST", score=0.3))
    assert low.gesture == "UNKNOWN" and "tracking confidence" in low.reason
    edge = classifier.classify(feats("FIST", cx=0.99))
    assert edge.gesture == "UNKNOWN" and "outside" in edge.reason
    assert classifier.classify(feats("FIST"), ["OPEN_PALM"]).gesture == "UNKNOWN"


def test_ambiguity_margin():
    c = StaticGestureClassifier({**CFG, "minScore": 0.5})
    c.scores = staticmethod(lambda f: {"OPEN_PALM": 0.8, "FIST": 0.0, "POINT": 0.7, "PINCH": 0.0})
    r = c.classify(feats("OPEN_PALM"))
    assert r.gesture == "UNKNOWN" and "ambiguous" in r.reason


# ---------------------------------------------------------------- stabilizer

def test_stabilizer_single_start_for_hold():
    s = GestureStabilizer(150, lambda g: 300)
    ev = [e for t in range(0, 2001, 33) for e in s.update("FIST", 0.9, t)]
    assert [e["type"] for e in ev] == ["start"] and ev[0]["t"] >= 300


def test_stabilizer_noise_and_dropouts():
    s = GestureStabilizer(150, lambda g: 300)
    ev = []
    ev += [e for t in range(0, 200, 33) for e in s.update("POINT", 0.9, t)]
    ev += [e for t in range(200, 400, 33) for e in s.update("UNKNOWN", 0, t)]
    assert ev == []
    ev += [e for t in range(400, 900, 33) for e in s.update("POINT", 0.9, t)]
    ev += s.update("OPEN_PALM", 0.7, 900)  # single noisy frame
    ev += [e for t in range(933, 1200, 33) for e in s.update("POINT", 0.9, t)]
    assert [(e["type"], e["gesture"]) for e in ev] == [("start", "POINT")]


def test_stabilizer_reset_condition_allows_retrigger():
    s = GestureStabilizer(150, lambda g: 200)
    ev = [e for t in range(0, 501, 50) for e in s.update("FIST", 0.9, t)]
    ev += [e for t in range(550, 801, 50) for e in s.update("NONE", 0, t)]
    ev += [e for t in range(850, 1201, 50) for e in s.update("FIST", 0.9, t)]
    assert [e["type"] for e in ev] == ["start", "end", "start"]


def test_stabilizer_release_on_tracking_loss():
    s = GestureStabilizer(150, lambda g: 0)
    s.update("PINCH", 1, 0)
    assert [e["type"] for e in s.release(10)] == ["end"]
    assert s.stable == "NONE" and s.release(20) == []


# ---------------------------------------------------------------- swipes

def run_swipe(d, from_x, to_x, duration, t0=0, y0=0.5, y1=0.5, palm=0.2):
    out = []
    t = t0
    while t <= t0 + duration:
        k = (t - t0) / duration
        r = d.update(t, (from_x + (to_x - from_x) * k, y0 + (y1 - y0) * k), palm)
        if r:
            out.append(r["gesture"])
        t += 33
    return out


def test_swipe_directions():
    assert run_swipe(SwipeDetector(CFG["swipe"]), 1.2, 0.5, 300) == ["SWIPE_LEFT"]
    assert run_swipe(SwipeDetector(CFG["swipe"]), 0.5, 1.2, 300) == ["SWIPE_RIGHT"]


def test_swipe_rejections():
    assert run_swipe(SwipeDetector(CFG["swipe"]), 0.5, 1.2, 2000) == []  # slow
    assert run_swipe(SwipeDetector(CFG["swipe"]), 0.5, 0.6, 100) == []  # short
    assert run_swipe(SwipeDetector(CFG["swipe"]), 0.5, 0.9, 300, y0=0.1, y1=0.9) == []  # vertical


def test_swipe_return_stroke_suppressed():
    d = SwipeDetector(CFG["swipe"])
    assert run_swipe(d, 0.5, 1.2, 300) == ["SWIPE_RIGHT"]
    assert run_swipe(d, 1.2, 0.5, 300, t0=400) == []
    assert run_swipe(d, 0.5, 1.2, 300, t0=1500) == ["SWIPE_RIGHT"]


# ---------------------------------------------------------------- engine

def feed(engine, pose, t0, t1, **view):
    events, last = [], None
    t = t0
    while t <= t1:
        last = engine.update(feats(pose, **view), t)
        events += last["events"]
        t += 33
    return events, last


def test_engine_hold_and_tracking_loss():
    e = GestureRecognitionEngine(CFG)
    events, last = feed(e, "OPEN_PALM", 0, 3000)
    assert [(x["type"], x["gesture"]) for x in events] == [("start", "OPEN_PALM")]
    assert last["stable"]["gesture"] == "OPEN_PALM"
    lost = e.update(None, 3033)
    assert not lost["hand"] and [x["type"] for x in lost["events"]] == ["end"]
    assert e.update(None, 3066)["events"] == []


def test_engine_no_swipe_while_pinching():
    e = GestureRecognitionEngine(CFG)
    events = []
    for t in range(0, 601, 33):
        cx = 0.5 if t < 300 else 0.5 - (t - 300) / 300 * 0.5
        events += e.update(feats("PINCH", cx=cx), t)["events"]
    assert not [x for x in events if x["type"] == "event"]
    assert [x["gesture"] for x in events if x["type"] == "start"] == ["PINCH"]


def test_engine_configurable_definitions():
    c = copy.deepcopy(CFG)
    c["definitions"]["FIST"]["enabled"] = False
    e = GestureRecognitionEngine(c)
    assert feed(e, "FIST", 0, 2000)[0] == []
