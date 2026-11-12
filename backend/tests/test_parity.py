"""Shared gesture-event contract: both engines emit identical events for identical input.

Synthetic, timestamped landmark sequences (test fixtures, not camera data) are fed to
the Python engine and, via node, to the JS engine (test/contract/gesture-parity.mjs).
"""

import copy
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from aura_cv.config import DEFAULT_CONFIG
from aura_cv.features import extract_features
from aura_cv.gestures import GestureRecognitionEngine
from aura_cv.posenet import HandPoseNet

from fixtures import POSES, THUMB_VIEW, make_hand, step_pose

REPO = Path(__file__).resolve().parents[2]
ASPECT = 16 / 9


def frame(t, pose=None, model=None, **view):
    if pose is None:
        return {"t": t, "aspect": ASPECT, "hand": None}
    img, world, _, score = make_hand(POSES[pose] if isinstance(pose, str) else pose, aspect=ASPECT, **view)
    return {"t": t, "aspect": ASPECT,
            "hand": {"landmarks": img.tolist(), "world": world.tolist() if world is not None else None, "score": score,
                     "model": model}}


def mdl(label, score):
    return {"label": label, "score": score, "scores": {label: score}}


def seq(pose, t0, t1, **view):
    return [frame(t, pose, **view) for t in range(t0, t1 + 1, 33)]


def sequences():
    s = {}
    s["hold_open_palm_then_loss"] = seq("OPEN_PALM", 0, 1500) + [frame(t) for t in range(1533, 1800, 33)]
    s["fist_hold_release_rehold"] = seq("FIST", 0, 1000) + seq("CLAW", 1033, 1400) + seq("FIST", 1433, 2400)
    # interrupted every 4th frame: never held for POINT's 250 ms -> no onset (by design)
    s["flicker_point"] = [frame(t, "POINT" if (t // 33) % 4 else "VICTORY") for t in range(0, 1500, 33)]
    # stable POINT survives a single-frame dropout: exactly one start
    s["point_hold_with_dropout"] = seq("POINT", 0, 600) + [frame(633, "VICTORY")] + seq("POINT", 666, 1200)
    s["pinch_hold_then_drag"] = [frame(t, "PINCH", cx=0.5 if t < 300 else 0.5 - (t - 300) / 300 * 0.5) for t in range(0, 601, 33)]
    s["swipe_right_then_return"] = ([frame(t, "OPEN_PALM", cx=0.75 - t / 300 * 0.5) for t in range(0, 301, 33)]
                                    + [frame(t, "OPEN_PALM", cx=0.25 + (t - 400) / 300 * 0.5) for t in range(400, 701, 33)])
    s["swipe_left_image_only"] = [frame(t, "OPEN_PALM", cx=0.25 + t / 300 * 0.5, world=False) for t in range(0, 301, 33)]
    s["low_tracking_score_unknown"] = seq("FIST", 0, 1500, rotation_deg=50, score=0.3)
    s["thumbs_up_never_locks"] = seq("THUMBS_UP", 0, 2000)
    # embodied motion: continuous spin (rate follows the finger), cadence walking, emotions
    # phase-continuous speed-up (a phase jump would be a tracking jump and correctly end the spin)
    ph = lambda t: t / 1600 if t < 1533 else 1533 / 1600 + (t - 1533) / 450  # noqa: E731
    s["spin_cw_slow_then_fast"] = [frame(t, "POINT", **disp(0.9 + 0.12 * math.cos(2 * math.pi * ph(t)), 0.65 + 0.12 * math.sin(2 * math.pi * ph(t))))
                                   for t in range(0, 3001, 33)]
    s["spin_ccw_then_loss"] = circle(-1, 0, 2000, period=900) + [frame(t) for t in range(2033, 2200, 33)]
    s["point_line_no_spin"] = [frame(t, "POINT", **disp(0.6 + 0.6 * t / 1500, 0.65)) for t in range(0, 1501, 33)]
    s["point_still_no_spin"] = seq("POINT", 0, 2500)
    s["static_v_never_walks"] = seq("VICTORY", 0, 3000)
    s["finger_walk_steps_then_stop"] = ([frame(t, step_pose(t, freq=1.25)) for t in range(0, 4000, 33)]
                                        + [frame(t, step_pose(3999, freq=1.25)) for t in range(4000, 5600, 33)])
    s["finger_walk_fast"] = [frame(t, step_pose(t, freq=2.2)) for t in range(0, 3000, 33)]
    s["finger_walk_in_phase_rejected"] = [frame(t, step_pose(t, freq=1.25, in_phase=True)) for t in range(0, 4000, 33)]
    s["pointing_wobble_not_walk"] = [frame(t, {**POSES["POINT"], "index": (20 + 20 * math.sin(2 * math.pi * 1.25 * t / 1000), 24 + 24 * math.sin(2 * math.pi * 1.25 * t / 1000), 14 + 14 * math.sin(2 * math.pi * 1.25 * t / 1000)),
                                                   "middle": (60 - 30 * math.sin(2 * math.pi * 1.25 * t / 1000), 90, 55)}) for t in range(0, 5000, 33)]
    s["thumb_up_laugh"] = seq("THUMBS_UP", 0, 1500, **THUMB_VIEW["up"])
    s["thumb_down_sad"] = seq("THUMBS_UP", 0, 1500, **THUMB_VIEW["down"])
    # deep-learning gesture model outputs (as GestureRecognizer reports them) through the fusion
    s["model_open_palm"] = [frame(t, "OPEN_PALM", mdl("Open_Palm", 0.9)) for t in range(0, 1500, 33)]
    s["model_open_palm_low_score"] = [frame(t, "OPEN_PALM", mdl("Open_Palm", 0.55)) for t in range(0, 1500, 33)]
    s["model_flat_palm_thumb_down_rejected"] = [frame(t, "OPEN_PALM", mdl("Thumb_Down", 0.77), rotation_deg=90) for t in range(0, 2000, 33)]
    s["model_fist_one_frame_spike"] = [frame(t, "OPEN_PALM", mdl("Closed_Fist", 0.99) if t == 330 else mdl("None", 0.9)) for t in range(0, 1500, 33)]
    s["model_none_sideways_point"] = [frame(t, "POINT", mdl("None", 0.88), rotation_deg=90) for t in range(0, 1200, 33)]
    s["model_pinch_over_open_palm"] = [frame(t, "PINCH", mdl("Open_Palm", 0.8)) for t in range(0, 900, 33)]
    s["model_fist_lock"] = [frame(t, "FIST", mdl("Closed_Fist", 0.93)) for t in range(0, 1500, 33)]
    s["model_none_on_fist_geometry"] = [frame(t, "FIST", mdl("None", 0.9)) for t in range(0, 1500, 33)]
    # walk / run / squat vocabulary
    s["three_fingers_rules"] = seq("THREE_FINGERS", 0, 1200)
    s["model_victory_on_three_fingers"] = [frame(t, "THREE_FINGERS", mdl("Victory", 0.8)) for t in range(0, 1200, 33)]
    s["v_then_three_then_v"] = seq("VICTORY", 0, 1000) + seq("THREE_FINGERS", 1033, 2000) + seq("VICTORY", 2033, 3000)
    s["pinch_tap"] = seq("PINCH", 0, 400) + seq("OPEN_PALM", 433, 1200)
    s["pinch_long_no_tap"] = seq("PINCH", 0, 1500) + seq("OPEN_PALM", 1533, 2200)
    s["pinch_then_hand_lost_no_tap"] = seq("PINCH", 0, 400) + [frame(t) for t in range(433, 800, 33)]
    return s


def fixture_weights(seed=5):
    """Deterministic small MLP weights (test fixture, not the trained model)."""
    rng = np.random.default_rng(seed)
    classes = ["OPEN_PALM", "FIST", "POINT", "TWO_FINGERS", "THREE_FINGERS", "THUMB_UP", "THUMB_DOWN", "OTHER"]
    sizes = [63, 16, len(classes)]
    layers = []
    for i, o in zip(sizes[:-1], sizes[1:]):
        layers.append({"in": i, "out": o, "W": [round(float(v), 6) for v in rng.normal(0, 0.4, i * o)], "b": [round(float(v), 6) for v in rng.normal(0, 0.1, o)]})
    return {"format": "aura-handpose-mlp", "version": 0, "classes": classes, "inputMean": [0.0] * 63, "inputStd": [1.0] * 63, "layers": layers}


def disp(xd, yd):
    return {"cx": 1 - xd / ASPECT, "cy": yd}


def circle(direction, t0, t1, period=1000):
    return [frame(t, "POINT", **disp(0.9 + 0.12 * math.cos(direction * 2 * math.pi * t / period),
                                     0.65 + 0.12 * math.sin(direction * 2 * math.pi * t / period)))
            for t in range(t0, t1 + 1, 33)]


def python_events(frames, net=None):
    engine = GestureRecognitionEngine(copy.deepcopy(DEFAULT_CONFIG["gestures"]), net)
    events, stable, motion, nets = [], [], [], []
    for f in frames:
        feats = None
        if f["hand"]:
            world = np.array(f["hand"]["world"]) if f["hand"]["world"] is not None else None
            feats = extract_features(np.array(f["hand"]["landmarks"]), world, aspect=f["aspect"], mirror=True,
                                     score=f["hand"]["score"], handedness="Right", model=f["hand"].get("model"))
        r = engine.update(feats, f["t"])
        events += [[e["type"], e["gesture"], round(e["t"])] for e in r["events"]]
        stable.append(r["stable"]["gesture"])
        motion.append([round(r["motion"]["spin"]["omega"], 3), round(r["motion"]["walk"]["cadence"], 3)])
        if net is not None:
            nets.append([feats.net["label"], round(feats.net["score"] * 1e4) / 1e4] if feats is not None and feats.net else None)
    out = {"events": events, "stable": stable, "motion": motion}
    if net is not None:
        out["nets"] = nets
    return out


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_both_engines_emit_identical_gesture_events(tmp_path):
    seqs = sequences()
    path = tmp_path / "sequences.json"
    path.write_text(json.dumps(seqs), encoding="utf-8")
    wpath = tmp_path / "weights.json"
    weights = fixture_weights()
    wpath.write_text(json.dumps(weights), encoding="utf-8")
    out = subprocess.run(["node", str(REPO / "test" / "contract" / "gesture-parity.mjs"), str(path), str(wpath)],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    js = json.loads(out)
    net = HandPoseNet(weights)
    for name, frames in seqs.items():
        for key, py in ((name, python_events(frames)), (f"{name}@net", python_events(frames, net))):
            assert py["events"] == js[key]["events"], key
            assert py["stable"] == js[key]["stable"], key
            # normalised speed / direction signals agree too (rounded to 1e-3)
            diffs = [(i, a, b) for i, (a, b) in enumerate(zip(py["motion"], js[key]["motion"])) if a != b]
            assert not diffs, (key, diffs[:3])
            if "nets" in py:
                nd = [(i, a, b) for i, (a, b) in enumerate(zip(py["nets"], js[key]["nets"])) if a != b]
                assert not nd, (key, "net prediction differs", nd[:3])
    labels = {n[0] for k in js if k.endswith("@net") for n in js[k]["nets"] if n}
    assert len(labels) >= 2, f"the fixture net exercises several classes: {labels}"
    # the sequences really exercise the vocabulary and the neutral cases
    seen = {e[1] for name in js for e in js[name]["events"]}
    assert {"OPEN_PALM", "FIST", "PINCH", "POINT", "SWIPE_RIGHT", "SWIPE_LEFT", "TWO_FINGERS", "SPIN", "FINGER_WALK",
            "THUMB_UP", "THUMB_DOWN"} <= seen, seen
    of = lambda n, g: [e[0] for e in js[n]["events"] if e[1] == g]  # noqa: E731
    assert of("spin_cw_slow_then_fast", "SPIN") == ["start"] and of("spin_ccw_then_loss", "SPIN") == ["start", "end"]
    om = [m[0] for m in js["spin_cw_slow_then_fast"]["motion"]]
    assert max(om[60:90]) > 2 * max(om[30:45]) > 0, "faster circling -> larger rate"
    assert min(m[0] for m in js["spin_ccw_then_loss"]["motion"]) < 0, "counter-clockwise is negative"
    assert of("point_line_no_spin", "SPIN") == [] and of("point_still_no_spin", "SPIN") == []
    assert of("static_v_never_walks", "FINGER_WALK") == [] and of("finger_walk_in_phase_rejected", "FINGER_WALK") == []
    assert of("pointing_wobble_not_walk", "FINGER_WALK") == [], "a curled finger is not a leg"
    assert of("finger_walk_steps_then_stop", "FINGER_WALK") == ["start", "end"]
    slow = max(m[1] for m in js["finger_walk_steps_then_stop"]["motion"])
    fast = max(m[1] for m in js["finger_walk_fast"]["motion"])
    assert fast > slow * 1.4 > 0, (slow, fast)
    assert of("thumb_up_laugh", "THUMB_UP") == ["start"] and of("thumb_down_sad", "THUMB_DOWN") == ["start"]
    starts = lambda n: [e[1] for e in js[n]["events"] if e[0] == "start"]  # noqa: E731
    assert starts("model_open_palm") == ["OPEN_PALM"] and starts("model_open_palm_low_score") == []
    assert starts("model_flat_palm_thumb_down_rejected") == [], "geometry disagreement guard"
    assert "FIST" not in starts("model_fist_one_frame_spike"), "no lock from a one-frame spike"
    assert starts("model_none_sideways_point") == ["POINT"] and starts("model_pinch_over_open_palm") == ["PINCH"]
    assert starts("model_fist_lock") == ["FIST"] and starts("model_none_on_fist_geometry") == [], "FIST only from the model"
    assert js["thumbs_up_never_locks"]["events"] == []
    assert js["low_tracking_score_unknown"]["events"] == []
    assert [e[1] for e in js["swipe_right_then_return"]["events"] if e[0] == "event"] == ["SWIPE_RIGHT"]
    assert js["flicker_point"]["events"] == []
    assert starts("three_fingers_rules") == ["THREE_FINGERS"]
    assert starts("model_victory_on_three_fingers") == ["THREE_FINGERS"], "the canned model has no three-finger class"
    assert starts("v_then_three_then_v") == ["TWO_FINGERS", "THREE_FINGERS", "TWO_FINGERS"]
    taps = lambda n: [e[1] for e in js[n]["events"] if e[0] == "event" and e[1] == "PINCH_TAP"]  # noqa: E731
    assert taps("pinch_tap") == ["PINCH_TAP"] and taps("pinch_long_no_tap") == [] and taps("pinch_then_hand_lost_no_tap") == []
    assert [e[:2] for e in js["point_hold_with_dropout"]["events"]] == [["start", "POINT"]]
    fist_starts = [e for e in js["fist_hold_release_rehold"]["events"] if e[0] == "start"]
    assert len(fist_starts) == 2, "release then re-hold triggers again, once"
