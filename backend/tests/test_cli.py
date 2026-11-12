"""CLI benchmark: measured output from the real engine code path (fake camera + scripted model)."""

import copy
import json

from aura_cv.__main__ import _benchmark
from aura_cv.config import DEFAULT_CONFIG
from aura_cv.engine import Engine

from fixtures import POSES, FakeCapture, ScriptedBackend, blank_frame, make_hand, mp_result


def test_benchmark_reports_measured_statistics(tmp_path, capsys):
    cap = FakeCapture([blank_frame(640, 360)], delay_s=0.01)
    eng = Engine(copy.deepcopy(DEFAULT_CONFIG), tmp_path, backend=ScriptedBackend([mp_result(make_hand(POSES["FIST"]))]),
                 capture_factory=lambda i, a: (cap, "FAKE"))
    assert _benchmark(eng, 1.0, "test-file") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["source"] == "test-file"
    assert out["captureFpsAvg"] > 10 and out["processedFpsAvg"] > 10
    assert out["inferenceMs"]["samples"] > 5 and out["previewEncodeMs"]["mean"] > 0
    assert out["previewBytes"]["mean"] > 100
    assert out["statuses"]["DETECTED"] > 5
    assert out["gestureEvents"] == {"start:FIST": 1}
    assert cap.released == 1, "camera released after the benchmark"


def test_benchmark_reports_camera_failure(tmp_path, capsys):
    eng = Engine(copy.deepcopy(DEFAULT_CONFIG), tmp_path, backend=ScriptedBackend([]), capture_factory=lambda i, a: (None, None))
    assert _benchmark(eng, 0.1, "camera") == 4
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "UNAVAILABLE"
