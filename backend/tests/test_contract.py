"""Frontend/backend compatibility: Python results consumed by the real JS code."""

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from aura_cv.camera import Frame
from aura_cv.config import DEFAULT_CONFIG
from aura_cv.pipeline import FrameProcessor
from aura_cv.protocol import dumps
from aura_cv.tracking import HandTracker

from fixtures import EMPTY, POSES, ScriptedBackend, blank_frame, make_hand, mp_result

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_python_results_drive_the_browser_controller(tmp_path):
    palm = mp_result(make_hand(POSES["OPEN_PALM"]))
    point = mp_result(make_hand(POSES["POINT"]))
    script = [EMPTY] * 3 + [palm] * 25 + [EMPTY] * 50 + [point] * 15 + [RuntimeError("inference crashed")]
    tracker = HandTracker(ScriptedBackend(script))
    tracker.init()
    proc = FrameProcessor(tracker, copy.deepcopy(DEFAULT_CONFIG["gestures"]), to_rgb=lambda x: x)
    lines = []
    for i in range(len(script)):
        r = proc.process(Frame(i + 1, blank_frame(640, 360), i * 33.0, 1.0))
        r["sentAt"] = r["capturedAt"] + 2
        r["ageAtSendMs"] = 2
        lines.append(dumps(r))
    lines.append(dumps(proc.camera_unavailable({"message": "Camera capture was interrupted"}, len(script) * 33.0)))
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")

    out = subprocess.run(["node", str(REPO / "test" / "contract" / "consume-results.mjs"), str(path)],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    summary = json.loads(out)
    assert summary["errors"] == [], summary["errors"]
    assert summary["count"] == len(script) + 1
    # Held open palm -> exactly one WAVE; held point -> AIM (accepted once WAVE has finished).
    assert summary["accepted"][0] == "WAVE"
    assert "AIM" in summary["accepted"] and summary["accepted"].count("WAVE") == 1
    assert summary["statuses"][0].startswith("NOT_DETECTED:0")
    assert any(s.startswith("DETECTED:1:OPEN_PALM") for s in summary["statuses"])
    assert any(s.startswith("TRACKING_LOST:0") for s in summary["statuses"])
    assert summary["statuses"][-2].startswith("PROCESSING_ERROR:0")
    assert summary["statuses"][-1].startswith("CAMERA_UNAVAILABLE:0")
    assert summary["finalState"] == "IDLE", "hold released after error / camera loss"
