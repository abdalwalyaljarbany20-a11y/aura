import copy
import json
import time

import numpy as np
import pytest

from aura_cv.camera import CameraManager, Frame, FrameSlot, validate_frame
from aura_cv.config import DEFAULT_CONFIG
from aura_cv.engine import Engine
from aura_cv.pipeline import FrameProcessor
from aura_cv.protocol import dumps
from aura_cv.tracking import HandTracker, TrackingStateMachine

from fixtures import EMPTY, POSES, FakeCapture, ScriptedBackend, blank_frame, make_hand, mp_result

CFG = copy.deepcopy(DEFAULT_CONFIG)


def factory(cap):
    return lambda index, api: (cap, "FAKE")


def wait_until(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------- frame validation

@pytest.mark.parametrize("frame, ok", [
    (blank_frame(), True),
    (None, False),
    ("x", False),
    (np.zeros((240, 320), dtype=np.uint8), False),
    (np.zeros((240, 320, 3), dtype=np.float32), False),
    (np.zeros((8, 8, 3), dtype=np.uint8), False),
])
def test_validate_frame(frame, ok):
    assert (validate_frame(frame) is None) == ok


# ---------------------------------------------------------------- camera manager

def test_camera_unavailable():
    cam = CameraManager(capture_factory=lambda i, a: (None, None))
    ok, err = cam.start()
    assert not ok and err["code"] == "UNAVAILABLE" and cam.status == "error"


def test_camera_factory_exception_reported():
    def boom(i, a):
        raise RuntimeError("driver crashed")

    ok, err = CameraManager(capture_factory=boom).start()
    assert not ok and err["code"] == "OPENCV_ERROR" and "driver crashed" in err["detail"]


def test_camera_live_frames_and_release():
    cap = FakeCapture([blank_frame()], delay_s=0.005)
    cam = CameraManager(capture_factory=factory(cap))
    ok, err = cam.start()
    assert ok and cam.status == "live" and cam.info["width"] == 320
    f = cam.slot.get(0, 1.0)
    assert f is not None and f.image.shape == (240, 320, 3) and f.acquire_ms >= 0
    cam.stop()
    assert cap.released == 1 and cam.status == "stopped"
    cam.stop()  # idempotent
    assert cap.released == 1


def test_camera_skips_invalid_frames():
    cap = FakeCapture([None, np.zeros((5, 5, 3), np.uint8), blank_frame()], delay_s=0.002)
    cam = CameraManager(capture_factory=factory(cap), max_consecutive_failures=5)
    ok, _ = cam.start()
    assert ok
    assert cam.stats["readFailures"] == 1 and cam.stats["invalid"] == 1
    cam.stop()


def test_camera_no_valid_frames():
    cam = CameraManager(capture_factory=factory(FakeCapture([None], loop_last=True)), max_consecutive_failures=3)
    ok, err = cam.start()
    assert not ok and err["code"] == "NO_FRAMES"


def test_camera_interrupted_after_frames_and_released():
    cap = FakeCapture([blank_frame()] * 3 + [None], loop_last=True, delay_s=0.002)
    cam = CameraManager(capture_factory=factory(cap), max_consecutive_failures=4)
    assert cam.start()[0]
    assert wait_until(lambda: cam.status == "error")
    assert cam.error["code"] == "INTERRUPTED" and cap.released == 1


def test_camera_release_after_read_exception():
    cap = FakeCapture([blank_frame(), RuntimeError("usb reset")], delay_s=0.002)
    cam = CameraManager(capture_factory=factory(cap))
    assert cam.start()[0]
    assert wait_until(lambda: cam.status == "error")
    assert cam.error["code"] == "OPENCV_ERROR" and cap.released == 1


def test_stop_does_not_block_on_hung_driver():
    cap = FakeCapture([blank_frame()], delay_s=0.0)
    cam = CameraManager(capture_factory=factory(cap), join_timeout_s=0.2)
    assert cam.start()[0]
    cap.delay_s = 5.0  # next read() hangs
    time.sleep(0.05)
    t0 = time.monotonic()
    cam.stop()
    assert time.monotonic() - t0 < 1.0


def test_frame_slot_keeps_only_newest():
    slot = FrameSlot()
    for i in range(1, 6):
        slot.put(Frame(i, blank_frame(), float(i), 0.0))
    f = slot.get(0, 0.1)
    assert f.id == 5 and slot.overwritten == 4
    assert slot.get(5, 0.05) is None


# ---------------------------------------------------------------- tracker & tracking state

def test_tracker_init_failure():
    t = HandTracker(ScriptedBackend([EMPTY], fail_init=FileNotFoundError("hand model not found")))
    assert not t.init() and t.status == "error" and "not found" in t.error
    assert not t.detect(np.zeros((4, 4, 3), np.uint8), 0).ok


def test_tracker_parses_validates_and_keeps_timestamps_monotonic():
    good = make_hand(POSES["OPEN_PALM"])
    bad = (good[0][:5], None, "Left", 0.9)
    backend = ScriptedBackend([mp_result(good, bad), EMPTY])
    t = HandTracker(backend)
    t.init()
    r = t.detect(np.zeros((4, 4, 3), np.uint8), 100.4)
    assert r.ok and len(r.hands) == 1 and r.rejected == 1 and r.primary == 0
    t.detect(np.zeros((4, 4, 3), np.uint8), 100.0)
    assert backend.timestamps[1] > backend.timestamps[0]


def test_tracker_backend_exception_contained():
    t = HandTracker(ScriptedBackend([RuntimeError("inference crashed")]))
    t.init()
    r = t.detect(np.zeros((4, 4, 3), np.uint8), 1)
    assert not r.ok and "inference crashed" in r.error


def test_tracking_state_machine():
    s = TrackingStateMachine(lost_hold_ms=500)
    assert s.update(False, 0) == "NOT_DETECTED"
    assert s.update(True, 100) == "DETECTED"
    assert s.update(False, 200) == "TRACKING_LOST"
    assert s.update(False, 600) == "TRACKING_LOST"  # 500 ms after the last detection
    assert s.update(False, 601) == "NOT_DETECTED"
    assert s.losses == 1
    s.update(True, 800)
    assert s.set("CAMERA_UNAVAILABLE") == "CAMERA_UNAVAILABLE" and s.losses == 2


# ---------------------------------------------------------------- frame processor

def processor(script):
    tracker = HandTracker(ScriptedBackend(script))
    tracker.init()
    return FrameProcessor(tracker, CFG["gestures"], to_rgb=lambda x: x)


def frame(i, t):
    return Frame(i, blank_frame(640, 360), t, 1.0)


def test_structured_result_and_gesture_events():
    palm = mp_result(make_hand(POSES["OPEN_PALM"]))
    p = processor([EMPTY] + [palm] * 30 + [EMPTY] * 3 + [EMPTY] * 30)
    results = [p.process(frame(i + 1, i * 33.0)) for i in range(64)]
    first, hand, lost, gone = results[0], results[25], results[31], results[63]
    assert first["status"] == "NOT_DETECTED" and first["hands"] == []
    assert hand["status"] == "DETECTED" and len(hand["hands"][0]["landmarks"]) == 21
    assert hand["features"]["valid"] and hand["gesture"]["stable"] == "OPEN_PALM"
    assert lost["status"] == "TRACKING_LOST" and lost["hands"] == [] and lost["primary"] is None
    assert gone["status"] == "NOT_DETECTED"
    starts = [e for r in results for e in r["events"] if e["type"] == "start"]
    assert [e["gesture"] for e in starts] == ["OPEN_PALM"]
    ends = [e for r in results for e in r["events"] if e["type"] == "end"]
    assert len(ends) == 1, "tracking loss releases the gesture"
    for key in ("acquireMs", "queueMs", "inferenceMs", "featuresMs", "gestureMs", "totalMs"):
        assert hand["timing"][key] is not None
    assert first["timing"]["warmup"] is True and hand["timing"]["warmup"] is False
    json.loads(dumps(hand))  # JSON-serialisable


def test_processing_error_propagates_and_releases_gesture():
    pinch = mp_result(make_hand(POSES["PINCH"]))
    p = processor([pinch] * 10 + [RuntimeError("gpu lost")])
    for i in range(10):
        r = p.process(frame(i + 1, i * 33.0))
    assert r["gesture"]["stable"] == "PINCH"
    err = p.process(frame(11, 400.0))
    assert err["status"] == "PROCESSING_ERROR" and "gpu lost" in err["error"]
    assert err["hands"] == [] and [e["type"] for e in err["events"]] == ["end"]


def test_invalid_frame_is_processing_error():
    p = processor([EMPTY])
    r = p.process(Frame(1, np.zeros((240, 320), np.uint8), 0.0, 0.0))
    assert r["status"] == "PROCESSING_ERROR" and "invalid frame" in r["error"]


def test_camera_unavailable_result():
    p = processor([EMPTY])
    r = p.camera_unavailable({"message": "Camera capture was interrupted"}, 10.0)
    assert r["status"] == "CAMERA_UNAVAILABLE" and r["hands"] == [] and "interrupted" in r["error"]


# ---------------------------------------------------------------- engine (camera + tracker + loop)

def test_engine_end_to_end_with_fakes_and_cleanup(tmp_path):
    palm = mp_result(make_hand(POSES["OPEN_PALM"]))
    cap = FakeCapture([blank_frame(640, 360)], delay_s=0.01)
    results = []
    eng = Engine(copy.deepcopy(CFG), tmp_path, backend=ScriptedBackend([palm]), capture_factory=factory(cap))
    eng.on_result = lambda r, preview: results.append((r, preview))
    assert eng.init_tracker()
    ok, err = eng.start()
    assert ok, err
    assert wait_until(lambda: any(r["gesture"]["stable"] == "OPEN_PALM" for r, _ in results))
    r, preview = results[-1]
    assert preview[:2] == b"\xff\xd8", "preview is a JPEG"
    assert r["timing"]["previewEncodeMs"] is not None
    m = eng.metrics()
    assert m["processedFps"] is None or m["processedFps"] > 0
    eng.shutdown()
    assert cap.released == 1 and eng.tracker.status == "closed"


def test_engine_refuses_start_without_tracker(tmp_path):
    eng = Engine(copy.deepcopy(CFG), tmp_path, backend=ScriptedBackend([EMPTY], fail_init=RuntimeError("no model")),
                 capture_factory=factory(FakeCapture([blank_frame()])))
    eng.init_tracker()
    ok, err = eng.start()
    assert not ok and err["code"] == "TRACKER_UNAVAILABLE"


def test_engine_reports_camera_interruption(tmp_path):
    cap = FakeCapture([blank_frame()] * 3 + [None], loop_last=True, delay_s=0.005)
    results = []
    cfg = copy.deepcopy(CFG)
    cfg["backend"]["maxConsecutiveReadFailures"] = 3
    eng = Engine(cfg, tmp_path, backend=ScriptedBackend([EMPTY]), capture_factory=factory(cap))
    eng.on_result = lambda r, p: results.append(r)
    eng.init_tracker()
    assert eng.start()[0]
    assert wait_until(lambda: any(r["status"] == "CAMERA_UNAVAILABLE" for r in results))
    assert eng.status()["camera"]["error"]["code"] == "INTERRUPTED"
    eng.shutdown()
