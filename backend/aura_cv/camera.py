"""Camera acquisition (OpenCV) with a single latest-frame slot.

Design:
  * exactly one ``cv2.VideoCapture`` per CameraManager; ``start`` refuses to open a
    second one while running
  * a capture thread reads continuously and overwrites a one-frame slot, so the
    processing loop always gets the newest frame and stale frames never queue up
    (overwritten frames are counted as ``dropped``)
  * empty / malformed frames are counted and skipped; after
    ``max_consecutive_failures`` consecutive failures the camera is reported as
    INTERRUPTED and capture stops
  * the capture is released in the capture thread's ``finally`` (normal stop and
    exceptions); ``stop`` waits at most ``join_timeout_s`` so a hung driver cannot
    block shutdown indefinitely

The VideoCapture factory is injectable so tests run without a camera.
"""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .metrics import RateMeter, now_ms

log = logging.getLogger("aura_cv.camera")

CAMERA_ERRORS = {
    "UNAVAILABLE": "Camera could not be opened (missing, in use, or blocked by privacy settings)",
    "NO_FRAMES": "Camera opened but delivered no valid frames",
    "INTERRUPTED": "Camera capture was interrupted (device disconnected or stopped delivering frames)",
    "OPENCV_ERROR": "OpenCV raised an error while accessing the camera",
}


def validate_frame(frame: Any) -> str | None:
    """Reason a frame is unusable, or None. Expects BGR uint8 (H, W, 3)."""
    if frame is None:
        return "empty frame"
    if not isinstance(frame, np.ndarray):
        return "frame is not an array"
    if frame.dtype != np.uint8:
        return f"unexpected dtype {frame.dtype}"
    if frame.ndim != 3 or frame.shape[2] != 3:
        return f"unexpected shape {frame.shape}"
    if frame.shape[0] < 16 or frame.shape[1] < 16:
        return f"frame too small {frame.shape[1]}x{frame.shape[0]}"
    return None


@dataclass
class Frame:
    id: int
    image: np.ndarray  # BGR uint8
    captured_at: float  # ms, monotonic (time read() returned)
    acquire_ms: float  # time spent inside read()


class FrameSlot:
    """Holds only the newest frame. ``get`` waits for a frame newer than ``after_id``."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self.overwritten = 0
        self._consumed_id = 0
        self._closed = False

    def put(self, frame: Frame) -> None:
        with self._cond:
            if self._frame is not None and self._frame.id > self._consumed_id:
                self.overwritten += 1
            self._frame = frame
            self._cond.notify_all()

    def get(self, after_id: int, timeout_s: float) -> Frame | None:
        with self._cond:
            self._cond.wait_for(lambda: self._closed or (self._frame is not None and self._frame.id > after_id), timeout_s)
            f = self._frame
            if f is None or f.id <= after_id:
                return None
            self._consumed_id = f.id
            return f

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def reopen(self) -> None:
        with self._cond:
            self._closed = False
            self._frame = None
            self._consumed_id = 0
            self.overwritten = 0


def default_capture_factory(index: int, api: str):
    import cv2

    if api == "dshow":
        apis = [cv2.CAP_DSHOW]
    elif api == "msmf":
        apis = [cv2.CAP_MSMF]
    elif sys.platform == "win32":
        # DirectShow usually opens much faster than Media Foundation on Windows.
        apis = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    else:
        apis = [cv2.CAP_ANY]
    last = None
    for a in apis:
        cap = cv2.VideoCapture(index, a)
        if cap.isOpened():
            return cap, cap.getBackendName()
        cap.release()
        last = a
    log.warning("camera %s could not be opened with backends %s (last %s)", index, apis, last)
    return None, None


class FileCapture:
    """TEST INPUT ONLY: replays an image or video file through the camera interface.

    Used for automated end-to-end checks without a physical camera. The real model and
    pipeline run on the file's pixels; the source is reported as "TEST-FILE" everywhere
    so it can never be mistaken for live camera input.
    """

    def __init__(self, path: str, fps: float = 30.0) -> None:
        import cv2

        self.path = path
        self.interval = 1.0 / max(1.0, fps)
        self._still = cv2.imread(path)
        self._video = None if self._still is not None else cv2.VideoCapture(path)
        self._next = 0.0
        self.source_label = f"TEST-FILE {path.replace(chr(92), '/').rsplit('/', 1)[-1]}"

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV API
        return self._still is not None or (self._video is not None and self._video.isOpened())

    def set(self, *_):
        return False

    def get(self, *_):
        return 1.0 / self.interval

    def read(self):
        import time

        delay = self._next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        self._next = time.perf_counter() + self.interval
        if self._still is not None:
            return True, self._still.copy()
        ok, frame = self._video.read()
        if not ok:  # loop the file
            self._video.set(1, 0)  # CAP_PROP_POS_FRAMES
            ok, frame = self._video.read()
        return ok, frame

    def release(self) -> None:
        if self._video is not None:
            self._video.release()


def file_capture_factory(path: str, fps: float = 30.0):
    def factory(index: int, api: str):
        cap = FileCapture(path, fps)
        return (cap, "TEST-FILE") if cap.isOpened() else (None, None)

    return factory


class CameraManager:
    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: float = 30,
        api: str = "auto",
        fourcc: str = "auto",
        max_consecutive_failures: int = 15,
        capture_factory: Callable[[int, str], tuple[Any, str | None]] = default_capture_factory,
        first_frame_timeout_s: float = 5.0,
        join_timeout_s: float = 2.0,
        on_status: Callable[[dict], None] | None = None,
    ) -> None:
        self.index, self.width, self.height, self.fps, self.api = index, width, height, fps, api
        self.fourcc = fourcc
        self.max_failures = max_consecutive_failures
        self.capture_factory = capture_factory
        self.first_frame_timeout_s = first_frame_timeout_s
        self.join_timeout_s = join_timeout_s
        self.on_status = on_status
        self.slot = FrameSlot()
        self.status = "stopped"  # stopped | starting | live | error
        self.error: dict | None = None
        self.info: dict | None = None
        self.stats = {"captured": 0, "invalid": 0, "readFailures": 0}
        self.capture_rate = RateMeter(1000)
        self._cap = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._first = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> tuple[bool, dict | None]:
        """Open the camera and wait for the first valid frame. Blocking (run off the event loop)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self.status == "live", self.error
            self._set_status("starting")
            self.error = None
            self.stats = {"captured": 0, "invalid": 0, "readFailures": 0}
            self.capture_rate = RateMeter(1000)
            self.slot.reopen()
            try:
                cap, backend = self.capture_factory(self.index, self.api)
            except Exception as exc:  # OpenCV can raise cv2.error on broken drivers
                return self._fail("OPENCV_ERROR", repr(exc))
            if cap is None:
                return self._fail("UNAVAILABLE", f"camera index {self.index}")
            self._configure(cap)
            self._cap = cap
            self._stop.clear()
            self._first.clear()
            self.info = {"index": self.index, "backend": backend, "width": None, "height": None,
                         "reportedFps": self._get(cap, "CAP_PROP_FPS"),
                         "source": getattr(cap, "source_label", "camera"),
                         "fourcc": self._fourcc_of(cap), "requestedFourcc": self.fourcc}
            self._thread = threading.Thread(target=self._run, name="aura-camera", daemon=True)
            self._thread.start()

        if not self._first.wait(self.first_frame_timeout_s) or self.status == "error":
            if self.status != "error":
                self.stop()
                return self._fail("NO_FRAMES", f"no valid frame within {self.first_frame_timeout_s}s")
            return False, self.error
        self._set_status("live")
        return True, None

    def stop(self) -> None:
        """Stop capture and release the device. Safe to call repeatedly."""
        self._stop.set()
        self.slot.close()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(self.join_timeout_s)
            if t.is_alive():
                log.warning("camera thread did not stop within %.1fs; it will release the device when read() returns",
                            self.join_timeout_s)
        self._thread = None
        if self.status in ("live", "starting"):
            self._set_status("stopped")

    def __enter__(self) -> "CameraManager":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------------ capture thread
    def _run(self) -> None:
        cap = self._cap
        frame_id = 0
        failures = 0
        try:
            while not self._stop.is_set():
                t0 = now_ms()
                try:
                    ok, image = cap.read()
                except Exception as exc:
                    self._fail("OPENCV_ERROR", repr(exc))
                    break
                t1 = now_ms()
                reason = None if ok else "read() returned False"
                if reason is None:
                    reason = validate_frame(image)
                if reason is not None:
                    failures += 1
                    self.stats["readFailures" if not ok else "invalid"] += 1
                    if failures >= self.max_failures:
                        code = "INTERRUPTED" if self._first.is_set() else "NO_FRAMES"
                        self._fail(code, f"{failures} consecutive failures (last: {reason})")
                        break
                    continue
                failures = 0
                frame_id += 1
                self.stats["captured"] += 1
                self.capture_rate.tick(t1)
                if self.info is not None and self.info["width"] is None:
                    self.info["height"], self.info["width"] = int(image.shape[0]), int(image.shape[1])
                self.slot.put(Frame(frame_id, image, t1, t1 - t0))
                self._first.set()
        finally:
            try:
                cap.release()
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("release failed: %r", exc)
            self._cap = None
            self.slot.close()
            self._first.set()  # unblock start() if we failed before the first frame

    # ------------------------------------------------------------------ helpers
    @property
    def dropped(self) -> int:
        return self.slot.overwritten

    @property
    def aspect(self) -> float:
        if self.info and self.info.get("width") and self.info.get("height"):
            return self.info["width"] / self.info["height"]
        return self.width / self.height

    def _configure(self, cap) -> None:
        try:
            import cv2

            if self.fourcc != "auto":  # must precede the size request on DirectShow
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # honoured by some backends only
        except Exception as exc:  # fake captures in tests may not support set()
            log.debug("capture configuration skipped: %r", exc)

    @staticmethod
    def _fourcc_of(cap) -> str | None:
        """Pixel format actually negotiated with the driver (as reported by OpenCV)."""
        try:
            import cv2

            v = int(cap.get(cv2.CAP_PROP_FOURCC))
            code = "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))
            return code if v and code.isprintable() and code.strip() else None
        except Exception:
            return None

    @staticmethod
    def _get(cap, prop: str):
        try:
            import cv2

            v = cap.get(getattr(cv2, prop))
            return float(v) if v and v > 0 else None
        except Exception:
            return None

    def _fail(self, code: str, detail: str | None = None) -> tuple[bool, dict]:
        self.error = {"code": code, "message": CAMERA_ERRORS.get(code, code), "detail": detail}
        self._set_status("error")
        log.error("camera error %s: %s", code, detail)
        return False, self.error

    def _set_status(self, status: str) -> None:
        self.status = status
        if self.on_status:
            try:
                self.on_status({"status": status, "error": self.error, "info": self.info})
            except Exception as exc:  # pragma: no cover
                log.warning("status callback failed: %r", exc)
