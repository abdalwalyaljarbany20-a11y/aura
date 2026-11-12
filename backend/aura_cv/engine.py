"""Engine facade: owns the one camera, the tracker and the processing loop."""

from __future__ import annotations

import logging
import platform
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from .posenet import HandPoseNet
from . import __version__
from .camera import CameraManager, Frame, default_capture_factory
from .faces import FaceTracker, MediaPipeFaceBackend
from .metrics import now_ms
from .pipeline import FrameProcessor, ProcessingLoop
from .tracking import HandBackend, HandTracker, MediaPipeHandBackend

log = logging.getLogger("aura_cv.engine")


def library_versions() -> dict:
    out = {"python": platform.python_version()}
    for name, mod in (("mediapipe", "mediapipe"), ("opencv", "cv2"), ("numpy", "numpy")):
        try:
            out[name] = __import__(mod).__version__
        except Exception as exc:  # pragma: no cover
            out[name] = f"unavailable ({exc.__class__.__name__})"
    return out


class Engine:
    def __init__(self, config: dict, repo_root: Path, *, backend: HandBackend | None = None, face_backend=None,
                 capture_factory=default_capture_factory,
                 on_status: Callable[[dict], None] | None = None) -> None:
        self.config = config
        b = config["backend"]
        cam = config["camera"]
        model_path = Path(b["modelPath"])
        if not model_path.is_absolute():
            model_path = repo_root / model_path
        gesture_path = Path(b["gestureModelPath"])
        if not gesture_path.is_absolute():
            gesture_path = repo_root / gesture_path
        self.tracker = HandTracker(backend or MediaPipeHandBackend(model_path, config["tracking"], gesture_path))
        # project-trained hand-pose net (optional; training/). Only with the real backend: injected
        # test backends keep the deterministic model + rule fusion.
        self.pose_net = None
        self.pose_net_info = {"status": "disabled"}
        net_path = Path(b.get("handPoseNetPath") or "")
        if not net_path.is_absolute():
            net_path = repo_root / net_path
        if backend is None and config["gestures"].get("net", {}).get("enabled", True) and net_path.is_file():
            try:
                self.pose_net = HandPoseNet.load(net_path)
                self.pose_net_info = {"status": "ready", "file": net_path.name, **self.pose_net.info}
            except Exception as exc:  # malformed weights: report, keep running without the net
                self.pose_net_info = {"status": "error", "error": repr(exc)}
        elif backend is None:
            self.pose_net_info = {"status": "missing" if not net_path.is_file() else "disabled"}
        fc = config["face"]
        if face_backend is None and fc["enabled"] and backend is None:
            face_path = Path(b["faceModelPath"])
            det_path = Path(b["faceDetectorPath"])
            face_backend = MediaPipeFaceBackend(face_path if face_path.is_absolute() else repo_root / face_path, fc["minScore"],
                                                det_path if det_path.is_absolute() else repo_root / det_path, fc["numFaces"])
        self.faces = FaceTracker(face_backend if fc["enabled"] else None, fc["intervalMs"], fc["minScore"])
        self.on_status = on_status
        self.camera = CameraManager(index=b["cameraIndex"], width=cam["width"], height=cam["height"], fps=cam["frameRate"],
                                    api=b["cameraApi"], fourcc=b["cameraFourcc"], max_consecutive_failures=b["maxConsecutiveReadFailures"],
                                    capture_factory=capture_factory, on_status=lambda _s: self._status_changed())
        self.processor: FrameProcessor | None = None
        self.loop: ProcessingLoop | None = None
        self.on_result: Callable[[dict, bytes | None], None] | None = None
        self._lock = threading.Lock()
        self.session = 0  # incremented per camera start; 0 = no session yet

    # ------------------------------------------------------------------ lifecycle
    def init_tracker(self) -> bool:
        ok = self.tracker.init()
        self.faces.init()  # optional: a failure only disables gaze
        self._status_changed()
        return ok

    def start(self) -> tuple[bool, dict | None]:
        """Open the camera and start processing (blocking; call from a worker thread)."""
        with self._lock:
            if self.tracker.status != "ready":
                return False, {"code": "TRACKER_UNAVAILABLE", "message": f"hand tracker {self.tracker.status}",
                               "detail": self.tracker.error}
            if self.loop is not None:
                return True, None
            # New session id *before* the camera reports "live": that status message
            # carries the id the browser will accept results for.
            self.session += 1
            ok, err = self.camera.start()
            if not ok:
                return False, err
            # Fresh temporal state for every camera session.
            self.processor = FrameProcessor(self.tracker, self.config["gestures"], mirror=self.config["camera"]["mirror"],
                                            lost_hold_ms=self.config["backend"]["trackingLostHoldMs"], session=self.session,
                                            face_tracker=self.faces, pose_net=self.pose_net)
            self.loop = ProcessingLoop(self.camera, self.processor, self._handle_result)
            self.loop.start()
            return True, None

    def stop(self) -> None:
        with self._lock:
            if self.loop is not None:
                self.loop.stop()
                self.loop = None
            self.camera.stop()

    def shutdown(self) -> None:
        self.stop()
        self.tracker.close()
        self.faces.close()

    # ------------------------------------------------------------------ results
    def _handle_result(self, result: dict, frame: Frame | None) -> None:
        preview = None
        if frame is not None:
            t0 = now_ms()
            preview = self.encode_preview(frame.image)
            if result.get("timing") is not None:
                result["timing"]["previewEncodeMs"] = round(now_ms() - t0, 3)
        if result.get("status") == "CAMERA_UNAVAILABLE":
            self._status_changed()
        if self.on_result:
            self.on_result(result, preview)

    def encode_preview(self, bgr: np.ndarray) -> bytes | None:
        import cv2

        b = self.config["backend"]
        h, w = bgr.shape[:2]
        if w > b["previewMaxWidth"]:
            scale = b["previewMaxWidth"] / w
            bgr = cv2.resize(bgr, (b["previewMaxWidth"], max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, b["previewJpegQuality"]])
        return buf.tobytes() if ok else None

    # ------------------------------------------------------------------ state
    def status(self) -> dict:
        return {
            "session": self.session,
            "camera": {"status": self.camera.status, "error": self.camera.error, "info": self.camera.info},
            "tracker": {"status": self.tracker.status, "error": self.tracker.error, **self.tracker.info},
            "face": self.faces.info(),
            "poseNet": self.pose_net_info,
        }

    def metrics(self) -> dict | None:
        if self.processor is None:
            return None
        return self.processor.metrics_snapshot(self.camera, now_ms())

    def hello(self, config_errors: list[str], config_source: str) -> dict:
        from .protocol import PROTOCOL_VERSION

        return {"type": "hello", "protocol": PROTOCOL_VERSION,
                "engine": {"name": "aura_cv", "version": __version__, **library_versions()},
                **self.status(), "config": {"source": config_source, "camera": self.config["camera"],
                                            "tracking": self.config["tracking"], "gestures": self.config["gestures"]},
                "configErrors": config_errors}

    def _status_changed(self) -> None:
        if self.on_status:
            try:
                self.on_status(self.status())
            except Exception:  # pragma: no cover
                log.exception("status callback failed")
