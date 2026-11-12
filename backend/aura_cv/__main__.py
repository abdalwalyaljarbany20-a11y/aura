"""Command line entry point.

    python -m aura_cv [--config ../public/aura.config.json] [--camera 0] [--port 8765] [--host 127.0.0.1]
    python -m aura_cv --check-image photo.jpg   # run the real model once on a still image and print the result
    python -m aura_cv --list-cameras            # probe camera indices 0-4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .config import load_config_file

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aura_cv", description="AURA VISION Python computer-vision engine")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "public" / "aura.config.json",
                   help="shared configuration file (optional)")
    p.add_argument("--camera", type=int, help="camera index (overrides backend.cameraIndex)")
    p.add_argument("--fourcc", help='camera pixel format, e.g. MJPG (overrides backend.cameraFourcc)')
    p.add_argument("--host", help="bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, help="port (default 8765)")
    p.add_argument("--check-image", type=Path, help="run the real model once on a still image and exit")
    p.add_argument("--list-cameras", action="store_true", help="probe camera indices 0-4 and exit")
    p.add_argument("--test-input", type=Path,
                   help="TEST ONLY: replay an image/video file instead of the camera (reported as TEST-FILE)")
    p.add_argument("--benchmark", type=float, metavar="SECONDS",
                   help="run the engine without the browser for SECONDS and print measured statistics (JSON)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config, errors, source = load_config_file(args.config)
    for e in errors:
        logging.getLogger("aura_cv").error("config: %s", e)
    b = config["backend"]
    if args.camera is not None:
        b["cameraIndex"] = args.camera
    if args.fourcc:
        b["cameraFourcc"] = args.fourcc
    if args.host:
        b["host"] = args.host
    if args.port:
        b["port"] = args.port

    if args.list_cameras:
        return _list_cameras(b["cameraApi"])
    if args.check_image:
        return _check_image(config, args.check_image)

    from .engine import Engine
    from .server import EngineServer

    factory = {}
    if args.test_input:
        from .camera import file_capture_factory

        logging.getLogger("aura_cv").warning("TEST INPUT: frames come from %s, not from a camera", args.test_input)
        factory = {"capture_factory": file_capture_factory(str(args.test_input), config["camera"]["frameRate"])}
    engine = Engine(config, REPO_ROOT, **factory)
    if args.benchmark:
        return _benchmark(engine, args.benchmark, "test-file" if args.test_input else "camera")
    try:
        asyncio.run(EngineServer(engine, errors, source).run(b["host"], b["port"]))
    except KeyboardInterrupt:
        logging.getLogger("aura_cv").info("interrupted; shutting down")
    except OSError as exc:
        logging.getLogger("aura_cv").error("could not start server on %s:%s: %s", b["host"], b["port"], exc)
        return 1
    finally:
        engine.shutdown()
    return 0


def _benchmark(engine, seconds: float, source_kind: str) -> int:
    """Measure the real pipeline (camera or TEST-FILE) without the browser.

    Frames are processed and JPEG-encoded exactly as for the browser, then discarded;
    nothing is written to disk. All durations use time.perf_counter (monotonic).
    """
    import time
    from collections import Counter

    from .metrics import RollingStats

    stats = {"previewBytes": RollingStats(100000), "encodeMs": RollingStats(100000), "inferenceMs": RollingStats(100000),
             "totalMs": RollingStats(100000), "acquireMs": RollingStats(100000)}
    statuses: Counter = Counter()
    events: list = []

    def on_result(r, preview):
        statuses[r["status"]] += 1
        events.extend(f"{e['type']}:{e['gesture']}" for e in r["events"])
        tm = r.get("timing") or {}
        if not tm.get("warmup") and r["status"] != "PROCESSING_ERROR":
            for k in ("inferenceMs", "totalMs", "acquireMs"):
                stats[k].push(tm.get(k))
        if tm.get("previewEncodeMs") is not None:
            stats["encodeMs"].push(tm["previewEncodeMs"])
        if preview:
            stats["previewBytes"].push(len(preview))

    engine.on_result = on_result
    if not engine.init_tracker():
        print(json.dumps({"error": f"tracker: {engine.tracker.error}"}))
        return 3
    t_open = time.perf_counter()
    ok, err = engine.start()
    open_ms = (time.perf_counter() - t_open) * 1000
    if not ok:
        print(json.dumps({"error": err, "cameraOpenMs": round(open_ms, 1)}))
        engine.shutdown()
        return 4
    t0 = time.perf_counter()
    try:
        time.sleep(seconds)
    except KeyboardInterrupt:
        pass
    elapsed = time.perf_counter() - t0
    m = engine.metrics()
    info = dict(engine.camera.info or {})
    captured = engine.camera.stats["captured"]
    engine.shutdown()

    def s(rs):
        d = rs.summary()
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k != "last"}

    out = {
        "source": source_kind,  # "camera" = physical device; "test-file" = replayed file (NOT a camera benchmark)
        "camera": info,
        "durationS": round(elapsed, 2),
        "cameraOpenMs": round(open_ms, 1),
        "captureFpsAvg": round(captured / elapsed, 2) if elapsed > 0 else None,
        "processedFpsAvg": round(m["session"]["frames"] / elapsed, 2) if elapsed > 0 else None,
        "droppedFrames": m["droppedFrames"],
        "warmupInferenceMs": m["warmupInferenceMs"],
        "inferenceMs": s(stats["inferenceMs"]),
        "acquireMs": s(stats["acquireMs"]),
        "captureToResultMs": s(stats["totalMs"]),
        "previewEncodeMs": s(stats["encodeMs"]),
        "previewBytes": s(stats["previewBytes"]),
        "statuses": dict(statuses),
        "gestureEvents": dict(Counter(events)),
        "session": m["session"],
    }
    print(json.dumps(out, indent=2))
    return 0


def _list_cameras(api: str) -> int:
    from .camera import default_capture_factory

    for i in range(5):
        cap, backend = default_capture_factory(i, api)
        if cap is None:
            print(f"camera {i}: not available")
            continue
        ok, frame = cap.read()
        cap.release()
        shape = f"{frame.shape[1]}x{frame.shape[0]}" if ok and frame is not None else "no frame"
        print(f"camera {i}: opened via {backend}, first frame {shape}")
    return 0


def _check_image(config: dict, path: Path) -> int:
    """Diagnostic: the real MediaPipe model on one still image (clearly not live input)."""
    import cv2

    from .camera import Frame
    from .metrics import now_ms
    from .pipeline import FrameProcessor
    from .tracking import HandTracker, MediaPipeHandBackend

    image = cv2.imread(str(path))
    if image is None:
        print(f"could not read image {path}", file=sys.stderr)
        return 2
    model = Path(config["backend"]["modelPath"])
    tracker = HandTracker(MediaPipeHandBackend(model if model.is_absolute() else REPO_ROOT / model, config["tracking"]))
    if not tracker.init():
        print(f"tracker error: {tracker.error}", file=sys.stderr)
        return 3
    proc = FrameProcessor(tracker, config["gestures"], mirror=config["camera"]["mirror"])
    # Feed the same still image for 1 s of timestamps so the stabiliser can settle.
    t0 = now_ms()
    result = None
    events = []
    for i in range(31):
        result = proc.process(Frame(i + 1, image, t0 + i * 33.3, 0.0))
        events.extend(result["events"])
    tracker.close()
    summary = {k: result[k] for k in ("status", "primary", "features", "gesture", "timing")}
    summary["source"] = f"still image {path.name} (diagnostic, not live camera input)"
    summary["hands"] = [{"handedness": h["handedness"], "score": h["score"]} for h in result["hands"]]
    summary["events"] = events
    summary["inferenceMs"] = proc.metrics["inferenceMs"].summary()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
