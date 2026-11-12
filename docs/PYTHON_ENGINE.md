# Python computer-vision engine (`backend/aura_cv`)

This is the Python implementation of the AURA VISION CV pipeline. It owns the camera, runs MediaPipe hand landmarks, extracts features, recognizes and stabilizes gestures, and streams structured results and preview frames to the existing browser frontend over a localhost WebSocket.

## Engine modes

There is exactly one camera owner per mode:

| `engine.mode` | Camera owner | Landmarks | Gestures | Start |
|---|---|---|---|---|
| `browser` (default) | browser `getUserMedia` | MediaPipe WASM in the page | JS engine (`src/gestures`) | `npm run dev` |
| `python` | Python OpenCV | MediaPipe Tasks (Python) | Python engine (`aura_cv.gestures`) | `npm run dev:python`, then open `/?engine=python` |

- **No double camera access.** In Python mode the page never calls `getUserMedia`, and the engine opens one `cv2.VideoCapture`. Windows webcams generally cannot be opened by two processes at once.
- **Single source of gesture decisions.** In Python mode the JS gesture engine is bypassed, and gesture decisions come only from Python. The browser uses the JS feature extractor only for render geometry (palm centre, arcs, aim point).
- **Shared downstream.** Both modes share the interaction controller, robot, effects and HUD unchanged.

## Pipeline

```
CameraManager (capture thread)          one cv2.VideoCapture; read() in a loop
  └─ FrameSlot (size 1)                 newest frame only; overwritten frames counted as dropped
ProcessingLoop (processing thread)
  └─ FrameProcessor.process(frame)
       validate_frame                   uint8 HxWx3, >= 16 px
       HandTracker.detect               BGR->RGB, MediaPipe HandLandmarker (VIDEO mode, monotonic ts)
         parse + landmark_set_error     21x3, finite, normalized x,y within [-0.5, 1.5]
         select_primary                 sticky handedness, else largest hand
       extract_features                 all hands (see "Features")
       GestureRecognitionEngine.update  primary hand (rules -> motion gate -> stabilizer, swipes)
       TrackingStateMachine.update      DETECTED / NOT_DETECTED / TRACKING_LOST
       -> result dict                   + JPEG preview (cv2.imencode)
Outbox (latest result, events merged)   processing thread -> asyncio loop
EngineServer (websockets, asyncio)      -> browser: result JSON + binary preview with the same frameId
```

There are two threads, and both are needed:
- **Capture thread.** Without it, OpenCV drivers buffer frames while inference runs and later return stale frames.
- **Processing thread.** It keeps blocking inference off the asyncio loop, so stop and disconnect stay responsive.

Each hand-off holds at most one item, so there is no unbounded queue anywhere. If the browser falls behind, an unsent result is replaced by the newer one. Its gesture events are carried over, so no command is lost.

Shutdown and cleanup:
- **On client stop, disconnect or Ctrl+C:** the processing loop stops, then the camera thread, and the capture is released in the thread's `finally`. Exceptions take the same path.
- **Hung driver:** `stop()` waits at most 2 s. A driver hung inside `read()` cannot block shutdown, and the device is released when `read()` returns.
- **Tracker:** closed on server exit.

## Tracking statuses

| Status | Meaning | Landmarks in the result |
|---|---|---|
| `DETECTED` | valid primary hand in this frame | yes (this frame only) |
| `TRACKING_LOST` | a hand was detected ≤ `trackingLostHoldMs` (500 ms) ago, but not now | **none**: stale landmarks are never repeated |
| `NOT_DETECTED` | no hand now and none recently | none |
| `PROCESSING_ERROR` | invalid frame, or inference raised | none; gestures and holds are released |
| `CAMERA_UNAVAILABLE` | the camera stopped delivering (interrupted, read failures) | none; gestures and holds are released |

The camera's error codes are `UNAVAILABLE`, `NO_FRAMES`, `INTERRUPTED` (after `maxConsecutiveReadFailures`, default 15) and `OPENCV_ERROR`. `TRACKER_UNAVAILABLE` is returned when the model failed to load.

## Features (`aura_cv/features.py`)

The input is a `(21, 3)` NumPy array in MediaPipe order: normalized image landmarks, plus optional metric world landmarks. Definitions match the browser engine exactly. A test asserts that the default thresholds are identical in both languages.

| Feature | Definition |
|---|---|
| `palm_size` | mean wrist→index-MCP, wrist→middle-MCP and wrist→pinky-MCP distance; all shape distances are divided by it |
| `straightness(f)` | \|base→tip\| / sum of segment lengths over the finger's 4-joint chain |
| `extension(f)` | `ramp(straightness, 0.55 → 0.92)` |
| thumb extension | additionally `min(…, ramp(abduction, 0.45 → 0.8))`, where abduction is thumb tip→index MCP divided by palm_size |
| `pinch_distance` | thumb tip→index tip, divided by palm_size |
| `palm_center`, `palm_size_image` | display space (mirrored if configured), in units of image height; used for motion |
| `in_frame_ratio` | fraction of landmarks inside the image |

These features are invariant to translation, scale and in-plane rotation, so there are no absolute pixel thresholds. Shape features use world landmarks when they are valid. Zero-length vectors produce `degenerate hand`, and non-finite values are rejected.

Joint angles and palm-normal estimates are not computed, because no implemented gesture needs them.

## Gestures and stabilization (`aura_cv/gestures.py`)

The rules are identical to the browser engine; see [ARCHITECTURE.md](ARCHITECTURE.md#gesture-definitions). In short:

- **Static gestures:** `OPEN_PALM`, `FIST`, `PINCH` and `POINT`, as rule scores. Low tracking score, partial visibility, a weak best score or an ambiguous margin all give `UNKNOWN`. No hand gives `NONE`.
- **Swipes:** `SWIPE_LEFT` and `SWIPE_RIGHT` come from a timestamped palm history. They are kept so that Python mode has the same vocabulary as browser mode.
- **Stabilizer (hysteresis):**
  - A gesture becomes stable after a continuous `holdMs` (per gesture) and is released after `exitMs` of absence.
  - Exactly one `start` event is emitted per continuous hold. The reset condition is: release the gesture, then hold it again.
  - Tracking loss, a processing error or camera loss release the gesture immediately.
- **Motion gate:** OPEN_PALM and FIST do not start while the hand moves faster than `maxStillSpeed`.
- **Configuration:** all parameters live in the shared config file's `gestures` section.

## Protocol (v2)

See [`aura_cv/protocol.py`](../backend/aura_cv/protocol.py) for the message list. The key points:

- **Messages:**
  - Text frames carry JSON: `hello`, `status`, `result`, `metrics` (2 Hz), `error`.
  - Binary frames carry the preview: `"AVF2"` + uint32 session + uint32 frameId + JPEG.
- **Sessions (new in v2):**
  - Each camera `start` gets a new `session` id. It is allocated before the camera reports `live`, and that status message announces it.
  - Results and previews carry the id.
  - The browser accepts only the current session, and only frameIds newer than the last accepted one.
  - v1 clients and previews are rejected.
- **Two delivery paths in the browser** (`RemoteCamera`, `RemoteFramePipeline`):
  - *State and events*: each accepted result is applied immediately. Gesture events reach the `InteractionController` without waiting for any preview, so a missing preview can never delay or lose a release.
  - *Visuals*: a preview is drawn only if it is the frame of the newest accepted result, and never older than the frame already on screen. The overlay then shows that frame's own result. If the newest result has no hand, the overlay is empty. If the newest result's preview is late by more than 250 ms, the overlay is hidden instead of showing a misaligned hand.
  - Counters: `previewsDrawn`, `previewsMissed`, `previewsStale`, `staleResults`.
- **Bounded buffers:** one frame in the capture slot, one result + preview in the outbox, one awaited preview in the browser, and at most 8 recent results kept for pairing.
- **Browser → engine:** `hello` (protocol checked), `start`, `stop`, `ping`. Messages are limited to 4 KB; anything else gets an `error` reply.
- **Security:**
  - The server binds to `127.0.0.1` only.
  - It accepts one client at a time; a second client is closed with code 1013.
  - It checks the browser Origin (`backend.allowedOrigins`, default `http://127.0.0.1` and `http://localhost` on any port), so other websites cannot read the camera stream through your browser.
  - Clients without an Origin header (local tools) are allowed.
  - Frames are never written to disk.

## Failure behaviour and recovery

| Situation | Engine | Browser | Recovery |
|---|---|---|---|
| Engine not running when the page opens | n/a | "not reachable … npm run backend"; robot ERROR | client retries with backoff of 0.5 → 1 → 2 → 4 → 5 s (±20 % jitter, capped); the dialog changes to "available" once reconnected; the user clicks Start |
| Engine exits / WebSocket closes | n/a | camera INTERRUPTED, robot ERROR, holds released, no landmarks; HUD counts outages | same backoff; the camera is **not** restarted automatically; the outage duration is measured |
| Engine busy (another tab) | closes with 1013 | "engine busy" | backoff is not reset by such connects (no reconnect storm) |
| Camera cannot open | `UNAVAILABLE` (DirectShow, then MSMF) | dialog + robot ERROR | fix the device or index, then Retry |
| Camera stops delivering | `INTERRUPTED` after 15 failed reads; `CAMERA_UNAVAILABLE` result releases gestures | robot ERROR | Retry |
| Model missing | tracker error at start-up; `start` → `TRACKER_UNAVAILABLE` | HUD + dialog | `npm run setup`, then restart the engine |
| Invalid message from the browser | `error` reply; connection kept | n/a | n/a |
| Malformed message from the engine | n/a | dropped and counted (`stats.malformed`) | n/a |
| Internal send failure | logged with traceback; connection closed with 1011 | treated as a disconnect | backoff reconnect |

Windows does not report camera-permission denial to OpenCV separately. A camera blocked in privacy settings appears as `UNAVAILABLE`.

## Benchmark (reproducible measurements)

```powershell
npm run bench:python                                  # physical camera, 20 s, no browser
npm run backend -- --benchmark 20 --camera 1          # another camera
npm run backend -- --benchmark 10 --test-input x.jpg  # pipeline check with a file (NOT a camera benchmark)
```

The benchmark prints JSON:
- `source` (`camera` or `test-file`) and the camera info;
- `cameraOpenMs`, `captureFpsAvg`, `processedFpsAvg`, `droppedFrames`, `warmupInferenceMs`;
- mean and p95 of `inferenceMs`, `acquireMs` (time blocked in `read()`), `captureToResultMs`, `previewEncodeMs` and `previewBytes`;
- tracking-status counts and gesture events.

Nothing is written to disk.

For the browser side (preview decode, frame → overlay, render FPS), use the HUD or `window.__aura.snapshot().engine.preview`.

## Configuration

The engine reads `public/aura.config.json`, the same optional file as the frontend. It uses the `camera`, `tracking` and `gestures` sections, plus a Python-only `backend` section:

| Key | Default | Meaning |
|---|---|---|
| `host` / `port` | `127.0.0.1` / `8765` | bind address |
| `cameraIndex` | `0` | OpenCV device index (CLI `--camera N`) |
| `cameraApi` | `auto` | `auto` (Windows: DirectShow, then Media Foundation), `dshow`, `msmf` |
| `modelPath` | `public/models/hand_landmarker.task` | relative to the repo root (shared with the browser engine) |
| `previewMaxWidth` / `previewJpegQuality` | `960` / `70` | preview stream size and quality |
| `trackingLostHoldMs` | `500` | TRACKING_LOST window |
| `maxConsecutiveReadFailures` | `15` | then INTERRUPTED |
| `allowedOrigins` | localhost origins | browser origins allowed to connect |

Invalid values are rejected individually: the default is kept and the error is logged and sent to the HUD.

To use the Python engine by default, set `"engine": { "mode": "python" }` in the same file.

## Setup (Windows 11)

```powershell
npm install
npm run setup          # WASM runtime + hand_landmarker.task (shared by both engines)
npm run setup:python   # creates .venv (py -3) and installs backend/requirements-dev.txt
npm run dev:python     # Python engine + Vite; open http://127.0.0.1:5173/?engine=python
```

Verified versions:

| Component | Version |
|---|---|
| Python | 3.14.7 |
| mediapipe | 1.1.1 |
| opencv-contrib-python | 5.0.0.93 |
| numpy | 2.5.4 |
| websockets | 17.3 |
| pytest | 9.1.1 |

Things to know about these versions:
- **API.** mediapipe 1.x has no `mediapipe.solutions`; this engine uses `mediapipe.tasks.python.vision.HandLandmarker`.
- **OpenCV package.** mediapipe depends on `opencv-contrib-python`. Do not install `opencv-python` alongside it, because both provide `cv2`.
- **GPU.** The Windows Python package is built without GPU support ("GPU processing is disabled in build flags"). A `GPU` delegate request falls back to CPU (XNNPACK) and is reported as `CPU (GPU unavailable)`.

Other commands:

```powershell
npm run backend -- --camera 1        # engine only (Ctrl+C to stop; releases the camera)
npm run backend -- --list-cameras    # probe indices 0-4 (opens each camera briefly)
npm run backend -- --check-image photo.jpg   # run the real model on one still image (diagnostic)
npm run test:py                      # Python tests
```

`--test-input FILE` replays an image or video in place of the camera, for automated checks. It is reported as `TEST-FILE` in the protocol, the logs and the HUD.

## Troubleshooting

| Symptom | Fix |
|---|---|
| HUD: "Python engine not reachable" | Start it (`npm run backend` or `npm run dev:python`). The page keeps retrying; click Start camera once the engine is up. |
| `UNAVAILABLE` camera error | Wrong index (`--list-cameras`), camera in use by another app, or Windows privacy settings (Settings → Privacy & security → Camera → allow desktop apps). |
| Very slow camera open | Media Foundation can take several seconds; try `"cameraApi": "dshow"`. |
| `engine busy` | Another page or tab is connected; only one client is allowed. |
| `NO_FRAMES` / `INTERRUPTED` | The driver delivered no or invalid frames; reconnect the camera or use another `cameraApi`. |
| `hand model not found` | `npm run setup`. |

## Extension points

| To add… | Do this |
|---|---|
| Another landmark model | Implement `HandBackend` (`init`, `detect`, `close`) returning MediaPipe-shaped results (21 landmarks). |
| A learned classifier | Replace `StaticGestureClassifier.classify(features, enabled)`; stabilization and protocol stay unchanged. |
| Two-hand gestures | All hands' observations are already in every result; add a recognizer over `det.hands` in `FrameProcessor`. |
| Another camera source | Pass a `capture_factory` to `Engine`; anything with `isOpened()`, `read()` and `release()` works. |
| An agent or voice integration | Go through the browser's `TaskBroker` ([AGENT_INTERFACE.md](AGENT_INTERFACE.md)); gesture results stay on their own path. |
