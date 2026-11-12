# AURA VISION

Real-time hand tracking and gesture interaction with a 3D robot companion, running entirely in the browser.

A live camera feed is processed frame by frame by local deep-learning models: MediaPipe **GestureRecognizer** (21-point hand landmarks plus a learned gesture classifier) and **FaceLandmarker** (478 face landmarks, 52 blendshape scores and head pose).
- **Puppet strings:** the robot hangs on glowing marionette strings from your fingertips, as in the reference interaction (`ww.mp4`). It turns with your hand, rises when you lift it and spins or walks as you move.
- **Face:** it looks at your face and reacts to your smile, sad face or surprise.
- **Temporal decisions:** gestures and motions pass temporal validation before they become validated commands.
- **The robot:** a procedural three.js robot carrying the Yemen flag on its belly. A translucent HUD shows the real system state: camera, model, gesture, robot state, measured FPS and latency, and an event log.

Everything runs locally. Camera frames are never uploaded and never recorded.

There are two interchangeable computer-vision engines, with exactly one camera owner at a time:

| Engine | Camera owner | Start |
|---|---|---|
| **browser** (default) | the page (`getUserMedia` + MediaPipe WASM) | `npm run dev` |
| **python** | the Python engine in `backend/aura_cv` (OpenCV + MediaPipe Tasks). It streams results and preview frames to the page over a localhost WebSocket. | `npm run dev:python`, then open `/?engine=python` |

Details: [docs/PYTHON_ENGINE.md](docs/PYTHON_ENGINE.md).

The active engine is shown in the HUD (*System → CV engine*) and in the start dialog, which also has an explicit **Switch to … engine** button (a full reload). If the selected engine fails, the app reports the failure. It never switches engines on its own.

---

## Features

- **Live camera pipeline**: `getUserMedia` with explicit error handling (permission denied, no camera, camera busy, unsupported settings, unplugged device). Only the newest frame is processed, so there is no backlog. Dropped frames are counted.
- **Deep-learning perception** (local inference, both engines):
  - **GestureRecognizer** gives 21 hand landmarks plus learned gesture classes; it is the same landmark model as before, with a classifier on top, so there is one hand pipeline.
  - **FaceLandmarker** gives face landmarks, blendshape scores and a head-pose matrix.
  - The GPU delegate is used where available, with automatic CPU fallback. Each model falls back gracefully if it is missing: rule classifier for hands, BlazeFace for gaze only.
- **Gesture decisions**:
  - the model's class, if its score is at least 0.6, decides open palm, fist (the safety lock), point, thumb up / down and V;
  - pinch comes from the pinch distance, which the model has no class for;
  - geometry may only *assist* (pointing sideways) or *veto* (a thumb class on a flat palm);
  - temporal validation: hold time, release filter, one onset per held gesture, motion gating.
  - Every decision reports its source (`model`, `rules (model: None)`, `pinch-distance`).
- **Motion gestures**: V to walk, three fingers to run (both steered by the hand's sideways offset); a "tornado" circle of the raised index finger to spin; a quick thumb-index tap to squat; pinch-and-hold to lift ("heavy lifting"); swipes. Finger-step (cadence) walking remains available as a configuration option.
- **Face understanding**:
  - gaze toward your face (always: also while you aim), with a sticky primary face, smoothing and neutral fallback;
  - head imitation: turn your head left / right or look up and the robot does the same (mirror image), on top of looking at you;
  - head pose; eye-gaze direction;
  - expressions (happy / sad / surprised / neutral) from blendshape scores, with smoothing and hysteresis. This is an estimate of visible facial movement, not a measure of emotion.
- **Interaction controller**:
  - a command registry and a documented robot state machine (`IDLE`, `TRACKING`, `CONTROL`, `EXECUTING`, `LOCKED`, `ERROR`)
  - every accepted or rejected command is logged with its reason; unknown commands are reported as `UNSUPPORTED`
- **3D robot**: a real three.js scene with a procedural robot.
  - PBR materials and environment lighting; idle bob, head and eye gaze, wave, spin with lean and vortex, stride-synced walk, heavy-lifting pose.
  - Emotions, plus reactions to your expression; aim beam; lock and error looks.
  - A status plate ("AGENT #: AURA-01 / STATUS: …") with glitch transitions, as in the reference video.
  - The **Yemen flag** (red, white, black at 2:3) is a 3D surface on its belly.
  - An optional GLB/glTF model is supported, with visible fallback.
- **Marionette strings**: five glowing string bundles from your fingertips to the robot's hands, head, chest and base. They never cross, sag under gravity, and fall and fade when the hand is lost.
- **Visual effects**: glowing landmarks, skeleton, fingertip trails sampled from real tracking results, palm arcs, and a pinch link. They fade out when tracking is lost.
- **HUD**: only measured or configured values: camera/processed/render FPS, inference time, frame→result and frame→overlay latency, dropped frames, session statistics, and asset/config status.
- **Session export**: downloads metrics and the event log as JSON. It contains no images and no landmarks.

## AI agent

The **Agent** button opens a panel where you type requests such as "wave", "spin left", "lock the gestures" or "what is the robot doing?".
- **Provider:** a Claude model (`claude-opus-5-5` by default), called server-side from the local Vite server. The API key lives in `.env.local` and never reaches the browser.
- **What it can do:** choose only allowlisted actions, which run through the same validated task path as gestures and keys. Gesture lock changes need your explicit confirmation in the panel.
- **What it reports:** an action is shown as succeeded only after it really completed. Unsupported requests are refused with the nearest available capability.
- **Diagnostics and workflows:** it can diagnose problems from live state ("why is nothing working?") and run two checked workflows: "fix the robot" (diagnose, unlock with your confirmation, reset, verify) and "make sure it waves" (check blockers, run, verify).
- **Without a key:** the panel shows "Unavailable" and the rest of the app works normally.

Setup: [docs/AGENT_SETUP.md](docs/AGENT_SETUP.md) · Design and guarantees: [docs/AGENT_INTERFACE.md](docs/AGENT_INTERFACE.md).

## Requirements

| | |
|---|---|
| Node.js | 20 or newer (tested with 24.20.0) |
| Python (optional, Python engine only) | 3.12–3.14 (tested with 3.14.7): mediapipe 1.1.1, opencv-contrib-python 5.0.0.93, numpy 2.5.4, websockets 17.3 |
| Browser | Chromium-based (Chrome, Edge) recommended; tested with Microsoft Edge (headless). Needs WebGL2 and `getUserMedia`. |
| Camera | Any webcam the browser can open. 720p at 30 fps is requested; the device may choose otherwise, and the HUD shows what it delivered. |
| Context | `http://localhost` / `127.0.0.1` or HTTPS (browsers only allow camera access in secure contexts) |
| Network | Only for `npm install` and the one-time model download (`npm run setup`) |

## Install and run

```bash
npm install
npm run setup     # copies the MediaPipe WASM runtime and downloads the models (all Apache-2.0, local inference):
                  #   hand_landmarker.task 7.8 MB, gesture_recognizer.task 8.4 MB, face_landmarker.task 3.8 MB,
                  #   blaze_face_short_range.tflite 0.2 MB; if a download fails it prints the curl command to run
npm run dev       # browser engine: http://127.0.0.1:5173
```

Python engine (Windows 11, Python 3.12–3.14; installs into `.venv`, never into the global Python):

```bash
npm run setup:python   # py -3 -m venv .venv + pinned backend/requirements-dev.txt
npm run dev:python     # engine on ws://127.0.0.1:8765 + Vite; open http://127.0.0.1:5173/?engine=python
```

Open the URL, click **Start camera** and allow camera access. `npm start` is an alias for `npm run dev`. For a static build: `npm run build && npm run preview`.

The dev server binds to `127.0.0.1` only, so the app (and the camera stream it processes) is not reachable from the network.

## Deep-learning models and their roles

| Model (file in `public/models/`) | Runs in | Role | If missing |
|---|---|---|---|
| GestureRecognizer (`gesture_recognizer.task`) | browser (WASM, GPU) / Python engine (CPU) | 21 hand landmarks + learned gesture classes (Open_Palm, Closed_Fist, Pointing_Up, Thumb_Up, Thumb_Down, Victory, ILoveYou, None) | HandLandmarker + rule classifier (labelled `rules`) |
| HandLandmarker (`hand_landmarker.task`) | same | fallback landmarks only | hand tracking unavailable (reported) |
| FaceLandmarker (`face_landmarker.task`) | browser / Python engine | face box, 478 landmarks, 52 blendshape scores (expressions, eye gaze), head-pose matrix | BlazeFace (gaze only, no expressions) |
| BlazeFace (`blaze_face_short_range.tflite`) | same | face detector fallback | neutral gaze |
| **AURA hand-pose network** (`aura_handpose.json`, trained in this project) | browser (JS) / Python engine (numpy) | 63 → 128 → 64 → 8 MLP on the hand landmarks: OPEN_PALM, FIST, POINT, TWO_FINGERS, **THREE_FINGERS**, THUMB_UP, THUMB_DOWN, OTHER. Adds the run gesture (the canned model has no three-finger class) and rejects poses the app does not use | canned model + rules, as before (reported in the HUD log / `hello`) |

The hand-pose network is trained here on the public HaGRIDv2 dataset (5,302 hand images, 95.9 % validation accuracy). The steps, dataset and licence are in [training/README.md](training/README.md), the results in [docs/VALIDATION.md §14](docs/VALIDATION.md). Rules are used only as documented post-processing (pinch distance, assisting POINT / V / three when both models are unsure, vetoing a thumb class that hand geometry contradicts) and as the labelled fallback. Faces, landmarks and dataset images never leave the machine.

## Gestures

| Gesture | How to perform it | Command | Robot behavior |
|---|---|---|---|
| **OPEN_PALM** | All four fingers straight, not pinching; hold still ~0.5 s | `WAVE` | Waves its right arm (1.8 s) |
| **FIST** | All four fingers curled, thumb not sticking out; hold still ~0.6 s | `TOGGLE_LOCK` | Locks/unlocks gesture commands (amber "sleep" look) |
| **PINCH** | Thumb tip touching index tip; hold, then move left/right or up | `GRAB_ROTATE` (hold) | Rotates with your hand while you pinch; raising the hand lifts it ("heavy lifting") |
| **PINCH_TAP** | Touch thumb and index tips and let go again within 0.6 s | `SQUAT` | **One squat**: sinks about 0.45 m with arms forward and stands up again (0.9 s). Tap again for another one |
| **POINT** | Index finger straight, other three curled; hold ~0.12 s | `AIM` (hold) | Aims a beam at your fingertip while you point (its eyes stay on you) |
| **SWIPE_LEFT** | Quick horizontal hand movement to the left (as seen on screen) | `SPIN_LEFT` | Turns a full circle to the left |
| **SWIPE_RIGHT** | Same, to the right | `SPIN_RIGHT` | Turns a full circle to the right |
| **TWO_FINGERS** (V) | Index and middle finger up, ring + pinky + thumb folded; ~0.15 s | `WALK` (hold) | **Walks forward** at 0.7 m/s (reached in about 0.1 s) while you hold the V; move the hand left / right to steer |
| **THREE_FINGERS** | Index, middle **and ring** finger up, pinky + thumb folded; ~0.15 s | `RUN` (hold) | **Runs** at 1.6 m/s (bigger stride, arms pumping). Switching between V and three fingers changes speed without stopping |
| **Spin** (`SPIN`) | Index finger straight up, **circle it like a tornado** | `SPIN` (hold) | **Rotates continuously at a rate that follows your finger** (starts about 0.26 s after you begin circling; up to 1080°/s = 3 turns/s). Clockwise on screen → right, counter-clockwise → left. Stop circling → it slows and stops |
| **THUMB_UP** | Thumb clearly extended and pointing up, other fingers folded; hold still ~0.5 s | `LAUGH` | Laughs: happy "^ ^" eyes, a smile arc, bouncing body, hands on the belly (2.2 s) |
| **THUMB_DOWN** | Same, thumb pointing down | `SAD` | Sad: head hangs, eyes droop and dim blue, a frown arc, slumped body, slow sway (2.6 s) |
| **Finger walking** (`FINGER_WALK`) | Step index and middle finger alternately like two legs | none by default | Detected; bind it to `WALK` (and set `TWO_FINGERS` / `THREE_FINGERS` to `null`) for cadence walking: the speed then follows your step rate |

Rules of thumb:

- Each static gesture triggers its command **once** per hold. Release the gesture (or drop your hand) to trigger it again.
- `WAVE` and `TOGGLE_LOCK` only start while the hand is still, so waving your hand around does not trigger them.
- While **LOCKED**, every command except `TOGGLE_LOCK` is rejected. A FIST or the `L` key unlocks.
- While an action animates (`EXECUTING`) or a hold is active (`CONTROL`), other actions are rejected as `BUSY` / `CONFLICT`.
- **Walking / running** lasts while you hold the V / three fingers. Lowering the fingers stops it; switching V ↔ three fingers within 0.6 s keeps the current speed (it accelerates or slows, it does not stop). Hand loss, camera stop and `L` stop it at once and never carry speed over. It stays inside a 2 m area; `R` brings it home.
- **Squat**: only a *quick* pinch that is released squats; a pinch held longer than 0.6 s is the lift / turn control, and a pinch that ends because the hand left the camera does nothing.
- **Spinning** needs a sustained rotational trajectory (radius, consistency, at least 100° within 0.7 s, held for 0.1 s). A still pointing finger, a line, a short arc, jitter or tracking jumps never rotate the robot. After you stop circling the robot slows down smoothly; hand loss, low confidence, camera failure and `L` stop it at once.
- A spin ends the `AIM` hold that pointing starts. Pinch-drag (`GRAB_ROTATE`) is unchanged and is never interrupted by a spin. Walking and spinning never run together (walking has priority in a tie).
- **Emotions** are short local animations: they never move the robot, are rejected while it moves or is locked, and `L` cancels them. FIST keeps its meaning (lock / unlock): the emotions need an *extended* thumb, the fist a folded one.
- **Gaze**: the robot looks at your face (local face model, about 10 detections/s on the same camera frames), also while it aims at your finger (`interaction.attention.handAttentionHolds: ["AIM"]` restores looking at the hand). If your face is lost it keeps looking for 0.7 s, then returns to a neutral, idle gaze. With several faces it sticks to one. Face loss never stops hand control.
- **Face mirroring** (visual only, never a command): smile → the robot smiles ("^ ^" eyes, smile arc); sad face → frown, drooping blue eyes, tilted head; surprise → wide eyes. Turn your head left / right or look up / down → its head does the same as a mirror image, up to 40°. Locked or in error it does not react.
- Walking, running, squats, spinning, emotions and gaze are fully offline: they need no API key, no network and no AI agent. The agent cannot walk, run, squat, spin or trigger emotions (its command list is unchanged).
- The exact rules and thresholds are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#gesture-definitions).

Keyboard (it uses the same validation path as gestures): `L` lock/unlock (also stops walking / running / a squat), `W` wave, `R` reset orientation and position, `H` hide/show HUD, `E` export session, `Esc` stop camera.

## HUD metrics

| Field | Definition |
|---|---|
| Camera FPS | Frames presented by the `<video>` element per second (from `requestVideoFrameCallback.presentedFrames`) |
| Processed FPS | Frames that went through the full pipeline per second |
| Render FPS | `requestAnimationFrame` callbacks per second (overlay and 3D) |
| Inference | Hand-model call duration, mean and p95 over the last 120 frames (the first call, the warm-up, is excluded) |
| Frame → result | From the frame's capture time (or presentation time if the browser does not report capture time) to the end of gesture/command processing |
| Frame → overlay | From the same frame timestamp to the moment its overlay was drawn |
| Dropped frames | Frames the video element presented that were never handed to the pipeline (typically because the previous frame was still being processed) |
| Score | The winning rule's membership score in [0, 1]. **Not a calibrated probability.** |

Camera sensor and USB latency happen before the browser sees a frame and cannot be measured from inside the page.

## Configuration

Copy `public/aura.config.example.json` to `public/aura.config.json` and edit it. Every key is optional; defaults and units are documented in [src/config/defaultConfig.js](src/config/defaultConfig.js). Each value is validated: an invalid value is rejected individually, the default is kept, and the error appears in the HUD event log. A binding to a command that does not exist is reported as unsupported.

URL overrides are also accepted: `?engine=python`, `?engineUrl=ws://…`, `?delegate=CPU` and `?numHands=2`. The `backend` section of the file is read only by the Python engine.

Optional 3D model: set `"scene": { "robotModelUrl": "/models/robot.glb" }`. The model is scaled to the robot's height. Animation clips whose names contain `idle`, `wave` or `spin` are played for the matching states. If loading fails, the procedural robot stays and the reason is shown.

## Project layout

```
index.html                 page shell (HUD markup, layers)
src/
  main.js                  entry: config loading, UI wiring, keyboard
  app/ApplicationCoordinator.js   init order, render loop, HUD view model, shutdown
  app/FramePipeline.js     per-frame: track -> features -> gestures -> commands
  camera/CameraManager.js  camera lifecycle, frame loop, error mapping
  tracking/HandTrackingEngine.js  backend-agnostic tracker wrapper, result validation
  tracking/mediapipeBackend.js    MediaPipe HandLandmarker backend
  gestures/                landmarks, FeatureExtractionEngine, StaticGestureClassifier,
                           SwipeDetector, GestureStabilizer, GestureRecognitionEngine
  interaction/             commands registry, RobotStateMachine, InteractionController
  scene/Scene3D.js         three.js scene + procedural robot; robotAsset.js (GLB fallback)
  fx/                      VisualEffectsEngine (Canvas2D), screenMapping (object-fit cover)
  hud/HUDController.js     DOM HUD (textContent only)
  core/                    EventBus, StateMachine, metrics (RateMeter, RollingStats)
  config/                  defaults + validation
  remote/                  PythonEngineClient (protocol v2), RemoteCamera/Tracker/FramePipeline adapters
backend/aura_cv/           Python engine: camera, tracking, features, gestures, pipeline, protocol, server
backend/tests/             pytest suite (no camera needed)
scripts/setup-assets.mjs   copies WASM, downloads the model
scripts/py.mjs, dev-python.mjs   venv runner; engine + Vite launcher
scripts/integration-check.mjs  browser check: headless with a fake camera; visible window for the physical camera
test/                      node:test unit tests + synthetic hand fixtures
  agent/                   TaskBroker, tasks (robot.command, app.status), tool catalog, AgentOrchestrator/Client/Panel
server/agent/             agent server (inside the Vite server): config, Claude adapter, conversations, HTTP API
docs/                      ARCHITECTURE.md, PYTHON_ENGINE.md, AGENT_INTERFACE.md, AGENT_SETUP.md, VALIDATION.md, ROADMAP.md
```

## Tests

```bash
npm test                                  # 199 JS tests (incl. agent and workflow tests with a MOCKED provider), no camera or browser needed
npm run test:py                           # 106 Python tests (engine, camera mocks, server, JS/Python contract + parity)
npm run test:all                          # both
node scripts/integration-check.mjs        # headless Edge/Chrome, fake camera (test pattern)
node scripts/integration-check.mjs --video hand.jpg --keys w,l   # feed an image as the camera
node scripts/integration-check.mjs --engine python --python-input hand.jpg   # full stack with the Python engine (TEST-FILE input)
npm run bench:python                      # physical-camera benchmark of the Python engine (no browser, nothing stored)
node scripts/integration-check.mjs --agent-mock   # agent panel in the real UI against a MOCK provider (not AI)
node scripts/integration-check.mjs --engine python --real-camera --seconds 30   # PHYSICAL webcam, Python engine: VISIBLE window
node scripts/integration-check.mjs --browser-real-camera --seconds 30           # PHYSICAL webcam, browser engine: VISIBLE window
node scripts/integration-check.mjs --engine python --real-camera --guided embodied    # guided on-screen gaze/spin/walk/emotion session (visible)
node scripts/integration-check.mjs --video hand.mjpeg --cpu-profile             # where the main thread spends its time
npm run agent:smoke                       # LIVE: one real request to the configured model (needs ANTHROPIC_API_KEY)
```

The step-by-step manual hardware checklist is in [docs/VALIDATION.md](docs/VALIDATION.md#7-manual-hardware-validation-checklist).

By default the integration check never opens your real webcam: it runs headless with Chromium's fake capture device (or a test file for the Python engine) and writes `report.json` and screenshots to `integration-output/`.

With `--real-camera` (Python engine) or `--browser-real-camera` (browser engine) it opens the **physical** webcam. Such a run:
- **always uses a visible Edge/Chrome window** in front. There is no headless option for it, so the camera is never used in the background;
- shows the real app with its live feed;
- shows a red **CAMERA ACTIVE** banner with the engine, camera, hand, gesture, robot state and time left;
- has a **Stop test** button that stops the camera and ends the run. Closing the window also ends it;
- saves no screenshots or frames, only `report.json`.

Run it only with the consent of whoever is in view. Recorded results are in [docs/VALIDATION.md](docs/VALIDATION.md).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "Camera API unavailable" | The page is not on a secure origin. Use `http://127.0.0.1:5173` or HTTPS. |
| "Camera permission was denied" | Allow the camera for the site (address-bar camera icon), then click **Retry**. |
| "The camera is in use…" | Close other apps using the webcam (video calls, camera app). |
| "No camera device was found" | Connect a camera, or check OS privacy settings (Windows: Settings → Privacy → Camera). |
| "hand model not found … run npm run setup" | The model file is missing from `public/models/`. Run `npm run setup`. |
| Model shows `CPU (GPU failed)` | The WebGL GPU delegate could not initialize; the CPU fallback is slower but functional. |
| Gestures flicker or never trigger | Keep the whole hand in the image (the HUD shows "hand partly outside the image"), face the palm toward the camera, and use even lighting. The HUD's `raw` label and note show why a frame was rejected. |
| Low FPS | Keep `numHands: 1`, lower `camera.width/height`, close other GPU-heavy tabs, or use the Python engine (measured about 3× faster inference on this machine). |
| "Python engine not reachable" | Start it with `npm run backend` (or `npm run dev:python`); see [docs/PYTHON_ENGINE.md](docs/PYTHON_ENGINE.md#troubleshooting). |
| Robot missing, HUD says WebGL unavailable | Enable hardware acceleration in the browser settings. |

## Privacy

- Frames are processed locally: in the page (browser engine), or by the local Python process (Python engine), which streams them only to `127.0.0.1` and only to a page from an allowed localhost origin. Nothing is uploaded, stored or recorded.
- Without the AI agent, no network requests are made at runtime except loading the app's own files from the local dev server.
- When you use the **AI agent**, your typed messages and the agent's read-only state reports go from the local server to the Anthropic API. The reports cover the robot state, the camera and tracking *status* and recent issue texts. **Camera frames and landmarks are never sent.** Without `ANTHROPIC_API_KEY`, nothing is sent.
- Face detection (gaze) runs locally on the frames the active engine already owns; face boxes go only to the local page, like hand landmarks, and are never sent to the AI agent or any other service.
- The audio track is never requested.
- The session export contains counters, timings and event texts only: no images, no landmarks.
- `window.__aura.snapshot()` is a read-only debugging hook in the page itself; it is not a network endpoint.

## Known limitations

See [docs/VALIDATION.md](docs/VALIDATION.md#known-limitations) for the full list. In short:
- The gesture rules are a hand-tuned baseline checked on synthetic poses and four real photos. That is not an accuracy evaluation.
- With a physical webcam, OPEN_PALM, FIST, PINCH and POINT onsets drove their commands in both engines (VALIDATION.md §10). Whether each detection matched the gesture actually performed was **not** evaluated, and swipes have not been exercised on a real camera.
- In browser mode, inference runs on the main thread. Profiling shows the time is spent waiting for the GPU inside MediaPipe's GPU delegate (synchronous `readPixels`), so a busy GPU lowers render FPS too. Python mode avoids this; a Web Worker backend is the roadmap fix.
- In Python mode, camera selection happens in the engine (`--camera N`), not in the page.
- Handedness labels are passed through from MediaPipe unchanged.

## License notes

The hand landmarker model is © Google, Apache-2.0 (downloaded by `npm run setup`, not committed). The AURA hand-pose weights are trained on HaGRIDv2, whose licence is a variant of CC BY-SA 4.0; check it before redistributing `public/models/aura_handpose.json`. The dataset images are not part of the project. three.js is MIT; `@mediapipe/tasks-vision` is Apache-2.0. The robot is built procedurally in code; no third-party 3D assets are included.
