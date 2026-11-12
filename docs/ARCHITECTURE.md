# Architecture

## Architectural invariants

1. **One camera owner per active engine.**
   - Browser mode: only `CameraManager` (`getUserMedia`).
   - Python mode: only the engine's single `cv2.VideoCapture`. The page never calls `getUserMedia`; the integration check counts the calls and requires 0.
   - Switching engines is an explicit, user-initiated full page reload (the "Switch to … engine" button or `?engine=`). When the selected engine fails, the app reports the failure. It never switches engines on its own.
2. **One gesture-event contract.** Both engines emit canonical `{type: start|end|event, gesture}` events (`src/interaction/gestureEvents.js`). `InteractionController.handleGestureEvents` validates every event before it can change state. A cross-language test feeds identical landmark sequences to both engines and requires identical events.
3. **Classification is separate from execution.** Engines only classify. The controller alone decides whether a command runs, and it owns the resulting application state: the state machine, `pose.yaw` and `lastResult`. The scene and HUD render that state; they never invent it.
4. **Delivery semantics (Python mode).**
   - *State and events*: every result is applied immediately and in order, and none is skipped. If the browser falls behind, the engine merges an unsent result's events into the next one.
   - *Visuals*: a preview may be dropped. The overlay only shows landmarks computed from the frame on screen.
5. **No stale data.** Results from an earlier camera session, or with a frameId that is not newer than the last one, are rejected. Landmarks are never shown when the newest result has no hand.
6. **Tasks are not gestures.** Agent and user tasks go through `TaskBroker` ([AGENT_INTERFACE.md](AGENT_INTERFACE.md)); a gesture cannot create a task.
7. **The AI agent only proposes.**
   - Model output is untrusted. Tool calls are allowlist- and schema-validated on the server and again in the browser, executed only through `TaskBroker` (origin `agent`), and reported from the task record.
   - The provider key exists only in the local server process (`server/agent/`, mounted in the Vite server).
   - The agent never touches the camera, the vision engine or gesture settings.
   - Multi-step work runs only as fixed workflows (`src/agent/workflows.js`). Each step goes through the broker, is dispatched at most once, and stops safely on failure; an unknown outcome is never retried. Blockers are checked again after any wait, so a hold or error that appears meanwhile stops the workflow before dispatch.

## Engine modes

Two interchangeable CV engines, with exactly one camera owner per mode (`engine.mode`):

- `browser`: the flow below. Camera, model and gesture engine all run in the page.
- `python`: `backend/aura_cv` owns the camera (OpenCV), runs MediaPipe Tasks, features, gestures and stabilization, and streams results plus frame-matched preview JPEGs over `ws://127.0.0.1:8765`.
  - In the browser, `RemoteCamera`, `RemoteTracker` and `RemoteFramePipeline` (`src/remote/`) implement the same interfaces as `CameraManager`, `HandTrackingEngine` and `FramePipeline`.
  - Everything from the `InteractionController` onward is shared and unchanged.
  - See [PYTHON_ENGINE.md](PYTHON_ENGINE.md).

## Components and data flow (browser engine)

```
                   requestVideoFrameCallback (newest frame only)
 CameraManager ─────────────────────────────────────────────┐
   getUserMedia, error mapping, track "ended" detection     │
                                                            ▼
                                                     FramePipeline.process(frame)
                                                            │
   HandTrackingEngine.detect ── backend (MediaPipe WASM) ───┤  landmarks, world landmarks,
     validation, primary-hand selection                     │  handedness score, inferenceMs
                                                            ▼
   FeatureExtractionEngine.extract (every hand) ────────────┤  normalised features
                                                            ▼
   GestureRecognitionEngine.update (primary hand) ──────────┤  raw label, stable label,
     StaticGestureClassifier → motion gate → GestureStabilizer│  start/end/event
     SwipeDetector                                          ▼
   InteractionController ───────────────────────────────────┤  validated commands,
     command registry, RobotStateMachine                    │  state transitions → EventBus
                                                            ▼
                                                     pipeline.latest
                                                            │
   requestAnimationFrame (render loop, independent) ◄───────┘
     VisualEffectsEngine.draw   (Canvas2D overlay)
     Scene3D.update + render    (three.js / WebGL)
     HUDController.update       (DOM, throttled to 10 Hz)
```

`EventBus` carries camera, tracker, tracking, gesture, command and state events to the HUD event log. It is also the hook for a future session recorder or agent bridge.

| Component | File | Knows about |
|---|---|---|
| CameraManager | `src/camera/CameraManager.js` | MediaStream, `<video>` |
| HandTrackingEngine | `src/tracking/HandTrackingEngine.js` | backend interface only |
| MediaPipe backend | `src/tracking/mediapipeBackend.js` | `@mediapipe/tasks-vision` |
| FeatureExtractionEngine | `src/gestures/FeatureExtractionEngine.js` | landmark geometry |
| GestureRecognitionEngine | `src/gestures/*.js` | features, time |
| InteractionController | `src/interaction/*.js` | gesture events, commands |
| Scene3D | `src/scene/Scene3D.js` | controller snapshot, hand position |
| VisualEffectsEngine | `src/fx/VisualEffectsEngine.js` | tracking result, screen mapping |
| HUDController | `src/hud/HUDController.js` | a view model of plain values |
| ApplicationCoordinator | `src/app/ApplicationCoordinator.js` | everything (wiring only) |

Everything from `HandTrackingEngine` to `InteractionController`, plus config, metrics and `FramePipeline`, is DOM-free and unit-tested in Node.

## Frame scheduling

- **No queue.** `requestVideoFrameCallback` fires for the most recently presented video frame. Inference is synchronous inside that callback, so frames presented during inference are simply never delivered. `CameraManager.stats.dropped` counts them from the gaps in `presentedFrames`. A slow model lowers the processed FPS; it never adds queueing latency.
- **Fallback.** Without `requestVideoFrameCallback` (older Firefox), a `requestAnimationFrame` loop polls `video.currentTime` and processes only when it has changed.
- **Rendering is decoupled.** The render loop draws at display rate from `pipeline.latest`. Robot animations are time-based and continue smoothly between results. Trails are sampled once per processed result, never interpolated or invented.
- **Timestamps.** The model receives the callback time, which is strictly increasing (enforced by `HandTrackingEngine`). Latency uses `captureTime`, else `presentationTime`, else the callback time (`metrics.frameTimeSource` records which).
- **Main-thread cost.** Inference blocks the main thread. Under GPU load, render FPS drops to roughly the processed FPS while a hand is visible.
  - Profiled in Phase 5: most of the blocked time is MediaPipe's GPU delegate waiting synchronously for the GPU (`readPixels` / `useProgram`), not JS or WASM compute.
  - A Web Worker backend is the fix (roadmap item 2). See VALIDATION.md §10.

## Robot state machine

| State | Meaning | Allowed next states |
|---|---|---|
| IDLE | no hand tracked | TRACKING, EXECUTING, LOCKED, ERROR |
| TRACKING | hand tracked; robot looks at it | IDLE, CONTROL, EXECUTING, LOCKED, ERROR |
| CONTROL | a hold command (GRAB_ROTATE, AIM, WALK, SPIN) is active | TRACKING, IDLE, ERROR |
| EXECUTING | a timed action (WAVE, SPIN_*, RESET_POSE) animates | TRACKING, IDLE, ERROR |
| LOCKED | only TOGGLE_LOCK accepted | IDLE, TRACKING, ERROR |
| ERROR | camera interrupted, camera failed to start, or model failed / 30 consecutive inference failures | IDLE (after a successful restart or inference recovery) |

### Transitions

| Trigger | Requires | Result | Visible response | Reset / recovery |
|---|---|---|---|---|
| valid hand appears | IDLE | TRACKING | eyes brighten (cyan); head follows the palm; forearms raise | hand lost → IDLE |
| hand lost (incl. TRACKING_LOST, PROCESSING_ERROR) | TRACKING / CONTROL | IDLE | eyes dim, idle head scan; holds end | hand returns → TRACKING |
| `start OPEN_PALM` → WAVE | IDLE/TRACKING, not locked/busy/conflict, cooldown over | EXECUTING | right arm raised and waving; eyes white, chest pulses | after 1.8 s → TRACKING/IDLE (`completed`) |
| `event SWIPE_LEFT/RIGHT` → SPIN_* | same | EXECUTING | full turn left/right (animation offset; `pose.yaw` unchanged) | after 1.1 s → resting state |
| key `R` / task → RESET_POSE | same | EXECUTING | robot turns and glides back; **`pose.yaw`, `pose.x`, `pose.z` animate to 0** | after 0.7 s, home pose → resting state |
| `start FINGER_WALK` → WALK | TRACKING (hand present), not locked/busy | CONTROL | gait (stride bob, lean, arm swing) whose phase advances with the distance walked; **`pose.x/z` advance at `walkSpeed`**, which follows `walkSpeedFor(cadence)` (dead zone below `minCadence`, rate-limited by `accel` / `decel`), clamped to `walk.maxDistance` | `end FINGER_WALK` (no step for `stopMs`), hand lost, camera stop, error, or the user's lock → resting state, speed 0 at once |
| `start SPIN` → SPIN | not locked/busy; ends an active **AIM** hold first (`supersedes`) | CONTROL | **`pose.yaw` integrates `spinRate`**, which follows `spinRateFor(finger ω)` (gain, dead zone, cap, accel / decel limits, trapezoid integration); a vortex of rings fades in with the rate | `end SPIN` → decelerates smoothly (`decelDegS2`); hand lost, lock, error, reset → rate 0 at once |
| `start THUMB_UP` / `THUMB_DOWN` → LAUGH / SAD | IDLE/TRACKING, not locked/busy/moving | EXECUTING | laugh / sad animation (no movement) | after `emotions.laughMs` / `sadMs` → resting state; the user's lock cancels it |
| `start PINCH` → GRAB_ROTATE | TRACKING (hand present), not locked/busy | CONTROL | orange eyes/ring; arms forward; **`pose.yaw` follows horizontal palm travel** | `end PINCH`, hand lost or error → TRACKING/IDLE; yaw keeps its new value |
| `start POINT` → AIM | same | CONTROL | head and right arm aim at the index tip; orange beam | `end POINT`, hand lost or error → resting state |
| `start FIST` / key `L` / task → TOGGLE_LOCK | IDLE/TRACKING | LOCKED | amber narrowed eyes, head down, arms lowered; all other commands rejected `LOCKED` | TOGGLE_LOCK again (after the cooldown) → TRACKING/IDLE |
| camera/engine/model failure | any | ERROR | red flickering eyes; holds and actions cancelled | successful camera restart / inference recovery → IDLE |

Spin and wave are animations by definition: their *result* is the completed animation, reported as `completed`. GRAB_ROTATE, SPIN and RESET_POSE change `pose.yaw`; WALK and RESET_POSE change `pose.x/z`. Motion is integrated in `InteractionController.tick(t)` from the tick clock (dt skipped when the clock does not advance, capped at 100 ms, trapezoid rule), so it is frame-rate independent in both engine modes. The continuous control signals (finger ω, step cadence) arrive per frame in `gesture.motion` and are passed to `updateHold(features, motion)`; they are targets, never applied directly. `snapshot().movement` is `WALKING` (speed, cadence, gait phase), `ROTATING` (direction, rate), `LAUGHING`, `SAD`, `RETURNING` or `IDLE`.

Priority and safety for the motion gestures:
- The gesture engine evaluates walking first; while one motion gesture is active the other cannot start (a tie in one frame goes to walking). The controller also rejects a second hold (`CONFLICT`), and starting a walk zeroes any spin coast.
- SPIN supersedes only AIM, because both come from the pointing shape. GRAB_ROTATE keeps its semantics.
- `TOGGLE_LOCK` from the user (keyboard / gesture) stops WALK / SPIN, or cancels LAUGH / SAD, then locks and zeroes all motion. A lock request from the agent is still `CONFLICT` / `BUSY`: the agent never overrides a held gesture. Unlocking never resumes motion.
- WALK, SPIN (holds) and LAUGH, SAD (`gestureOnly`) are not in the agent's command list.

### Perception → decision → animation (deep-learning upgrade)

Each layer only reads the one before it.

1. **Perception**: model inference on the active engine's own frames.
   - `GestureRecognizer`: hand landmarks plus learned gesture scores per hand (`hand.model`).
   - `FaceLandmarker`, throttled to `face.intervalMs`: face box, blendshape subset (`bs`), pose matrix.
   - Browser engine: `mediapipeBackend.js`. Python engine: `tracking.py` / `faces.py`, which forward the same outputs in `result.hands[].model` and `result.faces`.
2. **Decision**, deterministic:
   - `StaticGestureClassifier._fuse` is the model / rule fusion with the geometric veto; it is identical in `gestures.py`.
   - The stabilizer and the spin / walk detectors provide temporal validation.
   - `InteractionController` turns these into commands and motion: lift, steering, spin rate, walk speed.
   - `AttentionController` decides the gaze target.
   - `FaceExpressionEngine` (`src/perception/faceAnalysis.js`) produces expression, head pose and eye gaze. This logic is shared by both engines because the Python engine only forwards model outputs.
   - `_robotStatus` sets the plate text.
3. **Animation**: `Scene3D` renders the controller's pose and smooths everything time-based.
   - head and eye gaze; puppet coupling (visual offset only); emotions and expression reactions; lift; gait; spin lean and vortex; status plate; Yemen flag on the torso.
   - `VisualEffectsEngine` + `marionette.js` draw the strings.

   Animation never issues commands, and its visual offsets never feed back into the controller.

### Gaze (AttentionController)

- **Faces:** MediaPipe FaceDetector (BlazeFace short range) on the frames the active engine already owns.
  - Browser engine: `FaceTrackingEngine` on the same `<video>`, after hand inference.
  - Python engine: `aura_cv.faces.FaceTracker` in `FrameProcessor`; boxes are added to `result.faces`.
  - Both are throttled to `face.intervalMs` (120 ms). A missing model disables gaze only.
- **Policy:** the coordinator maps faces to screen-normalised coordinates, and `AttentionController` picks the target deterministically:
  - **HAND** while a hold in `attention.handAttentionHolds` (AIM) is active;
  - **USER** while the primary face was seen within `faceHoldMs`;
  - **NEUTRAL** otherwise.
- **Primary face:** sticky, re-matched within `maxJump`; switched only after `switchAfterMs` missing.
- **Angles:** `clamp((face − head) · gain)`, bounded by `maxYawDeg` / `maxPitchUp/DownDeg`. They are smoothed with a time constant and an angular-speed cap, and non-finite input is ignored.
- **Scene:** only the **head pivot** (and the eyes) follow the gaze, compensating the body's own yaw except while spinning. The locomotion root never turns to look.

What drives the transitions:
- Hand presence: `setHandPresent`.
- Accepted commands: `submit`.
- Continuous hold input: `updateHold`.
- Action completion: `tick`, after the command's `durationMs`.
- Failures: `fail` / `recover`.

Invalid transitions are refused by `StateMachine`. Hand presence never clears ERROR or LOCKED. A hold start that arrives when the hand is already gone is rejected as `NO_HAND`.

### Command validation (first failure wins)

`INVALID` (malformed request) → `UNSUPPORTED` (not in registry) → `ERROR_STATE` → `LOCKED` → `BUSY` (action running) → `CONFLICT` (hold active) → `NO_HAND` (hold without a hand) → `COOLDOWN` (same command within `interaction.cooldownMs`).

Commands only change robot animation or pose. None of them performs an external task, and the HUD never claims otherwise. A task's `succeeded` status is reported only after the controller's `completed` event ([AGENT_INTERFACE.md](AGENT_INTERFACE.md)).

## Gesture definitions

### Features (`FeatureExtractionEngine`)

The features are computed from MediaPipe *world* landmarks (metric, hand-centred) when present. Otherwise they use aspect-corrected image landmarks.

- `palmSize`: the mean of the wrist→index-MCP, wrist→middle-MCP and wrist→pinky-MCP distances. All shape distances are divided by it.
- `straightness(f)`: |base→tip| / (sum of segment lengths) over the finger's 4-joint chain. It is invariant to translation, scale and in-plane rotation.
- `extension(f)`: `ramp(straightness, 0.55 → 0.92)`.
- Thumb: additionally `min(…, ramp(abduction, 0.45 → 0.8))`, where abduction is the thumb-tip→index-MCP distance divided by palmSize.
- `pinchDistance`: thumb tip→index tip, divided by palmSize.
- Motion uses the palm centre in display space (mirrored if the view is mirrored), in units of image height, divided by the apparent palm size.

`ramp(v, a → b)` is linear: 0 at a, 1 at b, clamped. Write `e_x` for the extension of finger x and `pc = ramp(pinchDistance, 0.38 → 0.22)`.

### Static rules (`StaticGestureClassifier`)

| Gesture | Score |
|---|---|
| OPEN_PALM | `ramp(min(e_index, e_middle, e_ring, e_pinky), 0.6 → 0.9) · (1 − pc)` |
| FIST | `min(1−e_index, 1−e_middle, 1−e_ring, 1−e_pinky) · (1 − ramp(e_thumb, 0.7 → 0.95))` |
| POINT | `min(e_index, 1−e_middle, 1−e_ring, 1−e_pinky) · (1 − pc)` |
| PINCH | `pc · ramp(e_index, 0.1 → 0.35)` |
| TWO_FINGERS | `ramp(min(e_index, e_middle), 0.6 → 0.9) · ramp(max(e_ring, e_pinky), 0.9 → 0.7) · (1 − pc) · (1 − ramp(e_thumb, 0.75 → 1.0))` (`WALK`) |
| THREE_FINGERS | `ramp(min(e_index, e_middle, e_ring), 0.6 → 0.9) · ramp(e_pinky, 0.9 → 0.7) · (1 − pc) · (1 − ramp(e_thumb, 0.75 → 1.0))` (`RUN`) |
| THUMB_UP / THUMB_DOWN | `ramp(max(e_index…e_pinky), 0.95 → 0.75) · ramp(e_thumb, 0.7 → 0.95) · ramp(∓thumbDir.y, 0.5 → 0.8) · (1 − pc)`; thumbDir = thumb MCP → tip in display space |

TWO_FINGERS is decided from joint geometry (finger straightness), not from fingertip spread.
- Index and middle must be *clearly* straight, because the gesture moves the robot.
- Ring and pinky must be "not clearly straight". A frontal V sign foreshortens the folded fingers: the real `victory.jpg` measured ring 0.43 and pinky 0.66, so `1 − e` would reject it.
- A straight third finger (three fingers, open palm) or an extended thumb gives 0.

Decision:
- **NONE**: no valid hand.
- **UNKNOWN** when:
  - the handedness score is below 0.6 (low tracking confidence);
  - fewer than 90 % of landmarks are inside the image;
  - the best score is below 0.6;
  - or the best score beats the runner-up by less than 0.2 (ambiguous).
- Otherwise the best gesture wins.

### Temporal logic

- **Motion gate.** Gestures with `requireStill` (OPEN_PALM, FIST) are relabelled `MOVING` when the palm moves faster than 1.5 palm lengths/s, measured over about 150 ms.
- **Stabilizer.**
  - A label becomes stable after being observed continuously for its `holdMs`: OPEN_PALM 500, FIST 600, PINCH 150, POINT 250, TWO_FINGERS 300, THUMB_UP / THUMB_DOWN 500 (and still).
  - A stable label is released after being absent for 180 ms. Tracking loss releases it immediately.
  - One `start` and one `end` event are emitted per hold. Commands bind to `start`.
- **Swipes.**
  - A swipe fires when the palm travels at least 2.2 palm lengths horizontally within 450 ms, at an average of at least 4 palm lengths/s, with |dy| ≤ 0.6·|dx|.
  - After a swipe, the history is cleared and there is a 600 ms cooldown. A swipe in the opposite direction is suppressed for 1100 ms (the return stroke). A tracking gap over 150 ms resets the history.
  - Swipes are not detected while pinching, pointing, holding the V, walking or spinning, so drag, circle and step motions never fire a swipe.
  - Static onsets are suppressed for 700 ms after a swipe.

- **Spin** (`SpinDetector`, ported 1:1 to `aura_cv.gestures.SpinDetector`): index fingertip while the raw shape is POINT, in display space (y down: + = clockwise on screen), radii in palm lengths.
  - **Signature**, checked every frame over the last 800 ms (at least 250 ms, 6 points) around its centroid:
    - mean radius ≥ 0.25;
    - radius std/mean ≤ 0.5;
    - no step > 150°;
    - ≥ 75 % of steps turning the same way;
    - |swept angle| ≥ 120°.
  - **Rate:** the swept angle of the newest 300 ms divided by its duration.
  - **Hysteresis:**
    - starts after 150 ms of signature;
    - ends after 200 ms without it, 150 ms without the pointing shape, or at once on hand loss.
- **Finger walk** (`WalkCadenceDetector`, ported 1:1): `features.legs` = index / middle length projected on the palm axis (palm lengths).
  - Over 2 s, both fingers must:
    - move ≥ 0.12 and reach ≥ 0.6 (nearly straight) at some point;
    - be anti-correlated (≤ −0.3).
  - **Steps** are crossings of the difference through ±0.04 around its window mean, while the difference itself moved within the last 200 ms.
  - **Cadence** = steps/s over the last steps, lowered to 1/(time since the last step) when slowing.
  - **Starts** after 3 steps within the window. **Ends** after 1.1 s without a step, 200 ms without the legs pose, when the motion stops qualifying, or at once on hand loss.

All thresholds are in `src/config/defaultConfig.js` (`gestures.spin`, `gestures.walk`, `gestures.definitions`, `interaction.walk`, `interaction.spin`, `interaction.emotions`, `interaction.attention`, `face`). The Python engine uses identical defaults; a test enforces this. The calibration rationale is in [VALIDATION.md](VALIDATION.md).

### Known ambiguities

- **Side-on hands.** Curled fingers viewed side-on measured straightness 0.67–0.85, against 0.42–0.55 when facing the camera. FIST is therefore unreliable for side-on hands and falls to UNKNOWN; it does not fire falsely.
- **Half-curled hands.** A "claw" is UNKNOWN by design.
- **Shapes outside the vocabulary.** Victory, thumbs-up and similar shapes are UNKNOWN.

### Hand-pose network and fusion

`poseVector` (JS `src/perception/handPoseNet.js`, Python `backend/aura_cv/posenet.py`): the 21 display-space landmarks relative to the wrist, divided by the mean wrist → MCP 5 / 9 / 17 distance (63 values). The trained MLP (`public/models/aura_handpose.json`) returns a softmax over 8 classes; the feature extractor stores it as `features.net`.

`StaticGestureClassifier._fuse(model, rules, allowed, net)`, in order:
1. **PINCH** from the pinch distance (no learned model has a pinch class).
2. **THREE_FINGERS** when the canned model is weak or says Victory / Open_Palm / None: the net at `≥ gestures.net.minScore` (0.7), or without a net the rule at ≥ 0.8.
3. The canned model's class at `≥ model.minScore`, with the thumb-agreement veto. **FIST (the lock) only ever comes from the canned model.**
4. If the canned model is weak: the net for POINT / TWO_FINGERS / THREE_FINGERS / OPEN_PALM; then the rules for POINT / V / three, **only if the net is absent or unsure** (a confident OTHER rejects the pose).
5. Otherwise UNKNOWN. `source` says which path decided (`pinch-distance`, `model`, `aura-net (model: …)`, `rules (model: …)`).

### PINCH_TAP (squat)

`GestureRecognitionEngine` emits `PINCH_TAP` when a stable PINCH *ends* (the stabilizer's end event, not a release caused by hand loss) after at most `gestures.pinchTap.maxMs` (600 ms). It maps to `SQUAT`: a 900 ms gesture-only action, interruptible, no cooldown. A longer pinch stays the GRAB_ROTATE / lift control.

### Head cue (face mirroring)

`headCueFromLandmarks` (face landmarks 1, 234, 454, 10, 152): `yaw = (nose.x − cheek midpoint) / half cheek width`, `pitch = ((nose.y − forehead.y) / face span − 0.565) / 0.045` (calibrated on 99 faces of the reference video). The coordinator flips yaw for the mirrored view and smooths it (90 ms, 300 ms decay on loss). The scene adds `cue · gain · 25°` (max `interaction.mimic.maxDeg`, 40°) to the head yaw / pitch while the robot looks at the user. It is visual only and never issues a command.

## Extension points

| To add… | Do this |
|---|---|
| A static gesture | Add a score to `StaticGestureClassifier.scores`, a definition to `gestures.definitions` (config), and the name to `STATIC_GESTURES`. |
| A dynamic gesture | Add a detector like `SwipeDetector` and call it from `GestureRecognitionEngine.update`; emit `{ type: 'event' }`. |
| A learned classifier | Implement `classify(features, enabled) → { gesture, confidence, scores, reason }` and pass it to the engine. The stabilizer, gating and commands stay unchanged. |
| Another tracking model | Implement the backend interface (`init / detect / close`) returning MediaPipe-shaped landmarks (21 points, same order). |
| A command | Add an entry to `COMMANDS` (`kind: action/hold/toggle`), handle it in `Scene3D`, and bind it in config. |
| A 3D model | Set `scene.robotModelUrl` to a GLB file; name clips `idle`/`wave`/`spin…`. |
| Another camera | The camera selector appears automatically when more than one video input exists; `camera.deviceId` sets it in config. |
| Voice or an AI agent | Submit task requests to `TaskBroker` (`app.tasks.submit({ type, args, origin: 'agent' })`); register new task types with their own argument validation. See [AGENT_INTERFACE.md](AGENT_INTERFACE.md). No network endpoint exists today. |
| Session recording/analysis | Subscribe to `bus.on('*')`. `exportSession()` already writes metrics and events without biometric data. |

## Decisions

1. **Browser stack (Vite + MediaPipe Tasks Vision + three.js)** instead of Python, OpenCV and a native 3D library.
   - The repo was empty and Python is 3.14 here.
   - The browser gives WebGL rendering, a translucent DOM HUD and camera permission handling with few native dependencies.
   - Processing stays local.
2. **Deterministic rules first.** There is no labelled data, so no learned classifier is trained. Rules are explainable, testable and replaceable behind the same contract.
3. **Synchronous inference on the main thread.** This is the simplest correct scheduling and needs no queue. Its cost is measured and documented; moving it to a worker is roadmap item 1.
4. **Procedural robot.** No suitably licensed robot asset was available. Building it in code avoids licensing issues and lets every part be animated. GLB loading exists for real assets.
5. **Runtime assets served from the app's own origin.** WASM and the model are copied or downloaded once by `npm run setup`; nothing is fetched from a CDN while running.
6. **Default `numHands: 1`.** It measured 39 ms inference vs 69 ms with two visible hands on the test GPU. Multi-hand is supported through config.
7. **Python engine added as a second mode, not a replacement.**
   - The browser engine was complete and tested, so it was kept as the default.
   - In Python mode the browser does not touch the camera: one owner, no conflict over the device. Preview frames are streamed instead of opening the camera twice.
   - The transport is the `websockets` library only; no web framework is needed for one local client.
   - Defaults are mirrored in both languages, and a test enforces that they stay identical.
8. **Agent server inside the Vite server.**
   - The page is always served by Vite (`npm run dev` / `dev:python` / `preview`), so the agent API works in both vision modes without another process.
   - The key stays server-side, and the Python engine (the camera owner) stays independent of provider availability.
   - The loop is manual, not the SDK tool runner, because tools execute in the browser across HTTP round trips with confirmation pauses.
   - The SDK is pinned to `@anthropic-ai/sdk@0.133.0`; the 1.0.0 major release was a day old at integration time.
