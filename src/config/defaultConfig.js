/**
 * Default configuration. Every tunable threshold used by the pipeline lives
 * here so it can be inspected, overridden (public/aura.config.json) and
 * validated (see config.js). Units are given in each comment.
 */
export const DEFAULT_CONFIG = Object.freeze({
  // Which computer-vision engine owns the camera. Exactly one is active:
  //   'browser' - getUserMedia + MediaPipe WASM in this page (no Python needed)
  //   'python'  - the Python engine (backend/aura_cv) owns the camera and streams
  //               results + preview frames over a localhost WebSocket
  // The `backend` section of the config file is read by the Python engine only.
  engine: {
    mode: 'browser',
    url: 'ws://127.0.0.1:8765',
    connectTimeoutMs: 4000,
  },

  camera: {
    width: 1280, // requested capture width (px); the device may choose another
    height: 720, // requested capture height (px)
    frameRate: 30, // requested capture rate (fps); measured rate is shown in the HUD
    facingMode: 'user',
    deviceId: null, // specific camera id, or null for the default device
    mirror: true, // show a selfie view; gesture directions use the mirrored view
  },

  tracking: {
    modelAssetPath: '/models/hand_landmarker.task', // fallback when the gesture model is missing
    // Deep-learning gesture model (MediaPipe GestureRecognizer): the same 21-point hand
    // landmarker plus a learned gesture classifier, in one bundle (one hand pipeline).
    gestureModelAssetPath: '/models/gesture_recognizer.task',
    // Project-trained hand-pose net weights (training/train_handpose.py); optional.
    handPoseNetPath: '/models/aura_handpose.json',
    wasmPath: '/mediapipe/wasm',
    delegate: 'GPU', // 'GPU' | 'CPU'; GPU falls back to CPU automatically on failure
    numHands: 1, // 2+ supported; each extra visible hand roughly doubles inference time (measured)
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  },

  gestures: {
    // Learned gesture classifier (GestureRecognizer). When its result is present it decides
    // the static gesture (PINCH, which it has no class for, comes from the pinch distance);
    // without it the rule classifier is used as a labelled fallback.
    model: {
      enabled: true,
      minScore: 0.6, // model score needed for a gesture; below it the frame is UNKNOWN
    },
    // Project-trained hand-pose network (training/, HaGRIDv2 landmarks): decides the
    // three-finger RUN pose (no canned class) and assists when the canned model says None.
    net: {
      enabled: true,
      minScore: 0.7,
    },
    // A hand whose handedness score is below this is not classified (UNKNOWN).
    minTrackingScore: 0.6,
    // Fraction of the 21 landmarks that must lie inside the image.
    minInFrameRatio: 0.9,
    // Winning rule score must reach this and beat the runner-up by `minMargin`.
    minScore: 0.6,
    minMargin: 0.2,
    // Stabilizer: how long a different/absent label must persist before the
    // current stable gesture is released (ms).
    exitMs: 180,
    // Hands moving faster than this (palm lengths per second) are treated as
    // "moving" for gestures flagged requireStill.
    maxStillSpeed: 1.5,
    // After a swipe, static gesture onsets are suppressed for this long (ms).
    postSwipeSuppressMs: 700,
    definitions: {
      OPEN_PALM: { enabled: true, holdMs: 500, requireStill: true },
      FIST: { enabled: true, holdMs: 600, requireStill: true },
      PINCH: { enabled: true, holdMs: 150, requireStill: false },
      POINT: { enabled: true, holdMs: 120, requireStill: false },
      // index + middle up, others folded: WALK. holdMs is the (short) debounce.
      TWO_FINGERS: { enabled: true, holdMs: 150, requireStill: false },
      // index + middle + ring up, pinky + thumb folded: RUN.
      THREE_FINGERS: { enabled: true, holdMs: 150, requireStill: false },
      // thumb clearly extended and pointing up / down, other four fingers folded (emotions).
      THUMB_UP: { enabled: true, holdMs: 500, requireStill: true },
      THUMB_DOWN: { enabled: true, holdMs: 500, requireStill: true },
    },
    swipe: {
      enabled: true,
      windowMs: 450, // max duration of the stroke (ms)
      minDistance: 2.2, // min horizontal travel (palm lengths)
      minSpeed: 4.0, // min average speed over the stroke (palm lengths / s)
      maxVerticalRatio: 0.6, // |dy| / |dx| upper bound
      maxGapMs: 150, // a tracking gap longer than this resets the history
      cooldownMs: 600, // no swipe at all for this long after one fired
      oppositeSuppressMs: 1100, // suppress the return stroke in the opposite direction
    },
    // Pinch tap: thumb and index tips touch and part again quickly (PINCH_TAP -> SQUAT).
    // A longer pinch stays GRAB_ROTATE (turn / lift).
    pinchTap: {
      enabled: true,
      maxMs: 600, // longest stable pinch that still counts as a tap
    },
    // Continuous index-finger rotation while pointing up (SPIN, see SpinDetector.js).
    spin: {
      enabled: true,
      windowMs: 700, // trajectory that must show the rotational signature (ms)
      rateWindowMs: 200, // most recent part used for the angular-velocity estimate (ms)
      minWindowMs: 160, // shortest trajectory evaluated (ms)
      minSweepDeg: 100, // swept angle within windowMs: rejects brief arcs
      minRadius: 0.25, // mean radius (palm lengths): rejects a still or jittering finger
      maxRadiusCv: 0.5, // radius std / mean: rejects lines and zig-zags
      minConsistency: 0.75, // fraction of steps turning the same way: rejects noise
      maxStepDeg: 150, // largest angle step between two frames: rejects tracking jumps
      maxGapMs: 200, // a tracking gap longer than this restarts the trajectory
      poseGraceMs: 150, // leaving the pointing shape longer than this ends the spin
      enterMs: 100, // the signature must persist this long before rotation starts (100: a brief arc still never spins)
      releaseMs: 200, // the signature may vanish this long before rotation ends (brief gaps)
    },
    // Finger walking: index and middle finger alternate like two legs (FINGER_WALK, see WalkCadenceDetector.js).
    walk: {
      enabled: true,
      windowMs: 2000, // rolling window for amplitude, correlation and cadence (ms)
      minAmplitude: 0.12, // peak-to-peak movement of EACH finger (palm lengths): rejects a static V
      minReach: 0.6, // each finger must be nearly straight at some point (palm lengths): rejects circling while pointing
      hysteresis: 0.04, // band around the mean the difference signal must cross for a step (palm lengths)
      minSteps: 3, // alternating steps before walking starts (false-start filter)
      maxCorrelation: -0.3, // index vs middle movement correlation must be below this (anti-phase)
      stopMs: 1100, // no new step for this long: walking ends (speed already falls with 1/time-since-last-step)
      maxStepRate: 8, // steps/s above this are treated as shaking (rejected)
      poseGraceMs: 200, // leaving the legs pose longer than this ends walking
    },
  },

  interaction: {
    // gesture -> command. Unknown command names are reported and rejected.
    bindings: {
      OPEN_PALM: 'WAVE',
      FIST: 'TOGGLE_LOCK',
      PINCH: 'GRAB_ROTATE',
      POINT: 'AIM',
      SWIPE_LEFT: 'SPIN_LEFT',
      SWIPE_RIGHT: 'SPIN_RIGHT',
      TWO_FINGERS: 'WALK', // V: walk
      THREE_FINGERS: 'RUN', // three fingers: run
      PINCH_TAP: 'SQUAT', // touch and release thumb + index: one squat
      FINGER_WALK: null, // alternating finger steps (cadence) are still detected, not bound
      SPIN: 'SPIN',
      THUMB_UP: 'LAUGH',
      THUMB_DOWN: 'SAD',
    },
    // Minimum time between two accepted activations of the same command (ms).
    cooldownMs: 900,
    // WALK / RUN (holds): fixed speeds; steer with the hand's horizontal offset.
    // (FINGER_WALK, if bound, maps the step cadence to a speed instead.)
    walk: {
      speed: 0.7, // m/s while walking (two fingers)
      runSpeed: 1.6, // m/s while running (three fingers)
      minCadence: 1.0, // below this (dead zone) the target speed is 0
      minSpeed: 0.15, // m/s at minCadence
      speedPerCadence: 0.3, // m/s added per step/s above minCadence
      maxSpeed: 1.8, // m/s cap
      accel: 6.0, // m/s^2 limit when speeding up (fast response)
      decel: 4.0, // m/s^2 limit when the cadence drops; the walk itself ending stops at once
      maxDistance: 2.0, // m from the home position; the robot stops at the boundary
      strideLength: 0.35, // m per step: the gait animation advances with the distance walked
    },
    // SPIN (hold): robot yaw rate follows the finger's angular velocity.
    spin: {
      gain: 0.85, // robot rate = gain * (|finger rate| - deadZone)
      deadZoneDegS: 115, // finger rates below this give no rotation
      maxSpeedDegS: 1080, // cap (3 turns/s: a tornado)
      accelDegS2: 6000, // limit when speeding up (fast response)
      decelDegS2: 4000, // limit when slowing or after the gesture ends; lock / hand loss stop at once
    },
    emotions: {
      laughMs: 2200,
      sadMs: 2600,
    },
    squatMs: 900, // one squat (down and up)
    // Where the robot looks (AttentionController.js). Coordinates: normalised screen space.
    attention: {
      handAttentionHolds: [], // holds during which the robot looks at the hand; none: it always looks at the user
      minFaceScore: 0.6, // detections below this are ignored
      faceHoldMs: 700, // keep looking where the face was for this long after it is lost, then neutral
      switchAfterMs: 500, // another face becomes primary only after the primary was missing this long
      maxJump: 0.25, // a detection further than this from the primary face is a different face
      yawGain: 1.8, // rad of head yaw per screen width between the robot's head and the face
      pitchGain: 1.4, // rad of head pitch per screen height
      maxYawDeg: 46,
      maxPitchUpDeg: 22,
      maxPitchDownDeg: 26,
      smoothingMs: 80, // exponential time constant (fast but not jittery)
      maxSpeedDegS: 480, // head angular speed cap (prevents snapping)
    },
    // GRAB_ROTATE (pinch hold) also lifts: raising the pinched hand lifts the robot like a
    // marionette ("heavy lifting"); releasing lowers it smoothly.
    lift: {
      gain: 2.2, // metres per screen height of vertical hand travel
      maxHeight: 0.9, // m
      dropMs: 450, // time constant of the smooth descent after release / lock / hand loss
    },
    // Steering while walking: horizontal hand offset from where the walk started -> yaw rate.
    steer: {
      gainDegS: 260, // deg/s per unit of normalised horizontal offset
      deadZone: 0.05, // offsets below this do not steer
      maxDegS: 110,
    },
    // Facial expressions (FaceExpressionEngine, from the face model's blendshape scores).
    expression: {
      enabled: true,
      smoothingMs: 150, // exponential smoothing of the expression scores
      enter: 0.45, // score needed (for holdMs) before an expression is reported
      exit: 0.3, // below this the expression is released
      holdMs: 200,
      gain: { HAPPY: 2.5, SAD: 2, SURPRISED: 1 }, // score = gain * (raw - the user's resting level)
      baseline: { enabled: true, upMs: 30000, downMs: 1500 }, // per-user resting level (fast down, slow up)
      react: true, // the robot mirrors it (visual only; never moves it or changes its state)
    },
    // Head imitation: the robot turns / raises its head the way you do (mirror view).
    mimic: {
      enabled: true,
      yawGain: 1.6, // robot head yaw per unit of your head-turn cue
      pitchGain: 1.4,
      maxDeg: 40,
      // per-user resting head pose (HeadCueCalibrator): learnt in the first 1.5 s of face, then slowly
      calibration: { enabled: true, calibrateMs: 1500, baselineMs: 30000, relearnMs: 5000 },
    },
  },

  // Face perception (MediaPipe FaceLandmarker: detector + 478 landmarks + 52 blendshape
  // scores + head-pose matrix; runs locally). Falls back to the BlazeFace detector (gaze only).
  face: {
    enabled: true,
    modelAssetPath: '/models/face_landmarker.task',
    detectorAssetPath: '/models/blaze_face_short_range.tflite',
    numFaces: 2,
    minScore: 0.6,
    intervalMs: 100, // detection period: faces need far less than the hand rate
  },

  scene: {
    robotModelUrl: null, // optional GLB/glTF URL; null uses the procedural robot
    robotScale: 1.0,
    pixelRatioCap: 2, // limits renderer resolution on high-DPI screens
  },

  effects: {
    trailMs: 450, // fingertip trail lifetime (ms)
    maxTrailPoints: 40, // per fingertip
    trailFingertips: [4, 8, 12, 16, 20], // landmark indices that leave trails
    fadeOutMs: 350, // effects fade after tracking is lost
    glow: true,
  },

  hud: {
    maxEvents: 9,
    refreshMs: 100, // HUD text refresh period (ms); metrics are measured every frame
  },
});
