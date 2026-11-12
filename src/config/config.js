import { DEFAULT_CONFIG } from './defaultConfig.js';

/**
 * Validation rules keyed by dotted path. Paths not listed here are accepted
 * only if they exist in DEFAULT_CONFIG with the same JS type.
 * Each rule returns an error string or null.
 */
const num = (min, max) => (v) =>
  typeof v !== 'number' || !Number.isFinite(v) ? 'must be a finite number'
    : v < min || v > max ? `must be between ${min} and ${max}` : null;
const int = (min, max) => (v) =>
  !Number.isInteger(v) ? 'must be an integer' : num(min, max)(v);
const oneOf = (...values) => (v) => (values.includes(v) ? null : `must be one of ${values.join(', ')}`);
const bool = (v) => (typeof v === 'boolean' ? null : 'must be a boolean');
const str = (v) => (typeof v === 'string' && v.length > 0 ? null : 'must be a non-empty string');
const strOrNull = (v) => (v === null ? null : str(v));

const wsUrl = (v) => (typeof v === 'string' && /^wss?:\/\/[^\s]+$/.test(v) ? null : 'must be a ws:// or wss:// URL');

const RULES = {
  'engine.mode': oneOf('browser', 'python'),
  'engine.url': wsUrl,
  'engine.connectTimeoutMs': num(100, 60000),
  'camera.width': int(160, 3840),
  'camera.height': int(120, 2160),
  'camera.frameRate': num(1, 120),
  'camera.facingMode': oneOf('user', 'environment'),
  'camera.deviceId': strOrNull,
  'camera.mirror': bool,
  'tracking.modelAssetPath': str,
  'tracking.gestureModelAssetPath': strOrNull,
  'gestures.model.enabled': bool,
  'gestures.model.minScore': num(0, 1),
  'gestures.net.enabled': bool,
  'gestures.net.minScore': num(0, 1),
  'tracking.handPoseNetPath': strOrNull,
  'tracking.wasmPath': str,
  'tracking.delegate': oneOf('GPU', 'CPU'),
  'tracking.numHands': int(1, 4),
  'tracking.minHandDetectionConfidence': num(0, 1),
  'tracking.minHandPresenceConfidence': num(0, 1),
  'tracking.minTrackingConfidence': num(0, 1),
  'gestures.minTrackingScore': num(0, 1),
  'gestures.minInFrameRatio': num(0, 1),
  'gestures.minScore': num(0, 1),
  'gestures.minMargin': num(0, 1),
  'gestures.exitMs': num(0, 5000),
  'gestures.maxStillSpeed': num(0, 100),
  'gestures.postSwipeSuppressMs': num(0, 10000),
  'gestures.swipe.enabled': bool,
  'gestures.swipe.windowMs': num(50, 5000),
  'gestures.swipe.minDistance': num(0.1, 50),
  'gestures.swipe.minSpeed': num(0, 100),
  'gestures.swipe.maxVerticalRatio': num(0, 10),
  'gestures.swipe.maxGapMs': num(1, 5000),
  'gestures.swipe.cooldownMs': num(0, 10000),
  'gestures.swipe.oppositeSuppressMs': num(0, 10000),
  'gestures.spin.enabled': bool,
  'gestures.spin.windowMs': num(200, 3000),
  'gestures.spin.rateWindowMs': num(80, 2000),
  'gestures.spin.minWindowMs': num(50, 2000),
  'gestures.spin.minSweepDeg': num(30, 720),
  'gestures.spin.minRadius': num(0.01, 20),
  'gestures.spin.maxRadiusCv': num(0.01, 2),
  'gestures.spin.minConsistency': num(0.5, 1),
  'gestures.spin.maxStepDeg': num(5, 180),
  'gestures.spin.maxGapMs': num(1, 2000),
  'gestures.spin.poseGraceMs': num(0, 1000),
  'gestures.spin.enterMs': num(0, 2000),
  'gestures.spin.releaseMs': num(0, 1000),
  'gestures.pinchTap.enabled': bool,
  'gestures.pinchTap.maxMs': num(50, 3000),
  'gestures.walk.enabled': bool,
  'gestures.walk.windowMs': num(500, 5000),
  'gestures.walk.minAmplitude': num(0.01, 2),
  'gestures.walk.minReach': num(-1, 2),
  'gestures.walk.hysteresis': num(0, 1),
  'gestures.walk.minSteps': int(2, 20),
  'gestures.walk.maxCorrelation': num(-1, 0),
  'gestures.walk.stopMs': num(100, 3000),
  'gestures.walk.maxStepRate': num(1, 30),
  'gestures.walk.poseGraceMs': num(0, 1000),
  'interaction.cooldownMs': num(0, 60000),
  'interaction.walk.speed': num(0.05, 5),
  'interaction.walk.runSpeed': num(0.05, 5),
  'interaction.walk.minCadence': num(0.1, 10),
  'interaction.walk.minSpeed': num(0, 3),
  'interaction.walk.maxSpeed': num(0.05, 3),
  'interaction.walk.speedPerCadence': num(0, 3),
  'interaction.walk.accel': num(0.1, 50),
  'interaction.walk.decel': num(0.1, 50),
  'interaction.walk.maxDistance': num(0.1, 5),
  'interaction.walk.strideLength': num(0.05, 2),
  'interaction.spin.gain': num(0, 5),
  'interaction.spin.deadZoneDegS': num(0, 2000),
  'interaction.spin.maxSpeedDegS': num(10, 1440),
  'interaction.spin.accelDegS2': num(10, 20000),
  'interaction.spin.decelDegS2': num(10, 20000),
  'interaction.emotions.laughMs': num(300, 10000),
  'interaction.emotions.sadMs': num(300, 10000),
  'interaction.squatMs': num(300, 5000),
  'interaction.attention.handAttentionHolds': (v) => (Array.isArray(v) && v.every((x) => typeof x === 'string' && /^[A-Z_]+$/.test(x)) ? null : 'must be an array of hold command names'),
  'interaction.attention.minFaceScore': num(0, 1),
  'interaction.attention.faceHoldMs': num(0, 10000),
  'interaction.attention.switchAfterMs': num(0, 10000),
  'interaction.attention.maxJump': num(0.01, 2),
  'interaction.attention.yawGain': num(0, 10),
  'interaction.attention.pitchGain': num(0, 10),
  'interaction.attention.maxYawDeg': num(0, 90),
  'interaction.attention.maxPitchUpDeg': num(0, 60),
  'interaction.attention.maxPitchDownDeg': num(0, 60),
  'interaction.attention.smoothingMs': num(1, 5000),
  'interaction.attention.maxSpeedDegS': num(10, 2000),
  'face.enabled': bool,
  'face.modelAssetPath': str,
  'face.detectorAssetPath': strOrNull,
  'face.numFaces': int(1, 4),
  'interaction.lift.gain': num(0, 10),
  'interaction.lift.maxHeight': num(0, 3),
  'interaction.lift.dropMs': num(10, 5000),
  'interaction.steer.gainDegS': num(0, 2000),
  'interaction.steer.deadZone': num(0, 0.5),
  'interaction.steer.maxDegS': num(0, 720),
  'interaction.expression.enabled': bool,
  'interaction.expression.smoothingMs': num(1, 5000),
  'interaction.expression.enter': num(0.05, 1),
  'interaction.expression.exit': num(0, 1),
  'interaction.expression.holdMs': num(0, 5000),
  'interaction.expression.react': bool,
  'interaction.expression.gain.HAPPY': num(0.1, 10),
  'interaction.expression.gain.SAD': num(0.1, 10),
  'interaction.expression.gain.SURPRISED': num(0.1, 10),
  'interaction.expression.baseline.enabled': bool,
  'interaction.expression.baseline.upMs': num(100, 600000),
  'interaction.expression.baseline.downMs': num(10, 600000),
  'interaction.mimic.enabled': bool,
  'interaction.mimic.yawGain': num(0, 5),
  'interaction.mimic.pitchGain': num(0, 5),
  'interaction.mimic.maxDeg': num(0, 90),
  'interaction.mimic.calibration.enabled': bool,
  'interaction.mimic.calibration.calibrateMs': num(0, 60000),
  'interaction.mimic.calibration.baselineMs': num(100, 600000),
  'interaction.mimic.calibration.relearnMs': num(0, 600000),
  'face.minScore': num(0, 1),
  'face.intervalMs': num(16, 5000),
  'scene.robotModelUrl': strOrNull,
  'scene.robotScale': num(0.01, 100),
  'scene.pixelRatioCap': num(0.5, 4),
  'effects.trailMs': num(0, 5000),
  'effects.maxTrailPoints': int(2, 500),
  'effects.fadeOutMs': num(0, 5000),
  'effects.glow': bool,
  'hud.maxEvents': int(1, 100),
  'hud.refreshMs': num(16, 5000),
};

export const STATIC_GESTURES = Object.freeze(['OPEN_PALM', 'FIST', 'PINCH', 'POINT', 'TWO_FINGERS', 'THREE_FINGERS', 'THUMB_UP', 'THUMB_DOWN']);
export const DYNAMIC_GESTURES = Object.freeze(['SWIPE_LEFT', 'SWIPE_RIGHT', 'PINCH_TAP']);
/** Motion gestures with a start and an end and a continuous control signal (gesture.motion). */
export const CONTINUOUS_GESTURES = Object.freeze(['SPIN', 'FINGER_WALK']);
export const ALL_GESTURES = Object.freeze([...STATIC_GESTURES, ...DYNAMIC_GESTURES, ...CONTINUOUS_GESTURES]);

const isPlainObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);

function deepClone(v) {
  return JSON.parse(JSON.stringify(v));
}

/**
 * Merge `overrides` onto the defaults and validate the result.
 * Invalid or unknown values are rejected individually (the default is kept)
 * and reported, so one typo cannot silently disable the application.
 *
 * @param {unknown} overrides parsed JSON from the user config, or undefined
 * @param {{ knownCommands?: string[] }} [opts]
 * @returns {{ config: object, errors: string[], warnings: string[] }}
 */
export function loadConfig(overrides, opts = {}) {
  const config = deepClone(DEFAULT_CONFIG);
  const errors = [];
  const warnings = [];

  if (overrides !== undefined && overrides !== null && !isPlainObject(overrides)) {
    errors.push('configuration root must be a JSON object');
    return { config, errors, warnings };
  }

  const walk = (src, dst, path) => {
    for (const [key, value] of Object.entries(src ?? {})) {
      const p = path ? `${path}.${key}` : key;
      if (p === 'backend') continue; // owned and validated by the Python engine
      if (p === 'gestures.definitions') {
        mergeDefinitions(value, dst.definitions, errors);
        continue;
      }
      if (p === 'interaction.bindings') {
        mergeBindings(value, dst.bindings, errors);
        continue;
      }
      if (p === 'effects.trailFingertips') {
        if (Array.isArray(value) && value.every((i) => Number.isInteger(i) && i >= 0 && i <= 20)) {
          dst.trailFingertips = [...value];
        } else {
          errors.push(`${p}: must be an array of landmark indices 0-20`);
        }
        continue;
      }
      if (!(key in dst)) {
        errors.push(`${p}: unknown setting`);
        continue;
      }
      if (isPlainObject(dst[key])) {
        if (!isPlainObject(value)) errors.push(`${p}: must be an object`);
        else walk(value, dst[key], p);
        continue;
      }
      const rule = RULES[p];
      const err = rule ? rule(value) : typeof value === typeof dst[key] ? null : `must be a ${typeof dst[key]}`;
      if (err) errors.push(`${p}: ${err}`);
      else dst[key] = value;
    }
  };
  walk(overrides, config, '');

  if (opts.knownCommands) {
    for (const [gesture, command] of Object.entries(config.interaction.bindings)) {
      if (command !== null && !opts.knownCommands.includes(command)) {
        warnings.push(`interaction.bindings.${gesture}: command "${command}" is not implemented; it will be rejected as UNSUPPORTED`);
      }
    }
  }

  return { config, errors, warnings };
}

function mergeDefinitions(value, dst, errors) {
  if (!isPlainObject(value)) {
    errors.push('gestures.definitions: must be an object');
    return;
  }
  for (const [gesture, def] of Object.entries(value)) {
    const p = `gestures.definitions.${gesture}`;
    if (!STATIC_GESTURES.includes(gesture)) {
      errors.push(`${p}: unknown static gesture (known: ${STATIC_GESTURES.join(', ')})`);
      continue;
    }
    if (!isPlainObject(def)) {
      errors.push(`${p}: must be an object`);
      continue;
    }
    for (const [k, v] of Object.entries(def)) {
      const err =
        k === 'enabled' || k === 'requireStill' ? bool(v)
          : k === 'holdMs' ? num(0, 10000)(v)
            : 'unknown setting';
      if (err) errors.push(`${p}.${k}: ${err}`);
      else dst[gesture][k] = v;
    }
  }
}

function mergeBindings(value, dst, errors) {
  if (!isPlainObject(value)) {
    errors.push('interaction.bindings: must be an object');
    return;
  }
  for (const [gesture, command] of Object.entries(value)) {
    const p = `interaction.bindings.${gesture}`;
    if (!ALL_GESTURES.includes(gesture)) errors.push(`${p}: unknown gesture`);
    else if (command !== null && (typeof command !== 'string' || !/^[A-Z][A-Z0-9_]*$/.test(command))) {
      errors.push(`${p}: command must be an UPPER_SNAKE_CASE string or null`);
    } else dst[gesture] = command;
  }
}

/**
 * Browser helper: fetch an optional JSON override file.
 * A missing file (404) is normal; a malformed file is reported.
 * @param {string} url
 * @param {typeof fetch} fetchFn
 * @returns {Promise<{ overrides: unknown, source: string, error: string|null }>}
 */
export async function fetchConfigOverrides(url, fetchFn = fetch) {
  try {
    const res = await fetchFn(url, { cache: 'no-store' });
    if (res.status === 404) return { overrides: undefined, source: 'defaults', error: null };
    if (!res.ok) return { overrides: undefined, source: 'defaults', error: `config HTTP ${res.status}` };
    const text = await res.text();
    // Dev servers may answer unknown paths with index.html.
    if (/^\s*</.test(text)) return { overrides: undefined, source: 'defaults', error: null };
    try {
      return { overrides: JSON.parse(text), source: url, error: null };
    } catch (e) {
      return { overrides: undefined, source: 'defaults', error: `${url} is not valid JSON: ${e.message}` };
    }
  } catch (e) {
    return { overrides: undefined, source: 'defaults', error: `could not load ${url}: ${e.message}` };
  }
}
