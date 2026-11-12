import { COMMANDS, commandRequestError } from './commands.js';
import { gestureEventError } from './gestureEvents.js';
import { createRobotStateMachine } from './RobotStateMachine.js';

/** Radians of robot yaw per unit of horizontal palm travel (normalised display width). */
export const GRAB_ROTATE_GAIN = Math.PI * 2.2;
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const DEG = Math.PI / 180;
/** Wrap an angle to (-PI, PI]. */
const wrapAngle = (a) => {
  const w = a - 2 * Math.PI * Math.floor((a + Math.PI) / (2 * Math.PI));
  return w <= -Math.PI ? w + 2 * Math.PI : w;
};
const finite = (v, lo, hi) => (typeof v === 'number' && Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : 0);
/** Move `cur` toward `target` by at most `maxDelta`. */
const approach = (cur, target, maxDelta) => cur + Math.max(-maxDelta, Math.min(maxDelta, target - cur));

export const WALK_DEFAULTS = Object.freeze({ speed: 0.7, runSpeed: 1.6, minCadence: 1.0, minSpeed: 0.15, speedPerCadence: 0.3, maxSpeed: 1.8, accel: 6.0, decel: 4.0, maxDistance: 2.0, strideLength: 0.35 });
export const SPIN_DEFAULTS = Object.freeze({ gain: 0.85, deadZoneDegS: 115, maxSpeedDegS: 1080, accelDegS2: 6000, decelDegS2: 4000 });
export const EMOTION_DEFAULTS = Object.freeze({ laughMs: 2200, sadMs: 2600 });
/** Holds that move the robot along the floor (same integration, different speed source). */
const LOCOMOTION = new Set(['WALK', 'RUN']);
/** A WALK <-> RUN switch within this time keeps the current speed (no stop between them). */
const CARRY_MS = 600;
export const LIFT_DEFAULTS = Object.freeze({ gain: 2.2, maxHeight: 0.9, dropMs: 450 });
export const STEER_DEFAULTS = Object.freeze({ gainDegS: 260, deadZone: 0.05, maxDegS: 110 });

/** Steering yaw rate (rad/s) for a horizontal hand offset (normalised display units) while walking. */
export function steerRateFor(offset, c = STEER_DEFAULTS) {
  if (!Number.isFinite(offset)) return 0;
  const mag = Math.abs(offset) - c.deadZone;
  if (!(mag > 0)) return 0;
  return Math.sign(offset) * Math.min(c.maxDegS, c.gainDegS * mag) * DEG;
}

/** Target walking speed (m/s) for a finger-step cadence (steps/s); 0 inside the dead zone. */
export function walkSpeedFor(cadence, c = WALK_DEFAULTS) {
  if (!(cadence >= c.minCadence)) return 0;
  return Math.min(c.maxSpeed, Math.max(c.minSpeed, c.minSpeed + (cadence - c.minCadence) * c.speedPerCadence));
}

/** Target robot yaw rate (rad/s, signed) for a finger angular velocity (rad/s, + = clockwise on screen). */
export function spinRateFor(omega, c = SPIN_DEFAULTS) {
  const mag = Math.abs(omega) - c.deadZoneDegS * DEG;
  if (!(mag > 0)) return 0;
  return Math.sign(omega) * Math.min(c.maxSpeedDegS * DEG, c.gain * mag);
}

/**
 * InteractionController
 *
 * Translates recognised gesture events into commands, validates every
 * command (from any source) and owns the robot state machine. It has no
 * dependency on rendering or the DOM.
 *
 * Validation order for a request (first failure wins, reason in brackets):
 *   malformed request               [INVALID]
 *   name not in the registry        [UNSUPPORTED]
 *   system in ERROR                 [ERROR_STATE]
 *   locked and not TOGGLE_LOCK      [LOCKED]
 *   an action is animating          [BUSY]
 *   a hold control is active        [CONFLICT]
 *   hold command without a hand     [NO_HAND]
 *   same command within cooldown    [COOLDOWN]
 *
 * Every decision is published on the event bus as a `command` event, so
 * the HUD shows accepted *and* rejected commands with their reason.
 *
 * Application state owned here (rendered by Scene3D, never invented by it):
 *   state       robot state machine (RobotStateMachine.js)
 *   pose.yaw    robot orientation (radians): GRAB_ROTATE (pinch-drag), the SPIN
 *               yaw rate, RESET_POSE (animated back to 0)
 *   pose.x/z    robot position on the floor (metres, home = 0,0): advanced along the
 *               facing direction at the WALK speed, bounded by walk.maxDistance,
 *               animated home by RESET_POSE
 *   spinRate    robot yaw rate (rad/s): follows spinRateFor(finger omega) while SPIN is
 *               held, rate-limited (accel / decel); after the gesture is released it
 *               decelerates smoothly; lock, hand loss, error and reset stop it at once
 *   walkSpeed   m/s: follows walkSpeedFor(cadence) while WALK is held, rate-limited;
 *               when WALK ends (cadence stopped, release, hand loss, lock) it is 0 at once
 *   gaitPhase   advances with the distance walked (strideLength per step), so the
 *               walk animation is synchronised with the real movement
 *   lastResult  outcome of the most recently finished command
 * Motion is integrated in tick(t) from the tick clock only (skipped when the clock
 * does not advance, capped at 100 ms per step): frame-rate independent.
 *
 * Priority and safety:
 *   - SPIN (`supersedes: ['AIM']`) ends the AIM hold its pointing shape started.
 *     WALK and SPIN exclude each other (one hand; the gesture engine never has both
 *     active, and a second hold is CONFLICT here). Starting WALK zeroes any spin
 *     coast so yaw is stable while walking.
 *   - The user's lock (keyboard / gesture source) ends ANY hold (WALK, SPIN, GRAB_ROTATE
 *     lift, AIM) or cancels an interruptible action (LAUGH, SAD), then locks; a lock request
 *     from the agent is still rejected (CONFLICT / BUSY): the agent never overrides
 *     a held gesture. Locking zeroes all motion.
 *   - Emotions are timed actions: rejected while moving (CONFLICT) or locked; they
 *     stop a spin coast; they never move the robot.
 */
export class InteractionController {
  /**
   * @param {{ bindings: Record<string, string|null>, cooldownMs: number, walk?: object, spin?: object, emotions?: object }} cfg
   * @param {{ bus?: { emit: Function }, commands?: Record<string, any> }} [deps]
   */
  constructor(cfg, deps = {}) {
    this.bindings = { ...cfg.bindings };
    this.cooldownMs = cfg.cooldownMs;
    this.commands = deps.commands ?? COMMANDS;
    this.bus = deps.bus ?? null;
    this.handPresent = false;
    /** @type {null | { name: string, source: string, startedAt: number, durationMs: number, params: object }} */
    this.action = null;
    /** @type {null | { name: string, source: string, startedAt: number, params: object, gesture: string|null }} */
    this.hold = null;
    this.lastAccepted = new Map();
    this.error = null;
    this.counts = { accepted: 0, rejected: 0, invalidEvents: 0 };
    this.pose = { yaw: 0, x: 0, z: 0, lift: 0 };
    this.liftCfg = { ...LIFT_DEFAULTS, ...(cfg.lift ?? {}) };
    this.steerCfg = { ...STEER_DEFAULTS, ...(cfg.steer ?? {}) };
    this.liftTarget = 0;
    this.steerRate = 0; // rad/s while walking
    this._steerAnchorX = null;
    this._steerInput = 0;
    this.walkCfg = { ...WALK_DEFAULTS, ...(cfg.walk ?? {}) };
    this.spinCfg = { ...SPIN_DEFAULTS, ...(cfg.spin ?? {}) };
    const emo = { ...EMOTION_DEFAULTS, ...(cfg.emotions ?? {}) };
    this.durations = { LAUGH: emo.laughMs, SAD: emo.sadMs, ...(cfg.squatMs ? { SQUAT: cfg.squatMs } : {}) };
    this._carry = null; // { speed, t, yawRate } of a WALK / RUN released by the gesture
    this.motionInput = { omega: 0, cadence: 0 };
    this.spinRate = 0;
    this.walkSpeed = 0;
    this.gaitPhase = 0;
    this.atBoundary = false;
    this._lastTick = null;
    this.lastResult = null; // { name, source, outcome: 'completed'|'ended'|'cancelled', at, pose }
    this._grab = null; // { baseYaw, anchorX }
    this._resetFrom = null;
    this.fsm = createRobotStateMachine((from, to, reason) => this._emit('state', { from, to, reason }));
  }

  get state() {
    return this.fsm.state;
  }

  get locked() {
    return this.fsm.state === 'LOCKED';
  }

  /**
   * Feed GestureRecognitionEngine events.
   * @param {{ type: string, gesture: string, t: number }[]} events
   * @param {number} t
   * @param {{ features?: any }} [ctx]
   */
  handleGestureEvents(events, t, ctx = {}) {
    const results = [];
    for (const ev of events) {
      const invalid = gestureEventError(ev);
      if (invalid) {
        // Malformed events never reach command handling.
        this.counts.invalidEvents++;
        this._emit('gesture_rejected', { reason: invalid, t });
        continue;
      }
      if (ev.type === 'end') {
        if (this.hold && this.hold.gesture === ev.gesture) this._endHold(t, 'gesture released', { coast: true });
        continue;
      }
      const name = this.bindings[ev.gesture];
      if (name === undefined || name === null) continue; // gesture intentionally unbound
      if (this.hold && this.commands[name]?.supersedes?.includes(this.hold.name)) {
        this._endHold(t, `superseded by ${ev.gesture}`);
      }
      const params = { gesture: ev.gesture };
      const f = ctx.features;
      if (f?.valid) params.anchor = { ...f.palmCenterNorm };
      results.push(this.submit({ name, source: 'gesture', params }, t));
    }
    return results;
  }

  /**
   * Validate and execute a command request.
   * @param {{ name: string, source: string, params?: object }} req
   * @param {number} t
   * @returns {{ accepted: boolean, reason: string|null, name: string }}
   */
  submit(req, t) {
    const malformed = commandRequestError(req);
    if (malformed) return this._reject(req, 'INVALID', t, malformed);
    const def = this.commands[req.name];
    if (!def) return this._reject(req, 'UNSUPPORTED', t, `command "${req.name}" is not implemented`);

    let s = this.fsm.state;
    if (s === 'ERROR') return this._reject(req, 'ERROR_STATE', t, this.error ?? 'system error');
    if (req.name === 'TOGGLE_LOCK' && req.source !== 'agent') {
      // Safety clutch: the user's lock stops the movement / emotion first, then locks.
      // any hold (walk, spin, pinch-lift, aim): safety takes priority over every interaction
      if (s === 'CONTROL' && this.hold) {
        this._endHold(t, 'stopped by safety lock');
        s = this.fsm.state;
      } else if (s === 'EXECUTING' && this.commands[this.action?.name]?.interruptible) {
        this._cancelAction(t, 'safety lock');
        s = this.fsm.state;
      }
    }
    if (s === 'LOCKED' && req.name !== 'TOGGLE_LOCK') return this._reject(req, 'LOCKED', t, 'commands are locked');
    if (s === 'EXECUTING') return this._reject(req, 'BUSY', t, `${this.action?.name} is still running`);
    if (s === 'CONTROL') return this._reject(req, 'CONFLICT', t, `${this.hold?.name} control is active`);
    if (def.kind === 'hold' && !this.handPresent) return this._reject(req, 'NO_HAND', t, 'hold commands need a tracked hand');
    const last = this.lastAccepted.get(req.name);
    const cooldown = def.cooldownMs ?? this.cooldownMs;
    if (def.kind !== 'hold' && last !== undefined && t - last < cooldown) {
      return this._reject(req, 'COOLDOWN', t, `${Math.round(cooldown - (t - last))} ms remaining`);
    }

    const params = { ...(req.params ?? {}) };
    if (def.kind === 'toggle') {
      const to = s === 'LOCKED' ? this._restingState() : 'LOCKED';
      if (to === 'LOCKED') this._stopMotion();
      this.fsm.transition(to, t, req.name);
    } else if (def.kind === 'hold') {
      this.hold = { name: req.name, source: req.source, startedAt: t, params, gesture: params.gesture ?? null };
      if (req.name === 'GRAB_ROTATE') {
        this._grab = { baseYaw: this.pose.yaw, anchorX: params.anchor?.x ?? null, anchorY: params.anchor?.y ?? null, baseLift: this.pose.lift };
      }
      if (LOCOMOTION.has(req.name)) {
        this._steerAnchorX = params.anchor?.x ?? null;
        this._steerInput = 0;
        this.steerRate = 0;
        this.spinRate = 0; // stable yaw while walking
        // switching walk <-> run keeps the speed (the robot accelerates / slows, never stops)
        const c = this._carry;
        this.walkSpeed = c && t - c.t <= CARRY_MS ? c.speed : 0;
        this._carry = null;
      }
      this.fsm.transition('CONTROL', t, req.name);
    } else {
      const durationMs = this.durations[req.name] ?? def.durationMs;
      this.action = { name: req.name, source: req.source, startedAt: t, durationMs, params };
      this.spinRate = 0; // actions never start on a spinning robot
      if (req.name === 'RESET_POSE') this._resetFrom = { ...this.pose };
      this.fsm.transition('EXECUTING', t, req.name);
    }
    this.lastAccepted.set(req.name, t);
    this.counts.accepted++;
    this._emit('command', { status: 'accepted', name: req.name, source: req.source, t });
    return { accepted: true, reason: null, name: req.name };
  }

  /**
   * Update hand presence (once per processed frame).
   * @param {boolean} present
   * @param {number} t
   */
  setHandPresent(present, t) {
    if (present === this.handPresent) return;
    this.handPresent = present;
    const s = this.fsm.state;
    if (!present) {
      this.spinRate = 0; // no coasting without a hand
      this.motionInput = { omega: 0, cadence: 0 };
    }
    if (!present && s === 'CONTROL') this._endHold(t, 'hand lost');
    else if (s === 'IDLE' || s === 'TRACKING') this.fsm.transition(present ? 'TRACKING' : 'IDLE', t, present ? 'hand detected' : 'hand lost');
  }

  /**
   * Apply continuous hand input to the active hold (once per processed frame).
   * GRAB_ROTATE: yaw follows horizontal palm travel since the pinch began.
   * SPIN / WALK: the motion signal (finger omega, step cadence) becomes the target
   * of the rate-limited yaw rate / speed integrated in tick(). Values are clamped;
   * a missing or malformed signal counts as 0 (the robot slows, never jumps).
   * @param {any} features primary-hand features (valid) or null
   * @param {{ spin?: { omega: number }, walk?: { cadence: number } } | null} [motion]
   */
  updateHold(features, motion = null) {
    this.motionInput = { omega: finite(motion?.spin?.omega, -60, 60), cadence: finite(motion?.walk?.cadence, 0, 30) };
    const valid = features?.valid && Number.isFinite(features.palmCenterNorm?.x) && Number.isFinite(features.palmCenterNorm?.y);
    if (LOCOMOTION.has(this.hold?.name)) {
      // steering: horizontal offset of the hand from where the walk began
      if (!valid) {
        this._steerInput = 0;
      } else {
        if (this._steerAnchorX === null) this._steerAnchorX = features.palmCenterNorm.x;
        this._steerInput = features.palmCenterNorm.x - this._steerAnchorX;
      }
      return;
    }
    if (this.hold?.name !== 'GRAB_ROTATE' || !this._grab || !valid) return;
    const { x, y } = features.palmCenterNorm;
    if (this._grab.anchorX === null) this._grab.anchorX = x;
    if (this._grab.anchorY === null) this._grab.anchorY = y;
    this.pose.yaw = this._grab.baseYaw + (x - this._grab.anchorX) * GRAB_ROTATE_GAIN;
    // marionette lift ("heavy lifting"): raising the pinched hand (y decreases) lifts the robot
    const lc = this.liftCfg;
    this.liftTarget = Math.min(lc.maxHeight, Math.max(0, this._grab.baseLift + (this._grab.anchorY - y) * lc.gain));
  }

  /** Advance timed actions, spin and walking. @param {number} t */
  tick(t) {
    this._advanceMotion(t);
    const a = this.action;
    if (a?.name === 'RESET_POSE' && this._resetFrom !== null) {
      const k = 1 - easeInOut(Math.min(1, (t - a.startedAt) / a.durationMs));
      this.pose.yaw = this._resetFrom.yaw * k;
      this.pose.x = this._resetFrom.x * k;
      this.pose.z = this._resetFrom.z * k;
      this.liftTarget = 0;
    }
    if (a && t - a.startedAt >= a.durationMs) {
      this.action = null;
      if (a.name === 'RESET_POSE') {
        this.pose.yaw = 0;
        this.pose.x = 0;
        this.pose.z = 0;
        this._resetFrom = null;
      }
      if (this.fsm.state === 'EXECUTING') this.fsm.transition(this._restingState(), t, `${a.name} complete`);
      this.lastResult = { name: a.name, source: a.source, outcome: 'completed', at: t, pose: { ...this.pose } };
      this._emit('command', { status: 'completed', name: a.name, source: a.source, t, pose: { ...this.pose } });
    }
  }

  /**
   * Spin and walk integration. dt comes from this tick clock only, is skipped when
   * the clock does not advance and capped at 100 ms, so frame rate and stalls never
   * change the rate per second.
   */
  _advanceMotion(t) {
    if (this._lastTick === null || !(t > this._lastTick)) {
      if (this._lastTick === null) this._lastTick = t;
      return;
    }
    const dt = Math.min(0.1, (t - this._lastTick) / 1000);
    this._lastTick = t;

    // ---- lift: follows the pinched hand quickly while held, descends smoothly otherwise
    // (release, lock, hand loss, error): never snaps, never stays up without a hold
    if (this.hold?.name !== 'GRAB_ROTATE') this.liftTarget = 0;
    const tau = this.hold?.name === 'GRAB_ROTATE' ? 90 : this.liftCfg.dropMs;
    this.pose.lift += (this.liftTarget - this.pose.lift) * (1 - Math.exp(-(dt * 1000) / tau));
    if (this.pose.lift < 1e-3) this.pose.lift = 0; // below 1 mm: settled

    // ---- spin: yaw rate follows the finger, rate-limited
    const sc = this.spinCfg;
    const targetRate = this.hold?.name === 'SPIN' ? spinRateFor(this.motionInput.omega, sc) : 0;
    const slowing = Math.abs(targetRate) < Math.abs(this.spinRate) || Math.sign(targetRate) !== Math.sign(this.spinRate);
    const prevRate = this.spinRate;
    this.spinRate = approach(this.spinRate, targetRate, (slowing ? sc.decelDegS2 : sc.accelDegS2) * DEG * dt);
    if (Math.abs(this.spinRate) < 1e-4) this.spinRate = 0;
    // trapezoid rule: exact for the rate-limited ramp, so the turned angle does not depend on frame rate
    const turned = ((prevRate + this.spinRate) / 2) * dt;
    if (turned !== 0) this.pose.yaw = wrapAngle(this.pose.yaw + turned);

    // ---- walk: speed follows the cadence, rate-limited; bounded area
    const wc = this.walkCfg;
    if (!LOCOMOTION.has(this.hold?.name)) {
      this.walkSpeed = 0;
      this.steerRate = 0;
      return;
    }
    // steering (smoothed toward the target rate, so a jerk of the hand never snaps the yaw)
    const targetSteer = steerRateFor(this._steerInput, this.steerCfg);
    const prevSteer = this.steerRate;
    this.steerRate += (targetSteer - this.steerRate) * (1 - Math.exp(-(dt * 1000) / 150));
    const steered = ((prevSteer + this.steerRate) / 2) * dt;
    if (steered !== 0 && this.walkSpeed > 0) this.pose.yaw = wrapAngle(this.pose.yaw + steered);
    const target = this.targetWalkSpeed();
    const prevSpeed = this.walkSpeed;
    this.walkSpeed = approach(this.walkSpeed, target, (target > this.walkSpeed ? wc.accel : wc.decel) * dt);
    const step = ((prevSpeed + this.walkSpeed) / 2) * dt;
    if (step <= 0) return;
    let x = this.pose.x + Math.sin(this.pose.yaw) * step;
    let z = this.pose.z + Math.cos(this.pose.yaw) * step;
    const r = Math.sqrt(x * x + z * z);
    this.atBoundary = r >= wc.maxDistance;
    if (r > wc.maxDistance) {
      x *= wc.maxDistance / r;
      z *= wc.maxDistance / r;
    }
    const moved = Math.sqrt((x - this.pose.x) ** 2 + (z - this.pose.z) ** 2);
    this.pose.x = x;
    this.pose.z = z;
    // one step (half a gait cycle) per strideLength actually travelled
    this.gaitPhase = (this.gaitPhase + (moved / wc.strideLength) * Math.PI) % (2 * Math.PI);
  }

  /**
   * Target floor speed (m/s) of the active locomotion hold: RUN = runSpeed, WALK from the
   * static V = speed, WALK from finger stepping (FINGER_WALK, if bound) = walkSpeedFor(cadence).
   */
  targetWalkSpeed() {
    const h = this.hold;
    const wc = this.walkCfg;
    if (!LOCOMOTION.has(h?.name)) return 0;
    if (h.name === 'RUN') return Math.min(wc.maxSpeed, wc.runSpeed);
    if (h.gesture === 'FINGER_WALK') return walkSpeedFor(this.motionInput.cadence, wc);
    return Math.min(wc.maxSpeed, wc.speed);
  }

  /** IDLE | WALKING | RUNNING | ROTATING | SQUATTING | LAUGHING | SAD | RETURNING, from the active command. */
  movement() {
    const a = this.action?.name;
    const deg = (r) => Math.round(r / DEG);
    if (LOCOMOTION.has(this.hold?.name)) {
      const turn = Math.round(this.steerRate / DEG);
      return { state: this.hold.name === 'RUN' ? 'RUNNING' : 'WALKING', direction: turn > 2 ? 'RIGHT' : turn < -2 ? 'LEFT' : 'FORWARD', speed: this.walkSpeed, cadence: this.motionInput.cadence, turnDegS: turn, atBoundary: this.atBoundary, gaitPhase: this.gaitPhase };
    }
    if (this.hold?.name === 'SPIN' || this.spinRate !== 0) {
      return { state: 'ROTATING', direction: this.spinRate < 0 ? 'LEFT' : this.spinRate > 0 ? 'RIGHT' : null, rateDegS: deg(this.spinRate), source: 'SPIN' };
    }
    if (a === 'SPIN_LEFT') return { state: 'ROTATING', direction: 'LEFT', rateDegS: null, source: 'SWIPE' };
    if (a === 'SPIN_RIGHT') return { state: 'ROTATING', direction: 'RIGHT', rateDegS: null, source: 'SWIPE' };
    if (this.hold?.name === 'GRAB_ROTATE') return { state: this.pose.lift > 0.03 ? 'LIFTING' : 'ROTATING', direction: 'MANUAL', rateDegS: null, source: 'PINCH', lift: this.pose.lift };
    if (a === 'SQUAT') return { state: 'SQUATTING', direction: null };
    if (a === 'LAUGH') return { state: 'LAUGHING', direction: null };
    if (a === 'SAD') return { state: 'SAD', direction: null };
    if (a === 'RESET_POSE') return { state: 'RETURNING', direction: 'HOME' };
    return { state: 'IDLE', direction: null };
  }

  /** Enter ERROR (camera/tracker failure). @param {string} reason @param {number} t */
  fail(reason, t) {
    this.error = reason;
    if (this.hold) this._endHold(t, 'error');
    this.action = null;
    this._resetFrom = null;
    this._stopMotion();
    this.fsm.transition('ERROR', t, reason);
  }

  /** Leave ERROR after a successful restart. @param {number} t */
  recover(t) {
    if (this.fsm.state !== 'ERROR') return false;
    this.error = null;
    this.handPresent = false;
    return this.fsm.transition('IDLE', t, 'recovered');
  }

  /** Read-only view for the scene and HUD. @param {number} t */
  snapshot(t) {
    return {
      state: this.fsm.state,
      stateSince: this.fsm.enteredAt,
      locked: this.locked,
      handPresent: this.handPresent,
      action: this.action
        ? { ...this.action, progress: Math.min(1, (t - this.action.startedAt) / this.action.durationMs) }
        : null,
      hold: this.hold ? { ...this.hold } : null,
      pose: { ...this.pose },
      movement: this.movement(),
      lastResult: this.lastResult ? { ...this.lastResult } : null,
      error: this.error,
      counts: { ...this.counts },
    };
  }

  _restingState() {
    return this.handPresent ? 'TRACKING' : 'IDLE';
  }

  _stopMotion() {
    this.spinRate = 0;
    this.walkSpeed = 0;
    this.steerRate = 0;
    this.liftTarget = 0; // the robot descends smoothly (tick), never drops instantly
    this.motionInput = { omega: 0, cadence: 0 };
  }

  /**
   * @param {number} t
   * @param {string} reason
   * @param {{ coast?: boolean }} [opts] coast: a released SPIN decelerates smoothly
   *        (decelDegS2); every other end (hand lost, lock, error, superseded) stops at once.
   *        WALK always stops at once: the robot never coasts forward.
   */
  _endHold(t, reason, opts = {}) {
    const h = this.hold;
    this.hold = null;
    this._grab = null;
    // a WALK / RUN released by its gesture may hand its speed to the next one (walk <-> run);
    // hand loss, lock, error and supersede stop for good
    this._carry = LOCOMOTION.has(h?.name) && opts.coast ? { speed: this.walkSpeed, t } : null;
    this.walkSpeed = 0;
    if (h?.name === 'SPIN' && !opts.coast) this.spinRate = 0;
    if (this.fsm.state === 'CONTROL') this.fsm.transition(this._restingState(), t, reason);
    if (h) {
      this.lastResult = { name: h.name, source: h.source, outcome: 'ended', reason, at: t, pose: { ...this.pose } };
      this._emit('command', { status: 'ended', name: h.name, source: h.source, reason, t, pose: { ...this.pose } });
    }
  }

  _cancelAction(t, reason) {
    const a = this.action;
    this.action = null;
    if (this.fsm.state === 'EXECUTING') this.fsm.transition(this._restingState(), t, `${a.name} cancelled`);
    this.lastResult = { name: a.name, source: a.source, outcome: 'cancelled', reason, at: t, pose: { ...this.pose } };
    this._emit('command', { status: 'cancelled', name: a.name, source: a.source, reason, t, pose: { ...this.pose } });
  }

  _reject(req, reason, t, detail) {
    this.counts.rejected++;
    const name = typeof req?.name === 'string' ? req.name : '?';
    this._emit('command', { status: 'rejected', name, source: req?.source ?? '?', reason, detail, t });
    return { accepted: false, reason, name };
  }

  _emit(type, payload) {
    this.bus?.emit(type, payload);
  }
}
