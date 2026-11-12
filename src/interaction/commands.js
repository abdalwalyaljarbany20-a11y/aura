/**
 * Command registry: the single source of truth for which commands exist.
 *
 * kind:
 *   'action' - a timed robot animation; the robot is EXECUTING for durationMs
 *              and other actions are rejected meanwhile (BUSY).
 *   'hold'   - a continuous control active while its gesture is held
 *              (robot state CONTROL); ends when the gesture ends.
 *   'toggle' - flips a mode flag (only LOCK exists today).
 *
 * Optional fields:
 *   supersedes  hold names this command may end when it arrives (SPIN ends the AIM
 *               hold that its own pointing shape started)
 *   interruptible  the user's safety lock may cancel this running action (emotions)
 *   cooldownMs  per-command cooldown instead of interaction.cooldownMs
 *   gestureOnly not offered to the AI agent (agentTools.js)
 *   motion      the hold moves the robot; the user's safety lock stops it
 *
 * Every command here is implemented by the robot/scene as an animation or
 * pose change. None of them performs an external task. Adding a command =
 * add an entry here + handle `command.name` in Scene3D (and optionally bind
 * it to a gesture in the configuration).
 */
export const COMMANDS = Object.freeze({
  WAVE: { kind: 'action', durationMs: 1800, description: 'Robot waves its right arm' },
  SPIN_LEFT: { kind: 'action', durationMs: 1100, description: 'Robot turns a full circle to the left' },
  SPIN_RIGHT: { kind: 'action', durationMs: 1100, description: 'Robot turns a full circle to the right' },
  RESET_POSE: { kind: 'action', durationMs: 700, description: 'Robot returns to its default orientation and home position' },
  LAUGH: { kind: 'action', durationMs: 2200, gestureOnly: true, interruptible: true, description: 'Robot laughs: happy eyes, smile, bouncing body (thumb up)' },
  SAD: { kind: 'action', durationMs: 2600, gestureOnly: true, interruptible: true, description: 'Robot is sad: head down, drooping eyes, slumped body (thumb down)' },
  GRAB_ROTATE: { kind: 'hold', description: 'While pinching, horizontal hand motion rotates the robot' },
  AIM: { kind: 'hold', description: 'While pointing, the robot looks and aims at the index fingertip' },
  WALK: { kind: 'hold', motion: true, description: 'While index and middle finger are raised (V), the robot walks forward; move the hand sideways to steer' },
  RUN: { kind: 'hold', motion: true, gestureOnly: true, description: 'While index, middle and ring finger are raised, the robot runs; move the hand sideways to steer' },
  SQUAT: { kind: 'action', durationMs: 900, cooldownMs: 0, gestureOnly: true, interruptible: true, description: 'Robot does one squat: sits down and stands up (touch and release thumb and index tips)' },
  SPIN: { kind: 'hold', motion: true, supersedes: ['AIM'], description: 'While the index finger circles, the robot rotates at a rate that follows the finger' },
  TOGGLE_LOCK: { kind: 'toggle', description: 'Lock/unlock gesture commands (safety clutch)' },
});

export const COMMAND_NAMES = Object.freeze(Object.keys(COMMANDS));

/** Allowed command sources. An AI agent integration would use 'agent'. */
export const COMMAND_SOURCES = Object.freeze(['gesture', 'keyboard', 'agent', 'test']);

/**
 * Validate the shape of a command request coming from any source,
 * including untrusted ones (e.g. a future agent bridge).
 * @param {unknown} req
 * @returns {string|null} error, or null if well-formed
 */
export function commandRequestError(req) {
  if (!req || typeof req !== 'object' || Array.isArray(req)) return 'command request must be an object';
  const { name, source, params } = /** @type {any} */ (req);
  if (typeof name !== 'string' || !/^[A-Z][A-Z0-9_]{0,63}$/.test(name)) return 'command name must be UPPER_SNAKE_CASE';
  if (!COMMAND_SOURCES.includes(source)) return `command source must be one of ${COMMAND_SOURCES.join(', ')}`;
  if (params !== undefined && (params === null || typeof params !== 'object' || Array.isArray(params))) {
    return 'command params must be an object';
  }
  return null;
}
