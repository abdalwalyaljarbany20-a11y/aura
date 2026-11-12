import { StateMachine } from '../core/StateMachine.js';

/**
 * Robot interaction states (all implemented):
 *
 *   IDLE      no hand is tracked; robot idles
 *   TRACKING  a hand is tracked; robot looks at it; ready for commands
 *   CONTROL   a hold command (GRAB_ROTATE, AIM) is active
 *   EXECUTING a timed action (WAVE, SPIN_*, RESET_POSE) is animating
 *   LOCKED    gesture commands are disabled except TOGGLE_LOCK
 *   ERROR     camera or tracker failure; cleared only by a successful restart
 *
 * Transition table (anything not listed is rejected):
 */
export const ROBOT_TRANSITIONS = Object.freeze({
  IDLE: ['TRACKING', 'EXECUTING', 'LOCKED', 'ERROR'],
  TRACKING: ['IDLE', 'CONTROL', 'EXECUTING', 'LOCKED', 'ERROR'],
  CONTROL: ['TRACKING', 'IDLE', 'ERROR'],
  EXECUTING: ['TRACKING', 'IDLE', 'ERROR'],
  LOCKED: ['IDLE', 'TRACKING', 'ERROR'],
  ERROR: ['IDLE'],
});

export const ROBOT_STATES = Object.freeze(Object.keys(ROBOT_TRANSITIONS));

/** @param {(from: string, to: string, reason: string) => void} [onChange] */
export function createRobotStateMachine(onChange) {
  return new StateMachine(ROBOT_TRANSITIONS, 'IDLE', onChange);
}
