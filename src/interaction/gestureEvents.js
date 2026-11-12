import { CONTINUOUS_GESTURES, DYNAMIC_GESTURES, STATIC_GESTURES } from '../config/config.js';

/**
 * Canonical gesture-event contract shared by both CV engines.
 *
 * Produced by the JS GestureRecognitionEngine (browser mode) and by
 * aura_cv.gestures (Python mode, delivered in `result.events`). Consumed only
 * by InteractionController.handleGestureEvents, which validates every event
 * before it can change application state.
 *
 *   { type: 'start', gesture: STATIC_GESTURE }   a static gesture became stable (once per hold)
 *   { type: 'end',   gesture: STATIC_GESTURE }   that stable gesture was released
 *   { type: 'event', gesture: DYNAMIC_GESTURE }  a dynamic gesture (swipe) occurred
 *   { type: 'start'|'end', gesture: CONTINUOUS_GESTURE }  a motion gesture (SPIN,
 *       FINGER_WALK) began / ended; its control signal travels in gesture.motion
 *
 * Optional fields: t (ms, producer clock — informational only), confidence
 * (rule score or null), detail (object), durationMs (end events).
 * NONE / UNKNOWN / MOVING are classifier labels, never events.
 */
export const GESTURE_EVENT_TYPES = Object.freeze(['start', 'end', 'event']);

/** @returns {string|null} why `ev` is not a valid canonical gesture event */
export function gestureEventError(ev) {
  if (!ev || typeof ev !== 'object' || Array.isArray(ev)) return 'event must be an object';
  if (!GESTURE_EVENT_TYPES.includes(ev.type)) return `unknown event type ${JSON.stringify(ev.type)}`;
  if (typeof ev.gesture !== 'string') return 'gesture must be a string';
  if (ev.type === 'event') {
    if (!DYNAMIC_GESTURES.includes(ev.gesture)) return `${ev.gesture} is not a dynamic gesture`;
  } else if (!STATIC_GESTURES.includes(ev.gesture) && !CONTINUOUS_GESTURES.includes(ev.gesture)) {
    return `${ev.gesture} is not a static or continuous gesture`;
  }
  if (ev.confidence !== undefined && ev.confidence !== null && !(typeof ev.confidence === 'number' && ev.confidence >= 0 && ev.confidence <= 1)) {
    return 'confidence must be null or a number in [0, 1]';
  }
  if (ev.t !== undefined && !(typeof ev.t === 'number' && Number.isFinite(ev.t))) return 't must be a finite number';
  return null;
}
