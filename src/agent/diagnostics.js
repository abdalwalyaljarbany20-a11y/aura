/**
 * Diagnostics helpers (pure; no DOM). Everything here is derived from real
 * application state or real events; nothing is estimated or invented.
 */

/** Make an internal message safe to show the model/user: no paths, no key-like strings, bounded. */
export function sanitizeMessage(text) {
  if (text === null || text === undefined) return null;
  return String(text)
    .replace(/sk-ant-[A-Za-z0-9_-]+/g, '[redacted]')
    .replace(/[A-Za-z]:\\[^\s"')]+/g, '<path>') // Windows paths
    .replace(/(^|[\s(])\/(?:[\w.-]+\/)+[\w.-]+/g, '$1<path>') // POSIX-like paths
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 140);
}

/** Bounded log of recent operational issues (from real bus events). */
export class IssueLog {
  constructor(max = 8, now = () => performance.now()) {
    this.max = max;
    this.now = now;
    this.items = [];
  }

  record(source, code, message) {
    this.items.push({ at: this.now(), source, code: String(code ?? 'UNKNOWN').slice(0, 40), message: sanitizeMessage(message) });
    if (this.items.length > this.max) this.items.splice(0, this.items.length - this.max);
  }

  /** Most recent first, with age instead of an absolute clock. */
  recent(n = 5) {
    const t = this.now();
    return this.items.slice(-n).reverse().map((i) => ({ secondsAgo: Math.round((t - i.at) / 1000), source: i.source, code: i.code, message: i.message }));
  }
}

/**
 * What currently stops robot commands, from the controller snapshot and the
 * camera status. Each blocker names what it affects; a camera that is not live
 * blocks gestures, not agent-requested robot commands.
 * @param {ReturnType<import('../interaction/InteractionController.js').InteractionController['snapshot']>} s
 * @param {string} cameraStatus
 */
export function deriveBlockers(s, cameraStatus) {
  const out = [];
  if (s.state === 'ERROR') {
    out.push({ code: 'ROBOT_ERROR', detail: sanitizeMessage(s.error) ?? 'robot is in ERROR', affects: 'all robot commands until the camera/engine is restarted from the app' });
  }
  if (s.state === 'LOCKED') out.push({ code: 'GESTURE_LOCK_ON', detail: 'gesture lock is on', affects: 'all robot commands except TOGGLE_LOCK' });
  if (s.state === 'CONTROL' && s.hold) {
    out.push({ code: 'HOLD_ACTIVE', detail: `${s.hold.name} is active${s.hold.gesture ? ` (held ${s.hold.gesture} gesture)` : ''}`, affects: 'robot commands until the gesture is released' });
  }
  if (s.state === 'EXECUTING' && s.action) {
    out.push({ code: 'ACTION_RUNNING', detail: `${s.action.name} is running`, affects: 'other robot commands until it finishes' });
  }
  if (cameraStatus !== 'live') out.push({ code: 'CAMERA_NOT_LIVE', detail: `camera is ${cameraStatus}`, affects: 'gesture control (robot commands from the agent still work)' });
  return out;
}
