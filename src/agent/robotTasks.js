import { COMMANDS } from '../interaction/commands.js';
import { TaskError } from './TaskBroker.js';

/**
 * `robot.command`: run one existing robot command through the same validated
 * path as gestures and keys (InteractionController.submit, source 'agent').
 *
 *   args    { command: 'WAVE' | 'SPIN_LEFT' | 'SPIN_RIGHT' | 'RESET_POSE' | 'TOGGLE_LOCK' }
 *   result  { command, state, pose, completedAt }   only after the controller reports
 *           the action as completed (or the toggle as applied)
 *   errors  controller rejection codes (LOCKED, BUSY, CONFLICT, COOLDOWN, ERROR_STATE, ...),
 *           INTERRUPTED (robot entered ERROR before completion), TIMEOUT
 *
 * Hold commands (GRAB_ROTATE, AIM) need a live hand and are not available as tasks.
 */
export function registerRobotTasks(broker, { controller, bus, now = () => performance.now() }) {
  const allowed = Object.entries(COMMANDS).filter(([, d]) => d.kind !== 'hold').map(([n]) => n);
  broker.register('robot.command', {
    description: `Run a robot command: ${allowed.join(', ')}`,
    timeoutMs: 5000,
    validateArgs(args) {
      const keys = Object.keys(args);
      if (keys.length !== 1 || keys[0] !== 'command') return 'args must be exactly { command }';
      if (!allowed.includes(args.command)) return `command must be one of ${allowed.join(', ')}`;
      return null;
    },
    execute({ command }) {
      return new Promise((resolve, reject) => {
        const off = [];
        const cleanup = () => off.forEach((f) => f());
        // Subscribe before submitting so a synchronous outcome cannot be missed.
        off.push(bus.on('command', (e) => {
          if (e.name === command && e.source === 'agent' && e.status === 'completed') {
            cleanup();
            resolve({ command, state: controller.state, pose: { ...controller.pose }, completedAt: e.t });
          }
        }));
        off.push(bus.on('state', (e) => {
          if (e.to === 'ERROR') {
            cleanup();
            reject(new TaskError('INTERRUPTED', `robot entered ERROR: ${e.reason}`));
          }
        }));
        const r = controller.submit({ name: command, source: 'agent' }, now());
        if (!r.accepted) {
          cleanup();
          reject(new TaskError(r.reason, `command ${command} rejected: ${r.reason}`));
          return;
        }
        if (COMMANDS[command].kind === 'toggle') {
          cleanup();
          resolve({ command, state: controller.state, pose: { ...controller.pose }, completedAt: now() });
        }
      });
    },
  });
}
