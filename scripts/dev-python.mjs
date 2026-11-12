#!/usr/bin/env node
/**
 * Starts the Python CV engine and the Vite dev server together, then prints the
 * URL that selects the Python engine. Ctrl+C stops both (the engine releases the
 * camera on shutdown).
 *
 *   npm run dev:python [-- --camera 1]
 */
import { spawn } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const extra = process.argv.slice(2);
const children = [
  spawn(process.execPath, [join(ROOT, 'scripts', 'py.mjs'), '-m', 'aura_cv', ...extra], { cwd: ROOT, stdio: 'inherit' }),
  spawn(process.execPath, [join(ROOT, 'node_modules', 'vite', 'bin', 'vite.js')], { cwd: ROOT, stdio: 'inherit' }),
];
console.log('\n  AURA VISION (Python engine): open http://127.0.0.1:5173/?engine=python\n');
const stopAll = () => {
  for (const c of children) if (c.exitCode === null) c.kill('SIGINT');
};
process.on('SIGINT', stopAll);
process.on('SIGTERM', stopAll);
for (const c of children) c.on('exit', stopAll);
