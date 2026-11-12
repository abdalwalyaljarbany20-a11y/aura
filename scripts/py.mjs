#!/usr/bin/env node
/**
 * Runs the project's virtual-environment Python inside backend/.
 *
 *   node scripts/py.mjs setup          create .venv and install backend/requirements-dev.txt
 *   node scripts/py.mjs <python args>  e.g. "-m aura_cv", "-m pytest"
 *
 * The global Python installation is never modified.
 */
import { spawn, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const VENV_PY = process.platform === 'win32' ? join(ROOT, '.venv', 'Scripts', 'python.exe') : join(ROOT, '.venv', 'bin', 'python');
const args = process.argv.slice(2);

if (args[0] === 'setup') {
  if (!existsSync(VENV_PY)) {
    // Windows: the py launcher; elsewhere python3. Python 3.12-3.14 have been used with these pins.
    const [cmd, ...pre] = process.platform === 'win32' ? ['py', '-3'] : ['python3'];
    console.log(`[py] creating .venv with ${cmd} ${pre.join(' ')}`);
    const r = spawnSync(cmd, [...pre, '-m', 'venv', join(ROOT, '.venv')], { stdio: 'inherit' });
    if (r.status !== 0) process.exit(r.status ?? 1);
  }
  const r = spawnSync(VENV_PY, ['-m', 'pip', 'install', '--disable-pip-version-check', '-r', join(ROOT, 'backend', 'requirements-dev.txt')], { stdio: 'inherit' });
  process.exit(r.status ?? 1);
}

if (!existsSync(VENV_PY)) {
  console.error('[py] .venv not found. Run "npm run setup:python" first.');
  process.exit(1);
}
const child = spawn(VENV_PY, args, { cwd: join(ROOT, 'backend'), stdio: 'inherit' });
const forward = (sig) => () => child.kill(sig);
process.on('SIGINT', forward('SIGINT'));
process.on('SIGTERM', forward('SIGTERM'));
child.on('exit', (code) => process.exit(code ?? 0));
