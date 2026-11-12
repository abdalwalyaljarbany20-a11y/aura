import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fetchConfigOverrides, loadConfig } from '../src/config/config.js';
import { DEFAULT_CONFIG } from '../src/config/defaultConfig.js';
import { COMMAND_NAMES } from '../src/interaction/commands.js';

test('no overrides yields the defaults without errors', () => {
  const { config, errors, warnings } = loadConfig(undefined, { knownCommands: COMMAND_NAMES });
  assert.deepEqual(config, JSON.parse(JSON.stringify(DEFAULT_CONFIG)));
  assert.deepEqual(errors, []);
  assert.deepEqual(warnings, []);
});

test('valid overrides are merged deeply', () => {
  const { config, errors } = loadConfig({
    camera: { width: 640, mirror: false },
    tracking: { delegate: 'CPU' },
    gestures: { swipe: { minDistance: 3 }, definitions: { FIST: { holdMs: 900, enabled: false } } },
    interaction: { bindings: { OPEN_PALM: null } },
  });
  assert.deepEqual(errors, []);
  assert.equal(config.camera.width, 640);
  assert.equal(config.camera.height, DEFAULT_CONFIG.camera.height);
  assert.equal(config.camera.mirror, false);
  assert.equal(config.tracking.delegate, 'CPU');
  assert.equal(config.gestures.swipe.minDistance, 3);
  assert.equal(config.gestures.swipe.windowMs, DEFAULT_CONFIG.gestures.swipe.windowMs);
  assert.equal(config.gestures.definitions.FIST.holdMs, 900);
  assert.equal(config.gestures.definitions.FIST.enabled, false);
  assert.equal(config.interaction.bindings.OPEN_PALM, null);
  assert.equal(config.interaction.bindings.FIST, 'TOGGLE_LOCK');
});

test('invalid values are rejected individually and the default kept', () => {
  const { config, errors } = loadConfig({
    camera: { width: -5, frameRate: 'fast', bogus: 1 },
    tracking: { delegate: 'TPU', numHands: 2.5 },
    gestures: { minScore: 2, definitions: { WAVE: {}, FIST: { holdMs: -1, color: 'red' } } },
    interaction: { bindings: { SNAP: 'WAVE', POINT: 'lowercase' } },
    effects: { trailFingertips: [4, 99] },
    unknownSection: {},
  });
  assert.equal(config.camera.width, DEFAULT_CONFIG.camera.width);
  assert.equal(config.tracking.delegate, 'GPU');
  assert.equal(config.gestures.minScore, DEFAULT_CONFIG.gestures.minScore);
  assert.equal(config.interaction.bindings.POINT, 'AIM');
  for (const fragment of [
    'camera.width', 'camera.frameRate', 'camera.bogus', 'tracking.delegate', 'tracking.numHands',
    'gestures.minScore', 'definitions.WAVE', 'FIST.holdMs', 'FIST.color', 'bindings.SNAP', 'bindings.POINT',
    'trailFingertips', 'unknownSection',
  ]) {
    assert.ok(errors.some((e) => e.includes(fragment)), `expected an error mentioning ${fragment}: ${errors.join(' | ')}`);
  }
});

test('non-object root is rejected', () => {
  assert.equal(loadConfig([1, 2]).errors.length, 1);
  assert.equal(loadConfig('x').errors.length, 1);
});

test('bindings to unimplemented commands produce a warning', () => {
  const { warnings } = loadConfig({ interaction: { bindings: { FIST: 'SHUTDOWN_PC' } } }, { knownCommands: COMMAND_NAMES });
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /SHUTDOWN_PC/);
});

test('defaults are not mutated by loading', () => {
  loadConfig({ camera: { width: 320 } });
  assert.equal(DEFAULT_CONFIG.camera.width, 1280);
});

test('example config file in public/ is valid', async () => {
  const { readFile } = await import('node:fs/promises');
  const text = await readFile(new URL('../public/aura.config.example.json', import.meta.url), 'utf8');
  const { errors } = loadConfig(JSON.parse(text), { knownCommands: COMMAND_NAMES });
  assert.deepEqual(errors, []);
});

test('fetchConfigOverrides handles missing, HTML fallback, invalid and valid files', async () => {
  const res = (status, body) => async () => ({ status, ok: status >= 200 && status < 300, text: async () => body });
  assert.equal((await fetchConfigOverrides('/c.json', res(404, ''))).error, null);
  assert.equal((await fetchConfigOverrides('/c.json', res(200, '<!doctype html>'))).overrides, undefined);
  assert.match((await fetchConfigOverrides('/c.json', res(200, '{bad'))).error, /not valid JSON/);
  assert.deepEqual((await fetchConfigOverrides('/c.json', res(200, '{"camera":{"width":640}}'))).overrides, { camera: { width: 640 } });
  const failing = async () => {
    throw new Error('offline');
  };
  assert.match((await fetchConfigOverrides('/c.json', failing)).error, /offline/);
});
