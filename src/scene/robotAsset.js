/**
 * Resolves which robot model to show. Kept free of three.js imports so it
 * can be unit-tested: the actual GLTF loading function is injected.
 *
 * Result:
 *   { kind: 'gltf', gltf, url }            model loaded
 *   { kind: 'procedural', reason }         no URL configured, or loading failed
 *
 * Loading failures never throw: the procedural robot is always available
 * as a fallback, and `reason` is shown in the HUD.
 */
export async function resolveRobotAsset(url, { loadGLTF, timeoutMs = 15000 }) {
  if (!url) return { kind: 'procedural', reason: 'no model configured (procedural robot)' };
  let timer;
  try {
    const gltf = await Promise.race([
      loadGLTF(url),
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`timed out after ${timeoutMs} ms`)), timeoutMs);
      }),
    ]);
    if (!gltf || !gltf.scene) throw new Error('file contains no scene');
    return { kind: 'gltf', gltf, url };
  } catch (err) {
    return { kind: 'procedural', reason: `failed to load ${url}: ${err?.message ?? err}; using procedural robot` };
  } finally {
    clearTimeout(timer);
  }
}
