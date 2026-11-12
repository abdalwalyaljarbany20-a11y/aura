import { defineConfig, loadEnv } from 'vite';
import { auraAgentPlugin } from './server/agent/http.js';

// The dev server binds to localhost only: the camera stream is processed
// in the page and never exposed on the network. getUserMedia requires a
// secure context, which http://localhost satisfies.
//
// The AI agent API (/api/agent/*) runs inside this same server process. Its
// configuration (ANTHROPIC_API_KEY, AURA_AGENT_*) is read here, server side,
// from the shell environment or .env / .env.local; Vite only exposes
// VITE_-prefixed variables to browser code, so none of these reach the page.
export default defineConfig(({ mode }) => {
  const fileEnv = loadEnv(mode, process.cwd(), ['ANTHROPIC_', 'AURA_AGENT_']);
  const env = { ...fileEnv, ...process.env };
  return {
    plugins: [auraAgentPlugin(env)],
    server: { host: '127.0.0.1', port: 5173, strictPort: false },
    preview: { host: '127.0.0.1', port: 4173 },
    build: { target: 'es2022', chunkSizeWarningLimit: 1500 },
  };
});
