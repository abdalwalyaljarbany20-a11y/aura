# AI agent: setup and use

The agent lets you type requests such as "wave", "spin left" or "what is the robot doing?" in the **Agent** panel. A Claude model chooses one of the app's allowlisted actions. The app validates that choice and runs it through the same task path as gestures and keys. You only see "succeeded" after the action really completed.

## 1. Configure the credential (server side only)

```powershell
copy .env.example .env.local
notepad .env.local        # set ANTHROPIC_API_KEY=...
```

`.env.local` is git-ignored. Variables can also come from the shell (`$env:ANTHROPIC_API_KEY = "..."`), and the shell takes precedence over files. The key is read only by the local Vite server process; it is never sent to the browser, put in a URL, or included in the built bundle (verified by building with a test key and scanning `dist/`).

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | (none) | Claude API key. Without it, the panel shows "Unavailable" and the rest of the app works normally. |
| `AURA_AGENT_PROVIDER` | `anthropic` | Provider adapter. `anthropic` is the only one implemented. |
| `AURA_AGENT_MODEL` | `claude-opus-5-5` | Model id |
| `AURA_AGENT_EFFORT` | `low` | Thinking effort: `low` / `medium` / `high` / `xhigh` / `max`. `low` suits short tool-selection turns. |
| `AURA_AGENT_TIMEOUT_MS` | `30000` | Per provider request. One automatic retry; a whole request (all model calls) is capped at 90 s. |
| `AURA_AGENT_MAX_TOKENS` | `16000` | Per response; thinking counts toward it. |
| `AURA_AGENT_BASE_URL` | SDK default | Optional API gateway URL |
| `AURA_AGENT_REFUSAL_FALLBACK` | `off` | `default` opts into Anthropic's server-side refusal fallback (see below) |

What the adapter sends:
- **Tool choice:** `tool_choice: auto` (at most one call per turn) with strict tool schemas. Forced tool choice is not supported on current models.
- **Thinking:** adaptive thinking with `effort`, and `thinking.block_binding.prefix_mismatch_behavior: "drop_block"` (beta `thinking-binding-controls-2026-08-01`).
- **Strict schemas:** `strict: true` is set only on tools that have parameters (`robot_command`, `run_workflow`). Strict support for an empty object schema is not documented, so the parameter-less read tools don't use it. Server and browser validate every call either way.
- **Refusal fallback: off by default.** With `AURA_AGENT_REFUSAL_FALLBACK=default`, the request carries `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`), and only for models that offer it (not Haiku 5.5). If a safety classifier then declines a request, the API re-runs it on a recommended fallback model inside the same call. Per the official docs, only a classifier decline triggers it: rate limits, overload and server errors are returned as-is. The fallback run is billed at the fallback model's rates, and the declined attempt is also billed in some categories (mid-stream declines, or `bio` / `frontier_llm` / `reasoning_extraction`). Without the fallback, a decline is shown in the panel as "The model declined this request."

Audit of the defaults (2026-10-22, against the official docs): `claude-opus-5-5` is a current model (128K max output); `output_config.effort: "low"` is valid for it (its API default is `medium`); `max_tokens` 16000 is within limits; `tool_choice: auto` with `disable_parallel_tool_use` is used because forced tool choice is not supported on this model. The SDK is pinned at `@anthropic-ai/sdk@0.133.0`, whose types include every field sent.

## 2. Start

```powershell
npm run dev            # browser vision engine + agent -> http://127.0.0.1:5173
npm run dev:python     # Python vision engine + agent  -> http://127.0.0.1:5173/?engine=python
```

Click **Agent** in the top bar.

- **Ready** with the model name means the agent is configured.
- **Unavailable — …** gives the reason, for example a missing key.

The agent is independent of the vision engine and the camera: if the provider fails, tracking and gestures keep working.

## 3. What the agent can do

| Tool (model-facing) | Task | Arguments | Policy |
|---|---|---|---|
| `robot_command` | `robot.command` | `{ command: WAVE \| SPIN_LEFT \| SPIN_RIGHT \| RESET_POSE }` | runs directly (short, reversible animations) |
| `robot_command` | `robot.command` | `{ command: TOGGLE_LOCK }` | **needs your confirmation** (it switches gesture control off or on) |
| `app_status` | `app.status` | `{}` | runs directly (read-only live state) |
| `app_diagnostics` | `app.diagnostics` | `{}` | runs directly (read-only troubleshooting snapshot) |
| `run_workflow` | several tasks (see below) | `{ workflow: "recover_robot" }` or `{ workflow: "checked_command", command }` | the workflow runs directly; its consequential steps (TOGGLE_LOCK) ask you to confirm |

That is the complete list. The workflows are described in [AGENT_INTERFACE.md](AGENT_INTERFACE.md#workflows). The agent cannot see camera frames, change gesture thresholds, switch the vision engine, open or close the camera, touch files, run code, or browse the web. If you ask for something else, it says so and suggests the closest supported action. Hold gestures (`GRAB_ROTATE`, `AIM`) need a live hand and are not offered to the agent.

## 4. Using the panel

| Control | Behaviour |
|---|---|
| Enter / **Send** | Send. Disabled while a request runs, so a request can't be submitted twice. |
| Shift+Enter | New line |
| Esc / **Stop** | Abandon a request that is still planning. Nothing executes, and a late answer is ignored. A robot action that has already started cannot be cancelled; it finishes on its own (at most 1.8 s). |
| **Confirm** / **Cancel** | Shown only for confirmation-required actions. They refer to the exact action displayed and expire after 30 s. |
| **clear** | Forgets the conversation in the browser and on the server. |

Keys typed in the panel never trigger the app shortcuts (L, W, R, H, E, Esc).

Each proposed action appears as a card labelled **App**, and its status chip comes from the task record itself: `executing`, `succeeded`, `failed`, `rejected`, `declined`, `expired` or `awaiting confirmation`.

A workflow card lists every step with its own status, then the app's summary line:
- "Completed and verified …";
- "Completed, not verified …" when the action ran but its effect was not observed;
- "Stopped: …" with the reason.

Lines labelled **AURA · model** are the model's explanation. If the two ever disagree, the App card is authoritative. While a workflow runs, the phase line names the current step. The HUD shows **Agent** (phase or step) and **AI agent** (ready / unavailable).

Conversations are kept in memory only: up to 60 provider messages per conversation, 20 conversations, and a 30-minute idle expiry. They are lost when the server restarts and are never written to disk.

## 5. Tests

```powershell
npm test                 # JS suite incl. agent tests (MOCKED provider, no network, no credit)
npm run test:all         # JS + Python
node scripts/integration-check.mjs --agent-mock          # real browser UI, MOCK provider (keyword rules, not AI)
node scripts/integration-check.mjs --agent-unavailable   # panel without credentials
node scripts/integration-check.mjs --engine python --real-camera --agent-mock --agent-diagnose-only --seconds 30   # PHYSICAL camera: visible window, Stop button, no images saved
node scripts/integration-check.mjs --browser-real-camera --agent-mock --agent-diagnose-only --seconds 30           # PHYSICAL camera: visible window, Stop button, no images saved
npm run agent:smoke      # LIVE: one real request to your configured model (costs a little credit)
```

The integration check removes `ANTHROPIC_API_KEY` from the server it starts unless you pass `--agent-live`, so ordinary checks never make paid calls.

## 6. Troubleshooting

| Panel shows | Fix |
|---|---|
| Unavailable — no provider credential | Set `ANTHROPIC_API_KEY` in `.env.local` and restart `npm run dev` |
| the AI provider rejected the credential | The key is wrong or revoked |
| the configured model was not found | Check `AURA_AGENT_MODEL` |
| rate-limiting… retry in about N s / overloaded | Wait and send again (nothing was executed) |
| the agent server is not reachable | The page is served by something other than `npm run dev` / `npm run preview` (the API lives in that server) |
| the agent is still working on the previous request | A cancelled request is still finishing on the server; send again shortly |
| "Executing…" right after **clear** | A robot action that was already dispatched is finishing (at most about 2 s); the panel accepts messages again afterwards |
