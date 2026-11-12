# AI-agent integration

Phase 2 prepared the task boundary; Phase 3 connected a real model-provider adapter and the agent loop to it; Phase 4 added read-only diagnostics and two deterministic workflows. Setup and use: [AGENT_SETUP.md](AGENT_SETUP.md).

## Concepts (kept distinct)

| Concept | Produced by | Representation | Can it trigger work? |
|---|---|---|---|
| **Gesture event** | a vision engine | `{ type: 'start'|'end'|'event', gesture }` (`src/interaction/gestureEvents.js`) | Only its bound robot command, via `InteractionController`. **Never a task, never the agent.** |
| **Command request** | gesture binding, keyboard, task executor | `{ name, source, params }` + controller validation | Yes: one registered robot command |
| **Proposed tool call** | the model (untrusted) | `{ name, input }` | No. It is validated against the allowlist (`src/agent/agentTools.js`) on the server **and** in the browser, then becomes a task request. |
| **Task request** | the agent orchestrator, other origins | `{ id?, type, args, origin }` (`taskRequestError`) | Yes: one registered task type |
| **Task record** | `TaskBroker` | `{ id, type, args, origin, status, result, error, … }` | n/a: this is the real outcome |
| **Tool outcome** | browser → server | `{ status, result?, error?, taskId? }` (`toolOutcomeError`), built from the task record | n/a: the model only explains it |

## Request flow

```
Agent panel (src/agent/AgentPanel.js)                          DOM only, textContent rendering
  └ AgentOrchestrator (src/agent/AgentOrchestrator.js)         phases, epochs, confirmation, execution
      └ AgentClient ──HTTP──► /api/agent/* (server/agent/http.js, inside the Vite dev/preview server)
                                 └ AgentService (server/agent/agentService.js)   conversations, bounded model loop
                                     └ AnthropicProvider (server/agent/anthropicProvider.js) ──► Claude Messages API
      ◄── step: reply | tool_call { tool, args, policy } (already allowlist-validated)
      ├ re-validate (agentTools.validateToolCall)
      ├ policy 'confirm' → wait for a trusted click on this exact proposal (30 s expiry; invalid if the lock state changed)
      └ TaskBroker.submit({ type, args, origin: 'agent' }) → robotTasks / appStatusTask → InteractionController
          → task record → tool outcome ──HTTP──► model explains → reply
```

## Guarantees and where they are enforced

| Guarantee | Enforcement | Test |
|---|---|---|
| The key never reaches the browser | Read server-side from env; `/status` returns no key; no `VITE_` prefix | `agent-provider.test.js`, `agent-integration.test.js`; bundle scan after building with a test key |
| Only allowlisted tools with valid args | `validateToolCall` on the server (invalid calls go back to the model as errors and never leave the server) and in the browser | `agent-tools.test.js`, `agent-service.test.js` |
| No bypass of the task broker | The orchestrator's only execution path is `tasks.submit`; tools are task types | `agent-integration.test.js` |
| Success only after real completion | `robot.command` resolves on the controller's `completed` event; the UI chip shows the record status | integration test "success reported only after completion" |
| Real failures reported as failures | Controller rejection → task `failed` → outcome `failed` → chip `failed` | "robot locked by the user", CONFLICT runs in the browser UI |
| Confirmation only from the user | `confirm(proposalId)` from a trusted click on the exact proposal; 30 s expiry; invalid if the lock state changed; a new message supersedes it | "confirmation required", "stale confirmations" |
| Model text can't confirm or forge results | Outcomes enter the conversation only via `/tool-results` matched to the pending call id | "stale or mismatched tool results are rejected" |
| No duplicate execution | One turn at a time in the browser; per-conversation `BUSY`; `requestId` idempotency on the server | "duplicate requestId", "duplicate submit refused" |
| Late responses change nothing | Epoch + requestId checks in the orchestrator; server rolls back aborted requests | "cancel and clear" |
| Provider failure is isolated | Errors become typed messages in the panel; vision, gestures and rendering continue | "provider outage", browser UI runs |
| Cross-site requests can't spend credit | Localhost Host only, same-origin, `X-Aura-Agent` header, JSON content type, 16 KB bodies | "request guard" |

## Tools and tasks (exact)

| Tool (model) | Task (broker) | Input schema | Policy | Result |
|---|---|---|---|---|
| `robot_command` | `robot.command` | `{ command: "WAVE"\|"SPIN_LEFT"\|"SPIN_RIGHT"\|"RESET_POSE"\|"TOGGLE_LOCK" }`, strict | TOGGLE_LOCK: confirm; others: auto | `{ command, state, pose, completedAt }` only after the controller reports completion |
| `app_status` | `app.status` | `{}` | auto (read-only) | `{ visionEngine, camera, tracking, handsVisible, stableGesture, robot{ state, gestureLock, yawDegrees, runningAction, activeControl, lastFinishedCommand, error } }` |
| `app_diagnostics` | `app.diagnostics` | `{}` | auto (read-only) | see below |
| `run_workflow` | each step is its own task | `{ workflow: "recover_robot" }` or `{ workflow: "checked_command", command }`, strict | the workflow runs directly; TOGGLE_LOCK steps confirm | compact report: `{ workflow, outcome, verified, summary, steps[{ id, status, error?, note? }] }` |

### `app.diagnostics` data sources

| Field | Source | Unknown when |
|---|---|---|
| `visionEngine.{mode, connection, engineVersion}` | engine selection; Python client connection state and hello | `engineVersion` before the Python hello |
| `camera.{status, errorCode}` | the active camera adapter | never (status is always known) |
| `handTracking.{status, handDetected, stableGesture}` | the latest processed result | `handDetected` / `stableGesture` are **null** while the camera is not live or no frame has been processed. A `note` explains it; this is never reported as a tracking failure. |
| `handModel.{status, error}` | tracker status (error text sanitised) | — |
| `robot.*` | `InteractionController.snapshot()` | — |
| `blockers[{ code, detail, affects }]` | derived from the controller state and camera status: `ROBOT_ERROR`, `GESTURE_LOCK_ON`, `HOLD_ACTIVE`, `ACTION_RUNNING`, `CAMERA_NOT_LIVE` (affects gestures, not agent commands) | — |
| `agentProvider.{agentServerReachable, configured, model, unavailableReason, lastCall{ ok, code, at }, refusalFallback}` | `/api/agent/status`; the server records only the outcome of its last provider call | `lastCall` null before any call |
| `recentIssues[{ secondsAgo, source, code, message }]` (max 5) | real bus events: camera errors, hand-model errors, rejected commands, failed tasks, invalid gesture events, pipeline errors. Paths and key-like strings are redacted. | — |

No environment variables, keys, stack traces or file paths are included.

## Workflows

Fixed sequences in `src/agent/workflows.js` (`WorkflowRunner`). The model only chooses a workflow and a validated argument; trusted code makes every decision from real task outcomes.

**`recover_robot`**

1. `diagnose` (`app.diagnostics`). Then:
   - ERROR: stop with `ROBOT_ERROR`. Recovery needs the camera or engine restarted from the app, which the agent cannot do.
   - CONTROL: stop with `HOLD_ACTIVE`. A held gesture is never overridden.
   - EXECUTING: go to step 2.
   - LOCKED: go to step 3.
2. `wait` (polls `app.status`, at most 3 s). If the action is still running, stop with `STILL_EXECUTING`. If the robot entered ERROR or CONTROL during the wait, stop as in step 1. Nothing is dispatched into a new blocker.
3. `unlock` (`robot.command` TOGGLE_LOCK). Requires confirmation; declined or expired stops with `USER_DECLINED` / `CONFIRMATION_EXPIRED`.
4. `reset` (`robot.command` RESET_POSE) if |yaw| ≥ 1°; otherwise skipped ("already facing forward").
5. `verify` (`app.status`). Verified only if the robot is unlocked, not in ERROR or CONTROL, and yaw is 0.

**`checked_command { command }`**

1. `diagnose` (`app.diagnostics`). Stop **before any dispatch** if the robot is in ERROR, a hold is active (CONFLICT), or the lock is on (LOCKED, unless the command is TOGGLE_LOCK). If an action is running, wait for it as above, then **check the blockers again** against the state the wait observed.
2. `execute` (`robot.command`; TOGGLE_LOCK asks for confirmation, and the prompt names the actual effect: "Lock …" or "Unlock …").
3. `verify` (`app.status`), by command:
   - TOGGLE_LOCK: the lock flipped.
   - RESET_POSE: yaw is 0 and it was the last finished command.
   - Other commands: the last finished command is this one, with outcome `completed`.

   Otherwise the result is "executed, but the resulting state could not be confirmed".

**Termination and safety rules**
- **One pass:** each step is dispatched at most once, and a runner instance runs exactly one workflow.
- **Bounded:** at most 6 / 5 steps, a 60 s deadline (which includes up to 30 s for a confirmation), and a 3 s wait.
- **Failure handling:** a failed prerequisite stops the workflow. A step that fails with `TIMEOUT` or `INTERRUPTED` has an **unknown outcome**: it is never repeated, the state is read once (`observedAfterUnknown`), and the stop reason says `OUTCOME_UNKNOWN`.
- **Cancellation:**
  - Stop or Esc during a step lets that step finish and dispatches nothing more.
  - Cancel during a confirmation declines it.
  - Clearing the conversation stops the workflow, and its report is not sent. If a robot action or step was already dispatched, the panel stays **executing** until it finishes, and new messages are refused (BUSY) in the meantime. A cleared conversation can't run a second action alongside one that is still running.
- **Busy while running:** new messages are refused while a workflow runs or waits for confirmation.
- **Same path as everything else:** every step goes through `TaskBroker` and `InteractionController` validation. Workflows add no capability that gestures, keys or the single robot command don't already have.

## Adding a capability

1. Implement and register a task type in `TaskBroker` (with its own `validateArgs`, and an `execute` that resolves only on real completion).
2. Add a tool entry in `src/agent/agentTools.js` (strict JSON schema, task mapping, policy), and add the confirmation rule if the action is consequential.
3. Add tests like `test/agent-tools.test.js` and `test/agent-integration.test.js`.

The system prompt lists capabilities from the same catalog. Tools and the system prompt stay fixed for a server process; this keeps conversation history append-only, as current models require.

## Provider replacement

Implement `complete({ system, tools, messages, signal, requestId })`, returning `{ stopReason, text, toolCalls, content, refusalCategory, usage }` and throwing `ProviderError(code, …)`. Then select it in `server/agent/http.js`. No other component changes.
