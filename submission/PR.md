# plugin-catalog: add tironian

## What it does

Tironian trains a small local model on the requests you keep repeating in Hermes, tests it on held-out steps from your own past sessions, and only then lets it answer those requests. Every answer it gives is checked before Hermes sees it, and anything that doesn't pass goes to your usual model instead.

1. **Capture.** `llm_execution` middleware records each request and response; observer hooks record tool outcomes, approvals, interrupts and session boundaries. Secrets and personal details are masked before anything is written.
2. **Grade.** Deterministic signals from the turn itself: tool failures and exit codes, the user correcting or repeating the request, denied approvals, interrupts, reactions and `/tironian good|bad`. Only good turns produced by the user's own model are ever trained on.
3. **Group.** Recurring requests become task families (deterministic clustering; paths, names, file types and filler words are set aside).
4. **Train.** One small model per family: Qwen3 1.7B, LoRA, exported as a Q4_K_M GGUF. Runs on a GPU in the user's own Modal account (cost shown first, hard cap enforced as a timeout) or on their own NVIDIA GPU. The training prompt is built by the same code the router uses at serving time.
5. **Test.** The model answers held-out steps from sessions it never saw. The first step of a task and the steps after a tool result are scored separately, and a kind of step is served only if the lower bound of the 95% Wilson interval on its pass rate clears 85% (configurable), with at least 30 steps and 98% format validity. When the student's command differs from the user's model's, an optional blind judge decides. Sessions recorded after training join the test, so the evidence grows without retraining.
6. **Route.** Shadow mode first, then live. Each answer must parse, call only tools the request offers and pass their schemas; otherwise the user's model answers. After any failed or blocked tool step, the user's model finishes the turn. A model with too many fallbacks demotes itself.

**Results on my machine** (no GPU, free hosted model, student on Ollama): an early student scored 7 of 31 and was refused; the current one wrote the right first command on 51 of 51 held-out first steps (lower bound 93%), agreed with my model 18 of 20 times in shadow mode, and now writes the first command of those requests locally while my model writes the answer. Training cost about $0.14 on a Modal L4.

Demo (86 s): see the X post. Site and docs: https://tironian.vercel.app

## Hermes surfaces used

- `ctx.register_middleware("llm_execution", ...)`: capture and routing
- `ctx.register_hook(...)`: `on_session_start`, `pre_llm_call`, `post_llm_call`, `post_tool_call`, `api_request_error`, `on_session_end`, `on_session_finalize`, `on_session_delete`, `post_approval_response`, `subagent_start`
- `ctx.register_command("tironian", ...)`: status, report, good/bad marks, training, promotion
- `ctx.register_cli_command("tironian", ...)`: the whole pipeline from a terminal
- `ctx.register_auxiliary_task("tironian_judge", ...)`: optional judge model slot
- `plugins.plugin_storage.plugin_data_dir`, `agent.secret_scope.get_secret`, `hermes_cli.plugin_events.broadcast_plugin_event`
- Desktop half (`desktop/plugin.js`): a route, a sidebar entry and a status-bar chip, SDK imports only
- Dashboard backend (`dashboard/plugin_api.py`): routes under `/api/plugins/tironian`

No tools, no core overrides, no `sys.modules` or `setattr` on Hermes objects. The in-process code uses only the Python standard library.

## Disclosure (rule 13)

- **Reads outside its own data:** this profile's `state.db`, opened read only (`mode=ro`), on first use and on `hermes tironian import`. It never writes to it.
- **Stores:** masked request contexts and responses, tool outcomes, grades, datasets and trained GGUF files under `plugin-data/tironian/`. Contexts older than `retention_days` (90) are dropped. Deleting a session in Hermes deletes it here.
- **Network:** none during normal use. The router talks only to `student_endpoint` (default `http://127.0.0.1:8080/v1`, Hermes's local model server, or Ollama on `127.0.0.1:11434` when only Ollama has the model; both local). After training, if Ollama is installed, Tironian runs `ollama create` to register the model there. Only when the user starts a training run, a separate trainer process uploads that family's masked training examples to a private Volume in the user's own Modal account, trains, downloads the model and deletes the job's files from the Volume. The optional judge (off by default) sends held-out examples to the model the user assigns to the "Tironian judge" slot, and, while a family is in shadow mode, the request and both answers when they differ, within a daily call budget (`judge_daily_budget`).
- **Shell commands and installs:** `hermes tironian trainer setup`, run by the user, creates a venv under plugin data and installs the `modal` client from PyPI (`--local` also installs the training stack).
- **Background processes:** training runs as a detached process in that venv, started only by an explicit user action; never from cron, the gateway or a model turn. Shadow mode uses one short-lived background thread at a time.
- **Credentials:** the Modal token is read with `get_secret` and passed only to the trainer process's environment; never written to disk by the plugin.
- **Telemetry:** none.

## How to test

1. `hermes plugins validate plugin --install-deps` and `hermes plugins doctor plugin --ci` on a checkout of the pinned SHA.
2. `python -m pytest tests -q` in the repo: 75 tests covering redaction, grading, clustering, the dataset split, the per-step Wilson gate, the judge, and every routing fallback path (malformed output, unknown tool, schema failure, server error, timeout, sticky fallback, hand-back after a blocked tool, auto-demotion, shadow mode never blocking). CI runs them on Linux and Windows, Python 3.11 to 3.14.
3. Without any spend: `hermes plugins enable tironian`, then `/tironian report` in any session, and `hermes tironian train <family> --backend dry-run --follow`.

`validate`, `doctor` and the tests also pass on v0.23.0. Checked live on Hermes v0.22.7 on Windows: validate and doctor pass, capture records streaming turns, and a small model answered a real turn through the middleware; when Hermes blocked its command, the user's model took over and finished the task.

## Screenshots

(Desktop page and terminal screenshots attached below.)
