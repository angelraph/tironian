# Tironian

**Your own small model, learned from your own Hermes work.**

Tironian is a [Hermes Agent](https://hermes-agent.nousresearch.com) plugin that closes the loop
between how you actually use Hermes and a personal, local model:

1. **Captures** every model step Hermes takes, redacting secrets and personal data *before*
   anything is stored.
2. **Grades** every turn from what really happened — tool failures, exit codes, corrections,
   repeats, approvals, interrupts, your `/tironian good|bad` — never from the model's opinion of itself.
3. **Finds recurring tasks** ("what's using disk space in X", "count the .pdf files in Y"),
   deterministically.
4. **Fine-tunes a small model per task family** (Qwen3 1.7B LoRA → quantized GGUF) on a rented GPU
   in *your* Modal account (~$0.30 a run, hard-capped) or on your own NVIDIA GPU.
5. **Proves it** on held-out steps from your own sessions — sessions it never saw in training — and
   promotes it only if the *lower* bound of the 95% confidence interval on its pass rate clears 85%.
6. **Routes** matching requests to it through Hermes's local model runtime. Every answer is
   validated (well-formed, a tool the request actually offers, schema-valid arguments); anything
   off, and your own model answers instead — silently. A failed or blocked step hands the rest of
   the turn back to your model.

The result: the routine part of your work gets faster, cheaper, private and offline-capable,
week after week — and it can never get worse than what you have today.

> Named after Tiro, Cicero's secretary, who learned his master's work so well that he invented
> shorthand to carry it forward.

## Why

Every Hermes turn is training data that is thrown away. Hermes already writes trajectories in
Nous's Hermes function-calling format and already runs local GGUF models through its managed
llama.cpp runtime. Tironian is the missing link between the two: a data flywheel that turns
*your* usage into *your* model, with an evaluation gate strict enough that you can trust it.

## Install

```bash
hermes plugins install angelraph/tironian
hermes plugins enable tironian
```

Then, in any Hermes session:

```
/tironian report        what's recurring in your history and what it costs
/tironian families      task families and what each still needs
/tironian               status: learned, served locally, saved
```

Tironian reads your existing Hermes history **read-only** on first use, so the report has
something to show in the first minute — no keys, no GPU, no spend.

## Train your first student

```bash
hermes tironian trainer setup          # creates Tironian's own venv with the Modal client
hermes tironian trainer check          # verifies your Modal token (no compute, no cost)
hermes tironian train <family> --follow
```

Add your Modal token in **Settings → Plugins → Tironian** (or as `MODAL_TOKEN_ID` /
`MODAL_TOKEN_SECRET` in Hermes's `.env`). Modal's free Starter plan covers many runs; each run is
capped by `max_usd_per_run` (default $1.00), enforced as a GPU timeout. Prefer your own GPU? Set
`train_backend: local` and run `hermes tironian trainer setup --local`.

When training finishes, load the student — Hermes Desktop → **Settings → Providers → Local
Models → Add model file** (the path is printed) — then:

```bash
hermes tironian eval <student>         # held-out test on your own sessions; shows the gate verdict
hermes tironian shadow <family>        # answers in the background, compared with your model
hermes tironian promote <family>       # once shadow agreement is high enough
```

and turn on **Let promoted students answer** in the plugin settings.

## Guarantees

- **Never worse than baseline.** Your model's call is made at most once per request; any
  Tironian error before it falls through to your model; any error after it returns your model's
  answer. Hermes's own middleware is fail-open as well.
- **Validated or not used.** A student's answer must parse, call only tools the request offers,
  and pass each tool's argument schema. Otherwise your model answers.
- **Sticky hand-back.** After one fallback, or any failed or blocked tool step, your model finishes
  the turn.
- **Approvals still apply.** Student tool calls go through the same Hermes guardrails and
  approvals as your model's.
- **Auto-demotion.** A live student whose fallback rate exceeds 10% (over the last 50) or that you
  mark bad twice drops back to shadow mode.
- **Students never learn from students.** Only steps your own model produced, in turns graded
  good, are ever trained on.

## Disclosure (catalog rule 13)

- **Reads outside its own data:** this profile's `state.db`, **read-only** (`mode=ro`), once on first
  use and on `hermes tironian import`, to grade past sessions. Nothing is written to it.
- **Stores:** a SQLite database and job folders under `<HERMES_HOME>/plugin-data/tironian/`:
  redacted request contexts and responses, tool outcomes, grades, datasets, trained GGUF files.
  Captured contexts older than `retention_days` (90) are dropped. Deleting a session in Hermes
  deletes it here; `hermes tironian purge` deletes everything; `/tironian off` stops capture for a
  conversation and excludes what was already captured.
- **Redaction before storage:** API keys and tokens (OpenAI, Anthropic, OpenRouter, GitHub, AWS,
  Slack, Google, Stripe, Hugging Face, Modal), JWTs, private keys, bearer tokens, URL credentials,
  `password=`-style assignments, high-entropy strings; emails, phone numbers, card numbers (Luhn),
  IBANs; your home folder and login name. Steps with more than 5 sensitive masks are never trained on.
- **Network:** none during normal use. The router talks only to `student_endpoint`
  (default `http://127.0.0.1:8080/v1`, Hermes's local model runtime). **Only when you start
  training** does Tironian's separate trainer process upload that family's redacted training set
  to a private Volume in **your own Modal account**, run the job there, download the trained model,
  and delete the job's files from the Volume. The optional LLM judge (off by default) sends
  held-out examples to the model you assign to the "Tironian judge" auxiliary slot.
- **Processes and installs:** `hermes tironian trainer setup` (explicit) creates a venv under the
  plugin data folder and installs the `modal` client from PyPI (`--local` also installs the
  training stack). Training runs as a detached process in that venv; it is only ever started by
  you — never by cron, the gateway, or a model turn. Shadow mode uses one short-lived background
  thread at a time.
- **Credentials:** the Modal token is read only through Hermes's secret store and passed only to
  the trainer process's environment. It is never written to disk by Tironian.
- **Telemetry:** none.

## How it works

| Piece | Where |
|---|---|
| Capture + live routing (`llm_execution` middleware) and hooks | `plugin/core/engine.py` |
| Redaction | `plugin/core/redact.py` |
| Grading | `plugin/core/grade.py` |
| Task families (deterministic clustering) | `plugin/core/families.py` |
| Hermes function-calling format (one path for training *and* serving) | `plugin/core/hermes_format.py` |
| Datasets, held-out evaluation, promotion gate | `plugin/core/pipeline.py`, `plugin/core/evaluate.py` |
| Training (Unsloth LoRA → GGUF, Modal or local CUDA) | `plugin/trainer/` |

Uses only public plugin surfaces: hooks, `llm_execution` middleware, a slash command, a CLI
command, an auxiliary model slot, plugin storage and the secret store. No core patches. The
in-process plugin is standard-library only.

## Development

```bash
python -m pytest tests -q
hermes plugins validate plugin
hermes plugins doctor plugin --ci
```

`scripts/seed_sessions.py` runs real, read-only Hermes sessions to bootstrap a family for
development and the demo (the demo's student was trained on sessions seeded this way, with the
author's own model as the teacher).

## License

MIT
