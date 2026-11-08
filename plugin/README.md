# Tironian

A Hermes Agent plugin that trains a small local model on the work you already do with Hermes,
tests it against your own past sessions, and lets it take over the routine requests it has
proven it can handle.

Site: https://tironian.vercel.app

The name comes from Tiro, Cicero's secretary. He learned his master's work well enough to
invent a shorthand for it and keep it going.

## The idea

I noticed that a lot of what I ask Hermes is the same handful of things in different words:
which files are eating my disk, how many PDFs are in some folder, what changed recently. Each
time, a large hosted model does the job, and the whole exchange is thrown away afterwards.

Hermes already has the two pieces needed to do better. It records conversations in Nous's own
Hermes function calling format, and it can run GGUF models locally through its managed llama.cpp
runtime. Tironian connects them:

1. It records each model step Hermes takes. Secrets and personal details are masked before
   anything is written to disk.
2. It grades each turn from what actually happened: did the commands succeed, did you correct
   it or ask again, did you deny an approval, did you mark it with `/tironian good` or `bad`.
3. It groups recurring requests into task families. This part is plain, deterministic code, so
   the same history always gives the same families.
4. For a family with enough good examples, it fine-tunes a small model (Qwen3 1.7B, LoRA,
   exported as a quantized GGUF). Training runs on a GPU in your own Modal account, about
   $0.15 a run with a hard cap, or on your own NVIDIA card.
5. It tests that model on steps from sessions it never saw during training. The first step of
   a task (answering the request) and the steps after a tool result are scored separately, and
   the model is only promoted for a kind of step if the lower end of the 95% confidence
   interval on its pass rate there is at least 85%. So a model that writes the right command
   but can't yet summarise the result serves just the command, and your model writes the
   answer. When its command differs from your model's, an optional blind judge decides whether
   both do the job.
6. Once promoted, matching requests go to the small model. Its answer has to parse, call only
   tools the request actually offers, and pass those tools' argument schemas. If any of that
   fails, your usual model answers instead and you don't see the difference.

The goal is simple: routine requests get faster, cheaper and private, and nothing gets worse
than it is today.

## Install

```bash
hermes plugins install angelraph/tironian
hermes plugins enable tironian
```

Then in any Hermes session:

```
/tironian report      what keeps coming up in your history, and what it costs
/tironian families    the task families and what each one still needs
/tironian             current status
```

On first use Tironian reads your existing Hermes history (read only), so the report already has
something in it. That step needs no keys, no GPU and costs nothing.

## Training a model

```bash
hermes tironian trainer setup        # creates Tironian's own venv with the Modal client
hermes tironian trainer check        # checks your Modal token; uses no compute
hermes tironian train <family> --follow
```

Put your Modal token in Settings, Plugins, Tironian, or add `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET` to Hermes's `.env`. Each run is limited by `max_usd_per_run` (default
$1.00), which Tironian enforces as a timeout on the GPU job. To train on your own GPU instead,
set `train_backend: local` and run `hermes tironian trainer setup --local`.

When training is done, Tironian loads the model for you. If you run Ollama, it registers the
model there with the same prompt format it was trained on. If Hermes's own local model server
is running, that works too, and you can point `student_endpoint` at any other OpenAI-compatible
server. Then:

```bash
hermes tironian eval <student>       # test it on held-out steps from your own sessions
hermes tironian shadow <family>      # it answers in the background and gets compared
hermes tironian promote <family>     # after it has agreed with your model often enough
```

Finally, switch on "Let promoted students answer" in the plugin settings.

## What it promises

- Your normal model is called at most once per request. If something inside Tironian fails
  before that call, the request goes to your model unchanged. If something fails after it,
  your model's answer is returned.
- A small model's answer is only used if it parses, uses an offered tool, and passes that
  tool's schema.
- After one fallback in a turn, or after any tool step fails or is blocked, your model finishes
  the turn. Recovering from a failed step is exactly what a small model is worst at.
- Hermes's own guardrails and approvals apply to the small model's tool calls the same way they
  apply to your model's.
- A live model drops back to shadow mode if more than 10% of its last 50 requests fell back,
  or if you mark two of its answers as bad.
- Only steps your own model produced, in turns graded good, are used for training. A small
  model never learns from another small model.

## What it reads, stores and sends

This is the disclosure the Hermes plugin catalog asks for.

**Reads.** Your profile's `state.db`, opened read only, on first use and when you run
`hermes tironian import`. Nothing is written to it.

**Stores.** A SQLite database and job folders in `<HERMES_HOME>/plugin-data/tironian/`. That
includes masked request contexts and responses, tool results, grades, datasets and trained GGUF
files. Contexts older than `retention_days` (90 by default) are dropped. Deleting a session in
Hermes deletes it here too. `hermes tironian purge` deletes everything, and `/tironian off`
stops recording in the current conversation and excludes what was already recorded.

**Masks before storing.** API keys and tokens (OpenAI, Anthropic, OpenRouter, GitHub, AWS,
Slack, Google, Stripe, Hugging Face, Modal), JWTs, private keys, bearer tokens, passwords in
URLs, `password=` style values and other long random strings; email addresses, phone numbers,
card numbers, IBANs; your home folder and login name. A step with more than five masked secrets
or personal details is never used for training.

**Network.** Nothing during normal use. The router only talks to `student_endpoint`, which is
Hermes's local model server at `http://127.0.0.1:8080/v1` by default, or Ollama on
`http://127.0.0.1:11434/v1` when only Ollama has the model. Both stay on your machine. Only when you start a
training run does Tironian's separate trainer process upload that family's masked training set
to a private Volume in your own Modal account, train there, download the model and delete the
job's files from the Volume. The optional judge, off by default, sends held-out examples to
whichever model you assign to the "Tironian judge" slot. While a family is in shadow mode it
also sends that model the request and both answers when they differ, up to
`judge_daily_budget` calls a day.

**Processes and installs.** `hermes tironian trainer setup` creates a venv inside the plugin's
data folder and installs the `modal` client from PyPI (`--local` also installs the training
libraries). Training runs as a separate process in that venv, and only when you start it. Cron
jobs, the gateway and model turns never start training. Shadow mode uses one short background
thread at a time.

**Credentials.** The Modal token is read through Hermes's secret store and passed only to the
trainer process. Tironian never writes it to disk.

**Telemetry.** None.

## Where things are

| Part | File |
|---|---|
| Recording and routing (the `llm_execution` middleware) and hooks | `plugin/core/engine.py` |
| Masking secrets and personal details | `plugin/core/redact.py` |
| Grading turns | `plugin/core/grade.py` |
| Task families | `plugin/core/families.py` |
| Hermes function calling format, shared by training and serving | `plugin/core/hermes_format.py` |
| Datasets, held-out tests, the promotion gate | `plugin/core/pipeline.py`, `plugin/core/evaluate.py` |
| Training (Unsloth LoRA to GGUF, on Modal or a local GPU) | `plugin/trainer/` |

Tironian only uses public plugin surfaces: hooks, the `llm_execution` middleware, one slash
command, one CLI command, an auxiliary model slot, plugin storage and the secret store. It
doesn't patch Hermes, and the part that runs inside Hermes uses only the Python standard
library.

## Development

```bash
python -m pytest tests -q
hermes plugins validate plugin
hermes plugins doctor plugin --ci
```

`scripts/seed_sessions.py` runs real, read-only Hermes sessions to give a new task family
enough examples. The model in the demo was trained on sessions created this way, with my own
Hermes model as the teacher.

## License

MIT
