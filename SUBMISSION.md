# Tironian: hackathon submission

Entry for the Nous Research **Hermes Agent plugin hackathon** (entries closed November 11, 2026).

**The judged version is commit [`9b590cc`](https://github.com/angelraph/tironian/tree/9b590cc094f4b65cd70e979f632e1783424526a8),
the one Nous's catalog entry pins.** I haven't changed the catalog entry since the deadline. Everything
on `master` after November 11 is work done after entries closed (the fixes from an outside review,
`proof` and `verify`), and it's listed in the README's build log under Nov 13.

| | |
|---|---|
| **Plugin** | Tironian, `plugin/` in [angelraph/tironian](https://github.com/angelraph/tironian) |
| **Catalog PR** | [NousResearch/hermes-agent#151281](https://github.com/NousResearch/hermes-agent/pull/151281), merged by Nous on November 10, pinned to `9b590cc` |
| **Demo on X** | [x.com/RaphElevator/status/2119405172937961540](https://x.com/RaphElevator/status/2119405172937961540), tagging @NousResearch |
| **Demo video** | [tironian.vercel.app/#demo](https://tironian.vercel.app/#demo) (1:26), transcript in [demo/TRANSCRIPT.md](demo/TRANSCRIPT.md) |
| **Submission form** | Submitted on November 8, 2026 |
| **Site and docs** | [tironian.vercel.app](https://tironian.vercel.app) |

## Eligibility

| Requirement | Evidence |
|---|---|
| A Hermes Agent plugin | `plugin/plugin.yaml`, loaded by Hermes v0.22.7 and v0.23.0; `hermes plugins validate` and `doctor --ci` pass |
| Submitted to the plugin catalog by PR | [#151281](https://github.com/NousResearch/hermes-agent/pull/151281) adds `plugin-catalog/tironian.yaml` with an exact SHA pin and the rule 13 disclosure |
| Demo video under 90 seconds, posted on X, tagging @NousResearch | [The post](https://x.com/RaphElevator/status/2119405172937961540); the video is 86.7 seconds |
| Submission form | Filled in with the PR and X links |
| Open source | MIT, [LICENSE](LICENSE) |

## How it meets the judging criteria

**Usefulness.** It takes the requests you already repeat and gives them a model of your own, at
about $0.15 a training run. The first report works on your existing history with no keys, no GPU
and no cost, so anyone can see in a minute what they repeat.

**Reliability.** This is where most of the work went.

- A student is only trusted after a held-out test on sessions it never saw, judged by the lower
  end of a 95% confidence interval, then after shadow mode on real requests.
- Each kind of step is gated on its own, so a student serves only what it has proven.
- Every live answer must parse, use only offered tools and fit their schemas, or your usual model
  answers. A model that falls back too often demotes itself.
- Spot checks on live requests catch a task that changes after training, and one correction
  sends a student back to shadow mode.
- 99 tests cover every fallback path, with CI on Linux and Windows.
- A training run that loses its connection to Modal can be reattached with
  `hermes tironian resume`, with no new GPU time. I used it for real on Nov 9.
- Live evidence: the gate refused my early student at 7 of 31, passed the current one at 51 of 51
  (lower bound 93%), shadow mode agreed 18 of 20, and a live fallback with the student
  unreachable showed nothing to me.
- It also catches its own maker's blind spots. A reply on X suggested holding out whole folder
  trees, not just sessions. I built that, and it found that my live student had learned my home
  folder instead of the task: 5 of 13 on folders it never trained on. It took the student off duty
  by itself. Tironian would rather refuse a model than flatter it.

**Polish.**

- `/tironian demo` (or the Run the demo button on the page) shows the whole loop in about 30
  seconds, offline and with no keys, so a judge can see it work without training anything.
- A Tironian page and status chip inside Hermes, `/tironian` in chat and a full `hermes tironian`
  CLI.
- A site with docs, a cost page and an honest changelog, plus a demo video.
- Real screenshots throughout the [README](README.md).

**Creativity.** Routers choose between existing models; Tironian makes a new one that knows your
work, and refuses to use it until it has proven itself. What I haven't seen elsewhere is the
split by kind of step: the student takes over exactly the part of a task it has mastered (here,
writing the first command), proven on its own held-out data and in shadow mode, while your usual
model keeps the rest.

Two more things make the trust visible. A reply your own model helped write ends with a small
proof card in chat: the family, the held-out result, the shadow record and the tokens saved. And
every student gets a sealed passport (`hermes tironian passport`) tying the model file's hash to
its training data, its training run and every test it took, so anyone can check where a model
came from and `--verify` shows if a single number was changed.

**More than a prompt or a skill.** Middleware that captures and routes model steps, a grading
engine, deterministic clustering, a dataset builder that matches the serving format exactly, a
GPU trainer, a statistical gate, shadow mode, live routing with validation and fallback, a Desktop
page and a dashboard API. None of that can be written as a prompt.

## What is real and what was seeded

All numbers come from my own laptop. Most sessions were created with
[`scripts/seed_sessions.py`](scripts/seed_sessions.py), which runs real `hermes chat` turns with my
own model so that I had enough repeated work to train on within the hackathon. Those are genuine
Hermes turns, graded the same way as any other, but they are more uniform than everyday use. The
[internal review](AUDIT.md) lists what was checked and what is still open.
