# Tironian on the IMD board: what happened and what to fix

Source: hermes-hackathon.imd.site (unofficial, IMD's agent swarm, not Nous). Data pulled 2026-11-13.

## Where it landed

- 58th of 1,454. Score 70.6.
- Got into round 3: the 128 plugins actually installed and run inside Hermes. Did not make the final 20.
- Round 3 average: 26.26 of 30. The 20th finalist (Hypnos, our closest rival) had 27.37.

| Read | Useful | Reliable | Polish | Creative | Beyond | Total |
|---|---|---|---|---|---|---|
| Round 1 | | | | | | 27 |
| Round 2 | | | | | | 28 |
| Round 3, agent A | 4 | **2** | 5 | 5 | 5 | 26 |
| Round 3, agent B | 4 | 4 | 5 | 5 | 5 | 28 |
| Round 3, agent C | 4 | **2** | 4 | 5 | 5 | 25 |
| Finalists' average (round 3) | 4.59 | 4.29 | 4.84 | 4.35 | 4.81 | 27.69 |

Creative and Beyond are perfect, better than the finalists' average. Reliable is the whole gap.

## Why it missed the final

Two of the three round 3 agents ran `hermes tironian dataset missing-family`, got a Python traceback (`ValueError: unknown family`), and the rules cap Reliable at 2 for any crash. With Reliable 4 on those two reads, the round 3 totals become 28, 28, 27 (average 27.67). That beats Hypnos (27.37) and sits next to kata (27.97, 6th place). One missing try/except most likely cost a finalist spot.

## Every criticism, checked against the code

All of these are real. Ranked by how much they hurt.

1. **CLI crashes on bad input** (cost the final). `cli.run()` has no top-level error handling. The same crash happens on `dataset <unknown>`, `train <unknown>`, `eval <unknown student>`, `passport <unknown student>`, and `train` before the trainer is set up (RuntimeError). Fix: catch ValueError and RuntimeError in `run()`, print a plain sentence, and exit 1. Add a test that runs every subcommand with a bogus id and asserts there's no traceback.
2. **The gate can be looser than the README says.** In `evaluate.gated_scorecard`, when the whole set passes, a step kind with only a handful of examples rides along on its raw pass rate (around evaluate.py:304). The docstring promises "passing is never easier". Fix: a kind serves only if it passes its own `min_n` and Wilson bound.
3. **Usernames that are also chat roles get masked.** If the login name is `user` (common in Docker, CI and cloud VMs), the redactor replaces the message role `"user"` with `[REDACTED:user]` and breaks the dataset. A judge hit this. Fix: never mask the role/name structural fields; skip usernames that equal `user`, `assistant`, `system` or `tool`.
4. **Calls to the local model can go through a proxy.** `urllib` respects HTTP_PROXY, so on a machine with a proxy set, the calls to Ollama at 127.0.0.1 go to the proxy. A judge's tests failed this way. Fix: build the opener with an empty ProxyHandler for loopback endpoints.
5. **No cap on the size of student replies.** `resp.read()` has no limit. Fix: read at most a few MB and treat anything bigger as a failed reply (silent fallback).
6. **Manifest warning.** Hermes warns `provides_middleware is an unknown manifest field`. Cosmetic (Hypnos has it too), but it shows on every doctor run. Check what field name Hermes expects now.
7. **Nobody could verify a real trained model.** Every agent said the same thing: the demo is scripted, training is a dry run, and the laptop numbers are self-reported. That's what holds Useful at 4. Fix: publish the real v4 student (GGUF) and its frozen held-out set, plus a `hermes tironian verify` command that downloads both and reruns the gate on CPU in a few minutes. Then any judge can reproduce 51/51 themselves. Hypnos got marked down for "synthetic persona" evidence, so this would be a clear edge over it.
8. **No family is live right now.** The v5 check demoted v4, which is honest, but it means the author has nothing serving today. A count v6 trained on more varied folders (~$0.19 on Modal) that passes the new-folder check would turn this into the strongest story Tironian has: it caught its own model cheating, got retrained, and earned its spot back.

## My own rating, as a judge

| | Now | After fixes 1 to 7 |
|---|---|---|
| Useful | 4 | 4.5 (5 with a live, verifiable model) |
| Reliable | 3 | 4.5 |
| Polish | 4.5 | 5 |
| Creative | 5 | 5 |
| Beyond a prompt | 5 | 5 |
| **Total** | **26.5** | **28.5 to 29** |

Next to its neighbours:

- **Hypnos** (12th, finalist). Same idea: LoRA from your habits, gated promotion, routing. Tironian is stronger on proof: held-out sessions, Wilson bounds, a new-folder veto, one-correction demotion, and a cheap trainer that runs on Modal. Hypnos only placed above Tironian because it didn't crash.
- **kata** (6th). Learns repeated tool sequences and compiles them into deterministic tools. Easier to show (works after 3 conversations, no GPU), so it scores higher on Useful. Tironian is the more ambitious idea.
- **shadow** (10th, 2nd in round 3). Blind A/B of models on your own past turns. Its judges loved the rigor and the "undo". Tironian has the same rigor; it needs the same "anyone can check it" feel.

What the top five share: no crash on bad input, a clean catalog check, a large test suite (170 to 460 tests, against Tironian's 104), and something a judge can run themselves in minutes.

## Plan, in order

1. Fixes 1 to 6 with tests. Small, about an afternoon, no cost.
2. Verifiable model: publish v4 GGUF and the held-out set, add `hermes tironian verify`.
3. Retrain count v6 on more varied folders (needs your OK for about $0.19 on Modal), run it through the full gate, and promote it if it passes.
4. Re-pin the catalog entry to the new SHA (a PR to Nous; needs your OK).
5. X post: "The judges found a crash. Fixed. Here's the model, verify it yourself."
