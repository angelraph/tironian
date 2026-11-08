# Internal review

This is my own review of Tironian, not a third-party audit. It lists what I checked, what I found,
and what I did about it. Every fix has a commit that explains why.

## What was checked

| Area | How |
|---|---|
| Routing never makes things worse | Tests for every fallback path: malformed output, unknown tool, schema failure, server error, timeout, sticky fallback, hand-back after a blocked tool, auto-demotion, shadow mode never blocking. Live: a request with the student unreachable fell back with nothing shown to me |
| Your model is called at most once | Middleware tests count calls to the next handler for every path |
| No data leaks into the test | Datasets are split by session, and a test checks that no session is in both training and held-out sets |
| Training and serving match | The same code renders the training prompt and the serving prompt; a test compares them |
| Secrets stay out of storage | A redaction test corpus covers keys, tokens, JWTs, private keys, emails, phone and card numbers, IBANs and usernames |
| Disclosure is complete | README, privacy page, catalog entry and PR text were checked against what the code actually sends |
| Docs match reality | Every claim on the site and in the README was checked against the code and my runs |
| Hermes checks | `hermes plugins validate` (including its security scan) and `doctor --ci` at the pinned SHA |

## What I found and fixed

| Found | Fix |
|---|---|
| The trainer's environment could see Hermes's Python paths and skip installing its own dependencies | Every trainer call gets a clean environment |
| Sessions captured live were imported again from history, which could put one conversation in both training and test data | Already captured sessions are skipped and old duplicates removed |
| Families split by folder names and file types | Arguments become placeholders, synonyms merge, filler words are ignored |
| A training job whose process died mid-export blocked every later run | Dead drivers are detected and the job is marked failed |
| A small model's command was compared character by character, so correct commands failed | Shell commands are compared by words, and a blind judge decides when they still differ |
| One overall score hid that the student was strong on first steps and weak after tool results | Each kind of step is gated on its own |
| The held-out split counted steps overall, leaving too few first steps to judge | The split grows until every kind has enough |
| When one task was most of my history, its own key word was thrown away as "common" | A frequent word is ignored only if it is spread across kinds of request |
| The judge timed out on a slow free model, or ran out of its call budget, and quietly stopped judging | It retries, and the result says how many steps went unjudged |
| Shadow mode compared answers without the judge the test uses | Shadow mode uses the same judge, within a daily budget |
| Shadow mode only ran once students were allowed to answer, so the README's steps never got past shadow | Shadow mode runs on its own |
| Ollama unloads an idle model, and the reload missed the routing deadline | Sessions warm the student when they start; shadow replays get two minutes |
| The Hermes page showed the overall score of a family that passed on first steps, and counted relabelled shadow samples | It shows each kind of step as the gate judged it, and the same samples promotion counts |
| The docs said training cost $0.30 and promised speed | They now say about $0.15, and speed only with a GPU |
| A planned feature (a judge for unclear turns) was described but never built | The claim was removed |
| A task that changes after training wasn't noticed until answers started falling back | Spot checks judge the student against your model on about 1 in 10 live requests; under 85% of the last 20 sends it back to shadow |
| Demotion rules counted all-time history, so a re-promoted student could be knocked down by old mistakes, and a demoted one promoted on old agreements | Every status change starts a fresh window of evidence |
| It took two bad marks to take a family away after a corrected answer | One correction, re-ask or denial of a turn the student took part in is enough |
| Deleting a session removed its data, but a student trained on it kept answering | Forgetting a session takes every student that saw it off duty until retrained |
| Ollama unloaded an idle student after 5 minutes, so the first request after a pause fell back | The warm-up asks Ollama to keep it loaded for an hour |

## Still open

- Spot checks send about 1 in 10 of a live family's requests to your usual model. That is the
  price of catching drift; it can be lowered or turned off.
- Only tested live on Windows. CI covers Linux and Windows; macOS is untested.
- On a CPU-only machine the student takes about 6 seconds a step, so the gain there is cost and
  privacy rather than speed.
- My results come from seeded sessions, which are more uniform than everyday use.
