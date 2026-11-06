# Tironian demo video (under 90 seconds)

The rules: under 90 seconds, show the plugin working. Judges score usefulness, reliability,
polish and creativity, and favour plugins that do more than a prompt or a skill could. So every
shot shows something real happening, and the voiceover says plainly what it is.

Record at 1920x1080, Hermes Desktop on the left two thirds, a terminal on the right third when
needed. Hide the taskbar and any personal file names that aren't part of the shot. Speed up the
training segment; say that it's sped up.

## Shot list

| Time | Screen | Voiceover |
|---|---|---|
| 0:00 to 0:07 | Tironian logo on black, then the home page hero | "A lot of what I ask Hermes is the same few jobs, done by a big model, every time. Tironian fixes that." |
| 0:07 to 0:20 | `/tironian report` in Hermes, then the Tironian page in Desktop: recurring share, task families | "It reads my history, read only, and shows what I repeat. Sixty-four percent of my requests are the same few tasks." |
| 0:20 to 0:34 | Desktop page: click **Train** on the largest-files family; the confirmation shows example count and cost; progress bar moves (sped up, label "4x") | "One click trains a small model for that task, on a GPU in my own Modal account. About thirty cents, capped." |
| 0:34 to 0:48 | Terminal: `hermes tironian eval ...` printing the pass count, the confidence interval and **GATE PASSED** (or the honest failure and why) | "Before it answers anything, it has to pass a test on my own sessions it has never seen. The lower bound has to clear eighty-five percent." |
| 0:48 to 1:08 | Hermes chat: "What are the 5 largest files in my Downloads folder?" Answer appears; then `/tironian why` shows "answered by your student" with the time | "Now that request is answered on my own machine, by my own model." |
| 1:08 to 1:20 | A request where the small model's answer fails a check, or a command gets blocked; `/tironian why` shows the fallback | "And when it isn't sure, my usual model answers. I never see the difference." |
| 1:20 to 1:28 | Desktop page tiles: answered locally, silent fallbacks; then the site URL and the install command | "Tironian. Your own small model, learned from your work. Install it from the Hermes marketplace." |

## Before recording

- Hermes Desktop installed, Tironian's Desktop half switched on.
- A trained, loaded student that has passed (or a clear failed result we show honestly).
- Routing switched on, family promoted to live.
- Close notifications. Use a clean desktop wallpaper.

## Captions (burned in, short)

- "Reads history read only. Sends nothing until you train."
- "Trained on Modal, in your own account. Sped up 4x."
- "Held-out test: sessions it never saw."
- "Every answer checked. Silent fallback."
