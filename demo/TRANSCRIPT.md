# Tironian demo: transcript

Length 1:23. Every number on screen comes from real runs on my machine.

**0:00 Intro**

This is Tironian, a plugin for Hermes Agent. It turns the work you already do with Hermes into a small model of your own.

**0:08 What it does**

Most of what we ask an agent is the same few jobs, and each one goes to a big cloud model. Tironian records every step Hermes takes, masks secrets, grades how each turn went, and groups repeated requests into task families.

**0:23 The first report**

Your first report comes from the history you already have. No keys, no GPU, and nothing leaves your machine.

**0:30 Training**

Pick a family, and one command fine-tunes a Qwen3 1.7B model on Modal, in your own account. About ten minutes, for about fifteen cents.

**0:40 The gate**

Then it has to prove itself on sessions it has never seen, and a blind judge decides whenever its answer differs from yours. An early model got 7 out of 31, and the gate turned it down. This one wrote the right first command 51 times out of 51. So it takes that step, and your main model still writes the answer.

**0:59 Shadow mode**

Next comes shadow mode, where it answers real requests quietly in the background. It agreed 18 times out of 20, so it went live.

**1:08 Live, with fallback**

Now those requests run on your own machine. Every answer is checked, and if anything is off, your usual model takes over. You never see the difference.

**1:18 Install**

Tironian. Your own model, proven on your own work.
