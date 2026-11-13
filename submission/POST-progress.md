Submissions closed. We kept building anyway 🏛️

This week an independent swarm of AI judges installed Tironian and ran it inside real Hermes. 1,454 plugins ranked. We landed 58th.

Their notes stung a little, and they were right.

Creativity: 5 out of 5. Doing something a prompt can't: 5 out of 5. Then two judges typed a family name that didn't exist and got a crash. One missing error check, and our reliability score was capped at 2. That was most likely the difference between us and their final 20.

So we didn't argue with the scoreboard. We went back to work.

Within a day, everything they found was fixed:
• bad input now gets one clear sentence, never a crash
• a small group of test steps can no longer slip through on the overall score
• a login named "user" no longer breaks the training data
• calls to your local model skip any proxy on your machine
• replies from the local model are size capped

But one note mattered more than the rest. Every judge wrote some version of the same thing: "we couldn't confirm a real trained model."

They only had our word for it. That's not good enough for us either.

So we built a way to check it without trusting us at all.

hermes tironian verify downloads our real trained model and the exact test it took. It checks the file is the same one that was tested, then asks it every question again, on your own machine.

We ran it on a regular laptop CPU:

Same answer as in the test: 51 of 51
Passed now: 51 of 51
Gate reproduced

About 5 seconds a step. No GPU, no keys, nothing to take on faith.

The model, the full test and how every answer was graded are public now. Run it yourself and tell us what you get.

We're not done. Next up is getting a newly trained model through the same gate and live again.

github.com/angelraph/tironian/releases/tag/proof-count-v4

@NousResearch
