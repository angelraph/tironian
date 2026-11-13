Tironian is now live on the Hermes Marketplace 🏛️

A few weeks ago I noticed something about how I use Hermes. Most of my day is the same handful of jobs on repeat. Count these files. Find the biggest ones. Show me what changed recently. Every time, a big cloud model does the work from scratch, like it's never seen me before.

So I built a plugin that pays attention to that.

Here's how it works:

1. It learns your habits
Tironian quietly groups the requests you keep making into task families. Secrets and personal details are masked before anything gets saved.

2. It trains a small model for each one
When a family has enough good sessions, you can train a small model on them. On Modal it costs me about 15 to 19 cents a run, or you can use your own GPU. Your data is deleted from the cloud after the run, and the model ends up running locally on your machine through Ollama.

3. It has to earn the job
Before a small model answers anything, it gets tested on steps from your own sessions that it never trained on. It needs a score of at least 85%, counting the worst case and not just the average. Then it runs in shadow mode next to your usual model so you can compare their answers before it goes live.

4. It never gets a free pass
Once it's live, every answer still gets checked. If it slips, your usual model takes over without you noticing. One correction from you and it gets demoted.

Some real numbers from my own setup:
• my file counting model passed 51 out of 51 first steps
• in shadow mode it matched my main model 18 out of 20 times
• its first real request: it wrote the find command in 6.2 seconds and my main model wrote the reply
• about 12k cloud tokens saved every time it handles a step

Then I made the test harder. I held back entire folders it had never seen, like AppData, Downloads and C:\Users\Public. The newer version scored 44 out of 51 there, so it got demoted. I could have lowered the bar to keep it live, but I didn't. A model you can't trust on new ground has no business answering for you.

A few more things in the box:
• /tironian demo to watch the whole flow in a few seconds
• a sealed passport for every model, so you can check what it was trained and tested on
• no telemetry and no network calls during normal use

Huge thanks to @NousResearch and @Teknium1 for the review. The feedback made it better.

Try it:
hermes plugins install tironian

hermes-agent.nousresearch.com/marketplace/tironian
