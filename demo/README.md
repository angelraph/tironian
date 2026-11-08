# How the demo video was made

The 1:26 demo at [tironian.vercel.app/#demo](https://tironian.vercel.app/#demo) is built from this
folder, so anyone can rebuild it or check where a number came from.

| Step | File | What it does |
|---|---|---|
| Script | [`script.json`](script.json) | The narration, scene by scene |
| Voice | [`voice.py`](voice.py) | Reads the script with a Microsoft neural voice (edge-tts) and keeps word timings for the captions |
| Music | [`music.py`](music.py) | A quiet pad synthesised in code, so there are no licence questions |
| Real screen | [`capture_desktop.py`](capture_desktop.py) | Screenshots the real Tironian page in the Hermes web app |
| Motion graphics | [`scenes.html`](scenes.html) | Every scene, timed to the narration with GSAP |
| Render | [`render.py`](render.py) | Draws each frame by exact time with Playwright, then mixes voice and music with ffmpeg |
| Transcript | [`transcript.py`](transcript.py) | Writes [`TRANSCRIPT.md`](TRANSCRIPT.md) and an SRT file |

Every number on screen comes from real runs on my laptop: the report, the training log and loss
curve, the 7 of 31 refusal, the 51 of 51 pass, the 18 of 20 shadow samples in the order they
happened, the live 6.2 second request and the 2.0 second fallback.

```bash
cd demo && npm install && cd ..
python demo/voice.py
python demo/music.py
python demo/render.py
```
