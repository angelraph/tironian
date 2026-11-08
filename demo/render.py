"""Render scenes.html frame by frame, then mux narration and music into the final video.

usage: python demo/render.py              full video -> demo/out/tironian-demo.mp4
       python demo/render.py --stills     a few frames for review -> demo/out/stills/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
FPS = 30


def open_page(p):
    timeline = json.loads((OUT / "timeline.json").read_text(encoding="utf-8"))
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
    page.add_init_script(f"window.TIMELINE = {json.dumps(timeline)};")
    page.on("console", lambda m: print("console:", m.text) if m.type in ("warning", "error") else None)
    page.goto((HERE / "scenes.html").as_uri())
    page.evaluate("window.READY")
    return browser, page, timeline


def stills(times: list[float]) -> None:
    d = OUT / "stills"
    d.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser, page, _ = open_page(p)
        for t in times:
            page.evaluate(f"window.seek({t})")
            page.screenshot(path=str(d / f"t{t:06.2f}.png"))
        browser.close()


def video() -> Path:
    silent = OUT / "video-silent.mp4"
    with sync_playwright() as p:
        browser, page, timeline = open_page(p)
        n = int(timeline["total"] * FPS)
        enc = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "mjpeg",
                                "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart", str(silent)], stdin=subprocess.PIPE)
        for i in range(n):
            page.evaluate(f"window.seek({i / FPS})")
            enc.stdin.write(page.screenshot(type="jpeg", quality=95))
            if i % 150 == 0:
                print(f"frame {i}/{n}", flush=True)
        enc.stdin.close()
        enc.wait()
        browser.close()
    return silent


def audio() -> Path:
    """Narration placed at each scene's start, plus a quiet music bed under it."""
    timeline = json.loads((OUT / "timeline.json").read_text(encoding="utf-8"))
    total = timeline["total"]
    args, filt = [], []
    for i, s in enumerate(timeline["scenes"]):
        args += ["-i", str(OUT / "audio" / f"{s['id']}.mp3")]
        ms = int(s["start"] * 1000)
        filt.append(f"[{i}:a]aresample=48000,adelay={ms}|{ms}[v{i}]")
    k = len(timeline["scenes"])
    args += ["-i", str(OUT / "music.wav")]
    mix = "".join(f"[v{i}]" for i in range(k))
    filt.append(f"{mix}amix=inputs={k}:normalize=0[voice]")
    filt.append(f"[{k}:a]aresample=48000,volume=0.16,afade=t=in:d=1.5,afade=t=out:st={total - 2.5:.2f}:d=2.5[bed]")
    filt.append("[voice][bed]amix=inputs=2:normalize=0,alimiter=limit=0.95[out]")
    wav = OUT / "mix.wav"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", *args, "-filter_complex", ";".join(filt), "-map", "[out]",
                    "-t", f"{total:.3f}", str(wav)], check=True)
    return wav


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", nargs="*", type=float)
    ap.add_argument("--audio-only", action="store_true")
    a = ap.parse_args()
    if a.stills is not None:
        stills(a.stills)
        return
    wav = audio()
    if a.audio_only:
        return
    silent = video()
    final = OUT / "tironian-demo.mp4"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(silent), "-i", str(wav), "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(final)], check=True)
    print("wrote", final)


if __name__ == "__main__":
    sys.exit(main())
