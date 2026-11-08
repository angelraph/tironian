"""Narration: one audio file per scene, with word timings for the captions.

usage: python demo/voice.py
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
from pathlib import Path

import edge_tts
import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "audio"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def duration(path: Path) -> float:
    err = subprocess.run([FFMPEG, "-i", str(path)], capture_output=True, text=True).stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


async def speak(text: str, voice: str, rate: str, mp3: Path) -> list[dict]:
    words = []
    comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    with mp3.open("wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                words.append({"t": chunk["offset"] / 1e7, "d": chunk["duration"] / 1e7, "w": chunk["text"]})
    return words


async def main() -> None:
    spec = json.loads((HERE / "script.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    timeline, t = [], 0.0
    for sc in spec["scenes"]:
        mp3 = OUT / f"{sc['id']}.mp3"
        words = await speak(sc["text"], spec["voice"], spec["rate"], mp3)
        d = duration(mp3)
        timeline.append({"id": sc["id"], "start": round(t, 3), "speech": round(d, 3),
                         "dur": round(d + sc["pad"], 3), "text": sc["text"], "words": words})
        t += d + sc["pad"]
    (HERE / "out" / "timeline.json").write_text(json.dumps({"total": round(t, 3), "scenes": timeline}, indent=1),
                                               encoding="utf-8")
    for s in timeline:
        print(f"{s['id']:<9} {s['start']:6.2f}s  +{s['dur']:5.2f}s")
    print(f"total {t:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
