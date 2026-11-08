"""Transcript (Markdown) and subtitles (SRT) from the narration timings.

usage: python demo/transcript.py -> demo/TRANSCRIPT.md and demo/out/tironian-demo.srt
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TITLES = {"intro": "The problem, and Tironian", "pipeline": "What it does", "report": "The first report", "train": "Training",
          "gate": "The gate", "shadow": "Shadow mode", "live": "Live, with fallback", "desktop": "Inside Hermes",
          "outro": "Install"}


def ts(t: float, srt: bool = True) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    if not srt:
        return f"{int(m)}:{int(s):02d}"
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"


def main() -> None:
    tl = json.loads((HERE / "out" / "timeline.json").read_text(encoding="utf-8"))
    md = ["# Tironian demo: transcript", "",
          f"Length {ts(tl['total'], False)}. Every number on screen comes from real runs on my machine.", ""]
    srt, n = [], 1
    for s in tl["scenes"]:
        md += [f"**{ts(s['start'], False)} {TITLES[s['id']]}**", "", s["text"], ""]
        sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", s["text"]) if x.strip()]
        total_words = len(s["text"].split())
        idx = 0
        for sent in sentences:
            k = len(sent.split())
            a = s["start"] + s["speech"] * idx / total_words
            b = s["start"] + s["speech"] * (idx + k) / total_words
            srt += [str(n), f"{ts(a)} --> {ts(b)}", sent, ""]
            n += 1
            idx += k
    (HERE / "TRANSCRIPT.md").write_text("\n".join(md), encoding="utf-8")
    (HERE / "out" / "tironian-demo.srt").write_text("\n".join(srt), encoding="utf-8")
    print("wrote TRANSCRIPT.md and tironian-demo.srt")


if __name__ == "__main__":
    main()
