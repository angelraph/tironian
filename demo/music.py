"""A quiet ambient pad for under the narration, synthesised here so there are no licence questions.

usage: python demo/music.py   -> demo/out/music.wav (as long as the video)
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SR = 48000

# Am9 / Fmaj7 / Cmaj7 / G6, slow and soft (Hz)
CHORDS = [
    [110.00, 164.81, 220.00, 261.63, 329.63, 493.88],
    [87.31, 130.81, 174.61, 220.00, 261.63, 329.63],
    [65.41, 130.81, 196.00, 246.94, 329.63, 392.00],
    [98.00, 146.83, 196.00, 246.94, 293.66, 329.63],
]


def main() -> None:
    total = json.loads((OUT / "timeline.json").read_text(encoding="utf-8"))["total"] + 1.0
    n = int(total * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    seg = 8.0
    for k in range(int(total // seg) + 2):
        chord = CHORDS[k % len(CHORDS)]
        start = k * seg - 2.0
        env = np.clip((t - start) / 3.0, 0, 1) * np.clip((start + seg + 4.0 - t) / 3.0, 0, 1)
        env = np.sin(env * np.pi / 2) ** 2
        for j, f in enumerate(chord):
            for det in (-0.6, 0.6):
                out += env * np.sin(2 * np.pi * (f + det) * t + j) * (0.9 if j < 2 else 0.55)
    # slow tremolo and a gentle low-pass by averaging
    out *= 0.85 + 0.15 * np.sin(2 * np.pi * 0.11 * t)
    kernel = np.ones(24) / 24
    out = np.convolve(out, kernel, mode="same")
    out /= np.max(np.abs(out)) + 1e-9
    out *= 0.6
    stereo = np.stack([out, np.roll(out, 240)], axis=1)
    pcm = (stereo * 32767).astype("<i2")
    with wave.open(str(OUT / "music.wav"), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print("wrote", OUT / "music.wav", f"{total:.1f}s")


if __name__ == "__main__":
    main()
