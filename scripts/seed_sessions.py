"""Seed real Hermes sessions so Tironian has something to learn from (disclosed in the README).

Each session is a genuine `hermes chat -q` run with the user's own model: Hermes really runs
the (read-only) terminal commands and Tironian captures it exactly as it captures normal use.
Nothing is fabricated: the teacher's real answers are what the student learns.

usage: python scripts/seed_sessions.py --n 150 [--only count] [--hermes PATH] [--pause 5]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

FOLDERS = ["Downloads", "Documents", "Desktop", "Pictures", "Videos", "Music", "AppData\\Local\\Temp"]
EXTS = ["pdf", "exe", "zip", "png", "jpg", "txt", "mp4", "docx", "json", "py"]

DISK = [
    "What are the {k} largest files in my {f} folder? Use the terminal to check.",
    "Which {k} files take up the most space in {f}? Check with the terminal.",
    "Show me the {k} biggest files in my {f} folder (use the terminal).",
    "Find the {k} largest files under my {f} directory using the terminal.",
    "Use the terminal to list the {k} largest files in {f}, with their sizes.",
]
COUNT = [
    "How many .{e} files are in my {f} folder? Use the terminal.",
    "Count the .{e} files in {f} using the terminal.",
    "Use the terminal to tell me how many {e} files I have in {f}.",
    "How many files with the .{e} extension are in my {f} directory? Check with the terminal.",
]
RECENT = [
    "What are the {k} most recently modified files in my {f} folder? Use the terminal.",
    "Use the terminal to show the {k} newest files in {f}.",
    "Which {k} files in {f} changed most recently? Check with the terminal.",
    "List the {k} latest files in my {f} folder using the terminal.",
    "Using the terminal, which {k} files in {f} were modified last?",
    "Show me the {k} most recent files in {f} (use the terminal).",
]


def prompts(n: int, seed: int, only: str | None = None) -> list[str]:
    rng = random.Random(seed)
    pick = {"largest": (DISK,), "count": (COUNT,), "recent": (RECENT,)}.get(only or "", (DISK, COUNT, RECENT))
    out = []
    for i in range(n):
        fam = pick[i % len(pick)]
        out.append(rng.choice(fam).format(k=rng.choice([3, 5, 10]), f=rng.choice(FOLDERS), e=rng.choice(EXTS)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pause", type=float, default=5.0)
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--hermes", default=str(Path(os.environ.get("LOCALAPPDATA", "")) / "hermes" / "bin" / "hermes.exe"))
    ap.add_argument("--log", type=Path, default=Path("seed_sessions.jsonl"))
    ap.add_argument("--only", choices=["largest", "count", "recent"], help="seed one task family")
    a = ap.parse_args()
    done = {json.loads(x)["prompt"] for x in a.log.read_text(encoding="utf-8").splitlines()} if a.log.exists() else set()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for i, p in enumerate(prompts(a.n, a.seed, a.only)):
        if p in done:
            continue
        t = time.time()
        proc = subprocess.Popen([a.hermes, "chat", "-q", p], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", env=env)
        try:
            out, _ = proc.communicate(timeout=a.timeout)
            status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
            limited = "rate limit" in (out or "").lower()
        except subprocess.TimeoutExpired:
            # On Windows, killing hermes.exe leaves its child processes running and holding the
            # pipe open, so end the whole process tree.
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
            else:
                proc.kill()
            try:
                proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            status, limited = "timeout", False
        rec = {"i": i, "prompt": p, "status": status, "seconds": round(time.time() - t, 1), "rate_limited": limited}
        with a.log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        time.sleep(a.pause * (12 if limited else 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
