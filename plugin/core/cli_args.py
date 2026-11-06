"""Argument definitions for ``hermes tironian``.

Hermes builds its command-line parser on every start, so this module stays tiny: it only
describes the arguments. The code that does the work (``cli.py``) is imported when a
``hermes tironian`` command actually runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run(args: Any) -> int:
    from .cli import run as _run
    return _run(args)


def setup(sub: Any) -> None:
    sp = sub.add_subparsers(dest="tironian_cmd")
    sp.add_parser("status", help="What's learned, what's served locally, savings")
    imp = sp.add_parser("import", help="Read Hermes history (read-only) and rebuild task families")
    imp.add_argument("--days", type=int, default=90)
    sp.add_parser("report", help="Recurring tasks and spend in your history")
    sp.add_parser("families", help="Task families and their status")
    ds = sp.add_parser("dataset", help="Build a training set for a family (Hermes ShareGPT JSONL)")
    ds.add_argument("family")
    tr = sp.add_parser("trainer", help="Set up the training sidecar")
    tr.add_argument("action", choices=["setup", "status", "check"])
    tr.add_argument("--local", action="store_true", help="Also install the CUDA training stack for this machine")
    t = sp.add_parser("train", help="Train a student for a family")
    t.add_argument("family")
    t.add_argument("--yes", action="store_true", help="Skip the cost confirmation")
    t.add_argument("--backend", choices=["modal", "local", "dry-run"])
    t.add_argument("--follow", action="store_true", help="Stay attached and print progress")
    j = sp.add_parser("jobs", help="List training jobs")
    j.add_argument("--follow", action="store_true")
    c = sp.add_parser("cancel", help="Cancel a training job")
    c.add_argument("job")
    sp.add_parser("students", help="Trained students and how to load them")
    ev = sp.add_parser("eval", help="Test a student on held-out examples from your own sessions")
    ev.add_argument("student")
    ev.add_argument("--limit", type=int)
    for name in ("shadow", "promote", "demote"):
        p = sp.add_parser(name, help=f"{name.capitalize()} a family's student")
        p.add_argument("family")
    ex = sp.add_parser("export", help="Copy a family's dataset to a folder")
    ex.add_argument("family")
    ex.add_argument("--out", type=Path, required=True)
    pg = sp.add_parser("purge", help="Delete everything Tironian has captured")
    pg.add_argument("--yes", action="store_true")
    sub.set_defaults(func=run)
