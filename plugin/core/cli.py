"""``hermes tironian ...``: the whole pipeline from a terminal."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from . import commands, jobs, pipeline
from .app import get_app


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


def _p(text: str = "") -> None:
    print(text, flush=True)


def _follow(app: Any, job_id: str) -> int:
    last = None
    while True:
        row = app.poll_job(job_id)
        if row is None:
            _p("unknown job")
            return 1
        line = f"[{row['status']:<11}] {row['progress'] * 100:5.1f}%  {row['message'] or ''}"
        if line != last:
            _p(line)
            last = line
        if row["status"] in jobs.TERMINAL:
            if row["status"] == "done":
                st = app.store.one("SELECT * FROM students WHERE job_id=?", (job_id,))
                if st:
                    _p(f"\nStudent ready: {st['gguf_path']}")
                    _p("Load it: Hermes Desktop → Settings → Providers → Local Models → Add model file,")
                    _p(f"then run: hermes tironian eval {st['id']}")
            return 0 if row["status"] == "done" else 1
        time.sleep(3)


def run(args: Any) -> int:
    app = get_app()
    cmd = getattr(args, "tironian_cmd", None) or "status"
    s = app.settings()
    if cmd == "status":
        _p(commands.status_text(app))
    elif cmd == "import":
        res = pipeline.import_history(app.store, app.state_db, app.redactor, since_days=args.days)
        fams = pipeline.rebuild_families(app.store, s)
        app.store.meta_set("first_import_done", str(time.time()))
        _p(f"Imported {res['sessions']} sessions, {res['turns']} requests, {res['decisions']} model steps "
           f"(read-only from {app.state_db}). {len(fams)} task families.")
    elif cmd == "report":
        pipeline.rebuild_families(app.store, s)
        _p(commands.report_text(app))
    elif cmd == "families":
        _p(commands.families_text(app))
    elif cmd == "dataset":
        ds = pipeline.build_dataset(app.store, s, args.family, app.data_dir / "datasets")
        _p(f"{ds.id}: {ds.n_train} train / {ds.n_heldout} held-out examples → {ds.train_path.parent}")
        if ds.skipped:
            _p(f"skipped: {json.dumps(ds.skipped)}")
    elif cmd == "trainer":
        if args.action == "status":
            _p("ready" if jobs.sidecar_ready(app.data_dir) else "not set up: run `hermes tironian trainer setup`")
        elif args.action == "check":
            ok, msg = jobs.check_modal(app.data_dir, app.modal_secrets())
            _p(("✓ " if ok else "✗ ") + msg)
            return 0 if ok else 1
        else:
            _p("Creating Tironian's trainer environment (installs the Modal client from PyPI)…")
            ok, msg = jobs.setup_sidecar(app.data_dir, extra="local" if args.local else None)
            _p(msg)
            return 0 if ok else 1
    elif cmd == "train":
        prep = app.prepare_training(args.family)
        backend = args.backend or s.train_backend
        _p(f"{args.family}: {prep['n_train']} training / {prep['n_heldout']} held-out examples")
        if not prep["enough"]:
            _p(f"Not enough data yet (skipped: {json.dumps(prep['skipped'])}).")
            return 1
        cost = "free (your GPU)" if backend in ("local", "dry-run") else f"~${prep['estimate_usd']:.2f} on Modal {prep['gpu']}"
        _p(f"Estimate: ~{prep['estimate_s'] // 60} min, {cost}; hard cap ${s.max_usd_per_run:.2f}.")
        if not args.yes:
            if not sys.stdin.isatty():
                _p("Re-run with --yes to start (no terminal to confirm in).")
                return 1
            if input("Start training? [y/N] ").strip().lower() not in ("y", "yes"):
                return 1
        jid = app.start_training(args.family, prep["dataset_id"], backend=backend)
        _p(f"Started {jid}.")
        if args.follow:
            return _follow(app, jid)
    elif cmd == "jobs":
        rows = app.store.q("SELECT * FROM jobs ORDER BY started_at DESC LIMIT 10")
        for r in rows:
            r = app.poll_job(r["id"]) or r
            _p(f"{r['id']}  {r['family_id']:<10} {r['backend']:<7} {r['status']:<11} {r['progress'] * 100:5.1f}%  {r['message'] or ''}")
        if args.follow and rows:
            return _follow(app, rows[0]["id"])
        if not rows:
            _p("No training jobs yet.")
    elif cmd == "cancel":
        jobs.cancel(app.data_dir, args.job)
        _p("Cancellation requested.")
    elif cmd == "students":
        rows = app.store.q("SELECT * FROM students ORDER BY created_at DESC")
        available = set(app.student_client().available_models())
        for r in rows:
            loaded = "loaded" if r["model_id"] in available else "not loaded"
            _p(f"{r['id']}  family {r['family_id']}  {r['size_bytes'] / 1e9:.2f} GB  {loaded}\n  {r['gguf_path']}")
        if not rows:
            _p("No students yet: `hermes tironian train <family>`.")
    elif cmd == "eval":
        res = app.evaluate(args.student, limit=args.limit)
        _p(f"{res['passed']}/{res['n']} held-out steps passed · 95% CI {res['wilson_lo']:.0%} to {res['wilson_hi']:.0%} "
           f"· format valid {res['validity']:.0%} · p50 {res['p50_ms']} ms")
        _p(("GATE PASSED: " if res["gate_passed"] else "GATE NOT PASSED: ") + res["gate_reason"])
        if res["failures"]:
            _p("misses: " + ", ".join(f"{k} ×{v}" for k, v in sorted(res["failures"].items(), key=lambda kv: -kv[1])))
    elif cmd in ("shadow", "promote", "demote"):
        _p(commands.set_status(app, cmd, args.family))
    elif cmd == "export":
        ds = pipeline.build_dataset(app.store, s, args.family, args.out)
        _p(f"Wrote {ds.n_train + ds.n_heldout} examples to {ds.train_path.parent} (sha256 {ds.sha256[:12]}…)")
    elif cmd == "purge":
        if not args.yes:
            _p("This deletes every captured turn, dataset row and route record. Re-run with --yes.")
            return 1
        app.store.purge_all()
        app.store.meta_set("first_import_done", "")
        _p("Purged.")
    return 0
