"""``hermes tironian ...``: the whole pipeline from a terminal."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from . import commands, jobs, pipeline
from .app import get_app


from .cli_args import setup  # noqa: E402,F401  (kept for callers that import cli.setup)


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
                    ok, msg = app.serve_student(st["id"])
                    _p(f"Local model server: {msg}" if ok else
                       f"Couldn't load it automatically ({msg}). Load the file into any OpenAI-compatible server.")
                    _p(f"Next: hermes tironian eval {st['id']}")
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
        from .student import model_matches
        available = app.student_client().available_models()
        for r in rows:
            loaded = "loaded" if any(model_matches(m, r["model_id"]) for m in available) else "not loaded"
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
