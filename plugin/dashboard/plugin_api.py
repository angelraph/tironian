"""Backend for Tironian's Desktop page, mounted by Hermes at /api/plugins/tironian/.

It runs inside the gateway process and reads the same SQLite store the hooks write to.
Every write here is something the user clicked on the page.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parent.parent


def _pkg() -> Any:
    """Import the plugin package by path (the dashboard loads this file on its own)."""
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f and Path(f).resolve() == (ROOT / "__init__.py").resolve():
            return mod
    name = "tironian_dashboard_pkg"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _app() -> Any:
    pkg = _pkg()
    app_mod = importlib.import_module(f"{pkg.__name__}.core.app")
    return app_mod.get_app()


def _mods() -> tuple[Any, Any, Any]:
    pkg = _pkg()
    return (importlib.import_module(f"{pkg.__name__}.core.pipeline"),
            importlib.import_module(f"{pkg.__name__}.core.jobs"),
            importlib.import_module(f"{pkg.__name__}.core.commands"))


router = APIRouter()


@router.get("/summary")
def summary() -> dict[str, Any]:
    app = _app()
    pipeline, jobs, _ = _mods()
    app.first_run()
    s = app.settings()
    job = jobs.active_job(app.store)
    job_row = app.poll_job(job["id"]) if job else None
    rep = pipeline.report(app.store, s).to_dict()
    daily = app.store.q(
        "SELECT date(created_at, 'unixepoch', 'localtime') AS day, "
        "SUM(outcome='served') AS served, SUM(mode='live' AND outcome!='served') AS fallbacks "
        "FROM routes WHERE created_at > ? GROUP BY day ORDER BY day", (time.time() - 14 * 86400,))
    return {
        "stats": app.engine.summary(),
        "report": rep,
        "settings": {"routing_enabled": s.routing_enabled, "train_backend": s.train_backend,
                     "promotion_threshold": s.promotion_threshold, "max_usd_per_run": s.max_usd_per_run},
        "trainer_ready": jobs.sidecar_ready(app.data_dir),
        "job": dict(job_row) if job_row else None,
        "task": app.bg_status,
        "daily": [dict(r) for r in daily],
    }


@router.get("/families")
def families() -> dict[str, Any]:
    app = _app()
    rows = app.store.q(
        "SELECT f.*, e.n AS eval_n, e.passed AS eval_passed, e.wilson_lo, e.wilson_hi, e.validity, e.gate_passed, "
        "e.p50_ms, e.report_json "
        "FROM families f LEFT JOIN evals e ON e.id = (SELECT id FROM evals x WHERE x.family_id=f.id "
        "ORDER BY created_at DESC LIMIT 1) ORDER BY f.n_turns DESC LIMIT 40")
    out = []
    for r in rows:
        d = dict(r)
        d["tools"] = json.loads(d.pop("toolset_json") or "[]")
        d.pop("exemplars_json", None)
        rep = json.loads(d.pop("report_json") or "null")
        d["gate_reason"] = rep.get("gate_reason") if rep else None
        d["failures"] = rep.get("failures") if rep else None
        served = app.store.scalar("SELECT COUNT(*) FROM routes WHERE family_id=? AND outcome='served'", (r["id"],), 0)
        shadow = app.store.q("SELECT outcome FROM routes WHERE family_id=? AND mode='shadow' ORDER BY id DESC LIMIT 200", (r["id"],))
        d["served"] = served
        d["shadow_n"] = len(shadow)
        d["shadow_agree"] = sum(1 for x in shadow if x["outcome"] == "shadow_agree")
        out.append(d)
    return {"families": out, "min_family_turns": app.settings().min_family_turns}


@router.post("/families/{family_id}/mode")
def set_mode(family_id: str, body: dict[str, Any]) -> dict[str, Any]:
    app = _app()
    _, _, commands = _mods()
    mode = str(body.get("mode") or "")
    if mode not in ("shadow", "promote", "demote"):
        raise HTTPException(400, "mode must be shadow, promote or demote")
    return {"message": commands.set_status(app, mode, family_id)}


@router.post("/refresh")
def refresh() -> dict[str, Any]:
    app = _app()
    pipeline, _, _ = _mods()

    def run() -> dict[str, Any]:
        res = pipeline.import_history(app.store, app.state_db, app.redactor)
        fams = pipeline.rebuild_families(app.store, app.settings())
        app.store.meta_set("first_import_done", str(time.time()))
        return {"import": res, "families": len(fams)}
    return app.run_bg("refresh", run)


@router.post("/train/prepare")
def train_prepare(body: dict[str, Any]) -> dict[str, Any]:
    app = _app()
    try:
        return app.prepare_training(str(body.get("family") or ""))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/train/start")
def train_start(body: dict[str, Any]) -> dict[str, Any]:
    """Starts a paid run only with the dataset the user just saw the estimate for."""
    app = _app()
    try:
        jid = app.start_training(str(body["family"]), str(body["dataset_id"]))
    except (KeyError, ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"job": jid}


@router.post("/train/cancel")
def train_cancel(body: dict[str, Any]) -> dict[str, Any]:
    app = _app()
    _, jobs, _ = _mods()
    jobs.cancel(app.data_dir, str(body.get("job") or ""))
    return {"ok": True}


@router.get("/students")
def students() -> dict[str, Any]:
    app = _app()
    model_matches = importlib.import_module(f"{_pkg().__name__}.core.student").model_matches
    loaded = app.student_client().available_models()
    rows = [dict(r) | {"loaded": any(model_matches(m, r["model_id"]) for m in loaded)}
            for r in app.store.q("SELECT * FROM students ORDER BY created_at DESC LIMIT 20")]
    return {"students": rows}


@router.post("/eval")
def evaluate(body: dict[str, Any]) -> dict[str, Any]:
    app = _app()
    student = str(body.get("student") or "")
    return app.run_bg("eval", lambda: app.evaluate(student))
