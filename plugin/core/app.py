"""Process-wide Tironian application object.

Hermes may load Tironian in several processes at once (the gateway behind Desktop and
messaging, a ``hermes chat`` CLI, the dashboard backend). Each gets its own :class:`App`; they
share state only through the SQLite store and the job directories, both of which are safe
for concurrent use.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .config import PLUGIN_ID, Settings
from .engine import Engine, reentrant
from .redact import Redactor
from .store import Store
from .student import StudentClient

log = logging.getLogger("tironian")

_APP: "App | None" = None
_APP_LOCK = threading.Lock()


def _default_data_dir() -> Path:
    try:
        from plugins.plugin_storage import plugin_data_dir  # Hermes public storage API
        return Path(plugin_data_dir(PLUGIN_ID))
    except Exception:
        home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
        p = Path(home) / "plugin-data" / PLUGIN_ID
        p.mkdir(parents=True, exist_ok=True)
        return p


def _config_reader() -> Callable[..., Any]:
    """Read ``plugins.entries.tironian.settings`` outside a plugin ctx (dashboard backend)."""
    def get(key: str, default: Any = None) -> Any:
        try:
            from hermes_cli.config import load_config_readonly
            cfg = load_config_readonly() or {}
            return ((cfg.get("plugins") or {}).get("entries") or {}).get(PLUGIN_ID, {}).get("settings", {}).get(key, default)
        except Exception:
            return default
    return get


def _secret(name: str) -> str | None:
    try:
        from agent.secret_scope import get_secret
        return get_secret(name)
    except Exception:
        return None


def _broadcast(kind: str, payload: dict[str, Any]) -> None:
    try:
        from hermes_cli.plugin_events import broadcast_plugin_event
        broadcast_plugin_event(PLUGIN_ID, kind, payload)
    except Exception:
        pass


class App:
    def __init__(self, *, data_dir: Path | None = None, config_get: Callable[..., Any] | None = None,
                 llm: Any = None, secret: Callable[[str], str | None] | None = None,
                 broadcast: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        self.data_dir = data_dir or _default_data_dir()
        self._get = config_get or _config_reader()
        self.llm = llm
        self.secret = secret or _secret
        self.broadcast = broadcast or _broadcast
        self.store = Store(self.data_dir / "tironian.db")
        self._settings: Settings | None = None
        self._settings_at = 0.0
        s = self.settings()
        self.redactor = Redactor(pii=s.redact_pii)
        self.engine = Engine(self.store, self.settings, self.redactor, on_event=self.broadcast)
        self._bg_lock = threading.Lock()
        self.bg_status: dict[str, Any] = {}

    # Settings are re-read at most every 5 s so a Desktop save takes effect without a restart.
    def settings(self) -> Settings:
        now = time.monotonic()
        if self._settings is None or now - self._settings_at > 5:
            self._settings = Settings.load(self._get)
            self._settings_at = now
        return self._settings

    def reload_settings(self) -> None:
        self._settings = None

    # Paths
    @property
    def state_db(self) -> Path:
        """Hermes's session store for this profile: ``<HERMES_HOME>/state.db``."""
        return self.data_dir.parent.parent / "state.db"

    def student_client(self, model_id: str | None = None) -> StudentClient:
        from .serve import resolve_endpoint
        s = self.settings()
        return StudentClient(resolve_endpoint(s.student_endpoint, model_id), timeout_s=s.student_timeout_s)

    def serve_student(self, student_id: str) -> tuple[bool, str]:
        """Make a student available to the local model server: already listed, or load it into Ollama."""
        from . import serve
        st = self.store.one("SELECT * FROM students WHERE id=?", (student_id,))
        if st is None:
            return False, "unknown student"
        from .student import model_matches
        if any(model_matches(m, st["model_id"]) for m in self.student_client(st["model_id"]).available_models()):
            return True, "already loaded"
        ok, msg = serve.load_into_ollama(Path(st["gguf_path"]), st["model_id"])
        serve.forget()
        if ok:
            self.broadcast("student.served", {"student": student_id})
        return ok, msg

    # Judge
    def judge(self) -> Callable[[str, str, str], str | None] | None:
        s = self.settings()
        if not s.judge_enabled or self.llm is None:
            return None
        budget = {"left": s.judge_daily_budget}

        def ask(request: str, a: str, b: str) -> str | None:
            if budget["left"] <= 0:
                return None
            budget["left"] -= 1
            schema = {"type": "object", "properties": {"better": {"type": "string", "enum": ["a", "b", "tie"]}},
                      "required": ["better"]}
            try:
                with reentrant():
                    res = self.llm.complete_structured(
                        instructions=("Two assistant replies answer the same user request. A reply may be a "
                                      "final answer or a tool call the assistant is about to make. Judge only "
                                      "whether each would correctly serve that request. Answer 'tie' when both "
                                      "are equally acceptable."),
                        input=[{"type": "text", "text": f"REQUEST:\n{request[:3000]}\n\nREPLY A:\n{a[:3000]}\n\nREPLY B:\n{b[:3000]}"}],
                        json_schema=schema, task="tironian_judge", purpose="tironian held-out evaluation",
                        temperature=0, max_tokens=50)
                parsed = getattr(res, "parsed", None) or {}
                v = parsed.get("better") if isinstance(parsed, dict) else None
                return v if v in ("a", "b", "tie") else None
            except Exception as exc:
                log.debug("judge call failed", exc_info=True)
                self.judge_error = f"{type(exc).__name__}: {exc}"[:300]
                return None
        return ask

    judge_error: str | None = None

    # First run
    def first_run(self) -> dict[str, Any] | None:
        """Import history once (read-only) and build families, in the background."""
        if self.store.meta_get("first_import_done"):
            return None
        return self.run_bg("import", self._do_import)

    def _do_import(self) -> dict[str, Any]:
        from . import jobs, pipeline
        res = pipeline.import_history(self.store, self.state_db, self.redactor)
        fams = pipeline.rebuild_families(self.store, self.settings())
        self.store.meta_set("first_import_done", str(time.time()))
        rep = pipeline.report(self.store, self.settings()).to_dict()
        self.broadcast("report.ready", {"turns": rep["turns"], "routine_share": rep["routine_share"]})
        return {"import": res, "families": len(fams)}

    def run_bg(self, name: str, fn: Callable[[], Any]) -> dict[str, Any]:
        if not self._bg_lock.acquire(blocking=False):
            return {"started": False, "reason": f"busy: {self.bg_status.get('task')}"}
        self.bg_status = {"task": name, "state": "running", "started": time.time()}

        def run() -> None:
            try:
                out = fn()
                self.bg_status = {"task": name, "state": "done", "result": out, "ended": time.time()}
            except Exception as exc:
                log.warning("tironian %s failed", name, exc_info=True)
                self.bg_status = {"task": name, "state": "failed", "error": str(exc), "ended": time.time()}
            finally:
                self._bg_lock.release()
                self.broadcast("task.updated", {"task": name, "state": self.bg_status.get("state")})

        threading.Thread(target=run, name=f"tironian-{name}", daemon=True).start()
        return {"started": True}

    # Training flow
    def modal_secrets(self) -> dict[str, str]:
        return {k: v for k in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET") if (v := self.secret(k) or "")}

    def prepare_training(self, family_id: str) -> dict[str, Any]:
        from . import jobs, pipeline
        s = self.settings()
        ds = pipeline.build_dataset(self.store, s, family_id, self.data_dir / "datasets")
        est = jobs.estimate(ds.n_train, s)
        return {"dataset_id": ds.id, "n_train": ds.n_train, "n_heldout": ds.n_heldout, "skipped": ds.skipped,
                "estimate_s": est.seconds, "estimate_usd": est.usd, "gpu": est.gpu, "backend": s.train_backend,
                "enough": ds.n_train >= 8 and ds.n_heldout >= 1}

    def start_training(self, family_id: str, dataset_id: str, *, backend: str | None = None) -> str:
        from . import jobs, pipeline
        s = self.settings()
        if (backend or s.train_backend) == "modal" and len(self.modal_secrets()) < 2:
            raise RuntimeError("add your Modal token in Settings → Plugins → Tironian first")
        jid = jobs.start(self.store, self.data_dir, s, dataset_id=dataset_id, family_id=family_id,
                         secrets=self.modal_secrets(), backend=backend)
        self.broadcast("job.updated", {"job": jid, "state": "queued"})
        return jid

    def poll_job(self, job_id: str) -> dict[str, Any] | None:
        from . import jobs, pipeline
        row = jobs.reconcile(self.store, self.data_dir, job_id)
        if row and row["status"] == "done" and not self.store.one("SELECT 1 FROM students WHERE job_id=?", (job_id,)):
            path = jobs.install_student(self.store, self.data_dir, job_id)
            if path:
                params = json.loads(row["params_json"])
                pipeline.register_student(self.store, job_id=job_id, family_id=row["family_id"],
                                          base_model=params["base"], gguf_path=path, sha256=row["sha256"] or "")
                self.broadcast("student.ready", {"job": job_id, "path": str(path)})
                st = self.store.one("SELECT id FROM students WHERE job_id=?", (job_id,))
                if st and params.get("backend") != "dry-run":  # load it for the user, in the background
                    threading.Thread(target=self.serve_student, args=(st["id"],), daemon=True).start()
        return row

    def evaluate(self, student_id: str, *, limit: int | None = None) -> dict[str, Any]:
        from . import jobs, pipeline
        st = self.store.one("SELECT * FROM students WHERE id=?", (student_id,))
        if st is None:
            raise ValueError("unknown student")
        job = self.store.one("SELECT dataset_id FROM jobs WHERE id=?", (st["job_id"],)) if st["job_id"] else None
        ds_id = job["dataset_id"] if job else self.store.scalar(
            "SELECT id FROM datasets WHERE family_ids LIKE ? ORDER BY created_at DESC LIMIT 1",
            (f'%"{st["family_id"]}"%',))
        from .student import model_matches

        def listed(c: StudentClient) -> bool:
            return any(model_matches(m, st["model_id"]) for m in c.available_models())
        # The test measures answers, not the routing deadline: a slow CPU gets time to finish, and
        # its speed is still reported (p50) so the user can see it.
        def eval_client() -> StudentClient:
            c = self.student_client(st["model_id"])
            c.timeout_s = max(180.0, float(self.settings().student_timeout_s))
            return c
        client = eval_client()
        if not listed(client):
            ok, msg = self.serve_student(student_id)
            client = eval_client()
            if not listed(client):
                raise RuntimeError(
                    f"the local model server doesn't list '{st['model_id']}' ({msg}). Install Ollama, or load "
                    f"{st['gguf_path']} into any OpenAI-compatible server and set student_endpoint to it")

        def progress(i: int, n: int) -> None:
            if i == n or i % 5 == 0:
                self.broadcast("eval.progress", {"student": student_id, "done": i, "total": n})
        judge = self.judge()
        self.judge_error = None
        sc = pipeline.evaluate_student(self.store, self.settings(), student_id=student_id, dataset_id=ds_id,
                                       client=client, judge=judge, limit=limit, progress=progress)
        s = self.settings()
        judge_state = ("off" if not s.judge_enabled else "unavailable (no model access)" if judge is None
                       else f"failed: {self.judge_error}" if self.judge_error else "on")
        self.engine.invalidate()
        return {"n": sc.n, "passed": sc.passed, "wilson_lo": sc.wilson_lo, "wilson_hi": sc.wilson_hi,
                "validity": sc.validity, "gate_passed": sc.gate_passed, "gate_reason": sc.gate_reason,
                "p50_ms": sc.p50_ms, "failures": sc.failures, "steps": sc.steps, "serve_steps": sc.serve_steps,
                "judge": judge_state}


def get_app(**kwargs: Any) -> App:
    global _APP
    with _APP_LOCK:
        if _APP is None:
            _APP = App(**kwargs)
        elif kwargs.get("llm") is not None and _APP.llm is None:
            _APP.llm = kwargs["llm"]
        return _APP


def reset_app() -> None:
    """Tests only."""
    global _APP
    with _APP_LOCK:
        if _APP is not None:
            _APP.store.close()
        _APP = None
