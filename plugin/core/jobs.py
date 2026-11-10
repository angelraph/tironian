"""Training jobs, seen from inside Hermes.

Hermes never imports Modal or PyTorch. A job is a directory plus a detached driver process
in Tironian's own sidecar venv (``plugin-data/tironian/sidecar``); Hermes only writes
``job.json`` and reads ``status.jsonl``. That keeps the in-process plugin standard-library
only, survives Hermes restarts (any process can reconcile a job from its files), and means a
crash in the training stack can never take an agent turn down with it.

The sidecar is created only by an explicit ``hermes tironian trainer setup`` (it installs the
``modal`` client from PyPI). Training is only ever started by an explicit user action, never
from cron, the gateway, or a model turn.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import BASE_MODELS, MODAL_GPU_USD_PER_S, Settings
from .store import Store

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
TRAINER_DIR = PLUGIN_ROOT / "trainer"
TERMINAL = ("done", "failed", "cancelled")

#: Rough wall-clock model of a run, used for the estimate shown before anything is spent.
_OVERHEAD_S = {"qwen3-1.7b": 420, "qwen3-4b": 600}      # container start + load + GGUF export
_S_PER_EXAMPLE_EPOCH = {"qwen3-1.7b": 0.55, "qwen3-4b": 1.3}


#: What the trainer's processes may see of Hermes's environment, by name. Everything else is
#: left out: provider keys, bot tokens and any other secret Hermes holds never reach pip or
#: the third-party packages it installs. The Modal token is added explicitly, from Hermes's
#: secret store, only where it is needed.
_ALLOWED_ENV = frozenset({
    "PATH", "PATHEXT", "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "USER", "USERNAME", "LOGNAME",
    "TMPDIR", "TEMP", "TMP", "LANG", "LANGUAGE", "TZ", "TERM",
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA",
    "PROGRAMFILES", "PROGRAMFILES(X86)", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS",
    "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_RUNTIME_DIR",
    # Networks that need a proxy or their own certificates can't reach PyPI or Modal without these.
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
})
_ALLOWED_PREFIXES = ("LC_",)
#: Only for training on the user's own GPU: the CUDA stack, and Hugging Face for model downloads.
_LOCAL_TRAINING_PREFIXES = ("CUDA_", "NVIDIA_", "HF_", "HUGGING_FACE_", "TORCH_", "PYTORCH_")


def clean_env(extra: dict[str, str] | None = None, *, local_training: bool = False) -> dict[str, str]:
    """Environment for the trainer's processes, built from an allow-list.

    Only the names in ``_ALLOWED_ENV`` (and ``LC_*``) are copied from Hermes's environment, plus
    the CUDA and Hugging Face settings when training on the user's own GPU. ``extra`` adds what
    a call needs explicitly, which in practice is ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET``.
    Hermes's own Python settings (``PYTHONPATH`` and friends) are not on the list either, so pip
    doesn't mistake Hermes's packages for the sidecar's.
    """
    prefixes = _ALLOWED_PREFIXES + (_LOCAL_TRAINING_PREFIXES if local_training else ())
    env = {k: v for k, v in os.environ.items() if k in _ALLOWED_ENV or k.upper() in _ALLOWED_ENV
           or k.startswith(prefixes)}
    env.update({k: v for k, v in (extra or {}).items() if v})
    return env


def _modal_secrets(secrets: dict[str, str] | None) -> dict[str, str]:
    """Only the Modal token, whatever else a caller passes."""
    return {k: v for k, v in (secrets or {}).items() if k in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET") and v}


def sidecar_python(data_dir: Path) -> Path:
    base = data_dir / "sidecar"
    return base / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def sidecar_ready(data_dir: Path) -> bool:
    py = sidecar_python(data_dir)
    return py.exists() and (data_dir / "sidecar" / ".tironian-ready").exists()


def setup_sidecar(data_dir: Path, *, python: str | None = None, extra: str | None = None) -> tuple[bool, str]:
    """Create the sidecar venv and install the trainer client. Explicit user action only."""
    venv = data_dir / "sidecar"
    py = python or sys.executable
    try:
        if not sidecar_python(data_dir).exists():
            subprocess.run([py, "-m", "venv", str(venv)], check=True, capture_output=True, timeout=300, env=clean_env())
        req = TRAINER_DIR / ("requirements-train.txt" if extra == "local" else "requirements-sidecar.txt")
        subprocess.run([str(sidecar_python(data_dir)), "-m", "pip", "install", "--disable-pip-version-check",
                        "-q", "-r", str(req)], check=True, capture_output=True, timeout=1800,
                       env=clean_env(local_training=extra == "local"))
        if extra == "local":
            subprocess.run([str(sidecar_python(data_dir)), "-m", "pip", "install", "-q", "-r",
                            str(TRAINER_DIR / "requirements-sidecar.txt")], check=True, capture_output=True,
                           timeout=600, env=clean_env(local_training=True))
        (venv / ".tironian-ready").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        return True, f"trainer ready ({venv})"
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or b"").decode(errors="replace")[-400:]
        return False, f"setup failed: {tail.strip() or exc}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"setup failed: {exc}"


@dataclass
class Estimate:
    seconds: int
    usd: float
    gpu: str


def epochs_for(n_examples: int) -> int:
    """Passes over the data. A small family needs more of them: the first real run (64 examples,
    2 passes, 16 steps) learned the right tool calls but not the teacher's short answers.
    Aim for about 40 optimizer steps (batch 8), between 2 and 6 passes."""
    return max(2, min(6, -(-320 // max(n_examples, 1))))


def estimate(n_examples: int, s: Settings) -> Estimate:
    secs = int(_OVERHEAD_S[s.base_model] + n_examples * epochs_for(n_examples) * _S_PER_EXAMPLE_EPOCH[s.base_model])
    rate = MODAL_GPU_USD_PER_S.get(s.modal_gpu, MODAL_GPU_USD_PER_S["L4"])
    return Estimate(secs, round(secs * rate, 2) if s.train_backend == "modal" else 0.0, s.modal_gpu)


def start(store: Store, data_dir: Path, s: Settings, *, dataset_id: str, family_id: str,
          secrets: dict[str, str] | None = None, backend: str | None = None) -> str:
    ds = store.one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if ds is None:
        raise ValueError("unknown dataset")
    backend = backend or s.train_backend
    if backend != "dry-run" and not sidecar_ready(data_dir):
        raise RuntimeError("trainer not set up yet: run `hermes tironian trainer setup`")
    for r in store.q("SELECT id FROM jobs WHERE status NOT IN ('done','failed','cancelled')"):
        reconcile(store, data_dir, r["id"])   # a driver that died must not block new runs
    running = store.scalar("SELECT COUNT(*) FROM jobs WHERE status NOT IN ('done','failed','cancelled')", (), 0)
    if running:
        raise RuntimeError("a training job is already running")
    version = int(store.scalar("SELECT COUNT(*) FROM students WHERE family_id=?", (family_id,), 0)) + 1
    name = f"tironian-{family_id}-v{version}"
    job_id = "job-" + hashlib.sha1(f"{dataset_id}:{time.time()}".encode()).hexdigest()[:10]
    jdir = data_dir / "jobs" / job_id
    jdir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(ds["path"]) / "train.jsonl", jdir / "train.jsonl")
    est = estimate(int(ds["n_train"]), s)
    rate = MODAL_GPU_USD_PER_S.get(s.modal_gpu, MODAL_GPU_USD_PER_S["L4"])
    cap_s = int(max(600, min(3 * 3600, s.max_usd_per_run / rate)))   # the cost cap, enforced as a timeout
    params = {"job_id": job_id, "backend": backend, "base": BASE_MODELS[s.base_model], "name": name,
              "epochs": epochs_for(int(ds["n_train"])), "seed": 3407, "gpu": s.modal_gpu, "timeout_s": cap_s}
    (jdir / "job.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    store.upsert("jobs", {
        "id": job_id, "kind": "train", "backend": backend, "status": "queued", "remote_id": None,
        "dataset_id": dataset_id, "family_id": family_id, "params_json": json.dumps(params), "progress": 0.0,
        "message": "queued", "cost_est_usd": est.usd, "artifact_path": None, "sha256": None, "error": None,
        "started_at": time.time(), "ended_at": None})
    _spawn(data_dir, jdir, secrets or {}, dry=backend == "dry-run", local=backend == "local")
    return job_id


def _spawn(data_dir: Path, jdir: Path, secrets: dict[str, str], *, dry: bool, local: bool = False) -> None:
    py = str(sys.executable if dry else sidecar_python(data_dir))
    # A local run never talks to Modal, so it doesn't get the Modal token either.
    env = clean_env(None if local or dry else _modal_secrets(secrets), local_training=local)
    env["PYTHONUNBUFFERED"] = "1"
    log = open(jdir / "driver.log", "ab")
    kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": log, "stderr": log, "env": env,
                              "cwd": str(jdir), "close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED | NEW_GROUP | NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen([py, str(TRAINER_DIR / "driver.py"), "--job-dir", str(jdir)], **kwargs)
    (jdir / "driver.pid").write_text(str(proc.pid))


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _driver_gone(jdir: Path, params: dict[str, Any], last_t: float) -> bool:
    """True when the driver has stopped without writing a final state, so the job would
    otherwise look busy forever and block every later run."""
    pid_file = jdir / "driver.pid"
    if pid_file.exists():
        try:
            return not _pid_alive(int(pid_file.read_text().strip() or 0))
        except (OSError, ValueError):
            return False
    # Jobs started before the pid was recorded: silent for longer than the run could last.
    return time.time() - last_t > float(params.get("timeout_s") or 3600) + 900


def _log_tail(jdir: Path) -> str:
    try:
        lines = [x.strip() for x in (jdir / "driver.log").read_text(encoding="utf-8", errors="replace").splitlines()]
    except OSError:
        return ""
    lines = [x for x in lines if x and "DeprecationWarning" not in x and "set_event_loop_policy" not in x]
    return lines[-1][:300] if lines else ""


def check_modal(data_dir: Path, secrets: dict[str, str]) -> tuple[bool, str]:
    """Run the driver's credential check in the sidecar (no compute, no cost)."""
    if not sidecar_ready(data_dir):
        return False, "trainer not set up yet: run `hermes tironian trainer setup`"
    env = clean_env(_modal_secrets(secrets))
    try:
        out = subprocess.run([str(sidecar_python(data_dir)), str(TRAINER_DIR / "driver.py"), "--check"],
                             capture_output=True, text=True, timeout=120, env=env)
        line = (out.stdout.strip().splitlines() or ["{}"])[-1]
        res = json.loads(line)
        return bool(res.get("ok")), str(res.get("message") or out.stderr[-300:])
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return False, f"check failed: {exc}"


def resume(store: Store, data_dir: Path, job_id: str, secrets: dict[str, str] | None = None) -> str:
    """Reattach to a Modal run whose connection was lost, instead of paying to train again.

    The GPU job keeps running on Modal when this machine loses touch with it (a network blip, the
    laptop sleeping). Its id is in the job's status, so a new driver can wait for the same run
    and download its result.
    """
    row = store.one("SELECT * FROM jobs WHERE id=?", (job_id,))
    if row is None:
        raise ValueError(f"unknown job {job_id}")
    if row["backend"] != "modal":
        raise ValueError("only Modal jobs can be reattached; start a new run instead")
    if row["status"] not in ("failed", "timeout"):
        raise ValueError(f"{job_id} is {row['status']}; only a failed job can be resumed")
    other = active_job(store)
    if other is not None:
        raise ValueError(f"{other['id']} is still running; wait for it to finish first")
    jdir = data_dir / "jobs" / job_id
    remote = row["remote_id"] or next((r["remote_id"] for r in reversed(read_status(jdir)) if r.get("remote_id")), None)
    if not remote:
        raise ValueError("that job never reached Modal, so there is nothing to reattach to; start a new run")
    params = json.loads((jdir / "job.json").read_text(encoding="utf-8"))
    params["remote_id"] = remote
    (jdir / "job.json").write_text(json.dumps(params), encoding="utf-8")
    (jdir / "cancel").unlink(missing_ok=True)
    with (jdir / "status.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": time.time(), "state": "starting", "progress": 0.05,
                             "message": f"reattaching to Modal run {remote}", "remote_id": remote}) + "\n")
    with store.tx() as c:
        c.execute("UPDATE jobs SET status='starting', message=?, ended_at=NULL, error=NULL, remote_id=? WHERE id=?",
                  (f"reattaching to Modal run {remote}", remote, job_id))
    _spawn(data_dir, jdir, secrets or {}, dry=False)
    return remote


def cancel(data_dir: Path, job_id: str) -> None:
    jdir = data_dir / "jobs" / job_id
    if jdir.exists():
        (jdir / "cancel").write_text("1")


def read_status(jdir: Path) -> list[dict[str, Any]]:
    p = jdir / "status.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a line being written right now
    return out


def reconcile(store: Store, data_dir: Path, job_id: str) -> dict[str, Any] | None:
    """Fold ``status.jsonl`` into the jobs row. Safe to call from any process, any time."""
    row = store.one("SELECT * FROM jobs WHERE id=?", (job_id,))
    if row is None:
        return None
    if row["status"] in TERMINAL:
        return dict(row)
    jdir = data_dir / "jobs" / job_id
    recs = read_status(jdir)
    params = json.loads(row["params_json"] or "{}")
    last_t = recs[-1].get("t", row["started_at"]) if recs else row["started_at"]
    if (not recs or recs[-1].get("state") not in TERMINAL) and _driver_gone(jdir, params, float(last_t or 0)):
        why = _log_tail(jdir) or "no details in driver.log"
        if (jdir / "cancel").exists():
            recs.append({"t": time.time(), "state": "cancelled", "progress": 1.0, "message": "cancelled"})
        else:
            recs.append({"t": time.time(), "state": "failed", "progress": 1.0,
                         "message": f"the trainer stopped without finishing: {why}"})
    if not recs:
        return dict(row)
    last = recs[-1]
    remote = next((r["remote_id"] for r in reversed(recs) if r.get("remote_id")), row["remote_id"])
    upd = {"status": last["state"], "progress": last.get("progress", row["progress"]),
           "message": last.get("message", ""), "remote_id": remote}
    if last["state"] in TERMINAL:
        upd["ended_at"] = last.get("t", time.time())
        if last["state"] == "done":
            upd["artifact_path"] = last.get("gguf")
            upd["sha256"] = last.get("sha256")
        else:
            upd["error"] = last.get("message")
    with store.tx() as c:
        c.execute(f"UPDATE jobs SET {', '.join(f'{k}=?' for k in upd)} WHERE id=?", (*upd.values(), job_id))
    return dict(store.one("SELECT * FROM jobs WHERE id=?", (job_id,)))


def active_job(store: Store) -> Any:
    return store.one("SELECT * FROM jobs WHERE status NOT IN ('done','failed','cancelled') ORDER BY started_at DESC LIMIT 1")


def install_student(store: Store, data_dir: Path, job_id: str) -> Path | None:
    """Move a finished job's GGUF to ``students/`` under its versioned name; never overwrite."""
    row = store.one("SELECT * FROM jobs WHERE id=?", (job_id,))
    if row is None or row["status"] != "done" or not row["artifact_path"]:
        return None
    params = json.loads(row["params_json"])
    dst_dir = data_dir / "students"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{params['name']}.gguf"
    src = Path(row["artifact_path"])
    if not dst.exists() and src.exists():
        os.replace(src, dst)
    return dst if dst.exists() else None
