"""Detached training driver. Runs in Tironian's sidecar venv, never inside Hermes.

Protocol with the plugin (``core/jobs.py``), all files inside one job directory:

* ``job.json``     — parameters written by the plugin (read once).
* ``status.jsonl`` — one JSON object per line, appended by this driver:
  ``{"t", "state", "progress", "message", ...}``. States: ``queued → uploading → starting →
  loading → training → exporting → exported → downloading → done`` or ``failed`` / ``cancelled``.
* ``cancel``       — created by the plugin to ask for cancellation.
* ``student.gguf`` — the result, written atomically when ``done``.

Backends: ``modal`` (default; credentials come from ``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET``
in this process's environment, passed by the plugin from Hermes's secret store and never
written to disk), ``local`` (CUDA on this machine, same ``train_core``), and ``dry-run``
(scripted states, no compute — used by tests and the demo rehearsal).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Job:
    def __init__(self, job_dir: Path) -> None:
        self.dir = job_dir
        self.params = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        self.status_path = job_dir / "status.jsonl"

    def emit(self, state: str, progress: float, message: str, **extra) -> None:
        rec = {"t": time.time(), "state": state, "progress": round(progress, 4), "message": message, **extra}
        with self.status_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    @property
    def cancelled(self) -> bool:
        return (self.dir / "cancel").exists()

    def finish(self, src: Path) -> None:
        dst = self.dir / "student.gguf"
        h = hashlib.sha256()
        with src.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if src.resolve() != dst.resolve():
            os.replace(src, dst)
        self.emit("done", 1.0, "student ready", sha256=h.hexdigest(), size_bytes=dst.stat().st_size,
                  gguf=str(dst))


def run_dry(job: Job) -> int:
    steps = [("uploading", 0.03, "uploading dataset"), ("starting", 0.05, "starting GPU container"),
             ("loading", 0.08, "loading base model"), ("training", 0.4, "step 20/50 · loss 0.912"),
             ("training", 0.8, "step 50/50 · loss 0.388"), ("exporting", 0.9, "quantizing to Q4_K_M"),
             ("downloading", 0.97, "downloading student")]
    for state, p, msg in steps:
        if job.cancelled:
            job.emit("cancelled", 1.0, "cancelled")
            return 1
        job.emit(state, p, msg)
        time.sleep(float(job.params.get("dry_run_delay", 0.05)))
    out = job.dir / "dry.gguf"
    out.write_bytes(b"GGUF" + b"\0" * 60)
    job.finish(out)
    return 0


def run_local(job: Job) -> int:
    p = job.params
    out = job.dir / "out"
    cmd = [sys.executable, str(HERE / "train_core.py"), "--data", str(job.dir / "train.jsonl"), "--base", p["base"],
           "--out", str(out), "--name", p["name"], "--epochs", str(p.get("epochs", 2)), "--seed", str(p.get("seed", 3407)),
           "--progress-file", str(job.status_path)]
    job.emit("starting", 0.02, "training on this machine's GPU")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(job.dir / "train.log", "wb"))
    while proc.poll() is None:
        if job.cancelled:
            proc.terminate()
            job.emit("cancelled", 1.0, "cancelled")
            return 1
        time.sleep(2)
    if proc.returncode != 0:
        tail = (job.dir / "train.log").read_text(errors="replace")[-600:]
        job.emit("failed", 1.0, f"local training failed (exit {proc.returncode})", log_tail=tail)
        return 1
    job.finish(out / f"{p['name']}.gguf")
    return 0


def run_modal(job: Job) -> int:
    p = job.params
    os.environ["TIRONIAN_GPU"] = str(p.get("gpu", "L4"))
    os.environ["TIRONIAN_TIMEOUT"] = str(int(p.get("timeout_s", 3600)))
    if not (os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")):
        job.emit("failed", 1.0, "Modal token missing: set it in Settings → Plugins → Tironian")
        return 1
    sys.path.insert(0, str(HERE))
    import modal  # noqa: E402
    import modal_app  # noqa: E402

    job_id = p["job_id"]
    remote_id = p.get("remote_id")
    try:
        if not remote_id:
            job.emit("uploading", 0.02, "uploading dataset to your Modal volume")
            with modal_app.volume.batch_upload(force=True) as b:
                b.put_file(str(job.dir / "train.jsonl"), f"/jobs/{job_id}/train.jsonl")
            job.emit("starting", 0.04, f"starting a {p.get('gpu', 'L4')} GPU container (first run builds the image)")
            with modal_app.app.run(detach=True):
                call = modal_app.train.spawn(job_id, p["base"], p["name"], float(p.get("epochs", 2)),
                                             int(p.get("seed", 3407)))
            remote_id = call.object_id
            job.emit("starting", 0.05, "job accepted by Modal", remote_id=remote_id)
        call = modal.FunctionCall.from_id(remote_id)
        last = None
        while True:
            if job.cancelled:
                try:
                    call.cancel()
                finally:
                    job.emit("cancelled", 1.0, "cancelled; the GPU container was stopped")
                return 1
            try:
                result = call.get(timeout=0)
                break
            except TimeoutError:
                pass
            except modal.exception.TimeoutError:  # type: ignore[attr-defined]
                pass
            rec = modal_app.progress_dict.get(job_id)
            if rec and rec != last and rec.get("state") not in ("failed",):
                job.emit(rec.get("state", "training"), float(rec.get("progress", 0)), rec.get("message", ""),
                         **{k: v for k, v in rec.items() if k in ("step", "max_steps", "loss")})
                last = rec
            time.sleep(5)
        job.emit("downloading", 0.96, "downloading your student model")
        tmp = job.dir / "student.gguf.part"
        with tmp.open("wb") as fh:
            for chunk in modal_app.volume.read_file(result["remote_gguf"]):
                fh.write(chunk)
        h = hashlib.sha256()
        with tmp.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if result.get("sha256") and h.hexdigest() != result["sha256"]:
            job.emit("failed", 1.0, "downloaded file does not match the trained model's checksum")
            return 1
        try:
            modal_app.volume.remove_file(f"/jobs/{job_id}", recursive=True)
        except Exception:
            pass
        job.finish(tmp)
        return 0
    except Exception as exc:  # report, never hang
        job.emit("failed", 1.0, f"{type(exc).__name__}: {exc}"[:500], remote_id=remote_id)
        return 1


def check() -> int:
    """Verify the Modal token without spending anything: open the (free) progress Dict."""
    if not (os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")):
        print(json.dumps({"ok": False, "message": "MODAL_TOKEN_ID / MODAL_TOKEN_SECRET not set in Hermes"}))
        return 1
    try:
        import modal
        d = modal.Dict.from_name("tironian-progress", create_if_missing=True)
        d.put("tironian-check", time.time())
        ws = ""
        try:
            from modal.config import _profile
            ws = _profile or ""
        except Exception:
            pass
        print(json.dumps({"ok": True, "message": f"Modal token works{(' (' + ws + ')') if ws else ''}"}))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "message": f"{type(exc).__name__}: {str(exc)[:300]}"}))
        return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", type=Path)
    ap.add_argument("--check", action="store_true", help="verify Modal credentials (no compute)")
    a = ap.parse_args(argv)
    if a.check:
        return check()
    if a.job_dir is None:
        ap.error("--job-dir is required")
    job = Job(a.job_dir)
    backend = job.params.get("backend", "modal")
    try:
        return {"modal": run_modal, "local": run_local, "dry-run": run_dry}[backend](job)
    except Exception as exc:
        job.emit("failed", 1.0, f"{type(exc).__name__}: {exc}"[:500])
        return 1


if __name__ == "__main__":
    sys.exit(main())
