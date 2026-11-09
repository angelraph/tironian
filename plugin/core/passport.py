"""A student's passport: where it came from and what it has proven, in one checkable document.

``hermes tironian passport <student>`` writes it as JSON and Markdown. It holds the model file's
SHA-256 (checked again against the file on disk), the training set's SHA-256, the training run,
the held-out test result per kind of step, shadow mode, spot checks, live use and the family's
status history. A ``seal`` over all of that is the SHA-256 of its canonical JSON, so changing any
number afterwards is detectable: ``hermes tironian passport --verify <file.json>``.

It reveals no conversations: only counts, scores, hashes and names.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .store import Store

VERSION = 1


def _sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def seal(body: dict[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build(store: Store, student_id: str) -> dict[str, Any]:
    st = store.one("SELECT * FROM students WHERE id=?", (student_id,))
    if st is None:
        raise ValueError(f"unknown student {student_id}")
    fam = store.one("SELECT * FROM families WHERE id=?", (st["family_id"],))
    job = store.one("SELECT * FROM jobs WHERE id=?", (st["job_id"],)) if st["job_id"] else None
    ds = store.one("SELECT * FROM datasets WHERE id=?", (job["dataset_id"],)) if job else None
    params = json.loads(job["params_json"] or "{}") if job else {}

    on_disk = _sha256_file(Path(st["gguf_path"])) if st["gguf_path"] else None
    model = {"id": st["id"], "base": st["base_model"], "size_bytes": st["size_bytes"], "sha256": st["sha256"],
             "file_matches": (on_disk == st["sha256"]) if (on_disk and st["sha256"]) else None}
    training = None
    if job:
        training = {"job": job["id"], "backend": job["backend"], "gpu": params.get("gpu"), "epochs": params.get("epochs"),
                    "status": job["status"], "remote_id": job["remote_id"], "cost_estimate_usd": job["cost_est_usd"],
                    "started": _iso(job["started_at"]), "ended": _iso(job["ended_at"])}
    data = {"dataset": ds["id"], "sha256": ds["sha256"], "train_examples": ds["n_train"],
            "heldout_examples": ds["n_heldout"], "seed": ds["seed"], "split": "by session"} if ds else None

    tests = []
    for e in store.q("SELECT * FROM evals WHERE student_id=? ORDER BY created_at", (student_id,)):
        rep = json.loads(e["report_json"] or "{}")
        steps = {k: {"passed": v.get("passed"), "n": v.get("n"), "lower_bound": round(float(v.get("wilson_lo") or 0), 4),
                     "passes": bool(v.get("gate_passed"))} for k, v in (rep.get("steps") or {}).items()}
        tests.append({"at": _iso(e["created_at"]), "steps": steps, "passed": e["passed"], "n": e["n"],
                      "lower_bound": round(float(e["wilson_lo"] or 0), 4), "gate_passed": bool(e["gate_passed"]),
                      "serves": rep.get("serve_steps") or [], "fresh_steps": rep.get("fresh", 0),
                      "reason": rep.get("gate_reason")})

    def tally(mode: str) -> dict[str, int]:
        rows = store.q("SELECT outcome, COUNT(*) AS n FROM routes WHERE student_id=? AND mode=? GROUP BY outcome",
                       (st["model_id"], mode))
        return {r["outcome"]: r["n"] for r in rows}

    family = None
    if fam:
        family = {"id": fam["id"], "name": fam["name"], "status": fam["status"],
                  "status_reason": store.meta_get(f"status_reason:{fam['id']}") or None,
                  "serving_this_student": fam["student_id"] == st["id"]}
    np = json.loads(store.meta_get(f"new_places:{st['family_id']}") or "{}")
    new_places = None
    if np.get("student") == st["id"]:
        new_places = {"trained_on": np.get("trained_on", []), "new_trees": np.get("trees", []),
                      "steps": {k: {"passed": d["passed"], "n": d["n"]} for k, d in (np.get("steps") or {}).items()},
                      "at": _iso(np.get("at"))}
    body = {"passport": VERSION, "new_folders": new_places, "issued": _iso(time.time()), "tool": "tironian", "model": model, "family": family,
            "training": training, "data": data, "held_out_tests": tests,
            "shadow": tally("shadow"), "spot_checks": tally("spot"), "live": tally("live"),
            "forgotten_sessions": store.meta_get(f"forgotten_in:{st['id']}") is not None}
    return {**body, "seal": seal(body)}


def verify(doc: dict[str, Any]) -> tuple[bool, str]:
    body = {k: v for k, v in doc.items() if k != "seal"}
    if doc.get("seal") != seal(body):
        return False, "the seal doesn't match: something in this passport was changed after it was issued"
    return True, f"sealed and unchanged (issued {doc.get('issued')})"


def markdown(doc: dict[str, Any]) -> str:
    m, f, t, d = doc["model"], doc.get("family") or {}, doc.get("training") or {}, doc.get("data") or {}
    lines = [f"# Model passport: {m['id']}", "",
             f"Issued {doc['issued']} by Tironian. Seal `{doc['seal'][:16]}…` "
             f"(`hermes tironian passport --verify` checks it).", "",
             "| | |", "|---|---|",
             f"| Task family | {f.get('name', '?')} (`{f.get('id', '?')}`), now **{f.get('status', '?')}** |",
             f"| Base model | {m['base']} |",
             f"| Model file | SHA-256 `{(m['sha256'] or '?')[:16]}…`, "
             f"{'matches the file on disk' if m['file_matches'] else 'file not checked' if m['file_matches'] is None else 'DOES NOT match the file on disk'} |"]
    if t:
        lines.append(f"| Training | {t.get('backend')} {t.get('gpu') or ''}, {t.get('epochs')} epochs, "
                     f"about ${t.get('cost_estimate_usd') or 0:.2f}, {t.get('ended') or t.get('started')} |")
    if d:
        lines.append(f"| Data | {d['train_examples']} training and {d['heldout_examples']} held-out examples, split "
                     f"by session, SHA-256 `{(d['sha256'] or '?')[:16]}…` |")
    lines += ["", "## Held-out tests", ""]
    for e in doc["held_out_tests"]:
        verdict = "passed for " + ", ".join(e["serves"]) if e["gate_passed"] else "not passed"
        lines.append(f"- {e['at']}: {e['passed']}/{e['n']} steps, {verdict}"
                     + (f" ({e['fresh_steps']} steps from sessions recorded after training)" if e.get("fresh_steps") else ""))
        for k, s in e["steps"].items():
            lines.append(f"  - {k} steps: {s['passed']}/{s['n']}, lower bound {s['lower_bound']:.0%}"
                         f"{', passes' if s['passes'] else ''}")
    nf = doc.get("new_folders")
    if nf:
        lines += ["", "## Folders it never trained on", "",
                  f"- Trained on: {', '.join(nf['trained_on']) or 'none recorded'}",
                  f"- Tested on: {', '.join(nf['new_trees']) or 'none yet'}"]
        for k, s in nf["steps"].items():
            lines.append(f"  - {k} steps: {s['passed']}/{s['n']}")

    def fmt(tally: dict[str, int]) -> str:
        return ", ".join(f"{k} {v}" for k, v in sorted(tally.items())) or "none"
    lines += ["", "## In use", "",
              f"- Shadow mode: {fmt(doc['shadow'])}",
              f"- Spot checks: {fmt(doc['spot_checks'])}",
              f"- Live: {fmt(doc['live'])}"]
    if f.get("status_reason"):
        lines.append(f"- Last status change: {f['status_reason']}")
    if doc.get("forgotten_sessions"):
        lines.append("- A session it trained on was forgotten, so it is off duty until retrained.")
    lines += ["", "No conversations are included: only counts, scores, hashes and names."]
    return "\n".join(lines)


def _iso(t: Any) -> str | None:
    if not t:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(t)))
