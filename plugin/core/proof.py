"""A held-out test anyone can run again.

``hermes tironian proof <student>`` writes the exact steps a student was tested on, what it
answered, and how each answer was graded, sealed with a hash. ``hermes tironian verify`` takes
that file and the student's model file, checks the file is the one that was tested, loads it
into Ollama, asks every step again and grades the answers with the same code. Nothing from the
author's machine is needed, so a reviewer can check a claim like "51 of 51" for themselves.

Steps graded by the blind judge are the honest exception: a judge is a model, not a rule. When
the student answers exactly as it did in the test, the recorded verdict stands and the report
shows both answers so you can read them. When the answer differs, it is judged again by your own
Tironian judge if you have one, and otherwise counted as not passed.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Sequence

from .evaluate import Check, Judge, check_decision, gated_scorecard

PROOF_VERSION = 1


def seal(body: dict[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def check_seal(doc: dict[str, Any]) -> bool:
    body = {k: v for k, v in doc.items() if k != "seal"}
    return bool(doc.get("seal")) and seal(body) == doc["seal"]


# Export ---------------------------------------------------------------------------------------

def export(store: Any, settings: Any, student_id: str, *, eval_id: str | None = None,
           kinds: Sequence[str] = ("first",)) -> dict[str, Any]:
    """The sealed record of one evaluation of ``student_id`` (its latest passing one by default)."""
    from . import pipeline
    from .passport import _sha256_file

    st = store.one("SELECT * FROM students WHERE id=?", (student_id,))
    if st is None:
        raise ValueError(f"unknown student {student_id}")
    if eval_id:
        ev = store.one("SELECT * FROM evals WHERE id=? AND student_id=?", (eval_id, student_id))
        if ev is None:
            raise ValueError(f"no evaluation {eval_id} for {student_id}")
    else:
        ev = (store.one("SELECT * FROM evals WHERE student_id=? AND gate_passed=1 ORDER BY created_at DESC LIMIT 1",
                        (student_id,))
              or store.one("SELECT * FROM evals WHERE student_id=? ORDER BY created_at DESC LIMIT 1", (student_id,)))
        if ev is None:
            raise ValueError(f"{student_id} has no evaluation yet: run `hermes tironian eval {student_id}` first")
    ds = store.one("SELECT * FROM datasets WHERE id=?", (ev["dataset_id"],))
    if ds is None:
        raise ValueError(f"the dataset {ev['dataset_id']} behind that evaluation is gone")
    family_id = json.loads(ds["family_ids"])[0]
    fam = store.one("SELECT name, toolset_json FROM families WHERE id=?", (family_id,))
    schemas = pipeline._schemas_by_name(store)
    tools = [schemas[t] for t in json.loads((fam and fam["toolset_json"]) or "[]") if t in schemas]

    # The evaluation stored each step's decision id, output and grade. Its input lines are the
    # dataset's held-out file plus the sessions recorded after training, rebuilt the same way.
    lines = Path(ds["path"], "heldout.jsonl").read_text(encoding="utf-8").splitlines()
    lines += pipeline._fresh_heldout(store, settings, family_id, ds["id"], tools)
    by_id = {}
    for ln in lines:
        ex = json.loads(ln)
        by_id.setdefault(ex.get("decision_id"), ex)

    items = store.q("SELECT rowid, * FROM eval_items WHERE eval_id=? ORDER BY rowid", (ev["id"],))
    steps, missing = [], 0
    for i, it in enumerate(items):
        ex = by_id.get(it["decision_id"])
        if ex is None:
            missing += 1
            continue
        kind, msgs, teacher, user_req = pipeline.step_inputs(ex)
        if kinds and kind not in kinds:
            continue
        steps.append({
            "n": i, "decision_id": it["decision_id"], "kind": kind,
            "place": pipeline.fam.place(user_req) or "",
            "user_request": user_req, "messages": msgs, "teacher": teacher,
            "flip": bool(i % 2),   # the judge's A/B order in the original run
            "recorded": {"output": it["student_out"], "check": json.loads(it["checks_json"] or "{}"),
                         "passed": bool(it["pass"])},
        })
    if missing:
        raise ValueError(f"{missing} step(s) of {ev['id']} can't be rebuilt any more (their sessions were "
                         "forgotten or purged), so this test can't be exported as it was run")
    if not steps:
        raise ValueError(f"{ev['id']} has no {' or '.join(kinds)} steps")

    gguf = Path(st["gguf_path"])
    body = {
        "proof": PROOF_VERSION, "tool": "tironian",
        "issued": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "student": {"id": student_id, "base": st["base_model"], "size_bytes": st["size_bytes"],
                    "sha256": st["sha256"] or _sha256_file(gguf) or ""},
        "family": {"id": family_id, "name": fam["name"] if fam else ""},
        "eval": {"id": ev["id"], "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ev["created_at"])),
                 "kinds": list(kinds)},
        "gate": {"threshold": settings.promotion_threshold, "min_n": settings.min_eval_n,
                 "validity_floor": settings.validity_floor},
        "tools": tools,
        "steps": steps,
    }
    return {**body, "seal": seal(body)}


def review_markdown(doc: dict[str, Any]) -> str:
    """Every step in plain text: read this before sharing a proof, it is what others will see."""
    out = [f"# Proof for {doc['student']['id']}", "",
           f"Family **{doc['family']['name'] or doc['family']['id']}**, evaluation {doc['eval']['id']} "
           f"({doc['eval']['at']}), {len(doc['steps'])} steps ({', '.join(doc['eval']['kinds'])}).", "",
           "Read every step below before you share this file. It contains your requests and the",
           "commands your model ran, with your home folder and secrets already masked.", ""]
    for s in doc["steps"]:
        rc = s["recorded"]
        verdict = "passed" if rc["passed"] else f"missed ({rc['check'].get('reason') or 'no reason'})"
        how = rc["check"].get("text") or "rule"
        out += [f"## Step {s['n']} ({s['kind']}, {verdict}, {how})", "",
                f"**Request:** {s['user_request']}", "",
                "**Student:**", "```", (rc["output"] or "").strip()[:2000], "```",
                "**Your model:**", "```", _teacher_text(s["teacher"])[:2000], "```", ""]
    return "\n".join(out)


def _teacher_text(t: dict[str, Any]) -> str:
    calls = t.get("tool_calls") or []
    if calls:
        return "\n".join(f"{(c.get('function') or {}).get('name')}({(c.get('function') or {}).get('arguments')})"
                         for c in calls)
    return (t.get("content") or "").strip()


# Verify ---------------------------------------------------------------------------------------

def replay(doc: dict[str, Any], complete: Callable[[list[dict[str, str]]], Any], *,
           judge: Judge | None = None, progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Ask every step again and grade the answers. ``complete(messages)`` returns a StudentReply."""
    tools = doc.get("tools") or []
    checks: list[Check] = []
    kinds: list[str] = []
    lats: list[int] = []
    rows = []
    same = reused = rejudged = unjudged = 0
    for i, s in enumerate(doc["steps"]):
        reply = complete(s["messages"])
        lats.append(reply.latency_ms)
        kinds.append(s["kind"])
        rec = s["recorded"]
        identical = reply.ok and reply.text == rec["output"]
        same += identical
        rec_text = (rec.get("check") or {}).get("text") or ""
        if not reply.ok:
            chk = Check(False, False, reason=f"invalid:{reply.error}")
        elif identical and rec_text.startswith("judge-"):
            # Same answer, same question: the judge's recorded verdict applies to it.
            chk = Check(**{k: v for k, v in rec["check"].items() if k in Check.__dataclass_fields__})
            reused += 1
        else:
            seen: dict[str, bool] = {"asked": False}

            def tracked(req: str, a: str, b: str) -> str | None:
                seen["asked"] = True
                return judge(req, a, b) if judge else None
            chk = check_decision(reply.text, s["teacher"].get("content") or "", s["teacher"].get("tool_calls") or [],
                                 tools, user_request=s["user_request"], judge=tracked, flip=bool(s.get("flip")),
                                 context=_context_text(s["messages"]))
            if seen["asked"]:
                if (chk.text or "").startswith("judge-"):
                    rejudged += 1
                elif not chk.passed:
                    unjudged += 1
        checks.append(chk)
        rows.append({"n": s["n"], "kind": s["kind"], "identical": identical, "passed": chk.passed,
                     "was_passed": rec["passed"], "how": chk.text or "rule", "reason": chk.reason,
                     "output": reply.text if not identical else None})
        if progress:
            progress(i + 1, len(doc["steps"]))
    g = doc.get("gate") or {}
    sc = gated_scorecard(checks, lats, kinds, threshold=g.get("threshold", 0.85), min_n=g.get("min_n", 30),
                         validity_floor=g.get("validity_floor", 0.98))
    recorded = sum(s["recorded"]["passed"] for s in doc["steps"])
    return {"n": sc.n, "passed": sc.passed, "recorded_passed": recorded, "identical": same,
            "judge_reused": reused, "rejudged": rejudged, "unjudged": unjudged,
            "steps": sc.steps, "serves": sc.serve_steps, "gate_passed": sc.gate_passed,
            "gate_reason": sc.gate_reason, "p50_ms": sc.p50_ms, "rows": rows}


def report_text(doc: dict[str, Any], res: dict[str, Any]) -> str:
    lines = [f"Verified {doc['student']['id']} on {res['n']} held-out steps from evaluation {doc['eval']['id']}.",
             f"  Same answer as in the test: {res['identical']} of {res['n']}",
             f"  Passed now: {res['passed']} of {res['n']} (the test recorded {res['recorded_passed']})"]
    for kind, d in res["steps"].items():
        lines.append(f"  {kind} steps: {d['passed']} of {d['n']}, lower bound {d['wilson_lo']:.0%}"
                     f" -> {'passes' if d['gate_passed'] else d['gate_reason']}")
    if res["judge_reused"]:
        lines.append(f"  {res['judge_reused']} pass(es) rest on the blind judge's recorded verdict; the answer is"
                     " identical, and PROOF.md shows both commands so you can judge them yourself")
    if res["rejudged"]:
        lines.append(f"  {res['rejudged']} step(s) answered differently were judged again by your judge model")
    if res["unjudged"]:
        lines.append(f"  {res['unjudged']} step(s) answered differently needed a judge and had none, so they count"
                     " as not passed (turn on the Tironian judge to grade them)")
    changed = [r for r in res["rows"] if r["passed"] != r["was_passed"]]
    for r in changed[:10]:
        lines.append(f"  step {r['n']}: {'passed' if r['was_passed'] else 'missed'} in the test, "
                     f"{'passes' if r['passed'] else 'misses'} now ({r['reason'] or r['how']})")
    lines.append(("GATE REPRODUCED: " if res["gate_passed"] else "GATE NOT REPRODUCED: ") + res["gate_reason"])
    return "\n".join(lines)


# Files ----------------------------------------------------------------------------------------

def fetch(src: str, cache: Path, *, expect_sha256: str | None = None,
          say: Callable[[str], None] = print) -> Path:
    """A local path as is, or an http(s) URL downloaded once into ``cache``."""
    if not src.lower().startswith(("http://", "https://")):
        p = Path(src).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"no such file: {p}")
        return p
    import urllib.request
    name = Path(urllib.parse.urlsplit(src).path).name or "download"
    # Every release names its proof proof.json, so the cache is per URL: keyed by name alone,
    # verifying v7 after v4 handed back v4's proof.
    folder = cache / hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / name
    # Reuse a download only when there's a hash to check it against (the model file). Without
    # one (the proof), fetch it again: it's small, and the URL may now serve something else.
    if dest.exists() and expect_sha256 and file_sha256(dest) == expect_sha256:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    say(f"Downloading {src}")
    with urllib.request.urlopen(src, timeout=60) as resp, tmp.open("wb") as fh:  # noqa: S310 - URL the user gave
        total = int(resp.headers.get("Content-Length") or 0)
        got, last = 0, 0.0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            got += len(chunk)
            if total and time.monotonic() - last > 5:
                say(f"  {got / 1e6:.0f} of {total / 1e6:.0f} MB")
                last = time.monotonic()
    tmp.replace(dest)
    return dest


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _context_text(msgs: Sequence[dict[str, Any]]) -> str:
    from .pipeline import context_text
    return context_text(msgs)
