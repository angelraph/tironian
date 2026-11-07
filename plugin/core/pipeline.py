"""Offline pipeline: history import → families → report → dataset → evaluation → promotion.

Nothing here runs inside a model turn. These functions are driven by the CLI, the slash
command and the Desktop page, and they are all idempotent: running one twice gives the same
result, so a user (or a judge) can rerun any step and get the same numbers.
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import families as fam
from .config import Settings
from .evaluate import Check, Judge, Scorecard, check_decision, gated_scorecard, scorecard, step_kind  # noqa: F401
from .grade import ToolOutcome, TurnFacts, grade
from .hermes_format import compact_messages, render_target, to_sharegpt, tool_name
from .redact import TRAINING_HIT_LIMIT, Redactor, sensitive_count
from .store import Store, compress, decompress
from .student import StudentClient

ROUTINE_MIN = 5          # a task seen this many times counts as "recurring" in the report
EXCLUDED_SOURCES = ("cron", "kanban", "subagent")


# ============================================================== history import (read-only)

def import_history(store: Store, state_db: Path, redactor: Redactor, *, since_days: int = 90,
                   exclude_sources: Sequence[str] = EXCLUDED_SOURCES, now: float | None = None) -> dict[str, int]:
    """Read Hermes's ``state.db`` **read-only** and add past sessions as graded turns.

    Imported rows are prefixed ``h:`` and skipped if already present, so re-running is safe.
    """
    now = now or time.time()
    if not state_db.exists():
        return {"sessions": 0, "turns": 0, "decisions": 0}
    uri = state_db.resolve().as_uri() + "?mode=ro"
    src = sqlite3.connect(uri, uri=True, timeout=5)
    src.row_factory = sqlite3.Row
    try:
        scols = {r[1] for r in src.execute("PRAGMA table_info(sessions)")}
        mcols = {r[1] for r in src.execute("PRAGMA table_info(messages)")}
        where = ["started_at >= ?"]
        args: list[Any] = [now - since_days * 86400]
        if "parent_session_id" in scols:
            where.append("parent_session_id IS NULL")
        if "archived" in scols:
            pass  # archived sessions are still real work; keep them
        cost_col = "estimated_cost_usd" if "estimated_cost_usd" in scols else "NULL"
        sessions = src.execute(
            f"SELECT id, source, model, started_at, ended_at, "
            f"{'end_reason' if 'end_reason' in scols else 'NULL'} AS end_reason, {cost_col} AS cost "
            f"FROM sessions WHERE {' AND '.join(where)} ORDER BY started_at", args).fetchall()
        msg_filter = []
        if "active" in mcols:
            msg_filter.append("(active IS NULL OR active=1)")
        if "compacted" in mcols:
            msg_filter.append("(compacted IS NULL OR compacted=0)")
        mwhere = (" AND " + " AND ".join(msg_filter)) if msg_filter else ""
        n_sessions = n_turns = n_decisions = 0
        # A session Tironian captured live is already here with full detail; importing it again
        # would put the same conversation in the data twice (and maybe on both sides of the split).
        for (dup,) in store.q("SELECT h.id FROM sessions h JOIN sessions l ON l.id = substr(h.id, 3) "
                              "WHERE h.id LIKE 'h:%'"):
            store.purge_session(dup)
        for s in sessions:
            if (s["source"] or "") in exclude_sources:
                continue
            sid = "h:" + s["id"]
            if store.one("SELECT 1 FROM sessions WHERE id IN (?, ?)", (sid, s["id"])):
                continue
            rows = src.execute(
                f"SELECT role, content, tool_calls, tool_call_id, tool_name, timestamp FROM messages "
                f"WHERE session_id=?{mwhere} ORDER BY id", (s["id"],)).fetchall()
            t, d = _import_session(store, redactor, sid, s, rows)
            if t:
                n_sessions += 1
                n_turns += t
                n_decisions += d
        return {"sessions": n_sessions, "turns": n_turns, "decisions": n_decisions}
    finally:
        src.close()


def _import_session(store: Store, redactor: Redactor, sid: str, s: sqlite3.Row,
                    rows: Sequence[sqlite3.Row]) -> tuple[int, int]:
    completed = (s["end_reason"] or "") not in ("interrupted", "error", "cancelled")
    msgs: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for r in rows:
        role = r["role"]
        content = r["content"] or ""
        if role == "user":
            cur = {"user_text": content, "start": len(msgs), "decisions": [], "tools": [], "ts": r["timestamp"]}
            turns.append(cur)
            msgs.append({"role": "user", "content": content})
        elif role == "assistant":
            try:
                calls = json.loads(r["tool_calls"]) if r["tool_calls"] else []
            except json.JSONDecodeError:
                calls = []
            m = {"role": "assistant", "content": content or None, "tool_calls": calls or None}
            if cur is not None:
                cur["decisions"].append((len(msgs), m))
            msgs.append(m)
        elif role == "tool":
            msgs.append({"role": "tool", "tool_call_id": r["tool_call_id"], "name": r["tool_name"], "content": content})
            if cur is not None:
                cur["tools"].append((r["tool_name"] or "", _tool_status(content)))
    if not turns:
        return 0, 0
    session_cost = float(s["cost"] or 0.0)
    n_dec_total = sum(len(t["decisions"]) for t in turns) or 1
    store.upsert("sessions", {"id": sid, "platform": s["source"], "provider": None, "model": s["model"],
                              "started_at": s["started_at"], "ended_at": s["ended_at"],
                              "completed": int(completed), "source": "history"})
    n_dec = 0
    for i, t in enumerate(turns):
        tid = f"{sid}:{i}"
        red_user = redactor.redact_text(t["user_text"]).text
        store.upsert("turns", {"id": tid, "session_id": sid, "idx": i, "user_text": red_user,
                               "started_at": t["ts"], "label": "pending"})
        for j, (name, status) in enumerate(t["tools"]):
            store.upsert("tool_calls", {"id": f"{tid}:t{j}", "turn_id": tid, "session_id": sid, "name": name,
                                        "status": status, "created_at": t["ts"]})
        for k, (pos, m) in enumerate(t["decisions"]):
            ctx, h1 = redactor.redact_obj(msgs[:pos])
            out, h2 = redactor.redact_obj({"content": m["content"] or "", "tool_calls": m["tool_calls"] or []})
            store.upsert("decisions", {
                "id": f"{tid}:d{k}", "turn_id": tid, "session_id": sid, "call_idx": k,
                "api_mode": "chat_completions", "provider": None, "model": s["model"], "tools_hash": None,
                "context_z": compress(ctx), "output_json": json.dumps(out, ensure_ascii=False),
                "finish_reason": None, "in_tok": 0, "out_tok": 0,
                "cost_usd": session_cost / n_dec_total, "latency_ms": None, "served_by": "teacher",
                "redaction_hits": sensitive_count(h1) + sensitive_count(h2), "created_at": t["ts"]})
            n_dec += 1
    for i, t in enumerate(turns):
        nxt = turns[i + 1]["user_text"] if i + 1 < len(turns) else None
        g = grade(TurnFacts(t["user_text"], [ToolOutcome(n, st) for n, st in t["tools"]], nxt,
                            session_completed=completed))
        with store.tx() as c:
            c.execute("UPDATE turns SET score=?, label=?, grade_json=? WHERE id=?",
                      (g.score, g.label, json.dumps({"reasons": g.reasons, "hard": g.hard_negative}), f"{sid}:{i}"))
    return len(turns), n_dec


def _tool_status(content: str) -> str:
    s = (content or "").lstrip()[:400].lower()
    if s.startswith("{"):
        try:
            obj = json.loads(content)
            if isinstance(obj, dict):
                if obj.get("error") or obj.get("success") is False:
                    return "error"
                code = obj.get("exit_code", obj.get("returncode"))
                if isinstance(code, int) and code != 0:
                    return "error"
        except json.JSONDecodeError:
            pass
    if s.startswith(("error:", "traceback", "[error]")):
        return "error"
    return "ok"


# ============================================================== families

def rebuild_families(store: Store, settings: Settings) -> list[dict[str, Any]]:
    """Recluster every non-excluded turn and persist families, keeping promoted ones stable."""
    rows = store.q(
        "SELECT t.id, t.user_text, t.label FROM turns t JOIN sessions s ON s.id=t.session_id "
        "WHERE s.excluded=0 AND t.label NOT IN ('excluded','pending') AND t.user_text IS NOT NULL "
        "ORDER BY t.started_at, t.id")
    tools_by_turn: dict[str, list[str]] = {}
    for r in store.q("SELECT turn_id, name FROM tool_calls ORDER BY created_at"):
        tools_by_turn.setdefault(r["turn_id"], []).append(r["name"] or "")
    items = [fam.Item(r["id"], r["user_text"], tools_by_turn.get(r["id"], [])) for r in rows]
    labels = {r["id"]: r["label"] for r in rows}
    common = fam.common_features([fam.features(i.user_text, i.tools, i.skill) for i in items])
    clusters = fam.cluster(items, common=common)
    store.meta_set("common_features", json.dumps(sorted(common)))

    old = {r["id"]: r for r in store.q("SELECT * FROM families")}
    old_members: dict[str, set[str]] = {}
    for r in store.q("SELECT id, family_id FROM turns WHERE family_id IS NOT NULL"):
        old_members.setdefault(r["family_id"], set()).add(r["id"])

    # Each new family keeps the id (and status/student) of the old family it mostly is.
    # Families with a student go first and can also be recognised by what they ask for, so a
    # trained family keeps its id even if an earlier rebuild lost its members. Otherwise the
    # match is on shared members, measured against both sides so a one-request family can't
    # hand its id to a big one.
    def _content(fs: Any) -> frozenset[str]:
        return fam.without_common(frozenset(x for x in fs if not x.startswith(("<", "T:", "S:"))
                                            and x not in fam._PHRASING), common)

    pairs: list[tuple[float, int, str]] = []
    for i, f in enumerate(clusters):
        members = set(f.members)
        new_ex = [_content(e) for e in f.exemplars]
        for oid, r in old.items():
            om = old_members.get(oid, set())
            score = len(members & om) / len(members | om) if om else 0.0
            if r["student_id"]:
                old_ex = [_content(e) for e in json.loads(r["exemplars_json"] or "[]")]
                sim = max((fam.jaccard(a, b) for a in new_ex for b in old_ex if a and b), default=0.0)
                if sim >= 0.5:
                    score = 1.0 + sim
            if score >= 0.3:
                pairs.append((score, i, oid))
    fid_for: dict[int, str] = {}
    claimed: set[str] = set()
    for score, i, oid in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
        if i not in fid_for and oid not in claimed:
            fid_for[i] = oid
            claimed.add(oid)

    out = []
    with store.tx() as c:
        c.execute("UPDATE turns SET family_id=NULL")
        for i, f in enumerate(clusters):
            fid = fid_for.get(i, f.id)
            claimed.add(fid)
            prev = old.get(fid)
            n_good = sum(1 for m in f.members if labels.get(m) == "good")
            row = {
                "id": fid, "name": f.label(), "exemplars_json": json.dumps([sorted(e) for e in f.exemplars]),
                "toolset_json": json.dumps(f.toolset()), "n_turns": len(f.members), "n_good": n_good,
                "status": prev["status"] if prev else "candidate",
                "student_id": prev["student_id"] if prev else None, "updated_at": time.time()}
            cols = list(row)
            c.execute(f"INSERT INTO families({', '.join(cols)}) VALUES({', '.join('?' for _ in cols)}) "
                      f"ON CONFLICT(id) DO UPDATE SET {', '.join(f'{k}=excluded.{k}' for k in cols if k != 'id')}",
                      tuple(row[k] for k in cols))
            c.executemany("UPDATE turns SET family_id=? WHERE id=?", [(fid, m) for m in f.members])
            out.append(row)
        # Families that vanished keep their row only if a student depends on them.
        live_ids = {r["id"] for r in out}
        for oid, r in old.items():
            if oid not in live_ids:
                if r["student_id"]:
                    c.execute("UPDATE families SET n_turns=0, n_good=0, updated_at=? WHERE id=?", (time.time(), oid))
                else:
                    c.execute("DELETE FROM families WHERE id=?", (oid,))
    return out


# ============================================================== first-minute report

@dataclass
class Report:
    sessions: int
    turns: int
    good: int
    bad: int
    routine_turns: int
    routine_share: float
    spend_30d_usd: float
    routine_spend_30d_usd: float
    trainable_families: int
    families: list[dict[str, Any]] = field(default_factory=list)
    redactions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def report(store: Store, settings: Settings, *, now: float | None = None) -> Report:
    now = now or time.time()
    since = now - 30 * 86400
    q = store.scalar
    turns = q("SELECT COUNT(*) FROM turns t JOIN sessions s ON s.id=t.session_id WHERE s.excluded=0 AND t.label!='excluded'", (), 0)
    fams = store.q("SELECT * FROM families ORDER BY n_turns DESC")
    fam_rows = []
    routine = 0
    for f in fams:
        cost = q("SELECT SUM(d.cost_usd) FROM decisions d JOIN turns t ON t.id=d.turn_id "
                 "WHERE t.family_id=? AND d.created_at>=?", (f["id"], since), 0.0)
        if f["n_turns"] >= ROUTINE_MIN:
            routine += f["n_turns"]
        fam_rows.append({
            "id": f["id"], "name": f["name"], "turns": f["n_turns"], "good": f["n_good"],
            "status": f["status"], "spend_30d_usd": round(float(cost or 0.0), 4),
            "tools": json.loads(f["toolset_json"] or "[]"),
            "ready_to_train": f["n_good"] >= settings.min_family_turns,
            "needs": max(0, settings.min_family_turns - f["n_good"])})
    spend = float(q("SELECT SUM(cost_usd) FROM decisions WHERE created_at>=?", (since,), 0.0) or 0.0)
    routine_spend = sum(r["spend_30d_usd"] for r in fam_rows if r["turns"] >= ROUTINE_MIN)
    return Report(
        sessions=q("SELECT COUNT(*) FROM sessions WHERE excluded=0", (), 0), turns=turns,
        good=q("SELECT COUNT(*) FROM turns WHERE label='good'", (), 0),
        bad=q("SELECT COUNT(*) FROM turns WHERE label='bad'", (), 0),
        routine_turns=routine, routine_share=routine / turns if turns else 0.0,
        spend_30d_usd=round(spend, 4), routine_spend_30d_usd=round(routine_spend, 4),
        trainable_families=sum(1 for r in fam_rows if r["ready_to_train"]),
        families=fam_rows[:25],
        redactions=int(q("SELECT SUM(redaction_hits) FROM decisions", (), 0) or 0))


# ============================================================== dataset

@dataclass
class DatasetInfo:
    id: str
    family_id: str
    train_path: Path
    heldout_path: Path
    n_train: int
    n_heldout: int
    toolset: list[str]
    skipped: dict[str, int]
    sha256: str


def _schemas_by_name(store: Store) -> dict[str, dict[str, Any]]:
    """Latest captured schema for every tool name (live capture is the only source of schemas)."""
    out: dict[str, dict[str, Any]] = {}
    for r in store.q("SELECT tools_json FROM toolsets"):
        for t in json.loads(r["tools_json"] or "[]"):
            n = tool_name(t)
            if n:
                out[n] = t
    return out


def _example(store: Store, d: Any, toolset_schemas: list[dict[str, Any]], max_tokens: int):
    msgs = decompress(d["context_z"])
    out = json.loads(d["output_json"] or "{}")
    if msgs is None:
        return None, "expired"
    compact = compact_messages(msgs, toolset_schemas, max_tokens=max_tokens)
    if compact.had_image:
        return None, "image"
    if not compact.messages[1:]:
        return None, "no_context"
    names = {tool_name(t) for t in toolset_schemas}
    for tc in out.get("tool_calls") or []:
        if (tc.get("function") or {}).get("name") not in names:
            return None, "tool_outside_family"
    target = render_target(out.get("content"), out.get("tool_calls"))
    return {"compact": compact, "target": target, "out": out}, None


def build_dataset(store: Store, settings: Settings, family_id: str, out_dir: Path, *, seed: int = 3407,
                  heldout_share: float = 0.2) -> DatasetInfo:
    """Write ``train.jsonl`` / ``heldout.jsonl`` (Hermes ShareGPT format) for one family.

    Split is by **session** so no conversation leaks from training into evaluation.
    """
    f = store.one("SELECT * FROM families WHERE id=?", (family_id,))
    if f is None:
        raise ValueError(f"unknown family {family_id}")
    toolset = json.loads(f["toolset_json"] or "[]")
    schemas = _schemas_by_name(store)
    missing = [t for t in toolset if t not in schemas]
    if missing:
        raise ValueError("no captured schema yet for tool(s) " + ", ".join(missing) +
                         ". Use Hermes once with Tironian enabled so it can see them.")
    tool_schemas = [schemas[t] for t in toolset]

    rows = store.q(
        "SELECT d.*, t.session_id AS sid FROM decisions d JOIN turns t ON t.id=d.turn_id "
        "JOIN sessions s ON s.id=t.session_id "
        "WHERE t.family_id=? AND t.label='good' AND d.served_by='teacher' AND s.excluded=0 "
        "AND d.redaction_hits<=? AND d.context_z IS NOT NULL ORDER BY d.created_at, d.id",
        (family_id, TRAINING_HIT_LIMIT))
    skipped: dict[str, int] = {}
    examples: list[tuple[str, str, dict[str, Any]]] = []
    for d in rows:
        ex, why = _example(store, d, tool_schemas, settings.max_context_tokens)
        if ex is None:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        examples.append((d["sid"], d["id"], ex))

    sessions = sorted({sid for sid, _, _ in examples})
    rng = random.Random(seed)
    rng.shuffle(sessions)
    n_hold = max(1, round(len(sessions) * heldout_share)) if len(sessions) > 1 else 0
    # The gate needs min_eval_n held-out steps; with fewer, even a perfect student can't pass.
    # Hold out more sessions until it has them, but never more than 40% of the family.
    steps = {}
    for sid, _, _ in examples:
        steps[sid] = steps.get(sid, 0) + 1
    cap = int(len(sessions) * 0.4)
    while n_hold < cap and sum(steps[s] for s in sessions[:n_hold]) < settings.min_eval_n:
        n_hold += 1
    held_sessions = set(sessions[:n_hold])

    ds_id = "ds-" + hashlib.sha1(f"{family_id}:{seed}:{len(examples)}:{time.time()}".encode()).hexdigest()[:10]
    ddir = out_dir / ds_id
    ddir.mkdir(parents=True, exist_ok=True)
    train_p, held_p = ddir / "train.jsonl", ddir / "heldout.jsonl"
    h = hashlib.sha256()
    n_train = n_held = 0
    items = []
    with train_p.open("w", encoding="utf-8") as ft, held_p.open("w", encoding="utf-8") as fh:
        for sid, did, ex in examples:
            held = sid in held_sessions
            line = json.dumps({
                "conversations": to_sharegpt(ex["compact"], ex["target"]),
                "family": family_id, "decision_id": did, "completed": True,
                **({"teacher": ex["out"]} if held else {}),
            }, ensure_ascii=False)
            (fh if held else ft).write(line + "\n")
            h.update(line.encode("utf-8"))
            items.append((ds_id, did, "heldout" if held else "train"))
            if held:
                n_held += 1
            else:
                n_train += 1
    digest = h.hexdigest()
    with store.tx() as c:
        c.execute("INSERT INTO datasets(id, created_at, family_ids, seed, n_train, n_heldout, path, sha256) "
                  "VALUES(?,?,?,?,?,?,?,?)", (ds_id, time.time(), json.dumps([family_id]), seed, n_train, n_held,
                                              str(ddir), digest))
        c.executemany("INSERT INTO dataset_items(dataset_id, decision_id, split) VALUES(?,?,?)", items)
    return DatasetInfo(ds_id, family_id, train_p, held_p, n_train, n_held, toolset, skipped, digest)


# ============================================================== evaluation + promotion

def evaluate_student(store: Store, settings: Settings, *, student_id: str, dataset_id: str,
                     client: StudentClient, judge: Judge | None = None, limit: int | None = None,
                     progress: Callable[[int, int], None] | None = None) -> Scorecard:
    st = store.one("SELECT * FROM students WHERE id=?", (student_id,))
    ds = store.one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if st is None or ds is None:
        raise ValueError("unknown student or dataset")
    family_id = json.loads(ds["family_ids"])[0]
    f = store.one("SELECT toolset_json FROM families WHERE id=?", (family_id,))
    schemas = _schemas_by_name(store)
    tools = [schemas[t] for t in json.loads(f["toolset_json"] or "[]") if t in schemas]
    lines = Path(ds["path"], "heldout.jsonl").read_text(encoding="utf-8").splitlines()
    if limit:
        lines = lines[:limit]

    eval_id = "ev-" + hashlib.sha1(f"{student_id}:{dataset_id}:{time.time()}".encode()).hexdigest()[:10]
    checks: list[Check] = []
    lats: list[int] = []
    kinds: list[str] = []
    item_rows = []
    for i, line in enumerate(lines):
        ex = json.loads(line)
        convo = ex["conversations"]
        kinds.append(step_kind("tool" if len(convo) >= 2 and convo[-2]["from"] == "tool" else "user"))
        msgs = [{"role": {"system": "system", "human": "user", "tool": "user", "gpt": "assistant"}[c["from"]],
                 "content": c["value"]} for c in convo[:-1]]
        teacher = ex.get("teacher") or {}
        user_req = next((c["value"] for c in reversed(convo[:-1]) if c["from"] == "human"), "")
        reply = client.complete(st["model_id"], msgs)
        lats.append(reply.latency_ms)
        if not reply.ok:
            chk = Check(False, False, reason=f"invalid:{reply.error}")
        else:
            chk = check_decision(reply.text, teacher.get("content") or "", teacher.get("tool_calls") or [],
                                 tools, user_request=user_req, judge=judge, flip=bool(i % 2))
        checks.append(chk)
        item_rows.append((eval_id, ex.get("decision_id"), reply.text[:4000],
                          json.dumps(asdict(chk)), int(chk.passed)))
        if progress:
            progress(i + 1, len(lines))

    sc = gated_scorecard(checks, lats, kinds, threshold=settings.promotion_threshold, min_n=settings.min_eval_n,
                         validity_floor=settings.validity_floor)
    store.meta_set(f"serve_steps:{family_id}", json.dumps(sc.serve_steps if sc.gate_passed else []))
    with store.tx() as c:
        c.execute(
            "INSERT INTO evals(id, student_id, dataset_id, family_id, n, passed, wilson_lo, wilson_hi, validity, "
            "tool_acc, args_acc, judge_win, p50_ms, gate_passed, report_json, created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (eval_id, student_id, dataset_id, family_id, sc.n, sc.passed, sc.wilson_lo, sc.wilson_hi, sc.validity,
             (sc.tool_hits / sc.tool_steps) if sc.tool_steps else None, sc.args_mean, None, sc.p50_ms,
             int(sc.gate_passed), json.dumps(asdict(sc)), time.time()))
        c.executemany("INSERT INTO eval_items(eval_id, decision_id, student_out, checks_json, pass) VALUES(?,?,?,?,?)",
                      item_rows)
        if sc.gate_passed:
            c.execute("UPDATE families SET status=?, student_id=?, updated_at=? WHERE id=? AND status IN "
                      "('candidate','trained','evaluated','demoted','shadow')",
                      (settings.promotion_mode, student_id, time.time(), family_id))
        else:
            # Link the student even on a fail, so status and `/tironian shadow` can say *why* it can't serve.
            c.execute("UPDATE families SET status='evaluated', student_id=?, updated_at=? WHERE id=? AND status IN "
                      "('candidate','trained','evaluated')", (student_id, time.time(), family_id))
    return sc


def shadow_ready(store: Store, family_id: str, *, min_n: int = 20, threshold: float = 0.85) -> tuple[bool, str]:
    rows = store.q("SELECT outcome FROM routes WHERE family_id=? AND mode='shadow' AND outcome IN "
                   "('shadow_agree','shadow_disagree') ORDER BY id DESC LIMIT 200", (family_id,))
    n = len(rows)
    agree = sum(1 for r in rows if r["outcome"] == "shadow_agree")
    if n < min_n:
        return False, f"{n}/{min_n} live shadow samples so far"
    rate = agree / n
    if rate < threshold:
        return False, f"agrees with your model {rate:.0%} of the time (needs {threshold:.0%})"
    return True, f"agrees with your model {rate:.0%} over {n} live requests"


def register_student(store: Store, *, job_id: str | None, family_id: str, base_model: str, gguf_path: Path,
                     sha256: str) -> str:
    sid = gguf_path.stem
    store.upsert("students", {
        "id": sid, "job_id": job_id, "family_id": family_id, "base_model": base_model,
        "gguf_path": str(gguf_path), "model_id": sid, "sha256": sha256,
        "size_bytes": gguf_path.stat().st_size if gguf_path.exists() else 0, "created_at": time.time()})
    with store.tx() as c:
        c.execute("UPDATE families SET status='trained', updated_at=? WHERE id=? AND status IN ('candidate','evaluated')",
                  (time.time(), family_id))
    return sid


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)
