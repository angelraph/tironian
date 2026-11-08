"""The live half of Tironian: capture every decision, grade every turn, route when proven.

Safety contract (tested in ``tests/test_engine.py``):

* The teacher's ``next_call`` is called **at most once** per provider request.
* Any exception inside Tironian before ``next_call`` is caught and the request goes to the
  teacher unchanged; any exception after it is caught and the teacher's response is returned.
  (Hermes's own middleware runner is fail-open as well, so this is belt and braces.)
* A student answer is used only when the request's family is promoted, the match is confident
  and unambiguous, and the answer parses, names a tool the request actually offers, and passes
  that tool's argument schema. Anything else falls back to the teacher, silently.
* After one fallback in a turn the teacher finishes that turn (sticky), so a turn never
  ping-pongs between models.
* Tironian's own model calls (judge, shadow replays) never pass through its own capture.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from . import families as fam
from .config import Settings
from .evaluate import step_kind
from .grade import ToolOutcome, TurnFacts, grade
from .hermes_format import compact_messages, parse_output, select_tools, tool_name, tool_schema_map, validate_args
from .redact import TRAINING_HIT_LIMIT, Redactor, sensitive_count
from .store import Store, compress
from .student import StudentClient, model_matches
from .wire import build_response, normalize_request, normalize_response

log = logging.getLogger("tironian")

_local = threading.local()


class reentrant:
    """Mark Tironian's own model calls so capture and routing ignore them."""

    def __enter__(self) -> None:
        _local.depth = getattr(_local, "depth", 0) + 1

    def __exit__(self, *exc: Any) -> None:
        _local.depth -= 1


def in_tironian_call() -> bool:
    return getattr(_local, "depth", 0) > 0


@dataclass
class TurnState:
    turn_id: str
    session_id: str
    idx: int
    user_text: str
    call_idx: int = 0
    route_checked: bool = False
    route_family: str | None = None
    route_mode: str | None = None
    student_model: str | None = None
    sticky_fallback: bool = False
    served_by_student: bool = False


@dataclass
class PromotedFamily:
    family: fam.Family
    status: str               # shadow | live
    student_model: str
    toolset: list[str]
    avg_cost_usd: float
    steps: frozenset[str] = frozenset({"first", "follow"})   # step kinds that passed the gate


@dataclass
class Engine:
    store: Store
    settings: Callable[[], Settings]
    redactor: Redactor
    clock: Callable[[], float] = time.time
    student_factory: Callable[[Settings], StudentClient] | None = None
    on_event: Callable[[str, dict[str, Any]], None] | None = None
    judge_factory: Callable[[], Any] | None = None   # the same blind judge the held-out test uses
    _judge: Any = None
    _judge_day: str = ""
    _shadow_n: int = 0
    _turns: "OrderedDict[str, TurnState]" = field(default_factory=OrderedDict)
    _session_turn: dict[str, str] = field(default_factory=dict)
    _session_idx: dict[str, int] = field(default_factory=dict)
    _subagent_sessions: set[str] = field(default_factory=set)
    _excluded_sessions: set[str] = field(default_factory=set)
    _promoted: list[PromotedFamily] | None = None
    _promoted_at: float = 0.0
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _shadow_busy: threading.Lock = field(default_factory=threading.Lock)
    sync_capture: bool = False
    dropped: int = 0
    _queue: "queue.Queue[Any]" = field(default_factory=lambda: queue.Queue(maxsize=256))
    _worker: threading.Thread | None = None

    # Helpers
    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self.on_event:
            try:
                self.on_event(kind, payload)
            except Exception:  # pragma: no cover - UI notifications must never break a turn
                log.debug("tironian event failed", exc_info=True)

    def _student(self, s: Settings) -> StudentClient:
        if self.student_factory:
            return self.student_factory(s)
        from .serve import resolve_endpoint
        return StudentClient(resolve_endpoint(s.student_endpoint), timeout_s=s.student_timeout_s)

    def _turn(self, turn_id: str | None) -> TurnState | None:
        if not turn_id:
            return None
        with self._lock:
            return self._turns.get(turn_id)

    def _excluded(self, session_id: str, platform: str | None, s: Settings) -> bool:
        if session_id in self._excluded_sessions:
            return True
        if not s.capture_subagents and session_id in self._subagent_sessions:
            return True
        return bool(platform) and platform in s.exclude_platforms

    # Hooks
    def on_session_start(self, **kw: Any) -> None:
        sid = str(kw.get("session_id") or "")
        if not sid:
            return
        s = self.settings()
        platform = kw.get("platform")
        excluded = self._excluded(sid, platform, s)
        self.store.upsert("sessions", {
            "id": sid, "platform": platform, "provider": kw.get("provider"), "model": kw.get("model"),
            "started_at": self.clock(), "ended_at": None, "completed": None,
            "excluded": int(excluded), "exclude_reason": "platform" if excluded else None, "source": "live"})
        if not excluded and self.warm_enabled:
            self._warm(s)

    _warmed_at: float = -1e9
    warm_enabled: bool = True

    def _warm(self, s: Settings) -> None:
        """Load the students into the local server when a session starts. Ollama unloads an idle
        model after a few minutes, and reloading takes longer than the routing deadline, so
        without this the first matching request would always fall back."""
        now = time.monotonic()
        if now - self._warmed_at < 120:
            return
        self._warmed_at = now
        models = sorted({p.student_model for p in self.promoted()})
        if not models:
            return

        def run() -> None:
            from .serve import is_ollama, keep_warm
            for m in models:
                try:
                    client = self._student(s)
                    client.timeout_s = 120.0
                    endpoint = str(getattr(client, "endpoint", "") or "")
                    # With Ollama, load it and ask for it to stay loaded for an hour, so a pause
                    # between requests doesn't send the next one past the routing deadline.
                    if self.student_factory is None and is_ollama(endpoint) and keep_warm(endpoint, m):
                        continue
                    with reentrant():
                        client.complete(m, [{"role": "user", "content": "hi"}], max_tokens=1)
                except Exception:
                    log.debug("tironian: warm-up of %s failed", m, exc_info=True)
        threading.Thread(target=run, name="tironian-warm", daemon=True).start()

    def on_subagent_start(self, **kw: Any) -> None:
        child = kw.get("child_session_id")
        if child:
            self._subagent_sessions.add(str(child))

    def exclude_session(self, session_id: str, reason: str = "user") -> None:
        self._excluded_sessions.add(session_id)
        with self.store.tx() as c:
            c.execute("UPDATE sessions SET excluded=1, exclude_reason=? WHERE id=?", (reason, session_id))
            c.execute("UPDATE turns SET label='excluded' WHERE session_id=?", (session_id,))

    def include_session(self, session_id: str) -> None:
        self._excluded_sessions.discard(session_id)
        with self.store.tx() as c:
            c.execute("UPDATE sessions SET excluded=0, exclude_reason=NULL WHERE id=?", (session_id,))

    def pre_llm_call(self, **kw: Any) -> None:
        s = self.settings()
        if not s.capture_enabled:
            return
        sid = str(kw.get("session_id") or "")
        tid = str(kw.get("turn_id") or "")
        if not sid or not tid or self._excluded(sid, kw.get("platform"), s):
            return
        user_text = kw.get("user_message")
        if not isinstance(user_text, str):
            user_text = json.dumps(user_text, ensure_ascii=False) if user_text is not None else ""
        red = self.redactor.redact_text(user_text)
        with self._lock:
            prev = self._session_turn.get(sid)
            idx = self._session_idx.get(sid, 0)
            self._session_idx[sid] = idx + 1
            self._session_turn[sid] = tid
            self._turns[tid] = TurnState(tid, sid, idx, red.text)
            while len(self._turns) > 512:
                self._turns.popitem(last=False)
        if self.store.one("SELECT 1 FROM sessions WHERE id=?", (sid,)) is None:
            self.on_session_start(session_id=sid, platform=kw.get("platform"), model=kw.get("model"))
        self.store.upsert("turns", {"id": tid, "session_id": sid, "idx": idx, "user_text": red.text,
                                    "started_at": self.clock(), "label": "pending"})
        if prev:
            self.regrade(prev, next_user_text=red.text)

    def post_tool_call(self, **kw: Any) -> None:
        tid = kw.get("turn_id")
        sid = kw.get("session_id")
        st = self._turn(tid)
        if not tid or st is None:
            return
        status = str(kw.get("status") or "ok")
        exit_code = _exit_code(kw.get("result"))
        if st.served_by_student and (status != "ok" or exit_code not in (None, 0)):
            # Recovering from a failed or blocked step is the hardest part of a task; a small model
            # trained on successful work is weakest exactly there. Hand the rest of the turn back.
            st.sticky_fallback = True
            self._emit("route.handback", {"turn": tid, "reason": f"tool {status}"})
        call_id = str(kw.get("tool_call_id") or f"{tid}:{kw.get('tool_name')}:{self.clock()}")
        self.store.upsert("tool_calls", {
            "id": call_id, "turn_id": tid, "session_id": sid, "name": kw.get("tool_name"),
            "status": status, "error_type": kw.get("error_type"), "exit_code": exit_code,
            "duration_ms": kw.get("duration_ms"), "created_at": self.clock()})

    def api_request_error(self, **kw: Any) -> None:
        tid = kw.get("turn_id")
        if tid and self._turn(tid):
            with self.store.tx() as c:
                c.execute("UPDATE turns SET api_error=1 WHERE id=?", (tid,))

    def post_llm_call(self, **kw: Any) -> None:
        tid = kw.get("turn_id")
        st = self._turn(tid)
        if not st:
            return
        with self.store.tx() as c:
            c.execute("UPDATE turns SET ended_at=?, served_by=? WHERE id=?",
                      (self.clock(), "student" if st.served_by_student else "teacher", tid))
        self.regrade(tid)

    def post_approval_response(self, **kw: Any) -> None:
        if str(kw.get("choice") or "") != "deny":
            return
        sid = kw.get("session_id") or kw.get("session_key")
        tid = self._session_turn.get(str(sid)) if sid else None
        if tid:
            with self.store.tx() as c:
                c.execute("UPDATE turns SET approval_denied=1 WHERE id=?", (tid,))
            self.regrade(tid)

    def on_session_end(self, **kw: Any) -> None:
        sid = str(kw.get("session_id") or "")
        if not sid:
            return
        completed = kw.get("completed")
        interrupted = bool(kw.get("interrupted"))
        with self.store.tx() as c:
            c.execute("UPDATE sessions SET ended_at=?, completed=?, interrupted=? WHERE id=?",
                      (self.clock(), None if completed is None else int(bool(completed)), int(interrupted), sid))
            last = self._session_turn.get(sid)
            if interrupted and last:
                c.execute("UPDATE turns SET interrupted=1 WHERE id=?", (last,))
        last = self._session_turn.get(sid)
        if last:
            self.regrade(last)

    def on_session_finalize(self, **kw: Any) -> None:
        sid = str(kw.get("session_id") or "")
        if sid:
            last = self._session_turn.pop(sid, None)
            if last:
                self.regrade(last)
        try:
            self.store.sweep(self.settings().retention_days)
        except Exception:
            log.debug("retention sweep failed", exc_info=True)

    def on_session_delete(self, **kw: Any) -> None:
        sid = str(kw.get("session_id") or "")
        if sid:
            self.store.purge_session(sid)
            self._session_turn.pop(sid, None)

    def on_reaction(self, session_id: str, value: int) -> None:
        tid = self._session_turn.get(session_id)
        if tid:
            with self.store.tx() as c:
                c.execute("UPDATE turns SET reaction=? WHERE id=?", (1 if value > 0 else -1, tid))
            self.regrade(tid)

    def mark(self, session_id: str, value: int, *, turn_id: str | None = None) -> str | None:
        """``/tironian good|bad``: explicit label on the session's latest finished turn."""
        tid = turn_id or self._session_turn.get(session_id)
        if not tid:
            row = self.store.one("SELECT id FROM turns WHERE session_id=? ORDER BY idx DESC LIMIT 1", (session_id,))
            tid = row["id"] if row else None
        if not tid:
            return None
        with self.store.tx() as c:
            c.execute("UPDATE turns SET explicit=? WHERE id=?", (1 if value > 0 else -1, tid))
        self.regrade(tid)
        if value < 0:
            self._maybe_demote_for_turn(tid)
        return tid

    # Grading
    def regrade(self, turn_id: str, *, next_user_text: str | None = None) -> None:
        row = self.store.one("SELECT * FROM turns WHERE id=?", (turn_id,))
        if row is None or row["label"] == "excluded":
            return
        sess = self.store.one("SELECT completed, interrupted FROM sessions WHERE id=?", (row["session_id"],))
        if next_user_text is None:
            nxt = self.store.one("SELECT user_text FROM turns WHERE session_id=? AND idx=?",
                                 (row["session_id"], (row["idx"] or 0) + 1))
            next_user_text = nxt["user_text"] if nxt else None
        tools = [ToolOutcome(r["name"] or "", r["status"] or "ok", r["exit_code"])
                 for r in self.store.q("SELECT name, status, exit_code FROM tool_calls WHERE turn_id=?", (turn_id,))]
        g = grade(TurnFacts(
            user_text=row["user_text"] or "", tool_outcomes=tools, next_user_text=next_user_text,
            session_completed=None if sess is None or sess["completed"] is None else bool(sess["completed"]),
            interrupted=bool(row["interrupted"]), approval_denied=bool(row["approval_denied"]),
            api_error=bool(row["api_error"]), reaction=row["reaction"] or 0, explicit=row["explicit"] or 0))
        with self.store.tx() as c:
            c.execute("UPDATE turns SET score=?, label=?, grade_json=? WHERE id=?",
                      (g.score, g.label, json.dumps({"reasons": g.reasons, "hard": g.hard_negative}), turn_id))

    # Routing table
    def promoted(self, *, refresh: bool = False) -> list[PromotedFamily]:
        now = self.clock()
        if not refresh and self._promoted is not None and now - self._promoted_at < 30:
            return self._promoted
        out: list[PromotedFamily] = []
        for r in self.store.q(
                "SELECT f.*, s.model_id FROM families f JOIN students s ON s.id=f.student_id "
                "WHERE f.status IN ('shadow','live')"):
            exemplars = [frozenset(e) for e in json.loads(r["exemplars_json"] or "[]")]
            f = fam.Family(id=r["id"], exemplars=exemplars)
            avg = self.store.scalar(
                "SELECT AVG(d.cost_usd) FROM decisions d JOIN turns t ON t.id=d.turn_id "
                "WHERE t.family_id=? AND d.served_by='teacher' AND d.cost_usd IS NOT NULL", (r["id"],), 0.0)
            raw_steps = self.store.meta_get(f"serve_steps:{r['id']}")
            try:
                steps = frozenset(json.loads(raw_steps)) if raw_steps else frozenset({"first", "follow"})
            except (TypeError, ValueError):
                steps = frozenset()
            out.append(PromotedFamily(f, r["status"], r["model_id"], json.loads(r["toolset_json"] or "[]"),
                                      float(avg or 0.0), steps))
        self._promoted, self._promoted_at = out, now
        try:
            self._common = frozenset(json.loads(self.store.meta_get("common_features") or "[]"))
        except (TypeError, ValueError):
            self._common = frozenset()
        return out

    def invalidate(self) -> None:
        self._promoted = None

    def _route_for(self, st: TurnState, s: Settings, tools: list[dict[str, Any]], had_tools_names: set[str]) -> PromotedFamily | None:
        promoted = self.promoted()
        if not promoted:
            return None
        # Match the way families were built, minus what can't be known yet: which tools the turn
        # will use. Both sides are compared on the request's words only.
        feats = fam.without_common(fam.features(st.user_text), getattr(self, "_common", frozenset()))
        m = fam.match(feats, [p.family for p in promoted], text_only=True)
        if m.family_id is None or m.similarity < s.route_confidence_min or m.margin < s.route_margin_min:
            return None
        chosen = next(p for p in promoted if p.family.id == m.family_id)
        if not set(chosen.toolset) <= had_tools_names:
            return None  # the request doesn't offer the tools this student was trained with
        return chosen

    # Middleware
    def llm_execution(self, **kw: Any) -> Any:
        next_call = kw["next_call"]
        request = kw.get("request")
        if in_tironian_call() or not isinstance(request, dict):
            return next_call(request)
        try:
            plan = self._plan(kw)
        except Exception:
            log.warning("tironian: planning failed; using your model", exc_info=True)
            plan = None

        if plan is not None and plan["mode"] == "live":
            try:
                served = self._serve_student(plan, kw)
            except Exception:
                log.warning("tironian: student path failed; using your model", exc_info=True)
                served = None
            if served is not None:
                return served

        started = time.monotonic()
        response = next_call(request)  # the teacher; never called twice
        latency = int((time.monotonic() - started) * 1000)
        try:
            self._capture_soon(kw, request, response, latency)
            if plan is not None and plan["mode"] in ("shadow", "spot"):
                self._shadow(plan, kw, response)
        except Exception:
            log.debug("tironian capture failed", exc_info=True)
        return response

    def _plan(self, kw: dict[str, Any]) -> dict[str, Any] | None:
        s = self.settings()
        st = self._turn(kw.get("turn_id"))
        if st is None or st.sticky_fallback:
            return None
        # Shadow mode never answers, so it runs whether or not students may answer. Only a live
        # family needs the "let promoted students answer" switch.
        if not s.routing_enabled and not any(p.status == "shadow" for p in self.promoted()):
            return None
        api_mode = str(kw.get("api_mode") or "chat_completions")
        if api_mode not in ("chat_completions", "anthropic_messages"):
            return None
        messages, tools = normalize_request(kw["request"], api_mode)
        names = {tool_name(t) for t in tools}
        if not st.route_checked:
            st.route_checked = True
            chosen = self._route_for(st, s, tools, names)
            if chosen:
                st.route_family, st.route_mode, st.student_model = chosen.family.id, chosen.status, chosen.student_model
        if not st.route_family:
            return None
        chosen = next((p for p in self.promoted() if p.family.id == st.route_family), None)
        if chosen is None:
            return None
        # Only the kinds of step this student passed: e.g. it runs the first command and your
        # model writes the answer from the result. Not sticky; a later step may qualify again.
        if step_kind(messages[-1].get("role") if messages else None) not in chosen.steps:
            return None
        family_tools = select_tools(tools, chosen.toolset)
        compact = compact_messages(messages, family_tools, max_tokens=s.max_context_tokens)
        if compact.had_image or compact.tokens > s.max_context_tokens:
            return None
        if chosen.status == "live" and not s.routing_enabled:
            return None
        mode = chosen.status
        if mode == "live" and self._spot_check(st.turn_id, s):
            # Now and then your model answers a live family's request as before and the student is
            # judged in the background, so a task that changes after training gets noticed.
            mode = "spot"
        return {"mode": mode, "family": chosen, "turn": st, "compact": compact,
                "tools": family_tools, "api_mode": api_mode, "settings": s}

    def _validate(self, text: str, tools: list[dict[str, Any]], st: TurnState) -> tuple[Any, str | None]:
        p = parse_output(text)
        if not p.ok:
            return p, f"invalid:{p.error}"
        schemas = tool_schema_map(tools)
        for c in p.tool_calls:
            if c["name"] not in schemas:
                return p, f"unknown_tool:{c['name']}"
            bad = validate_args(c["arguments"], schemas[c["name"]])
            if bad:
                return p, f"schema:{bad}"
        if not p.tool_calls and len(p.content) < 2:
            return p, "empty"
        return p, None

    def _serve_student(self, plan: dict[str, Any], kw: dict[str, Any]) -> Any | None:
        s: Settings = plan["settings"]
        st: TurnState = plan["turn"]
        chosen: PromotedFamily = plan["family"]
        with reentrant():
            reply = self._student(s).complete(chosen.student_model, plan["compact"].messages)
        outcome = None
        response = None
        if not reply.ok:
            outcome = "fb_error" if reply.error != "truncated" else "fb_invalid"
        elif not model_matches(reply.model, chosen.student_model):
            outcome = "fb_wrong_model"
        else:
            parsed, why = self._validate(reply.text, plan["tools"], st)
            if why:
                outcome = "fb_invalid"
            else:
                response = build_response(plan["api_mode"], parsed.content, parsed.tool_calls,
                                          model=f"tironian/{chosen.student_model}",
                                          in_tok=reply.in_tok, out_tok=reply.out_tok)
                outcome = "served" if response is not None else "fb_unsupported"
        saved = chosen.avg_cost_usd if outcome == "served" else 0.0
        self.store.insert("routes", {
            "turn_id": st.turn_id, "decision_id": kw.get("api_request_id"), "family_id": chosen.family.id,
            "mode": "live", "student_id": chosen.student_model, "outcome": outcome,
            "latency_ms": reply.latency_ms, "saved_usd": saved, "created_at": self.clock()})
        if response is None:
            st.sticky_fallback = True
            self._check_auto_demote(chosen.family.id)
            self._emit("route.fallback", {"family": chosen.family.id, "reason": outcome})
            return None
        st.served_by_student = True
        st.call_idx += 1
        self._emit("route.served", {"family": chosen.family.id, "latency_ms": reply.latency_ms, "saved_usd": saved})
        return response

    def _shadow_judge(self) -> Any:
        """One judge per day, so its call budget is a daily one here."""
        day = time.strftime("%Y-%m-%d", time.localtime(self.clock()))
        if day != self._judge_day:
            self._judge_day = day
            try:
                self._judge = self.judge_factory() if self.judge_factory else None
            except Exception:
                log.debug("tironian: no judge for shadow", exc_info=True)
                self._judge = None
        return self._judge

    def _shadow(self, plan: dict[str, Any], kw: dict[str, Any], teacher_response: Any) -> None:
        """Ask the student the same question in the background and record agreement."""
        if not self._shadow_busy.acquire(blocking=False):
            return  # one shadow replay at a time; dropping samples is fine, blocking is not
        teacher = normalize_response(teacher_response, plan["api_mode"]) or {}

        def run() -> None:
            try:
                from .evaluate import check_decision
                s: Settings = plan["settings"]
                chosen: PromotedFamily = plan["family"]
                # Nothing waits on a shadow answer, so a slow cold start isn't a failure here.
                client = self._student(s)
                client.timeout_s = max(120.0, float(s.student_timeout_s))
                with reentrant():
                    reply = client.complete(chosen.student_model, plan["compact"].messages)
                if not reply.ok:
                    outcome = "shadow_error"
                else:
                    # Shadow agreement is judged like the held-out test, or a different but
                    # equally right command would always count against the student.
                    self._shadow_n += 1
                    chk = check_decision(reply.text, teacher.get("content") or "", teacher.get("tool_calls") or [],
                                         plan["tools"], user_request=plan["turn"].user_text,
                                         judge=self._shadow_judge(), flip=bool(self._shadow_n % 2))
                    outcome = "shadow_agree" if chk.passed else "shadow_disagree"
                    # Keep the why, so `/tironian why` can explain a student that isn't promoted.
                    self.store.meta_set(f"shadow_last:{chosen.family.id}", json.dumps({
                        "outcome": outcome, "reason": chk.reason, "judge": chk.text,
                        "judge_on": self._judge is not None, "student": reply.text[:600],
                        "teacher": (teacher.get("tool_calls") or teacher.get("content") or "")}, default=str)[:4000])
                self.store.insert("routes", {
                    "turn_id": plan["turn"].turn_id, "decision_id": kw.get("api_request_id"),
                    "family_id": chosen.family.id, "mode": plan["mode"], "student_id": chosen.student_model,
                    "outcome": outcome, "latency_ms": reply.latency_ms, "saved_usd": 0.0,
                    "created_at": self.clock()})
                self._emit("route.shadow", {"family": chosen.family.id, "outcome": outcome, "mode": plan["mode"]})
                if plan["mode"] == "spot":
                    self._check_drift(chosen.family.id, s)
            except Exception:
                log.debug("tironian shadow replay failed", exc_info=True)
            finally:
                self._shadow_busy.release()

        threading.Thread(target=run, name="tironian-shadow", daemon=True).start()

    def _spot_check(self, turn_id: str, s: Settings) -> bool:
        """Pick about `spot_check_rate` of a live family's requests, the same way for the same turn."""
        rate = float(getattr(s, "spot_check_rate", 0.0) or 0.0)
        if rate <= 0:
            return False
        h = int(hashlib.sha1(str(turn_id).encode()).hexdigest()[:8], 16) % 10000
        return h < rate * 10000

    def _since(self, family_id: str) -> float:
        """When the family last changed status. Evidence from before that doesn't count."""
        try:
            return float(self.store.meta_get(f"status_since:{family_id}") or 0)
        except (TypeError, ValueError):
            return 0.0

    def _check_drift(self, family_id: str, s: Settings, *, window: int = 20, min_n: int = 10) -> None:
        """A live student that stops agreeing with your model on spot checks goes back to shadow."""
        rows = self.store.q("SELECT outcome FROM routes WHERE family_id=? AND mode='spot' AND outcome IN "
                            "('shadow_agree','shadow_disagree') AND created_at>=? ORDER BY id DESC LIMIT ?",
                            (family_id, self._since(family_id), window))
        n = len(rows)
        if n < min_n:
            return
        agree = sum(1 for r in rows if r["outcome"] == "shadow_agree")
        if agree / n < s.promotion_threshold:
            reason = f"drift: agreed with your model on {agree} of its last {n} spot checks"
            self.store.meta_set(f"drift:{family_id}", json.dumps({"agree": agree, "n": n, "at": self.clock()}))
            self.set_family_status(family_id, "shadow", reason=reason)

    def _check_auto_demote(self, family_id: str) -> None:
        rows = self.store.q("SELECT outcome FROM routes WHERE family_id=? AND mode='live' AND created_at>=? "
                            "ORDER BY id DESC LIMIT 50", (family_id, self._since(family_id)))
        if len(rows) >= 20:
            fb = sum(1 for r in rows if r["outcome"] != "served")
            if fb / len(rows) > 0.10:
                self.set_family_status(family_id, "shadow", reason="fallback rate above 10%")

    def _maybe_demote_for_turn(self, turn_id: str) -> None:
        r = self.store.one("SELECT family_id FROM routes WHERE turn_id=? AND outcome='served' LIMIT 1", (turn_id,))
        if not r:
            return
        bad = self.store.scalar(
            "SELECT COUNT(*) FROM turns t WHERE t.explicit<0 AND t.id IN "
            "(SELECT turn_id FROM routes WHERE family_id=? AND outcome='served' AND created_at>=?)",
            (r["family_id"], self._since(r["family_id"])), 0)
        if bad >= 2:
            self.set_family_status(r["family_id"], "shadow", reason="marked bad twice")

    def set_family_status(self, family_id: str, status: str, *, reason: str = "") -> None:
        with self.store.tx() as c:
            c.execute("UPDATE families SET status=?, updated_at=? WHERE id=?", (status, self.clock(), family_id))
        # Each status starts a fresh window of evidence: shadow samples, fallbacks, bad marks and
        # spot checks from before this change say nothing about the student's next stretch.
        self.store.meta_set(f"status_since:{family_id}", repr(self.clock()))
        self.store.meta_set(f"status_reason:{family_id}", reason)
        self.invalidate()
        self._emit("family.status", {"family": family_id, "status": status, "reason": reason})

    # Capture
    def _capture_soon(self, kw: dict[str, Any], request: dict[str, Any], response: Any, latency_ms: int) -> None:
        """Hand the step to a background worker so the turn never waits on masking and storage.

        The turn's step counter advances here, synchronously, so steps keep their order. If the
        queue is full (a burst on a slow disk), the step is dropped from capture, never delayed.
        """
        if not self.settings().capture_enabled:
            return
        st = self._turn(kw.get("turn_id"))
        if st is None:
            return
        idx = st.call_idx
        st.call_idx += 1
        item = (dict(kw), request, response, latency_ms, idx)
        if self.sync_capture:
            self._capture(*item[:4], served_by="teacher", call_idx=idx)
            return
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._capture_worker, name="tironian-capture", daemon=True)
                self._worker.start()
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def _capture_worker(self) -> None:
        while True:
            kw, request, response, latency_ms, idx = self._queue.get()
            try:
                self._capture(kw, request, response, latency_ms, served_by="teacher", call_idx=idx)
            except Exception:
                log.debug("tironian capture failed", exc_info=True)
            finally:
                self._queue.task_done()

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait until captured steps are written (used by commands that read right after a turn)."""
        end = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.01)
        return not self._queue.unfinished_tasks

    def _capture(self, kw: dict[str, Any], request: dict[str, Any], response: Any, latency_ms: int,
                 *, served_by: str, call_idx: int) -> None:
        s = self.settings()
        if not s.capture_enabled:
            return
        st = self._turn(kw.get("turn_id"))
        if st is None:
            return
        api_mode = str(kw.get("api_mode") or "chat_completions")
        messages, tools = normalize_request(request, api_mode)
        out = normalize_response(response, api_mode)
        if out is None:
            return
        red_msgs, h1 = self.redactor.redact_obj(messages)
        red_out, h2 = self.redactor.redact_obj({"content": out["content"], "tool_calls": out["tool_calls"]})
        hits = sensitive_count(h1) + sensitive_count(h2)
        tools_json = json.dumps(tools, sort_keys=True, ensure_ascii=False)
        tools_hash = hashlib.sha1(tools_json.encode("utf-8")).hexdigest()[:16]
        if self.store.one("SELECT 1 FROM toolsets WHERE hash=?", (tools_hash,)) is None:
            red_tools, _ = self.redactor.redact_obj(tools)
            self.store.upsert("toolsets", {"hash": tools_hash, "tools_json": json.dumps(red_tools, ensure_ascii=False)},
                              key="hash")
        cost = out.get("cost")
        if cost is None:
            cost = (out["in_tok"] * s.teacher_usd_per_mtok_in + out["out_tok"] * s.teacher_usd_per_mtok_out) / 1e6
        did = str(kw.get("api_request_id") or f"{st.turn_id}:{call_idx}")
        self.store.upsert("decisions", {
            "id": did, "turn_id": st.turn_id, "session_id": st.session_id, "call_idx": call_idx,
            "api_mode": api_mode, "provider": kw.get("provider"), "model": kw.get("model"),
            "tools_hash": tools_hash, "context_z": compress(red_msgs), "output_json": json.dumps(red_out, ensure_ascii=False),
            "finish_reason": out.get("finish_reason"), "in_tok": out["in_tok"], "out_tok": out["out_tok"],
            "cost_usd": float(cost), "latency_ms": latency_ms, "served_by": served_by,
            "redaction_hits": hits, "created_at": self.clock()})

    # Stats
    def summary(self) -> dict[str, Any]:
        q = self.store.scalar
        served = q("SELECT COUNT(*) FROM routes WHERE outcome='served'", (), 0)
        fallbacks = q("SELECT COUNT(*) FROM routes WHERE mode='live' AND outcome!='served'", (), 0)
        decisions = q("SELECT COUNT(*) FROM decisions WHERE served_by='teacher'", (), 0)
        return {
            "turns": q("SELECT COUNT(*) FROM turns WHERE label!='excluded'", (), 0),
            "good": q("SELECT COUNT(*) FROM turns WHERE label='good'", (), 0),
            "bad": q("SELECT COUNT(*) FROM turns WHERE label='bad'", (), 0),
            "decisions": decisions,
            "trainable": q("SELECT COUNT(*) FROM decisions d JOIN turns t ON t.id=d.turn_id WHERE t.label='good' "
                           "AND d.served_by='teacher' AND d.redaction_hits<=?", (TRAINING_HIT_LIMIT,), 0),
            "redactions": q("SELECT SUM(redaction_hits) FROM decisions", (), 0),
            "served": served, "fallbacks": fallbacks,
            "local_share": served / (served + decisions) if (served + decisions) else 0.0,
            "saved_usd": round(float(q("SELECT SUM(saved_usd) FROM routes", (), 0.0)), 4),
            "families_live": q("SELECT COUNT(*) FROM families WHERE status='live'", (), 0),
            "families_shadow": q("SELECT COUNT(*) FROM families WHERE status='shadow'", (), 0),
        }


def _exit_code(result: Any) -> int | None:
    """Pull an exit code out of a terminal tool result (JSON string or dict), if any."""
    obj = result
    if isinstance(result, str):
        s = result.strip()
        if not s.startswith("{"):
            return None
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return None
    if isinstance(obj, dict):
        for k in ("exit_code", "returncode", "exitCode", "status_code"):
            v = obj.get(k)
            if isinstance(v, int) and not isinstance(v, bool):
                return v
    return None
