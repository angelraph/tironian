"""Held-out evaluation and the promotion gate.

A student never serves a request because it *seems* good. For each held-out decision (taken
from sessions it never saw in training) the student receives exactly the context the teacher
had and produces its next step. That step passes only if:

* it parses (valid tool-call JSON, no stray tags),
* it takes the same kind of action (tool call vs. final answer),
* it calls the same tool(s) with schema-valid arguments that match the teacher's
  (≥ ``ARGS_PASS`` of keys equal after normalisation; long free-text values such as shell
  commands match when ≥ ``FREE_TEXT_PASS`` of their words agree), and
* for a final answer, a blind pairwise judge rates it as good as the teacher's (or, without a
  judge, its token-level F1 against the teacher's answer is ≥ ``TEXT_F1_PASS``).

The family is promoted only if the **lower** bound of the Wilson 95% interval on the pass rate
clears the threshold, with a minimum sample size and a near-perfect format-validity floor.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .hermes_format import Parsed, parse_output, tool_schema_map, validate_args

ARGS_PASS = 0.8
TEXT_F1_PASS = 0.6
Z95 = 1.959963984540054


def wilson(passed: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = passed / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


_WS = re.compile(r"\s+")


def _norm_val(v: Any) -> Any:
    if isinstance(v, str):
        s = _WS.sub(" ", v.strip().strip("'\"")).replace("\\", "/").rstrip("/")
        try:
            f = float(s)
            return int(f) if f.is_integer() else f
        except ValueError:
            return s.lower() if len(s) < 64 else s
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, dict):
        return {k: _norm_val(x) for k, x in sorted(v.items())}
    if isinstance(v, list):
        return [_norm_val(x) for x in v]
    return v


#: Two long free-text values (a shell command, a search query) count as the same when this share
#: of their words agree. Exact equality would fail `find "$HOME/Videos" -iname '*.docx'` against
#: `find ~/Videos -type f -iname "*.docx"`, which do the same thing.
FREE_TEXT_PASS = 0.6

_HOME = re.compile(r"(?i)(?:\$\{?home\}?|\$\{?userprofile\}?|%userprofile%|(?:/c|c:)?/users/(?:\[redacted:user\]|[^/\s\"']+))")
_LOCALAPPDATA = re.compile(r"(?i)\$\{?localappdata\}?|%localappdata%")
_NOISE = {"2>/dev/null", "2>nul", "2>&1"}


def _words(s: str) -> set[str]:
    s = s.replace("\\", "/")
    s = _LOCALAPPDATA.sub("~/appdata/local", s)
    s = _HOME.sub("~", s)
    toks = re.findall(r"[^\s\"'`]+", s.lower())
    return {t.rstrip(";") for t in toks if t not in _NOISE}


def _same(a: Any, b: Any) -> bool:
    na, nb = _norm_val(a), _norm_val(b)
    if na == nb:
        return True
    if isinstance(a, str) and isinstance(b, str) and (len(a) > 20 or len(b) > 20):
        wa, wb = _words(a), _words(b)
        return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= FREE_TEXT_PASS
    return False


def args_score(student: dict[str, Any], teacher: dict[str, Any]) -> float:
    keys = set(student) | set(teacher)
    if not keys:
        return 1.0
    same = sum(1 for k in keys if k in student and k in teacher and _same(student[k], teacher[k]))
    return same / len(keys)


def token_f1(a: str, b: str) -> float:
    ta, tb = re.findall(r"\w+", a.lower()), re.findall(r"\w+", b.lower())
    if not ta or not tb:
        return 1.0 if ta == tb else 0.0
    common: dict[str, int] = {}
    cb: dict[str, int] = {}
    for t in tb:
        cb[t] = cb.get(t, 0) + 1
    for t in ta:
        if cb.get(t, 0) > 0:
            common[t] = common.get(t, 0) + 1
            cb[t] -= 1
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(ta), overlap / len(tb)
    return 2 * prec * rec / (prec + rec)


#: judge(user_request, answer_a, answer_b) -> "a" | "b" | "tie" | None (None = unavailable)
Judge = Callable[[str, str, str], "str | None"]


@dataclass
class Check:
    passed: bool
    valid: bool
    action_match: bool = False
    tool_match: bool = False
    args: float = 0.0
    text: str = ""            # how a final answer was judged: judge-win|judge-tie|judge-loss|f1=…
    reason: str = ""


def check_decision(student_text: str, teacher_content: str, teacher_calls: Sequence[dict[str, Any]],
                   tools: Sequence[dict[str, Any]], *, user_request: str = "", judge: Judge | None = None,
                   flip: bool = False) -> Check:
    """Compare one student step with the teacher's recorded step.

    ``teacher_calls`` are OpenAI-style ``tool_calls`` entries. ``flip`` swaps the A/B order shown
    to the judge (callers alternate it so position bias cancels out).
    """
    p: Parsed = parse_output(student_text)
    if not p.ok:
        return Check(False, False, reason=f"invalid:{p.error}")

    t_calls = []
    for tc in teacher_calls or []:
        fn = tc.get("function") or {}
        raw = fn.get("arguments")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                raw = {}
        t_calls.append({"name": fn.get("name"), "arguments": raw or {}})

    if bool(p.tool_calls) != bool(t_calls):
        return Check(False, True, reason="action_mismatch")

    if t_calls:
        if [c["name"] for c in p.tool_calls] == [c["name"] for c in t_calls]:
            pairs = list(zip(p.tool_calls, t_calls))
        else:
            # The teacher sometimes runs a few commands at once. A student that runs some of
            # those same commands took a correct step; anything else is a different step.
            pairs, left = [], list(t_calls)
            for s in p.tool_calls:
                same = [t for t in left if t["name"] == s["name"]]
                best = max(same, key=lambda t: args_score(s["arguments"], t["arguments"]), default=None)
                if best is None or args_score(s["arguments"], best["arguments"]) < ARGS_PASS:
                    return Check(False, True, action_match=True, reason="tool_mismatch")
                pairs.append((s, best))
                left.remove(best)
        schemas = tool_schema_map(tools)
        for c in p.tool_calls:
            if c["name"] not in schemas:
                return Check(False, True, action_match=True, reason=f"unknown_tool:{c['name']}")
            bad = validate_args(c["arguments"], schemas[c["name"]])
            if bad:
                return Check(False, True, action_match=True, tool_match=True, reason=f"schema:{bad}")
        score = min(args_score(s["arguments"], t["arguments"]) for s, t in pairs)
        if score < ARGS_PASS and judge is not None:
            # A different command can be just as right (`ls *.png | wc -l` against `find -iname`).
            # The same blind pairwise judge decides; position alternates with ``flip``.
            def show(calls: Sequence[dict[str, Any]]) -> str:
                return "\n".join(f"Runs {c['name']} with {json.dumps(c['arguments'], ensure_ascii=False)}" for c in calls)
            a, b = (show(t_calls), show(p.tool_calls)) if flip else (show(p.tool_calls), show(t_calls))
            raw = judge(user_request, a, b)
            if raw in ("a", "b", "tie"):
                student_slot = "b" if flip else "a"
                ok = raw == "tie" or raw == student_slot
                return Check(ok, True, True, True, score, text=f"judge-{'tie' if raw == 'tie' else ('win' if ok else 'loss')}",
                             reason="" if ok else "judge_preferred_teacher")
        return Check(score >= ARGS_PASS, True, True, True, score,
                     reason="" if score >= ARGS_PASS else f"args={score:.2f}")

    # Final-answer step.
    verdict = None
    if judge is not None:
        a, b = (teacher_content, p.content) if flip else (p.content, teacher_content)
        raw = judge(user_request, a, b)
        if raw in ("a", "b", "tie"):
            student_slot = "b" if flip else "a"
            verdict = "tie" if raw == "tie" else ("win" if raw == student_slot else "loss")
    if verdict is not None:
        return Check(verdict in ("win", "tie"), True, True, True, 1.0, text=f"judge-{verdict}",
                     reason="" if verdict != "loss" else "judge_preferred_teacher")
    f1 = token_f1(p.content, teacher_content)
    return Check(f1 >= TEXT_F1_PASS, True, True, True, 1.0, text=f"f1={f1:.2f}",
                 reason="" if f1 >= TEXT_F1_PASS else f"text_f1={f1:.2f}")


@dataclass
class Scorecard:
    n: int
    passed: int
    valid: int
    tool_steps: int
    tool_hits: int
    args_mean: float
    wilson_lo: float
    wilson_hi: float
    validity: float
    p50_ms: int
    gate_passed: bool
    gate_reason: str
    failures: dict[str, int] = field(default_factory=dict)
    #: Per step kind ("first": answering the request; "follow": the step after a tool result),
    #: each with its own n, pass count, bounds and gate. ``serve_steps`` lists the kinds that passed.
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)
    serve_steps: list[str] = field(default_factory=list)
    fresh: int = 0   # held-out steps from sessions recorded after the dataset was built

    @property
    def pass_rate(self) -> float:
        return self.passed / self.n if self.n else 0.0


def scorecard(checks: Sequence[Check], latencies_ms: Sequence[int], *, threshold: float = 0.85,
              min_n: int = 30, validity_floor: float = 0.98) -> Scorecard:
    n = len(checks)
    passed = sum(c.passed for c in checks)
    valid = sum(c.valid for c in checks)
    tool_steps = [c for c in checks if c.tool_match or c.reason == "tool_mismatch"]
    lo, hi = wilson(passed, n)
    validity = valid / n if n else 0.0
    failures: dict[str, int] = {}
    for c in checks:
        if not c.passed:
            key = c.reason.split("=")[0].split(":")[0] or "failed"
            failures[key] = failures.get(key, 0) + 1
    if n < min_n:
        ok, why = False, f"needs ≥{min_n} held-out examples (has {n})"
    elif validity < validity_floor:
        ok, why = False, f"format validity {_below(validity, validity_floor)} < {validity_floor:.0%}"
    elif lo < threshold:
        ok, why = False, f"lower bound {_below(lo, threshold)} < {threshold:.0%}"
    else:
        ok, why = True, f"lower bound {lo:.0%} ≥ {threshold:.0%}"
    return Scorecard(
        n=n, passed=passed, valid=valid,
        tool_steps=len(tool_steps), tool_hits=sum(c.tool_match for c in tool_steps),
        args_mean=statistics.fmean([c.args for c in checks if c.tool_match]) if any(c.tool_match for c in checks) else 0.0,
        wilson_lo=lo, wilson_hi=hi, validity=validity,
        p50_ms=int(statistics.median(latencies_ms)) if latencies_ms else 0,
        gate_passed=ok, gate_reason=why, failures=failures)


STEP_KINDS = ("first", "follow")


def step_kind(last_role: str | None) -> str:
    """A step answers the request ("first") or continues after a tool result ("follow")."""
    return "follow" if last_role in ("tool", "function") else "first"


def gated_scorecard(checks: Sequence[Check], latencies_ms: Sequence[int], kinds: Sequence[str], *,
                    threshold: float = 0.85, min_n: int = 30, validity_floor: float = 0.98) -> Scorecard:
    """The overall scorecard, plus the same gate applied to each step kind on its own.

    A student that has mastered the first step of a task but not what comes after a tool result
    can still serve that first step; your model takes the rest of the turn. Each kind needs its
    own ``min_n`` held-out examples and its own Wilson lower bound, so passing is never easier.
    """
    sc = scorecard(checks, latencies_ms, threshold=threshold, min_n=min_n, validity_floor=validity_floor)
    for kind in STEP_KINDS:
        idx = [i for i, k in enumerate(kinds) if k == kind]
        if not idx:
            continue
        sub = scorecard([checks[i] for i in idx], [latencies_ms[i] for i in idx if i < len(latencies_ms)],
                        threshold=threshold, min_n=min_n, validity_floor=validity_floor)
        sc.steps[kind] = {"n": sub.n, "passed": sub.passed, "wilson_lo": sub.wilson_lo, "wilson_hi": sub.wilson_hi,
                          "validity": sub.validity, "gate_passed": sub.gate_passed, "gate_reason": sub.gate_reason}
        if sub.gate_passed:
            sc.serve_steps.append(kind)
    if sc.serve_steps and not sc.gate_passed:
        sc.gate_passed = True
        sc.gate_reason = "passed for " + " and ".join(
            f"{k} steps ({sc.steps[k]['gate_reason']})" for k in sc.serve_steps) + "; your model handles the rest"
    elif sc.gate_passed:
        # The whole set passed. A kind too small to pass alone rides along only if its own
        # pass rate and validity are as good as the gate asks for, so a weak kind can't hide.
        for kind, d in sc.steps.items():
            if kind not in sc.serve_steps and d["n"] and d["passed"] / d["n"] >= threshold \
                    and d["validity"] >= validity_floor:
                sc.serve_steps.append(kind)
        sc.serve_steps = [k for k in STEP_KINDS if k in sc.serve_steps]
        if not sc.serve_steps:
            sc.gate_passed, sc.gate_reason = False, "no step kind is good enough on its own"
    elif sc.steps:
        # Say why each kind failed, not just the first overall check that tripped.
        sc.gate_reason = "; ".join(f"{k} steps: {d['gate_reason']}" for k, d in sc.steps.items())
    return sc


def _below(x: float, floor: float) -> str:
    """``x`` as a percentage that never rounds up to ``floor`` when it is below it."""
    s = f"{x:.0%}"
    return s if s != f"{floor:.0%}" else f"{x:.1%}"
