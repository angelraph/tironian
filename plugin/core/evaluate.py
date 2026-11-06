"""Held-out evaluation and the promotion gate.

A student never serves a request because it *seems* good. For each held-out decision (taken
from sessions it never saw in training) the student receives exactly the context the teacher
had and produces its next step. That step passes only if:

* it parses (valid tool-call JSON, no stray tags),
* it takes the same kind of action (tool call vs. final answer),
* it calls the same tool(s) with schema-valid arguments that match the teacher's
  (≥ ``ARGS_PASS`` of keys equal after normalisation), and
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


def args_score(student: dict[str, Any], teacher: dict[str, Any]) -> float:
    keys = set(student) | set(teacher)
    if not keys:
        return 1.0
    same = sum(1 for k in keys if k in student and k in teacher and _norm_val(student[k]) == _norm_val(teacher[k]))
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
        if [c["name"] for c in p.tool_calls] != [c["name"] for c in t_calls]:
            return Check(False, True, action_match=True, reason="tool_mismatch")
        schemas = tool_schema_map(tools)
        for c in p.tool_calls:
            if c["name"] not in schemas:
                return Check(False, True, action_match=True, reason=f"unknown_tool:{c['name']}")
            bad = validate_args(c["arguments"], schemas[c["name"]])
            if bad:
                return Check(False, True, action_match=True, tool_match=True, reason=f"schema:{bad}")
        score = min(args_score(s["arguments"], t["arguments"]) for s, t in zip(p.tool_calls, t_calls))
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
        ok, why = False, f"format validity {validity:.0%} < {validity_floor:.0%}"
    elif lo < threshold:
        ok, why = False, f"lower bound {lo:.0%} < {threshold:.0%}"
    else:
        ok, why = True, f"lower bound {lo:.0%} ≥ {threshold:.0%}"
    return Scorecard(
        n=n, passed=passed, valid=valid,
        tool_steps=len(tool_steps), tool_hits=sum(c.tool_match for c in tool_steps),
        args_mean=statistics.fmean([c.args for c in checks if c.tool_match]) if any(c.tool_match for c in checks) else 0.0,
        wilson_lo=lo, wilson_hi=hi, validity=validity,
        p50_ms=int(statistics.median(latencies_ms)) if latencies_ms else 0,
        gate_passed=ok, gate_reason=why, failures=failures)
