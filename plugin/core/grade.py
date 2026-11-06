"""Deterministic turn grading.

A student is only ever trained on turns that went *well*, so grading decides what Tironian
learns. Signals come from what actually happened, never from the model's opinion of itself:
tool outcomes, exit codes, whether the user corrected or repeated the request, approvals,
interrupts, reactions, and explicit ``/tironian good|bad`` marks.

An optional LLM judge (budgeted, off by default) can only resolve turns that the deterministic
signals leave ``unknown``; it never overrides a hard negative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from .families import features, jaccard

CORRECTION = re.compile(
    r"^\s*(?:no\b|nope\b|wrong\b|that'?s not\b|that is not\b|not what i\b|undo\b|revert\b|"
    r"you didn'?t\b|you did not\b|stop\b|actually[, ]|try again\b|that'?s incorrect\b|incorrect\b|"
    r"doesn'?t work\b|didn'?t work\b|still (?:broken|failing|wrong)\b|why did you\b)",
    re.IGNORECASE,
)
ACCEPT = re.compile(
    r"^\s*(?:thanks|thank you|thx|perfect|great|nice|awesome|works|that worked|exactly|cool|ok(?:ay)?\b.*thanks)",
    re.IGNORECASE,
)

GOOD_AT = 0.3
BAD_AT = -0.3


@dataclass
class ToolOutcome:
    name: str
    status: str = "ok"            # ok | error | blocked | cancelled | timeout
    exit_code: int | None = None


@dataclass
class TurnFacts:
    user_text: str
    tool_outcomes: Sequence[ToolOutcome] = ()
    next_user_text: str | None = None      # None = no follow-up (yet)
    session_completed: bool | None = None
    interrupted: bool = False
    approval_denied: bool = False
    api_error: bool = False
    reaction: int = 0                      # +1 thumbs up, -1 thumbs down, 0 none
    explicit: int = 0                      # +1 /tironian good, -1 /tironian bad


@dataclass
class Grade:
    score: float
    label: str                             # good | bad | unknown
    reasons: list[str] = field(default_factory=list)
    hard_negative: bool = False


def grade(f: TurnFacts) -> Grade:
    reasons: list[str] = []
    hard = False

    if f.explicit:
        label = "good" if f.explicit > 0 else "bad"
        return Grade(float(f.explicit), label, [f"explicit:{label}"], hard_negative=f.explicit < 0)

    score = 0.0
    if f.session_completed:
        score += 0.2
        reasons.append("session_completed")

    errors = [t for t in f.tool_outcomes if t.status != "ok" or (t.exit_code not in (None, 0))]
    if f.tool_outcomes and not errors:
        score += 0.2
        reasons.append("tools_ok")
    for t in errors:
        score -= 0.3
        reasons.append(f"tool_failed:{t.name}")

    if f.next_user_text is not None:
        nxt = f.next_user_text
        if CORRECTION.search(nxt):
            score -= 0.5
            hard = True
            reasons.append("user_corrected")
        elif jaccard(features(f.user_text), features(nxt)) >= 0.7:
            score -= 0.4
            hard = True
            reasons.append("user_repeated")
        elif ACCEPT.search(nxt):
            score += 0.3
            reasons.append("user_thanked")
        else:
            score += 0.1
            reasons.append("moved_on")

    if f.approval_denied:
        score -= 0.5
        hard = True
        reasons.append("approval_denied")
    if f.interrupted:
        score -= 0.5
        hard = True
        reasons.append("interrupted")
    if f.api_error:
        score -= 0.2
        reasons.append("api_error")
    if f.reaction:
        score += 0.7 * (1 if f.reaction > 0 else -1)
        reasons.append("reaction_up" if f.reaction > 0 else "reaction_down")
        hard = hard or f.reaction < 0

    score = max(-1.0, min(1.0, score))
    if score >= GOOD_AT and not hard:
        label = "good"
    elif score <= BAD_AT:
        label = "bad"
    else:
        label = "unknown"
    return Grade(round(score, 3), label, reasons, hard_negative=hard)
