"""Redaction that runs before anything Tironian stores.

Everything that reaches the database (user messages, model output, tool arguments, tool
results, tool schemas) passes through :func:`redact_text` or :func:`redact_obj` first. The
rules are conservative: a false positive masks a harmless string, a false negative would put a
secret into a training set, so ties go to masking.

Masks are stable ASCII placeholders (``[REDACTED:kind]``) so that the model learns the
*shape* of the interaction without ever seeing the value.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# Ordered: specific vendor formats first, generic shapes last, so the most precise label wins.
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private_key", re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z0-9 ]*PRIVATE KEY-----")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openrouter_key", re.compile(r"\bsk-or-[A-Za-z0-9_\-]{20,}")),
    ("openai_key", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("stripe_key", re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("modal_token", re.compile(r"\b(?:ak|as)-[A-Za-z0-9]{20,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}")),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]{16,}=*")),
    ("url_credentials", re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^\s:/@]+:[^\s@/]+@")),
    ("assignment", re.compile(
        r"(?i)\b((?:api[_-]?key|secret|token|passwd|password|pwd|auth|access[_-]?key|client[_-]?secret)"
        r"[\"']?\s*[:=]\s*[\"']?)([^\s\"',;]{6,})")),
]

_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,3})?\b")),
    ("card", re.compile(r"(?<![\w\-])\d(?:[ \-]?\d){12,18}(?![\w\-])")),
    # A phone number needs a leading '+' or at least two separator-delimited groups; bare digit
    # runs (pids, order ids, timestamps) and hex/uuid fragments are left alone.
    ("phone", re.compile(
        r"(?<![\w.\-])(?:\+\d{1,3}[\s.\-]?\(?\d{1,4}\)?(?:[\s.\-]?\d{2,4}){2,3}"
        r"|\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4})(?![\w.\-])")),
]

# Candidate tokens for the entropy check: long, mixed, no spaces.
_TOKEN_CANDIDATE = re.compile(r"[A-Za-z0-9_\-+/=]{32,}")
# Shapes that look random but are not secrets; masking them would destroy useful training signal.
_BENIGN_SHAPES = [
    re.compile(r"^[0-9a-f]{40}$"),                       # git SHA-1
    re.compile(r"^[0-9a-f]{64}$"),                       # sha256 digest
    re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"),  # uuid
    re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$"),             # identifiers: snake_case_function_names
    re.compile(r"^[/\\A-Za-z0-9_.\-]+$"),                # paths without any digit-letter noise
]


def _shannon(s: str) -> float:
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _looks_random(tok: str) -> bool:
    if any(p.match(tok) for p in _BENIGN_SHAPES[:3]):
        return False
    has_digit = any(c.isdigit() for c in tok)
    has_alpha = any(c.isalpha() for c in tok)
    if not (has_digit and has_alpha):
        return False
    if _BENIGN_SHAPES[3].match(tok) and tok.count("_") >= 2:
        return False
    if "/" in tok and _shannon(tok) < 4.5:
        return False  # a path or URL fragment
    return _shannon(tok) > 4.0


@dataclass
class RedactionResult:
    text: str
    hits: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.hits.values())


#: Words that name chat roles. Never treated as a login name.
_ROLE_WORDS = frozenset({"user", "assistant", "system", "tool", "function", "developer", "model"})

#: JSON keys whose string values describe structure, not content.
_STRUCTURAL_KEYS = frozenset({"role", "type", "tool_call_id", "finish_reason"})
#: ...but only when the value looks like an identifier. A path or a sentence is still masked.
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]{0,63}")


class Redactor:
    """Stateless apart from configuration; safe to share across threads."""

    def __init__(self, *, pii: bool = True, home_dirs: Iterable[str] | None = None,
                 usernames: Iterable[str] | None = None) -> None:
        self.pii = pii
        homes = list(home_dirs) if home_dirs is not None else [os.path.expanduser("~")]
        names = set(usernames) if usernames is not None else {os.environ.get("USERNAME") or os.environ.get("USER") or ""}
        for h in homes:
            base = os.path.basename(h.rstrip("/\\"))
            if base:
                names.add(base)
        self._homes = sorted({h for h in homes if h and len(h) > 3}, key=len, reverse=True)
        # A login called "user" (common in containers and cloud VMs) must not mask the word, or
        # every message role and every "the user asked" turns into a mask. The home folder
        # itself is still masked as a path above.
        self._usernames = sorted({n for n in names if n and len(n) >= 3 and n.lower() not in _ROLE_WORDS},
                                 key=len, reverse=True)

    def redact_text(self, text: str) -> RedactionResult:
        """Mask one string. Long strings are cached: each model call resends the whole
        conversation, so the same system prompt and history would otherwise be scanned
        again on every step."""
        if not text:
            return RedactionResult(text or "")
        if len(text) < 64:
            return self._redact(text)
        cache = self.__dict__.setdefault("_cache", {})
        hit = cache.get(text)
        if hit is None:
            hit = self._redact(text)
            if len(cache) >= 2048:
                cache.pop(next(iter(cache)))
            cache[text] = hit
        return RedactionResult(hit.text, dict(hit.hits))

    def _redact(self, text: str) -> RedactionResult:
        hits: dict[str, int] = {}

        def bump(kind: str) -> None:
            hits[kind] = hits.get(kind, 0) + 1

        out = text
        for kind, pat in _SECRET_PATTERNS:
            if kind == "assignment":
                def _assign(m: re.Match[str]) -> str:
                    bump(kind)
                    return f"{m.group(1)}[REDACTED:{kind}]"
                out = pat.sub(_assign, out)
            elif kind == "url_credentials":
                def _creds(m: re.Match[str]) -> str:
                    bump(kind)
                    return f"{m.group(1)}[REDACTED:{kind}]@"
                out = pat.sub(_creds, out)
            else:
                def _mask(m: re.Match[str], kind: str = kind) -> str:
                    bump(kind)
                    return f"[REDACTED:{kind}]"
                out = pat.sub(_mask, out)

        def _entropy(m: re.Match[str]) -> str:
            tok = m.group(0)
            if "REDACTED" in tok or not _looks_random(tok):
                return tok
            bump("high_entropy")
            return "[REDACTED:high_entropy]"
        out = _TOKEN_CANDIDATE.sub(_entropy, out)

        if self.pii:
            for kind, pat in _PII_PATTERNS:
                def _pii(m: re.Match[str], kind: str = kind) -> str:
                    raw = m.group(0)
                    if kind == "card":
                        digits = re.sub(r"\D", "", raw)
                        if not (13 <= len(digits) <= 19 and _luhn_ok(digits)):
                            return raw
                    if kind == "phone":
                        digits = re.sub(r"\D", "", raw)
                        if len(digits) < 9 or re.fullmatch(r"(19|20)\d{2}[\-./]\d{2}[\-./]\d{2}.*", raw.strip()):
                            return raw  # too short for a phone number, or a date
                    bump(kind)
                    return f"[REDACTED:{kind}]"
                out = pat.sub(_pii, out)

        for home in self._homes:
            for variant in {home, home.replace("\\", "/"), home.replace("/", "\\")}:
                if variant in out:
                    hits["home_path"] = hits.get("home_path", 0) + out.count(variant)
                    out = out.replace(variant, "~")
        for name in self._usernames:
            # Case-sensitive: "Admin" in `ls -l` is the login; "admin" in a sentence is a word.
            pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])")
            out, n = pat.subn("[REDACTED:user]", out)
            if n:
                hits["username"] = hits.get("username", 0) + n
        return RedactionResult(out, hits)

    def redact_obj(self, obj: Any) -> tuple[Any, dict[str, int]]:
        """Redact every string inside a JSON-like structure (keys untouched)."""
        totals: dict[str, int] = {}

        def merge(h: dict[str, int]) -> None:
            for k, v in h.items():
                totals[k] = totals.get(k, 0) + v

        def walk(x: Any) -> Any:
            if isinstance(x, str):
                r = self.redact_text(x)
                merge(r.hits)
                return r.text
            if isinstance(x, dict):
                # Structural fields (a message's role, a block's type, call ids) are never
                # personal data, and masking them breaks the conversation's shape.
                return {k: v if k in _STRUCTURAL_KEYS and isinstance(v, str) and _TOKEN.fullmatch(v) else walk(v)
                        for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [walk(v) for v in x]
            return x

        return walk(obj), totals


#: Decisions with more masked values than this are kept for analytics but never trained on.
TRAINING_HIT_LIMIT = 5

#: Masks that are routine on any machine (your home folder, your login name in `ls -l`). They are
#: still masked, but they don't make a step "risky", so they don't count toward the limit above.
ROUTINE_KINDS = frozenset({"home_path", "username"})


def sensitive_count(hits: dict[str, int]) -> int:
    return sum(v for k, v in hits.items() if k not in ROUTINE_KINDS)
