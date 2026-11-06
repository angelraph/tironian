"""Task families: recurring kinds of work, found deterministically.

A *family* is a cluster of turns that ask for the same kind of thing ("what's using disk
space in X", "turn this text into JSON", "write a commit message"). Students are trained and
promoted per family, which keeps each one small, specialised and testable.

Everything here is deterministic and dependency-free: the same history always produces the
same families, so a scorecard can be reproduced and a judge can rerun it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

_URL = re.compile(r"https?://\S+")
_PATH = re.compile(r"(?:[A-Za-z]:)?(?:[~./\\][\w.\-]*)?(?:[/\\][\w.\-~]+)+[/\\]?")
_QUOTED = re.compile(r"\"[^\"]{1,200}\"|'[^']{1,200}'|`[^`]{1,200}`")
_NUM = re.compile(r"\b\d+(?:[.,:]\d+)*\b")
_WORD = re.compile(r"[a-z][a-z0-9_\-]*|<[a-z]+>")
_REDACTED = re.compile(r"\[REDACTED:[a-z_]+\]")

# Words that carry no task signal. Kept short on purpose: verbs like "find", "list", "fix" stay.
_STOP = frozenset("""
a an the and or but if then so to of in on at for with by from as is are was were be been it its
this that these those i me my we our you your he she they them please can could would will just
some any all into about up out over do does did there here what which who how why when
""".split())


def normalize(text: str) -> str:
    """Collapse the variable parts of a request so the *shape* of the task remains."""
    t = text or ""
    t = _REDACTED.sub(" <secret> ", t)
    t = _URL.sub(" <url> ", t)
    t = _QUOTED.sub(" <s> ", t)
    t = _PATH.sub(" <path> ", t)
    t = t.lower()
    t = _NUM.sub(" <n> ", t)
    return " ".join(t.split())


def features(user_text: str, tools: Sequence[str] = (), skill: str | None = None) -> frozenset[str]:
    """Unigrams + bigrams of the normalized request, plus the first tools used and any skill."""
    words = [w for w in _WORD.findall(normalize(user_text)) if w not in _STOP]
    feats: set[str] = set(words)
    feats.update(f"{a} {b}" for a, b in zip(words, words[1:]))
    for name in list(dict.fromkeys(tools))[:3]:
        feats.add(f"T:{name}")
    if skill:
        feats.add(f"S:{skill}")
    return frozenset(feats)


def jaccard(a: frozenset[str] | set[str], b: frozenset[str] | set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter) if inter else 0.0


# --- MinHash prototypes (cheap, stable similarity at route time) ---------------------------

_NUM_PERM = 64
_MASK = (1 << 61) - 1


def _hash(token: str, seed: int) -> int:
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8, salt=seed.to_bytes(8, "little")).digest()
    return int.from_bytes(h, "little") & _MASK


def minhash(feats: Iterable[str]) -> tuple[int, ...]:
    feats = list(feats)
    if not feats:
        return tuple([_MASK] * _NUM_PERM)
    return tuple(min(_hash(f, seed) for f in feats) for seed in range(_NUM_PERM))


def minhash_sim(a: Sequence[int], b: Sequence[int]) -> float:
    return sum(1 for x, y in zip(a, b) if x == y) / _NUM_PERM


# --- Clustering ------------------------------------------------------------------------------

@dataclass
class Family:
    id: str
    members: list[str] = field(default_factory=list)       # turn ids, in time order
    exemplars: list[frozenset[str]] = field(default_factory=list)
    tool_counts: dict[str, int] = field(default_factory=dict)
    term_counts: dict[str, int] = field(default_factory=dict)

    def similarity(self, feats: frozenset[str]) -> float:
        return max((jaccard(feats, e) for e in self.exemplars), default=0.0)

    def add(self, turn_id: str, feats: frozenset[str], tools: Sequence[str], max_exemplars: int) -> None:
        self.members.append(turn_id)
        # Keep exemplars diverse: only add one that is not near-identical to an existing one.
        if len(self.exemplars) < max_exemplars and all(jaccard(feats, e) < 0.9 for e in self.exemplars):
            self.exemplars.append(feats)
        for t in set(tools):
            self.tool_counts[t] = self.tool_counts.get(t, 0) + 1
        for f in feats:
            if not f.startswith(("T:", "S:")) and "<" not in f:
                self.term_counts[f] = self.term_counts.get(f, 0) + 1

    def label(self, k: int = 3) -> str:
        """Name from the most frequent distinctive terms, bigrams preferred."""
        terms = sorted(self.term_counts.items(), key=lambda kv: (-kv[1] - (0.5 if " " in kv[0] else 0), kv[0]))
        picked: list[str] = []
        for term, _ in terms:
            if any(term in p or p in term for p in picked):
                continue
            picked.append(term)
            if len(picked) == k:
                break
        return " · ".join(picked) if picked else self.id

    def toolset(self, min_share: float = 0.2) -> list[str]:
        n = max(len(self.members), 1)
        return sorted(t for t, c in self.tool_counts.items() if c / n >= min_share)


@dataclass
class Item:
    turn_id: str
    user_text: str
    tools: Sequence[str] = ()
    skill: str | None = None


def cluster(items: Sequence[Item], *, threshold: float = 0.35, max_exemplars: int = 8) -> list[Family]:
    """Deterministic leader clustering in input (time) order, then a merge pass.

    Each item joins the most similar family above ``threshold`` or starts a new one. Family ids
    are content hashes of their first member, so they are stable across re-runs.
    """
    families: list[Family] = []
    for it in items:
        feats = features(it.user_text, it.tools, it.skill)
        if not feats:
            continue
        best, best_sim = None, 0.0
        for fam in families:
            s = fam.similarity(feats)
            if s > best_sim:
                best, best_sim = fam, s
        if best is None or best_sim < threshold:
            fid = "f" + hashlib.sha1(it.turn_id.encode()).hexdigest()[:8]
            best = Family(id=fid)
            families.append(best)
        best.add(it.turn_id, feats, list(it.tools), max_exemplars)

    # Merge pass: families whose exemplars overlap strongly are the same task phrased differently.
    merged = True
    while merged:
        merged = False
        for i in range(len(families)):
            for j in range(i + 1, len(families)):
                a, b = families[i], families[j]
                link = max((a.similarity(e) for e in b.exemplars), default=0.0)
                if link >= threshold + 0.15:
                    for tid in b.members:
                        a.members.append(tid)
                    for e in b.exemplars:
                        if len(a.exemplars) < max_exemplars and all(jaccard(e, x) < 0.9 for x in a.exemplars):
                            a.exemplars.append(e)
                    for t, c in b.tool_counts.items():
                        a.tool_counts[t] = a.tool_counts.get(t, 0) + c
                    for t, c in b.term_counts.items():
                        a.term_counts[t] = a.term_counts.get(t, 0) + c
                    del families[j]
                    merged = True
                    break
            if merged:
                break
    families.sort(key=lambda f: (-len(f.members), f.id))
    return families


@dataclass
class RouteMatch:
    family_id: str | None
    similarity: float
    margin: float


def match(feats: frozenset[str], families: Sequence[Family]) -> RouteMatch:
    """Best family for a new request, with the margin over the runner-up (ambiguity guard)."""
    scored = sorted(((f.similarity(feats), f.id) for f in families), reverse=True)
    if not scored:
        return RouteMatch(None, 0.0, 0.0)
    best_sim, best_id = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    return RouteMatch(best_id, best_sim, best_sim - second)
