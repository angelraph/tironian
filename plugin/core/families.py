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
_EXT = re.compile(r"(?<![\w])\.(?:[a-z0-9]{1,5})\b", re.IGNORECASE)
# Capitalised words mid-sentence, including CamelCase names like AppData or OneDrive.
_NAME = re.compile(r"([a-z,;:]\s+)(?:[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)")
_REDACTED = re.compile(r"\[REDACTED:[a-z_]+\]")

# Words that carry no task signal. Kept short on purpose: verbs like "find", "list", "fix" stay.
_STOP = frozenset("""
a an the and or but if then so to of in on at for with by from as is are was were be been it its
this that these those i me my we our you your he she they them please can could would will just
some any all into about up out over under do does did there here what which who how why when their
his her
""".split())


# Different words people use for the same request. Phrases are matched before single words.
# Kept small and boring on purpose: it should merge obvious paraphrases, not guess intent.
_SYNONYMS: list[tuple[re.Pattern[str], str]] = [(re.compile(rf"\b(?:{a})\b"), b) for a, b in [
    (r"take(?:s)? up the most (?:space|room)|use(?:s)? the most (?:space|room)|eating (?:up )?(?:space|disk)|"
     r"biggest|largest|heaviest", "largest"),
    (r"most recently (?:modified|changed|edited|updated)|changed most recently|modified most recently|"
     r"(?:modified|changed|edited|updated) last|newest|latest|most recent|recently (?:modified|changed)", "recent"),
    (r"how many|count|number of", "count"),
    (r"directory|directories|dir|folders?", "folder"),
    (r"show(?: me)?|list|display|find|give me|tell me|what are|which", "show"),
    (r"turn into|transform|convert(?:ing)?|change into", "convert"),
    (r"summari[sz]e|sum up|tl;?dr|give (?:me )?a summary of", "summarize"),
]]


# Words that say how something was asked, not what was asked. Kept out of family names only.
_LABEL_FILLER = frozenset("""
use using used check checking show tell help want need get see look run terminal command commands
can could would should give make let quickly please also just
""".split())


# Words about how a request is phrased, not what it asks for. "Check with the terminal" and
# "using the terminal" are the same request as "use the terminal". Dropped from features so
# that phrasing alone doesn't split one task into several families.
_PHRASING = frozenset("""
check checking using use used have has extension extensions command quickly also just let
show folder
""".split())

# File types written without the dot ("how many docx files") are arguments too.
_BARE_EXT = re.compile(r"\b(?:pdf|exe|zip|png|jpe?g|gif|txt|csv|json|xml|ya?ml|mp3|mp4|mov|docx?|xlsx?|pptx?|"
                       r"py|js|ts|html?|css|md|log|iso|msi|dll|tmp|bak)\b(?=\s+files?\b)")


def normalize(text: str) -> str:
    """Collapse the variable parts of a request so the *shape* of the task remains."""
    t = text or ""
    t = _REDACTED.sub(" <secret> ", t)
    t = _URL.sub(" <url> ", t)
    t = _QUOTED.sub(" <s> ", t)
    t = _PATH.sub(" <path> ", t)
    # Arguments, not task: ".pdf" files, and names written with a capital mid-sentence
    # ("in my Downloads folder", "email Sarah"). The first word of a sentence keeps its case meaning.
    t = _EXT.sub(" <ext> ", t)
    t = _NAME.sub(lambda m: m.group(1) + " <name>", t)
    t = t.lower()
    t = _BARE_EXT.sub(" <ext> ", t)
    t = _NUM.sub(" <n> ", t)
    for pat, word in _SYNONYMS:
        t = pat.sub(word, t)
    return " ".join(t.split())


def common_features(item_feats: Sequence[frozenset[str]], *, max_df: float = 0.5, min_items: int = 20,
                    hard_df: float = 0.9) -> frozenset[str]:
    """Features that say nothing about *which* task a request is ("use the terminal", "my",
    the one tool everything uses). They are ignored when grouping.

    A feature in nearly every request (over ``hard_df``) is always common. One in more than
    ``max_df`` is common only if it is spread across most kinds of request. If one task makes
    up most of someone's history, its own word ("count") is frequent too, but it marks that
    task, so it stays.
    """
    n = len(item_feats)
    if n < min_items:
        return frozenset()
    df: dict[str, int] = {}
    for fs in item_feats:
        for f in fs:
            df[f] = df.get(f, 0) + 1
    hard = frozenset(f for f, c in df.items() if c / n > hard_df)
    frequent = [f for f, c in df.items() if max_df < c / n <= hard_df]
    if not frequent:
        return hard
    # Rough groups with only the near-universal words removed, then see where each
    # frequent word lives.
    groups = _group([str(i) for i in range(n)], list(item_feats), [()] * n, hard,
                    threshold=0.35, max_exemplars=8, merge_at=None)
    by_id = dict(zip((str(i) for i in range(n)), item_feats))
    big = [g for g in groups if len(g.members) >= max(3, n * 0.05)]
    if len(big) < 2:
        return hard | frozenset(frequent)   # one kind of request: nothing to tell apart
    spread = set()
    for f in frequent:
        typical = sum(1 for g in big if sum(f in by_id[m] for m in g.members) * 2 >= len(g.members))
        if typical > len(big) / 2:
            spread.add(f)
    return hard | frozenset(spread)


def without_common(feats: frozenset[str], common: frozenset[str]) -> frozenset[str]:
    """Drop common features, and word pairs built on a common word ("recent files" when nearly
    every request says "files"), so the rare word decides the family on its own."""
    if not common:
        return feats
    return frozenset(f for f in feats if f not in common
                     and not (" " in f and any(w in common for w in f.split(" "))))


def task_features(raw: frozenset[str], common: frozenset[str]) -> frozenset[str]:
    """What grouping and routing both compare: a request's features minus the common ones.

    When one task is nearly all of someone's history, all of its words are "common". Taking them
    away would leave nothing, so such a request keeps its own words (tools aside). Grouping and
    routing must use this same rule, or a family could be built that requests can never match.
    """
    return without_common(raw, common) or frozenset(f for f in raw if not f.startswith(("T:", "S:")))


def features(user_text: str, tools: Sequence[str] = (), skill: str | None = None) -> frozenset[str]:
    """Unigrams + bigrams of the normalized request, plus the first tools used and any skill."""
    # Single letters are leftovers of contractions ("what's" gives "s"), not words.
    words = [w for w in _WORD.findall(normalize(user_text))
             if w not in _STOP and w not in _PHRASING and (len(w) > 1 or w.startswith("<"))]
    # A placeholder on its own ("<n>", "<ext>") is an argument, not a task: "the 5 largest" and
    # "the 5 newest" share the number, not the job. Word pairs carry order ("largest files"),
    # but a pair with a placeholder in it only says where the argument went. Both are left out.
    feats: set[str] = {w for w in words if not w.startswith("<")}
    feats.update(f"{a} {b}" for a, b in zip(words, words[1:]) if "<" not in a and "<" not in b)
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


# MinHash prototypes (cheap, stable similarity at route time)

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


# Clustering

@dataclass
class Family:
    id: str
    members: list[str] = field(default_factory=list)       # turn ids, in time order
    exemplars: list[frozenset[str]] = field(default_factory=list)
    tool_counts: dict[str, int] = field(default_factory=dict)
    term_counts: dict[str, int] = field(default_factory=dict)

    def similarity(self, feats: frozenset[str]) -> float:
        return max((jaccard(feats, e) for e in self.exemplars), default=0.0)

    def add(self, turn_id: str, feats: frozenset[str], tools: Sequence[str], max_exemplars: int,
            name_feats: frozenset[str] | None = None) -> None:
        self.members.append(turn_id)
        # Keep exemplars diverse: only add one that is not near-identical to an existing one.
        if len(self.exemplars) < max_exemplars and all(jaccard(feats, e) < 0.9 for e in self.exemplars):
            self.exemplars.append(feats)
        for t in set(tools):
            self.tool_counts[t] = self.tool_counts.get(t, 0) + 1
        # Names come from the full wording ("recent files"), so they read well even though
        # matching ignores the common word "files".
        for f in (name_feats if name_feats is not None else feats):
            if not f.startswith(("T:", "S:")) and "<" not in f:
                self.term_counts[f] = self.term_counts.get(f, 0) + 1

    def label(self, k: int = 2) -> str:
        """Name from the most frequent distinctive terms, bigrams preferred, filler skipped."""
        terms = sorted(self.term_counts.items(), key=lambda kv: (-kv[1] - (0.5 if " " in kv[0] else 0), kv[0]))
        picked: list[str] = []
        for term, _ in terms:
            if any(w in _LABEL_FILLER for w in term.split()):
                continue
            if any(term in p or p in term for p in picked):
                continue
            if " " not in term:
                # "recent files" reads better than "recent" when people often say it that way.
                pairs = [(c, p) for p, c in self.term_counts.items() if " " in p and term in p.split(" ")
                         and not any(w in _LABEL_FILLER for w in p.split(" "))]
                if pairs:
                    c, p = max(pairs)
                    if c >= 0.4 * self.term_counts[term]:
                        term = p
            if any(term in p or p in term or set(term.split()) == set(p.split()) for p in picked):
                continue   # "files largest" adds nothing to "largest files"
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


def cluster(items: Sequence[Item], *, threshold: float = 0.35, max_exemplars: int = 8,
            common: frozenset[str] | None = None, merge_at: float | None = None) -> list[Family]:
    """Deterministic leader clustering in input (time) order, then a merge pass.

    Each item joins the most similar family above ``threshold`` or starts a new one. Family ids
    are content hashes of their first member, so they are stable across re-runs. Features in
    ``common`` (see :func:`common_features`) are ignored.
    """
    all_feats = [features(it.user_text, it.tools, it.skill) for it in items]
    if common is None:
        common = common_features(all_feats)
    return _group([it.turn_id for it in items], all_feats, [list(it.tools) for it in items], common,
                  threshold=threshold, max_exemplars=max_exemplars, merge_at=merge_at)


def _group(ids: Sequence[str], all_feats: Sequence[frozenset[str]], tools: Sequence[Sequence[str]],
           common: frozenset[str], *, threshold: float, max_exemplars: int,
           merge_at: float | None) -> list[Family]:
    families: list[Family] = []
    for tid, raw, used in zip(ids, all_feats, tools):
        feats = task_features(raw, common)
        if not feats:
            continue
        best, best_sim = None, 0.0
        for fam in families:
            s = fam.similarity(feats)
            if s > best_sim:
                best, best_sim = fam, s
        if best is None or best_sim < threshold:
            fid = "f" + hashlib.sha1(tid.encode()).hexdigest()[:8]
            best = Family(id=fid)
            families.append(best)
        name_feats = frozenset(f for f in raw if f not in common or " " in f)
        best.add(tid, feats, list(used), max_exemplars, name_feats=name_feats)

    # Merge pass: families whose exemplars overlap strongly are the same task phrased differently.
    merged = True
    while merged:
        merged = False
        for i in range(len(families)):
            for j in range(i + 1, len(families)):
                a, b = families[i], families[j]
                link = max((a.similarity(e) for e in b.exemplars), default=0.0)
                if link >= (threshold + 0.15 if merge_at is None else merge_at):
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


def _text_only(fs: frozenset[str]) -> frozenset[str]:
    return frozenset(f for f in fs if not f.startswith(("T:", "S:")))


def match(feats: frozenset[str], families: Sequence[Family], *, text_only: bool = False) -> RouteMatch:
    """Best family for a new request, with the margin over the runner-up (ambiguity guard).

    ``text_only`` compares words only, for requests whose tool use isn't known yet.
    """
    if text_only:
        feats = _text_only(feats)
        sims = [(max((jaccard(feats, _text_only(e)) for e in f.exemplars), default=0.0), f.id) for f in families]
    else:
        sims = [(f.similarity(feats), f.id) for f in families]
    scored = sorted(sims, reverse=True)
    if not scored:
        return RouteMatch(None, 0.0, 0.0)
    best_sim, best_id = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    return RouteMatch(best_id, best_sim, best_sim - second)


# ============================================================== where a request acts

_HOME_DIRS = ("desktop", "documents", "downloads", "music", "pictures", "videos", "appdata", "onedrive")
_WIN_PATH = re.compile(r"\b([a-z]):[\\/]+((?:[^\\/\s\"']+[\\/]*){0,3})", re.I)
_NIX_PATH = re.compile(r"(?:~|\$home|/[a-z]/users/[^/\s]+|/home/[^/\s]+|/users/[^/\s]+)[\\/]+([^\\/\s\"'?,]+)", re.I)
_HOME_WORD = re.compile(r"\b(" + "|".join(_HOME_DIRS) + r")\b", re.I)


def place(text: str) -> str:
    """The folder tree a request acts on: ``videos``, ``appdata``, ``d:/data``, or "" if none.

    Used to hold out whole folder trees and drives, not just whole sessions, so the test asks
    whether a student learned the job or only learned my usual paths.
    """
    t = text or ""
    m = _WIN_PATH.search(t)
    if m:
        drive = m.group(1).lower()
        rest = [p.strip(".?!,") for p in re.split(r"[\\/]+", m.group(2)) if p.strip(".?!,")]
        if drive == "c" and rest and rest[0].lower() == "users":
            return rest[2].lower() if len(rest) >= 3 else "home"
        return f"{drive}:/" + (rest[0].lower() if rest else "")
    m = _NIX_PATH.search(t)
    if m:
        return m.group(1).strip(".?!,").lower()
    m = _HOME_WORD.search(t)
    return m.group(1).lower() if m else ""
