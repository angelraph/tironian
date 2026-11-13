"""`hermes tironian proof` and `verify`: a held-out test someone else can rerun."""

from __future__ import annotations

import json

import pytest

from tironian.core import proof
from tironian.core.student import StudentReply

TOOLS = [{"type": "function", "function": {"name": "terminal", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def call(cmd: str) -> str:
    return '<tool_call>\n{"name": "terminal", "arguments": {"command": "%s"}}\n</tool_call>' % cmd


def step(n: int, out: str, teacher_cmd: str, *, check: dict | None = None, passed: bool = True) -> dict:
    return {"n": n, "decision_id": f"d{n}", "kind": "first", "place": "music", "user_request": "count mp3s",
            "messages": [{"role": "user", "content": f"count mp3s {n}"}],
            "teacher": {"tool_calls": [{"function": {"name": "terminal",
                                                     "arguments": json.dumps({"command": teacher_cmd})}}]},
            "flip": bool(n % 2),
            "recorded": {"output": out, "passed": passed, "check": check or {"passed": passed, "valid": True}}}


def make_doc(steps: list[dict], min_n: int = 3) -> dict:
    body = {"proof": 1, "tool": "tironian", "student": {"id": "s", "sha256": "x"}, "family": {"id": "f", "name": "count"},
            "eval": {"id": "ev", "at": "", "kinds": ["first"]},
            "gate": {"threshold": 0.5, "min_n": min_n, "validity_floor": 0.9}, "tools": TOOLS, "steps": steps}
    return {**body, "seal": proof.seal(body)}


FIND = "find $HOME/Music -iname '*.mp3' | wc -l"
LS = "ls $HOME/Music/*.mp3 | wc -l"


def answers(texts):
    it = iter(texts)
    return lambda msgs: StudentReply(True, next(it), "s", 5)


def test_seal_catches_any_edit():
    doc = make_doc([step(0, call(FIND), FIND)])
    assert proof.check_seal(doc)
    doc["steps"][0]["recorded"]["passed"] = False
    assert not proof.check_seal(doc)


def test_identical_answers_reproduce_the_gate():
    steps = [step(i, call(FIND), FIND) for i in range(4)]
    res = proof.replay(make_doc(steps), answers([call(FIND)] * 4))
    assert res["identical"] == 4 and res["passed"] == 4 and res["gate_passed"]
    assert "GATE REPRODUCED" in proof.report_text(make_doc(steps), res)


def test_a_judged_pass_stands_only_for_the_same_answer():
    judged = {"passed": True, "valid": True, "action_match": True, "tool_match": True, "args": 0.0,
              "text": "judge-win", "reason": ""}
    steps = [step(0, call(LS), FIND, check=judged), step(1, call(LS), FIND, check=judged)]
    # Step 0 answers the same: the recorded verdict stands. Step 1 answers differently, no judge: a miss.
    res = proof.replay(make_doc(steps, min_n=1), answers([call(LS), call("ls $HOME/Music | wc -l")]))
    assert res["judge_reused"] == 1 and res["unjudged"] == 1
    assert [r["passed"] for r in res["rows"]] == [True, False]
    # With a judge of your own, the different answer is judged again.
    res = proof.replay(make_doc(steps, min_n=1), answers([call(LS), call("ls $HOME/Music | wc -l")]),
                       judge=lambda req, a, b: "tie")
    assert res["rejudged"] == 1 and all(r["passed"] for r in res["rows"])


def test_a_worse_answer_is_reported():
    steps = [step(i, call(FIND), FIND) for i in range(4)]
    res = proof.replay(make_doc(steps), answers([call(FIND)] * 3 + ["I can't help with that."]))
    assert res["identical"] == 3 and res["passed"] == 3
    assert "passed in the test, misses now" in proof.report_text(make_doc(steps), res)


def test_unreachable_server_counts_as_a_miss():
    steps = [step(0, call(FIND), FIND)]
    res = proof.replay(make_doc(steps, min_n=1), lambda m: StudentReply(False, error="unreachable:URLError"))
    assert res["passed"] == 0 and not res["gate_passed"]


def test_fetch_local_and_missing(tmp_path):
    f = tmp_path / "p.json"
    f.write_text("{}")
    assert proof.fetch(str(f), tmp_path / "cache") == f
    with pytest.raises(FileNotFoundError):
        proof.fetch(str(tmp_path / "nope.json"), tmp_path / "cache")
    assert proof.file_sha256(f) == "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"


def test_fetch_keeps_each_release_apart(tmp_path, monkeypatch):
    """Found verifying v7's public release right after v4's: both are called proof.json, and the
    cache handed back v4's proof, so the v7 model 'failed' its sha256 check."""
    import io
    import urllib.request
    served = {"https://x.test/v4/proof.json": b'{"v": 4}', "https://x.test/v7/proof.json": b'{"v": 7}'}

    class Resp(io.BytesIO):
        headers: dict = {}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=60: Resp(served[url]))
    cache = tmp_path / "cache"
    say = lambda s: None
    assert proof.fetch("https://x.test/v4/proof.json", cache, say=say).read_bytes() == b'{"v": 4}'
    assert proof.fetch("https://x.test/v7/proof.json", cache, say=say).read_bytes() == b'{"v": 7}'
    # Same URL serving something new: a file with no hash to check is fetched again.
    served["https://x.test/v7/proof.json"] = b'{"v": 7.1}'
    assert proof.fetch("https://x.test/v7/proof.json", cache, say=say).read_bytes() == b'{"v": 7.1}'
