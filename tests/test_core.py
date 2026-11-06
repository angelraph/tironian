import json

import pytest

from tironian.core import evaluate, families, grade, hermes_format as hf
from tironian.core.redact import Redactor

R = Redactor(home_dirs=[r"C:\Users\alice"], usernames=["alice"])


@pytest.mark.parametrize("text,kind", [
    ("key sk-ant-api03-AbCdEf0123456789abcdefGHIJ", "anthropic_key"),
    ("OPENAI_API_KEY=sk-proj-abcDEF1234567890abcdef", "openai_key"),
    ("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", "github_token"),
    ("AKIAIOSFODNN7EXAMPLE", "aws_access_key"),
    ("Authorization: Bearer abcdefghijklmnop1234567890", "bearer"),
    ("postgres://bob:hunter2pass@db:5432/x", "url_credentials"),
    ("token=Zx9Qm2Lp8Rt5Vw3Ny7Bk4Hj6Gf1Ds0Aa", "assignment"),
    ("mail jane.doe@example.com", "email"),
    ("call +1 415 555 0132", "phone"),
    ("card 4111 1111 1111 1111", "card"),
    ("IBAN DE89 3704 0044 0532 0130 00", "iban"),
])
def test_redacts_secrets_and_pii(text, kind):
    r = R.redact_text(text)
    assert r.hits.get(kind), (text, r.text, r.hits)
    assert f"[REDACTED:{kind}]" in r.text


@pytest.mark.parametrize("text", [
    "commit 3f2a9c1e5b7d4f6a8c0e2b4d6f8a0c2e4b6d8f0a",
    "uuid 123e4567-e89b-12d3-a456-426614174000",
    "date 2026-11-05 12:30:00, version 3.11.6, port 8080, pid 123456789",
    "order 1234567890123",
    "def compute_monthly_revenue_for_customer_accounts():",
    "ls -la /usr/local/lib/python3.11/site-packages/numpy/core",
])
def test_leaves_benign_values(text):
    r = R.redact_text(text)
    assert r.text == text and not r.hits


def test_home_and_username():
    r = R.redact_text(r"opened C:\Users\alice\Downloads\x.txt for alice")
    assert r.text == r"opened ~\Downloads\x.txt for [REDACTED:user]"


def test_redact_obj_walks_structures():
    obj, hits = R.redact_obj({"a": ["x sk-ant-api03-AbCdEf0123456789abcdefGHIJ"], "b": 3})
    assert obj["a"][0] == "x [REDACTED:anthropic_key]" and obj["b"] == 3 and hits == {"anthropic_key": 1}


def test_clustering_is_deterministic_and_groups_paraphrases():
    items = [families.Item(f"t{i}", u, ["terminal"]) for i, u in enumerate([
        "what's using disk space in ~/Downloads", "what is using the most disk space in /var/log",
        "show disk space usage for C:\\Users\\me\\Videos", "convert this csv to json: a,b",
        "convert the following csv into json", "what's using disk space in /tmp"])]
    a = families.cluster(items)
    b = families.cluster(items)
    assert [f.members for f in a] == [f.members for f in b]
    disk = next(f for f in a if "t0" in f.members)
    assert {"t0", "t1", "t5"} <= set(disk.members)
    assert "t3" not in disk.members


def test_grade_signals():
    good = grade.grade(grade.TurnFacts("list files", [grade.ToolOutcome("terminal")], "thanks!", True))
    assert good.label == "good"
    corrected = grade.grade(grade.TurnFacts("list files", [grade.ToolOutcome("terminal")], "no, that's wrong", True))
    assert corrected.label != "good" and corrected.hard_negative
    failed = grade.grade(grade.TurnFacts("x", [grade.ToolOutcome("terminal", "ok", 2)], "ok next", True))
    assert failed.label != "good"
    explicit = grade.grade(grade.TurnFacts("x", explicit=-1))
    assert explicit.label == "bad"


TOOLS = [{"type": "function", "function": {"name": "terminal", "description": "run",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def test_format_roundtrip():
    target = hf.render_target("checking", [{"function": {"name": "terminal", "arguments": '{"command": "du -sh ~"}'}}])
    p = hf.parse_output(target)
    assert p.ok and p.content == "checking" and p.tool_calls == [{"name": "terminal", "arguments": {"command": "du -sh ~"}}]


@pytest.mark.parametrize("bad,err", [
    ("<think>\nstill thinking", "unterminated_think"),
    ("<tool_call>{not json}</tool_call>", "tool_call_json"),
    ("hello </tool_call>", "stray_tag"),
    ("<think></think>", "empty"),
])
def test_parse_rejects(bad, err):
    p = hf.parse_output(bad)
    assert not p.ok and p.error == err


def test_compact_context_groups_tools_and_drops_system():
    msgs = [{"role": "system", "content": "long hermes prompt"}, {"role": "user", "content": "disk?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "terminal", "arguments": "{\"command\":\"du\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "terminal", "content": "4G ."}]
    c = hf.compact_messages(msgs, TOOLS)
    roles = [m["role"] for m in c.messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert "<tools>" in c.messages[0]["content"] and "long hermes prompt" not in c.messages[0]["content"]
    assert c.messages[3]["content"].startswith("<tool_response>")
    convo = hf.to_sharegpt(c, "<think>\n</think>\nYou use 4G.")
    assert [x["from"] for x in convo] == ["system", "human", "gpt", "tool", "gpt"]


def test_validate_args():
    schema = TOOLS[0]["function"]["parameters"]
    assert hf.validate_args({"command": "ls"}, schema) is None
    assert hf.validate_args({}, schema) == "missing:command"
    assert hf.validate_args({"command": 3}, schema) == "type:command"


def test_wilson_known_values():
    lo, hi = evaluate.wilson(55, 60)
    assert round(lo, 3) == 0.819 and round(hi, 3) == 0.964
    assert evaluate.wilson(0, 0) == (0.0, 0.0)


def test_check_decision_tool_step():
    teacher = [{"function": {"name": "terminal", "arguments": json.dumps({"command": "du -sh ~/Downloads"})}}]
    ok = evaluate.check_decision(hf.render_target("", [{"function": {"name": "terminal", "arguments": '{"command":"du -sh ~/Downloads/"}'}}]),
                                 "", teacher, TOOLS)
    assert ok.passed
    wrong = evaluate.check_decision("<think>\n</think>\nIt's big.", "", teacher, TOOLS)
    assert not wrong.passed and wrong.reason == "action_mismatch"


def test_gate_requires_lower_bound():
    passing = [evaluate.Check(True, True)] * 58 + [evaluate.Check(False, True, reason="args=0.5")] * 2
    sc = evaluate.scorecard(passing, [100] * 60)
    assert sc.gate_passed
    small = evaluate.scorecard([evaluate.Check(True, True)] * 10, [1] * 10)
    assert not small.gate_passed and "needs" in small.gate_reason
    weak = evaluate.scorecard([evaluate.Check(True, True)] * 45 + [evaluate.Check(False, True)] * 15, [1] * 60)
    assert not weak.gate_passed
