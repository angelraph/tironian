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


def test_phrasing_alone_does_not_split_a_task():
    """Found in real use: the same three tasks came out as nine families, split by how they were asked."""
    asks = {
        "recent": ["What are the 5 most recently modified files in my Downloads folder? Use the terminal.",
                   "Use the terminal to show the 3 newest files in Desktop.",
                   "Which 10 files in Documents changed most recently? Check with the terminal."],
        "largest": ["What are the 3 largest files in my Videos folder? Use the terminal to check.",
                    "Which 5 files take up the most space in Music? Check with the terminal.",
                    "Use the terminal to list the 10 largest files in Pictures, with their sizes.",
                    "Find the 5 largest files under my Downloads directory using the terminal."],
        "count": ["How many .pdf files are in my Desktop folder? Use the terminal.",
                  "Use the terminal to tell me how many docx files I have in Documents.",
                  "How many files with the .zip extension are in my Downloads directory? Check with the terminal."],
    }
    items, truth = [], {}
    for rep in range(3):  # enough requests for "files", "terminal" to count as common words
        for task, texts in asks.items():
            for k, text in enumerate(texts):
                tid = f"{task}{rep}{k}"
                items.append(families.Item(tid, text, ["terminal"]))
                truth[tid] = task
    fams = families.cluster(items, common=families.common_features(
        [families.features(i.user_text, i.tools) for i in items], min_items=10))
    assert len(fams) == 3
    for f in fams:
        assert len({truth[m] for m in f.members}) == 1

    # When one task is most of the history, its own word ("count") is in most requests too.
    # It still marks that task, so it must not be thrown away like "files" or "terminal".
    for rep in range(3, 12):
        for k, text in enumerate(asks["count"]):
            tid = f"count{rep}{k}"
            items.append(families.Item(tid, text, ["terminal"]))
            truth[tid] = "count"
    fams = families.cluster(items, common=families.common_features(
        [families.features(i.user_text, i.tools) for i in items], min_items=10))
    assert len(fams) == 3
    for f in fams:
        assert len({truth[m] for m in f.members}) == 1


def test_same_task_on_another_folder_is_not_asking_again():
    """Found while testing savings: a second routine request read as a re-ask and demoted a student."""
    again = grade.grade(grade.TurnFacts("How many .pdf files are in Downloads?", [grade.ToolOutcome("terminal")],
                                        "how many pdf files are in my Downloads", True))
    assert "user_repeated" in again.reasons and again.hard_negative
    other = grade.grade(grade.TurnFacts("How many .pdf files are in Downloads?", [grade.ToolOutcome("terminal")],
                                        "How many .zip files are in Desktop?", True))
    assert "user_repeated" not in other.reasons and not other.hard_negative


def test_a_history_that_is_nearly_one_task_still_groups_and_routes():
    """Found by the demo: when one task is 97% of history, all its words were dropped as common,
    so the family vanished, and routing (which strips the same way) could never match it."""
    texts = [f"what's using disk space in ~/{f}" for f in ("Downloads", "Videos", "Music")] * 30
    texts += ["write a haiku about autumn", "translate good morning to French"]
    items = [families.Item(f"t{i}", t, ["terminal"]) for i, t in enumerate(texts)]
    common = families.common_features([families.features(i.user_text, i.tools) for i in items])
    fams = families.cluster(items, common=common)
    disk = max(fams, key=lambda f: len(f.members))
    assert len(disk.members) == 90 and "disk" in disk.label()
    probe = families.task_features(families.features("what's using disk space in ~/Pictures"), common)
    m = families.match(probe, fams, text_only=True)
    assert m.family_id == disk.id and m.similarity > 0.5


def test_acknowledgements_are_not_tasks():
    from tironian.core.grade import ACCEPT
    assert ACCEPT.search("thanks") and ACCEPT.search("Perfect, thank you")
    assert not ACCEPT.search("count the pdf files")


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


def test_shell_escape_inside_tool_call_json_is_kept_literally():
    """Found live: `-iname \\*.docx` written raw inside the JSON string."""
    raw = '<tool_call>\n{"name": "terminal", "arguments": {"command": "find ~/Videos -iname \\*.docx | wc -l"}}\n</tool_call>'
    p = hf.parse_output(raw)
    assert p.ok and p.tool_calls[0]["arguments"]["command"] == "find ~/Videos -iname \\*.docx | wc -l"
    ok = '<tool_call>\n{"name": "terminal", "arguments": {"command": "echo \\"a\\\\nb\\""}}\n</tool_call>'
    assert hf.parse_output(ok).tool_calls[0]["arguments"]["command"] == 'echo "a\\nb"'


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


def test_judge_decides_when_commands_differ():
    from tironian.core.evaluate import check_decision
    tools = [{"type": "function", "function": {"name": "terminal", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
    teacher = [{"function": {"name": "terminal", "arguments": '{"command": "find ~/Pictures -iname \'*.png\' | wc -l"}'}}]
    student = hf.render_target("", [{"function": {"name": "terminal", "arguments": '{"command": "ls ~/Pictures/*.png | wc -l"}'}}])
    seen = []

    def judge(req, a, b):
        seen.append((a, b))
        return "tie"
    c = check_decision(student, "", teacher, tools, user_request="how many png", judge=judge)
    assert c.passed and c.text == "judge-tie" and "ls ~/Pictures" in seen[0][0]
    c = check_decision(student, "", teacher, tools, user_request="x", judge=lambda r, a, b: "b")  # teacher in slot b
    assert not c.passed and c.reason == "judge_preferred_teacher"
    assert not check_decision(student, "", teacher, tools, judge=None).passed


def test_gate_per_step_kind():
    from tironian.core.evaluate import Check, gated_scorecard
    good, bad = Check(True, True), Check(False, False, reason="invalid:x")
    checks = [good] * 30 + [bad] * 20
    kinds = ["first"] * 30 + ["follow"] * 20
    sc = gated_scorecard(checks, [100] * 50, kinds, min_n=30)
    assert sc.gate_passed and sc.serve_steps == ["first"] and not sc.steps["follow"]["gate_passed"]
    # Too few examples of the kind that is good: nothing passes.
    sc = gated_scorecard([good] * 20 + [bad] * 20, [100] * 40, ["first"] * 20 + ["follow"] * 20, min_n=30)
    assert not sc.gate_passed and sc.serve_steps == []
    # Everything passes together: a small but perfect kind may ride along, a weak one may not.
    sc = gated_scorecard([good] * 40 + [good] * 9 + [bad], [1] * 50, ["first"] * 40 + ["follow"] * 10, min_n=30)
    assert sc.gate_passed and sc.serve_steps == ["first"]


def test_shell_commands_compared_by_words_not_characters():
    """Found live: correct student commands failed for quoting and home-folder spelling."""
    from tironian.core.evaluate import args_score
    teacher = {"command": 'find "/c/Users/[REDACTED:user]/Videos" -iname "*.docx" -type f 2>/dev/null | wc -l'}
    assert args_score({"command": "find \"$HOME/Videos\" -type f -iname '*.docx' | wc -l"}, teacher) == 1.0
    # A different approach is still a different answer.
    assert args_score({"command": 'ls -1 "$HOME/Videos"/*.docx | wc -l'}, teacher) == 0.0
    assert args_score({"path": "a"}, {"path": "b"}) == 0.0


def test_one_of_the_teachers_parallel_commands_counts():
    """Found live: the teacher ran two finds at once; the student ran exactly the second."""
    call = lambda cmd: {"function": {"name": "terminal", "arguments": json.dumps({"command": cmd})}}
    teacher = [call("find ~/Temp -maxdepth 1 -type f -iname '*.py' | wc -l"),
               call("find ~/Temp -type f -iname '*.py' | wc -l")]
    student = hf.render_target("", [call("find ~/Temp -type f -iname '*.py' | wc -l")])
    assert evaluate.check_decision(student, "", teacher, TOOLS).passed
    other = hf.render_target("", [call("du -sh ~/Temp")])
    assert evaluate.check_decision(other, "", teacher, TOOLS).reason == "tool_mismatch"


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
