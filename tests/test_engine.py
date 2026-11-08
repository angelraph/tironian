"""The router's safety contract: never worse than the user's own model."""

import json

import pytest

from tironian.core.hermes_format import render_target

TOOLS = [{"type": "function", "function": {"name": "terminal", "description": "run a shell command",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def teacher_response(content="done", calls=None):
    from tironian.core.student import to_chat_completion
    return to_chat_completion(content, calls or [], model="teacher", in_tok=100, out_tok=20)


class Teacher:
    def __init__(self):
        self.calls = 0

    def __call__(self, request):
        self.calls += 1
        return teacher_response()


def request(user="what's using disk space in ~/Downloads"):
    return {"model": "teacher", "messages": [{"role": "system", "content": "hermes"}, {"role": "user", "content": user}],
            "tools": TOOLS}


def promote(app, llama, status="live"):
    """Seed one promoted family whose exemplar matches disk-usage questions."""
    from tironian.core import families
    feats = families.features("what's using disk space in ~/Downloads")
    app.store.upsert("families", {"id": "fdisk", "name": "disk space", "exemplars_json": json.dumps([sorted(feats)]),
                                  "toolset_json": json.dumps(["terminal"]), "n_turns": 40, "n_good": 40,
                                  "status": status, "student_id": "stu", "updated_at": 0})
    app.store.upsert("students", {"id": "stu", "job_id": None, "family_id": "fdisk", "base_model": "q", "gguf_path": "x",
                                  "model_id": llama.model, "sha256": "", "size_bytes": 1, "created_at": 0})
    app.engine.invalidate()


def run_turn(app, user, teacher, turn="t1", api_mode="chat_completions", req=None):
    e = app.engine
    e.pre_llm_call(session_id="s1", turn_id=turn, user_message=user, platform="cli")
    return e.llm_execution(request=req or request(user), next_call=teacher, session_id="s1", turn_id=turn,
                           api_request_id=f"{turn}:0", api_mode=api_mode, provider="p", model="teacher")


def test_capture_without_routing_calls_teacher_once(app_factory):
    app = app_factory()
    t = Teacher()
    resp = run_turn(app, "hello there", t)
    assert t.calls == 1 and resp.choices[0].message.content == "done"
    assert app.store.scalar("SELECT COUNT(*) FROM decisions") == 1


def test_live_student_serves_valid_tool_call(app_factory, llama):
    app = app_factory(routing_enabled=True, student_endpoint=llama.url)
    promote(app, llama)
    llama.replies = [render_target("", [{"function": {"name": "terminal", "arguments": '{"command": "du -sh ~/Downloads"}'}}])]
    t = Teacher()
    resp = run_turn(app, "what's using disk space in ~/Downloads", t)
    assert t.calls == 0
    call = resp.choices[0].message.tool_calls[0]
    assert call.function.name == "terminal" and json.loads(call.function.arguments) == {"command": "du -sh ~/Downloads"}
    assert app.store.scalar("SELECT outcome FROM routes") == "served"


def test_student_prompt_matches_training_render(llama):
    """Found live: earlier assistant turns reached the student with a think block training had removed."""
    from tironian.core.student import StudentClient
    StudentClient(llama.url).complete("m", [{"role": "user", "content": "hi"},
                                             {"role": "assistant", "content": "<think>\n</think>\n<tool_call>x</tool_call>"}])
    sent = llama.calls[-1]["messages"]
    assert sent[1]["content"] == "<tool_call>x</tool_call>" and sent[0]["content"] == "hi"


def test_ollama_reply_in_reasoning_field_is_served(app_factory, llama):
    """Found live: Ollama put the student's whole answer under "reasoning" and left content empty."""
    app = app_factory(routing_enabled=True, student_endpoint=llama.url)
    promote(app, llama)
    llama.replies = [(render_target("", [{"function": {"name": "terminal", "arguments": '{"command": "ls"}'}}]),)]
    t = Teacher()
    resp = run_turn(app, "what's using disk space in ~/Downloads", t)
    assert t.calls == 0 and resp.choices[0].message.tool_calls[0].function.name == "terminal"


@pytest.mark.parametrize("reply", [
    "<tool_call>{broken</tool_call>",                                   # malformed JSON
    render_target("", [{"function": {"name": "rm_everything", "arguments": "{}"}}]),   # tool not offered
    render_target("", [{"function": {"name": "terminal", "arguments": "{}"}}]),        # schema: missing arg
    500,                                                                # server error
])
def test_any_bad_student_answer_falls_back_silently(app_factory, llama, reply):
    app = app_factory(routing_enabled=True, student_endpoint=llama.url)
    promote(app, llama)
    llama.replies = [reply]
    t = Teacher()
    resp = run_turn(app, "what's using disk space in ~/Downloads", t)
    assert t.calls == 1 and resp.choices[0].message.content == "done"
    assert app.store.scalar("SELECT outcome FROM routes").startswith("fb_")


def test_fallback_is_sticky_for_the_turn(app_factory, llama):
    app = app_factory(routing_enabled=True, student_endpoint=llama.url)
    promote(app, llama)
    llama.replies = ["<tool_call>{broken</tool_call>", render_target("fine", [])]
    t = Teacher()
    run_turn(app, "what's using disk space in ~/Downloads", t)
    app.engine.llm_execution(request=request(), next_call=t, session_id="s1", turn_id="t1", api_request_id="t1:1",
                             api_mode="chat_completions")
    assert t.calls == 2 and len(llama.calls) == 1


def test_failed_tool_hands_turn_back_to_teacher(app_factory, llama):
    """Found live: Hermes blocked a student's command; the student must not carry on regardless."""
    app = app_factory(routing_enabled=True, student_endpoint=llama.url)
    promote(app, llama)
    llama.replies = [render_target("", [{"function": {"name": "terminal", "arguments": '{"command": "du -sh ~"}'}}]),
                     render_target("all done", [])]
    t = Teacher()
    run_turn(app, "what's using disk space in ~/Downloads", t)
    app.engine.post_tool_call(session_id="s1", turn_id="t1", tool_name="terminal", status="blocked", tool_call_id="c1")
    resp = app.engine.llm_execution(request=request(), next_call=t, session_id="s1", turn_id="t1",
                                    api_request_id="t1:1", api_mode="chat_completions")
    assert t.calls == 1 and len(llama.calls) == 1 and resp.choices[0].message.content == "done"


def test_student_serves_only_the_step_kinds_it_passed(app_factory, llama):
    """First command from the student, the answer after the tool result from your model."""
    app = app_factory(routing_enabled=True, student_endpoint=llama.url)
    promote(app, llama)
    app.store.meta_set("serve_steps:fdisk", json.dumps(["first"]))
    app.engine.invalidate()
    llama.replies = [render_target("", [{"function": {"name": "terminal", "arguments": '{"command": "du -sh ~"}'}}])]
    t = Teacher()
    run_turn(app, "what's using disk space in ~/Downloads", t)
    assert t.calls == 0 and len(llama.calls) == 1
    follow = request()
    follow["messages"] += [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": '{"command": "du -sh ~"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "4.0G ~"}]
    resp = app.engine.llm_execution(request=follow, next_call=t, session_id="s1", turn_id="t1",
                                    api_request_id="t1:1", api_mode="chat_completions")
    assert t.calls == 1 and len(llama.calls) == 1 and resp.choices[0].message.content == "done"


def test_unrelated_request_goes_to_teacher(app_factory, llama):
    app = app_factory(routing_enabled=True, student_endpoint=llama.url)
    promote(app, llama)
    t = Teacher()
    run_turn(app, "write a haiku about autumn leaves", t)
    assert t.calls == 1 and not llama.calls


def test_unreachable_student_falls_back(app_factory, llama):
    app = app_factory(routing_enabled=True, student_endpoint="http://127.0.0.1:9/v1", student_timeout_s=1)
    promote(app, llama)
    t = Teacher()
    resp = run_turn(app, "what's using disk space in ~/Downloads", t)
    assert t.calls == 1 and resp.choices[0].message.content == "done"


def test_planning_crash_still_reaches_teacher(app_factory, llama, monkeypatch):
    app = app_factory(routing_enabled=True, student_endpoint=llama.url)
    promote(app, llama)
    monkeypatch.setattr(app.engine, "_plan", lambda kw: 1 / 0)
    t = Teacher()
    run_turn(app, "what's using disk space in ~/Downloads", t)
    assert t.calls == 1


def test_teacher_error_propagates_not_swallowed(app_factory):
    app = app_factory()

    def boom(_):
        raise RuntimeError("provider down")
    with pytest.raises(RuntimeError):
        run_turn(app, "hi", boom)


def test_shadow_mode_never_blocks_and_records(app_factory, llama):
    import time
    app = app_factory(routing_enabled=True, student_endpoint=llama.url)
    promote(app, llama, status="shadow")
    llama.replies = [render_target("done", [])]
    t = Teacher()
    resp = run_turn(app, "what's using disk space in ~/Downloads", t)
    assert t.calls == 1 and resp.choices[0].message.content == "done"
    for _ in range(50):
        if app.store.scalar("SELECT COUNT(*) FROM routes"):
            break
        time.sleep(0.05)
    assert app.store.scalar("SELECT outcome FROM routes") == "shadow_agree"


def test_shadow_uses_the_judge(app_factory, llama):
    """Found live: shadow compared answers word for word, so a different but right answer
    always counted as a disagreement, unlike the held-out test."""
    import time
    app = app_factory(routing_enabled=True, student_endpoint=llama.url)
    promote(app, llama, status="shadow")
    llama.replies = [render_target("Your Downloads folder is mostly one big ISO image.", [])]
    seen = []
    app.engine.judge_factory = lambda: (lambda r, a, b: seen.append((a, b)) or "tie")
    run_turn(app, "what's using disk space in ~/Downloads", Teacher())
    for _ in range(50):
        if app.store.scalar("SELECT COUNT(*) FROM routes"):
            break
        time.sleep(0.05)
    assert seen and app.store.scalar("SELECT outcome FROM routes") == "shadow_agree"


def test_shadow_runs_without_the_answer_switch(app_factory, llama):
    """Found live: shadow only collected samples once students were allowed to answer, so
    following the README (shadow, then promote, then switch on) never got anywhere."""
    import time
    app = app_factory(routing_enabled=False, student_endpoint=llama.url)
    promote(app, llama, status="shadow")
    llama.replies = [render_target("done", [])]
    t = Teacher()
    run_turn(app, "what's using disk space in ~/Downloads", t)
    for _ in range(50):
        if app.store.scalar("SELECT COUNT(*) FROM routes"):
            break
        time.sleep(0.05)
    assert t.calls == 1 and app.store.scalar("SELECT mode FROM routes") == "shadow"


def test_live_family_needs_the_answer_switch(app_factory, llama):
    app = app_factory(routing_enabled=False, student_endpoint=llama.url)
    promote(app, llama)
    t = Teacher()
    run_turn(app, "what's using disk space in ~/Downloads", t)
    assert t.calls == 1 and not llama.calls


def test_session_start_warms_the_student(app_factory, llama):
    """Found live: Ollama unloads an idle student and the cold start blew the routing deadline."""
    import time
    app = app_factory(routing_enabled=True, student_endpoint=llama.url)
    promote(app, llama)
    app.engine.warm_enabled = True
    before = len(llama.calls)
    app.engine.on_session_start(session_id="warm-1", platform="cli")
    app.engine.on_session_start(session_id="warm-2", platform="cli")   # once is enough
    for _ in range(50):
        if len(llama.calls) > before:
            break
        time.sleep(0.05)
    time.sleep(0.1)
    assert len(llama.calls) == before + 1


def test_auto_demote_on_high_fallback_rate(app_factory, llama):
    app = app_factory(routing_enabled=True, student_endpoint=llama.url)
    promote(app, llama)
    for i in range(20):
        llama.replies = ["<tool_call>{broken</tool_call>"]
        run_turn(app, "what's using disk space in ~/Downloads", Teacher(), turn=f"t{i}")
    assert app.store.scalar("SELECT status FROM families WHERE id='fdisk'") == "shadow"


def test_excluded_session_not_captured(app_factory):
    app = app_factory()
    app.engine.exclude_session("s1")
    run_turn(app, "secret project", Teacher())
    assert app.store.scalar("SELECT COUNT(*) FROM decisions") == 0


def test_grading_from_followup_and_delete_purges(app_factory):
    app = app_factory()
    run_turn(app, "list files", Teacher(), turn="t1")
    app.engine.post_llm_call(turn_id="t1")
    run_turn(app, "no, that's wrong", Teacher(), turn="t2")
    assert app.store.scalar("SELECT label FROM turns WHERE id='t1'") != "good"
    app.engine.on_session_delete(session_id="s1")
    assert app.store.scalar("SELECT COUNT(*) FROM turns") == 0


def test_anthropic_capture_normalizes(app_factory):
    app = app_factory()
    req = {"system": "hermes", "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
           "tools": [{"name": "terminal", "input_schema": {"type": "object"}}]}
    resp = {"content": [{"type": "text", "text": "hello"}, {"type": "tool_use", "id": "x", "name": "terminal", "input": {"command": "ls"}}],
            "usage": {"input_tokens": 5, "output_tokens": 3}}
    run_turn(app, "hi", lambda r: resp, api_mode="anthropic_messages", req=req)
    out = json.loads(app.store.scalar("SELECT output_json FROM decisions"))
    assert out["content"] == "hello" and out["tool_calls"][0]["function"]["name"] == "terminal"


def test_capture_runs_in_background_and_never_delays_the_turn(app_factory):
    """Masking and storage happen after the model's answer is handed back."""
    import time as _t
    app = app_factory(async_capture=True)
    big = request("x " * 20000)
    app.engine.pre_llm_call(session_id="s1", turn_id="t1", user_message="hello", platform="cli")
    started = _t.perf_counter()
    for i in range(3):
        app.engine.llm_execution(request=big, next_call=Teacher(), session_id="s1", turn_id="t1",
                                 api_request_id=f"t1:{i}", api_mode="chat_completions")
    assert _t.perf_counter() - started < 1.0  # headroom for a busy machine
    assert app.engine.flush()
    rows = app.store.q("SELECT call_idx FROM decisions ORDER BY call_idx")
    assert [r["call_idx"] for r in rows] == [0, 1, 2]
