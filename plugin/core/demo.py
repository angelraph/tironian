"""``/tironian demo``: the whole loop in about half a minute, offline, with nothing to set up.

Everything here runs Tironian's real code: history import, grading, task families, the dataset
builder, the training driver (in its dry-run mode), the held-out gate, shadow mode, live routing
and the silent fallback. Only two things are scripted, and the output says so: the history (a
throwaway Hermes session store with 90 disk-usage questions) and the models (a tiny local server
that plays a weak student, then a good one, then a broken one).

It works in its own folder under the plugin's data directory and never touches your real data.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable

FOLDERS = ["Downloads", "Videos", "Documents", "Desktop", "Music", "Pictures", "projects", "tmp", "src"]
TOOLS = [{"type": "function", "function": {"name": "terminal", "description": "run a shell command",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def _command(folder: str) -> str:
    return f"du -sh ~/{folder}/* | sort -h | tail -5"


def _answer(folder: str) -> str:
    return f"The biggest item in {folder} is big.iso (1.2G)."


def _history(path: Path, n: int = 90) -> None:
    """A Hermes ``state.db`` with n sessions of the same routine question, plus a few one-offs."""
    db = sqlite3.connect(path)
    db.executescript("""
      CREATE TABLE sessions(id TEXT PRIMARY KEY, source TEXT NOT NULL, model TEXT, started_at REAL NOT NULL,
        ended_at REAL, end_reason TEXT, estimated_cost_usd REAL, parent_session_id TEXT);
      CREATE TABLE messages(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT,
        tool_call_id TEXT, tool_calls TEXT, tool_name TEXT, timestamp REAL);""")
    now = time.time()
    rows = []
    for i in range(n):
        sid, folder = f"demo{i}", FOLDERS[i % len(FOLDERS)]
        db.execute("INSERT INTO sessions VALUES(?,?,?,?,?,?,?,NULL)", (sid, "cli", "your-model", now - 600 * i, now, "done", 0.02))
        call = [{"id": f"c{i}", "type": "function", "function": {"name": "terminal",
                 "arguments": json.dumps({"command": _command(folder)})}}]
        rows += [(sid, "user", f"what's using disk space in ~/{folder}", None, None, None, now - 600 * i),
                 (sid, "assistant", "", None, json.dumps(call), None, now - 600 * i),
                 (sid, "tool", json.dumps({"output": "1.2G big.iso", "exit_code": 0}), f"c{i}", None, "terminal", now - 600 * i),
                 (sid, "assistant", _answer(folder), None, None, None, now - 600 * i),
                 (sid, "user", "thanks", None, None, None, now - 600 * i),
                 (sid, "assistant", "You're welcome.", None, None, None, now - 600 * i)]
    for j, one_off in enumerate(["write a haiku about autumn", "translate 'good morning' to French",
                                 "what's the capital of Peru"]):
        sid = f"other{j}"
        db.execute("INSERT INTO sessions VALUES(?,?,?,?,?,?,?,NULL)", (sid, "cli", "your-model", now, now, "done", 0.01))
        rows += [(sid, "user", one_off, None, None, None, now), (sid, "assistant", "Here you go.", None, None, None, now)]
    db.executemany("INSERT INTO messages(session_id, role, content, tool_call_id, tool_calls, tool_name, timestamp) "
                   "VALUES(?,?,?,?,?,?,?)", rows)
    db.commit()
    db.close()


class ScriptedModels:
    """A local OpenAI-compatible server that plays the student: weak, good or broken."""

    def __init__(self) -> None:
        self.mode = "weak"
        self.model = "tironian-demo"
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def _send(self, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                self._send({"data": [{"id": outer.model}]})

            def do_POST(self) -> None:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                self._send({"model": outer.model, "usage": {"prompt_tokens": 900, "completion_tokens": 30},
                            "choices": [{"message": {"content": outer.reply(req.get("messages") or [])},
                                         "finish_reason": "stop"}]})

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def reply(self, messages: list[dict[str, Any]]) -> str:
        from .hermes_format import render_target
        text = " ".join(str(m.get("content") or "") for m in messages if m.get("role") == "user")
        folder = (re.findall(r"~/(\w+)", text) or ["Downloads"])[-1]
        if self.mode == "broken":
            return "<tool_call>{not json</tool_call>"
        if self.mode == "weak":
            return "<think>\n</think>\nI'm not sure how to check that."
        last = str(messages[-1].get("content") or "") if messages else ""
        if last.startswith("<tool_response>"):
            return render_target(_answer(folder), [])
        return render_target("", [{"function": {"name": "terminal", "arguments": json.dumps({"command": _command(folder)})}}])

    def close(self) -> None:
        self.server.shutdown()


class _Teacher:
    """Plays your usual model during shadow mode and the live checks."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: dict[str, Any]) -> Any:
        from types import SimpleNamespace as NS
        self.calls += 1
        msgs = request.get("messages") or []
        text = " ".join(str(m.get("content") or "") for m in msgs if m.get("role") == "user")
        folder = (re.findall(r"~/(\w+)", text) or ["Downloads"])[-1]
        call = NS(id="t1", type="function", function=NS(name="terminal", arguments=json.dumps({"command": _command(folder)})))
        msg = NS(role="assistant", content="", tool_calls=[call])
        return NS(choices=[NS(message=msg, finish_reason="tool_calls", index=0)], model="your-model",
                  usage=NS(prompt_tokens=12000, completion_tokens=30, total_tokens=12030))


def run(base: Path, say: Callable[[str], None]) -> dict[str, Any]:
    """Run the demo in ``base/demo`` and narrate it with ``say``. Returns the key numbers."""
    from . import commands, jobs, pipeline
    from .app import App

    started = time.monotonic()
    root = base / "demo"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    if root.exists():   # a previous demo's database still open in this process: use a fresh folder
        root = base / f"demo-{int(time.time())}"
    data = root / "plugin-data" / "tironian"
    data.mkdir(parents=True, exist_ok=True)
    _history(root / "state.db")
    models = ScriptedModels()
    settings = {"student_endpoint": models.url, "routing_enabled": True, "spot_check_rate": 0.0,
                "judge_enabled": False, "min_eval_n": 30, "promotion_mode": "shadow"}
    app = App(data_dir=data, config_get=lambda k, default=None: settings.get(k, default),
              secret=lambda n: None, broadcast=lambda k, p: None)
    app.engine.warm_enabled = False
    app.engine.sync_capture = True   # so the savings at the end count every step of the demo
    out: dict[str, Any] = {}
    try:
        say("Tironian demo. Real Tironian code end to end; the history and the models are scripted.\n")

        say("1. Your history (a throwaway store with 90 routine questions and a few one-offs)")
        pipeline.import_history(app.store, app.state_db, app.redactor)
        fams = pipeline.rebuild_families(app.store, app.settings())
        app.store.upsert("toolsets", {"hash": "demo", "tools_json": json.dumps(TOOLS)}, key="hash")
        say(commands.report_text(app) + "\n")
        fam = max(fams, key=lambda f: f["n_good"])
        if fam["n_good"] < app.settings().min_family_turns:
            say("   No family has enough good examples to train, so the demo stops here.")
            return {"stopped": "no family ready"}

        def train() -> dict[str, Any]:
            prep = app.prepare_training(fam["id"])
            jid = app.start_training(fam["id"], prep["dataset_id"], backend="dry-run")
            # A dry run takes a second or two, but a busy machine can stretch it, so wait up to 3 minutes.
            deadline = time.monotonic() + 180
            st = None
            while time.monotonic() < deadline:
                done = (app.poll_job(jid) or {}).get("status") in jobs.TERMINAL
                st = app.store.one("SELECT * FROM students WHERE job_id=?", (jid,))
                if st is not None or done:
                    break
                time.sleep(0.05)
            if st is None:
                raise RuntimeError(f"the demo's training run didn't finish ({(app.poll_job(jid) or {}).get('message')})")
            models.model = st["model_id"]
            say(f"   trained {st['model_id']} on {prep['n_train']} examples, {prep['n_heldout']} held out "
                f"(dry run: on a real machine this is a GPU job, about $0.15)")
            return dict(st)

        def gate(st: dict[str, Any]) -> dict[str, Any]:
            res = app.evaluate(st["id"])
            for kind, d in (res.get("steps") or {}).items():
                say(f"   {kind} steps: {d['passed']}/{d['n']}, lower bound {d['wilson_lo']:.0%}"
                    f" ({'passes' if d['gate_passed'] else d['gate_reason']})")
            say(("   GATE PASSED: " if res["gate_passed"] else "   GATE REFUSED: ") + res["gate_reason"])
            return res

        say("2. A weak first model, tested on sessions it never saw")
        models.mode = "weak"
        weak = gate(train())
        out["weak_passed"] = weak["gate_passed"]

        say("\n3. A better model, same test")
        models.mode = "good"
        good_st = train()
        good = gate(good_st)
        out["good_passed"] = good["gate_passed"]

        say("\n4. Shadow mode: 20 real-looking requests, your model answers, the student is compared")
        teacher = _Teacher()
        for i in range(20):
            _turn(app, f"what's using disk space in ~/{FOLDERS[i % len(FOLDERS)]}", teacher, f"shadow{i}")
            for _ in range(100):
                if app.store.scalar("SELECT COUNT(*) FROM routes WHERE mode='shadow'", (), 0) > i:
                    break
                time.sleep(0.02)
        ok, why = pipeline.shadow_ready(app.store, fam["id"])
        say(f"   {why}")
        say("   " + commands.set_status(app, "promote", fam["id"]))

        say("\n5. Live: a matching request")
        teacher = _Teacher()
        _turn(app, "what's using disk space in ~/Music", teacher, "live1")
        r = _route(app, "live1")
        say(f"   route: {r['outcome']} by your student in {r['latency_ms']} ms; your model called {teacher.calls} times")
        out["live"] = r["outcome"]

        say("\n6. The student breaks: its answer doesn't parse")
        models.mode = "broken"
        teacher = _Teacher()
        _turn(app, "what's using disk space in ~/Videos", teacher, "live2")
        r = _route(app, "live2")
        say(f"   route: {r['outcome']}; your model answered instead ({teacher.calls} call), nothing shown to you")
        out["fallback"] = r["outcome"]

        sv = pipeline.savings(app.store)
        say(f"\nCloud prompt tokens avoided: {sv['tokens_avoided']:,} (steps the student answered, times the "
            f"average first-step prompt your model was sent for this task)")
        say(f"Done in {time.monotonic() - started:.1f} s. On your own history: /tironian report, then "
            f"`hermes tironian train <family>`.")
        out["seconds"] = round(time.monotonic() - started, 1)
        return out
    finally:
        models.close()


def _route(app: Any, turn: str, wait_s: float = 30.0) -> Any:
    """The route record for a demo turn. Routes are written as the turn finishes, which a busy
    machine can delay, so wait a little rather than read a row that isn't there yet."""
    deadline = time.monotonic() + wait_s
    while True:
        r = app.store.one("SELECT outcome, latency_ms FROM routes WHERE turn_id=? ORDER BY id DESC LIMIT 1", (turn,))
        if r is not None or time.monotonic() > deadline:
            return r or {"outcome": "not recorded", "latency_ms": 0}
        time.sleep(0.05)


def _turn(app: Any, user: str, teacher: Callable[[dict[str, Any]], Any], turn: str) -> Any:
    e = app.engine
    e.pre_llm_call(session_id=f"s-{turn}", turn_id=turn, user_message=user, platform="cli")
    req = {"model": "your-model", "tools": TOOLS,
           "messages": [{"role": "system", "content": "hermes"}, {"role": "user", "content": user}]}
    return e.llm_execution(request=req, next_call=teacher, session_id=f"s-{turn}", turn_id=turn,
                           api_request_id=f"{turn}:0", api_mode="chat_completions", provider="demo", model="your-model")
