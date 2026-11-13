"""History â†’ report â†’ dataset â†’ (dry-run) training â†’ evaluation â†’ gate â†’ shadow, end to end."""

import json
import sqlite3
import time
from pathlib import Path


from tironian.core import commands, jobs, pipeline
from tironian.core.hermes_format import render_target

TOOLS = [{"type": "function", "function": {"name": "terminal", "description": "run a shell command",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
FOLDERS = ["Downloads", "Videos", "Documents", "Desktop", "Music", "Pictures", "projects", "tmp"]


def make_state_db(path: Path, n: int = 60) -> None:
    db = sqlite3.connect(path)
    db.executescript("""
      CREATE TABLE sessions(id TEXT PRIMARY KEY, source TEXT NOT NULL, model TEXT, started_at REAL NOT NULL,
        ended_at REAL, end_reason TEXT, estimated_cost_usd REAL, parent_session_id TEXT);
      CREATE TABLE messages(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT,
        tool_call_id TEXT, tool_calls TEXT, tool_name TEXT, timestamp REAL);""")
    now = time.time()
    for i in range(n):
        sid = f"s{i}"
        folder = FOLDERS[i % len(FOLDERS)]
        db.execute("INSERT INTO sessions VALUES(?,?,?,?,?,?,?,NULL)", (sid, "cli", "teacher", now - 3600 * i, now, "done", 0.02))
        call = [{"id": f"c{i}", "type": "function", "function": {"name": "terminal",
                 "arguments": json.dumps({"command": f"du -sh ~/{folder}/* | sort -h | tail -5"})}}]
        rows = [("user", f"what's using disk space in ~/{folder}", None, None, None),
                ("assistant", "", None, json.dumps(call), None),
                ("tool", json.dumps({"output": "1.2G big.iso", "exit_code": 0}), f"c{i}", None, "terminal"),
                ("assistant", f"The biggest item in {folder} is big.iso (1.2G).", None, None, None),
                ("user", "thanks", None, None, None),
                ("assistant", "You're welcome.", None, None, None)]
        for role, content, tcid, calls, tname in rows:
            db.execute("INSERT INTO messages(session_id, role, content, tool_call_id, tool_calls, tool_name, timestamp) "
                       "VALUES(?,?,?,?,?,?,?)", (sid, role, content, tcid, calls, tname, now - 3600 * i))
    db.commit()
    db.close()


def test_end_to_end(app_factory, llama, tmp_path):
    # About 15 held-out steps of each kind: even a perfect student's Wilson lower bound is 80%
    # there, and each kind must now pass on its own, so this pipeline test uses a 75% bar.
    app = app_factory(student_endpoint=llama.url, min_eval_n=5, promotion_threshold=0.75)
    make_state_db(app.state_db)

    # 1. first-minute report from history (read-only import)
    res = pipeline.import_history(app.store, app.state_db, app.redactor)
    assert res["sessions"] == 60 and res["turns"] == 120
    assert pipeline.import_history(app.store, app.state_db, app.redactor)["sessions"] == 0   # idempotent
    fams = pipeline.rebuild_families(app.store, app.settings())
    disk = max(fams, key=lambda f: f["n_turns"])
    assert disk["n_turns"] >= 55 and json.loads(disk["toolset_json"]) == ["terminal"]
    rep = pipeline.report(app.store, app.settings())
    assert rep.routine_share > 0.4 and rep.trainable_families >= 1
    assert "recurring" in commands.report_text(app)

    # 2. a dataset needs real tool schemas, which only live capture sees
    try:
        pipeline.build_dataset(app.store, app.settings(), disk["id"], tmp_path / "ds")
        raise AssertionError("expected missing-schema error")
    except ValueError as exc:
        assert "schema" in str(exc)
    app.store.upsert("toolsets", {"hash": "h1", "tools_json": json.dumps(TOOLS)}, key="hash")
    prep = app.prepare_training(disk["id"])
    assert prep["n_train"] > 0 and prep["n_heldout"] > 0 and prep["enough"]
    ds = app.store.one("SELECT * FROM datasets WHERE id=?", (prep["dataset_id"],))
    held = [json.loads(x) for x in Path(ds["path"], "heldout.jsonl").read_text(encoding="utf-8").splitlines()]
    train_sessions = {r["decision_id"].split(":")[1] for r in
                      map(json.loads, Path(ds["path"], "train.jsonl").read_text(encoding="utf-8").splitlines())}
    held_sessions = {h["decision_id"].split(":")[1] for h in held}
    assert not train_sessions & held_sessions                       # no session leaks into the test set
    assert held[0]["conversations"][0]["from"] == "system" and "<tools>" in held[0]["conversations"][0]["value"]

    # 3. training job (dry-run driver: real process, real status protocol, no GPU)
    jid = app.start_training(disk["id"], prep["dataset_id"], backend="dry-run")
    for _ in range(200):
        row = app.poll_job(jid)
        if row["status"] in jobs.TERMINAL:
            break
        time.sleep(0.1)
    assert row["status"] == "done", row
    student = app.store.one("SELECT * FROM students WHERE job_id=?", (jid,))
    assert student and Path(student["gguf_path"]).exists()

    # 4. evaluation against held-out steps; a student that copies the teacher passes the gate
    llama.model = student["model_id"]
    replies = []
    for h in held:
        t = h["teacher"]
        replies.append(render_target(t.get("content"), t.get("tool_calls")))
    llama.replies = list(replies)   # pop() would empty the list the replay below reuses
    out = app.evaluate(student["id"])
    assert out["n"] == len(held) and out["passed"] == len(held) and out["gate_passed"], out
    assert app.store.scalar("SELECT status FROM families WHERE id=?", (disk["id"],)) == "shadow"

    # 4b. the same test, exported and replayed by someone else, reproduces the result
    from tironian.core import proof
    from tironian.core.student import StudentClient
    doc = proof.export(app.store, app.settings(), student["id"], kinds=())
    assert proof.check_seal(doc) and len(doc["steps"]) == len(held)
    assert all(s["recorded"]["passed"] for s in doc["steps"])
    llama.replies = list(replies)
    res = proof.replay(doc, lambda m: StudentClient(llama.url).complete(llama.model, m))
    assert res["identical"] == res["n"] == len(held) and res["passed"] == len(held) and res["gate_passed"], res
    assert "Request:" in proof.review_markdown(doc)

    # 5. promotion to live needs live shadow agreement first
    msg = commands.set_status(app, "promote", disk["id"])
    assert "not yet" in msg

    # 6. a session the student never saw (recorded after training) joins the held-out test
    train_ids = [json.loads(x)["decision_id"] for x in Path(ds["path"], "train.jsonl").read_text(encoding="utf-8").splitlines()]
    sid = app.store.scalar("SELECT t.session_id FROM decisions d JOIN turns t ON t.id=d.turn_id WHERE d.id=?", (train_ids[0],))
    unseen = [r["id"] for r in app.store.q("SELECT d.id FROM decisions d JOIN turns t ON t.id=d.turn_id "
                                            "WHERE t.session_id=?", (sid,))]
    with app.store.tx() as c:
        c.executemany("DELETE FROM dataset_items WHERE decision_id=?", [(u,) for u in unseen])
    llama.replies = replies + ["<think>\n</think>\nok"] * 10
    out = app.evaluate(student["id"])
    assert out["fresh"] >= 1 and out["n"] == len(held) + out["fresh"]


def test_dead_driver_fails_the_job_and_frees_the_trainer(app_factory, tmp_path):
    """Found live: the driver died mid-export without a final line and the job blocked every new run."""
    app = app_factory()
    jdir = app.data_dir / "jobs" / "job-dead"
    jdir.mkdir(parents=True)
    (jdir / "status.jsonl").write_text(json.dumps({"t": time.time(), "state": "exporting", "progress": 0.82,
                                                   "message": "merging"}) + "\n")
    (jdir / "driver.log").write_text("DeprecationWarning: noise\nGGUF export produced no file\n")
    (jdir / "driver.pid").write_text("999999")  # no such process
    app.store.upsert("jobs", {"id": "job-dead", "kind": "train", "backend": "modal", "status": "exporting",
                              "remote_id": None, "dataset_id": "d", "family_id": "f", "params_json": "{}",
                              "progress": 0.82, "message": "", "cost_est_usd": 0.1, "artifact_path": None,
                              "sha256": None, "error": None, "started_at": time.time(), "ended_at": None})
    row = jobs.reconcile(app.store, app.data_dir, "job-dead")
    assert row["status"] == "failed" and "GGUF export produced no file" in row["error"]
    assert jobs.active_job(app.store) is None


def test_the_demo_runs_end_to_end_offline(app_factory):
    """`/tironian demo`: weak student refused, good one passes, shadow, live, fallback."""
    from tironian.core import demo
    app = app_factory()
    lines = []
    out = demo.run(app.data_dir, lines.append)
    text = "\n".join(lines)
    assert out["weak_passed"] is False and out["good_passed"] is True
    assert out["live"] == "served" and out["fallback"] == "fb_invalid"
    assert "GATE REFUSED" in text and "GATE PASSED" in text and "live." in text
    # It works in its own folder and leaves the real database alone.
    assert app.store.scalar("SELECT COUNT(*) FROM turns", (), 0) == 0


def test_trained_family_keeps_its_id(app_factory):
    app = app_factory()
    make_state_db(app.state_db)
    pipeline.import_history(app.store, app.state_db, app.redactor)
    disk = max(pipeline.rebuild_families(app.store, app.settings()), key=lambda f: f["n_turns"])

    def x(sql, args=()):
        with app.store.tx() as c:
            c.execute(sql, args)

    x("UPDATE families SET student_id='st1' WHERE id=?", (disk["id"],))
    # A bad rebuild lost the members, and one of them now sits alone in a tiny family.
    one = app.store.one("SELECT id FROM turns WHERE family_id=? LIMIT 1", (disk["id"],))["id"]
    x("UPDATE turns SET family_id=NULL")
    x("INSERT INTO families(id, name, n_turns, n_good, status, updated_at) VALUES('ftiny','x',1,0,'candidate',0)")
    x("UPDATE turns SET family_id='ftiny' WHERE id=?", (one,))
    again = max(pipeline.rebuild_families(app.store, app.settings()), key=lambda f: f["n_turns"])
    assert again["id"] == disk["id"] and again["student_id"] == "st1"
    assert app.store.one("SELECT id FROM families WHERE id='ftiny'") is None


def test_forgetting_a_session_takes_its_student_off_duty(app_factory, llama, tmp_path):
    app = app_factory(student_endpoint=llama.url, min_eval_n=5)
    make_state_db(app.state_db)
    pipeline.import_history(app.store, app.state_db, app.redactor)
    disk = max(pipeline.rebuild_families(app.store, app.settings()), key=lambda f: f["n_turns"])
    app.store.upsert("toolsets", {"hash": "h1", "tools_json": json.dumps(TOOLS)}, key="hash")
    prep = app.prepare_training(disk["id"])
    jid = app.start_training(disk["id"], prep["dataset_id"], backend="dry-run")
    for _ in range(200):
        if app.poll_job(jid)["status"] in jobs.TERMINAL:
            break
        time.sleep(0.1)
    student = app.store.one("SELECT * FROM students WHERE job_id=?", (jid,))
    with app.store.tx() as c:
        c.execute("UPDATE families SET status='live', student_id=? WHERE id=?", (student["id"], disk["id"]))
    ds = app.store.one("SELECT * FROM datasets WHERE id=?", (prep["dataset_id"],))
    did = json.loads(Path(ds["path"], "train.jsonl").read_text(encoding="utf-8").splitlines()[0])["decision_id"]
    sid = app.store.scalar("SELECT t.session_id FROM decisions d JOIN turns t ON t.id=d.turn_id WHERE d.id=?", (did,))

    msg = commands.forget_text(app, sid)
    assert "learned from it" in msg
    assert app.store.scalar("SELECT COUNT(*) FROM turns WHERE session_id=?", (sid,)) == 0
    assert app.store.scalar("SELECT status FROM families WHERE id=?", (disk["id"],)) == "evaluated"
    assert app.store.meta_get(f"forgotten_in:{student['id']}") == sid
    # A session the student never trained on leaves it alone.
    held = json.loads(Path(ds["path"], "heldout.jsonl").read_text(encoding="utf-8").splitlines()[0])["decision_id"]
    hsid = app.store.scalar("SELECT t.session_id FROM decisions d JOIN turns t ON t.id=d.turn_id WHERE d.id=?", (held,))
    assert "No trained student learned from it" in commands.forget_text(app, hsid)


def test_a_lost_modal_job_can_be_resumed_without_a_new_run(app_factory, monkeypatch):
    """Found live: a network blip ended a Modal job halfway, though the GPU run carried on."""
    app = app_factory()
    jdir = app.data_dir / "jobs" / "job-x"
    jdir.mkdir(parents=True)
    (jdir / "job.json").write_text(json.dumps({"job_id": "job-x", "base": "q", "name": "n"}), encoding="utf-8")
    (jdir / "status.jsonl").write_text(json.dumps({"t": 1, "state": "failed", "progress": 1.0,
                                                   "message": "ConnectionError: Deadline exceeded",
                                                   "remote_id": "fc-123"}) + "\n", encoding="utf-8")
    app.store.upsert("jobs", {"id": "job-x", "kind": "train", "backend": "modal", "status": "failed",
                              "remote_id": "fc-123", "dataset_id": "d", "family_id": "f", "params_json": "{}",
                              "progress": 1.0, "message": "x", "started_at": 1})
    spawned = []
    monkeypatch.setattr(jobs, "_spawn", lambda *a, **k: spawned.append(a))
    assert app.resume_training("job-x") == "fc-123"
    assert json.loads((jdir / "job.json").read_text())["remote_id"] == "fc-123" and spawned
    assert app.store.scalar("SELECT status FROM jobs WHERE id='job-x'") == "starting"
    # A job that never reached Modal has nothing to reattach to.
    app.store.upsert("jobs", {"id": "job-y", "kind": "train", "backend": "modal", "status": "failed", "remote_id": None,
                              "dataset_id": "d", "family_id": "f", "params_json": "{}", "progress": 1.0,
                              "message": "x", "started_at": 1})
    (app.data_dir / "jobs" / "job-y").mkdir()
    with app.store.tx() as c:
        c.execute("UPDATE jobs SET status='done' WHERE id='job-x'")
    try:
        app.resume_training("job-y")
        raise AssertionError("expected a refusal")
    except ValueError as exc:
        assert "never reached Modal" in str(exc)


def test_driver_treats_network_blips_as_transient():
    import importlib.util
    spec = importlib.util.spec_from_file_location("driver", Path(__file__).resolve().parent.parent / "plugin" / "trainer" / "driver.py")
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)

    class ModalConnErr(Exception):
        pass
    assert d.is_transient(ConnectionError("Deadline exceeded"))
    assert d.is_transient(ModalConnErr("Deadline exceeded"))
    assert not d.is_transient(RuntimeError("GGUF export produced no file"))


def test_bad_student_fails_gate(app_factory, llama, tmp_path):
    app = app_factory(student_endpoint=llama.url, min_eval_n=5)
    make_state_db(app.state_db)
    pipeline.import_history(app.store, app.state_db, app.redactor)
    disk = max(pipeline.rebuild_families(app.store, app.settings()), key=lambda f: f["n_turns"])
    app.store.upsert("toolsets", {"hash": "h1", "tools_json": json.dumps(TOOLS)}, key="hash")
    prep = app.prepare_training(disk["id"])
    jid = app.start_training(disk["id"], prep["dataset_id"], backend="dry-run")
    for _ in range(200):
        if app.poll_job(jid)["status"] in jobs.TERMINAL:
            break
        time.sleep(0.1)
    student = app.store.one("SELECT * FROM students WHERE job_id=?", (jid,))
    llama.model = student["model_id"]
    llama.replies = ["<think>\n</think>\nI don't know."] * 200
    out = app.evaluate(student["id"])
    assert not out["gate_passed"]
    assert app.store.scalar("SELECT status FROM families WHERE id=?", (disk["id"],)) == "evaluated"
    assert "hasn't passed" in commands.set_status(app, "shadow", disk["id"])


def test_a_teacher_step_with_a_masked_home_folder_is_not_trained_on():
    """The student can't know what [REDACTED:user] stands for, so it learns to write the mask."""
    from tironian.core.store import compress
    msgs = [{"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "Count every .png file under Videos, subfolders included."}]

    def row(cmd):
        out = {"content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "terminal", "arguments": json.dumps({"command": cmd})}}]}
        return {"context_z": compress(msgs), "output_json": json.dumps(out)}

    ex, why = pipeline._example(None, row('find "/c/Users/[REDACTED:user]/Videos" -iname "*.png" | wc -l'), TOOLS, 4000)
    assert ex["masked"] and why is None  # kept for the test, left out of training by build_dataset
    ex, why = pipeline._example(None, row('find "$HOME/Videos" -iname "*.png" | wc -l'), TOOLS, 4000)
    assert not ex["masked"]
