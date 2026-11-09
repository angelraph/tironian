import copy
import hashlib
import json
import time

from plugin.core import passport
from plugin.core.store import Store


def _seed(tmp_path):
    store = Store(tmp_path / "t.db")
    gguf = tmp_path / "s.gguf"
    gguf.write_bytes(b"model bytes")
    sha = hashlib.sha256(b"model bytes").hexdigest()
    now = time.time()
    with store.tx() as c:
        c.execute("INSERT INTO datasets(id, created_at, family_ids, seed, n_train, n_heldout, path, sha256) "
                  "VALUES('ds1', ?, 'fam1', 7, 40, 30, '', 'dsha')", (now,))
        c.execute("INSERT INTO jobs(id, kind, backend, status, remote_id, dataset_id, family_id, params_json, progress, "
                  "cost_est_usd, started_at, ended_at) VALUES('job1', 'train', 'modal', 'done', 'fc-1', 'ds1', 'fam1', "
                  "'{\"gpu\": \"L4\", \"epochs\": 2}', 1.0, 0.14, ?, ?)", (now, now))
        c.execute("INSERT INTO students(id, job_id, family_id, base_model, gguf_path, model_id, sha256, size_bytes, "
                  "created_at) VALUES('stu1', 'job1', 'fam1', 'Qwen3-1.7B', ?, 'stu1', ?, 11, ?)", (str(gguf), sha, now))
        rep = {"steps": {"first": {"passed": 30, "n": 30, "wilson_lo": 0.886, "gate_passed": True}},
               "serve_steps": ["first"], "fresh": 5}
        c.execute("INSERT INTO evals(id, student_id, dataset_id, family_id, n, passed, wilson_lo, wilson_hi, validity, "
                  "gate_passed, report_json, created_at) VALUES('ev1', 'stu1', 'ds1', 'fam1', 30, 30, 0.886, 1.0, 1.0, 1, ?, ?)",
                  (json.dumps(rep), now))
    return store, gguf


def test_passport_seals_and_detects_edits(tmp_path):
    store, _ = _seed(tmp_path)
    doc = passport.build(store, "stu1")
    assert doc["model"]["file_matches"] is True
    assert doc["held_out_tests"][0]["steps"]["first"]["passed"] == 30
    assert passport.verify(doc)[0]
    bad = copy.deepcopy(doc)
    bad["held_out_tests"][0]["steps"]["first"]["passed"] = 29
    assert not passport.verify(bad)[0]
    md = passport.markdown(doc)
    assert "30/30" in md and "matches the file on disk" in md


def test_passport_notices_a_swapped_model_file(tmp_path):
    store, gguf = _seed(tmp_path)
    gguf.write_bytes(b"something else")
    doc = passport.build(store, "stu1")
    assert doc["model"]["file_matches"] is False
    assert "DOES NOT match" in passport.markdown(doc)


def test_slash_passport_picks_the_live_student(tmp_path):
    from types import SimpleNamespace

    from plugin.core import commands
    store, _ = _seed(tmp_path)
    with store.tx() as c:
        c.execute("INSERT INTO families(id, name, status, student_id, updated_at) VALUES('fam1', 'count files', 'live', 'stu1', 1)")
    out = commands.handle(SimpleNamespace(store=store), "passport")
    assert "Model passport: stu1" in out and "count files" in out
