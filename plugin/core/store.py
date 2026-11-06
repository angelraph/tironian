"""SQLite store for everything Tironian learns, under the plugin's own data directory.

One connection, serialized by a lock: Hermes calls hooks and middleware from the agent thread,
the gateway executor and background workers, and every write here is a few rows, so a lock is
simpler and safer than a writer thread. WAL mode keeps the Desktop page's readers unblocked.

Large payloads (request contexts) are stored zlib-compressed and *already redacted*: the store
never sees raw secrets.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);

CREATE TABLE IF NOT EXISTS sessions(
  id TEXT PRIMARY KEY, platform TEXT, provider TEXT, model TEXT,
  started_at REAL, ended_at REAL, completed INTEGER, interrupted INTEGER DEFAULT 0,
  excluded INTEGER DEFAULT 0, exclude_reason TEXT, source TEXT DEFAULT 'live');

CREATE TABLE IF NOT EXISTS turns(
  id TEXT PRIMARY KEY, session_id TEXT NOT NULL, idx INTEGER, user_text TEXT,
  started_at REAL, ended_at REAL, family_id TEXT, score REAL, label TEXT, grade_json TEXT,
  explicit INTEGER DEFAULT 0, reaction INTEGER DEFAULT 0, approval_denied INTEGER DEFAULT 0,
  api_error INTEGER DEFAULT 0, interrupted INTEGER DEFAULT 0, served_by TEXT DEFAULT 'teacher');
CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, idx);
CREATE INDEX IF NOT EXISTS turns_family ON turns(family_id, label);

CREATE TABLE IF NOT EXISTS decisions(
  id TEXT PRIMARY KEY, turn_id TEXT, session_id TEXT, call_idx INTEGER, api_mode TEXT,
  provider TEXT, model TEXT, tools_hash TEXT, context_z BLOB, output_json TEXT,
  finish_reason TEXT, in_tok INTEGER, out_tok INTEGER, cost_usd REAL, latency_ms INTEGER,
  served_by TEXT, redaction_hits INTEGER DEFAULT 0, created_at REAL);
CREATE INDEX IF NOT EXISTS decisions_turn ON decisions(turn_id, call_idx);

CREATE TABLE IF NOT EXISTS toolsets(hash TEXT PRIMARY KEY, tools_json TEXT);

CREATE TABLE IF NOT EXISTS tool_calls(
  id TEXT PRIMARY KEY, turn_id TEXT, session_id TEXT, name TEXT, status TEXT,
  error_type TEXT, exit_code INTEGER, duration_ms INTEGER, created_at REAL);
CREATE INDEX IF NOT EXISTS tool_calls_turn ON tool_calls(turn_id);

CREATE TABLE IF NOT EXISTS families(
  id TEXT PRIMARY KEY, name TEXT, exemplars_json TEXT, toolset_json TEXT,
  n_turns INTEGER, n_good INTEGER, status TEXT DEFAULT 'candidate', student_id TEXT,
  updated_at REAL);

CREATE TABLE IF NOT EXISTS datasets(
  id TEXT PRIMARY KEY, created_at REAL, family_ids TEXT, seed INTEGER, n_train INTEGER,
  n_heldout INTEGER, path TEXT, sha256 TEXT);
CREATE TABLE IF NOT EXISTS dataset_items(dataset_id TEXT, decision_id TEXT, split TEXT);

CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY, kind TEXT, backend TEXT, status TEXT, remote_id TEXT, dataset_id TEXT,
  family_id TEXT, params_json TEXT, progress REAL DEFAULT 0, message TEXT, cost_est_usd REAL,
  artifact_path TEXT, sha256 TEXT, error TEXT, started_at REAL, ended_at REAL);

CREATE TABLE IF NOT EXISTS students(
  id TEXT PRIMARY KEY, job_id TEXT, family_id TEXT, base_model TEXT, gguf_path TEXT,
  model_id TEXT, sha256 TEXT, size_bytes INTEGER, created_at REAL);

CREATE TABLE IF NOT EXISTS evals(
  id TEXT PRIMARY KEY, student_id TEXT, dataset_id TEXT, family_id TEXT, n INTEGER,
  passed INTEGER, wilson_lo REAL, wilson_hi REAL, validity REAL, tool_acc REAL, args_acc REAL,
  judge_win REAL, p50_ms INTEGER, gate_passed INTEGER, report_json TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS eval_items(
  eval_id TEXT, decision_id TEXT, student_out TEXT, checks_json TEXT, pass INTEGER);

CREATE TABLE IF NOT EXISTS routes(
  id INTEGER PRIMARY KEY AUTOINCREMENT, turn_id TEXT, decision_id TEXT, family_id TEXT,
  mode TEXT, student_id TEXT, outcome TEXT, latency_ms INTEGER, saved_usd REAL, created_at REAL);
CREATE INDEX IF NOT EXISTS routes_family ON routes(family_id, created_at);
"""


def compress(obj: Any) -> bytes:
    return zlib.compress(json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8"), 6)


def decompress(blob: bytes | None) -> Any:
    if not blob:
        return None
    return json.loads(zlib.decompress(blob).decode("utf-8"))


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            row = self._conn.execute("SELECT v FROM meta WHERE k='schema_version'").fetchone()
            if row is None:
                self._conn.execute("INSERT INTO meta(k, v) VALUES('schema_version', ?)", (str(SCHEMA_VERSION),))
            self._conn.commit()

    # -- primitives --------------------------------------------------------------------------
    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def q(self, sql: str, args: tuple | dict = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def one(self, sql: str, args: tuple | dict = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, args).fetchone()

    def scalar(self, sql: str, args: tuple | dict = (), default: Any = None) -> Any:
        row = self.one(sql, args)
        return default if row is None or row[0] is None else row[0]

    def upsert(self, table: str, row: dict[str, Any], key: str = "id") -> None:
        cols = list(row)
        placeholders = ", ".join("?" for _ in cols)
        updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != key)
        sql = f"INSERT INTO {table}({', '.join(cols)}) VALUES({placeholders})"
        sql += f" ON CONFLICT({key}) DO UPDATE SET {updates}" if updates else f" ON CONFLICT({key}) DO NOTHING"
        with self.tx() as c:
            c.execute(sql, tuple(row[k] for k in cols))

    def insert(self, table: str, row: dict[str, Any]) -> int:
        cols = list(row)
        with self.tx() as c:
            cur = c.execute(
                f"INSERT INTO {table}({', '.join(cols)}) VALUES({', '.join('?' for _ in cols)})",
                tuple(row[k] for k in cols))
            return int(cur.lastrowid or 0)

    def meta_get(self, k: str, default: str | None = None) -> str | None:
        v = self.scalar("SELECT v FROM meta WHERE k=?", (k,))
        return default if v is None else v

    def meta_set(self, k: str, v: str) -> None:
        self.upsert("meta", {"k": k, "v": v}, key="k")

    # -- privacy -----------------------------------------------------------------------------
    def purge_session(self, session_id: str) -> int:
        """Remove every trace of a session (called when the user deletes it in Hermes)."""
        with self.tx() as c:
            n = 0
            for table, col in (("decisions", "session_id"), ("tool_calls", "session_id"),
                               ("turns", "session_id"), ("sessions", "id")):
                n += c.execute(f"DELETE FROM {table} WHERE {col}=?", (session_id,)).rowcount
            return n

    def sweep(self, retention_days: int) -> int:
        """Drop captured contexts older than the retention window (aggregates are kept)."""
        if retention_days <= 0:
            return 0
        cutoff = time.time() - retention_days * 86400
        with self.tx() as c:
            return c.execute(
                "UPDATE decisions SET context_z=NULL, output_json=NULL WHERE created_at < ? AND context_z IS NOT NULL",
                (cutoff,)).rowcount

    def purge_all(self) -> None:
        with self.tx() as c:
            for table in ("decisions", "tool_calls", "turns", "sessions", "toolsets", "families",
                          "datasets", "dataset_items", "routes", "eval_items", "evals"):
                c.execute(f"DELETE FROM {table}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()
