"""SQLite-backed session/turn/message store. JSON would have worked but SQLite
makes the eval harness and analytics dead-simple."""
import json
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List


DB_PATH = os.environ.get("DB_PATH", "./data/sessions.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT,
    rolling_summary TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    search_queries TEXT,
    urls_opened TEXT,
    context_snippets TEXT,
    answer TEXT,
    citations TEXT,
    critique TEXT,
    elapsed_s REAL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_turn_session ON turns(session_id);
"""


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _conn()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def create_session(title: str = "New session") -> str:
    sid = uuid.uuid4().hex[:12]
    conn = _conn()
    try:
        conn.execute(
            "INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)",
            (sid, title, _now()),
        )
        conn.commit()
    finally:
        conn.close()
    return sid


def list_sessions() -> List[Dict]:
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT id, title, created_at FROM sessions ORDER BY created_at DESC"
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def rename_session(session_id: str, title: str):
    conn = _conn()
    try:
        conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
        conn.commit()
    finally:
        conn.close()


def add_message(session_id: str, role: str, content: str):
    conn = _conn()
    try:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, _now()),
        )
        conn.commit()
    finally:
        conn.close()


def get_messages(session_id: str) -> List[Dict]:
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def save_turn(session_id: str, turn: Dict):
    conn = _conn()
    try:
        conn.execute(
            """INSERT INTO turns
               (session_id, query, search_queries, urls_opened, context_snippets,
                answer, citations, critique, elapsed_s, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                turn["query"],
                json.dumps(turn.get("search_queries", [])),
                json.dumps(turn.get("urls_opened", [])),
                json.dumps(turn.get("context_snippets", [])),
                turn.get("answer", ""),
                json.dumps(turn.get("citations", [])),
                json.dumps(turn.get("critique", {})),
                turn.get("elapsed_s"),
                _now(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_turns(session_id: str) -> List[Dict]:
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("search_queries", "urls_opened", "context_snippets",
                  "citations", "critique"):
            try:
                d[k] = json.loads(d[k]) if d[k] else ([] if k != "critique" else {})
            except json.JSONDecodeError:
                d[k] = [] if k != "critique" else {}
        out.append(d)
    return out


def get_summary(session_id: str) -> str:
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT rolling_summary FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    finally:
        conn.close()
    return (row["rolling_summary"] if row else "") or ""


def update_summary(session_id: str, summary: str):
    conn = _conn()
    try:
        conn.execute(
            "UPDATE sessions SET rolling_summary = ? WHERE id = ?",
            (summary.strip(), session_id),
        )
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"