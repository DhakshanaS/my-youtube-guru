"""Chat session persistence (SQLite via the standard library).

Stores conversations so the UI can offer a sessions sidebar (create, open,
rename, delete) — the same shape as ChatGPT/Claude. SQLite is a deliberate
choice: it gives real, transactional, on-disk persistence that survives server
restarts with ZERO extra dependencies, which suits a local single-user app and
keeps the deployment story simple.

Schema
------
sessions(id, title, created_at, updated_at)
messages(id, session_id → sessions.id ON DELETE CASCADE, role, content, data, created_at)

`data` holds the JSON for an assistant turn (grounded flag, sources, retrieval,
etc.) so a past answer can be re-rendered exactly — including its cited sources.

Concurrency
-----------
The app is multi-threaded (background jobs, SSE workers), and SQLite
connections can't be shared across threads. So every call opens its own
short-lived connection (cheap for SQLite), and writes are serialised behind a
lock to avoid "database is locked" under the occasional concurrent write.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

from app.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_TITLE = "New chat"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id         TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    data       TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _derive_title(text: str, limit: int = 48) -> str:
    """Name a session after its first user message (collapsed + truncated)."""
    t = " ".join((text or "").split())
    if len(t) > limit:
        t = t[:limit].rstrip() + "…"
    return t or DEFAULT_TITLE


class ChatStore:
    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or get_settings().chat_db_path
        self._wlock = threading.Lock()
        parent = os.path.dirname(self._db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._db_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")  # enforce ON DELETE CASCADE
        return con

    def _init_db(self) -> None:
        with self._wlock:
            con = self._connect()
            try:
                con.executescript(_SCHEMA)
                con.commit()
            finally:
                con.close()
        logger.info("Chat store ready at '%s'", self._db_path)

    # ── sessions ─────────────────────────────────────────────────────────
    def create_session(self, title: str | None = None) -> dict:
        sid = uuid.uuid4().hex
        now = _now()
        title = (title or DEFAULT_TITLE).strip() or DEFAULT_TITLE
        with self._wlock:
            con = self._connect()
            try:
                con.execute(
                    "INSERT INTO sessions(id, title, created_at, updated_at) VALUES (?,?,?,?)",
                    (sid, title, now, now),
                )
                con.commit()
            finally:
                con.close()
        return self.get_session_meta(sid)  # type: ignore[return-value]

    def list_sessions(self) -> list[dict]:
        con = self._connect()
        try:
            rows = con.execute(
                """SELECT s.id, s.title, s.created_at, s.updated_at,
                          (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id)
                              AS message_count
                   FROM sessions s
                   ORDER BY s.updated_at DESC"""
            ).fetchall()
        finally:
            con.close()
        return [dict(r) for r in rows]

    def get_session_meta(self, session_id: str) -> dict | None:
        con = self._connect()
        try:
            row = con.execute(
                """SELECT s.id, s.title, s.created_at, s.updated_at,
                          (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id)
                              AS message_count
                   FROM sessions s WHERE s.id = ?""",
                (session_id,),
            ).fetchone()
        finally:
            con.close()
        return dict(row) if row else None

    def get_session(self, session_id: str) -> dict | None:
        """Full session including its messages in chronological order."""
        meta = self.get_session_meta(session_id)
        if meta is None:
            return None
        con = self._connect()
        try:
            rows = con.execute(
                """SELECT id, session_id, role, content, data, created_at
                   FROM messages WHERE session_id = ? ORDER BY rowid ASC""",
                (session_id,),
            ).fetchall()
        finally:
            con.close()
        messages = []
        for r in rows:
            messages.append({
                "id": r["id"], "session_id": r["session_id"], "role": r["role"],
                "content": r["content"],
                "data": json.loads(r["data"]) if r["data"] else None,
                "created_at": r["created_at"],
            })
        return {**meta, "messages": messages}

    def rename_session(self, session_id: str, title: str) -> dict | None:
        title = (title or "").strip()
        if not title:
            return None
        with self._wlock:
            con = self._connect()
            try:
                cur = con.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                    (title, _now(), session_id),
                )
                con.commit()
                changed = cur.rowcount
            finally:
                con.close()
        return self.get_session_meta(session_id) if changed else None

    def delete_session(self, session_id: str) -> bool:
        with self._wlock:
            con = self._connect()
            try:
                cur = con.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                con.commit()  # messages cascade-delete via the FK
                return cur.rowcount > 0
            finally:
                con.close()

    # ── messages ─────────────────────────────────────────────────────────
    def add_message(self, session_id: str, role: str, content: str,
                    data: dict | None = None) -> dict | None:
        """Append a message; bump the session and auto-title it on first ask."""
        meta = self.get_session_meta(session_id)
        if meta is None:
            return None
        mid = uuid.uuid4().hex
        now = _now()
        data_json = json.dumps(data) if data is not None else None
        # Name the chat after the first user message (while still untitled).
        new_title = _derive_title(content) if (
            role == "user" and meta["title"] == DEFAULT_TITLE
        ) else None

        with self._wlock:
            con = self._connect()
            try:
                con.execute(
                    """INSERT INTO messages(id, session_id, role, content, data, created_at)
                       VALUES (?,?,?,?,?,?)""",
                    (mid, session_id, role, content, data_json, now),
                )
                if new_title:
                    con.execute("UPDATE sessions SET updated_at = ?, title = ? WHERE id = ?",
                                (now, new_title, session_id))
                else:
                    con.execute("UPDATE sessions SET updated_at = ? WHERE id = ?",
                                (now, session_id))
                con.commit()
            finally:
                con.close()
        return {"id": mid, "session_id": session_id, "role": role,
                "content": content, "data": data, "created_at": now}


# Process-wide singleton.
chat_store = ChatStore()
