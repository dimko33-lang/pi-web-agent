import sqlite3
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

APP_DIR = Path("/opt/my-agent")
DB_PATH = APP_DIR / "agent.db"
SESSIONS_DIR = APP_DIR / "sessions"

def init_db():
    SESSIONS_DIR.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                message TEXT,
                provider TEXT,
                model TEXT,
                timestamp INTEGER NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                html TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS redo_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                html TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
        """)
        conn.commit()

def get_or_create_session(name: str, provider: str = "groq", model: str = "llama-3.1-8b-instant") -> Dict[str, Any]:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT id, name, provider, model, created_at FROM sessions WHERE name = ?", (name,))
        row = cur.fetchone()
        if row:
            return {"id": row[0], "name": row[1], "provider": row[2], "model": row[3], "created_at": row[4]}
        now = int(time.time())
        cur = conn.execute(
            "INSERT INTO sessions (name, provider, model, created_at) VALUES (?, ?, ?, ?)",
            (name, provider, model, now)
        )
        return {"id": cur.lastrowid, "name": name, "provider": provider, "model": model, "created_at": now}

def get_session(name: str) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT id, name, provider, model, created_at FROM sessions WHERE name = ?", (name,))
        row = cur.fetchone()
        if row:
            return {"id": row[0], "name": row[1], "provider": row[2], "model": row[3], "created_at": row[4]}
    return None

def get_session_by_id(session_id: int) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT id, name, provider, model, created_at FROM sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        if row:
            return {"id": row[0], "name": row[1], "provider": row[2], "model": row[3], "created_at": row[4]}
    return None

def list_sessions() -> List[Dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT id, name, provider, model, created_at FROM sessions ORDER BY name")
        return [{"id": r[0], "name": r[1], "provider": r[2], "model": r[3], "created_at": r[4]} for r in cur]

def update_session_state(session_id: int, provider: str, model: str) -> Dict[str, Any]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE sessions SET provider = ?, model = ? WHERE id = ?", (provider, model, session_id))
    return get_session_by_id(session_id)

def session_html_path(session_id: int) -> Path:
    return SESSIONS_DIR / f"session_{session_id}.html"

def read_session_html(session_id: int) -> str:
    path = session_html_path(session_id)
    if path.exists():
        return path.read_text(encoding="utf-8")
    default = APP_DIR / "default_terminal.html"
    if default.exists():
        return default.read_text(encoding="utf-8")
    return "<!doctype html><html><body>Error: no template</body></html>"

def take_snapshot(session_id: int, html: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO snapshots (session_id, html, created_at) VALUES (?, ?, ?)",
            (session_id, html, int(time.time()))
        )

def push_redo_snapshot(session_id: int, html: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO redo_snapshots (session_id, html, created_at) VALUES (?, ?, ?)",
            (session_id, html, int(time.time()))
        )

def clear_redo_snapshots(session_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM redo_snapshots WHERE session_id = ?", (session_id,))

def undo_last_snapshot(session_id: int) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT id, html FROM snapshots WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
            (session_id,)
        )
        row = cur.fetchone()
        if not row:
            return False
        snap_id, old_html = row
        current_html = read_session_html(session_id)
        conn.execute(
            "INSERT INTO redo_snapshots (session_id, html, created_at) VALUES (?, ?, ?)",
            (session_id, current_html, int(time.time()))
        )
        path = session_html_path(session_id)
        path.write_text(old_html, encoding="utf-8")
        conn.execute("DELETE FROM snapshots WHERE id = ?", (snap_id,))
        return True

def redo_last_snapshot(session_id: int) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT id, html FROM redo_snapshots WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
            (session_id,)
        )
        row = cur.fetchone()
        if not row:
            return False
        redo_id, redo_html = row
        current_html = read_session_html(session_id)
        conn.execute(
            "INSERT INTO snapshots (session_id, html, created_at) VALUES (?, ?, ?)",
            (session_id, current_html, int(time.time()))
        )
        path = session_html_path(session_id)
        path.write_text(redo_html, encoding="utf-8")
        conn.execute("DELETE FROM redo_snapshots WHERE id = ?", (redo_id,))
        return True

def save_session_html(session_id: int, new_html: str):
    current_html = read_session_html(session_id)
    if current_html == new_html:
        return
    take_snapshot(session_id, current_html)
    clear_redo_snapshots(session_id)
    path = session_html_path(session_id)
    path.write_text(new_html, encoding="utf-8")

def clear_history(session_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))

def get_history(session_id: int, limit: int = 200) -> List[Dict]:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT id, role, message, provider, model, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp ASC LIMIT ?",
            (session_id, limit)
        )
        return [{"id": r[0], "role": r[1], "message": r[2], "provider": r[3], "model": r[4], "timestamp": r[5]} for r in cur]

def add_message(session_id: int, role: str, message: str, provider: str = None, model: str = None):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, message, provider, model, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, message, provider, model, int(time.time()))
        )

def last_message_id(session_id: int) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT MAX(id) FROM messages WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        return row[0] if row and row[0] else 0

def html_mtime_ns(session_id: int) -> int:
    path = session_html_path(session_id)
    return int(path.stat().st_mtime_ns) if path.exists() else 0
