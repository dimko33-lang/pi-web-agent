import os
import re
import shutil
import sqlite3
import tempfile
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / "agent.db"
SESSIONS_DIR = BASE_DIR / "sessions"
DEFAULT_HTML_FILE = BASE_DIR / "default_terminal.html"

LOCK = threading.Lock()


def _conn():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def _dict(row):
    return dict(row) if row else None


def _slug(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-zA-Z0-9а-яА-Я_-]+", "-", value, flags=re.UNICODE)
    value = value.strip("-_")
    return value or "default"


def atomic_write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def init_db():
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    with _conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            slug TEXT NOT NULL UNIQUE,
            provider TEXT NOT NULL DEFAULT 'groq',
            model TEXT NOT NULL DEFAULT 'llama-3.1-8b-instant',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            provider TEXT,
            model TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        );

        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            html_content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        );
        """)

    get_or_create_session("default")


def get_session(name_or_id="default"):
    with _conn() as conn:
        if isinstance(name_or_id, int) or (str(name_or_id).isdigit() and str(name_or_id) == str(int(name_or_id))):
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (int(name_or_id),)).fetchone()
        else:
            row = conn.execute("SELECT * FROM sessions WHERE name = ?", (str(name_or_id),)).fetchone()
        return _dict(row)


def list_sessions():
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM sessions ORDER BY id ASC").fetchall()
        return [_dict(r) for r in rows]


def get_or_create_session(name="default", provider="groq", model="llama-3.1-8b-instant"):
    with LOCK:
        session = get_session(name)
        if session:
            path = session_html_path(session["id"])
            if not path.exists() and DEFAULT_HTML_FILE.exists():
                shutil.copy2(DEFAULT_HTML_FILE, path)
            return session

        with _conn() as conn:
            slug = _slug(name)
            base_slug = slug
            n = 1
            while conn.execute("SELECT 1 FROM sessions WHERE slug = ?", (slug,)).fetchone():
                n += 1
                slug = f"{base_slug}-{n}"

            conn.execute(
                "INSERT INTO sessions(name, slug, provider, model) VALUES (?, ?, ?, ?)",
                (name, slug, provider, model),
            )
            conn.commit()

        session = get_session(name)
        path = session_html_path(session["id"])
        if DEFAULT_HTML_FILE.exists() and not path.exists():
            shutil.copy2(DEFAULT_HTML_FILE, path)
        return session


def update_session_state(name_or_id, provider=None, model=None):
    session = get_session(name_or_id)
    if not session:
        raise ValueError("Session not found")

    provider = provider or session["provider"]
    model = model or session["model"]

    with _conn() as conn:
        conn.execute(
            "UPDATE sessions SET provider = ?, model = ? WHERE id = ?",
            (provider, model, session["id"]),
        )
        conn.commit()

    return get_session(session["id"])


def session_html_path(name_or_id="default") -> Path:
    session = get_session(name_or_id)
    if not session:
        raise ValueError("Session not found")
    return SESSIONS_DIR / f"{session['slug']}.html"


def read_session_html(name_or_id="default") -> str:
    session = get_or_create_session(str(name_or_id)) if not str(name_or_id).isdigit() else get_session(int(name_or_id))
    if not session:
        raise ValueError("Session not found")

    path = session_html_path(session["id"])
    if not path.exists():
        if DEFAULT_HTML_FILE.exists():
            shutil.copy2(DEFAULT_HTML_FILE, path)
        else:
            atomic_write(path, "<!DOCTYPE html><html><body><h1>Missing default_terminal.html</h1></body></html>")
    return path.read_text(encoding="utf-8")


def save_session_html(name_or_id, new_html: str):
    session = get_session(name_or_id)
    if not session:
        raise ValueError("Session not found")

    path = session_html_path(session["id"])
    current_html = path.read_text(encoding="utf-8") if path.exists() else ""

    with _conn() as conn:
        if current_html:
            conn.execute(
                "INSERT INTO snapshots(session_id, html_content) VALUES (?, ?)",
                (session["id"], current_html),
            )
            conn.commit()

    atomic_write(path, new_html)


def undo_last_snapshot(name_or_id="default"):
    session = get_session(name_or_id)
    if not session:
        raise ValueError("Session not found")

    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM snapshots WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (session["id"],),
        ).fetchone()

        if not row:
            return False

        atomic_write(session_html_path(session["id"]), row["html_content"])
        conn.execute("DELETE FROM snapshots WHERE id = ?", (row["id"],))
        conn.commit()
        return True


def add_message(name_or_id, role: str, content: str, provider=None, model=None):
    session = get_session(name_or_id)
    if not session:
        raise ValueError("Session not found")

    with _conn() as conn:
        conn.execute(
            "INSERT INTO messages(session_id, role, content, provider, model) VALUES (?, ?, ?, ?, ?)",
            (session["id"], role, content, provider, model),
        )
        conn.commit()


def get_history(name_or_id="default", limit=200):
    session = get_session(name_or_id)
    if not session:
        raise ValueError("Session not found")

    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT id, role, content AS message, provider, model, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session["id"], limit),
        ).fetchall()

    items = [_dict(r) for r in rows]
    items.reverse()
    return items


def clear_history(name_or_id="default"):
    session = get_session(name_or_id)
    if not session:
        raise ValueError("Session not found")

    with _conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session["id"],))
        conn.commit()


def last_message_id(name_or_id="default") -> int:
    session = get_session(name_or_id)
    if not session:
        return 0

    with _conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(id), 0) AS max_id FROM messages WHERE session_id = ?",
            (session["id"],),
        ).fetchone()
        return int(row["max_id"] or 0)


def html_mtime_ns(name_or_id="default") -> int:
    try:
        return session_html_path(name_or_id).stat().st_mtime_ns
    except Exception:
        return 0
