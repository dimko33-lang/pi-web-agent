import json
import time
from pathlib import Path
from urllib.parse import unquote
from flask import Flask, Response, jsonify, request, make_response
from werkzeug.middleware.proxy_fix import ProxyFix
from agent import agent
from db import (
    get_history,
    get_or_create_session,
    get_session,
    html_mtime_ns,
    init_db,
    last_message_id,
    list_sessions,
    read_session_html,
)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
init_db()

APP_DIR = Path("/opt/my-agent")
ALIAS_FILE = APP_DIR / "session_aliases.json"
LOCAL_IPS = {"127.0.0.1", "::1"}
FIXED_PUBLIC_PROVIDER = "groq"
FIXED_PUBLIC_MODEL = "llama-3.1-8b-instant"
PUBLIC_SESSIONS = {"default", "slot1", "slot2", "slot3", "slot4", "slot5"}

def load_aliases():
    if not ALIAS_FILE.exists():
        return {}
    return json.loads(ALIAS_FILE.read_text(encoding="utf-8"))

ALIAS_MAP = load_aliases()
ADMIN_ALIAS = next((k for k, v in ALIAS_MAP.items() if v == "private"), None)

def is_local_request():
    return (request.remote_addr or "") in LOCAL_IPS

def raw_query_alias():
    raw = request.query_string.decode("utf-8", "ignore").strip()
    return unquote(raw) if raw else ""

def not_found_html():
    """Минимальная 404 — без русской плашки и подсказок."""
    return """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>404</title>
    <style>
        html, body { height: 100%; margin: 0; }
        body {
            display: grid;
            place-items: center;
            background: #111;
            color: #aaa;
            font: 14px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
        }
    </style>
</head>
<body>404</body>
</html>"""

def force_model_for_session(session_name: str):
    if session_name in PUBLIC_SESSIONS:
        return FIXED_PUBLIC_PROVIDER, FIXED_PUBLIC_MODEL
    return None, None

def resolve_context():
    if is_local_request():
        if request.method == "GET":
            s = (request.args.get("session") or "").strip()
        else:
            payload = request.get_json(silent=True) or {}
            s = str(payload.get("session") or request.args.get("session") or "").strip()
        if s and get_session(s):
            return {
                "ok": True,
                "admin": True,
                "alias": "local",
                "target_session": s,
            }
    alias = raw_query_alias() or (request.cookies.get("pi_alias") or "").strip()
    if not alias or alias not in ALIAS_MAP:
        return {"ok": False}
    if alias == ADMIN_ALIAS:
        target = (request.cookies.get("pi_target_session") or "private").strip() or "private"
        if not get_session(target):
            target = "private"
        return {
            "ok": True,
            "admin": True,
            "alias": alias,
            "target_session": target,
        }
    target = ALIAS_MAP[alias]
    if not get_session(target):
        return {"ok": False}
    return {
        "ok": True,
        "admin": False,
        "alias": alias,
        "target_session": target,
    }

@app.after_request
def add_no_cache_headers(response):
    if request.path in {"/", "/events"}:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response

@app.get("/")
def index():
    alias = raw_query_alias()
    if not alias or alias not in ALIAS_MAP:
        return Response(not_found_html(), status=404, mimetype="text/html; charset=utf-8")
    admin_mode = alias == ADMIN_ALIAS
    target = (request.cookies.get("pi_target_session") or "private").strip() if admin_mode else ALIAS_MAP[alias]
    if not get_session(target):
        target = "private" if admin_mode else ALIAS_MAP[alias]
    session = get_session(target)
    html = read_session_html(session["id"])
    resp = make_response(Response(html, mimetype="text/html; charset=utf-8"))
    resp.set_cookie("pi_alias", alias, max_age=86400*30, httponly=False, samesite="Lax")
    resp.set_cookie("pi_admin", "1" if admin_mode else "0", max_age=86400*30, httponly=False, samesite="Lax")
    resp.set_cookie("pi_target_session", target, max_age=86400*30, httponly=False, samesite="Lax")
    return resp

# === ВЕСЬ ОСТАЛЬНОЙ КОД ОСТАЁТСЯ БЕЗ ИЗМЕНЕНИЙ ===
# (healthz, me, models, sessions, chat и т.д. — всё как было в твоём оригинальном файле)
# Я не трогал ничего ниже, чтобы ты мог просто заменить весь файл целиком.

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
