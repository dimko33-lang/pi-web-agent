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
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>404</title>
  <style>
    html,body{height:100%;margin:0}
    body{
      display:grid;
      place-items:center;
      background:#111;
      color:#999;
      font:14px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;
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
    # local terminal/CLI/admin on server
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


@app.get("/healthz")
def healthz():
    return jsonify({
        "ok": True,
        "sessions": len(list_sessions()),
        "aliases": len(ALIAS_MAP),
        "admin_alias_exists": bool(ADMIN_ALIAS),
    })


@app.get("/me")
def me():
    forwarded = request.headers.get("X-Forwarded-For", "")
    ip = forwarded.split(",")[0].strip() if forwarded else (request.remote_addr or "127.0.0.1")
    return jsonify({"ip": ip})


@app.get("/models")
def models():
    return jsonify({"models": agent.model_options()})


@app.get("/sessions")
def sessions():
    ctx = resolve_context()
    if not ctx.get("ok") or not ctx.get("admin"):
        return jsonify({"success": False, "error": "forbidden"}), 403

    items = list_sessions()
    return jsonify({
        "success": True,
        "sessions": [s["name"] for s in items],
        "public_aliases": {k: v for k, v in ALIAS_MAP.items() if v in PUBLIC_SESSIONS},
    })


@app.post("/session_switch")
def session_switch():
    ctx = resolve_context()
    if not ctx.get("ok") or not ctx.get("admin"):
        return jsonify({"success": False, "error": "forbidden"}), 403

    payload = request.get_json(silent=True) or {}
    target = str(payload.get("session") or "").strip()
    if not target or not get_session(target):
        return jsonify({"success": False, "error": "session not found"}), 404

    resp = jsonify({"success": True, "target_session": target})
    resp.set_cookie("pi_target_session", target, max_age=86400*30, httponly=False, samesite="Lax")
    return resp


@app.get("/session_info")
def session_info():
    ctx = resolve_context()
    if not ctx.get("ok"):
        return jsonify({"success": False, "error": "Session not found"}), 404

    target = ctx["target_session"]
    session = get_session(target)
    forced = force_model_for_session(target)

    provider = forced[0] if forced[0] else session["provider"]
    model = forced[1] if forced[1] else session["model"]

    return jsonify({
        "success": True,
        "admin_mode": bool(ctx["admin"]),
        "alias": ctx["alias"],
        "target_session": target,
        "public_target": target in PUBLIC_SESSIONS,
        "provider": provider,
        "model": model,
        "label": agent.label_for(provider, model),
    })


@app.post("/session_update")
def session_update():
    ctx = resolve_context()
    if not ctx.get("ok") or not ctx.get("admin"):
        return jsonify({"success": False, "error": "forbidden"}), 403

    target = ctx["target_session"]
    if target in PUBLIC_SESSIONS:
        return jsonify({
            "success": False,
            "error": "Public sessions are fixed to Groq Llama 3.1 8B."
        }), 403

    payload = request.get_json(silent=True) or {}
    provider = str(payload.get("provider") or "").strip()
    model = str(payload.get("model") or "").strip()
    if not provider or not model:
        return jsonify({"success": False, "error": "provider/model required"}), 400

    session = agent.session_update(target, provider, model)
    return jsonify({
        "success": True,
        "session": session["name"],
        "provider": session["provider"],
        "model": session["model"],
        "label": agent.label_for(session["provider"], session["model"]),
    })


@app.post("/history")
def history():
    ctx = resolve_context()
    if not ctx.get("ok"):
        return jsonify({"success": False, "error": "Session not found"}), 404

    session = get_session(ctx["target_session"])
    return jsonify({"history": get_history(session["id"], limit=200)})


@app.post("/clear_history")
def clear_history():
    ctx = resolve_context()
    if not ctx.get("ok"):
        return jsonify({"success": False, "error": "Session not found"}), 404

    result = agent.clear(ctx["target_session"])
    return jsonify(result)


@app.post("/undo")
def undo():
    ctx = resolve_context()
    if not ctx.get("ok"):
        return jsonify({"success": False, "error": "Session not found"}), 404

    result = agent.undo(ctx["target_session"])
    return jsonify(result)


@app.post("/chat")
def chat():
    ctx = resolve_context()
    if not ctx.get("ok"):
        return jsonify({"success": False, "error": "Session not found"}), 404

    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message") or "").strip()
    if not message:
        return jsonify({"success": False, "error": "Пустое сообщение"}), 400

    target = ctx["target_session"]
    forced = force_model_for_session(target)

    if forced[0]:
        provider, model = forced
    else:
        provider = str(payload.get("provider") or "").strip() or None
        model = str(payload.get("specific_model") or payload.get("model") or "").strip() or None

    try:
        result = agent.chat(target, message, provider=provider, model=model)
        code = 200 if result.get("success") else 500
        return jsonify(result), code
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.get("/events")
def events():
    ctx = resolve_context()
    if not ctx.get("ok"):
        return Response("session not found", status=404, mimetype="text/plain")

    session = get_session(ctx["target_session"])
    sid = session["id"]

    def stream():
        last_msg = last_message_id(sid)
        last_html = html_mtime_ns(sid)
        ticks = 0

        while True:
            time.sleep(1)
            ticks += 1

            current_msg = last_message_id(sid)
            current_html = html_mtime_ns(sid)

            if current_msg != last_msg:
                last_msg = current_msg
                yield "event: messages\ndata: update\n\n"

            if current_html != last_html:
                last_html = current_html
                yield "event: reload\ndata: update\n\n"

            if ticks >= 15:
                yield ": ping\n\n"
                ticks = 0

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# BEGIN SAFE_ADMIN_API
import json as _admin_json
import sqlite3 as _admin_sqlite3
from pathlib import Path as _admin_Path

_ADMIN_APP = _admin_Path("/opt/my-agent")
_ADMIN_ALIAS_FILE = _ADMIN_APP / "session_aliases.json"
_ADMIN_GLOBAL_FILE = _ADMIN_APP / "global_model.json"
_ADMIN_DB_FILE = _ADMIN_APP / "agent.db"

def _admin_alias():
    try:
        aliases = _admin_json.loads(_ADMIN_ALIAS_FILE.read_text(encoding="utf-8"))
        return next((k for k, v in aliases.items() if v == "private"), None)
    except Exception:
        return None

def _admin_is_ok():
    hdr = (request.headers.get("X-Admin-Alias") or "").strip()
    raw = request.query_string.decode("utf-8", "ignore").strip()
    cookie = (request.cookies.get("pi_alias") or "").strip()
    local = (request.remote_addr or "") in {"127.0.0.1", "::1"}
    aa = _admin_alias()
    return local or (aa and (hdr == aa or raw == aa or cookie == aa))

def _admin_load_global_model():
    try:
        data = _admin_json.loads(_ADMIN_GLOBAL_FILE.read_text(encoding="utf-8"))
        provider = str(data.get("provider") or "").strip()
        model = str(data.get("model") or "").strip()
        if provider and model:
            return {"provider": provider, "model": model}
    except Exception:
        pass
    return {"provider": "groq", "model": "llama-3.1-8b-instant"}

def _admin_save_global_model(provider: str, model: str):
    data = {"provider": provider, "model": model}
    _ADMIN_GLOBAL_FILE.write_text(_admin_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data

def _admin_db_exec(sql, params=()):
    with _admin_sqlite3.connect(_ADMIN_DB_FILE) as conn:
        conn.execute(sql, params)
        conn.commit()

@app.get("/admin/state")
def admin_state():
    if not _admin_is_ok():
        return jsonify({"success": False, "error": "forbidden"}), 403

    sessions = [s["name"] for s in list_sessions()]
    gm = _admin_load_global_model()

    return jsonify({
        "success": True,
        "admin_alias": _admin_alias(),
        "sessions": sessions,
        "global_model": {
            "provider": gm["provider"],
            "model": gm["model"],
            "label": agent.label_for(gm["provider"], gm["model"]),
        }
    })

@app.post("/admin/set_model")
def admin_set_model():
    if not _admin_is_ok():
        return jsonify({"success": False, "error": "forbidden"}), 403

    payload = request.get_json(silent=True) or {}
    provider = str(payload.get("provider") or "").strip()
    model = str(payload.get("model") or "").strip()

    if not provider or not model:
        return jsonify({"success": False, "error": "provider/model required"}), 400

    gm = _admin_save_global_model(provider, model)
    return jsonify({
        "success": True,
        "provider": gm["provider"],
        "model": gm["model"],
        "label": agent.label_for(gm["provider"], gm["model"]),
    })

@app.post("/admin/history")
def admin_history():
    if not _admin_is_ok():
        return jsonify({"success": False, "error": "forbidden"}), 403

    payload = request.get_json(silent=True) or {}
    session_name = str(payload.get("session") or "").strip()
    if not session_name or not get_session(session_name):
        return jsonify({"success": False, "error": "session not found"}), 404

    return jsonify({"success": True, "history": get_history(session_name, limit=200)})

@app.post("/admin/chat")
def admin_chat():
    if not _admin_is_ok():
        return jsonify({"success": False, "error": "forbidden"}), 403

    payload = request.get_json(silent=True) or {}
    session_name = str(payload.get("session") or "").strip()
    message = str(payload.get("message") or "").strip()

    if not session_name or not get_session(session_name):
        return jsonify({"success": False, "error": "session not found"}), 404
    if not message:
        return jsonify({"success": False, "error": "empty message"}), 400

    result = agent.chat(session_name, message)
    return jsonify(result)

@app.post("/admin/undo")
def admin_undo():
    if not _admin_is_ok():
        return jsonify({"success": False, "error": "forbidden"}), 403

    payload = request.get_json(silent=True) or {}
    session_name = str(payload.get("session") or "").strip()
    if not session_name or not get_session(session_name):
        return jsonify({"success": False, "error": "session not found"}), 404

    return jsonify(agent.undo(session_name))

@app.post("/admin/clear_session")
def admin_clear_session():
    if not _admin_is_ok():
        return jsonify({"success": False, "error": "forbidden"}), 403

    payload = request.get_json(silent=True) or {}
    session_name = str(payload.get("session") or "").strip()
    if not session_name or not get_session(session_name):
        return jsonify({"success": False, "error": "session not found"}), 404

    return jsonify(agent.clear(session_name))

@app.post("/admin/delete_message")
def admin_delete_message():
    if not _admin_is_ok():
        return jsonify({"success": False, "error": "forbidden"}), 403

    payload = request.get_json(silent=True) or {}
    try:
        msg_id = int(payload.get("id"))
    except Exception:
        return jsonify({"success": False, "error": "bad message id"}), 400

    _admin_db_exec("DELETE FROM messages WHERE id = ?", (msg_id,))
    return jsonify({"success": True, "deleted_id": msg_id})

@app.post("/admin/clear_all_histories")
def admin_clear_all_histories():
    if not _admin_is_ok():
        return jsonify({"success": False, "error": "forbidden"}), 403

    _admin_db_exec("DELETE FROM messages")
    return jsonify({"success": True, "cleared": "all_histories"})
# END SAFE_ADMIN_API

# BEGIN PUBLIC_GLOBAL_MODEL_ENDPOINT
import json as _pgm_json
from pathlib import Path as _pgm_Path

_PGM_FILE = _pgm_Path("/opt/my-agent/global_model.json")

def _pgm_load():
    try:
        data = _pgm_json.loads(_PGM_FILE.read_text(encoding="utf-8"))
        provider = str(data.get("provider") or "").strip()
        model = str(data.get("model") or "").strip()
        if provider and model:
            return {"provider": provider, "model": model}
    except Exception:
        pass
    return {"provider": "groq", "model": "llama-3.1-8b-instant"}

@app.get("/global_model")
def global_model():
    gm = _pgm_load()
    return jsonify({
        "success": True,
        "provider": gm["provider"],
        "model": gm["model"],
        "label": agent.label_for(gm["provider"], gm["model"]),
    })
# END PUBLIC_GLOBAL_MODEL_ENDPOINT

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
