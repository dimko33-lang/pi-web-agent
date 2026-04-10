import json
import os
import re
import subprocess
import time
import pwd
from typing import List, Dict
from datetime import datetime
from pathlib import Path

import requests

from db import (
    add_message,
    clear_history,
    get_history,
    get_or_create_session,
    get_session,
    read_session_html,
    save_session_html,
    undo_last_snapshot,
    redo_last_snapshot,
    update_session_state,
)


class Agent:
    GROQ_LABELS = {
        "llama-3.1-8b-instant": "Llama 3.1 8B",
        "llama-3.3-70b-versatile": "Llama 3.3 70B",
        "qwen-qwq-32b": "Qwen QWQ 32B",
    }

    KIMI_MODELS = [
        ("Kimi K2.5", "kimi-k2.5"),
        ("Kimi K2", "kimi-k2"),
    ]

    def __init__(self):
        self.groq_key = os.getenv("GROQ_API_KEY", "").strip()
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.kimi_key = os.getenv("KIMI_API_KEY", "").strip()

        self.default_provider = os.getenv("PI_DEFAULT_PROVIDER", "groq").strip() or "groq"
        self.default_groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
        self.default_openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o").strip()
        self.default_kimi_model = os.getenv("KIMI_MODEL", "kimi-k2.5").strip()

        self.timeout = 120
        self.roles_dir = Path("/opt/my-agent/roles")
        self.roles_dir.mkdir(exist_ok=True)

    def default_model_for(self, provider: str) -> str:
        if provider == "openrouter":
            return self.default_openrouter_model
        if provider == "kimi":
            return self.default_kimi_model
        return self.default_groq_model

    def label_for(self, provider: str, model: str) -> str:
        provider = (provider or "").strip().lower()
        model = (model or "").strip()
        if provider == "groq":
            return self.GROQ_LABELS.get(model, model)
        return model or "Unknown"

    def model_options(self) -> List[Dict]:
        models = []
        
        # Groq
        if self.groq_key:
            try:
                r = requests.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {self.groq_key}"},
                    timeout=10,
                )
                r.raise_for_status()
                data = r.json()
                groq_ids = sorted({item["id"] for item in data.get("data", []) if item.get("id")})
                for mid in groq_ids:
                    name = self.GROQ_LABELS.get(mid, f"Groq · {mid}")
                    models.append({"name": name, "provider": "groq", "model": mid})
            except Exception:
                for mid, name in self.GROQ_LABELS.items():
                    models.append({"name": f"Groq · {name}", "provider": "groq", "model": mid})
        
        # Kimi
        if self.kimi_key:
            for name, mid in self.KIMI_MODELS:
                models.append({"name": f"Kimi · {name}", "provider": "kimi", "model": mid})
        
        # OpenRouter
        if self.openrouter_key:
            try:
                r = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
                r.raise_for_status()
                or_data = r.json().get("data", [])
                or_models = []
                for item in or_data:
                    mid = item.get("id")
                    if not mid:
                        continue
                    raw_name = item.get("name") or mid
                    clean_name = re.sub(r'^OR:\s*', '', raw_name)
                    is_free = ":free" in mid
                    if is_free:
                        display_name = f"FREE · {clean_name}"
                    else:
                        display_name = clean_name
                    or_models.append({
                        "name": display_name,
                        "provider": "openrouter",
                        "model": mid,
                        "_is_free": is_free,
                        "_id": mid
                    })
                def sort_key(m):
                    if m["_id"] == "openrouter/auto": return (0, "")
                    if m["_id"] == "openrouter/free": return (1, "")
                    if m["_is_free"]: return (2, m["name"])
                    return (3, m["name"])
                or_models.sort(key=sort_key)
                for m in or_models:
                    models.append({"name": m["name"], "provider": m["provider"], "model": m["model"]})
            except Exception:
                pass

        return models or [{"name": "Llama 3.1 8B", "provider": "groq", "model": "llama-3.1-8b-instant"}]

    def ensure_session(self, session_name: str, provider=None, model=None):
        provider = (provider or self.default_provider).strip().lower()
        model = (model or self.default_model_for(provider)).strip()
        session = get_or_create_session(session_name, provider=provider, model=model)
        return update_session_state(session["id"], provider, model)

    def undo(self, session_name: str):
        session = get_or_create_session(session_name)
        ok = undo_last_snapshot(session["id"])
        return {
            "success": True,
            "reply": "↩️ Откатил." if ok else "ℹ️ Нечего откатывать.",
            "changed": bool(ok),
        }

    def redo(self, session_name: str):
        session = get_or_create_session(session_name)
        ok = redo_last_snapshot(session["id"])
        return {
            "success": True,
            "reply": "↪️ Вернул." if ok else "ℹ️ Нечего возвращать.",
            "changed": bool(ok),
        }

    def clear(self, session_name: str):
        session = get_or_create_session(session_name)
        clear_history(session["id"])
        return {"success": True, "reply": "🧹 История очищена.", "changed": False}

    def _execute_command(self, command: str) -> dict:
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=30, cwd="/opt/my-agent"
            )
            output = result.stdout + result.stderr
            return {"output": output or "(no output)", "exit_code": result.returncode}
        except subprocess.TimeoutExpired:
            return {"output": "⏱️ Таймаут 30с", "exit_code": -1}
        except Exception as e:
            return {"output": str(e), "exit_code": -1}

    def _apply_css(self, html: str, css: str) -> str:
        style_tag = f'<style id="agent-style">\n{css}\n</style>'
        html = re.sub(r'<style\s+id="agent-style">.*?</style>', '', html, flags=re.DOTALL)
        return html.replace('</head>', f'{style_tag}\n</head>') if '</head>' in html else html + style_tag

    def _parse_response(self, text: str) -> Dict:
        result = {"mode": "chat", "assistant": text, "command": None, "css": None}
        
        cmd = re.search(r'\[CMD\](.*?)\[/CMD\]', text, re.DOTALL)
        if cmd:
            result["mode"] = "shell"
            result["command"] = cmd.group(1).strip()
            result["assistant"] = re.sub(r'\[CMD\].*?\[/CMD\]', '', text, flags=re.DOTALL).strip()
        
        css = re.search(r'\[CSS\](.*?)\[/CSS\]', text, re.DOTALL)
        if css:
            result["mode"] = "edit_css"
            result["css"] = css.group(1).strip()
            result["assistant"] = re.sub(r'\[CSS\].*?\[/CSS\]', '', text, flags=re.DOTALL).strip()
        
        return result

    def _call_llm(self, provider: str, model: str, messages: list) -> str:
        provider = provider.lower()
        
        if provider == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.groq_key}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": messages, "temperature": 0.9}
        elif provider == "kimi":
            url = "https://api.moonshot.ai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.kimi_key}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": messages}
        elif provider == "openrouter":
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv("PI_PUBLIC_URL", "http://localhost"),
            }
            payload = {"model": model, "messages": messages, "temperature": 0.9}
        else:
            raise RuntimeError(f"Unknown provider: {provider}")

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        if not resp.ok:
            raise RuntimeError(f"API error {resp.status_code}")
        return resp.json()["choices"][0]["message"]["content"] or ""

    def chat(self, session_name: str, message: str, provider=None, model=None):
        session_name = (session_name or "default").strip() or "default"
        clean_message = (message or "").strip()
        if not clean_message:
            return {"success": False, "error": "Пустое сообщение"}

        lower = clean_message.lower()
        if lower in {"/undo", "undo"}:
            return self.undo(session_name)
        if lower in {"/clear", "очисти"}:
            return self.clear(session_name)

        if "прочти файл" in lower or "read file" in lower:
            match = re.search(r'(?:прочти файл|read file)\s+(\S+\.\w+)', lower)
            if match:
                filename = match.group(1)
                role_file = self.roles_dir / filename
                if role_file.exists():
                    content = role_file.read_text(encoding="utf-8")
                    session = self.ensure_session(session_name, provider, model)
                    add_message(session["id"], "assistant", f"📄 *Прочитал {filename}:*\n\n{content}", provider, model)
                    return {
                        "success": True,
                        "reply": f"✅ Загрузил {filename}. Я в теме!",
                        "changed": False,
                    }

        session = self.ensure_session(session_name, provider, model)
        provider = session["provider"]
        model = session["model"]

        add_message(session["id"], "user", clean_message, provider, model)

        history = get_history(session["id"], limit=15)
        messages = []
        for item in history:
            role = item["role"]
            content = str(item["message"] or "")[:1000]
            if content:
                messages.append({"role": role, "content": content})

        try:
            raw_reply = self._call_llm(provider, model, messages)
            data = self._parse_response(raw_reply)
        except Exception as e:
            error_text = f"❌ Ошибка: {e}"
            add_message(session["id"], "assistant", error_text, provider, model)
            return {"success": False, "reply": error_text, "changed": False}

        assistant_text = data["assistant"] or "Готово!"
        changed = False

        if data["mode"] == "shell" and data["command"]:
            exec_result = self._execute_command(data["command"])
            assistant_text = f"{assistant_text}\n\n💻 `{data['command']}`\n```\n{exec_result['output']}\n```"
        elif data["mode"] == "edit_css" and data["css"]:
            current_html = read_session_html(session["id"])
            new_html = self._apply_css(current_html, data["css"])
            save_session_html(session["id"], new_html)
            changed = True
            assistant_text = f"{assistant_text}\n\n🎨 CSS обновлён!"

        add_message(session["id"], "assistant", assistant_text, provider, model)

        return {
            "success": True,
            "reply": assistant_text,
            "changed": changed,
            "provider": provider,
            "model": model,
        }


agent = Agent()

import json as _gmo_json
from pathlib import Path as _gmo_Path

_GMO_FILE = _gmo_Path("/opt/my-agent/global_model.json")

def _gmo_load():
    try:
        data = _gmo_json.loads(_GMO_FILE.read_text(encoding="utf-8"))
        return data.get("provider"), data.get("model")
    except:
        return None, None

_original_chat = Agent.chat

def _patched_chat(self, session_name: str, message: str, provider=None, model=None):
    gp, gm = _gmo_load()
    if gp and gm:
        provider, model = gp, gm
    return _original_chat(self, session_name, message, provider=provider, model=model)

Agent.chat = _patched_chat
