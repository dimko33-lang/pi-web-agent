import json
import os
import re
import subprocess
import shlex
import time
from typing import List, Dict, Optional

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

COMMAND_PREFIX = "Ким"


class Agent:
    GROQ_LABELS = {
        "openai/gpt-oss-120b": "GPT-OSS 120B",
        "moonshotai/kimi-k2-instruct": "Kimi K2",
        "moonshotai/kimi-k2-instruct-0905": "Kimi K2-0905",
        "llama-3.3-70b-versatile": "Llama 3.3 70B",
        "deepseek-r1-distill-llama-70b": "DeepSeek R1 70B",
        "llama3-70b-8192": "Llama 3 70B",
        "qwen-qwq-32b": "Qwen QWQ 32B",
        "qwen/qwen3-32b": "Qwen3 32B",
        "mistral-saba-24b": "Mistral Saba 24B",
        "openai/gpt-oss-20b": "GPT-OSS 20B",
        "openai/gpt-oss-safeguard-20b": "GPT-OSS Safe 20B",
        "meta-llama/llama-4-maverick-17b-128e-instruct": "Llama 4 Maverick 17B",
        "meta-llama/llama-4-scout-17b-16e-instruct": "Llama 4 Scout 17B",
        "gemma2-9b-it": "Gemma 2 9B",
        "llama-3.1-8b-instant": "Llama 3.1 8B",
        "llama3-8b-8192": "Llama 3 8B",
        "groq/compound": "Compound",
        "groq/compound-mini": "Compound Mini",
    }

    OPENROUTER_FAVORITES = [
        ("GPT-4o", "openai/gpt-4o"),
        ("GPT-4.1", "openai/gpt-4.1"),
        ("GPT-4.1 Mini", "openai/gpt-4.1-mini"),
        ("Claude Sonnet 4", "anthropic/claude-sonnet-4"),
    ]

    KIMI_MODELS = [
        ("Kimi K2.5", "kimi-k2.5"),
        ("Kimi K2", "kimi-k2"),
        ("Kimi K2 Thinking", "kimi-k2-thinking"),
        ("Kimi K2 Thinking Turbo", "kimi-k2-thinking-turbo"),
    ]

    def __init__(self):
        self.groq_key = os.getenv("GROQ_API_KEY", "").strip()
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.kimi_key = os.getenv("KIMI_API_KEY", "").strip()

        self.default_provider = os.getenv("PI_DEFAULT_PROVIDER", "groq").strip() or "groq"
        self.default_groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
        self.default_openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o").strip()
        self.default_kimi_model = os.getenv("KIMI_MODEL", "kimi-k2.5").strip()

        # Увеличиваем таймаут для медленных моделей
        self.timeout = 120
        self.max_retries = 2

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
        if provider == "openrouter":
            for name, mid in self.OPENROUTER_FAVORITES:
                if mid == model:
                    return name
            return model
        if provider == "kimi":
            for name, mid in self.KIMI_MODELS:
                if mid == model:
                    return name
            return model
        return model or "Unknown"

    def model_options(self) -> List[Dict]:
        """Порядок: GROQ → Kimi → OpenRouter"""
        models = []

        # 1. GROQ
        if self.groq_key:
            try:
                r = requests.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {self.groq_key}", "Content-Type": "application/json"},
                    timeout=10,
                )
                r.raise_for_status()
                data = r.json()
                groq_ids = sorted({item["id"] for item in data.get("data", []) if item.get("id")})
                for mid in groq_ids:
                    name = self.GROQ_LABELS.get(mid, f"Groq · {mid}")
                    models.append({"name": name, "provider": "groq", "model": mid})
            except Exception:
                fallback = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "qwen-qwq-32b"]
                for mid in fallback:
                    name = self.GROQ_LABELS.get(mid, f"Groq · {mid}")
                    models.append({"name": name, "provider": "groq", "model": mid})

        # 2. Kimi
        if self.kimi_key:
            for name, mid in self.KIMI_MODELS:
                models.append({"name": f"Kimi · {name}", "provider": "kimi", "model": mid})

        # 3. OpenRouter
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
                    if mid == "openrouter/auto":
                        display_name = "OpenRouter · auto (automatic)"
                    elif mid == "openrouter/free":
                        display_name = "OpenRouter · free (automatic free tier)"
                    else:
                        if is_free:
                            display_name = f"FREE · OpenRouter · {clean_name}"
                        else:
                            display_name = f"OpenRouter · {clean_name}"
                    or_models.append({
                        "name": display_name,
                        "provider": "openrouter",
                        "model": mid,
                        "_is_free": is_free,
                        "_id": mid
                    })
                def sort_key(m):
                    mid = m["_id"]
                    if mid == "openrouter/auto":
                        return (0, "")
                    if mid == "openrouter/free":
                        return (1, "")
                    if m["_is_free"]:
                        return (2, m["name"])
                    return (3, m["name"])
                or_models.sort(key=sort_key)
                for m in or_models:
                    models.append({"name": m["name"], "provider": m["provider"], "model": m["model"]})
            except Exception as e:
                print(f"OpenRouter API error: {e}")
                models.append({"name": "OpenRouter · auto (automatic)", "provider": "openrouter", "model": "openrouter/auto"})
                models.append({"name": "OpenRouter · free (automatic free tier)", "provider": "openrouter", "model": "openrouter/free"})
                for name, mid in self.OPENROUTER_FAVORITES:
                    models.append({"name": f"OpenRouter · {name}", "provider": "openrouter", "model": mid})

        if not models:
            models = [{"name": "Llama 3.1 8B", "provider": "groq", "model": "llama-3.1-8b-instant"}]

        return models

    def ensure_session(self, session_name: str, provider=None, model=None):
        provider = (provider or self.default_provider).strip().lower()
        model = (model or self.default_model_for(provider)).strip()
        session = get_or_create_session(session_name, provider=provider, model=model)
        session = update_session_state(session["id"], provider, model)
        return session

    def session_update(self, session_name: str, provider: str, model: str):
        session = get_or_create_session(
            session_name,
            provider=provider or self.default_provider,
            model=model or self.default_model_for(provider or self.default_provider),
        )
        return update_session_state(session["id"], provider, model)

    def undo(self, session_name: str):
        session = get_or_create_session(session_name)
        ok = undo_last_snapshot(session["id"])
        return {
            "success": True,
            "reply": "Откатил последнее изменение." if ok else "Откатывать нечего.",
            "changed": bool(ok),
            "provider": session["provider"],
            "model": session["model"],
            "label": self.label_for(session["provider"], session["model"]),
        }

    def redo(self, session_name: str):
        session = get_or_create_session(session_name)
        ok = redo_last_snapshot(session["id"])
        return {
            "success": True,
            "reply": "Вернул изменение вперёд." if ok else "Возвращать нечего.",
            "changed": bool(ok),
            "provider": session["provider"],
            "model": session["model"],
            "label": self.label_for(session["provider"], session["model"]),
        }

    def clear(self, session_name: str):
        session = get_or_create_session(session_name)
        clear_history(session["id"])
        return {
            "success": True,
            "reply": "История очищена.",
            "changed": False,
            "provider": session["provider"],
            "model": session["model"],
            "label": self.label_for(session["provider"], session["model"]),
        }

    def _execute_command(self, command: str) -> dict:
        """Полный доступ — без ограничений, любые shell-команды"""
        try:
            result = subprocess.run(
                shlex.split(command),
                capture_output=True,
                text=True,
                timeout=30,
                cwd="/opt/my-agent"
            )
            output = result.stdout + result.stderr
            if not output:
                output = "(no output)"
            return {"output": output, "exit_code": result.returncode}
        except subprocess.TimeoutExpired:
            return {"output": "Command timed out after 30 seconds", "exit_code": -1}
        except Exception as e:
            return {"output": str(e), "exit_code": -1}

    def _apply_css_to_html(self, html: str, css: str) -> str:
        style_tag = f'<style id="agent-style">\n{css}\n</style>'
        html = re.sub(r'<style\s+id="agent-style">.*?</style>', '', html, flags=re.DOTALL)
        if '</head>' in html:
            new_html = html.replace('</head>', f'{style_tag}\n</head>')
        else:
            new_html = html + style_tag
        if not self._check_html_integrity(new_html):
            return html
        return new_html

    def _safe_parse_json(self, text: str) -> dict:
        """Надёжный парсер JSON с восстановлением после ошибок"""
        raw = (text or "").strip()
        if not raw:
            return {"mode": "chat", "assistant": "Модель вернула пустой ответ"}

        # Попытка 1: прямой парсинг
        try:
            return json.loads(raw)
        except:
            pass

        # Попытка 2: ищем JSON в markdown-блоке
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S | re.I)
        if fenced:
            try:
                return json.loads(fenced.group(1).strip())
            except:
                pass

        # Попытка 3: ищем фигурные скобки
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end+1])
            except:
                pass

        # Если всё сломалось — возвращаем безопасную заглушку
        return {"mode": "chat", "assistant": raw[:500]}

    def _system_prompt(self, simple: bool = False) -> str:
        if simple:
            return """Ты — помощник Ким. Отвечай на русском языке в формате JSON:
{"mode": "chat", "assistant": "твой ответ"}
Если пользователь просит выполнить команду: {"mode": "shell", "assistant": "пояснение", "command": "команда"}
Если просит изменить дизайн: {"mode": "edit_css", "assistant": "пояснение", "css": "CSS код"}

ВАЖНО: Только JSON, без лишнего текста!"""
        
        return f"""Ты — дружелюбный помощник по имени {COMMAND_PREFIX}. Твоя задача — помогать пользователю управлять сервером через shell-команды и менять CSS дизайн страницы.

ПРАВИЛА РАБОТЫ:
1. Если пользователь просто общается (нет команды "{COMMAND_PREFIX}") — отвечай в режиме chat.
2. Если пользователь обратился "{COMMAND_PREFIX}, сделай X" — используй режим shell или edit_css.

ФОРМАТ ОТВЕТА (СТРОГО JSON, БЕЗ КОММЕНТАРИЕВ):
- Разговор: {{"mode": "chat", "assistant": "текст ответа"}}
- Команда: {{"mode": "shell", "assistant": "пояснение", "command": "shell команда"}}
- CSS: {{"mode": "edit_css", "assistant": "пояснение", "css": "CSS код"}}

ПРИМЕРЫ:
Пользователь: "Привет, как дела?"
Ответ: {{"mode": "chat", "assistant": "Привет! Всё отлично, как у тебя?"}}

Пользователь: "{COMMAND_PREFIX}, покажи список файлов"
Ответ: {{"mode": "shell", "assistant": "Показываю файлы", "command": "ls -la"}}

Пользователь: "{COMMAND_PREFIX}, сделай фон чёрным"
Ответ: {{"mode": "edit_css", "assistant": "Меняю фон", "css": "body {{ background: black; }}"}}

ВАЖНО: Отвечай ТОЛЬКО JSON. Никакого текста до или после фигурных скобок!"""

    def _call_provider(self, provider: str, model: str, messages: list, retry_count: int = 0) -> str:
        provider = (provider or "").strip().lower()
        
        if provider == "groq":
            if not self.groq_key:
                raise RuntimeError("GROQ_API_KEY не задан")
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.groq_key}", "Content-Type": "application/json"}
            payload = {
                "model": model, 
                "messages": messages,
                "response_format": {"type": "json_object"},  # Принудительный JSON для Groq
                "temperature": 0.7 if retry_count == 0 else 0.3  # Снижаем температуру при повторе
            }
        elif provider == "openrouter":
            if not self.openrouter_key:
                raise RuntimeError("OPENROUTER_API_KEY не задан")
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv("PI_PUBLIC_URL", "http://localhost"),
                "X-Title": "Pi Browser Agent",
            }
            payload = {
                "model": model, 
                "messages": messages,
                "response_format": {"type": "json_object"},
                "temperature": 0.7 if retry_count == 0 else 0.3
            }
        elif provider == "kimi":
            if not self.kimi_key:
                raise RuntimeError("KIMI_API_KEY не задан")
            url = "https://api.moonshot.ai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.kimi_key}", "Content-Type": "application/json"}
            payload = {
                "model": model, 
                "messages": messages,
                "response_format": {"type": "json_object"},
                "temperature": 0.7 if retry_count == 0 else 0.3
            }
        else:
            raise RuntimeError(f"Неизвестный provider: {provider}")

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            if not resp.ok:
                try:
                    err = resp.json()
                    raise RuntimeError(f"{provider} error {resp.status_code}: {json.dumps(err, ensure_ascii=False)}")
                except:
                    raise RuntimeError(f"{provider} error {resp.status_code}: {resp.text[:800]}")
            data = resp.json()
            return data["choices"][0]["message"]["content"] or ""
        except requests.Timeout:
            if retry_count < self.max_retries:
                time.sleep(2)
                return self._call_provider(provider, model, messages, retry_count + 1)
            raise RuntimeError(f"Таймаут при обращении к {provider}")

    def chat(self, session_name: str, message: str, provider=None, model=None):
        session_name = (session_name or "default").strip() or "default"
        clean_message = (message or "").strip()
        if not clean_message:
            return {"success": False, "error": "Пустое сообщение"}

        # Проверяем, является ли сообщение командой
        lower_msg = clean_message.lower()
        prefix_lower = COMMAND_PREFIX.lower()
        has_command_keyword = (
            lower_msg.startswith(prefix_lower + " ") or 
            lower_msg.startswith(prefix_lower + ",") or
            f" {prefix_lower} " in lower_msg or
            lower_msg.endswith(f" {prefix_lower}")
        )

        # Обработка встроенных команд
        lower = clean_message.lower()
        if lower in {"/undo", "undo", "откати назад", "откатить назад", "верни назад", "верни как было"}:
            return self.undo(session_name)
        if lower in {"/clear", "/clear-history", "очисти историю", "сотри историю"}:
            return self.clear(session_name)

        # Получаем или создаём сессию
        session = self.ensure_session(session_name, provider=provider, model=model)
        provider = session["provider"]
        model = session["model"]

        add_message(session["id"], "user", clean_message, None, None)

        # Получаем историю (только последние 10 сообщений, без HTML)
        history = get_history(session["id"], limit=20)
        compact_history = []
        for item in history[-10:]:
            role = "assistant" if item["role"] == "assistant" else "user"
            content = str(item["message"] or "").strip()
            if content:
                # Обрезаем длинные сообщения
                compact_history.append({"role": role, "content": content[:500]})

        # Формируем сообщения для модели
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": json.dumps({
                "history": compact_history,
                "user_message": clean_message,
                "is_command": has_command_keyword
            }, ensure_ascii=False)}
        ]

        # Пробуем получить ответ от модели (с повторными попытками при ошибке)
        for attempt in range(self.max_retries + 1):
            try:
                raw_reply = self._call_provider(provider, model, messages, attempt)
                data = self._safe_parse_json(raw_reply)
                
                # Проверяем, что данные корректные
                if not isinstance(data, dict):
                    data = {"mode": "chat", "assistant": str(data)}
                break
            except Exception as e:
                if attempt == self.max_retries:
                    # Последняя попытка — возвращаем ошибку
                    error_text = f"Ошибка: {str(e)}"
                    add_message(session["id"], "assistant", error_text, provider, model)
                    return {
                        "success": False,
                        "reply": error_text,
                        "changed": False,
                        "provider": provider,
                        "model": model,
                        "label": self.label_for(provider, model),
                        "error": str(e)
                    }
                # Пробуем с упрощённым промптом
                messages[0]["content"] = self._system_prompt(simple=True)
                time.sleep(1)

        mode = str(data.get("mode") or "chat").strip().lower()
        assistant_text = str(data.get("assistant") or "").strip() or "Готово."
        changed = False

        # Выполняем действие в зависимости от режима
        if mode == "shell" and has_command_keyword:
            command = data.get("command", "").strip()
            if command:
                exec_result = self._execute_command(command)
                assistant_text = f"$ {command}\n{exec_result['output']}"
                
        elif mode == "edit_css" and has_command_keyword:
            css = data.get("css", "").strip()
            if css:
                current_html = read_session_html(session["id"])
                new_html = self._apply_css_to_html(current_html, css)
                if self._check_html_integrity(new_html):
                    save_session_html(session["id"], new_html)
                    changed = True
                    assistant_text = "CSS обновлён."
                else:
                    assistant_text = "Ошибка: нельзя удалять кнопки управления."
            else:
                assistant_text = "Не указан CSS код."

        # Сохраняем ответ
        add_message(session["id"], "assistant", assistant_text, provider, model)
        
        return {
            "success": True,
            "reply": assistant_text,
            "changed": changed,
            "provider": provider,
            "model": model,
            "label": self.label_for(provider, model),
        }

    def _check_html_integrity(self, html: str) -> bool:
        required_ids = [
            "undoBtn", "redoBtn", "clearBtn", "refreshBtn",
            "sessionSelect", "modelCurrent", "messageInput", "sendLabel",
        ]
        for elem_id in required_ids:
            if elem_id not in html:
                return False
        return True


agent = Agent()

# GLOBAL_MODEL_OVERRIDE
import json as _gmo_json
from pathlib import Path as _gmo_Path

_GMO_FILE = _gmo_Path("/opt/my-agent/global_model.json")

def _gmo_load():
    try:
        data = _gmo_json.loads(_GMO_FILE.read_text(encoding="utf-8"))
        provider = str(data.get("provider") or "").strip()
        model = str(data.get("model") or "").strip()
        if provider and model:
            return provider, model
    except Exception:
        pass
    return None, None

_original_chat = Agent.chat

def _patched_chat(self, session_name: str, message: str, provider=None, model=None):
    gp, gm = _gmo_load()
    if gp and gm:
        provider, model = gp, gm
    return _original_chat(self, session_name, message, provider=provider, model=model)

Agent.chat = _patched_chat
