import json
import os
import re
from typing import List, Dict

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
    update_session_state,
)


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

        self.timeout = 300  # увеличен до 5 минут

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
            # для красоты можно вернуть красивое имя, но оставим model
            return model

        if provider == "kimi":
            for name, mid in self.KIMI_MODELS:
                if mid == model:
                    return name
            return model

        return model or "Unknown"

    def model_options(self) -> List[Dict]:
        models = []

        if self.groq_key:
            models.extend(self._groq_models())

        if self.openrouter_key:
            models.extend(self._openrouter_models())

        if self.kimi_key:
            models.extend(self._kimi_models())

        if not models:
            models = [
                {"name": "Llama 3.1 8B", "provider": "groq", "model": "llama-3.1-8b-instant"},
            ]
        return models

    def _groq_models(self):
        out = []
        try:
            r = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={
                    "Authorization": f"Bearer {self.groq_key}",
                    "Content-Type": "application/json",
                },
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            ids = sorted({item["id"] for item in data.get("data", []) if item.get("id")})
        except Exception:
            ids = list(self.GROQ_LABELS.keys())

        for model_id in ids:
            out.append({
                "name": self.GROQ_LABELS.get(model_id, model_id),
                "provider": "groq",
                "model": model_id,
            })
        return out

    def _openrouter_models(self):
        """
        Возвращает ВСЕ модели, которые доступны через OpenRouter API.
        Без фильтрации по избранному.
        """
        out = []
        try:
            r = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
            r.raise_for_status()
            data = r.json()
            for item in data.get("data", []):
                model_id = item.get("id")
                if model_id:
                    # Берём красивое имя, если есть
                    display_name = item.get("name") or model_id
                    out.append({
                        "name": display_name,
                        "provider": "openrouter",
                        "model": model_id,
                    })
            # Сортируем по имени для удобства
            out.sort(key=lambda x: x["name"].lower())
        except Exception:
            # Fallback: если API не отвечает, показываем хотя бы избранные модели
            for name, model_id in self.OPENROUTER_FAVORITES:
                out.append({
                    "name": name,
                    "provider": "openrouter",
                    "model": model_id,
                })
        return out

    def _kimi_models(self):
        return [
            {"name": name, "provider": "kimi", "model": model_id}
            for name, model_id in self.KIMI_MODELS
        ]

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

    def _extract_json(self, text: str) -> dict:
        raw = (text or "").strip()
        if not raw:
            raise ValueError("Модель вернула пустой ответ")

        try:
            return json.loads(raw)
        except Exception:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.S | re.I)
        if fenced:
            return json.loads(fenced.group(1).strip())

        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end + 1])

        raise ValueError("Модель вернула невалидный JSON")

    def _extract_html(self, text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            raise ValueError("Пустой HTML")

        fenced = re.search(r"```(?:html)?\s*(.*?)```", raw, re.S | re.I)
        if fenced:
            raw = fenced.group(1).strip()

        doc = re.search(r"(?is)(<!doctype html>.*?</html>|<html\b.*?</html>)", raw)
        html = doc.group(1).strip() if doc else raw.strip()

        lower = html.lower()
        if "<html" not in lower and "<!doctype" not in lower:
            # fallback: оборачиваем в базовую страницу
            html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Agent Page</title></head>
<body>
{html}
</body>
</html>"""
        return html

    def _system_prompt(self) -> str:
        return """
Ты — один и тот же агент для браузера и терминала.

У тебя два режима:

1) chat
Если пользователь просто разговаривает, шутит, спрашивает мнение, обсуждает жизнь
или задаёт вопрос не про изменение сайта — просто ответь текстом и НЕ меняй HTML.

2) edit
Если пользователь просит изменить сайт: фон, цвета, шрифты, кнопки, блоки, layout,
контент, секции, карточки, отступы, заголовки — измени текущий HTML
и верни ПОЛНЫЙ новый HTML целиком.

Возвращай СТРОГО JSON и ничего кроме JSON:
{
  "mode": "chat" или "edit",
  "assistant": "короткий ответ пользователю",
  "html": "полный HTML целиком или пустая строка"
}

ПРАВИЛА:
- никаких пояснений вне JSON;
- если mode = chat, поле html должно быть пустой строкой;
- если mode = edit, поле html должно содержать полный HTML;
- отвечай по-русски;
- стиль по умолчанию минималистичный;
- если пользователь не попросил другое, сохраняй спокойную PI-like эстетику;
- если пользователь не попросил другое, базовый фон держи в духе rgb(20, 40, 76).
        """.strip()

    def _call_provider(self, provider: str, model: str, messages: list) -> str:
        provider = (provider or "").strip().lower()

        if provider == "groq":
            if not self.groq_key:
                raise RuntimeError("GROQ_API_KEY not set")
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.groq_key}",
                "Content-Type": "application/json",
            }

        elif provider == "openrouter":
            if not self.openrouter_key:
                raise RuntimeError("OPENROUTER_API_KEY not set")
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv("PI_PUBLIC_URL", "http://localhost"),
                "X-Title": "Pi Browser Agent",
            }

        elif provider == "kimi":
            if not self.kimi_key:
                raise RuntimeError("KIMI_API_KEY not set")
            url = "https://api.moonshot.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.kimi_key}",
                "Content-Type": "application/json",
            }

        else:
            raise RuntimeError(f"Unknown provider: {provider}")

        payload = {
            "model": model,
            "messages": messages,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

        if not resp.ok:
            try:
                err = resp.json()
                raise RuntimeError(f"{provider} error {resp.status_code}: {json.dumps(err, ensure_ascii=False)}")
            except Exception:
                raise RuntimeError(f"{provider} error {resp.status_code}: {resp.text[:800]}")

        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""

    def chat(self, session_name: str, message: str, provider=None, model=None):
        session_name = (session_name or "default").strip() or "default"
        clean_message = (message or "").strip()

        if not clean_message:
            return {"success": False, "error": "Empty message"}

        lower = clean_message.lower()

        if lower in {"/undo", "undo", "откати назад", "откатить назад", "верни назад", "верни как было"}:
            return self.undo(session_name)

        if lower in {"/clear", "/clear-history", "очисти историю", "сотри историю"}:
            return self.clear(session_name)

        session = self.ensure_session(session_name, provider=provider, model=model)
        provider = session["provider"]
        model = session["model"]

        add_message(session["id"], "user", clean_message, None, None)

        history = get_history(session["id"], limit=30)
        compact_history = []
        for item in history[-20:]:
            role = "assistant" if item["role"] == "assistant" else "user"
            content = str(item["message"] or "").strip()
            if content:
                compact_history.append({"role": role, "content": content[:1500]})

        current_html = read_session_html(session["id"])

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "session": session["name"],
                        "provider": provider,
                        "model": model,
                        "history": compact_history,
                        "current_html": current_html,
                        "user_message": clean_message,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        raw_reply = self._call_provider(provider, model, messages)
        data = self._extract_json(raw_reply)

        mode = str(data.get("mode") or "chat").strip().lower()
        assistant_text = str(data.get("assistant") or "").strip() or "Done."
        html = str(data.get("html") or "").strip()

        changed = False
        if mode == "edit":
            new_html = self._extract_html(html)
            save_session_html(session["id"], new_html)
            changed = True

        add_message(session["id"], "assistant", assistant_text, provider, model)

        return {
            "success": True,
            "reply": assistant_text,
            "changed": changed,
            "provider": provider,
            "model": model,
            "label": self.label_for(provider, model),
        }


agent = Agent()

# Глобальный оверрайд модели полностью удалён.
# Если нужно принудительно задать модель для всех сессий,
# используйте переменные окружения или удалите/отредактируйте файл global_model.json
# и раскомментируйте код ниже (НО ЭТО НЕ РЕКОМЕНДУЕТСЯ, ТАК КАК ЛОМАЕТ ВЫБОР В UI).

# import json as _gmo_json
# from pathlib import Path as _gmo_Path
# _GMO_FILE = _gmo_Path("/opt/my-agent/global_model.json")
# def _gmo_load():
#     try:
#         data = _gmo_json.loads(_GMO_FILE.read_text(encoding="utf-8"))
#         provider = str(data.get("provider") or "").strip()
#         model = str(data.get("model") or "").strip()
#         if provider and model:
#             return provider, model
#     except Exception:
#         pass
#     return None, None
# if not getattr(Agent.chat, "__name__", "") == "_agent_chat_with_global_model":
#     _AGENT_CHAT_ORIG = Agent.chat
#     def _agent_chat_with_global_model(self, session_name: str, message: str, provider=None, model=None):
#         gp, gm = _gmo_load()
#         if gp and gm:
#             provider, model = gp, gm
#         return _AGENT_CHAT_ORIG(self, session_name, message, provider=provider, model=model)
#     Agent.chat = _agent_chat_with_global_model
