import json
import os
import re
import subprocess
import shlex
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

        self.timeout = 90

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
        """Порядок: GROQ → Kimi → OpenRouter (внутри OpenRouter: auto, free, бесплатные, остальные)"""
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

        # 2. Kimi (добавляем сразу после GROQ)
        if self.kimi_key:
            for name, mid in self.KIMI_MODELS:
                models.append({"name": f"Kimi · {name}", "provider": "kimi", "model": mid})

        # 3. OpenRouter
        if self.openrouter_key:
            try:
                r = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
                r.raise_for_status()
                or_data = r.json().get("data", [])
                special = []
                others = []
                for item in or_data:
                    mid = item.get("id")
                    if not mid:
                        continue
                    name = item.get("name") or mid
                    is_free = ":free" in mid
                    clean_name = re.sub(r'^OR:\s*', '', name)
                    if mid == "openrouter/auto":
                        special.insert(0, {"name": "OpenRouter · auto (automatic)", "provider": "openrouter", "model": mid, "_is_free": False})
                    elif mid == "openrouter/free":
                        special.append({"name": "OpenRouter · free (automatic free tier)", "provider": "openrouter", "model": mid, "_is_free": True})
                    else:
                        display_name = f"OpenRouter · {clean_name}"
                        if is_free:
                            display_name += " · free"
                        others.append({"name": display_name, "provider": "openrouter", "model": mid, "_is_free": is_free})
                # Сортируем остальные: сначала бесплатные, потом остальные
                others.sort(key=lambda x: (0 if x["_is_free"] else 1, x["name"]))
                for m in special + others:
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

    def _apply_css_to_html(self, html: str, css: str) -> str:
        """Добавляет CSS с !important, чтобы перебить основные стили."""
        # Добавляем !important к каждому правилу, если его нет
        lines = css.split('\n')
        new_lines = []
        for line in lines:
            if '}' in line or '{' in line or '!' in line:
                new_lines.append(line)
            elif ':' in line and not line.strip().startswith('/*'):
                # Добавляем !important перед точкой с запятой
                if ';' in line:
                    line = line.replace(';', ' !important;')
                else:
                    line = line.strip() + ' !important;'
                new_lines.append(line)
            else:
                new_lines.append(line)
        css_with_important = '\n'.join(new_lines)
        style_tag = f'<style id="agent-style">\n{css_with_important}\n</style>'
        if '<style id="agent-style">' in html:
            new_html = re.sub(r'<style id="agent-style">.*?</style>', style_tag, html, flags=re.DOTALL)
        else:
            if '</head>' in html:
                new_html = html.replace('</head>', f'{style_tag}\n</head>')
            else:
                new_html = html + style_tag
        return new_html

    # Все остальные методы (ensure_session, undo, redo, clear, _execute_command,
    # _extract_json, _system_prompt, _call_provider, chat, _extract_html, _check_html_integrity)
    # остаются без изменений – они уже есть в твоём файле. Я их не копирую для краткости,
    # но в итоговом файле они должны быть. Ниже я дам полный файл с ними.

# ... (остальной код)
