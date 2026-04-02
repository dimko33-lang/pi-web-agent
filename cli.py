import os
import sys
import requests

BASE_URL = os.getenv("PI_SERVER_URL", "http://127.0.0.1").rstrip("/")
TIMEOUT = 180

current_session = os.getenv("PI_SESSION", "default")
current_provider = "groq"
current_model = "llama-3.1-8b-instant"


def api(method, path, **kwargs):
    url = BASE_URL + path
    resp = requests.request(method, url, timeout=TIMEOUT, **kwargs)
    data = resp.json()
    if not resp.ok:
        raise RuntimeError(data.get("error") or f"HTTP {resp.status_code}")
    return data


def sync_session_info():
    global current_provider, current_model
    data = api("GET", f"/session_info?session={current_session}")
    current_provider = data.get("provider", current_provider)
    current_model = data.get("model", current_model)


def show_help():
    print("""
Команды:
  /help
  /sessions
  /new ИМЯ
  /use ИМЯ
  /models
  /set PROVIDER MODEL
  /undo
  /clear-history
  /history
  /exit

Примеры:
  /set groq llama-3.1-8b-instant
  /set openrouter openai/gpt-4.1
  /set kimi kimi-k2.5
""".strip())


def show_sessions():
    data = api("GET", "/sessions")
    for s in data.get("sessions", []):
        mark = "*" if s["name"] == current_session else " "
        print(f"{mark} {s['name']}  [{s['provider']} | {s['model']}]")

def show_models():
    data = api("GET", "/models")
    for item in data.get("models", []):
        mark = "*" if (item["provider"] == current_provider and item["model"] == current_model) else " "
        print(f"{mark} {item['provider']:10}  {item['name']}  ->  {item['model']}")

def create_session(name):
    api("POST", "/session_create", json={"session": name, "provider": current_provider, "model": current_model})

def update_session(provider, model):
    api("POST", "/session_update", json={
        "session": current_session,
        "provider": provider,
        "model": model
    })

def show_history():
    data = api("POST", "/history", json={"session": current_session})
    items = data.get("history", [])
    if not items:
        print("(история пуста)")
        return
    for item in items[-30:]:
        role = "assistant" if item["role"] == "assistant" else "user"
        print(f"{role}> {item['message']}")

def send_message(text):
    data = api("POST", "/chat", json={
        "session": current_session,
        "provider": current_provider,
        "specific_model": current_model,
        "message": text
    })
    print("assistant>", data.get("reply", "Готово."))
    if data.get("changed"):
        print("[site updated]")
    sync_session_info()

def main():
    global current_session, current_provider, current_model

    try:
        sync_session_info()
    except Exception as e:
        print("Не удалось подключиться к backend:", e)
        print("Проверь сервис и URL:", BASE_URL)
        sys.exit(1)

    print("PI Terminal CLI")
    print("Server:", BASE_URL)
    print("Session:", current_session)
    print("Напиши /help\n")

    while True:
        try:
            line = input(f"[{current_session} | {current_provider}:{current_model}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line == "/help":
            show_help()
            continue

        if line == "/sessions":
            show_sessions()
            continue

        if line.startswith("/new "):
            name = line[5:].strip()
            if not name:
                print("Укажи имя сессии")
                continue
            try:
                create_session(name)
                current_session = name
                sync_session_info()
                print("Создана сессия:", current_session)
            except Exception as e:
                print("Ошибка:", e)
            continue

        if line.startswith("/use "):
            name = line[5:].strip()
            if not name:
                print("Укажи имя сессии")
                continue
            current_session = name
            try:
                sync_session_info()
                print("Переключено на:", current_session)
            except Exception as e:
                print("Ошибка:", e)
            continue

        if line == "/models":
            try:
                show_models()
            except Exception as e:
                print("Ошибка:", e)
            continue

        if line.startswith("/set "):
            parts = line.split(maxsplit=2)
            if len(parts) < 3:
                print("Формат: /set PROVIDER MODEL")
                continue
            provider = parts[1].strip()
            model = parts[2].strip()
            try:
                update_session(provider, model)
                sync_session_info()
                print("OK:", current_provider, current_model)
            except Exception as e:
                print("Ошибка:", e)
            continue

        if line == "/undo":
            try:
                send_message("/undo")
            except Exception as e:
                print("Ошибка:", e)
            continue

        if line == "/clear-history":
            try:
                send_message("/clear-history")
            except Exception as e:
                print("Ошибка:", e)
            continue

        if line == "/history":
            try:
                show_history()
            except Exception as e:
                print("Ошибка:", e)
            continue

        if line == "/exit":
            break

        try:
            send_message(line)
        except Exception as e:
            print("Ошибка:", e)


if __name__ == "__main__":
    main()
