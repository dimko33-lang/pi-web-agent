#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "ERROR: line $LINENO: $BASH_COMMAND" >&2' ERR
export DEBIAN_FRONTEND=noninteractive
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="/opt/my-agent"
SERVICE_USER="my-agent"
ENV_FILE="/etc/my-agent.env"
SERVICE_FILE="/etc/systemd/system/my-agent.service"
NGINX_SITE="/etc/nginx/sites-available/my-agent"
NGINX_LINK="/etc/nginx/sites-enabled/my-agent"

log() {
  echo
  echo "==> $*"
}

aptx() {
  apt-get -o DPkg::Lock::Timeout=300 -o Acquire::Retries=3 "$@"
}

if [ "$(id -u)" -ne 0 ]; then
  echo "Run as root."
  exit 1
fi

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemd not found. This script needs a normal Ubuntu server."
  exit 1
fi

DEFAULT_IP="$(hostname -I | awk '{print $1}')"
PUBLIC_HOST="${PUBLIC_HOST:-}"
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
GROQ_API_KEY="${GROQ_API_KEY:-}"
GEMINI_API_KEY="${GEMINI_API_KEY:-}"
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
KIMI_API_KEY="${KIMI_API_KEY:-}"

PROMPT_FD=""
if [ -t 0 ]; then
  PROMPT_FD=0
elif [ -r /dev/tty ]; then
  exec 3<>/dev/tty
  PROMPT_FD=3
fi

prompt_text() {
  local __var="$1"
  local __label="$2"
  local __default="${3:-}"
  local __value=""
  [ -z "$PROMPT_FD" ] && return 0
  if [ -n "$__default" ]; then
    printf "%s [%s]: " "$__label" "$__default" > /dev/tty
  else
    printf "%s: " "$__label" > /dev/tty
  fi
  IFS= read -r -u "$PROMPT_FD" __value || true
  [ -z "$__value" ] && __value="$__default"
  printf -v "$__var" '%s' "$__value"
}

prompt_key_visible() {
  local __var="$1"
  local __label="$2"
  local __value=""
  [ -z "$PROMPT_FD" ] && return 0
  printf "%s (press Enter to skip): " "$__label" > /dev/tty
  IFS= read -r -u "$PROMPT_FD" __value || true
  printf -v "$__var" '%s' "$__value"
}

if [ -z "$PUBLIC_HOST" ]; then
  prompt_text PUBLIC_HOST "Public host/IP" "$DEFAULT_IP"
fi
PUBLIC_HOST="${PUBLIC_HOST:-$DEFAULT_IP}"

if [ -n "$PROMPT_FD" ]; then
  printf "\nYou can paste keys directly into the terminal.\n" > /dev/tty
  printf "If you do not have a key for a provider, just press Enter.\n" > /dev/tty
  printf "Current built-in providers in the app: OpenRouter, GROQ, Kimi.\n" > /dev/tty
  printf "Extra keys can also be stored now for future use: Gemini, Anthropic, OpenAI.\n\n" > /dev/tty
fi

[ -z "$OPENROUTER_API_KEY" ] && prompt_key_visible OPENROUTER_API_KEY "OPENROUTER_API_KEY"
[ -z "$GROQ_API_KEY" ] && prompt_key_visible GROQ_API_KEY "GROQ_API_KEY"
[ -z "$GEMINI_API_KEY" ] && prompt_key_visible GEMINI_API_KEY "GEMINI_API_KEY"
[ -z "$ANTHROPIC_API_KEY" ] && prompt_key_visible ANTHROPIC_API_KEY "ANTHROPIC_API_KEY"
[ -z "$OPENAI_API_KEY" ] && prompt_key_visible OPENAI_API_KEY "OPENAI_API_KEY"
[ -z "$KIMI_API_KEY" ] && prompt_key_visible KIMI_API_KEY "KIMI_API_KEY"

if [ -z "${OPENROUTER_API_KEY}${GROQ_API_KEY}${KIMI_API_KEY}" ]; then
  echo "At least one currently supported key is required: OPENROUTER_API_KEY, GROQ_API_KEY, or KIMI_API_KEY."
  exit 1
fi

PI_PUBLIC_URL="http://${PUBLIC_HOST}"

log "Installing packages"
aptx update
aptx install -y --no-install-recommends python3-full python3-venv nginx rsync git curl

log "Creating service user"
id -u "$SERVICE_USER" >/dev/null 2>&1 || useradd --system --create-home --home-dir "$APP_DIR" --shell /usr/sbin/nologin "$SERVICE_USER"
mkdir -p "$APP_DIR" "$APP_DIR/sessions"

log "Syncing app files"
rsync -a --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'sessions' \
  --exclude 'agent.db*' \
  --exclude '.env*' \
  --exclude 'global_model.json' \
  --exclude 'session_aliases.json' \
  --exclude 'bootstrap.sh' \
  --exclude 'install.sh' \
  --exclude 'update.sh' \
  --exclude 'README.md' \
  --exclude '.gitignore' \
  --exclude 'env.example' \
  "$REPO_DIR/" "$APP_DIR/"

chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_DIR"

log "Creating virtualenv and installing Python deps"
runuser -u "$SERVICE_USER" -- bash -lc "
set -Eeuo pipefail
cd '$APP_DIR'
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m py_compile main.py agent.py cli.py db.py
"

log "Writing env file"
{
  printf 'OPENROUTER_API_KEY=%s\n' "$OPENROUTER_API_KEY"
  printf 'GROQ_API_KEY=%s\n' "$GROQ_API_KEY"
  printf 'GEMINI_API_KEY=%s\n' "$GEMINI_API_KEY"
  printf 'ANTHROPIC_API_KEY=%s\n' "$ANTHROPIC_API_KEY"
  printf 'OPENAI_API_KEY=%s\n' "$OPENAI_API_KEY"
  printf 'KIMI_API_KEY=%s\n' "$KIMI_API_KEY"
  printf 'PI_PUBLIC_URL=%s\n' "$PI_PUBLIC_URL"
} > "$ENV_FILE"
chmod 600 "$ENV_FILE"

log "Writing systemd service"
cat > "$SERVICE_FILE" <<'UNITEOF'
[Unit]
Description=PI Browser Agent
After=network.target
[Service]
Type=simple
User=my-agent
Group=my-agent
WorkingDirectory=/opt/my-agent
EnvironmentFile=/etc/my-agent.env
ExecStart=/opt/my-agent/.venv/bin/gunicorn -k gevent -w 1 -b 127.0.0.1:8000 main:app --access-logfile -
Restart=on-failure
RestartSec=2
NoNewPrivileges=true
PrivateTmp=true
[Install]
WantedBy=multi-user.target
UNITEOF

log "Writing nginx config"
cat > "$NGINX_SITE" <<'NGINXEOF'
server {
    listen 80;
    server_name _;
    location = / {
        if ($query_string = "") { return 404; }
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    location /events {
        proxy_pass http://127.0.0.1:8000/events;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 1h;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
NGINXEOF

ln -sfn "$NGINX_SITE" "$NGINX_LINK"
rm -f /etc/nginx/sites-enabled/default

log "Initializing aliases / sessions / global model"
"$APP_DIR/.venv/bin/python" - <<'PY'
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, "/opt/my-agent")
from db import init_db, get_or_create_session, update_session_state, session_html_path
APP = Path("/opt/my-agent")
ALIAS_FILE = APP / "session_aliases.json"
GLOBAL_FILE = APP / "global_model.json"
TEMPLATE = APP / "default_terminal.html"

aliases = {
    "admin": "private",
    "default": "default",
    "slot1": "slot1",
    "slot2": "slot2"
}

ALIAS_FILE.write_text(json.dumps(aliases, ensure_ascii=False, indent=2), encoding="utf-8")
GLOBAL_FILE.write_text(json.dumps({"provider": "groq", "model": "llama-3.1-8b-instant"}, ensure_ascii=False, indent=2), encoding="utf-8")

init_db()
for name in ["default", "slot1", "slot2", "slot3", "slot4", "slot5", "private"]:
    s = get_or_create_session(name, provider="groq", model="llama-3.1-8b-instant")
    update_session_state(s["id"], "groq", "llama-3.1-8b-instant")
    shutil.copy2(TEMPLATE, session_html_path(s["id"]))
PY

chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_DIR"

log "Starting services"
systemctl daemon-reload
systemctl enable my-agent
systemctl restart my-agent

nginx -t
systemctl enable nginx
systemctl restart nginx

log "Health"
curl -fsS http://127.0.0.1:8000/healthz || true

echo
echo "Installation completed."
echo
echo "Admin panel:"
echo "   ${PI_PUBLIC_URL}/?admin"
echo
echo "Client panels:"
echo "   ${PI_PUBLIC_URL}/?default"
echo "   ${PI_PUBLIC_URL}/?slot1"
echo "   ${PI_PUBLIC_URL}/?slot2"
echo
echo "More client panels: edit /opt/my-agent/session_aliases.json then run: systemctl restart my-agent"
echo
echo "Helpful commands:"
echo "   systemctl status my-agent"
echo "   journalctl -u my-agent -f"
echo "   cd /opt/my-agent"
echo
echo "Done."
echo
