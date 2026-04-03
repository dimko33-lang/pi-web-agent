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
GROQ_API_KEY="${GROQ_API_KEY:-}"
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
KIMI_API_KEY="${KIMI_API_KEY:-}"

# Если есть нормальная консоль — можно спросить
if [ -t 0 ]; then
  if [ -z "$PUBLIC_HOST" ]; then
    read -rp "Public host/IP [$DEFAULT_IP]: " PUBLIC_HOST
    PUBLIC_HOST="${PUBLIC_HOST:-$DEFAULT_IP}"
  fi

  if [ -z "$GROQ_API_KEY" ]; then
    read -srp "GROQ_API_KEY: " GROQ_API_KEY
    echo
  fi

  if [ -z "$OPENROUTER_API_KEY" ]; then
    read -srp "OPENROUTER_API_KEY (optional): " OPENROUTER_API_KEY
    echo
  fi

  if [ -z "$KIMI_API_KEY" ]; then
    read -srp "KIMI_API_KEY (optional): " KIMI_API_KEY
    echo
  fi
else
  # Если установка идет автоматом — не спрашиваем ничего
  PUBLIC_HOST="${PUBLIC_HOST:-$DEFAULT_IP}"
fi

if [ -z "$GROQ_API_KEY" ]; then
  echo "GROQ_API_KEY is required."
  echo "Set it like this:"
  echo "GROQ_API_KEY=xxx PUBLIC_HOST=1.2.3.4 bash install.sh"
  exit 1
fi

PI_PUBLIC_URL="http://${PUBLIC_HOST}"

log "Installing packages"
aptx update
aptx install -y --no-install-recommends python3-full python3-venv nginx rsync git curl

log "Creating service user"
id -u "$SERVICE_USER" >/dev/null 2>&1 || \
  useradd --system --create-home --home-dir "$APP_DIR" --shell /usr/sbin/nologin "$SERVICE_USER"

mkdir -p "$APP_DIR"
mkdir -p "$APP_DIR/sessions"

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
  printf 'GROQ_API_KEY=%s\n' "$GROQ_API_KEY"
  printf 'OPENROUTER_API_KEY=%s\n' "$OPENROUTER_API_KEY"
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
        if ($query_string = "") {
            return 404;
        }

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
import secrets
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/opt/my-agent")
from db import init_db, get_or_create_session, update_session_state, session_html_path

APP = Path("/opt/my-agent")
ALIAS_FILE = APP / "session_aliases.json"
GLOBAL_FILE = APP / "global_model.json"
TEMPLATE = APP / "default_terminal.html"

base = "-projects-pi-ru"
admin_alias = f"{base}-admin-{secrets.token_hex(6)}"

aliases = {
    base: "default",
    f"{base}/1": "slot1",
    f"{base}/2": "slot2",
    f"{base}/3": "slot3",
    f"{base}/4": "slot4",
    f"{base}/5": "slot5",
    admin_alias: "private",
}

ALIAS_FILE.write_text(json.dumps(aliases, ensure_ascii=False, indent=2), encoding="utf-8")
GLOBAL_FILE.write_text(json.dumps({
    "provider": "groq",
    "model": "llama-3.1-8b-instant"
}, ensure_ascii=False, indent=2), encoding="utf-8")

init_db()

for name in ["default", "slot1", "slot2", "slot3", "slot4", "slot5", "private"]:
    s = get_or_create_session(name, provider="groq", model="llama-3.1-8b-instant")
    update_session_state(s["id"], "groq", "llama-3.1-8b-instant")
    shutil.copy2(TEMPLATE, session_html_path(s["id"]))

print("ADMIN_ALIAS=" + admin_alias)
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
echo "Install complete."
echo "Public: ${PI_PUBLIC_URL}/?-projects-pi-ru"
echo "Backup: ${PI_PUBLIC_URL}/?-projects-pi-ru/1"
echo "Admin alias file: /opt/my-agent/session_aliases.json"
