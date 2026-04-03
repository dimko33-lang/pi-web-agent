#!/usr/bin/env bash
set -Eeuo pipefail

REPO_URL="${REPO_URL:-https://github.com/dimko33-lang/pi-web-agent.git}"
WORKDIR="/root/pi-web-agent-src"

if [ "$(id -u)" -ne 0 ]; then
  echo "Please run as root (or with sudo)."
  exit 1
fi

while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
  echo "Waiting for dpkg lock..."
  sleep 2
done

apt update
apt install -y git

rm -rf "$WORKDIR"
git clone "$REPO_URL" "$WORKDIR"

if ! cd "$WORKDIR"; then
  echo "ERROR: Can't cd to $WORKDIR"
  exit 1
fi

if [ ! -f install.sh ]; then
  echo "ERROR: install.sh not found in $WORKDIR"
  exit 1
fi

chmod +x install.sh
exec bash install.sh
