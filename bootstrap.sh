#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "ERROR: line $LINENO: $BASH_COMMAND" >&2' ERR
export DEBIAN_FRONTEND=noninteractive

REPO_URL="${REPO_URL:-https://github.com/dimko33-lang/pi-web-agent.git}"
REPO_REF="${REPO_REF:-main}"
WORKDIR="${WORKDIR:-/root/pi-web-agent-src}"

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

if command -v cloud-init >/dev/null 2>&1; then
  log "Waiting for cloud-init"
  cloud-init status --wait || true
fi

log "Installing git"
aptx update
aptx install -y git ca-certificates

log "Cloning repo"
rm -rf "$WORKDIR"
git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$WORKDIR"

cd "$WORKDIR"
if [ ! -f install.sh ]; then
  echo "install.sh not found"
  exit 1
fi

chmod +x install.sh
log "Running install.sh"
./install.sh
