#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

start_backend() {
  cd "$ROOT/backend"
  if [[ ! -d .venv ]]; then
    python3.12 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install -U pip
    pip install -r requirements.txt
  else
    # shellcheck disable=SC1091
    source .venv/bin/activate
  fi
  exec uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
}

start_frontend() {
  cd "$ROOT/frontend"
  if [[ ! -d node_modules ]]; then
    npm install
  fi
  exec npm run dev
}

case "${1:-}" in
  backend) start_backend ;;
  frontend) start_frontend ;;
  *)
    echo "Usage: $0 {backend|frontend}"
    exit 1
    ;;
esac
