#!/usr/bin/env bash
# Agentic Jira MCP create-issue benchmark (LLM + MCP).
#
# Edit the user prompt first:
#   $EDITOR benchmarks/prompts/jira_create/user.txt
#
# System prompt (checked in):
#   benchmarks/prompts/jira_create/system.txt
#
# Example:
#   ./scripts/benchmark_mcp_jira_create.sh --requests 1
#
# Creates real Jira issues — keep --requests low until prompts are correct.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_DIR="$ROOT/benchmarks"
VENV_DIR="$BENCH_DIR/.venv"
export PHOENIX_WORKING_DIR="${PHOENIX_WORKING_DIR:-$BENCH_DIR/.phoenix}"
mkdir -p "$PHOENIX_WORKING_DIR"

export BENCH_SCENARIO="${BENCH_SCENARIO:-create}"
export BENCH_MCP_SERVER_URL="${BENCH_MCP_SERVER_URL:-http://mcp-atlassian-acidonpe.apps.ocp.zd4ms.sandbox2306.opentlc.com/sse}"
export BENCH_MCP_SERVER_LABEL="${BENCH_MCP_SERVER_LABEL:-jira}"
export BENCH_SYSTEM_PROMPT_FILE="${BENCH_SYSTEM_PROMPT_FILE:-$BENCH_DIR/prompts/jira_create/system.txt}"
export BENCH_USER_PROMPT_FILE="${BENCH_USER_PROMPT_FILE:-$BENCH_DIR/prompts/jira_create/user.txt}"
export PHOENIX_PROJECT_NAME="${PHOENIX_PROJECT_NAME:-jira-mcp-create}"

if [[ -f "$ROOT/backend/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/backend/.env"
  set +a
fi
export PHOENIX_JUDGE_BASE_URL="${PHOENIX_JUDGE_BASE_URL:-${LLAMA_STACK_BASE_URL:-}}"
if [[ "${PHOENIX_JUDGE_BASE_URL:-}" == *"api.openai.com"* ]]; then
  export PHOENIX_JUDGE_BASE_URL="${LLAMA_STACK_BASE_URL:-}"
fi

PY=python3.12
if ! command -v "$PY" >/dev/null 2>&1; then
  PY=python3
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PY" -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  pip install -U pip
  pip install -r "$BENCH_DIR/requirements.txt"
else
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python -c "import phoenix, phoenix.evals" 2>/dev/null || pip install -r "$BENCH_DIR/requirements.txt"
fi

if [[ "${BENCH_PHOENIX:-1}" != "0" ]]; then
  has_flag=0
  for a in "$@"; do
    if [[ "$a" == "--phoenix" || "$a" == "--no-phoenix" ]]; then
      has_flag=1
      break
    fi
  done
  if [[ "$has_flag" -eq 0 ]]; then
    set -- --phoenix "$@"
  fi
fi

has_scenario=0
for a in "$@"; do
  if [[ "$a" == "--scenario" ]]; then
    has_scenario=1
    break
  fi
done
if [[ "$has_scenario" -eq 0 ]]; then
  set -- --scenario create "$@"
fi

if [[ ! -f "$BENCH_USER_PROMPT_FILE" ]]; then
  if [[ -f "$BENCH_DIR/prompts/jira_create/user.txt.example" ]]; then
    cp "$BENCH_DIR/prompts/jira_create/user.txt.example" "$BENCH_USER_PROMPT_FILE"
    echo "Created $BENCH_USER_PROMPT_FILE from the example template."
    echo "Edit it (replace every <PLACEHOLDER>), then re-run this script."
    exit 2
  fi
  echo "Missing user prompt file: $BENCH_USER_PROMPT_FILE" >&2
  exit 2
fi

exec python "$BENCH_DIR/agentic_bench.py" "$@"
