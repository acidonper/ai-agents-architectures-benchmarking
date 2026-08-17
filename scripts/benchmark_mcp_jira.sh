#!/usr/bin/env bash
# Agentic Jira MCP benchmark (LLM + MCP) with optional Phoenix semantic evals.
#
# Metrics with --phoenix / BENCH_PHOENIX=1:
#   - orchestration_time_s   E2E LLM+MCP wall clock
#   - semantic_success_rate  answer fulfills the Jira task (LLM judge)
#   - trajectory_success_rate tool path is correct (LLM judge)
#   - combined_success_rate  both judges pass (+ HTTP ok)
#
# Phoenix judge always uses a Llama Stack LLM (/v1), not OpenAI.
# Set PHOENIX_JUDGE_MODEL to a Stack model id different from --model when available.
#
# GuideLLM load test (no semantic judge):
#   ./scripts/benchmark_guidellm.sh mcp-jira
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_DIR="$ROOT/benchmarks"
VENV_DIR="$BENCH_DIR/.venv"
export PHOENIX_WORKING_DIR="${PHOENIX_WORKING_DIR:-$BENCH_DIR/.phoenix}"
mkdir -p "$PHOENIX_WORKING_DIR"

export BENCH_MCP_SERVER_URL="${BENCH_MCP_SERVER_URL:-http://mcp-atlassian-acidonpe.apps.ocp.zd4ms.sandbox2306.opentlc.com/sse}"
export BENCH_MCP_SERVER_LABEL="${BENCH_MCP_SERVER_LABEL:-jira}"

# Load Stack URL / keys; force Phoenix judge onto Llama Stack.
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

# Default: enable Phoenix semantic/trajectory judges for this script.
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

exec python "$BENCH_DIR/agentic_bench.py" "$@"
