#!/usr/bin/env bash
# Agentic RAG + MCP benchmark (file_search + Jira MCP).
#
# Tools injected into Llama Stack /v1/responses:
#   - file_search  (vector store / Milvus + embedding model)
#   - mcp          (Jira MCP server)
#
# Metrics: total_time, ttft, itl, rag_time_s, mcp_time_s, tokens.
#
# Requires:
#   export BENCH_VECTOR_STORE_IDS=vs_...
#   BENCH_MCP_SERVER_URL reachable from Llama Stack
#
# Example:
#   ./scripts/benchmark_rag.sh --requests 1 --no-phoenix
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_DIR="$ROOT/benchmarks"
VENV_DIR="$BENCH_DIR/.venv"
export PHOENIX_WORKING_DIR="${PHOENIX_WORKING_DIR:-$BENCH_DIR/.phoenix}"
mkdir -p "$PHOENIX_WORKING_DIR"

export BENCH_SCENARIO="${BENCH_SCENARIO:-rag}"
export BENCH_SYSTEM_PROMPT_FILE="${BENCH_SYSTEM_PROMPT_FILE:-$BENCH_DIR/prompts/rag/system.txt}"
export BENCH_USER_PROMPT_FILE="${BENCH_USER_PROMPT_FILE:-$BENCH_DIR/prompts/rag/user.txt}"
export PHOENIX_PROJECT_NAME="${PHOENIX_PROJECT_NAME:-jira-mcp-rag}"
export BENCH_MCP_SERVER_URL="${BENCH_MCP_SERVER_URL:-http://mcp-atlassian-acidonpe.apps.ocp.zd4ms.sandbox2306.opentlc.com/sse}"
export BENCH_MCP_SERVER_LABEL="${BENCH_MCP_SERVER_LABEL:-jira}"

if [[ -f "$ROOT/backend/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/backend/.env"
  set +a
fi

# Prefer explicit bench ids; fall back to app default vector stores.
export BENCH_VECTOR_STORE_IDS="${BENCH_VECTOR_STORE_IDS:-${DEFAULT_VECTOR_STORE_IDS:-}}"
export BENCH_MCP_SERVER_URL="${BENCH_MCP_SERVER_URL:-${MCP_SERVER_URL:-}}"

# Phoenix judge → Amazon Bedrock (not Llama Stack playground)
# shellcheck disable=SC1091
source "$ROOT/scripts/load_phoenix_judge_env.sh"

if [[ -z "${BENCH_VECTOR_STORE_IDS}" ]]; then
  echo "Set BENCH_VECTOR_STORE_IDS to one or more comma-separated vector store ids." >&2
  echo "Create a store via the StackChat UI or POST /api/rag/vector-stores, then upload docs." >&2
  exit 2
fi

if [[ -z "${BENCH_MCP_SERVER_URL}" ]]; then
  echo "Set BENCH_MCP_SERVER_URL to a Jira MCP SSE URL reachable from Llama Stack." >&2
  exit 2
fi

if [[ ! -f "$BENCH_USER_PROMPT_FILE" ]]; then
  if [[ -f "$BENCH_DIR/prompts/rag/user.txt.example" ]]; then
    cp "$BENCH_DIR/prompts/rag/user.txt.example" "$BENCH_USER_PROMPT_FILE"
    echo "Created $BENCH_USER_PROMPT_FILE from the example template."
    echo "Edit it (or pass --prompt), then re-run."
    exit 2
  fi
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
  set -- --scenario rag "$@"
fi

exec python "$BENCH_DIR/agentic_bench.py" "$@"
