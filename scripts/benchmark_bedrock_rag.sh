#!/usr/bin/env bash
# Bedrock KB RAG (Retrieve API) + Runtime MCP + Converse LLM.
#
# Requires backend/.env.bedrock:
#   BENCH_MODEL_INFERENCE_PROFILE_ID
#   BEDROCK_AGENT_RUNTIME_ARN
#   BEDROCK_KNOWLEDGE_BASE_ID
#   Cognito GW_AGENTIC_* (runtime JWT)
#
# Example:
#   $EDITOR benchmarks/prompts/rag/user.txt
#   ./scripts/benchmark_bedrock_rag.sh --requests 1
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_DIR="$ROOT/benchmarks"
VENV_DIR="$BENCH_DIR/.venv"
export PHOENIX_WORKING_DIR="${PHOENIX_WORKING_DIR:-$BENCH_DIR/.phoenix}"
mkdir -p "$PHOENIX_WORKING_DIR"

export BENCH_AGENTIC_TARGET="${BENCH_AGENTIC_TARGET:-bedrock-runtime}"
export BENCH_SCENARIO="${BENCH_SCENARIO:-rag}"
export BENCH_RAG_MODE="${BENCH_RAG_MODE:-bedrock_kb}"
export BENCH_SYSTEM_PROMPT_FILE="${BENCH_SYSTEM_PROMPT_FILE:-$BENCH_DIR/prompts/rag/system.txt}"
export BENCH_USER_PROMPT_FILE="${BENCH_USER_PROMPT_FILE:-$BENCH_DIR/prompts/rag/user.txt}"
export PHOENIX_PROJECT_NAME="${PHOENIX_PROJECT_NAME:-bedrock-rag-mcp}"

if [[ -f "$ROOT/backend/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/backend/.env"
  set +a
fi
if [[ -f "$ROOT/backend/.env.bedrock" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/backend/.env.bedrock"
  set +a
fi
# shellcheck disable=SC1091
source "$ROOT/scripts/load_bedrock_agentic_env.sh"
# Phoenix judge → Amazon Bedrock (not Llama Stack playground)
# shellcheck disable=SC1091
source "$ROOT/scripts/load_phoenix_judge_env.sh"

if [[ -z "${BEDROCK_AGENT_RUNTIME_ARN:-}" ]]; then
  echo "Set BEDROCK_AGENT_RUNTIME_ARN in backend/.env.bedrock (AgentCore MCP runtime)." >&2
  exit 2
fi

if [[ ! -f "$BENCH_USER_PROMPT_FILE" ]]; then
  if [[ -f "$BENCH_DIR/prompts/rag/user.txt.example" ]]; then
    cp "$BENCH_DIR/prompts/rag/user.txt.example" "$BENCH_USER_PROMPT_FILE"
    echo "Created $BENCH_USER_PROMPT_FILE from the example template."
    echo "Edit it, then re-run."
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
  python -c "import boto3" 2>/dev/null || pip install -r "$BENCH_DIR/requirements.txt"
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

exec python "$BENCH_DIR/agentic_bench.py" \
  --target bedrock-runtime \
  --scenario rag \
  --rag-mode bedrock_kb \
  "$@"
