#!/usr/bin/env bash
# Bedrock AgentCore agentic benchmark
#
# Modes (--target):
#   bedrock-runtime  — LLM = Bedrock Converse (BENCH_MODEL_INFERENCE_PROFILE_ID)
#                      MCP = AgentCore Runtime InvokeAgentRuntime JSON-RPC
#                      RAG = bedrock-agent-runtime Retrieve when --scenario rag
#   bedrock-gateway  — POST /inference/v1/responses + gateway /mcp (legacy)
#
# Required (backend/.env.bedrock):
#   BENCH_MODEL_INFERENCE_PROFILE_ID=eu.amazon.nova-2-lite-v1:0
#   BEDROCK_AGENT_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/...
#   GW_AGENTIC_CLIENT_ID / SECRET + Cognito discovery (runtime JWT)
#   BEDROCK_KNOWLEDGE_BASE_ID=7PTWKIDPIA   (for --scenario rag)
#
# Example:
#   source scripts/load_bedrock_agentic_env.sh
#   ./scripts/benchmark_bedrock_agentic.sh --target bedrock-runtime --requests 1
#   ./scripts/benchmark_bedrock_rag.sh --requests 1
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_DIR="$ROOT/benchmarks"
VENV_DIR="$BENCH_DIR/.venv"
export PHOENIX_WORKING_DIR="${PHOENIX_WORKING_DIR:-$BENCH_DIR/.phoenix}"
mkdir -p "$PHOENIX_WORKING_DIR"

export BENCH_AGENTIC_TARGET="${BENCH_AGENTIC_TARGET:-bedrock-runtime}"
export BENCH_SCENARIO="${BENCH_SCENARIO:-list}"
export PHOENIX_PROJECT_NAME="${PHOENIX_PROJECT_NAME:-bedrock-agentic}"

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
# Normalize AgentCore gateway + Cognito OAuth env (GW_AGENTIC_* → BEDROCK_*)
# shellcheck disable=SC1091
source "$ROOT/scripts/load_bedrock_agentic_env.sh"
# Phoenix judge → Amazon Bedrock (not Llama Stack playground)
# shellcheck disable=SC1091
source "$ROOT/scripts/load_phoenix_judge_env.sh"

TARGET="${1:-}"
if [[ "$TARGET" == "--target" ]]; then
  TARGET="${2:-}"
  shift 2
elif [[ "$TARGET" == bedrock-gateway || "$TARGET" == bedrock-runtime ]]; then
  shift
fi

if [[ -n "$TARGET" ]]; then
  set -- --target "$TARGET" "$@"
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

exec python "$BENCH_DIR/agentic_bench.py" "$@"
