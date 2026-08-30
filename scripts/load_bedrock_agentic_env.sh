#!/usr/bin/env bash
# Load AgentCore Gateway + Cognito OAuth settings for bedrock-gateway benchmarks.
#
# Sources (first match wins for secrets):
#   backend/.env.bedrock  (gitignored — copy from backend/.env.bedrock.example)
#   backend/.env
#
# Usage (bash or zsh):
#   source ./scripts/load_bedrock_agentic_env.sh
#   ./scripts/benchmark_bedrock_agentic.sh --target bedrock-gateway --requests 1
set -euo pipefail

# Resolve repo root when sourced from bash or zsh
if [[ -n "${BASH_VERSION:-}" ]]; then
  _load_script="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  _load_script="${(%):-%x}"
else
  _load_script="$0"
fi
ROOT="$(cd "$(dirname "$_load_script")/.." && pwd)"
unset _load_script

if [[ -f "$ROOT/backend/.env.bedrock" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/backend/.env.bedrock"
  set +a
elif [[ -f "$ROOT/backend/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/backend/.env"
  set +a
fi

# --- Agentic gateway (GW_HOSTNAME wins over constructed URL) ---
export BEDROCK_REGION="${BEDROCK_REGION:-${AWS_REGION:-us-east-1}}"

# Legacy typo: GW_AGENTIC_id → GW_AGENTIC_ID
if [[ -z "${GW_AGENTIC_ID:-}" && -n "${GW_AGENTIC_id:-}" ]]; then
  export GW_AGENTIC_ID="${GW_AGENTIC_id}"
fi

# Full gateway URL from console (preferred)
if [[ -n "${GW_HOSTNAME:-}" ]]; then
  GW_HOSTNAME="${GW_HOSTNAME%/}"
  export GW_AGENTIC_GATEWAY_URL="${GW_HOSTNAME}"
  export BEDROCK_GATEWAY_URL="${GW_HOSTNAME}"
  # Align region with hostname when BEDROCK_REGION was left as a default
  host_region="$(printf '%s' "${GW_HOSTNAME}" | sed -n 's/.*gateway\.bedrock-agentcore\.\([a-z0-9-]*\)\.amazonaws\.com.*/\1/p')"
  if [[ -n "${host_region}" && "${BEDROCK_REGION}" != "${host_region}" ]]; then
    echo "NOTE: BEDROCK_REGION=${BEDROCK_REGION} → ${host_region} (from GW_HOSTNAME)" >&2
    export BEDROCK_REGION="${host_region}"
  fi
elif [[ -n "${GW_AGENTIC_ID:-}" ]]; then
  expected_host="${GW_AGENTIC_ID}.gateway.bedrock-agentcore.${BEDROCK_REGION}.amazonaws.com"
  if [[ -z "${GW_AGENTIC_GATEWAY_URL:-}" ]]; then
    export GW_AGENTIC_GATEWAY_URL="https://${expected_host}"
  elif [[ "${GW_AGENTIC_GATEWAY_URL}" != *"${GW_AGENTIC_ID}"* ]]; then
    echo "WARN: GW_AGENTIC_GATEWAY_URL does not match GW_AGENTIC_ID; using ID-based URL." >&2
    export GW_AGENTIC_GATEWAY_URL="https://${expected_host}"
  fi
  export BEDROCK_GATEWAY_URL="${BEDROCK_GATEWAY_URL:-${GW_AGENTIC_GATEWAY_URL:-}}"
else
  export BEDROCK_GATEWAY_URL="${BEDROCK_GATEWAY_URL:-${GW_AGENTIC_GATEWAY_URL:-}}"
fi

# Cognito OAuth for gateway ingress (AgentCore quick-start pattern)
export BEDROCK_OAUTH_DISCOVERY_URL="${BEDROCK_OAUTH_DISCOVERY_URL:-${GW_AGENTIC_COGNITO_DISCOVERY_URL:-}}"
export BEDROCK_OAUTH_AUTHORITY="${BEDROCK_OAUTH_AUTHORITY:-${GW_AGENTIC_OAUTH_AUTHORITY:-}}"
export BEDROCK_OAUTH_CLIENT_ID="${BEDROCK_OAUTH_CLIENT_ID:-${GW_AGENTIC_CLIENT_ID:-}}"
export BEDROCK_OAUTH_CLIENT_SECRET="${BEDROCK_OAUTH_CLIENT_SECRET:-${GW_AGENTIC_CLIENT_SECRET:-}}"
export BEDROCK_OAUTH_SCOPE="${BEDROCK_OAUTH_SCOPE:-${GW_AGENTIC_OAUTH_SCOPE:-}}"

# Agent target path (NOT OAuth scope): GW_AGENTIC_TARGET_QUICK_START=target-quick-start-717dc2
export GW_AGENTIC_TARGET_QUICK_START="${GW_AGENTIC_TARGET_QUICK_START:-}"

# Managed knowledge base (gateway MCP Retrieve / AgenticRetrieveStream)
export BEDROCK_KNOWLEDGE_BASE_ID="${BEDROCK_KNOWLEDGE_BASE_ID:-${GW_KNOWLEDGE_BASE_QUICK_START:-}}"
export GW_KNOWLEDGE_BASE_QUICK_START="${GW_KNOWLEDGE_BASE_QUICK_START:-${BEDROCK_KNOWLEDGE_BASE_ID:-}}"
export BEDROCK_KB_EMBEDDING_MODEL="${BEDROCK_KB_EMBEDDING_MODEL:-${GW_KB_EMBEDDING_MODEL:-}}"
export BEDROCK_KB_MCP_TARGET_NAME="${BEDROCK_KB_MCP_TARGET_NAME:-${GW_KB_TARGET_QUICK_START:-}}"
export BENCH_RAG_MODE="${BENCH_RAG_MODE:-}"

# Default auth: Cognito when client credentials + agent target are configured
if [[ -z "${BEDROCK_AUTH_MODE:-}" ]]; then
  if [[ -n "${BEDROCK_OAUTH_CLIENT_ID:-}" && -n "${BEDROCK_OAUTH_CLIENT_SECRET:-}" ]]; then
    export BEDROCK_AUTH_MODE=cognito
  elif [[ -n "${BEDROCK_API_KEY:-}" ]]; then
    export BEDROCK_AUTH_MODE=api-key
  else
    export BEDROCK_AUTH_MODE=iam
  fi
fi

# Agent invocation via gateway target requires Cognito client credentials
if [[ -n "${GW_AGENTIC_TARGET_QUICK_START:-}" ]]; then
  if [[ -z "${BEDROCK_OAUTH_CLIENT_ID:-}" || -z "${BEDROCK_OAUTH_CLIENT_SECRET:-}" ]]; then
    echo "ERROR: GW_AGENTIC_TARGET_QUICK_START is set but Cognito client id/secret are missing." >&2
    exit 2
  fi
  export BEDROCK_AUTH_MODE=cognito
fi

# Static Bedrock API key → bearer for gateway when mode is api-key
if [[ "${BEDROCK_AUTH_MODE}" == api-key && -n "${BEDROCK_API_KEY:-}" ]]; then
  export BEDROCK_GATEWAY_TOKEN="${BEDROCK_GATEWAY_TOKEN:-$BEDROCK_API_KEY}"
fi

export AWS_REGION="${AWS_REGION:-${BEDROCK_REGION}}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION}}"
export AWS_REGION_NAME="${AWS_REGION_NAME:-${AWS_REGION}}"

export BENCH_AGENTIC_TARGET="${BENCH_AGENTIC_TARGET:-bedrock-runtime}"

# AgentCore Runtime (MCP server — InvokeAgentRuntime JSON-RPC)
export BEDROCK_AGENT_RUNTIME_ARN="${BEDROCK_AGENT_RUNTIME_ARN:-${GW_AGENTIC_RUNTIME_ARN:-}}"
export BEDROCK_RUNTIME_QUALIFIER="${BEDROCK_RUNTIME_QUALIFIER:-DEFAULT}"

# Inference model (prefer .env.bedrock profile id over stale shell BENCH_MODEL)
if [[ -n "${BENCH_MODEL_INFERENCE_PROFILE_ID:-}" ]]; then
  export BENCH_MODEL="${BENCH_MODEL_INFERENCE_PROFILE_ID}"
elif [[ -z "${BENCH_MODEL:-}" ]]; then
  export BENCH_MODEL="${BENCH_MODEL_INFERENCE_PROFILE_ID:-}"
fi

echo "Bedrock bench env:"
echo "  GW_HOSTNAME=${GW_HOSTNAME:-<unset>}"
echo "  BEDROCK_GATEWAY_URL=${BEDROCK_GATEWAY_URL:-<unset>}"
echo "  BEDROCK_REGION=${BEDROCK_REGION}"
echo "  BEDROCK_AUTH_MODE=${BEDROCK_AUTH_MODE}"
echo "  BENCH_MODEL=${BENCH_MODEL:-<unset>}"
echo "  PHOENIX_JUDGE_PROVIDER=${PHOENIX_JUDGE_PROVIDER:-bedrock}"
echo "  PHOENIX_JUDGE_MODEL=${PHOENIX_JUDGE_MODEL:-${BENCH_MODEL_INFERENCE_PROFILE_ID:-<unset>}}"
echo "  BEDROCK_OAUTH_SCOPE=${BEDROCK_OAUTH_SCOPE:-<unset>}"
if [[ -n "${BEDROCK_AGENT_RUNTIME_ARN:-}" ]]; then
  echo "  BEDROCK_AGENT_RUNTIME_ARN=${BEDROCK_AGENT_RUNTIME_ARN}"
  echo "  BEDROCK_RUNTIME_QUALIFIER=${BEDROCK_RUNTIME_QUALIFIER}"
fi
if [[ -n "${GW_AGENTIC_TARGET_QUICK_START:-}" ]]; then
  echo "  GW_AGENTIC_TARGET=${GW_AGENTIC_TARGET_QUICK_START}"
  echo "  invoke_url=${BEDROCK_GATEWAY_URL:-}/${GW_AGENTIC_TARGET_QUICK_START#/}/invocations"
fi
if [[ -n "${BEDROCK_KNOWLEDGE_BASE_ID:-}" ]]; then
  echo "  BEDROCK_KNOWLEDGE_BASE_ID=${BEDROCK_KNOWLEDGE_BASE_ID}"
  echo "  BEDROCK_KB_EMBEDDING_MODEL=${BEDROCK_KB_EMBEDDING_MODEL:-<unset>}"
  echo "  BENCH_RAG_MODE=${BENCH_RAG_MODE:-<unset>}"
fi
