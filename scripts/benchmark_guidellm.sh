#!/usr/bin/env bash
# Benchmark the Llama Stack model with GuideLLM from the terminal.
#
# GuideLLM talks to an OpenAI-compatible endpoint and generates its own
# synthetic traffic. It does NOT use the StackChat FastAPI /api/chat backend.
#
# Target: Llama Stack  →  /v1/chat/completions
# (same inference path the chat app ultimately uses, without RAG/MCP overhead)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_DIR="$ROOT/benchmarks"
RESULTS_DIR="$BENCH_DIR/results"
VENV_DIR="$BENCH_DIR/.venv"
# Keep HF tokenizer downloads inside the project.
export HF_HOME="${HF_HOME:-$BENCH_DIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

# macOS + torch/ObjC: GuideLLM's default fork multiprocessing segfaults (SIGSEGV).
# spawn is required; override with GUIDELLM__MP_CONTEXT_TYPE if needed.
if [[ "$(uname -s)" == "Darwin" ]]; then
  export GUIDELLM__MP_CONTEXT_TYPE="${GUIDELLM__MP_CONTEXT_TYPE:-spawn}"
fi

# Load StackChat backend env if present (LLAMA_STACK_BASE_URL, API key, model).
if [[ -f "$ROOT/backend/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/backend/.env"
  set +a
fi

TARGET="${BENCH_TARGET:-${GUIDELLM_TARGET:-${LLAMA_STACK_BASE_URL:-http://localhost:8321}}}"
TARGET="${TARGET%/}"
MODEL="${BENCH_MODEL:-${GUIDELLM_MODEL:-${DEFAULT_MODEL:-vllm-inference-1/llama-32-fp8}}}"
# Stack model ids are not Hugging Face repos. Default to a public tokenizer so
# the smoke run works without HF auth; for accurate Llama counts use
# BENCH_TOKENIZER=meta-llama/Llama-3.2-3B-Instruct (needs HF_TOKEN if gated).
TOKENIZER="${BENCH_TOKENIZER:-${GUIDELLM_TOKENIZER:-gpt2}}"
API_KEY="${BENCH_API_KEY:-${GUIDELLM_API_KEY:-${LLAMA_STACK_API_KEY:-}}}"
SCENARIO="${1:-quick}"
DURATION="${BENCH_DURATION:-${GUIDELLM_DURATION:-30}}"
PROMPT_TOKENS="${BENCH_PROMPT_TOKENS:-${GUIDELLM_PROMPT_TOKENS:-256}}"
OUTPUT_TOKENS="${BENCH_OUTPUT_TOKENS:-${GUIDELLM_OUTPUT_TOKENS:-128}}"
MAX_REQUESTS="${BENCH_MAX_REQUESTS:-${GUIDELLM_MAX_REQUESTS:-}}"
MCP_URL="${BENCH_MCP_SERVER_URL:-${MCP_SERVER_URL:-}}"
MCP_LABEL="${BENCH_MCP_SERVER_LABEL:-jira}"
MCP_TIMEOUT="${BENCH_TIMEOUT:-180}"

usage() {
  cat <<EOF
Usage: $0 [scenario|raw]

  scenario   quick (default) | sweep | concurrent | throughput | mcp-jira
  raw        Pass remaining args straight to \`guidellm run\`

Environment (optional):
  BENCH_TARGET / LLAMA_STACK_BASE_URL   Llama Stack base URL (no trailing /v1)
  BENCH_MODEL  / DEFAULT_MODEL          Model id (default: vllm-inference-1/llama-32-fp8)
  BENCH_TOKENIZER                       HF tokenizer id (default: gpt2)
  BENCH_API_KEY / LLAMA_STACK_API_KEY   Bearer token if the stack requires auth
  BENCH_PROMPT_TOKENS                   Synthetic prompt size (default: 256)
  BENCH_OUTPUT_TOKENS                   Synthetic output size (default: 128)
  BENCH_DURATION                        Seconds per strategy (default: 30)
  BENCH_MAX_REQUESTS                    Cap requests instead of duration
  BENCH_MCP_SERVER_URL                  Required for mcp-jira (reachable from Llama Stack)
  BENCH_MCP_SERVER_LABEL                MCP label (default: jira)
  GUIDELLM__MP_CONTEXT_TYPE             Multiprocessing context (default: spawn on macOS)
  HF_TOKEN                              Needed for gated Hugging Face tokenizers

Examples:
  $0 quick
  $0 sweep
  BENCH_MCP_SERVER_URL=https://jira-mcp.example/sse $0 mcp-jira
  BENCH_TOKENIZER=meta-llama/Llama-3.2-3B-Instruct $0 quick
EOF
}

python_bin() {
  if command -v python3.12 >/dev/null 2>&1; then
    echo python3.12
  elif command -v python3.11 >/dev/null 2>&1; then
    echo python3.11
  elif command -v python3.10 >/dev/null 2>&1; then
    echo python3.10
  else
    echo "GuideLLM requires Python >= 3.10 (prefer 3.12)." >&2
    exit 1
  fi
}

ensure_venv() {
  local py
  py="$(python_bin)"
  if [[ -d "$VENV_DIR" ]]; then
    local ver
    ver="$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo 0)"
    if [[ "$(printf '%s\n' "$ver" "3.10" | sort -V | head -n1)" != "3.10" ]]; then
      echo "Recreating benchmarks venv (found Python ${ver}; need >= 3.10)..."
      rm -rf "$VENV_DIR"
    fi
  fi
  if [[ ! -d "$VENV_DIR" ]]; then
    "$py" -m venv "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install -U pip
    pip install -r "$BENCH_DIR/requirements.txt"
  else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
  fi
}

# Drop helper env vars that GuideLLM would warn about (it only understands GUIDELLM__).
scrub_helper_env() {
  unset GUIDELLM_TARGET GUIDELLM_MODEL GUIDELLM_TOKENIZER GUIDELLM_API_KEY \
    GUIDELLM_PROFILE GUIDELLM_DURATION GUIDELLM_PROMPT_TOKENS \
    GUIDELLM_OUTPUT_TOKENS GUIDELLM_MAX_REQUESTS \
    BENCH_TARGET BENCH_MODEL BENCH_TOKENIZER BENCH_API_KEY \
    BENCH_DURATION BENCH_PROMPT_TOKENS BENCH_OUTPUT_TOKENS BENCH_MAX_REQUESTS \
    BENCH_MCP_SERVER_URL BENCH_MCP_SERVER_LABEL BENCH_TIMEOUT MCP_SERVER_URL \
    2>/dev/null || true
}

backend_arg() {
  # Llama Stack readiness is /v1/health (not /health). Remap GuideLLM's probe.
  # http2=false avoids quirks with some OpenShift routes.
  local request_format="${1:-/v1/chat/completions}"
  local stream="${2:-true}"
  local timeout_part=""
  if [[ -n "${3:-}" ]]; then
    timeout_part=",timeout=${3}"
  fi
  local backend="kind=openai_http,target=${TARGET},model=${MODEL},request_format=${request_format},stream=${stream},http2=false,api_routes./health=v1/health${timeout_part}"
  if [[ -n "$API_KEY" ]]; then
    backend+=",api_key=${API_KEY}"
  fi
  printf '%s' "$backend"
}

mcp_backend_json() {
  # JSON backend so we can inject Responses tools=[{type:mcp,...}].
  python3 - <<'PY' "$TARGET" "$MODEL" "$API_KEY" "$MCP_URL" "$MCP_LABEL" "$MCP_TIMEOUT"
import json, sys
target, model, api_key, mcp_url, mcp_label, timeout = sys.argv[1:7]
cfg = {
    "kind": "openai_http",
    "target": target,
    "model": model,
    "request_format": "/v1/responses",
    "stream": False,
    "http2": False,
    "timeout": float(timeout),
    "api_routes": {"/health": "v1/health"},
    "extras": {
        "body": {
            "ignore_eos": False,
            "tool_choice": "auto",
            "tools": [
                {
                    "type": "mcp",
                    "server_label": mcp_label,
                    "server_url": mcp_url,
                    "require_approval": "never",
                }
            ],
        }
    },
}
if api_key:
    cfg["api_key"] = api_key
print(json.dumps(cfg))
PY
}

constraint_args() {
  if [[ -n "$MAX_REQUESTS" ]]; then
    printf -- '--constraint kind=max_requests,count=%s' "$MAX_REQUESTS"
  else
    printf -- '--constraint kind=max_duration,seconds=%s' "$DURATION"
  fi
}

run_guidellm() {
  mkdir -p "$RESULTS_DIR"
  local stamp
  stamp="$(date +%Y%m%d-%H%M%S)"
  local name="$1"
  shift

  echo "GuideLLM → ${TARGET}/v1/chat/completions"
  echo "Model:     ${MODEL}"
  echo "Tokenizer: ${TOKENIZER}"
  echo "MP context:${GUIDELLM__MP_CONTEXT_TYPE:-default}"
  echo "Scenario:  ${name}"
  echo

  local backend
  backend="$(backend_arg)"
  scrub_helper_env
  guidellm run \
    --backend "${backend}" \
    --tokenizer "kind=huggingface_auto,model=${TOKENIZER}" \
    --data "kind=synthetic_text,prompt_tokens=${PROMPT_TOKENS},output_tokens=${OUTPUT_TOKENS}" \
    --output "kind=json,path=${RESULTS_DIR}/${name}-${stamp}.json" \
    --output "kind=csv,path=${RESULTS_DIR}/${name}-${stamp}.csv" \
    "$@"
}

run_mcp_jira() {
  if [[ -z "$MCP_URL" ]]; then
    echo "BENCH_MCP_SERVER_URL is required for mcp-jira." >&2
    echo "Example: BENCH_MCP_SERVER_URL=https://jira-mcp.example/sse $0 mcp-jira" >&2
    echo "The URL must be reachable from the Llama Stack pod/host." >&2
    exit 2
  fi

  mkdir -p "$RESULTS_DIR"
  local stamp
  stamp="$(date +%Y%m%d-%H%M%S)"
  local backend
  backend="$(mcp_backend_json)"

  echo "GuideLLM → ${TARGET}/v1/responses  (LLM + MCP)"
  echo "Model:     ${MODEL}"
  echo "MCP:       ${MCP_LABEL} → ${MCP_URL}"
  echo "Dataset:   benchmarks/datasets/jira_list_sup.jsonl"
  echo "Scenario:  mcp-jira"
  echo

  scrub_helper_env
  # Run from repo root so relative dataset paths in scenarios resolve.
  cd "$ROOT"
  guidellm run \
    --backend "${backend}" \
    --tokenizer "kind=huggingface_auto,model=${TOKENIZER}" \
    --data "kind=json_file,path=${BENCH_DIR}/datasets/jira_list_sup.jsonl" \
    --profile kind=synchronous \
    --constraint kind=max_requests,count="${MAX_REQUESTS:-5}" \
    --output "kind=json,path=${RESULTS_DIR}/mcp-jira-${stamp}.json" \
    --output "kind=csv,path=${RESULTS_DIR}/mcp-jira-${stamp}.csv"
}

case "${SCENARIO}" in
  -h|--help|help)
    usage
    exit 0
    ;;
  raw)
    shift || true
    ensure_venv
    scrub_helper_env
    exec guidellm run "$@"
    ;;
  quick)
    ensure_venv
    run_guidellm quick \
      --profile kind=synchronous \
      --constraint kind=max_requests,count="${MAX_REQUESTS:-20}"
    ;;
  sweep)
    ensure_venv
    # shellcheck disable=SC2046
    run_guidellm sweep \
      --profile kind=sweep,sweep_size=5 \
      $(constraint_args)
    ;;
  concurrent)
    ensure_venv
    # shellcheck disable=SC2046
    run_guidellm concurrent \
      --profile kind=concurrent,streams=4 \
      $(constraint_args)
    ;;
  throughput)
    ensure_venv
    # shellcheck disable=SC2046
    run_guidellm throughput \
      --profile kind=throughput,max_concurrency=8 \
      $(constraint_args)
    ;;
  mcp-jira|mcp_jira)
    ensure_venv
    run_mcp_jira
    ;;
  *)
    echo "Unknown scenario: ${SCENARIO}" >&2
    usage >&2
    exit 1
    ;;
esac
