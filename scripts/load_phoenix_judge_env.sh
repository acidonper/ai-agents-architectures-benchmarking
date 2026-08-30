#!/usr/bin/env bash
# Phoenix LLM-as-judge → Amazon Bedrock Converse (not the retired Llama Stack playground).
#
# Defaults:
#   PHOENIX_JUDGE_PROVIDER=bedrock
#   PHOENIX_JUDGE_MODEL=$BENCH_MODEL_INFERENCE_PROFILE_ID
#
# Opt out (OpenAI-compatible /v1 on a live Stack):
#   PHOENIX_JUDGE_PROVIDER=llama-stack
#   PHOENIX_JUDGE_BASE_URL=http://localhost:8321
#
# Usage: source from a bench script after ROOT is set (and after backend/.env if needed).

if [[ -z "${ROOT:-}" ]]; then
  if [[ -n "${BASH_VERSION:-}" ]]; then
    _pj_script="${BASH_SOURCE[0]}"
  elif [[ -n "${ZSH_VERSION:-}" ]]; then
    _pj_script="${(%):-%x}"
  else
    _pj_script="$0"
  fi
  ROOT="$(cd "$(dirname "$_pj_script")/.." && pwd)"
  unset _pj_script
fi

_pj_url="${PHOENIX_JUDGE_BASE_URL:-}"
if [[ "$_pj_url" == *"lsd-genai-playground"* || "$_pj_url" == *"api.openai.com"* ]]; then
  unset PHOENIX_JUDGE_BASE_URL
fi
unset _pj_url

# Load region + inference profile for the judge without clobbering a llama-stack BENCH_MODEL.
if [[ -f "$ROOT/backend/.env.bedrock" ]]; then
  _keep_bench_model="${BENCH_MODEL:-}"
  _keep_agentic_target="${BENCH_AGENTIC_TARGET:-}"
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/backend/.env.bedrock"
  set +a
  if [[ -n "$_keep_bench_model" ]]; then
    export BENCH_MODEL="$_keep_bench_model"
  fi
  if [[ -n "$_keep_agentic_target" ]]; then
    export BENCH_AGENTIC_TARGET="$_keep_agentic_target"
  fi
  unset _keep_bench_model _keep_agentic_target
fi

export PHOENIX_JUDGE_PROVIDER="${PHOENIX_JUDGE_PROVIDER:-bedrock}"
export BEDROCK_REGION="${BEDROCK_REGION:-${AWS_REGION:-eu-north-1}}"
export AWS_REGION="${AWS_REGION:-${BEDROCK_REGION}}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION}}"
export AWS_REGION_NAME="${AWS_REGION_NAME:-${AWS_REGION}}"
export PHOENIX_JUDGE_MODEL="${PHOENIX_JUDGE_MODEL:-${BENCH_MODEL_INFERENCE_PROFILE_ID:-eu.amazon.nova-2-lite-v1:0}}"
