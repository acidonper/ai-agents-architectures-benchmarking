#!/usr/bin/env bash
# Test Bedrock model inference (bedrock-runtime Converse API).
#
# Uses BENCH_MODEL or BENCH_MODEL_INFERENCE_PROFILE_ID from backend/.env.bedrock.
# EU models need inference profile ids (e.g. eu.amazon.nova-2-lite-v1:0).
#
# Usage:
#   source scripts/load_bedrock_agentic_env.sh
#   ./scripts/test_bedrock_inference.sh
#   ./scripts/test_bedrock_inference.sh "Explain RAG in one sentence."
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck disable=SC1091
source "$ROOT/scripts/load_bedrock_agentic_env.sh"

PROMPT="${1:-Say hello in one word.}"
MODEL="${BENCH_MODEL:-${BENCH_MODEL_INFERENCE_PROFILE_ID:-eu.amazon.nova-2-lite-v1:0}}"
REGION="${BEDROCK_REGION:-eu-north-1}"

VENV_DIR="$ROOT/benchmarks/.venv"
PY=python3.12
if ! command -v "$PY" >/dev/null 2>&1; then
  PY=python3
fi
if [[ -d "$VENV_DIR" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PY=python
fi

exec "$PY" - "$MODEL" "$REGION" "$PROMPT" <<'PY'
import sys
import boto3
from botocore.exceptions import ClientError

model_id, region, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
print(f"Region:  {region}")
print(f"Model:   {model_id}")
print(f"API:     bedrock-runtime converse")
print(f"Prompt:  {prompt}")
print()

client = boto3.client("bedrock-runtime", region_name=region)
try:
    resp = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )
except ClientError as exc:
    err = exc.response.get("Error", {})
    print(f"FAILED: {err.get('Code')}: {err.get('Message')}")
    sys.exit(1)

out = resp.get("output", {}).get("message", {}).get("content", [])
usage = resp.get("usage", {})
text_parts = [c.get("text", "") for c in out if isinstance(c, dict) and c.get("text")]
print("Response:")
print("\n".join(text_parts) or out)
print()
print(f"Usage: input={usage.get('inputTokens')} output={usage.get('outputTokens')} total={usage.get('totalTokens')}")
PY
