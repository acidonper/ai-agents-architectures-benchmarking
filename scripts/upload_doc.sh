#!/usr/bin/env bash
# Upload a document into a Llama Stack vector store (Milvus-backed).
#
# Usage:
#   ./scripts/upload_doc.sh [path-to-file] [vector-store-id]
#
# Defaults:
#   file  = benchmarks/assests/TFM_RAG.pdf
#   store = DEFAULT_VECTOR_STORE_IDS / BENCH_VECTOR_STORE_IDS from backend/.env
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$ROOT/backend/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/backend/.env"
  set +a
fi

BASE="${LLAMA_STACK_BASE_URL:-http://lsd-genai-playground-service-acidonpe.apps.ocp.zd4ms.sandbox2306.opentlc.com}"
PDF="${1:-$ROOT/benchmarks/assests/TFM_RAG.pdf}"
VS="${2:-${BENCH_VECTOR_STORE_IDS:-${DEFAULT_VECTOR_STORE_IDS:-}}}"
VS="${VS%%,*}"  # first id if comma-separated

if [[ -z "$VS" ]]; then
  echo "No vector store id. Pass as arg2 or set DEFAULT_VECTOR_STORE_IDS." >&2
  exit 2
fi
if [[ ! -f "$PDF" ]]; then
  echo "File not found: $PDF" >&2
  exit 2
fi

echo "Stack:  $BASE"
echo "Store:  $VS"
echo "File:   $PDF"

curl -sS "$BASE/v1/vector_stores/$VS" | python3 -c "
import sys, json
d = json.load(sys.stdin)
meta = d.get('metadata') or {}
print('provider:', meta.get('provider_id'))
print('embedding:', meta.get('embedding_model'))
print('files before:', d.get('file_counts'))
"

FILE_JSON=$(curl -sS -m 180 -F "file=@${PDF}" -F "purpose=assistants" "$BASE/v1/files")
echo "uploaded file: $FILE_JSON"
FID=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['id'])" "$FILE_JSON")

ATTACH=$(curl -sS -m 300 -H 'Content-Type: application/json' \
  -d "{\"file_id\":\"$FID\"}" \
  "$BASE/v1/vector_stores/$VS/files")
echo "attached: $ATTACH"

for i in $(seq 1 90); do
  STATUS_JSON=$(curl -sS "$BASE/v1/vector_stores/$VS/files/$FID")
  STATUS=$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('status',''))" "$STATUS_JSON")
  echo "index poll $i: $STATUS"
  case "$STATUS" in
    completed|failed|cancelled)
      echo "$STATUS_JSON" | python3 -m json.tool
      [[ "$STATUS" == "completed" ]] || exit 1
      break
      ;;
  esac
  sleep 2
done

curl -sS "$BASE/v1/vector_stores/$VS" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('files after:', d.get('file_counts'))
print('OK — document is in Milvus vector store', d.get('id'))
"
