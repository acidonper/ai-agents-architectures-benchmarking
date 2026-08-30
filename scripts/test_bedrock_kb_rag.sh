#!/usr/bin/env bash
# Retrieve content from Amazon Bedrock Knowledge Base (native RAG — not MCP).
#
# Uses bedrock-agent-runtime Retrieve with Titan embeddings configured on the KB.
#
# Usage:
#   source scripts/load_bedrock_agentic_env.sh
#   ./scripts/test_bedrock_kb_rag.sh
#   ./scripts/test_bedrock_kb_rag.sh "How do I create a Jira issue?"
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck disable=SC1091
source "$ROOT/scripts/load_bedrock_agentic_env.sh"

QUERY="${1:-What information is in the knowledge base?}"

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

exec "$PY" - "$QUERY" <<'PY'
import json
import sys

from benchmarks.bedrock_client import (
    format_kb_retrieval_context,
    resolve_kb_embedding_model,
    resolve_knowledge_base_api_id,
    resolve_knowledge_base_name,
    retrieve_knowledge_base,
)

query = sys.argv[1]
kb_name = resolve_knowledge_base_name()
kb_id = resolve_knowledge_base_api_id()
emb = resolve_kb_embedding_model() or "amazon.titan-embed-text-v2:0"

print(f"KB name:      {kb_name or '<unset>'}")
print(f"KB API id:    {kb_id or '<unset>'}")
print(f"Embed model:  {emb}")
print(f"API:          bedrock-agent-runtime Retrieve")
print(f"Query:        {query}")
print()

chunks = retrieve_knowledge_base(query, number_of_results=5)
print(f"OK: {len(chunks)} passage(s)")
print()
print(format_kb_retrieval_context(chunks))
for i, chunk in enumerate(chunks):
    loc = chunk.get("location")
    if loc:
        print(f"\n[passage {i} location] {json.dumps(loc)[:400]}")
PY
