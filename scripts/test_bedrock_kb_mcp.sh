#!/usr/bin/env bash
# Deprecated alias — KB RAG uses Bedrock Retrieve API, not MCP.
exec "$(dirname "$0")/test_bedrock_kb_rag.sh" "$@"
