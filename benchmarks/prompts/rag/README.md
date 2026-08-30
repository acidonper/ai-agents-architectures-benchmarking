# RAG + MCP benchmark prompts

| File | Role |
|------|------|
| `system.txt` | System / instructions (RAG + MCP) |
| `user.txt.example` | Example user prompt |
| `user.txt` | **Edit this** before each run (gitignored) |

The `rag` scenario with `--target bedrock-runtime` uses:

1. **RAG** — Amazon Bedrock Knowledge Base **Retrieve** (not `file_search`)
2. **MCP** — AgentCore Runtime InvokeAgentRuntime (`jira_*` tools)
3. **LLM** — Bedrock Converse (`BENCH_MODEL_INFERENCE_PROFILE_ID`)

## Run

```bash
export BENCH_VECTOR_STORE_IDS=vs_your_store_id
$EDITOR benchmarks/prompts/rag/user.txt
./scripts/benchmark_rag.sh --requests 1
```

Metrics include `rag_time_s`, `mcp_time_s`, TTFT, ITL, and token counts.
