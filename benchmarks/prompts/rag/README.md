# RAG + MCP benchmark prompts

| File | Role |
|------|------|
| `system.txt` | System / instructions (RAG + MCP) |
| `user.txt.example` | Example user prompt |
| `user.txt` | **Edit this** before each run (gitignored) |

The `rag` scenario injects **both** tools:

```json
[
  {"type": "file_search", "vector_store_ids": ["vs_…"]},
  {"type": "mcp", "server_label": "jira", "server_url": "…"}
]
```

## Run

```bash
export BENCH_VECTOR_STORE_IDS=vs_your_store_id
$EDITOR benchmarks/prompts/rag/user.txt
./scripts/benchmark_rag.sh --requests 1
```

Metrics include `rag_time_s`, `mcp_time_s`, TTFT, ITL, and token counts.
