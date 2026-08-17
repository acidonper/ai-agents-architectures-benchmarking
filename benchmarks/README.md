# Benchmark

## Benchmark model performance (GuideLLM)

Use [GuideLLM](https://github.com/vllm-project/guidellm) from the terminal to measure latency and throughput of the **inference model** behind Llama Stack.

**You do not need the StackChat FastAPI backend for this.** GuideLLM generates its own synthetic HTTP traffic and expects an OpenAI-compatible server. Point it at Llama Stack’s `/v1/chat/completions` API (the same model path the chat app uses, without RAG/MCP orchestration overhead).

```
GuideLLM (CLI)  →  Llama Stack /v1/chat/completions  →  vLLM / provider
```

```bash
# Reads LLAMA_STACK_BASE_URL from backend/.env when present
./scripts/benchmark_guidellm.sh quick        # ~20 sync requests (smoke)
./scripts/benchmark_guidellm.sh sweep        # capacity sweep (~minutes)
./scripts/benchmark_guidellm.sh concurrent   # fixed concurrency
./scripts/benchmark_guidellm.sh throughput   # peak throughput
```

Override target/model/tokenizer:

```bash
BENCH_TARGET=https://your-llama-stack.example.com \
BENCH_MODEL=vllm-inference-1/llama-32-fp8 \
BENCH_TOKENIZER=meta-llama/Llama-3.2-3B-Instruct \
./scripts/benchmark_guidellm.sh quick
```

Or call GuideLLM directly:

```bash
cd benchmarks && python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# On macOS, spawn avoids fork/ObjC segfaults with torch:
export GUIDELLM__MP_CONTEXT_TYPE=spawn

guidellm run \
  --backend kind=openai_http,target="$LLAMA_STACK_BASE_URL",model=vllm-inference-1/llama-32-fp8,request_format=/v1/chat/completions,stream=true,http2=false,api_routes./health=v1/health \
  --tokenizer kind=huggingface_auto,model=gpt2 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=20 \
  --output kind=json,path=results/run.json
```

Results land in `benchmarks/results/`. Scenario YAML examples live under `benchmarks/scenarios/`.

Notes:
- GuideLLM’s default health probe is `/health`; Llama Stack serves it at `/v1/health`, so the script remaps that route.
- Default tokenizer is `gpt2` (public). For Llama-accurate token counts set `BENCH_TOKENIZER=meta-llama/Llama-3.2-3B-Instruct` (may need `HF_TOKEN`).
- On macOS the script sets `GUIDELLM__MP_CONTEXT_TYPE=spawn` automatically.

### Agentic benchmark: LLM + Jira MCP

The synthetic `quick`/`sweep` scenarios only hit `/v1/chat/completions` (no tools). To measure **“list Jira issues in project SUP”** you need the Responses API with an MCP tool:

```
GuideLLM / agentic harness  →  Llama Stack /v1/responses + tools[mcp]  →  Jira MCP  →  Jira
```

1. Set a Jira MCP URL that **Llama Stack can reach** (cluster DNS / public URL, not only localhost on your laptop):

```bash
export BENCH_MCP_SERVER_URL='https://your-jira-mcp.example/sse'   # or /mcp
export BENCH_MCP_SERVER_LABEL=jira
```

2. GuideLLM load test (latency/throughput with MCP tools injected):

```bash
./scripts/benchmark_guidellm.sh mcp-jira
```

3. Agentic harness with **Phoenix semantic + trajectory evals** (default on):

```bash
# Reports orchestration_time, semantic_success_rate, trajectory_success_rate,
# combined_success_rate. Opens a local Phoenix UI when PHOENIX_COLLECTOR_ENDPOINT
# is unset.
./scripts/benchmark_mcp_jira.sh --target llama-stack --requests 3

# Through StackChat FastAPI
./scripts/benchmark_mcp_jira.sh --target stackchat --requests 3

# Disable Phoenix judges
BENCH_PHOENIX=0 ./scripts/benchmark_mcp_jira.sh --no-phoenix --requests 3
```

Phoenix judge model defaults to a **Llama Stack** LLM (OpenAI-compatible `/v1`).
It prefers a model different from the agent under test when more than one LLM is registered.

```bash
# Agent model (tool-calling run)
export BENCH_MODEL=vllm-inference-1/llama-32-fp8

# Dedicated judge on Llama Stack (recommended once you register a 2nd LLM)
export PHOENIX_JUDGE_BASE_URL="$LLAMA_STACK_BASE_URL"
export PHOENIX_JUDGE_MODEL=vllm-inference-2/your-judge-model

./scripts/benchmark_mcp_jira.sh --target llama-stack --requests 3 \
  --judge-model "$PHOENIX_JUDGE_MODEL"
```

Or source `script.sh` after setting `PHOENIX_JUDGE_MODEL`. OpenAI is not used for judging.
| Metric | Meaning |
|--------|---------|
| `orchestration_time_s` | End-to-end wall clock for one LLM+MCP turn |
| `ttft_s` | Time to first streamed output token/delta (requires `--stream`, default on) |
| `itl_s` | Inter-token latency (mean; MCP tool waits >2s excluded from pairwise gaps) |
| `input_tokens` / `output_tokens` / `total_tokens` | From API `usage` when present; else `output_tokens_estimated` |
| `tokens_per_second` | Output tokens ÷ generation span (first→last text delta) |
| `semantic_success_rate` | Phoenix LLM-judge: answer fulfills the Jira task |
| `trajectory_success_rate` | Phoenix LLM-judge: tool path is sensible vs reference |
| `combined_success_rate` | HTTP ok ∧ semantic correct ∧ trajectory correct |

Disable streaming (no TTFT/ITL): `--no-stream` or `BENCH_STREAM=0`.

Each run also writes a dedicated Jira/MCP payload file:

`benchmarks/results/agentic-jira-<stamp>-jira-responses.json`

It contains per-request `mcp_tool_responses` (raw tool outputs) and parsed `jira_issues` (key/status/summary). The main report JSON references it via `jira_responses_file`.

Dataset prompts live in `benchmarks/datasets/jira_list_sup.jsonl`. Results JSON is written under `benchmarks/results/`.

### Create-issue bench (prompt files)

Creates a real Jira issue via LLM + MCP. System and user prompts are file-based:

| File | Role |
|------|------|
| `benchmarks/prompts/jira_create/system.txt` | System / instructions (checked in) |
| `benchmarks/prompts/jira_create/user.txt.example` | Template with `<PLACEHOLDER>`s |
| `benchmarks/prompts/jira_create/user.txt` | **Fill this** before each run (gitignored) |

```bash
cp benchmarks/prompts/jira_create/user.txt.example \
   benchmarks/prompts/jira_create/user.txt
$EDITOR benchmarks/prompts/jira_create/user.txt   # replace every <PLACEHOLDER>

# Creates a real issue — start with --requests 1
./scripts/benchmark_mcp_jira_create.sh --requests 1
```

The harness rejects unfilled `<PLACEHOLDER>` values. Results use the prefix
`agentic-jira-create-<stamp>`. Phoenix judges use create-specific semantic/trajectory prompts.

### RAG + MCP bench (file_search + Jira)

Runs Llama Stack Responses with **both** tools:

```json
[
  {"type": "file_search", "vector_store_ids": ["vs_…"]},
  {"type": "mcp", "server_label": "jira", "server_url": "…"}
]
```

Typical flow: retrieve company procedure from Milvus → create/list Jira issues via MCP.

```bash
export BENCH_VECTOR_STORE_IDS=vs_your_store_id
# optional: BENCH_MCP_SERVER_URL=http://…/sse

$EDITOR benchmarks/prompts/rag/user.txt
./scripts/benchmark_rag.sh --requests 1
```

| Metric | Meaning |
|--------|---------|
| `rag_time_s` | Wall time in file_search (`searching`→`completed`) |
| `time_to_rag_s` | Request start → first file_search event |
| `mcp_time_s` | Wall time in mcp_call (`in_progress`→`completed`) |
| `time_to_mcp_s` | Request start → first mcp_call event |
| `ttft_s` | Time to first output token |
| `itl_s` / tokens | Same as other agentic benches |

Results: `benchmarks/results/agentic-rag-<stamp>.json`, plus `*-rag-responses.json` and `*-jira-responses.json`.
