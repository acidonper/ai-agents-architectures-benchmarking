# StackChat — Llama Stack Chat Application

Full-stack chat app that talks to a [Llama Stack](https://github.com/llamastack/llama-stack) server using the **Responses API**. One request can orchestrate:

- **LLM** inference (any model registered in Llama Stack)
- **RAG** via the built-in `file_search` tool and vector stores
- **MCP** tool calling against configured Model Context Protocol servers

```
Browser (React)  →  FastAPI backend  →  Llama Stack (:8321)
                                         ├─ LLM provider
                                         ├─ Vector stores / file_search
                                         └─ MCP servers
```

## Project layout

```
ia/
├── backend/          # FastAPI + llama-stack-client
│   └── app/
│       ├── main.py
│       ├── routers/  # chat, models, rag, mcp, health
│       └── services/ # Llama Stack orchestration
└── frontend/         # React + Vite chat UI
```

## Prerequisites

- Python 3.11+ (3.12 recommended)
- Node.js 18+
- A running Llama Stack server (default `http://localhost:8321`)

## Backend setup

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: LLAMA_STACK_BASE_URL, DEFAULT_MODEL, DEFAULT_EMBEDDING_MODEL, optional MCP / vector stores
uvicorn app.main:app --reload --port 8000
```

Or from the repo root: `./scripts/dev.sh backend`

API docs: http://localhost:8000/docs

### Key endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/health` | Backend + Llama Stack reachability |
| `GET` | `/api/config` | Public defaults (model, embedding, vector stores, MCP) |
| `GET` | `/api/models?model_type=llm\|embedding\|all` | List models from Llama Stack |
| `POST` | `/api/chat` | Non-streaming chat (Responses API) |
| `POST` | `/api/chat/stream` | SSE streaming chat |
| `GET/POST` | `/api/rag/vector-stores` | List / create vector stores (create requires embedding model) |
| `POST` | `/api/rag/vector-stores/{id}/files` | Upload a document for RAG |
| `GET` | `/api/mcp/defaults` | Default MCP servers from env |

### Example chat payload

```json
{
  "message": "Summarize our onboarding policy and check open tickets",
  "model": "meta-llama/Llama-3.2-3B-Instruct",
  "instructions": "Be concise. Prefer tools when helpful.",
  "enable_rag": true,
  "vector_store_ids": ["vs_abc123"],
  "enable_mcp": true,
  "mcp_servers": [
    {
      "server_label": "tickets",
      "server_url": "http://localhost:3000/sse"
    }
  ]
}
```

The backend turns that into a Llama Stack `responses.create` call with:

```python
tools=[
  {"type": "file_search", "vector_store_ids": ["vs_abc123"]},
  {"type": "mcp", "server_label": "tickets", "server_url": "http://localhost:3000/sse"},
]
```

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — Vite proxies `/api` to the backend on port 8000.

### UI features

- Model picker (LLM only) + system instructions + temperature
- **RAG**: pick embedding model + dimension, create vector store, upload docs, enable `file_search`
- Toggle **MCP**, add server label/URL pairs
- Streaming or non-streaming replies
- Tool-call chips when Llama Stack uses `file_search` / MCP

## RAG with embedding models

Llama Stack indexes uploaded files with a registered **embedding model**, stores vectors (e.g. Milvus), and retrieves chunks via the Responses API `file_search` tool.

On this playground Stack the embedding models are:

- `sentence-transformers/ibm-granite/granite-embedding-125m-english` (768d)
- `sentence-transformers/nomic-ai/nomic-embed-text-v1.5` (768d)

```bash
# backend/.env
DEFAULT_EMBEDDING_MODEL=sentence-transformers/ibm-granite/granite-embedding-125m-english
DEFAULT_EMBEDDING_DIMENSION=768
# optional: DEFAULT_VECTOR_STORE_PROVIDER=milvus
```

Create a store (UI or API):

```bash
curl -sS http://localhost:8000/api/rag/vector-stores \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "policies",
    "embedding_model": "sentence-transformers/ibm-granite/granite-embedding-125m-english",
    "embedding_dimension": 768
  }'
```

Then upload a file to `/api/rag/vector-stores/{id}/files`, enable RAG with that store id, and chat. StackChat injects:

```python
{"type": "file_search", "vector_store_ids": ["vs_…"]}
```

## Environment variables

See `backend/.env.example`:

| Variable | Description |
|----------|-------------|
| `LLAMA_STACK_BASE_URL` | Llama Stack URL (default `http://localhost:8321`) |
| `LLAMA_STACK_API_KEY` | Optional API key |
| `DEFAULT_MODEL` | Fallback inference LLM id |
| `DEFAULT_EMBEDDING_MODEL` | Default embedding model for new vector stores |
| `DEFAULT_EMBEDDING_DIMENSION` | Embedding size (e.g. `768`) |
| `DEFAULT_VECTOR_STORE_PROVIDER` | Optional vector IO provider (`milvus`, …) |
| `DEFAULT_MCP_SERVERS` | JSON array of MCP server configs |
| `DEFAULT_VECTOR_STORE_IDS` | Comma-separated vector store ids used when RAG is on |
| `CORS_ORIGINS` | Allowed frontend origins |

## Typical workflow

1. Start Llama Stack with an inference provider, an **embedding model**, and vector IO (e.g. Milvus).
2. Start the backend and frontend.
3. In the UI, pick an LLM.
4. For RAG: choose an embedding model, create a vector store, upload docs, enable RAG.
5. For MCP: add your MCP SSE URL, enable MCP.
6. Chat — Llama Stack orchestrates tool discovery, retrieval, and final answer.

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

Phoenix judge defaults to **Amazon Bedrock** (`LLM(provider="bedrock")`, same inference
profile as the agent: `BENCH_MODEL_INFERENCE_PROFILE_ID`). The retired Llama Stack
playground URL is ignored.

```bash
# Default judge = Bedrock Nova (from backend/.env.bedrock)
./scripts/benchmark_mcp_jira.sh --target llama-stack --requests 3

# Override judge model (still Bedrock)
export PHOENIX_JUDGE_MODEL=eu.amazon.nova-2-lite-v1:0

# Optional: OpenAI-compatible judge on a live Llama Stack (not the old playground)
export PHOENIX_JUDGE_PROVIDER=llama-stack
export PHOENIX_JUDGE_BASE_URL=http://localhost:8321
export PHOENIX_JUDGE_MODEL=vllm-inference-2/your-judge-model
```

Or source `script.sh`. The judge does not call OpenAI.com.
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

### Bedrock AgentCore agentic benchmark

Default stack (`--target bedrock-runtime`):

```
LLM     Bedrock Converse  BENCH_MODEL_INFERENCE_PROFILE_ID (e.g. eu.amazon.nova-2-lite-v1:0)
MCP     AgentCore Runtime InvokeAgentRuntime JSON-RPC (Cognito Bearer)
RAG     bedrock-agent-runtime Retrieve  BEDROCK_KNOWLEDGE_BASE_ID
```

| Component | How the bench uses it |
|-----------|------------------------|
| LLM | `bedrock-runtime` **Converse** with `BENCH_MODEL_INFERENCE_PROFILE_ID` |
| MCP (Jira) | Runtime URL `…/runtimes/<encoded-arn>/invocations?qualifier=DEFAULT` |
| RAG | Native **Retrieve** API (`--rag-mode bedrock_kb`) |

```bash
# LLM + Jira MCP via the AgentCore Runtime (no gateway /mcp)
source scripts/load_bedrock_agentic_env.sh
./scripts/benchmark_bedrock_agentic.sh --target bedrock-runtime --requests 1

# RAG (KB Retrieve) + Runtime MCP + same LLM
./scripts/benchmark_bedrock_rag.sh --requests 1
```

`backend/.env.bedrock` must include:

```bash
BENCH_MODEL_INFERENCE_PROFILE_ID=eu.amazon.nova-2-lite-v1:0
BEDROCK_AGENT_RUNTIME_ARN=arn:aws:bedrock-agentcore:eu-north-1:ACCOUNT:runtime/mcp_atlassian_runtime-…
BEDROCK_KNOWLEDGE_BASE_ID=7PTWKIDPIA
BEDROCK_KB_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
BENCH_RAG_MODE=bedrock_kb
```

Legacy **gateway** path (`--target bedrock-gateway`): inference at `/inference/v1/responses` and MCP at `{gateway}/mcp`.

```bash
./scripts/benchmark_bedrock_agentic.sh --target bedrock-gateway --requests 1
```

Bearer auth to the gateway: `BEDROCK_AUTH_MODE=bearer` and `BEDROCK_GATEWAY_TOKEN=...`.

**Agentic gateway + Cognito** (matches AgentCore quick-start with `GW_AGENTIC_*` vars):

```bash
cp backend/.env.bedrock.example backend/.env.bedrock
# Edit GW_AGENTIC_CLIENT_ID, GW_AGENTIC_CLIENT_SECRET, GW_AGENTIC_COGNITO_DISCOVERY_URL, GW_AGENTIC_ID

source scripts/load_bedrock_agentic_env.sh   # sets BEDROCK_AUTH_MODE=cognito, fetches token at runtime
./test.sh                                    # or ./scripts/benchmark_bedrock_agentic.sh ...
```

The harness calls Cognito’s `token_endpoint` (from the OIDC discovery URL) with `client_credentials`, caches the access token, and sends `Authorization: Bearer …` to `{gateway}/inference/v1/responses`.

If token requests fail, set `GW_AGENTIC_OAUTH_SCOPE` or `BEDROCK_OAUTH_SCOPE` to the scope configured on your Cognito app client (not always the target path).

**Bedrock API key** (`ABSK…`): use `BEDROCK_AUTH_MODE=api-key` and `BEDROCK_API_KEY=...` when the gateway accepts that key directly (not Cognito ingress).

Phoenix judges use Amazon Bedrock (`PHOENIX_JUDGE_PROVIDER=bedrock`, model
`BENCH_MODEL_INFERENCE_PROFILE_ID`). Opt out with `PHOENIX_JUDGE_PROVIDER=llama-stack`
and a live `PHOENIX_JUDGE_BASE_URL` (the old playground host is ignored).

Results use the prefix `agentic-bedrock-*`.

## Notes

- Conversation continuity uses `previous_response_id` from the Responses API when available.
- If your Llama Stack client build does not support streaming on `responses.create`, the backend falls back to a single non-stream response over SSE.
- Ensure MCP servers are reachable from the **Llama Stack** host, not only from the browser.
