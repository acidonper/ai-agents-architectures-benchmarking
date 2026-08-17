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

## Notes

- Conversation continuity uses `previous_response_id` from the Responses API when available.
- If your Llama Stack client build does not support streaming on `responses.create`, the backend falls back to a single non-stream response over SSE.
- Ensure MCP servers are reachable from the **Llama Stack** host, not only from the browser.
