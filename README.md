# AI Agents Architectures Benchmarking

Comprehensive platform for building, deploying, and benchmarking AI agent architectures with LLM inference, RAG, and MCP tool integration.

## Project Structure

### `/apps` - Application Stack
- **`backend/`** - FastAPI backend orchestrating Llama Stack
  - Chat API with multi-turn conversation support
  - RAG (retrieval-augmented generation) endpoints
  - MCP (Model Context Protocol) tool integration
  - Built with FastAPI & Uvicorn
  
- **`frontend/`** - React/TypeScript UI
  - Real-time chat interface (ChatWindow, Composer, MessageBubble)
  - Sidebar for conversation management
  - Built with Vite, React 19, TypeScript

### `/benchmarks` - Performance & Evaluation
- **`agentic_bench.py`** - Main benchmark suite for LLM + MCP latency analysis
  - Scenarios: Jira list/create, RAG, combined operations
  - Phoenix semantic/trajectory evaluation support
  - OTEL metrics export
- **`phoenix_semantic.py`** - Phoenix-based semantic evaluation
- **`datasets/`** - Test data (Jira issues in JSONL format)
- **`prompts/`** - System/user prompts for scenarios (jira_create, rag)
- **`results/`** - Benchmark output results (JSON)

### `/charts` - Infrastructure & Deployment
- OGX
  - **`milvus/`** - Helm chart for Milvus vector database
  - **`rhoai/`** - RHOAI/OpenShift deployment configurations
    - LLM inference service
    - MCP Atlassian integration
    - Llama Stack distribution configs

### `/scripts` - Utility & Setup Scripts
- `dev.sh` - Development environment setup
- `benchmark_*.sh` - Scripts to run specific benchmarks (RAG, MCP Jira, etc.)
- `upload_doc.sh` - Document upload utility

### `/informs` - Results & Artifacts
- Benchmark outputs and evaluation results from previous runs

## Quick Start

### Backend
```bash
cd apps/backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
# API docs at http://localhost:8000/docs
```

### Frontend
```bash
cd apps/frontend
npm install
npm run dev
# UI at http://localhost:5173
```

### Benchmarking
```bash
cd benchmarks
pip install -r requirements.txt
export BENCH_VECTOR_STORE_IDS=vs_f0f99ac9-0904-43a5-94e0-6fefd6305bf4                                                       
./scripts/benchmark_rag.sh --requests 50
```

## Key Technologies
- **Backend**: FastAPI & Uvicorn
- **Frontend**: React 19, TypeScript & Vite
- **Infrastructure**:
  - OGX: Kubernetes, OpenShift/RHOAI, Milvus, OGX (LlamaStack) & vLLM
- **Evaluation**: GuideLLM, Phoenix (semantic eval) & OTEL metrics
- **Tools**: Jira MCP integration, file_search RAG & multi-turn conversations

## Roadmap

- Integrate AWS Amazon Bedrock
- Integrate Google Vertex AI

## Author 

Asier Cidon