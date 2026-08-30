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

### `/assets` - Notebooks & Configuration
- **`milvus-onboard.ipynb`** - Jupyter notebook for Milvus setup and testing
- **`system_prompt`** - Default system prompt for AI agents

### `/ogx` - Infrastructure & Deployment
Complete Kubernetes/OpenShift deployment stack with vector database, LLM inference, and Llama Stack orchestration.

- **`milvus/`** - Helm chart for Milvus vector database (RAG/file_search backend)
- **`rhoai/`** - RHOAI/OpenShift AI stack with:
  - LLM inference service (vLLM-based, KServe-managed)
  - MCP Atlassian server (Jira tool integration)
  - Llama Stack distribution (agent orchestration engine)
  - Configuration for connecting all components

**Quick Start:** See [ogx/README.md](ogx/README.md) for detailed deployment instructions, configuration options, and troubleshooting.

### `/aws` - AWS Bedrock Deployment
AWS Bedrock integration for cloud-based AI agent architecture.

- **`Dockerfile`** - Container image for MCP server compatible with AWS AgentCore (port 8000)
- **`README.md`** - Comprehensive guide for:
  - Building and pushing MCP Docker image to ECR
  - Configuring Bedrock AgentCore runtime
  - Setting up Knowledge Bases with RAG
  - Integrating LLM services
  - Troubleshooting deployment issues

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