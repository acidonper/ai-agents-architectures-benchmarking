# OGX Infrastructure & Deployment

Kubernetes/OpenShift deployment manifests for a complete AI agent architecture stack, including Milvus vector database, LLM inference, MCP server integration, and Llama Stack distribution.

## Overview

This folder contains Helm-style Kubernetes manifests for deploying:
- **Milvus** — Open-source vector database for RAG retrieval
- **RHOAI** — Red Hat OpenShift AI stack with inference, MCP, and Llama Stack

```
┌──────────────────────────────────────────────────────┐
│         OpenShift/Kubernetes Cluster                 │
├──────────────────────────────────────────────────────┤
│ Namespace: acidonpe                                  │
│                                                      │
│  ┌─────────────────┐  ┌──────────────────┐           │
│  │   Milvus VectorDB   │  LLM Inference   │           │
│  │   (file_search)     │  (vLLM/KServe)   │           │
│  └─────────────────┘  └──────────────────┘           │
│         ↓                        ↓                    │
│  ┌──────────────────────────────────────┐            │
│  │     Llama Stack Distribution          │            │
│  │  (Responses API, tool runtime)        │            │
│  └──────────────────────────────────────┘            │
│         ↑                                             │
│  ┌─────────────────┐                                 │
│  │  MCP Atlassian  │  (Jira tools)                   │
│  │  (SSE transport)│                                 │
│  └─────────────────┘                                 │
└──────────────────────────────────────────────────────┘
```

## Project Structure

### `/milvus` — Vector Database Helm Chart

Milvus is an open-source vector database used for storing embeddings and performing semantic search during RAG operations.

**Contents:**
- `Chart.yaml` — Helm chart metadata
- `values.yaml` — Default Helm values for Milvus deployment
- `charts/` — Milvus dependency charts

**Key Configuration:**
- Namespace: `acidonpe`
- Replica count: Configurable (default 1 for dev)
- Storage: PersistentVolume (S3, NFS, or local)
- Embedding dimensions: 768 (default)

**Deployment:**
```bash
helm install milvus ./milvus -n acidonpe --create-namespace
```

**Verification:**
```bash
# Check pod status
kubectl get pods -n acidonpe -l app=milvus

# Port-forward for testing (if needed)
kubectl port-forward -n acidonpe svc/milvus 19530:19530
```

### `/rhoai` — RHOAI/OpenShift AI Stack

Red Hat OpenShift AI deployment configuration including LLM inference, MCP server, and Llama Stack orchestration.

**Contents:**
- `README.md` — Detailed RHOAI-specific documentation
- `templates/` — Kubernetes manifests organized by component

#### `/rhoai/templates/namespace.yaml`

Creates the dedicated `acidonpe` namespace for all AI agent workloads.

**Resources:**
- Namespace: `acidonpe`
- Labels: `app.kubernetes.io/name=ai-agents`

#### `/rhoai/templates/llm/inference-service.yaml`

Deploys an LLM inference service using KServe with vLLM backend.

**Resources:**
- `InferenceService` (KServe) — Managed model serving
- `ServingRuntime` — Runtime configuration for vLLM
- `Route` — External HTTPS endpoint

**Key Configuration:**
- Model: `llama-32-fp8` (3B parameter Llama 2)
- Framework: vLLM with FP8 quantization
- GPU Resources: 1× NVIDIA GPU (configurable)
- Memory: 8Gi request, 16Gi limit
- Storage URI: `oci://quay.io/rhoai-genaiops/llama32-3b-instruct-fp8:latest`

**Example Inference Call:**
```bash
# Get the InferenceService endpoint
ENDPOINT=$(oc get route -n acidonpe llama-32-fp8-service -o jsonpath='{.spec.host}')

# Send a request
curl -X POST https://$ENDPOINT/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Describe Jira integration",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

#### `/rhoai/templates/mcp/mcp-atlassian.yaml`

Deploys the Atlassian MCP server (Jira integration) for tool calling.

**Resources:**
- `Secret` (`mcp-atlassian-env`) — Jira credentials and configuration
- `Deployment` (`mcp-atlassian`) — MCP server pod
- `Service` (`mcp-atlassian`) — ClusterIP service on port 80 → 9000

**Environment Variables:**
- `JIRA_URL` — Jira instance URL (e.g., `https://asiercidon.atlassian.net`)
- `JIRA_USERNAME` — Jira user email
- `JIRA_API_TOKEN` — Jira API token
- `TOOLSETS` — `jira_issues` (enables issue listing and creation)
- `ALLOW_GLOBAL_CRED_FALLBACK` — `true` for credential fallback

**Transport:**
- Type: SSE (Server-Sent Events)
- Port: 9000 (internal)
- Exposed via Service on port 80

**Setup:**
1. Edit `mcp-atlassian.yaml` and replace:
   - `<YOUR_JIRA_API_TOKEN>` with your actual Jira token
2. Deploy:
   ```bash
   kubectl apply -f templates/mcp/mcp-atlassian.yaml
   ```
3. Verify:
   ```bash
   kubectl logs -n acidonpe deployment/mcp-atlassian -f
   # Should show: "SSE transport listening on :9000"
   ```

#### `/rhoai/templates/mcp/rhoai-config-mcp-atlassian.yaml`

Creates a ConfigMap that registers the MCP server with RHOAI/OpenShift Data Hub.

**Resources:**
- `ConfigMap` (`gen-ai-aa-mcp-servers`) in namespace `redhat-ods-applications`

**Configuration:**
```yaml
Atlassian-MCP-Server: |
  url: http://mcp-atlassian.acidonpe.svc.cluster.local/sse
  transport: sse
  description: An MCP server for managing Atlassian Suite
```

**Deployment:**
```bash
# Requires access to redhat-ods-applications namespace
kubectl apply -f templates/mcp/rhoai-config-mcp-atlassian.yaml
```

#### `/rhoai/templates/ogx/configmap.yaml`

Llama Stack configuration providing endpoints for inference, embeddings, vector stores, and file search.

**Key Settings:**
- **LLM Inference Provider:** vLLM at `http://llama-32-fp8-predictor.acidonpe.svc.cluster.local:8080/v1`
- **Embedding Model:** `sentence-transformers/ibm-granite/granite-embedding-125m-english` (768d)
- **Vector Store:** Milvus at `http://milvus.acidonpe.svc.cluster.local:19530`
- **File Search:** Local SQLite with Milvus backend
- **Tool Runtime:** MCP server discovery
- **Persistence:** SQLite under `/opt/app-root/src/.llama/distributions/rh`

**Deployment:**
```bash
kubectl apply -f templates/ogx/configmap.yaml
```

#### `/rhoai/templates/ogx/LlamaStackDistribution.yaml`

Deploys the Llama Stack distribution (orchestration engine) with exposed Route.

**Resources:**
- `LlamaStackDistribution` (`lsd-genai-playground`) — Llama Stack runtime
- `ConfigMap` (`llama-stack-config`) — Stack configuration
- `Route` (`lsd-genai-playground-service`) — HTTPS external endpoint

**Configuration:**
- Namespace: `acidonpe`
- Replicas: 1 (configurable)
- Expose Route: `true`

**Usage:**
```bash
# Get the external URL
LLAMA_STACK_URL=$(oc get route -n acidonpe lsd-genai-playground-service -o jsonpath='{.spec.host}')

# Test the health endpoint
curl https://$LLAMA_STACK_URL/api/health

# List available models
curl https://$LLAMA_STACK_URL/api/models
```

#### `/rhoai/templates/ogx/LlamaStackDistribution copy.yaml`

Alternate deployment variant (test/staging) in the `test` namespace. Use for non-production testing.

**Differences from main:**
- Namespace: `test` (separate from `acidonpe`)
- Route exposure: Can be toggled
- Config provider: Alternate URL (if needed)

## Deployment Instructions

### Prerequisites

- OpenShift/Kubernetes cluster (4.x or later)
- `oc` or `kubectl` CLI installed
- KServe operator installed (for inference)
- LlamaStack operator installed (for Llama Stack CRDs)
- Namespace `redhat-ods-applications` exists (RHOAI default)
- GPU node(s) available (for LLM inference)

### Full Stack Deployment

```bash
# 1. Create namespace
kubectl apply -f rhoai/templates/namespace.yaml

# 2. Deploy Milvus vector database
helm install milvus ./milvus -n acidonpe

# 3. Deploy LLM inference service
kubectl apply -f rhoai/templates/llm/inference-service.yaml

# 4. Deploy MCP Atlassian server (edit credentials first!)
# Edit: rhoai/templates/mcp/mcp-atlassian.yaml
#   - Replace <YOUR_JIRA_API_TOKEN>
kubectl apply -f rhoai/templates/mcp/mcp-atlassian.yaml

# 5. Register MCP with RHOAI
kubectl apply -f rhoai/templates/mcp/rhoai-config-mcp-atlassian.yaml

# 6. Deploy Llama Stack configuration
kubectl apply -f rhoai/templates/ogx/configmap.yaml

# 7. Deploy Llama Stack distribution
kubectl apply -f rhoai/templates/ogx/LlamaStackDistribution.yaml

# 8. Wait for all pods to be ready
kubectl wait --for=condition=Ready pod -l app=milvus -n acidonpe --timeout=5m
kubectl wait --for=condition=Ready pod -l app=mcp-atlassian -n acidonpe --timeout=5m
kubectl wait --for=condition=Ready pod -l app.kubernetes.io/name=lsd-genai-playground -n acidonpe --timeout=10m
```

### Verification Steps

```bash
# Check all pods are running
kubectl get pods -n acidonpe

# Verify inference service endpoint
oc get route -n acidonpe llama-32-fp8-service -o wide

# Verify Llama Stack endpoint
oc get route -n acidonpe lsd-genai-playground-service -o wide

# Check MCP server logs
kubectl logs -n acidonpe deployment/mcp-atlassian

# Test Llama Stack health
LLAMA_URL=$(oc get route -n acidonpe lsd-genai-playground-service -o jsonpath='{.spec.host}')
curl https://$LLAMA_URL/api/health
```

## Configuration Customization

### Update Jira Credentials

```bash
# 1. Edit the secret
kubectl edit secret mcp-atlassian-env -n acidonpe

# 2. Update Base64-encoded values (optional simpler approach):
kubectl patch secret mcp-atlassian-env -n acidonpe -p \
  '{"stringData":{"JIRA_API_TOKEN":"new-token-here"}}'

# 3. Force pod restart
kubectl rollout restart deployment/mcp-atlassian -n acidonpe
```

### Scale Inference Service

```bash
# Increase replicas
kubectl patch -n acidonpe knative-serving serving.knative.dev/llama-32-fp8 \
  --type='json' -p='[{"op": "replace", "path": "/spec/maxReplicas", "value":3}]'
```

### Change Embedding Model

```bash
# 1. Edit the ConfigMap
kubectl edit configmap llama-stack-config -n acidonpe

# 2. Update embedding_model field
# 3. Restart Llama Stack distribution
kubectl rollout restart deployment/lsd-genai-playground -n acidonpe
```

## Troubleshooting

### Inference Service Not Ready

```bash
# Check KServe conditions
kubectl describe isvc llama-32-fp8 -n acidonpe

# Check vLLM pod logs
kubectl logs -n acidonpe -l model=llama-32-fp8
```

### MCP Tool Discovery Fails

```bash
# Verify MCP pod is running
kubectl get pod -n acidonpe -l app=mcp-atlassian

# Check for credential errors
kubectl logs -n acidonpe deployment/mcp-atlassian | grep -i error

# Test Jira connectivity directly
kubectl exec -it pod/mcp-atlassian-xxxx -n acidonpe -- \
  curl -u $JIRA_USERNAME:$JIRA_API_TOKEN $JIRA_URL/rest/api/3/myself
```

### Milvus Connection Issues

```bash
# Check Milvus pod
kubectl get pod -n acidonpe -l app=milvus

# Test Milvus connectivity
kubectl run -it --rm milvus-test --image=python:3.9 -n acidonpe -- bash
# Inside the pod:
pip install pymilvus
python -c "from pymilvus import connections; connections.connect('default', host='milvus', port=19530)"
```

### Llama Stack Distribution Stuck

```bash
# Check pod status and events
kubectl describe pod -n acidonpe -l app.kubernetes.io/name=lsd-genai-playground

# Check logs
kubectl logs -n acidonpe deployment/lsd-genai-playground

# Force restart
kubectl rollout restart deployment/lsd-genai-playground -n acidonpe
```

## Storage & Persistence

### Milvus Data Persistence

- **Storage Class:** Configurable in `milvus/values.yaml`
- **Default:** Local storage (PVC)
- **For Production:** Use S3 or NFS storage backend

### Llama Stack Persistence

- **Location:** `/opt/app-root/src/.llama/distributions/rh/`
- **Storage:** PersistentVolume mounted to pod
- **Data Includes:** Vector store indices, model cache, conversation logs

## Security Considerations

⚠️ **Important:** The current manifests contain credentials in plain text:
- Jira API token in `mcp-atlassian.yaml`
- Potential secrets in ConfigMap

**For Production:**
1. Use Kubernetes Secrets instead of environment variables
2. Use OpenShift sealed-secrets or external secret operators
3. Restrict RBAC to service accounts
4. Enable network policies to isolate namespaces
5. Use TLS for all routes
6. Implement pod security policies

## Scaling & Performance

### Horizontal Scaling

```bash
# Scale inference replicas
kubectl scale deployment llama-32-fp8-predictor -n acidonpe --replicas=3

# Scale Llama Stack distribution
kubectl scale deployment lsd-genai-playground -n acidonpe --replicas=2
```

### Resource Limits

Default resource limits are set in manifests. Adjust based on:
- Available GPU memory (for inference)
- CPU/RAM available per node
- Expected concurrent user load

## Related Documentation

- [Milvus Documentation](https://milvus.io/docs)
- [OpenShift AI Documentation](https://docs.openshift.com/ai/)
- [KServe Documentation](https://kserve.github.io/website/)
- [Llama Stack Operator](https://github.com/llamastack/llama-stack)
- Local stack reference: [StackChat README](../apps/README.md)
- Benchmarking reference: [Benchmarks README](../benchmarks/README.md)
- AWS alternative: [AWS Bedrock README](../aws/README.md)
