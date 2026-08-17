# rhoai Helm chart

This folder contains a Helm chart-like collection of Kubernetes and OpenShift manifests for deploying an RhoAI proof-of-concept stack.

> Note: This chart currently does not include `Chart.yaml` or `values.yaml`, and the manifests are mostly static YAML files. The current repository structure is best used as a documented deployment manifest set, not a packaged Helm chart.

## Purpose

The chart deploys a small AI inference stack with the following pieces:

- A dedicated `acidonpe` namespace
- An Atlassian MCP server deployment, service, and secret
- A ConfigMap that registers the MCP server with the OpenShift Data Hub/MCP framework
- A KServe `InferenceService` and `ServingRuntime` for a vLLM-based model (`llama-32-fp8`)
- A LlamaStack distribution deployment and route
- A LlamaStack configmap for connecting inference and vector providers

## Chart contents

### templates/namespace.yaml

Creates the `acidonpe` namespace.

### templates/mcp/mcp-atlassian.yaml

Deploys the Atlassian MCP server and exposes it via a ClusterIP service.

Resources:
- `Secret` named `mcp-atlassian-env`
- `Deployment` named `mcp-atlassian`
- `Service` named `mcp-atlassian`

Important:
- This secret contains Atlassian credentials and a token in plain text. Replace these values with secure secrets before using in production.
- The service listens on port `80` and forwards to container port `9000`.

### templates/mcp/rhoai-config-mcp-atlassian.yaml

Creates a `ConfigMap` named `gen-ai-aa-mcp-servers` in the `redhat-ods-applications` namespace.

This register the MCP server for Atlasian integration:
- `url`: `http://mcp-atlassian.acidonpe.svc.cluster.local/sse`
- `transport`: `sse`
- `description`: `An MCP server for managing Atlassian Suite`

### templates/llm/inference-service.yaml

Deploys a KServe inference stack for `llama-32-fp8`.

Resources:
- `InferenceService` (`serving.kserve.io/v1beta1`)
- `ServingRuntime` (`serving.kserve.io/v1alpha1`)
- `Route` for external access

Important values:
- GPU resources: requests/limits for `nvidia.com/gpu: 1`
- Model runtime: `llama-32-fp8`
- Storage URI: `oci://quay.io/rhoai-genaiops/llama32-3b-instruct-fp8:latest`
- Namespace: `acidonpe`

### templates/ogx/configmap.yaml

Creates the LlamaStack configuration used by the LlamaStack distribution.

Key configuration:
- vLLM inference provider base_url: `http://llama-32-fp8-predictor.acidonpe.svc.cluster.local:8080/v1`
- Default embedding model: `ibm-granite/granite-embedding-125m-english`
- Local SQLite persistence stores under `/opt/app-root/src/.llama/distributions/rh`
- Tool runtime providers, local file search, vector store, and response persistence

### templates/ogx/LlamaStackDistribution.yaml

Deploys a `LlamaStackDistribution` with a route.

Resources:
- `LlamaStackDistribution` named `lsd-genai-playground`
- `ConfigMap` named `llama-stack-config`
- `Route` named `lsd-genai-playground-service`

The distribution is configured to run inside the `acidonpe` namespace.

### templates/ogx/LlamaStackDistribution copy.yaml

A duplicate/test variant of the distribution with:
- Namespace `test`
- `exposeRoute: true`
- A different LlamaStack config provider base URL

This file appears to be a sample or alternate deployment and should be used carefully.

## Requirements

- OpenShift cluster with route support
- KServe operator installed and CRDs available
- LlamaStack operator installed and CRDs available
- Namespace `redhat-ods-applications` present for MCP config
- GPU node(s) available if using the GPU inference configuration
- Complete <YOUR_JIRA_API_TOKEN> in mcp-atlassian.yaml file.

## Usage

Because the chart does not include `Chart.yaml`, the simplest usage is direct `kubectl`/`oc` apply of the templates.

Example:

```bash
oc apply -f charts/rhoai/templates/namespace.yaml
oc apply -f charts/rhoai/templates/mcp/mcp-atlassian.yaml # (Complete <YOUR_JIRA_API_TOKEN> before)
oc apply -f charts/rhoai/templates/mcp/rhoai-config-mcp-atlassian.yaml
oc apply -f charts/rhoai/templates/llm/inference-service.yaml
oc apply -f charts/rhoai/templates/ogx/configmap.yaml
oc apply -f charts/rhoai/templates/ogx/LlamaStackDistribution.yaml
```

## Customization

Replace the following values for your environment:

- `namespace` in all templates if you want a different deployment namespace
- Atlassian credentials and API token in `mcp-atlassian.yaml`
- MCP server URL in `rhoai-config-mcp-atlassian.yaml`
- Model image and resources in `inference-service.yaml`
- LlamaStack provider endpoints and secrets in `configmap.yaml`
- Route names and namespace values in `LlamaStackDistribution.yaml`

## Notes

- This repository folder is not a fully packaged Helm chart yet. It contains static manifests under `templates/`.
- A proper Helm chart should add:
  - `Chart.yaml`
  - `values.yaml`
  - optional `.helmignore`
  - Helm template expressions for environment-specific values
- Sensitive values are present in plain text and should not be committed in a production deployment.
- If you want, I can also scaffold a complete Helm chart with parameterized values and chart metadata.
