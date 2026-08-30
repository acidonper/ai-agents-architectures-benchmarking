# AWS Bedrock Agent Architecture

This document describes the manual steps required to set up AWS Bedrock for implementing an AI agent-based architecture with Llama Stack integration, MCP (Model Context Protocol) for tool calling, and RAG (Retrieval-Augmented Generation) capabilities.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Step 1: MCP Docker Image Setup](#step-1-mcp-docker-image-setup)
4. [Step 2: AgentCore Runtime Configuration](#step-2-agentcore-runtime-configuration)
5. [Step 3: Bedrock Knowledge Base & RAG](#step-3-bedrock-knowledge-base--rag)
6. [Step 4: LLM Service Configuration](#step-4-llm-service-configuration)
7. [Testing & Verification](#testing--verification)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

- AWS Account with Bedrock, ECR, and AgentCore access
- AWS CLI and `podman` or `docker` installed locally
- ECR repository created in `eu-north-1` region (e.g., `282516655659.dkr.ecr.eu-north-1.amazonaws.com`)
- Jira credentials (username and API token) for MCP Atlassian server
- S3 bucket for Knowledge Base storage (if using RAG)
- PDF documents or knowledge base files ready for upload

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│         AWS Bedrock AgentCore Gateway                   │
│  (Orchestrates LLM, MCP tools, and RAG retrieval)       │
└────┬────────────────────────────────────────────────────┘
     │
     ├─→ MCP Runtime (Port 8000)
     │   ├─ Atlassian MCP Server (Jira)
     │   ├─ Tool discovery & execution
     │   └─ JWT-based authentication
     │
     ├─→ Knowledge Base (RAG)
     │   ├─ Vector Store (S3)
     │   ├─ Embedding Model (Titan)
     │   └─ Document Retrieval
     │
     └─→ LLM Service
         ├─ Model selection
         └─ Response generation
```

## Step 1: MCP Docker Image Setup

### Important: Port 8000 Requirement

Amazon Bedrock AgentCore Runtime **requires** port 8000 for MCP communication. If this port is not exposed in the container, AWS cannot integrate with the MCP server. The provided `Dockerfile` in the `aws/` folder must be used to build a compliant image.

### Build & Push MCP Image to ECR

Execute these commands from the `aws/` folder:

```bash
# 1. Authenticate with ECR
aws ecr get-login-password --region eu-north-1 | podman login \
  --username AWS \
  --password-stdin 282516655659.dkr.ecr.eu-north-1.amazonaws.com

# 2. Build the MCP container image locally
podman build -t mcp-atlassian:latest .

# 3. Tag the image for ECR
# (Replace IMAGE_ID with the hash output from podman build)
podman tag mcp-atlassian:latest \
  282516655659.dkr.ecr.eu-north-1.amazonaws.com/tfm/mcp-atlassian:latest

# 4. Push to ECR
podman push 282516655659.dkr.ecr.eu-north-1.amazonaws.com/tfm/mcp-atlassian:latest

# (Optional) Tag and push a specific version
podman tag mcp-atlassian:latest \
  282516655659.dkr.ecr.eu-north-1.amazonaws.com/tfm/mcp-atlassian:v1.0.0
podman push 282516655659.dkr.ecr.eu-north-1.amazonaws.com/tfm/mcp-atlassian:v1.0.0
```

**Verification:**
- Confirm image is present in ECR via AWS Console or CLI:
  ```bash
  aws ecr describe-images --repository-name tfm/mcp-atlassian --region eu-north-1
  ```

## Step 2: AgentCore Runtime Configuration

### Create MCP Runtime in Bedrock AgentCore Console

1. **Select MCP Image**
   - Navigate to Bedrock → AgentCore Runtimes
   - Create a new runtime
   - Select the MCP image pushed to ECR: `282516655659.dkr.ecr.eu-north-1.amazonaws.com/tfm/mcp-atlassian:latest`

2. **Configure Authentication**
   - Auth Type: JWT token-based
   - Generate or provide a JWT secret
   - Store the token securely for authentication

3. **Security Configuration**
   - Enable internet egress (required for Jira API calls)
   - Restrict inbound access to Bedrock AgentCore client (*This user is created by default when a new runtime is generated)

4. **Environment Variables**
   - `JIRA_URL`: Jira instance URL (e.g., `https://your-domain.atlassian.net`)
   - `JIRA_USERNAME`: Jira user email
   - `JIRA_API_TOKEN`: Jira API token
   - `TOOLSETS`: Set to `jira_issues` to enable Jira tool discovery
   - `ALLOW_GLOBAL_CRED_FALLBACK`: Set to `true` (optional, for credential fallback)
   - `STATELESS`: Set to `true` if using stateless MCP operations

5. **Network Configuration**
   - Expose port 8000 internally

### Verify MCP Runtime

- In AgentCore Console, confirm the runtime status is "Running"
- Check Amazon CloudWatch logs
- Test tool discovery: `/mcp` endpoint should return available tools
- Example: `tools/list` should include `jira_search`, `jira_get_project_issues`, etc.

## Step 3: Bedrock Knowledge Base & RAG

### Create Knowledge Base (KB) in Bedrock Console

1. **Set Up Vector Store**
   - Navigate to Bedrock → Knowledge Bases
   - Create a new knowledge base
   - Storage backend: **Amazon S3**
   - Vector store: **Amazon OpenSearch Serverless** or **S3-backed vectors**

2. **Configure Embedding Model**
   - Embedding model: **Titan Text Embeddings V2** (or similar)
   - Dimensions: 1024 or configured value
   - Ensure model is available in your region

3. **Create/Select S3 Bucket**
   - Use existing S3 bucket or create a new one
   - Bucket name example: `bedrock-kb-documents-eu-north-1`
   - Ensure proper IAM permissions for Bedrock service role

4. **Upload Documents**
   - Supported formats: PDF, TXT, DOCX, PPTX, JSON, CSV
   - Upload documents to the S3 path linked to your KB
   - Bedrock automatically indexes and embeds the content
   - Monitor ingestion progress in the console

### Test Knowledge Base

- Use the "Test" option in Bedrock Console
- Query examples: "What is the onboarding policy?" or "List open tickets"
- Verify retrieved chunks are relevant and accurately sourced

## Step 4: LLM Service Configuration

### Select & Configure LLM

1. **Model Selection**
   - Navigate to Bedrock → Models
   - Bedrock provides several LLM options by default:
     - Claude (Anthropic)
     - Llama (Meta)
     - Mistral
     - Cohere
   - Select the model best suited for your use case

2. **Model Parameters**
   - Temperature: 0.7 (balanced) to 1.0 (creative)
   - Max tokens: 2048 (adjust based on response needs)
   - Top-p: 0.9 (nucleus sampling)

3. **Integration with Agent**
   - The LLM service is pre-deployed in Bedrock
   - No additional deployment needed
   - Simply select your preferred model when creating an agent or invoking responses

### Example: Invoke LLM via Agent

The Bedrock agent automatically routes requests through:
1. **LLM (reasoning)** → decides if tools are needed
2. **MCP Tools (execution)** → calls Jira/external services
3. **RAG (retrieval)** → embeds and searches knowledge base
4. **LLM (synthesis)** → generates final response

## Testing & Verification

### Test MCP Connectivity

```bash
# Authenticate with AgentCore gateway (using exposed credentials)
TOKEN=$(curl -s -X POST https://gateway-quick-start-xxx.bedrock-agentcore.eu-north-1.amazonaws.com/oauth2/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&client_id=YOUR_CLIENT_ID&client_secret=YOUR_SECRET" | jq -r '.access_token')

# List available MCP tools
curl -s -X POST https://gateway-quick-start-xxx.bedrock-agentcore.eu-north-1.amazonaws.com/mcp \
  -H "Authorization: Bearer $TOKEN" \
  -H "MCP-Protocol-Version: 2025-11-25" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

### Test RAG Retrieval

- Use Bedrock Console's "Retrieve" feature
- Query the knowledge base with a test question
- Verify document chunks are retrieved and ranked by relevance

### Test Full Agent Workflow

- Create an agent in Bedrock Console
- Enable MCP and Knowledge Base
- Send a multi-turn conversation that requires:
  - LLM reasoning (e.g., "Should I create a ticket?")
  - MCP tool call (e.g., `jira_create_issue`)
  - RAG retrieval (e.g., "Check the onboarding policy")

## Troubleshooting

### MCP Connection Failures

**Symptom:** "Cannot connect to MCP server" or "Port 8000 not accessible"

**Solution:**
- Verify Dockerfile exposes port 8000
- Rebuild and re-push the image to ECR
- Confirm AgentCore runtime status is "Running"
- Check security groups allow egress on 443 (for Jira API)

### Jira Authentication Errors

**Symptom:** "Invalid credentials" or "Unauthorized" from Jira MCP

**Solution:**
- Verify `JIRA_API_TOKEN` is valid (regenerate in Jira if needed)
- Ensure `JIRA_USERNAME` is the full email address
- Test Jira API directly:
  ```bash
  curl -u username:token https://your-domain.atlassian.net/rest/api/3/myself
  ```

### Knowledge Base Indexing Slow

**Symptom:** Documents uploaded but not yet searchable

**Solution:**
- Indexing is asynchronous; allow 5–15 minutes
- Check KB ingestion logs in CloudWatch
- Ensure S3 bucket has sufficient permissions

### Bedrock Model Not Available

**Symptom:** "Model access denied" or "Model not found"

**Solution:**
- Request model access in Bedrock Console → Model Access
- Wait for AWS approval (usually instant for common models)
- Verify region matches your deployment region

---

## Additional Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Bedrock AgentCore Runtime Guide](https://docs.aws.amazon.com/bedrock/agentcore/)
- [MCP (Model Context Protocol) Specification](https://modelcontextprotocol.io/)
- [Jira API Documentation](https://developer.atlassian.com/cloud/jira/rest/)
- Local stack reference: [StackChat README](../apps/README.md) 