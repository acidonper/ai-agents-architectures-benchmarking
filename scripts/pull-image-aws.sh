#!/bin/bash

MCP_IMAGE="ghcr.io/sooperset/mcp-atlassian:latest"
AWS_ECR="282516655659.dkr.ecr.eu-north-1.amazonaws.com"
AWS_ECR_REPO="${AWS_ECR}/tfm/mcp-atlassian:latest"

podman pull ${MCP_IMAGE}
aws ecr get-login-password --region eu-north-1 | podman login --username AWS --password-stdin ${AWS_ECR}
podman tag ${MCP_IMAGE} ${AWS_ECR_REPO}
podman push ${AWS_ECR_REPO}