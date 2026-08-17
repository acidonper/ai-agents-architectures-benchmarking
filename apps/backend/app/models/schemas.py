"""Request/response schemas for the chat API."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class McpServerConfig(BaseModel):
    server_label: str = Field(..., description="Human-readable MCP server label")
    server_url: str = Field(..., description="MCP server SSE/HTTP URL")
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Optional allow-list of MCP tool names",
    )


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    model: str | None = None
    instructions: str | None = Field(
        default=None,
        description="System instructions for the agent",
    )
    previous_response_id: str | None = Field(
        default=None,
        description="Continue a multi-turn conversation via Responses API",
    )
    history: list[ChatMessage] = Field(default_factory=list)
    enable_rag: bool = False
    vector_store_ids: list[str] = Field(default_factory=list)
    enable_mcp: bool = False
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class ToolCallSummary(BaseModel):
    type: str
    name: str | None = None
    status: str | None = None
    detail: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    response_id: str | None = None
    model: str | None = None
    output_text: str
    tool_calls: list[ToolCallSummary] = Field(default_factory=list)
    raw_output: list[dict[str, Any]] | None = None


class ModelInfo(BaseModel):
    identifier: str
    model_type: str | None = None
    provider_id: str | None = None
    embedding_dimension: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorStoreInfo(BaseModel):
    id: str
    name: str | None = None
    status: str | None = None
    file_counts: dict[str, Any] | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    provider_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateVectorStoreRequest(BaseModel):
    name: str
    embedding_model: str | None = Field(
        default=None,
        description="Llama Stack embedding model id (falls back to DEFAULT_EMBEDDING_MODEL)",
    )
    embedding_dimension: int | None = Field(
        default=None,
        description="Embedding vector size (falls back to DEFAULT_EMBEDDING_DIMENSION)",
    )
    provider_id: str | None = Field(
        default=None,
        description="Vector IO provider id (e.g. milvus); optional if Stack has a default",
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    llama_stack: Literal["reachable", "unreachable"]
    detail: str | None = None


class AppConfig(BaseModel):
    llama_stack_base_url: str
    default_model: str | None = None
    default_embedding_model: str | None = None
    default_embedding_dimension: int | None = None
    default_vector_store_provider: str | None = None
    default_vector_store_ids: list[str] = Field(default_factory=list)
    default_mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
