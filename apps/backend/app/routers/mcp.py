"""MCP configuration helpers."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.models.schemas import McpServerConfig

router = APIRouter(prefix="/mcp", tags=["mcp"])


class McpDefaultsResponse(BaseModel):
    servers: list[McpServerConfig] = Field(default_factory=list)


@router.get("/defaults", response_model=McpDefaultsResponse)
def get_default_mcp_servers(
    settings: Settings = Depends(get_settings),
) -> McpDefaultsResponse:
    servers = [McpServerConfig(**item) for item in settings.default_mcp_servers]
    return McpDefaultsResponse(servers=servers)
