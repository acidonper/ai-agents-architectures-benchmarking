"""Health and configuration endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import get_llama_service
from app.models.schemas import AppConfig, HealthResponse
from app.services.llamastack import LlamaStackService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    service: LlamaStackService = Depends(get_llama_service),
) -> HealthResponse:
    ok, detail = service.health_check()
    return HealthResponse(
        status="ok" if ok else "degraded",
        llama_stack="reachable" if ok else "unreachable",
        detail=detail,
    )


@router.get("/config", response_model=AppConfig)
def public_config(settings: Settings = Depends(get_settings)) -> AppConfig:
    return AppConfig(
        llama_stack_base_url=settings.llama_stack_base_url,
        default_model=settings.default_model,
        default_embedding_model=settings.default_embedding_model,
        default_embedding_dimension=settings.default_embedding_dimension,
        default_vector_store_provider=settings.default_vector_store_provider,
        default_vector_store_ids=settings.default_vector_store_ids,
        default_mcp_servers=settings.default_mcp_servers,
    )
