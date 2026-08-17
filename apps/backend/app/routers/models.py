"""Model discovery endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from app.dependencies import get_llama_service
from app.models.schemas import ModelInfo
from app.services.llamastack import LlamaStackService

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelInfo])
def list_models(
    model_type: Literal["llm", "embedding", "all"] | None = Query(
        default="all",
        description="Filter by model type (llm, embedding, or all)",
    ),
    service: LlamaStackService = Depends(get_llama_service),
) -> list[ModelInfo]:
    try:
        models = service.list_models()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if model_type in (None, "all"):
        return models
    wanted = model_type.lower()
    return [m for m in models if (m.model_type or "").lower() == wanted]
