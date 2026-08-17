"""Shared FastAPI dependencies."""

from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.services.llamastack import LlamaStackService


@lru_cache
def get_llama_service() -> LlamaStackService:
    settings: Settings = get_settings()
    return LlamaStackService(settings)
