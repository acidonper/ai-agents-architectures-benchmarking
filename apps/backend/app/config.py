"""Application settings loaded from environment variables."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llama_stack_base_url: str = "http://localhost:8321"
    llama_stack_api_key: str | None = None
    default_model: str | None = None
    default_embedding_model: str | None = None
    default_embedding_dimension: int | None = 768
    default_vector_store_provider: str | None = None

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )

    default_mcp_servers: Annotated[list[dict[str, Any]], NoDecode] = Field(
        default_factory=list
    )
    default_vector_store_ids: Annotated[list[str], NoDecode] = Field(
        default_factory=list
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> list[str]:
        if value is None or value == "":
            return [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
            ]
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                return json.loads(stripped)
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("default_mcp_servers", mode="before")
    @classmethod
    def parse_mcp_servers(cls, value: Any) -> list[dict[str, Any]]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return json.loads(value)
        return value

    @field_validator("default_vector_store_ids", mode="before")
    @classmethod
    def parse_vector_store_ids(cls, value: Any) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                return json.loads(stripped)
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
