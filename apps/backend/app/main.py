"""FastAPI entrypoint for the Llama Stack chat backend."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import chat, health, mcp, models, rag

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

settings = get_settings()

app = FastAPI(
    title="Llama Stack Chat API",
    description=(
        "Backend that orchestrates chat against Llama Stack, "
        "including LLM inference, RAG (file_search), and MCP tools."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(rag.router, prefix="/api")
app.include_router(mcp.router, prefix="/api")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "llama-stack-chat-api",
        "docs": "/docs",
        "health": "/api/health",
    }
