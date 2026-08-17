"""Chat endpoints backed by Llama Stack Responses API."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.dependencies import get_llama_service
from app.models.schemas import ChatRequest, ChatResponse
from app.services.llamastack import LlamaStackService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    service: LlamaStackService = Depends(get_llama_service),
) -> ChatResponse:
    if body.stream:
        raise HTTPException(
            status_code=400,
            detail="Use POST /chat/stream for streaming responses",
        )
    try:
        return service.chat(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat request failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    service: LlamaStackService = Depends(get_llama_service),
) -> EventSourceResponse:
    body.stream = True

    def event_generator():
        try:
            for event in service.chat_stream(body):
                yield {
                    "event": event["event"],
                    "data": json.dumps(event["data"]),
                }
        except ValueError as exc:
            yield {
                "event": "error",
                "data": json.dumps({"detail": str(exc)}),
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming chat failed")
            yield {
                "event": "error",
                "data": json.dumps({"detail": str(exc)}),
            }

    return EventSourceResponse(event_generator())
