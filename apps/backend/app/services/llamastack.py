"""Llama Stack client wrapper for LLMs, RAG, and MCP workflows."""

from __future__ import annotations

import logging
from typing import Any, Iterator

from llama_stack_client import LlamaStackClient

from app.config import Settings
from app.models.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    McpServerConfig,
    ModelInfo,
    ToolCallSummary,
    VectorStoreInfo,
)

logger = logging.getLogger(__name__)


class LlamaStackService:
    """Thin orchestration layer over the Llama Stack Responses API."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        kwargs: dict[str, Any] = {"base_url": settings.llama_stack_base_url}
        if settings.llama_stack_api_key:
            kwargs["api_key"] = settings.llama_stack_api_key
        self.client = LlamaStackClient(**kwargs)

    def health_check(self) -> tuple[bool, str | None]:
        try:
            models = self.client.models.list()
            count = len(list(models)) if models is not None else 0
            return True, f"Connected; {count} model(s) visible"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Llama Stack health check failed: %s", exc)
            return False, str(exc)

    def list_models(self) -> list[ModelInfo]:
        raw = self.client.models.list()
        items = list(raw) if raw is not None else []
        results: list[ModelInfo] = []
        for item in items:
            data = _to_dict(item)
            identifier = (
                data.get("identifier")
                or data.get("id")
                or data.get("model_id")
                or ""
            )
            if not identifier:
                continue
            meta = (
                data.get("metadata")
                or data.get("custom_metadata")
                or {}
            )
            if not isinstance(meta, dict):
                meta = {}
            model_type = (
                data.get("model_type")
                or data.get("type")
                or meta.get("model_type")
                or meta.get("type")
            )
            provider_id = (
                data.get("provider_id")
                or meta.get("provider_id")
                or data.get("owned_by")
            )
            dim = (
                data.get("embedding_dimension")
                or meta.get("embedding_dimension")
                or meta.get("dimension")
            )
            try:
                embedding_dimension = int(dim) if dim is not None else None
            except (TypeError, ValueError):
                embedding_dimension = None
            # Heuristic fallback when Stack omits model_type.
            if not model_type:
                low = str(identifier).lower()
                if "embed" in low or "sentence-transformers" in low:
                    model_type = "embedding"
                else:
                    model_type = "llm"
            results.append(
                ModelInfo(
                    identifier=identifier,
                    model_type=str(model_type) if model_type else None,
                    provider_id=str(provider_id) if provider_id else None,
                    embedding_dimension=embedding_dimension,
                    metadata=meta,
                )
            )
        return results

    def list_embedding_models(self) -> list[ModelInfo]:
        return [
            m
            for m in self.list_models()
            if (m.model_type or "").lower() == "embedding"
        ]

    def list_llm_models(self) -> list[ModelInfo]:
        return [
            m
            for m in self.list_models()
            if (m.model_type or "").lower() in {"", "llm"}
        ]

    def list_vector_stores(self) -> list[VectorStoreInfo]:
        raw = self.client.vector_stores.list()
        data = _to_dict(raw)
        items = data.get("data") or data.get("items") or list(raw or [])
        results: list[VectorStoreInfo] = []
        for item in items:
            entry = _to_dict(item)
            store_id = entry.get("id")
            if not store_id:
                continue
            meta = entry.get("metadata") or {}
            if not isinstance(meta, dict):
                meta = {}
            dim = meta.get("embedding_dimension") or entry.get("embedding_dimension")
            try:
                embedding_dimension = int(dim) if dim is not None else None
            except (TypeError, ValueError):
                embedding_dimension = None
            results.append(
                VectorStoreInfo(
                    id=store_id,
                    name=entry.get("name"),
                    status=entry.get("status"),
                    file_counts=entry.get("file_counts"),
                    embedding_model=meta.get("embedding_model")
                    or entry.get("embedding_model"),
                    embedding_dimension=embedding_dimension,
                    provider_id=meta.get("provider_id") or entry.get("provider_id"),
                    metadata=meta,
                )
            )
        return results

    def resolve_embedding_defaults(
        self,
        embedding_model: str | None = None,
        embedding_dimension: int | None = None,
        provider_id: str | None = None,
    ) -> tuple[str, int | None, str | None]:
        """Resolve embedding model/dimension/provider for vector store create."""
        model = (embedding_model or self.settings.default_embedding_model or "").strip()
        dimension = embedding_dimension
        if dimension is None:
            dimension = self.settings.default_embedding_dimension
        provider = (provider_id or self.settings.default_vector_store_provider or None)

        embeddings = self.list_embedding_models()
        if not model:
            if not embeddings:
                raise ValueError(
                    "No embedding model specified and none are registered on Llama Stack. "
                    "Set DEFAULT_EMBEDDING_MODEL or pass embedding_model when creating a store."
                )
            model = embeddings[0].identifier
            if dimension is None and embeddings[0].embedding_dimension:
                dimension = embeddings[0].embedding_dimension
            logger.info("Auto-selected embedding model for RAG: %s", model)
        else:
            match = next((m for m in embeddings if m.identifier == model), None)
            if match and dimension is None and match.embedding_dimension:
                dimension = match.embedding_dimension

        if dimension is None:
            raise ValueError(
                "embedding_dimension is required. Set DEFAULT_EMBEDDING_DIMENSION "
                "or pass embedding_dimension when creating a store."
            )
        return model, dimension, provider

    def create_vector_store(
        self,
        name: str,
        embedding_model: str | None = None,
        embedding_dimension: int | None = None,
        provider_id: str | None = None,
    ) -> VectorStoreInfo:
        model, dimension, provider = self.resolve_embedding_defaults(
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            provider_id=provider_id,
        )
        extra_body: dict[str, Any] = {
            "embedding_model": model,
            "embedding_dimension": dimension,
        }
        if provider:
            extra_body["provider_id"] = provider

        created = self.client.vector_stores.create(
            name=name,
            extra_body=extra_body,
        )
        entry = _to_dict(created)
        meta = entry.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        return VectorStoreInfo(
            id=entry["id"],
            name=entry.get("name", name),
            status=entry.get("status"),
            file_counts=entry.get("file_counts"),
            embedding_model=meta.get("embedding_model") or model,
            embedding_dimension=meta.get("embedding_dimension") or dimension,
            provider_id=meta.get("provider_id") or provider,
            metadata=meta,
        )

    def upload_file_to_vector_store(
        self,
        vector_store_id: str,
        filename: str,
        content: bytes,
        purpose: str = "assistants",
    ) -> dict[str, Any]:
        file_info = self.client.files.create(
            file=(filename, content),
            purpose=purpose,
        )
        file_data = _to_dict(file_info)
        file_id = file_data.get("id")
        if not file_id:
            raise RuntimeError("Llama Stack did not return a file id")

        attached = self.client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file_id,
        )
        return {
            "file": file_data,
            "vector_store_file": _to_dict(attached),
        }

    def build_tools(self, request: ChatRequest) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []

        vector_ids = list(request.vector_store_ids)
        if request.enable_rag:
            if not vector_ids:
                vector_ids = list(self.settings.default_vector_store_ids)
            if not vector_ids:
                raise ValueError(
                    "RAG is enabled but no vector_store_ids were provided and "
                    "DEFAULT_VECTOR_STORE_IDS is empty. Create or select a vector store first."
                )
            tools.append(
                {
                    "type": "file_search",
                    "vector_store_ids": vector_ids,
                }
            )

        mcp_servers = list(request.mcp_servers)
        if request.enable_mcp and not mcp_servers:
            mcp_servers = [
                McpServerConfig(**item)
                for item in self.settings.default_mcp_servers
            ]

        if request.enable_mcp:
            for server in mcp_servers:
                tool: dict[str, Any] = {
                    "type": "mcp",
                    "server_label": server.server_label,
                    "server_url": server.server_url,
                }
                if server.allowed_tools:
                    tool["allowed_tools"] = server.allowed_tools
                tools.append(tool)

        return tools

    def _resolve_model(self, request: ChatRequest) -> str:
        model = (request.model or self.settings.default_model or "").strip()
        if not model:
            raise ValueError(
                "No model specified. Set DEFAULT_MODEL or pass model in the request."
            )
        return model

    def chat(self, request: ChatRequest) -> ChatResponse:
        model = self._resolve_model(request)

        tools = self.build_tools(request)
        payload = self._build_response_payload(request, model, tools)
        response = self.client.responses.create(**payload)
        return self._parse_response(response, model)

    def chat_stream(self, request: ChatRequest) -> Iterator[dict[str, Any]]:
        model = self._resolve_model(request)

        tools = self.build_tools(request)
        payload = self._build_response_payload(request, model, tools)
        payload["stream"] = True

        try:
            stream = self.client.responses.create(**payload)
        except TypeError:
            # Older clients may not accept stream=True; fall back to non-stream.
            payload.pop("stream", None)
            response = self.client.responses.create(**payload)
            parsed = self._parse_response(response, model)
            yield {"event": "message", "data": parsed.model_dump()}
            yield {"event": "done", "data": {"response_id": parsed.response_id}}
            return

        accumulated = ""
        response_id: str | None = None
        tool_calls: list[ToolCallSummary] = []

        for event in stream:
            event_dict = _to_dict(event)
            event_type = event_dict.get("type") or event_dict.get("event")

            if event_type in {
                "response.output_text.delta",
                "response.text.delta",
                "content.delta",
            }:
                delta = (
                    event_dict.get("delta")
                    or event_dict.get("text")
                    or event_dict.get("content")
                    or ""
                )
                if isinstance(delta, dict):
                    delta = delta.get("text") or delta.get("content") or ""
                accumulated += str(delta)
                yield {
                    "event": "delta",
                    "data": {"text": str(delta)},
                }
            elif event_type in {"response.completed", "response.done"}:
                response_obj = event_dict.get("response") or event_dict
                parsed = self._parse_response(response_obj, model)
                response_id = parsed.response_id
                tool_calls = parsed.tool_calls
                if not accumulated and parsed.output_text:
                    accumulated = parsed.output_text
                yield {
                    "event": "message",
                    "data": {
                        "response_id": response_id,
                        "model": model,
                        "output_text": accumulated or parsed.output_text,
                        "tool_calls": [t.model_dump() for t in tool_calls],
                    },
                }
            elif event_type and "tool" in str(event_type):
                tool_calls.append(
                    ToolCallSummary(
                        type=str(event_type),
                        name=event_dict.get("name")
                        or event_dict.get("tool_name"),
                        status=event_dict.get("status"),
                        detail=event_dict,
                    )
                )
                yield {"event": "tool", "data": event_dict}
            else:
                # Forward unknown events for debugging in the UI.
                yield {"event": "raw", "data": event_dict}

        yield {
            "event": "done",
            "data": {
                "response_id": response_id,
                "output_text": accumulated,
                "tool_calls": [t.model_dump() for t in tool_calls],
            },
        }

    def _build_response_payload(
        self,
        request: ChatRequest,
        model: str,
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "input": self._build_input(request),
        }
        if request.instructions:
            payload["instructions"] = request.instructions
        if request.previous_response_id:
            payload["previous_response_id"] = request.previous_response_id
        if tools:
            payload["tools"] = tools
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        return payload

    def _build_input(self, request: ChatRequest) -> Any:
        """Build Responses API input from history + latest message."""
        if not request.history:
            return request.message

        messages: list[dict[str, str]] = []
        for item in request.history:
            messages.append({"role": item.role.value, "content": item.content})
        messages.append({"role": "user", "content": request.message})
        return messages

    def _parse_response(self, response: Any, model: str) -> ChatResponse:
        data = _to_dict(response)
        output_text = (
            data.get("output_text")
            or _extract_output_text(data)
            or ""
        )
        tool_calls = _extract_tool_calls(data)
        return ChatResponse(
            response_id=data.get("id"),
            model=data.get("model") or model,
            output_text=output_text,
            tool_calls=tool_calls,
            raw_output=data.get("output")
            if isinstance(data.get("output"), list)
            else None,
        )


def _to_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return {
            key: value
            for key, value in vars(obj).items()
            if not key.startswith("_")
        }
    return {"value": str(obj)}


def _extract_output_text(data: dict[str, Any]) -> str:
    output = data.get("output")
    if isinstance(output, str):
        return output
    if not isinstance(output, list):
        return ""

    chunks: list[str] = []
    for item in output:
        entry = _to_dict(item)
        if entry.get("type") == "message":
            content = entry.get("content") or []
            if isinstance(content, str):
                chunks.append(content)
                continue
            for part in content:
                part_dict = _to_dict(part)
                text = part_dict.get("text") or part_dict.get("content")
                if text:
                    chunks.append(str(text))
        elif entry.get("type") in {"output_text", "text"}:
            text = entry.get("text") or entry.get("content")
            if text:
                chunks.append(str(text))
    return "".join(chunks)


def _extract_tool_calls(data: dict[str, Any]) -> list[ToolCallSummary]:
    output = data.get("output")
    if not isinstance(output, list):
        return []

    calls: list[ToolCallSummary] = []
    for item in output:
        entry = _to_dict(item)
        entry_type = str(entry.get("type") or "")
        if any(
            token in entry_type
            for token in ("mcp", "file_search", "function_call", "tool")
        ):
            calls.append(
                ToolCallSummary(
                    type=entry_type,
                    name=entry.get("name") or entry.get("server_label"),
                    status=entry.get("status"),
                    detail=entry,
                )
            )
    return calls
