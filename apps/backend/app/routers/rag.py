"""RAG / vector store endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.dependencies import get_llama_service
from app.models.schemas import CreateVectorStoreRequest, VectorStoreInfo
from app.services.llamastack import LlamaStackService

router = APIRouter(prefix="/rag", tags=["rag"])


@router.get("/vector-stores", response_model=list[VectorStoreInfo])
def list_vector_stores(
    service: LlamaStackService = Depends(get_llama_service),
) -> list[VectorStoreInfo]:
    try:
        return service.list_vector_stores()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/vector-stores", response_model=VectorStoreInfo)
def create_vector_store(
    body: CreateVectorStoreRequest,
    service: LlamaStackService = Depends(get_llama_service),
) -> VectorStoreInfo:
    try:
        return service.create_vector_store(
            name=body.name,
            embedding_model=body.embedding_model,
            embedding_dimension=body.embedding_dimension,
            provider_id=body.provider_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/vector-stores/{vector_store_id}/files")
async def upload_file(
    vector_store_id: str,
    file: UploadFile = File(...),
    service: LlamaStackService = Depends(get_llama_service),
) -> dict:
    try:
        content = await file.read()
        filename = file.filename or "upload.bin"
        return service.upload_file_to_vector_store(
            vector_store_id=vector_store_id,
            filename=filename,
            content=content,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
