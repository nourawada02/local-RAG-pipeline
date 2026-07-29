from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.extraction import DocumentError
from app.schemas import (
    ApiInfo,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.service import EmptyIndexError, RAGService


app = FastAPI(
    title="Local Production RAG API",
    version="1.0.0",
    description=(
        "Upload native or scanned documents and ask grounded questions "
        "using local Ollama models."
    ),
)


@lru_cache(maxsize=1)
def get_service() -> RAGService:
    return RAGService(settings)


@app.get("/", response_model=ApiInfo)
def api_info() -> ApiInfo:
    return ApiInfo(
        name="Local Production RAG API",
        version="1.0.0",
        documentation="/docs",
    )


@app.get("/health", response_model=HealthResponse)
def health(service: RAGService = Depends(get_service)) -> HealthResponse:
    return service.health()


@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_document(
    file: UploadFile = File(...),
    service: RAGService = Depends(get_service),
) -> IngestResponse:
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Uploaded file exceeds the 50 MB default limit.",
        )
    try:
        return await run_in_threadpool(
            service.ingest, file.filename or "", data
        )
    except DocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Ingestion failed. Confirm that Ollama, mxbai-embed-large, "
                f"and Tesseract when required are available. Error: {exc}"
            ),
        ) from exc


@app.post("/query", response_model=QueryResponse)
def query_documents(
    request: QueryRequest,
    service: RAGService = Depends(get_service),
) -> QueryResponse:
    try:
        return service.query(request.question)
    except EmptyIndexError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Query failed. Confirm that Ollama is running and both "
                f"models are installed. Error: {exc}"
            ),
        ) from exc

