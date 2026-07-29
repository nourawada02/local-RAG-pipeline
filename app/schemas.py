from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class IngestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    filename: str
    pages: int
    chunks: int
    native_pages: int
    ocr_pages: int
    replaced_existing: bool
    message: str


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    document_id: str
    filename: str
    page: int
    chunk: int
    extraction_method: str
    cited: bool


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    abstained: bool
    decomposition_applied: bool
    subqueries: list[str]
    sources: list[Source]
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
    embedding_model: str
    generation_model: str


class ApiInfo(BaseModel):
    name: str
    version: str
    documentation: str

