from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    database_dir: Path
    uploads_dir: Path
    collection_name: str
    embedding_model: str
    generation_model: str
    chunk_size: int
    chunk_overlap: int
    dense_candidate_k: int
    bm25_candidate_k: int
    final_top_k: int
    max_upload_bytes: int
    ocr_dpi: int
    ocr_language: str

    @classmethod
    def from_environment(cls) -> "Settings":
        data_dir = Path(
            os.getenv("RAG_DATA_DIR", str(PROJECT_ROOT / "data"))
        ).resolve()
        return cls(
            data_dir=data_dir,
            database_dir=data_dir / "chroma_db",
            uploads_dir=data_dir / "uploads",
            collection_name=os.getenv(
                "RAG_COLLECTION", "production_documents"
            ),
            embedding_model=os.getenv(
                "RAG_EMBEDDING_MODEL", "mxbai-embed-large"
            ),
            generation_model=os.getenv(
                "RAG_GENERATION_MODEL", "qwen3.5:2b"
            ),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "200")),
            dense_candidate_k=int(os.getenv("RAG_DENSE_K", "20")),
            bm25_candidate_k=int(os.getenv("RAG_BM25_K", "20")),
            final_top_k=int(os.getenv("RAG_FINAL_K", "3")),
            max_upload_bytes=int(
                os.getenv("RAG_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024))
            ),
            ocr_dpi=int(os.getenv("RAG_OCR_DPI", "300")),
            ocr_language=os.getenv("RAG_OCR_LANGUAGE", "eng"),
        )

    def prepare_directories(self) -> None:
        self.database_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)


settings = Settings.from_environment()

