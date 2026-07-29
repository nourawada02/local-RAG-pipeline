from __future__ import annotations

import hashlib
import re
import subprocess
import threading
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.extraction import DocumentError, extract_document
from app.retrieval import (
    BM25Index,
    Chunk,
    FusedCandidate,
    decompose_question,
    reciprocal_rank_fusion,
    select_balanced,
)
from app.schemas import (
    HealthResponse,
    IngestResponse,
    QueryResponse,
    Source,
)


ABSTENTION_TEXT = "I do not have enough information in the retrieved documents."
SOURCE_PATTERN = re.compile(
    r"\[([^\]]*Sources?[^\]]*)\]", flags=re.IGNORECASE
)


class EmptyIndexError(RuntimeError):
    pass


def stop_ollama_model(model_name: str) -> None:
    """Unload a local model so two models do not compete for 8 GB of RAM."""
    try:
        subprocess.run(
            ["ollama", "stop", model_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


class RAGService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.prepare_directories()
        self._lock = threading.RLock()
        self._embeddings = OllamaEmbeddings(
            model=self.settings.embedding_model
        )
        self._vector_store = Chroma(
            collection_name=self.settings.collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(self.settings.database_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def _stored_chunks(self) -> list[Chunk]:
        collection = self._vector_store.get(
            include=["documents", "metadatas"]
        )
        texts = collection.get("documents") or []
        metadatas = collection.get("metadatas") or []
        if len(texts) != len(metadatas):
            raise RuntimeError("Chroma returned inconsistent stored data.")
        chunks = [
            Chunk(
                text=str(text),
                document_id=str(metadata.get("document_id", "")),
                filename=str(metadata.get("source", "unknown")),
                page=int(metadata.get("page", 1)),
                chunk=int(metadata.get("chunk", 0)),
                extraction_method=str(
                    metadata.get("extraction_method", "unknown")
                ),
            )
            for text, metadata in zip(texts, metadatas)
            if metadata
        ]
        chunks.sort(key=lambda item: item.key)
        return chunks

    @staticmethod
    def _safe_filename(filename: str) -> str:
        safe = Path(filename).name.strip()
        if not safe or safe in {".", ".."}:
            raise DocumentError("A valid filename is required.")
        return safe

    def ingest(self, filename: str, data: bytes) -> IngestResponse:
        safe_filename = self._safe_filename(filename)
        if not data:
            raise DocumentError("The uploaded file is empty.")
        if len(data) > self.settings.max_upload_bytes:
            raise DocumentError(
                "The uploaded file exceeds the configured size limit."
            )

        with self._lock:
            pages = extract_document(
                filename=safe_filename,
                data=data,
                ocr_dpi=self.settings.ocr_dpi,
                ocr_language=self.settings.ocr_language,
            )
            document_id = hashlib.sha256(data).hexdigest()[:24]
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            page_documents = [
                Document(
                    page_content=page.text,
                    metadata={
                        "document_id": document_id,
                        "source": safe_filename,
                        "page": page.page,
                        "extraction_method": page.extraction_method,
                    },
                )
                for page in pages
            ]
            chunks = splitter.split_documents(page_documents)
            if not chunks:
                raise DocumentError("Chunking produced no indexable text.")

            ids: list[str] = []
            for chunk_number, chunk in enumerate(chunks, start=1):
                chunk.metadata["chunk"] = chunk_number
                ids.append(f"{document_id}:{chunk_number}")

            existing = self._vector_store.get(
                where={"document_id": document_id},
                include=["metadatas"],
            )
            replaced_existing = bool(existing.get("ids"))
            if replaced_existing:
                self._vector_store.delete(where={"document_id": document_id})

            stop_ollama_model(self.settings.generation_model)
            try:
                batch_size = 16
                for start in range(0, len(chunks), batch_size):
                    self._vector_store.add_documents(
                        chunks[start : start + batch_size],
                        ids=ids[start : start + batch_size],
                    )
            finally:
                stop_ollama_model(self.settings.embedding_model)

            stored_path = (
                self.settings.uploads_dir
                / f"{document_id}__{safe_filename}"
            )
            stored_path.write_bytes(data)

            native_pages = sum(
                page.extraction_method != "ocr" for page in pages
            )
            ocr_pages = sum(
                page.extraction_method == "ocr" for page in pages
            )
            return IngestResponse(
                document_id=document_id,
                filename=safe_filename,
                pages=len(pages),
                chunks=len(chunks),
                native_pages=native_pages,
                ocr_pages=ocr_pages,
                replaced_existing=replaced_existing,
                message="Document indexed successfully.",
            )

    @staticmethod
    def _chunk_from_document(document: Document) -> Chunk:
        metadata = document.metadata
        return Chunk(
            text=document.page_content.strip(),
            document_id=str(metadata.get("document_id", "")),
            filename=str(metadata.get("source", "unknown")),
            page=int(metadata.get("page", 1)),
            chunk=int(metadata.get("chunk", 0)),
            extraction_method=str(
                metadata.get("extraction_method", "unknown")
            ),
        )

    def _rank_query(
        self,
        question: str,
        bm25: BM25Index,
        corpus_size: int,
    ) -> tuple[
        list[tuple[Chunk, float]],
        list[tuple[Chunk, float]],
        list[FusedCandidate],
    ]:
        dense_documents = self._vector_store.similarity_search_with_score(
            query=question,
            k=min(self.settings.dense_candidate_k, corpus_size),
        )
        dense = [
            (self._chunk_from_document(document), float(distance))
            for document, distance in dense_documents
        ]
        lexical = bm25.search(
            question, min(self.settings.bm25_candidate_k, corpus_size)
        )
        return dense, lexical, reciprocal_rank_fusion(dense, lexical)

    def _retrieve(self, question: str) -> tuple[list[FusedCandidate], list[str]]:
        chunks = self._stored_chunks()
        if not chunks:
            raise EmptyIndexError(
                "No documents are indexed. Upload a document before querying."
            )
        bm25 = BM25Index(chunks)
        subqueries = decompose_question(question)

        if len(subqueries) == 1:
            dense, lexical, fused = self._rank_query(
                question, bm25, len(chunks)
            )
            return (
                select_balanced(
                    dense,
                    lexical,
                    fused,
                    top_k=self.settings.final_top_k,
                ),
                [],
            )

        selected: list[FusedCandidate] = []
        selected_pages: set[tuple[str, int]] = set()
        for subquery in subqueries:
            _, _, fused = self._rank_query(subquery, bm25, len(chunks))
            for candidate in fused:
                if candidate.chunk.page_key not in selected_pages:
                    selected.append(candidate)
                    selected_pages.add(candidate.chunk.page_key)
                    break

        dense, lexical, full_fused = self._rank_query(
            question, bm25, len(chunks)
        )
        full_balanced = select_balanced(
            dense,
            lexical,
            full_fused,
            top_k=self.settings.final_top_k,
        )
        for candidate in [*full_balanced, *full_fused]:
            if len(selected) >= self.settings.final_top_k:
                break
            if candidate.chunk.page_key in selected_pages:
                continue
            selected.append(candidate)
            selected_pages.add(candidate.chunk.page_key)
        return selected, subqueries

    @staticmethod
    def _context(candidates: list[FusedCandidate]) -> str:
        parts = []
        for rank, candidate in enumerate(candidates, start=1):
            chunk = candidate.chunk
            parts.append(
                f"[Source {rank}: {chunk.filename}, physical PDF page "
                f"{chunk.page}]\n{chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

    def _generate(
        self, question: str, candidates: list[FusedCandidate]
    ) -> str:
        llm = ChatOllama(
            model=self.settings.generation_model,
            temperature=0.2,
            num_ctx=4096,
            num_predict=256,
            reasoning=False,
        )
        messages = [
            (
                "system",
                "You answer questions using only the supplied document "
                "context. Do not use outside knowledge or invent details. "
                f"If the evidence is insufficient, reply exactly: "
                f"'{ABSTENTION_TEXT}' Cite every supported claim with the "
                "provided labels, such as [Source 1]. Do not reveal a "
                "thinking process.",
            ),
            (
                "human",
                f"Context:\n{self._context(candidates)}\n\n"
                f"Question: {question}\n\n"
                "Give a concise but complete grounded answer.",
            ),
        ]
        response = llm.invoke(messages)
        answer = str(response.content).strip()
        return answer or ABSTENTION_TEXT

    @staticmethod
    def _cited_ranks(answer: str) -> set[int]:
        ranks: set[int] = set()
        for label in SOURCE_PATTERN.findall(answer):
            ranks.update(int(value) for value in re.findall(r"\d+", label))
        return ranks

    def query(self, question: str) -> QueryResponse:
        normalized_question = re.sub(r"\s+", " ", question).strip()
        if not normalized_question:
            raise ValueError("Question must not be empty.")

        with self._lock:
            total_start = time.perf_counter()
            stop_ollama_model(self.settings.generation_model)
            retrieval_start = time.perf_counter()
            candidates, subqueries = self._retrieve(normalized_question)
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

            stop_ollama_model(self.settings.embedding_model)
            generation_start = time.perf_counter()
            try:
                answer = self._generate(normalized_question, candidates)
            finally:
                stop_ollama_model(self.settings.generation_model)
            generation_ms = (time.perf_counter() - generation_start) * 1000
            total_ms = (time.perf_counter() - total_start) * 1000

            cited_ranks = self._cited_ranks(answer)
            sources = [
                Source(
                    rank=rank,
                    document_id=candidate.chunk.document_id,
                    filename=candidate.chunk.filename,
                    page=candidate.chunk.page,
                    chunk=candidate.chunk.chunk,
                    extraction_method=candidate.chunk.extraction_method,
                    cited=rank in cited_ranks,
                )
                for rank, candidate in enumerate(candidates, start=1)
            ]
            return QueryResponse(
                answer=answer,
                abstained=(
                    ABSTENTION_TEXT.casefold() in answer.casefold()
                ),
                decomposition_applied=bool(subqueries),
                subqueries=subqueries,
                sources=sources,
                retrieval_ms=round(retrieval_ms, 3),
                generation_ms=round(generation_ms, 3),
                total_ms=round(total_ms, 3),
            )

    def health(self) -> HealthResponse:
        with self._lock:
            count = len(self._stored_chunks())
        return HealthResponse(
            status="ok",
            indexed_chunks=count,
            embedding_model=self.settings.embedding_model,
            generation_model=self.settings.generation_model,
        )
