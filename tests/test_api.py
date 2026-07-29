from __future__ import annotations

import importlib.util
import unittest


API_DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in (
        "fastapi",
        "httpx",
        "langchain_chroma",
        "langchain_core",
        "langchain_ollama",
        "langchain_text_splitters",
        "multipart",
        "pytesseract",
    )
)


@unittest.skipUnless(
    API_DEPENDENCIES_AVAILABLE,
    "Install requirements-dev.txt to run the API contract tests.",
)
class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from fastapi.testclient import TestClient

        from app.main import app, get_service
        from app.schemas import HealthResponse, IngestResponse, QueryResponse

        class FakeService:
            def health(self) -> HealthResponse:
                return HealthResponse(
                    status="ok",
                    indexed_chunks=2,
                    embedding_model="test-embed",
                    generation_model="test-chat",
                )

            def ingest(self, filename: str, data: bytes) -> IngestResponse:
                return IngestResponse(
                    document_id="abc123",
                    filename=filename,
                    pages=1,
                    chunks=1,
                    native_pages=1,
                    ocr_pages=0,
                    replaced_existing=False,
                    message="Document indexed successfully.",
                )

            def query(self, question: str) -> QueryResponse:
                return QueryResponse(
                    answer="Grounded answer [Source 1].",
                    abstained=False,
                    decomposition_applied=False,
                    subqueries=[],
                    sources=[],
                    retrieval_ms=1.0,
                    generation_ms=2.0,
                    total_ms=3.0,
                )

        app.dependency_overrides[get_service] = lambda: FakeService()
        cls.app = app
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.app.dependency_overrides.clear()

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["indexed_chunks"], 2)

    def test_ingest(self) -> None:
        response = self.client.post(
            "/ingest",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["filename"], "notes.txt")

    def test_query(self) -> None:
        response = self.client.post(
            "/query", json={"question": "What does the document say?"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["abstained"])


if __name__ == "__main__":
    unittest.main()

