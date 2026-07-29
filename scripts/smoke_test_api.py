from __future__ import annotations

import argparse
import json
import mimetypes
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def request_json(request: urllib.request.Request, timeout: int) -> dict:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            if response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}: {body}")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def multipart_file(path: Path) -> tuple[bytes, str]:
    boundary = f"----rag-smoke-{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    suffix = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return prefix + path.read_bytes() + suffix, boundary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ingestion and query against a live local RAG API."
    )
    parser.add_argument("document", type=Path)
    parser.add_argument("question")
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000"
    )
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    document = args.document.resolve()
    if not document.is_file():
        raise SystemExit(f"Document not found: {document}")

    body, boundary = multipart_file(document)
    ingest_request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/ingest",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    ingest = request_json(ingest_request, args.timeout)
    print("INGEST")
    print(json.dumps(ingest, indent=2, ensure_ascii=False))

    query_body = json.dumps({"question": args.question}).encode("utf-8")
    query_request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/query",
        data=query_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    query = request_json(query_request, args.timeout)
    print("\nQUERY")
    print(json.dumps(query, indent=2, ensure_ascii=False))

    health_request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/health"
    )
    health = request_json(health_request, args.timeout)
    print("\nHEALTH")
    print(json.dumps(health, indent=2, ensure_ascii=False))

    required_query_fields = {
        "answer",
        "abstained",
        "sources",
        "retrieval_ms",
        "generation_ms",
        "total_ms",
    }
    missing = required_query_fields - set(query)
    if missing:
        raise SystemExit(f"Smoke test failed; missing fields: {sorted(missing)}")
    if not ingest.get("chunks") or not health.get("indexed_chunks"):
        raise SystemExit("Smoke test failed; no indexed chunks were reported.")
    print("\nEnd-to-end API smoke test passed.")


if __name__ == "__main__":
    main()

