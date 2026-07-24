from pathlib import Path
import shutil

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


DOCUMENTS_DIR = Path("documents")
DATABASE_DIR = Path("chroma_db")

COLLECTION_NAME = "university_documents"
EMBEDDING_MODEL = "mxbai-embed-large"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_pdf_pages() -> list[Document]:
    """Extract text from every readable page in the documents folder."""
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files were found inside: {DOCUMENTS_DIR.resolve()}"
        )

    documents: list[Document] = []

    for pdf_path in pdf_files:
        print(f"Reading: {pdf_path.name}")

        try:
            reader = PdfReader(str(pdf_path))
        except Exception as exc:
            print(f"  Skipped because the PDF could not be opened: {exc}")
            continue

        readable_pages = 0

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if not text:
                print(f"  Skipping empty page {page_number}")
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_number,
                    },
                )
            )
            readable_pages += 1

        print(
            f"  Extracted {readable_pages} readable pages "
            f"out of {len(reader.pages)}"
        )

    if not documents:
        raise ValueError("No readable text was extracted from the PDFs.")

    return documents


def main() -> None:
    pages = load_pdf_pages()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(pages)

    for chunk_number, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk"] = chunk_number

    print(f"\nReadable pages: {len(pages)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Chunk overlap: {CHUNK_OVERLAP} characters")

    # Rebuild the database cleanly whenever this script is run.
    if DATABASE_DIR.exists():
        print(f"\nRemoving old database: {DATABASE_DIR}")
        shutil.rmtree(DATABASE_DIR)

    print(f"\nCreating embeddings with: {EMBEDDING_MODEL}")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=str(DATABASE_DIR),
    )

    print("\nIndexing completed successfully.")
    print(f"Vector database saved to: {DATABASE_DIR.resolve()}")


if __name__ == "__main__":
    main()