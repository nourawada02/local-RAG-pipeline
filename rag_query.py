from pathlib import Path
import subprocess

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings


DATABASE_DIR = Path("chroma_db")

COLLECTION_NAME = "university_documents"
EMBEDDING_MODEL = "mxbai-embed-large"
GENERATION_MODEL = "qwen3.5:2b"

TOP_K = 3


def stop_ollama_model(model_name: str) -> None:
    """Unload an Ollama model to free memory."""
    try:
        subprocess.run(
            ["ollama", "stop", model_name],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        # The main query will report an error if Ollama is unavailable.
        pass


def open_vector_store() -> Chroma:
    """Open the existing persistent Chroma database."""
    if not DATABASE_DIR.exists():
        raise FileNotFoundError(
            "The chroma_db folder does not exist. "
            "Run build_index.py before querying the documents."
        )

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(DATABASE_DIR),
    )


def retrieve_chunks(
    vector_store: Chroma,
    question: str,
) -> list[tuple[Document, float]]:
    """Retrieve the chunks most semantically related to the question."""
    return vector_store.similarity_search_with_score(
        query=question,
        k=TOP_K,
    )


def print_retrieved_chunks(
    results: list[tuple[Document, float]],
) -> None:
    """Display retrieved text and metadata for manual evaluation."""
    print("\n" + "=" * 70)
    print("RETRIEVED CHUNKS")
    print("=" * 70)

    for rank, (document, distance) in enumerate(results, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")
        chunk = document.metadata.get("chunk", "Unknown chunk")

        # Chroma returns a distance, so lower generally means a closer match.
        print(f"\nResult {rank}")
        print(f"Source: {source}")
        print(f"Page: {page}")
        print(f"Chunk: {chunk}")
        print(f"Distance: {distance:.4f}")
        print("-" * 70)
        print(document.page_content.strip())


def create_context(
    results: list[tuple[Document, float]],
) -> str:
    """Combine retrieved chunks into a clearly labelled context."""
    context_parts = []

    for number, (document, _) in enumerate(results, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        context_parts.append(
            f"[Source {number}: {source}, page {page}]\n"
            f"{document.page_content.strip()}"
        )

    return "\n\n---\n\n".join(context_parts)


def generate_answer(
    llm: ChatOllama,
    question: str,
    results: list[tuple[Document, float]],
) -> str:
    """Generate an answer grounded only in the retrieved chunks."""
    context = create_context(results)

    messages = [
        (
            "system",
            (
                "You are a university-document question-answering assistant. "
                "Answer using only the supplied context. "
                "Do not use outside knowledge or invent missing details. "
                "If the context does not contain enough information, say: "
                "'I do not have enough information in the retrieved documents.' "
                "Use a clear, professional tone. "
                "Cite supporting information using the source labels, "
                "for example [Source 1]. "
                "Do not reveal a thinking process."
            ),
        ),
        (
            "human",
            (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Give a concise but complete answer."
            ),
        ),
    ]

    response = llm.invoke(messages)
    answer = str(response.content).strip()

    if not answer:
        return "The generation model returned an empty response."

    return answer


def main() -> None:
    try:
        vector_store = open_vector_store()
    except Exception as exc:
        print(f"Database error: {exc}")
        return

    llm = ChatOllama(
        model=GENERATION_MODEL,
        temperature=0.2,
        num_ctx=4096,
        num_predict=256,
	reasoning=False,
    )

    print("University PDF RAG")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Generation model: {GENERATION_MODEL}")
    print(f"Retrieved chunks per question: {TOP_K}")
    print("Type exit to stop.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in {"exit", "quit"}:
            stop_ollama_model(EMBEDDING_MODEL)
            stop_ollama_model(GENERATION_MODEL)
            print("Exiting.")
            break

        if not question:
            print("Enter a non-empty question.")
            continue

        try:
            # Prevent Qwen from occupying memory while the embedding
            # model is being used for retrieval.
            stop_ollama_model(GENERATION_MODEL)

            results = retrieve_chunks(
                vector_store=vector_store,
                question=question,
            )

            if not results:
                print("No chunks were retrieved.")
                continue

            print_retrieved_chunks(results)

            # Retrieval is complete, so unload the embedding model
            # before starting the generation model.
            stop_ollama_model(EMBEDDING_MODEL)

            print(f"\nGenerating answer with {GENERATION_MODEL}...")

            answer = generate_answer(
                llm=llm,
                question=question,
                results=results,
            )

            print("\n" + "=" * 70)
            print("GENERATED ANSWER")
            print("=" * 70)
            print(answer)

            # Free memory after generation.
            stop_ollama_model(GENERATION_MODEL)

        except Exception as exc:
            stop_ollama_model(EMBEDDING_MODEL)
            stop_ollama_model(GENERATION_MODEL)

            print(f"\nQuery failed: {exc}")
            print(
                "Confirm that Ollama is running and both models are installed."
            )


if __name__ == "__main__":
    main()