from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from legal_system_rag.config.settings import (
    EMBEDDING_MODEL,
    PERSIST_DIRECTORY,
)


QUESTION = "Wie lang ist die Kündigungsfrist für den Mieter?"


def print_results(title, results):
    print(f"\n\n{'#' * 80}")
    print(f"# {title}")
    print(f"{'#' * 80}")

    print(f"\nAnzahl Treffer: {len(results)}")

    for i, doc in enumerate(results, 1):
        print(f"\n{'=' * 80}")
        print(f"Treffer {i}")
        print(f"{'=' * 80}")

        print("\nMetadata:")
        print(doc.metadata)

        print("\nPage Content:")
        print(doc.page_content)


def main():
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
    )

    vector_store = Chroma(
        persist_directory=str(Path(PERSIST_DIRECTORY)),
        embedding_function=embeddings,
    )

    # --------------------------------------------------------
    # Test 1: Standard similarity search
    # --------------------------------------------------------

    similarity_results = vector_store.similarity_search(
        QUESTION,
        k=5,
    )

    print_results(
        "STANDARD SIMILARITY SEARCH",
        similarity_results,
    )

    # --------------------------------------------------------
    # Test 2: MMR - same retrieval strategy as chain.py
    # --------------------------------------------------------

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5},
    )

    mmr_results = retriever.invoke(QUESTION)

    print_results(
        "MMR RETRIEVAL",
        mmr_results,
    )


if __name__ == "__main__":
    main()