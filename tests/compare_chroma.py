from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


DB_A = Path("./chroma_legal_rag_A")
DB_B = Path("./chroma_legal_rag_B")

COLLECTION_NAME = "legal_documents"


def load_vector_store(path: Path) -> Chroma:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(path),
        embedding_function=embeddings,
    )


def inspect_database(name: str, vector_store: Chroma) -> None:
    collection = vector_store._collection

    count = collection.count()

    print()
    print("=" * 70)
    print(f"DATABASE {name}")
    print("=" * 70)
    print(f"Document count: {count}")

    data = collection.get(
        include=["documents", "metadatas"]
    )

    documents = data["documents"]
    metadatas = data["metadatas"]

    for i, (document, metadata) in enumerate(
        zip(documents[:5], metadatas[:5])
    ):
        print()
        print(f"Document {i}")
        print("-" * 70)
        print("Metadata:")
        print(metadata)
        print("Content:")
        print(document[:500])


def retrieve(
    name: str,
    vector_store: Chroma,
    query: str,
) -> None:

    print()
    print("=" * 70)
    print(f"RETRIEVAL {name}")
    print("=" * 70)
    print(f"Query: {query}")

    results = vector_store.similarity_search(
        query,
        k=5,
    )

    for rank, document in enumerate(results, start=1):
        print()
        print(f"Rank {rank}")
        print("-" * 70)
        print("Metadata:")
        print(document.metadata)
        print("Content:")
        print(document.page_content[:500])


def main() -> None:

    vector_store_a = load_vector_store(DB_A)
    vector_store_b = load_vector_store(DB_B)

    inspect_database("A", vector_store_a)
    inspect_database("B", vector_store_b)

    query = "Wie lang ist die Kündigungsfrist für den Mieter?"

    retrieve("A", vector_store_a, query)
    retrieve("B", vector_store_b, query)


if __name__ == "__main__":
    main()