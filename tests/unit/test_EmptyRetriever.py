from unittest.mock import MagicMock
import pytest

# 1. Funktionen und Klasse direkt aus deiner chain.py importieren:
from legal_system_rag.rag.chain import (
    _prepare_retrievers_symmetrical,
    _create_hybrid_retriever_with_score,
    EmptyRetriever,
)


# Testet, ob prepare_retrievers_symmetrical bei nicht existierendem Filter is_empty=True liefert
def test_prepare_retrievers_symmetrical_returns_is_empty_true():
    # 1. Mock für den Vector Store aufsetzen
    mock_vector_store = MagicMock()
    # Simuliere, dass die DB für den Filter 0 Dokumente findet
    mock_vector_store.get.return_value = {"documents": [], "metadatas": []}

    mock_bm25 = MagicMock()
    
    search_kwargs = {
        "k": 8,
        "filter": {"paragraph": "12345"}  # Nicht existierender Filter
    }

    # 2. Funktion ausführen
    bm25_res, kwargs_res, is_empty = _prepare_retrievers_symmetrical(
        vector_store=mock_vector_store,
        search_kwargs=search_kwargs,
        global_bm25_retriever=mock_bm25,
        top_k=8
    )

    # 3. Assertions
    assert is_empty is True
    assert bm25_res is None
    mock_vector_store.get.assert_called_once_with(
        where={"paragraph": "12345"},
        include=["documents", "metadatas"]
    )


# Testet, ob _create_hybrid_retriever_with_score direkt den EmptyRetriever zurückgibt
def test_create_hybrid_retriever_returns_empty_retriever():
    mock_vector_store = MagicMock()
    mock_vector_store.get.return_value = {"documents": [], "metadatas": []}
    mock_bm25 = MagicMock()

    search_kwargs = {
        "k": 8,
        "filter": {"paragraph": "NON_EXISTENT_PARAGRAPH"}
    }

    retriever = _create_hybrid_retriever_with_score(
        vector_store=mock_vector_store,
        search_kwargs=search_kwargs,
        global_bm25_retriever=mock_bm25
    )

    # Prüfen, ob eine Instanz des EmptyRetrievers erzeugt wurde
    assert isinstance(retriever, EmptyRetriever)
    
    # Sicherstellen, dass das Aufrufen des Retrievers sofort eine leere Liste zurückgibt
    docs = retriever.invoke("Beliebige Frage")
    assert docs == []



    