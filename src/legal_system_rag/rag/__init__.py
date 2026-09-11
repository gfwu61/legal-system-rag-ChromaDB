"""
RAG pipeline initialization module.
Provides access to prompt structures, retrieval logic, chain building, and reranking utilities.
"""

from .chain import (
    build_page_content,
    build_rag_chain,
    enrich_chunk,
    generate_answer,
    retrieve_documents,
)
from .prompts import (
    EnrichmentOutput,
    QueryExtraction,
    build_answer_prompt,
    build_query_prompt,
)

from .reranker import (
    rerank_documents,
)
from .utils import (
    build_page_content_for_rerank_from_page_content,
    create_structured_body,
    min_max_normalize,
    print_result,
    select_diverse_top_k,
    temperature_softmax_normalize,
)

__all__ = [
    # Prompts & Schemas
    "EnrichmentOutput",
    "QueryExtraction",
    "build_answer_prompt",
    "build_query_prompt",
    # Chain & Retrieval Core
    "enrich_chunk",
    "build_page_content",
    "retrieve_documents",
    "generate_answer",
    "build_rag_chain",
    # Text Processing & Normalization
    "create_structured_body",
    "min_max_normalize",
    "temperature_softmax_normalize",
    "build_page_content_for_rerank_from_page_content",
    "print_result",
    "select_diverse_top_k",
    # Models & Rankers
    "_NATIVE_RANKER",
    "rerank_documents",
]