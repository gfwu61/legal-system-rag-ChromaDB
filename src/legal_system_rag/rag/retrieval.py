from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from .router import RouterDecision
from legal_system_rag.config import RETRIEVER_MODE, TOP_K


def get_routed_retriever(
    decision: RouterDecision,
    vector_store,
    global_bm25: BM25Retriever,
    top_k: int = TOP_K,
    k_multiplier: int = 4
) -> EnsembleRetriever:
    """
    Constructs a routed EnsembleRetriever based on the router's decision.
    Fetches an expanded candidate set (top_k * k_multiplier) to provide sufficient
    depth for subsequent reranking.
    """
    # Calculate expanded candidate count for high recall prior to reranking
    retrieval_k = top_k * k_multiplier

    # ROUTE 1: Specific Norm Retrieval
    if decision.route == "specific_norm" and decision.paragraph_filters:
        clean_filters = [p.replace("§", "").strip() for p in decision.paragraph_filters if p]
        
        if len(clean_filters) == 1:
            db_filter = {"paragraph": clean_filters[0]}
        elif len(clean_filters) > 1:
            db_filter = {"$or": [{"paragraph": p} for p in clean_filters]}
        else:
            db_filter = None

        if db_filter:
            print(f"🔀 [Route: Specific Norm] Applying strict metadata filter: {db_filter} (Retrieval K={retrieval_k})")

            # Pre-filter vector store chunks to build a targeted local BM25 index
            raw_db_data = vector_store.get(where=db_filter, include=["documents", "metadatas"])
            filtered_docs = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(raw_db_data.get("documents", []), raw_db_data.get("metadatas", []))
            ]

            if filtered_docs:
                # Build temporary in-memory BM25 index restricted to filtered chunks
                bm25 = BM25Retriever.from_documents(filtered_docs)
                bm25.k = retrieval_k
                
                vector_retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": retrieval_k, "filter": db_filter}
                )
                return EnsembleRetriever(retrievers=[vector_retriever, bm25], weights=[0.5, 0.5])
            else:
                print(f"⚠️ [Route: Specific Norm] No documents found for filter {db_filter}. Falling back to Concept Search.")

    # ROUTE 2: Concept Search (Default / Fallback)
    print(f"🔀 [Route: Concept Search] Executing global search without metadata filters (Retrieval K={retrieval_k}).")
    
    global_bm25.k = retrieval_k
    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": retrieval_k}
    )
    return EnsembleRetriever(retrievers=[vector_retriever, global_bm25], weights=[0.5, 0.5])




    