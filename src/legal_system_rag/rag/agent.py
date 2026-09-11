from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from legal_system_rag.config import TOP_K

from .router import build_llm_router, RouterDecision
from .retrieval import get_routed_retriever
from .evaluators import evaluate_document_relevance, rewrite_search_query
from .reranker import rerank_documents
from .prompts import build_query_prompt



    
def run_agentic_retrieval_loop(
    question: str,
    llm_query: BaseChatModel,
    vector_store: Any,
    global_bm25: Any,
    max_retries: int = 2,
    top_k: int = TOP_K
) -> dict:
    """
    Executes an agentic retrieval loop using query routing, hybrid retrieval,
    native cross-encoder reranking, and relevance evaluation with query rewriting.
    """
    router = build_llm_router(llm=llm_query)
    current_question = question
    attempt = 0

    while attempt < max_retries:
        print(f"\n🔄 [Agentic Loop] Attempt {attempt + 1}/{max_retries}")


        # Step 1: Router decides search strategy
        decision: RouterDecision = router.invoke({"question": current_question})
        print( f"🔀 question='{question}'")    
        print(
            f"🔀 [Router] Route='{decision.route}' | "
            f"Query='{decision.search_query}' | "
            f"Filters={decision.paragraph_filters}"
        )

       
        # Step 2: Retrieve candidate chunks (retrieval_k = top_k * 4)
        retriever = get_routed_retriever(
            decision=decision,
            vector_store=vector_store,
            global_bm25=global_bm25,
            top_k=top_k,
            k_multiplier=4  # Fetches top_k * 4 candidate chunks
        )

        # raw_docs = retriever.invoke(decision.search_query)
        raw_docs = retriever.invoke(decision.search_query)

        # Step 3: Native Reranking & Diversity Selection (Filters down to exact top_k)
        print("🎯 [Reranker] Reranking and filtering candidates...")
        reranked_docs = rerank_documents(
            query=decision.search_query,
            docs_hybrid=raw_docs,
            top_k=top_k  # Keeps ONLY the top_k best chunks
        )

        # Step 4: Relevance evaluation on the filtered top_k chunks
        relevance = evaluate_document_relevance(
            question=question, 
            docs=reranked_docs, 
            llm=llm_query
        )
        print(f"📊 [Evaluation] Relevance score: '{relevance}'")

        if relevance == "yes" or attempt == max_retries - 1:
            return {
                "question": question,
                "docs": reranked_docs,
                "attempts": attempt + 1,
                "route_used": decision.route
            }

        # Step 5: Rewrite query if documents are deemed irrelevant
        print("⚠️ Chunks insufficient after reranking. Rewriting search query...")
        rewritten_query = rewrite_search_query(
            question=question, 
            current_query=decision.search_query,
            llm=llm_query
        )
        current_question = rewritten_query
        attempt += 1

    return {"question": question, "docs": [], "attempts": attempt, "route_used": None}


