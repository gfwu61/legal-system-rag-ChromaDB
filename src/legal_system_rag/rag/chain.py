# legal_system_rag/rag/chain.py

from typing import Any, List, Optional, Tuple, Dict
from copy import deepcopy
from collections import defaultdict
import re
import textwrap
import numpy as np

from pydantic import Field, ConfigDict
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from legal_system_rag.config import (
    RETRIEVER_MODE,
    TOP_K,
    TOP_BM25_K,
    WEIGHT_HYBRID,
    WEIGHT_RERANKER,
    VECTOR_STORE,
)

# Relative Imports innerhalb des Paket-Ordners
from .prompts import (
    EnrichmentOutput,
    QueryExtraction,
    build_answer_prompt,
    build_query_prompt,
)
from .reranker import rerank_documents
from .utils import (
    create_structured_body,
    min_max_normalize,
    temperature_softmax_normalize,
    build_page_content_for_rerank_from_page_content,
    print_result,
    print_hybrid_result,
    select_diverse_top_k,
)



# WICHTIG: _NATIVE_RANKER = Ranker(...) hier ENTFERNEN, da es in reranker.py steht!
# ============================================================
# LLM ENRICHMENT RETRY
# ============================================================

@retry(
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(
        multiplier=1,
        min=1,
        max=8,
    ),
    reraise=True,
)
def _invoke_enrichment_with_retry(
    prompt: str,
    structured_llm,
) -> EnrichmentOutput:
    return structured_llm.invoke(prompt)


# ============================================================
# CHUNK ENRICHMENT
# ============================================================

def enrich_chunk(
    text: str,
    structured_llm,
) -> EnrichmentOutput:
    """
    Enrich a legal text chunk with structured search-oriented information.
    """
    prompt = f"""Du bist ein juristischer KI-Assistent für deutsches Recht.

Analysiere ausschließlich den folgenden ORIGINALTEXT und erstelle daraus
ein strukturiertes JSON.

GRUNDREGEL:
Der ORIGINALTEXT ist die einzige rechtliche Quelle.
Das Enrichment darf keine zusätzlichen rechtlichen Informationen
enthalten und darf die Bedeutung des Gesetzestextes nicht verändern.

REGELN:

1. "topic"
   Beschreibe den tatsächlichen Regelungsgegenstand neutral und präzise.
   Weise eine Regelung nur dann dem Mieter oder Vermieter zu, wenn dies
   im ORIGINALTEXT ausdrücklich erkennbar ist.

2. "plain_language_summary"
   Fasse den vollständigen Inhalt des ORIGINALTEXTES verständlich und
   neutral zusammen.
   Lass keine rechtlich relevanten Voraussetzungen, Fristen, Ausnahmen
   oder Einschränkungen weg.
   Übertrage eine Einschränkung auf eine Partei niemals auf andere
   Sätze oder Regelungen.

3. "user_questions"
   Erstelle 3 bis 5 typische Fragen, die Bürger unmittelbar zu diesem
   ORIGINALTEXT stellen könnten.

   Die Fragen müssen:
   - direkt aus dem ORIGINALTEXT ableitbar sein,
   - die wesentlichen Regelungen abdecken,
   - konkrete Fristen, Voraussetzungen und Ausnahmen berücksichtigen,
   - zwischen Mieter und Vermieter unterscheiden, wenn der ORIGINALTEXT
     dies ausdrücklich tut.

   Verwende keine Frage, deren Beantwortung zusätzliches Wissen über
   deutsches Recht voraussetzt.

4. "keywords"
   Erzeuge zentrale Begriffe aus dem ORIGINALTEXT.
   Die Keywords sollen für semantische Suche und typische Nutzerfragen
   relevant sein.

WICHTIG:
- Keine Halluzinationen.
- Keine Ergänzung aus allgemeinem Rechtswissen.
- Keine Umdeutung des ORIGINALTEXTES.
- ORIGINALTEXT > topic > plain_language_summary > user_questions > keywords.

ORIGINALTEXT:
{text}
"""

    try:
        return _invoke_enrichment_with_retry(prompt, structured_llm)
    except Exception as e:
        print(f"  ⚠️ LLM-Enrichment endgültig fehlgeschlagen nach Retries: {e}")

    return EnrichmentOutput(
        topic="Unbekannt",
        plain_language_summary=text,
        user_questions=[
            "Welche Regelung enthält dieser Gesetzestext?",
            "Welche Voraussetzungen nennt der Gesetzestext?",
            "Welche Fristen oder Ausnahmen enthält der Gesetzestext?",
        ],
        keywords=[],
    )



    

# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(
    input_data: dict,
    llm_answer,
    answer_prompt,
) -> dict:
    """
    Generate the final answer exclusively from retrieved documents.
    """
    if not isinstance(input_data, dict) or "docs" not in input_data:
        return {
            "answer": "Fehler in der Verarbeitungskette.",
            "docs": [],
        }

    question = input_data["question"]
    docs = input_data["docs"]
    

    if not docs:
        return {
            "answer": "Ich konnte keine passenden Dokumente finden.",
            "docs": [],
        }

    context_str = "\n\n".join([doc.page_content for doc in docs])

    print("\n" + "=" * 80)
    print("ANSWER CONTEXT")
    print("=" * 80)
    print(context_str)
    print("=" * 80 + "\n")

    prompt_value = answer_prompt.invoke(
        {
            "context": context_str,
            "question": question,
        }
    )

    response = llm_answer.invoke(prompt_value)

    return {
        "answer": response.content,
        "docs": docs,
    }


# ============================================================
# BUILD RAG CHAIN
# ============================================================

# retriever_mode=1, without score;2: with score
    
def build_rag_chain(
    llm_query, 
    llm_answer, 
    vector_store, 
    global_bm25, 
    retriever_mode: int = 1  
):
    query_chain = build_query_prompt() | llm_query.with_structured_output(QueryExtraction)
    answer_prompt = build_answer_prompt()

    # 1: normal, 2: with score
    # RETRIEVER_MODE=1, without score;2: with score
    return (
        {"question": RunnablePassthrough()}
        | RunnableLambda(lambda x: retrieve_documents(x, query_chain, vector_store, global_bm25, retriever_mode))
        | RunnableLambda(lambda x: generate_answer(x, llm_answer, answer_prompt))
    )

    
# ============================================================
# BUILD PAGE CONTENT
# ============================================================



def build_page_content(
    gesetz: str,
    paragraph: str,
    paragraph_title: str,
    absatz: str,
    nummer: str | None,
    subsection_text: str | None,
    nummer_text: str | None,
    references: List[str],
    enrichment: EnrichmentOutput,
) -> str:
    # 1. Fallbacks behandeln
    num_str = nummer if nummer else "-"
    ref_str = ", ".join(references) if references else "-"
    user_questions = "\n".join([f"- {q}" for q in enrichment.user_questions])
    keywords = ", ".join(enrichment.keywords)

    # 2. Originaltext zusammenbauen
    structured_body = create_structured_body(subsection_text, nummer_text)

    # 3. Finalen Chunk aufbauen
    content = f"""GESETZ: {gesetz}
PARAGRAPH: §{paragraph}
PARAGRAPH_TITEL: {paragraph_title}
ABSATZ: {absatz}
NUMMER: {num_str}
REFERENZEN: {ref_str}

ORIGINALTEXT:
{structured_body}

THEMA: {enrichment.topic}
KEYWORDS: {keywords}
TYPISCHE NUTZERFRAGEN:
{user_questions}
KLARTEXT: {enrichment.plain_language_summary}"""

    return textwrap.dedent(content).strip()


# ============================================================
# HYBRID RETRIEVAL HELPERS & SCORE BLENDING
# ============================================================

def initialize_global_bm25(vector_store) -> BM25Retriever:
    """
    Lädt alle Dokumente einmalig aus ChromaDB und baut den globalen BM25-Index.
    Die Datenbank selbst wird dabei NICHT verändert.
    """
    print("🔄 Initialisiere globalen BM25-Retriever für die gesamte DB...")
    
    # Alle Dokumente aus der ChromaDB abrufen
    raw_db_data = vector_store.get(include=["documents", "metadatas"])
    
    all_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw_db_data["documents"], raw_db_data["metadatas"])
    ]
    
    # BM25 über die VOLLSTÄNDIGE Datenbank erstellen
    global_bm25 = BM25Retriever.from_documents(all_docs)
    global_bm25.k = TOP_BM25_K  # Wie viele Keyword-Treffer standardmäßig geholt werden
    
    print(f"✅ BM25-Index für {len(all_docs)} Dokumente erfolgreich im RAM geladen.\n")
    return global_bm25




# weight_hybrid/weight_reranker: 0.4/0.6, now: 0.7/0.3, 0.8/0.2
# base_retriever - top_k: Pydantic model fields
# base_retriever: BaseRetriever: not Pydantic known data type
# ranker:  RANKER,  not Pydantic known data type

# ---------------------------------------------------------------------------
# 1. BlendedFlashRankRetriever (Pydantic V2 kompatibel)
# ---------------------------------------------------------------------------

# A lightweight retriever that immediately returns an empty list.
# It avoids any database or vector-store queries when no documents match a filter.
class EmptyRetriever(BaseRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return []


def _prepare_retrievers_symmetrical(
    vector_store,
    search_kwargs: dict,
    global_bm25_retriever: BM25Retriever,
    top_k: int,
) -> Tuple[Optional[BM25Retriever], dict, bool]:
    """
    Prepare the BM25 and vector retrievers using the same filter context.

    Returns:
        A tuple containing:
        - bm25_retriever: A filtered BM25 retriever, the global retriever,
          or None if no documents match the filter.
        - vector_search_kwargs: Search settings passed to the vector retriever.
        - is_empty: True when the filter matches zero documents.
    """
    db_filter = search_kwargs.get("filter")
    k = search_kwargs.get("k", top_k)

    # Copy the settings to avoid modifying the original dictionary.
    vector_search_kwargs = deepcopy(search_kwargs)

    # Case 1: No metadata filter is provided.
    # Both retrievers work on the complete document collection.
    if not db_filter:
        global_bm25_retriever.k = k
        return global_bm25_retriever, vector_search_kwargs, False

    # Case 2: A metadata filter is provided.
    # Retrieve all documents matching the filter to build a matching BM25 index.
    raw_db_data = vector_store.get(
        where=db_filter,
        include=["documents", "metadatas"],
    )

    # Convert vector-store results into LangChain Document objects.
    filtered_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(
            raw_db_data.get("documents", []),
            raw_db_data.get("metadatas", []),
        )
    ]

    # If the filter matches no documents, do not create retrievers
    # and signal the caller to return an empty result immediately.
    if not filtered_docs:
        return None, vector_search_kwargs, True

    # Build a temporary BM25 retriever containing only filtered documents.
    # Limit k so it never exceeds the number of available documents.
    bm25_retriever = BM25Retriever.from_documents(filtered_docs)
    bm25_retriever.k = min(k, len(filtered_docs))

    return bm25_retriever, vector_search_kwargs, False


def _create_hybrid_retriever(
    vector_store,
    search_kwargs: dict,
    global_bm25_retriever: BM25Retriever,
) -> BaseRetriever:
    """
    Create a hybrid retriever that combines vector similarity search and BM25.

    If a supplied metadata filter matches no documents, return EmptyRetriever
    to prevent unnecessary database queries.
    """
    k = search_kwargs.get("k", TOP_K)
    bm25_retriever, vector_search_kwargs, is_empty = _prepare_retrievers_symmetrical(
        vector_store=vector_store,
        search_kwargs=search_kwargs,
        global_bm25_retriever=global_bm25_retriever,
        top_k=k,
    )

    # Hard constraint:
    # If no documents satisfy the filter, return an empty retriever immediately.
    if is_empty:
        return EmptyRetriever()

    # Create a vector retriever using similarity search and the same filters.
    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=vector_search_kwargs,
    )

    # Combine semantic vector search and lexical BM25 search equally.
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )


    
# ---------------------------------------------------------------------------
# Hybrid-Retriever mit Score
# ---------------------------------------------------------------------------
def _create_hybrid_retriever_with_score(
    vector_store,
    search_kwargs: dict,
    global_bm25_retriever: BM25Retriever,
):
    """
    Creates a hybrid retriever that retrieves real scores from dense (vector) 
    and sparse (BM25) searches, then combines them using weighted score blending.
    
    Uses `_prepare_retrievers_symmetrical` to ensure consistent filter application 
    across both search methods. Returns an `EmptyRetriever` if no documents match the filter.
    """
    k = search_kwargs.get("k", TOP_K)
    # Prepare both retrievers using the symmetrical helper function.
    bm25_retriever, vector_search_kwargs, is_empty = _prepare_retrievers_symmetrical(
        vector_store=vector_store,
        search_kwargs=search_kwargs,
        global_bm25_retriever=global_bm25_retriever,
        top_k=k,
    )

    # Return immediately if the metadata filter yields zero matching documents.
    if is_empty:
        return EmptyRetriever()

    # Custom LangChain retriever class that extracts and merges dense and sparse scores.
    class CustomHybridScoreRetriever(BaseRetriever):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        v_store: Any
        bm25: Any
        retrieval_k: int
        filter_kwargs: Optional[dict] = None

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        ) -> List[Document]:
            # Extract the metadata filter parameter for dense retrieval.
            filter_value = self.filter_kwargs.get("filter") if self.filter_kwargs else None

            # 1) Dense retrieval:
            # Fetch relevance scores from the vector store if supported, otherwise standard similarity scores.
            if hasattr(self.v_store, "similarity_search_with_relevance_scores"):
                docs_scores = self.v_store.similarity_search_with_relevance_scores(
                    query,
                    k=self.retrieval_k,
                    filter=filter_value,
                )
            else:
                docs_scores = self.v_store.similarity_search_with_score(
                    query,
                    k=self.retrieval_k,
                    filter=filter_value,
                )

            # Convert dense search output into explicit (Document, float_score) pairs.
            dense_pairs = [(doc, float(score)) for doc, score in docs_scores]

            print("\n=========================================")
            print("\n--- Dense: search_with_score, sorted, not normalized ---")
            for i, (doc, score) in enumerate(dense_pairs, start=1):
                print(
                    f"Document {i} | Score: {score:.4f} | "
                    f"paragraph={doc.metadata.get('paragraph')} | "
                    f"absatz={doc.metadata.get('absatz')} | "
                    f"nummer={doc.metadata.get('nummer')}"
                )
            print("----------------------\n")

            # 2) Sparse retrieval:
            # Calculate BM25 scores using the internal preprocessor and vectorizer.
            processed_query = self.bm25.preprocess_func(query)
            scores = self.bm25.vectorizer.get_scores(processed_query)
            docs = list(self.bm25.docs)

            # Pair documents with their BM25 score, sort in descending order, and limit to k.
            sparse_pairs = [(doc, float(score)) for doc, score in zip(docs, scores)]
            sparse_pairs.sort(key=lambda x: x[1], reverse=True)
            sparse_pairs = sparse_pairs[: self.bm25.k]

            print("\n--- Sparse: BM25 scores, sorted, not normalized ---")
            for i, (doc, score) in enumerate(sparse_pairs, start=1):
                print(
                    f"Document {i} | Score: {score:.4f} | "
                    f"paragraph={doc.metadata.get('paragraph')} | "
                    f"absatz={doc.metadata.get('absatz')} | "
                    f"nummer={doc.metadata.get('nummer')}"
                )
            print("----------------------\n")

            # 3) Score fusion:
            # Merge dense and sparse scores into a unified mapping using page_content as the primary key.
            score_map = defaultdict(lambda: {"doc": None, "dense": None, "sparse": None})

            for doc, score in dense_pairs:
                key = doc.page_content
                score_map[key]["doc"] = doc
                score_map[key]["dense"] = score

            for doc, score in sparse_pairs:
                key = doc.page_content
                score_map[key]["doc"] = doc
                score_map[key]["sparse"] = score

            # Collect existing scores to calculate Min-Max normalization bounds.
            dense_scores = [v["dense"] for v in score_map.values() if v["dense"] is not None]
            sparse_scores = [v["sparse"] for v in score_map.values() if v["sparse"] is not None]

            dense_minmax = {}
            sparse_minmax = {}

            # Perform Min-Max normalization for dense scores [0, 1].
            if dense_scores:
                d_arr = np.array(dense_scores, dtype=float)
                d_mn, d_mx = d_arr.min(), d_arr.max()
                for key, v in score_map.items():
                    if v["dense"] is not None:
                        dense_minmax[key] = 1.0 if d_mx == d_mn else (v["dense"] - d_mn) / (d_mx - d_mn)

            # Perform Min-Max normalization for sparse scores [0, 1].
            if sparse_scores:
                s_arr = np.array(sparse_scores, dtype=float)
                s_mn, s_mx = s_arr.min(), s_arr.max()
                for key, v in score_map.items():
                    if v["sparse"] is not None:
                        sparse_minmax[key] = 1.0 if s_mx == s_mn else (v["sparse"] - s_mn) / (s_mx - s_mn)

            # Blend the normalized dense and sparse scores with equal weight (0.5 / 0.5).
            blended = []
            for key, v in score_map.items():
                d = dense_minmax.get(key, 0.0)
                s = sparse_minmax.get(key, 0.0)
                final = 0.5 * d + 0.5 * s

                # Attach individual and final scores to document metadata for debugging/inspection.
                doc = v["doc"]
                meta = dict(doc.metadata)
                meta["dense_score"] = round(float(d), 4)
                meta["bm25_score"] = round(float(s), 4)
                meta["score"] = round(float(final), 4)

                blended.append(Document(page_content=doc.page_content, metadata=meta))

            # Sort candidate documents by final blended score in descending order.
            blended.sort(key=lambda d: d.metadata["score"], reverse=True)

            print_hybrid_result(
                "Hybrid Result (blended=dense*0.5 + bm25*0.5) -> sorted by blended score, normalized",
                blended,
            )

            return blended[: self.retrieval_k]

    return CustomHybridScoreRetriever(
        v_store=vector_store,
        bm25=bm25_retriever,
        retrieval_k=k,
        filter_kwargs=vector_search_kwargs,
    )

class BlendedFlashRankRetriever(BaseRetriever):
    """
    Custom LangChain Retriever, der ein Hybrid-Retrieval mit FlashRank Cross-Encoder
    über Weighted Score Blending kombiniert.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    base_retriever: Any
    weight_hybrid: float = 0.4
    weight_reranker: float = 0.6
    top_k: int = TOP_K
    temperature: float = 0.2  # Steuerparameter für die Reranker-Spreizung

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:

        config = {}
        if run_manager:
            config = {"callbacks": run_manager.get_child()}

        initial_docs = self.base_retriever.invoke(query, config=config)

        blended_docs = rerank_documents(
            query=query,
            docs_hybrid=initial_docs,
            weight_hybrid=self.weight_hybrid,
            weight_reranker=self.weight_reranker,
            top_k=self.top_k,   
            temperature=self.temperature  
        )

        return blended_docs

        


# ---------------------------------------------------------------------------
# 2. Haupt-Router (Gefixt: Korrekter Funktionsaufruf im else-Zweig)
# ---------------------------------------------------------------------------

def retrieve_documents(
    input_data: dict,
    query_chain,
    vector_store,
    global_bm25: BM25Retriever,
    mode: int = 2
) -> dict:

    if mode == 1:
        # Nur Reranking ohne Score-Blending der Basis-Retriever
        return retrieve_documents_no_score(input_data, query_chain, vector_store, global_bm25)
    else:
        # Korrigiert: Richtiges Routing für Mode 2
        return retrieve_documents_with_score(input_data, query_chain, vector_store, global_bm25)


# ---------------------------------------------------------------------------
# 3. retrieve_documents_no_score
# ---------------------------------------------------------------------------

def retrieve_documents_no_score(
    input_data: dict,
    query_chain,
    vector_store,
    global_bm25: BM25Retriever
) -> dict:
    # Accept either a dictionary containing "question" or a raw question string.
    question = input_data["question"] if isinstance(input_data, dict) else input_data

    # 1. Extract a refined search query and optional paragraph filters.
    # If query extraction fails, use the original user question without filters.
    try:
        extracted = query_chain.invoke({"question": question})
        search_phrase = extracted.search_query if extracted.search_query else question
        p_filters = extracted.paragraph_filter if extracted.paragraph_filter else []
    except Exception as e:
        print(f"  ⚠️ Query extraction failed: {e}")
        search_phrase = question
        p_filters = []

    # Log the original question, extracted search phrase, and active paragraph filters.
    print(f"\n🔍 [Original Query] Question: '{question}'")
    print(f"🔍🔍 [Query Analysis] Search phrase: '{search_phrase}' | Active paragraph filters: {p_filters}")

    # 2. Prepare metadata filters for symmetrical vector and BM25 retrieval.
    clean_filters = []
    if p_filters:
        # Remove paragraph symbols, trim whitespace, and ignore empty values.
        clean_filters = [
            str(p).replace("§", "").strip()
            for p in p_filters
            if str(p).strip()
        ]

    has_filter = len(clean_filters) > 0

    # Retrieve more candidates than the final TOP_K because reranking and
    # diversity selection will reduce the final result set later.
    RETRIEVAL_K = TOP_K * 4

    search_kwargs = {}

    if has_filter:
        # Scale the candidate count based on the number of requested paragraphs.
        # Each paragraph should have enough candidates available for reranking.
        num_paragraphs = len(clean_filters)
        # Dynamically scale retrieval depth based on paragraph filter count.
        calculated_k = max(num_paragraphs * 4, RETRIEVAL_K)
        search_kwargs["k"] = calculated_k
        
        # Use a direct metadata filter for one paragraph and an OR filter
        # when the query references multiple paragraphs.        
        if num_paragraphs == 1:
            search_kwargs["filter"] = {"paragraph": clean_filters[0]}
        else:
            search_kwargs["filter"] = {"$or": [{"paragraph": p} for p in clean_filters]}
    else:
        search_kwargs["k"] = RETRIEVAL_K


    # 3. Create and execute the hybrid retriever.
    # It combines semantic vector search with lexical BM25 retrieval.
    base_hybrid_retriever = _create_hybrid_retriever(
        vector_store=vector_store,
        search_kwargs=search_kwargs,
        global_bm25_retriever=global_bm25,
    )

    docs_hybrid = base_hybrid_retriever.invoke(search_phrase)

    # Stop immediately if no documents were found.
    # This can happen when the metadata filter matches no stored documents.
    if not docs_hybrid:
        print("  ⚠️ No documents found during hybrid retrieval.")
        return {"question": question, "docs": []}
        
    print_result("docs: base_hybrid_retriever()", docs_hybrid)
    
    reranked_docs = rerank_documents(
            query= search_phrase,
            docs_hybrid= docs_hybrid,
            weight_hybrid= 0.0,
            weight_reranker= 1.0,
            top_k=TOP_K  # Keeps ONLY the top_k best chunks
        )


        
    # Log final retrieval settings and selected documents for debugging.
    print(f"search_phrase={search_phrase}")
    print(f"p_filters={p_filters}")
    print(f"search_kwargs={search_kwargs}")
    print(f"\nFINAL RETRIEVED DOCUMENTS (TOP_K={TOP_K}):")


    print_result("HYBRID RETRIEVAL + RERANKING:select_diverse_top_k+sorted", reranked_docs)
 
    return {"question": question, "docs": reranked_docs}


# ---------------------------------------------------------------------------
# 4. retrieve_documents_with_score
# ---------------------------------------------------------------------------
def retrieve_documents_with_score(
    input_data: dict,
    query_chain,
    vector_store,
    global_bm25: BM25Retriever,
) -> dict:
    """
    Retrieves documents using a hybrid search (Dense + Sparse BM25) combined 
    with FlashRank cross-encoder reranking and diverse top-k selection.
    """
    # Extract the user question safely from dict or string input.
    question = input_data["question"] if isinstance(input_data, dict) else input_data

    # Extract search query and paragraph filters via query analysis chain.
    try:
        extracted = query_chain.invoke({"question": question})
        search_phrase = extracted.search_query if extracted.search_query else question
        p_filters = extracted.paragraph_filter if extracted.paragraph_filter else []
    except Exception as e:
        print(f"  ⚠️ Query analysis extraction failed: {e}")
        search_phrase = question
        p_filters = []
        
    print(f"\n🔍 [Original Query] Question: '{question}'")
    print(f"\n🔍 [Query Analysis] Search Phrase: '{search_phrase}' | Active Paragraph Filters: {p_filters}")

    has_filter = False
    clean_filters = []
    RETRIEVAL_K = TOP_K * 4

    # Sanitize and format paragraph filters if provided.
    if p_filters:
        clean_filters = [str(p).replace("§", "").strip() for p in p_filters if p]
        if clean_filters:
            has_filter = True

    search_kwargs = {}
    if has_filter:
        num_paragraphs = len(clean_filters)
        # Dynamically scale retrieval depth based on paragraph filter count.
        calculated_k = max(num_paragraphs * 4, RETRIEVAL_K)
        search_kwargs["k"] = calculated_k
        
        if num_paragraphs == 1:
            search_kwargs["filter"] = {"paragraph": clean_filters[0]}
        else:
            search_kwargs["filter"] = {"$or": [{"paragraph": p} for p in clean_filters]}
    else:
        search_kwargs["k"] = RETRIEVAL_K

    print(" >>> Initializing Custom Hybrid Score Retriever...\n")



    
    # 1. Instantiate the score-aware hybrid retriever (Vector + Symmetrical BM25).
    base_hybrid_retriever = _create_hybrid_retriever_with_score(
        vector_store=vector_store,
        search_kwargs=search_kwargs,
        global_bm25_retriever=global_bm25,
    )

    # 2. Combine Hybrid Search results with FlashRank Cross-Encoder Reranking.
    print(" >>> Initializing BlendedFlashRankRetriever (Hybrid + Reranker)...")
    print(f" >>> WEIGHT_HYBRID: {WEIGHT_HYBRID}, WEIGHT_RERANKER: {1.0 - WEIGHT_HYBRID}\n")
    
    blended_retriever = BlendedFlashRankRetriever(
        base_retriever= base_hybrid_retriever,
        weight_hybrid= WEIGHT_HYBRID,
        weight_reranker= 1.0 - WEIGHT_HYBRID,
        top_k= TOP_K,
        temperature=0.2,
    )

    # Execute document retrieval pipeline.
    docs = blended_retriever.invoke(search_phrase)


    # 3. Apply post-retrieval diversity filter and truncate to final TOP_K limit.

    print(f"search_phrase = {search_phrase}")
    print(f"p_filters = {p_filters}")
    print(f"search_kwargs = {search_kwargs}")
    print(f"\nFINAL RETRIEVED DOCUMENTS (TOP_K={TOP_K}):")

    print_result("2. Final Selected Documents (after diversity filter)", docs)

    return {
        "question": question,
        "docs": docs,
    }

    




    
        