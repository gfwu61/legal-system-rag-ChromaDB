import textwrap

from typing import Any, List, Optional, Tuple, Dict
from collections import defaultdict
import re
import textwrap
import numpy as np

from pydantic import Field, ConfigDict
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from flashrank import Ranker, RerankRequest

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from legal_system_rag.config import RETRIEVER_MODE, TOP_K, TOP_BM25_K, WEIGHT_HYBRID, WEIGHT_RERANKER, VECTOR_STORE
from legal_system_rag.rag.prompts import (
    EnrichmentOutput,
    QueryExtraction,
    build_answer_prompt,
    build_query_prompt,
)

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

# Globales Ranker-Modell einmalig initialisieren, um Overhead bei jedem Call zu vermeiden
_NATIVE_RANKER = Ranker(model_name="ms-marco-MiniLM-L-12-v2")


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

def build_rag_chain(llm_query, llm_answer, vector_store):
    global_bm25 = initialize_global_bm25(vector_store)
    
    query_chain = build_query_prompt() | llm_query.with_structured_output(QueryExtraction)
    answer_prompt = build_answer_prompt()

    # 1: normal, 2: with score
    return (
        {"question": RunnablePassthrough()}
        | RunnableLambda(lambda x: retrieve_documents(x, query_chain, vector_store, global_bm25, RETRIEVER_MODE))
        | RunnableLambda(lambda x: generate_answer(x, llm_answer, answer_prompt))
    )

    
# ============================================================
# BUILD PAGE CONTENT
# ============================================================

def create_structured_body(subsection_text: str | None, nummer_text: str | None) -> str:
    body_lines = [p.strip() for p in [subsection_text, nummer_text] if p and p.strip() != "-"]
    return "\n".join(body_lines)


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

def print_result(titel: str, docs):
    print()
    print(f"  >>>  {titel} <<< ")
    print("=" * 40 + f" docs: {titel} " + "=" * 40)

    for i, doc in enumerate(docs, start=1):
        md = doc.metadata
        print(
            f"Document {i} | "
            f"hybrid_norm_score={md.get('hybrid_norm_score', 0.0):.4f} | "
            f"rerank_norm_score={md.get('rerank_norm_score', 0.0):.4f} | "
            f"blended_score={md.get('blended_score', 0.0):.4f} | "
            f"paragraph={md.get('paragraph', '')} | "
            f"absatz={md.get('absatz', '')} | "
            f"nummer={md.get('nummer', '')} | "
            f"thema={md.get('thema', '')}"
        )

    print("=" * 80 + "\n")
    

    

def _create_hybrid_retriever(
    vector_store,
    search_kwargs: dict,
    global_bm25_retriever: BM25Retriever,
) -> EnsembleRetriever:
    """
    Erstellt den Hybrid Retriever. Nutzen Sie den globalen BM25 für reine Fragen
    oder filtern Sie BM25 bei Paragraphen-Suchen dynamisch.
    """
    # 1. Dense Vector Retriever (sucht über die ganze DB oder mit Filter)
    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    db_filter = search_kwargs.get("filter")
    k = search_kwargs.get("k", TOP_K)
    
    # 2. Sparse BM25 Retriever
    if db_filter:
        # FALL A: Wenn ein Paragraphen-Filter aktiv ist (z. B. § 573c),
        # bauen wir BM25 kurz nur für diesen spezifischen Paragraphen
        raw_db_data = vector_store.get(where=db_filter, include=["documents", "metadatas"])
        filtered_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(raw_db_data["documents"], raw_db_data["metadatas"])
        ]
        # (calculate the index for filtered_docs here)
        # later in: base_hybrid_retriever.invoke(search_phrase)
        bm25_retriever = BM25Retriever.from_documents(filtered_docs)
        bm25_retriever.k = min(len(filtered_docs), 30)
    else:
        # FALL B: Laien-Frage OHNE Filter -> Echter globaler BM25-Index!
        # search_phrase compares with global_docs, but do  it later base_hybrid_retriever.invoke(search_phrase)
        bm25_retriever = global_bm25_retriever
        bm25_retriever.k = search_kwargs.get("k", TOP_K)

    # 3. Hybrid Search (RRF mit Vector + BM25)
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )

def min_max_normalize(scores: List[float]) -> np.ndarray:
    """Normalisiert ein Score-Array strikt auf den Bereich [0.0, 1.0]."""
    if len(scores) == 0:
        return np.array([])
    
    scores_arr = np.array(scores, dtype=float)
    min_val = float(np.min(scores_arr))
    max_val = float(np.max(scores_arr))
    
    # Skalare implizit abfangen (min_val == max_val prüft zwei Floats)
    if np.isclose(min_val, max_val):
        return np.ones_like(scores_arr)
    diff= (max_val - min_val)
    return (scores_arr - min_val) / diff

    

def extract_text_from_page_content(search_text: str, page_content: str) -> str:
    pattern = rf"{re.escape(search_text)}:\s*(.*?)(?=\n[A-Z_]+:|$)"
    match = re.search(pattern, page_content, re.DOTALL)
    return match.group(1).strip() if match else ""

def build_page_content_for_rerank_from_page_content(page_content: str) -> str:
    paragraph = extract_text_from_page_content("PARAGRAPH", page_content)
    paragraph_title = extract_text_from_page_content("PARAGRAPH_TITEL", page_content)
    original_text = extract_text_from_page_content("ORIGINALTEXT", page_content)

    # Baut eine natürliche Überschrift: "§ 573 Ordentliche Kündigung des Vermieters"
    header = f"{paragraph} {paragraph_title}".strip()
    if not header.startswith("§"):
        header = f"§ {header}"

    return f"{header}\n{original_text}".strip()



# ============================================================
# RETRIEVAL PIPELINE
# ============================================================
# get the max_k docs (e.g. 30 docs, return  max_k=6  docs)
#
def select_diverse_top_k(docs, max_k=6, max_per_paragraph=2):
    """
    Selects the best documents but avoids too many paragraphs from the same section 
    to leave room for other statutes (such as § 573c).
    """
    selected = []
    seen_paragraphs = {}
    skipped_docs = []

    # 1. Durchgang: Diversität sichern
    for doc in docs:
        p_num = doc.metadata.get("paragraph")
        count = seen_paragraphs.get(p_num, 0)
        
        if count < max_per_paragraph:
            selected.append(doc)
            seen_paragraphs[p_num] = count + 1
        else:
            skipped_docs.append(doc)
            
        if len(selected) == max_k:
            return selected

    # 2. Durchgang (Fallback): Mit übersprungenen Dokumenten auffüllen, falls max_k nicht erreicht wurde
    for doc in skipped_docs:
        if len(selected) < max_k:
            selected.append(doc)
        else:
            break

    return selected


# eight_hybrid/weight_reranker: 0.4/0.6, now: 0.7/0.3, 0.8/0.2
# base_retriever - top_k: Pydantic model fields
# base_retriever: BaseRetriever: not Pydantic known data type
# ranker:  RANKER,  not Pydantic known data type

# ---------------------------------------------------------------------------
# 1. BlendedFlashRankRetriever (Pydantic V2 kompatibel)
# ---------------------------------------------------------------------------

def temperature_softmax_normalize(scores: List[float], temp: float = 0.05) -> np.ndarray:
    """
    Wendet Softmax mit Temperatur an, um eng beieinander liegende Scores
    stark zu spreizen, und skaliert das Ergebnis anschließend sauber auf [0.0, 1.0].
    
    Eine niedrige Temperatur (z. B. 0.05 oder 0.02) hebt feine Differenzen
    bei gecroppten/gesättigten Reranker-Probabilities deutlich hervor.
    """
    if len(scores) == 0:
        return np.array([])
    
    scores_arr = np.array(scores, dtype=float)
    
    # Numerisch stabile Softmax mit Temperatur
    max_score = np.max(scores_arr)
    exp_scores = np.exp((scores_arr - max_score) / temp)
    softmax_scores = exp_scores / np.sum(exp_scores)
    
    # Abschließende Min-Max-Skalierung auf [0.0, 1.0] für das Score Blending
    return min_max_normalize(softmax_scores.tolist())


class BlendedFlashRankRetriever(BaseRetriever):
    """
    Custom LangChain Retriever, der ein Hybrid-Retrieval mit FlashRank Cross-Encoder
    über Weighted Score Blending kombiniert.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: Any
    ranker: Any = Field(default_factory=lambda: _NATIVE_RANKER)
    weight_hybrid: float = 0.7
    weight_reranker: float = 0.3
    top_k: int = 8
    temperature: float = 0.05  # Steuerparameter für die Reranker-Spreizung

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:

        config = {}
        if run_manager:
            config = {"callbacks": run_manager.get_child()}

        initial_docs = self.base_retriever.invoke(query, config=config)
        if not initial_docs:
            return []

        # 1. Hybrid Scores extrahieren und standardmäßig min-max normalisieren
        raw_hybrid_scores = [
            float(doc.metadata.get("score", 0.0)) for doc in initial_docs
        ]
        norm_hybrid_scores = min_max_normalize(raw_hybrid_scores)

        # 2. Passagen für FlashRank Reranker aufbereiten
        passages = [
            {
                "id": idx,
                "text": build_page_content_for_rerank_from_page_content(doc.page_content),
                "meta": doc.metadata,
            }
            for idx, doc in enumerate(initial_docs)
        ]

        # 3. FlashRank Reranking ausführen
        rerank_request = RerankRequest(query=query, passages=passages)
        rerank_results = self.ranker.rerank(rerank_request)

        # 4. Rohe Rerank-Scores dem ursprünglichen Index zuordnen
        raw_rerank_scores = np.zeros(len(initial_docs))
        for res in rerank_results:
            raw_rerank_scores[res["id"]] = res["score"]


        # --- DEBUG PRINT ---
 

        print("\n=========================================")
        print("--- FlashRank: raw_rerank_scores (nach Reranking sortiert, not normalized) ---\n")

        
        for rank, res in enumerate(rerank_results, start=1):
            original_idx = res["id"]
            score = res["score"]
            meta = res.get("meta", {})  # Alternativ: initial_docs[original_idx].metadata
            
            print(
                f"Rank {rank:2d} (Orig Doc {original_idx + 1:2d}) | "
                f"Score: {score:.4f} | "
                f"paragraph={meta.get('paragraph')} | "
                f"absatz={meta.get('absatz')} | "
                f"nummer={meta.get('nummer')}"
            )
            
        
        # 5. Rerank-Scores mittels Temperature-Softmax stark spreizen und auf [0.0, 1.0] bringen
        norm_rerank_scores = temperature_softmax_normalize(
            raw_rerank_scores.tolist(), 
            temp=self.temperature
        )

        # 6. Gewichten und Blended Score berechnen

        print("\n==== calculate belended score for every Doc ====")
        print("--- the final rang after fusion of hybrid and rerank is not the same as the rerank ---")
        print("--- this ist because final_score = (self.weight_hybrid * h_score) + (self.weight_reranker * r_score)---")
        
        blended_docs: List[Document] = []
        for idx, doc in enumerate(initial_docs):
            h_score = norm_hybrid_scores[idx]
            r_score = norm_rerank_scores[idx]
            final_score = (self.weight_hybrid * h_score) + (self.weight_reranker * r_score)

            new_metadata = {
                **doc.metadata,
                "hybrid_norm_score": round(float(h_score), 4),
                "rerank_norm_score": round(float(r_score), 4),
                "blended_score": round(float(final_score), 4),
            }

            blended_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata=new_metadata,
                )
            )

        # 7. Nach gewichtetem Blended Score absteigend sortieren
        blended_docs.sort(key=lambda d: d.metadata["blended_score"], reverse=True)
        return blended_docs[:self.top_k]



        

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
    question = input_data["question"] if isinstance(input_data, dict) else input_data

    try:
        extracted = query_chain.invoke({"question": question})
        search_phrase = extracted.search_query if extracted.search_query else question
        p_filters = extracted.paragraph_filter if extracted.paragraph_filter else []
    except Exception as e:
        print(f"  ⚠️ Query Extraction fehlgeschlagen: {e}")
        search_phrase = question
        p_filters = []

    print(f"\n🔍 [Original Query] Question: '{question}'")
    print(f"🔍🔍 [Query Analyse] Suchphrase: '{search_phrase}' | Aktive Filter-§: {p_filters}")

    has_filter = False
    clean_filters = []
    RETRIEVAL_K = TOP_K * 4

    if p_filters:
        clean_filters = [str(p).replace("§", "").strip() for p in p_filters if p]
        if clean_filters:
            has_filter = True

    search_kwargs = {}
    if has_filter:
        anzahl_paragraphen = len(clean_filters)
        berechnetes_k = max(anzahl_paragraphen * 4, RETRIEVAL_K)
        search_kwargs["k"] = berechnetes_k
        if anzahl_paragraphen == 1:
            search_kwargs["filter"] = {"paragraph": clean_filters[0]}
        else:
            search_kwargs["filter"] = {"$or": [{"paragraph": p} for p in clean_filters]}
    else:
        search_kwargs["k"] = RETRIEVAL_K

    base_hybrid_retriever = _create_hybrid_retriever(
        vector_store=vector_store,
        search_kwargs=search_kwargs,
        global_bm25_retriever=global_bm25,
    )
    # needed to check if docs_hybrid is None
    docs_hybrid = base_hybrid_retriever.invoke(search_phrase)

    # Fallback bei 0 Treffern mit Filter
    if not docs_hybrid and has_filter:
        print(
            "  🔀 [Fallback] Keine Dokumente mit Metadaten-Filter gefunden. "
            "Starte ungefilterte Hybrid-Suche..."
        )
        fallback_kwargs = {"k": RETRIEVAL_K}
        fallback_base_retriever = _create_hybrid_retriever(
            vector_store=vector_store,
            search_kwargs=fallback_kwargs,
            global_bm25_retriever=global_bm25,
        )
        docs_hybrid = fallback_base_retriever.invoke(search_phrase)

    if not docs_hybrid:
        return {"question": question, "docs": []}

    # Logging für das Retrieval-Ergebnis
    print_result ("docs: base_hybrid_retriever()", docs_hybrid)


    # --------------------------------------------------------
    # build_page_content_for_rerank_from_page_content()
    # Pure Reranker: compact normalized text only, no additional information
    # problem with additional information
    # 1. The problem of attention dilution
    # 2. Semantic noise / false positives
    # 3. Distribution match of the reranker training data
    # solution:  build_page_content_for_rerank_from_page_content()
    # --------------------------------------------------------
    passages = [
        {
            "id": idx,
            "text": build_page_content_for_rerank_from_page_content(doc.page_content),            
            "meta": doc.metadata,
        }
        for idx, doc in enumerate(docs_hybrid)
    ]

    """
    example:
    rerank_results: sorted by score, but the id shows the identity of docs.
    [
    {
        "id": 2,
        "text": "Dies ist der Text der am besten zur Suchanfrage passt...",
        "meta": {"source": "document_2.pdf", "page": 4},
        "score": 0.9823
    },
    {
        "id": 0,
        "text": "Dies ist der Text der am zweitbesten passt...",
        "meta": {"source": "document_1.pdf", "page": 1},
        "score": 0.7412
    }, 
    {
        "id": 1,
        "text": "Dieser Text hat kaum noch Relevanz für die Suchanfrage...",
        "meta": {"source": "document_3.pdf", "page": 12},
        "score": 0.1256
    },...
    ]
    """
    rerank_request = RerankRequest(query=search_phrase, passages=passages)
    rerank_results = _NATIVE_RANKER.rerank(rerank_request)

    raw_rerank_scores = np.zeros(len(docs_hybrid))
    for res in rerank_results:
        raw_rerank_scores[res["id"]] = res["score"]

    norm_rerank_scores = min_max_normalize(raw_rerank_scores)

    docs = []
    for idx, doc in enumerate(docs_hybrid):
        new_metadata = {
            **doc.metadata,
            "rerank_norm_score": round(float(norm_rerank_scores[idx]), 4),
        }
        docs.append(
            Document(
                page_content=doc.page_content,
                metadata=new_metadata,
            )
        )

    docs.sort(key=lambda d: d.metadata["rerank_norm_score"], reverse=True)

    # Logging für Rerank-Ergebnisse

    print_result (" docs: rerank_retriever() ", docs)


    # 6. Top-K Abschneiden & finale Ausgabe
    #docs = docs[:TOP_K]
    docs= select_diverse_top_k(docs, max_k=TOP_K)

    print(f"search_phrase={search_phrase}")
    print(f"p_filters={p_filters}")
    print(f"search_kwargs={search_kwargs}")
    print(f"RETRIEVED DOCUMENTS (RETRIEVAL_K={TOP_K}):")

    print_result (" HYBRID RETRIEVAL + FLASHRANK RERANKING after select_diverse_top_k ", docs)

    return {"question": question, "docs": docs}


# ---------------------------------------------------------------------------
# 4. retrieve_documents_with_score
# ---------------------------------------------------------------------------
def retrieve_documents_with_score(
    input_data: dict,
    query_chain,
    vector_store,
    global_bm25: BM25Retriever
) -> dict:
    question = input_data["question"] if isinstance(input_data, dict) else input_data

    try:
        extracted = query_chain.invoke({"question": question})
        search_phrase = extracted.search_query if extracted.search_query else question
        p_filters = extracted.paragraph_filter if extracted.paragraph_filter else []
    except Exception as e:
        print(f"  ⚠️ Query Extraction fehlgeschlagen: {e}")
        search_phrase = question
        p_filters = []

    print(f"\n🔍 [Query Analyse] Suchphrase: '{search_phrase}' | Aktive Filter-§: {p_filters}")

    has_filter = False
    clean_filters = []
    RETRIEVAL_K = TOP_K * 4

    if p_filters:
        clean_filters = [str(p).replace("§", "").strip() for p in p_filters if p]
        if clean_filters:
            has_filter = True

    search_kwargs = {}
    if has_filter:
        anzahl_paragraphen = len(clean_filters)
        berechnetes_k = max(anzahl_paragraphen * 4, RETRIEVAL_K)
        search_kwargs["k"] = berechnetes_k
        if anzahl_paragraphen == 1:
            search_kwargs["filter"] = {"paragraph": clean_filters[0]}
        else:
            search_kwargs["filter"] = {"$or": [{"paragraph": p} for p in clean_filters]}
    else:
        search_kwargs["k"] = RETRIEVAL_K
        
    print(" >>> base_hybrid_retriever: 1. time")
    print(" >>> to check if docs_hybrid with p_filter is None\n")
    base_hybrid_retriever = _create_hybrid_retriever_with_score(
        vector_store=vector_store,
        search_kwargs=search_kwargs,
        global_bm25_retriever=global_bm25
    )

    docs_hybrid = base_hybrid_retriever.invoke(search_phrase)

    # Fallback ungefiltert
    if not docs_hybrid and has_filter:
        print(
            "  🔀 [Fallback] Keine Dokumente mit Metadaten-Filter gefunden. "
            "Starte ungefilterte Hybrid-Suche mit Score Blending..."
        )
        search_kwargs = {"k": RETRIEVAL_K}
        # create instance :  CustomHybridScoreRetriever()
        base_hybrid_retriever = _create_hybrid_retriever_with_score(
            vector_store=vector_store,
            search_kwargs=search_kwargs,
            global_bm25_retriever=global_bm25
        )
        docs_hybrid = base_hybrid_retriever.invoke(search_phrase)

    if not docs_hybrid:
        return {"question": question, "docs": []}

    # Logging der Ergebnisse der Basis-Suche
    # print_result ( "1. base_hybrid_retriever() ", docs_hybrid)

    # 5. Reranking / Blended Retrieval

    print(" >>> base_hybrid_retriever: 2. time")
    print(f">>> BlendedFlashRankRetriever= Hybrid + Reranker ")
    print(f">>> WEIGHT_HYBRID: {WEIGHT_HYBRID}, WEIGHT_RERANKER: {WEIGHT_RERANKER}\n")
    blended_retriever = BlendedFlashRankRetriever(
        base_retriever=base_hybrid_retriever,
        ranker=_NATIVE_RANKER,
        weight_hybrid= WEIGHT_HYBRID,
        weight_reranker= 1.0- WEIGHT_HYBRID,
        top_k=search_kwargs.get("k", TOP_K)
    )

    docs = blended_retriever.invoke(search_phrase)

    print_result ("2. BlendedFlashRankRetriever(): normalized, sorted after blended_score", docs)


    # 6. Top-K Abschneiden & finale Auswertung
    docs = select_diverse_top_k(docs, max_k=TOP_K)
    
    print(f"search_phrase={search_phrase}")
    print(f"p_filters={p_filters}")
    print(f"search_kwargs={search_kwargs}")
    print(f"\nRETRIEVED DOCUMENTS (TOP_K={TOP_K}):")
    
    print_result (" docs: select_diverse_top_k ", docs)

    return {
        "question": question,
        "docs": docs,
    }

# ---------------------------------------------------------------------------
# 5. Helper: Hybrid-Retriever mit Score
# ---------------------------------------------------------------------------
def _create_hybrid_retriever_with_score(
    vector_store,
    search_kwargs: dict,
    global_bm25_retriever: BM25Retriever,
):
    """
    Erstellt einen Hybrid-Retriever mit echten Scores für Dense + Sparse,
    dann gewichteter Fusion über Score-Blending.
    """
    k = search_kwargs.get("k", 8) # k= RETRIEVAL_K= 4*TOP_K=32
    db_filter = search_kwargs.get("filter")

    if db_filter:
        raw_db_data = vector_store.get(where=db_filter, include=["documents", "metadatas"])
        filtered_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(raw_db_data["documents"], raw_db_data["metadatas"])
        ]
        bm25_retriever = BM25Retriever.from_documents(filtered_docs)
        bm25_retriever.k = min(len(filtered_docs), 30)
    else:
        bm25_retriever = global_bm25_retriever
        bm25_retriever.k = k

    # Erbt von BaseRetriever für saubere LangChain-Schnittstelle
    class CustomHybridScoreRetriever(BaseRetriever):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        v_store: Any
        bm25: Any
        retrieval_k: int
        filter_kwargs: Optional[dict] = None
        # automatic call when .invoke due to pybantic: BaseRetriever-> BaseModel
        def _get_relevant_documents(
            self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
        ) -> List[Document]:
            
            # 1. Dense Search (Relevance Scores bevorzugen)
            if hasattr(self.v_store, "similarity_search_with_relevance_scores"):
                docs_scores = self.v_store.similarity_search_with_relevance_scores(
                    query, k=self.retrieval_k, filter=self.filter_kwargs
                )
            else:
                docs_scores = self.v_store.similarity_search_with_score(
                    query, k=self.retrieval_k, filter=self.filter_kwargs
                )
            # dense_pairs: already sorted
            dense_pairs = [(doc, float(score)) for doc, score in docs_scores]
            
            print("\n=========================================")

            print("\n---Dense : search_with_score, sorted, not normalized---")
            for i, (doc,score) in enumerate(dense_pairs,  start=1):
                print(f"Document {i} | Score: {score:.4f}  | "
                f"paragraph={doc.metadata.get('paragraph')} | "
                f"absatz={doc.metadata.get('absatz')} | nummer={doc.metadata.get('nummer')}"
                )
            print("----------------------\n")


       
            # 2. Sparse BM25 Search
            # scores: not sorted
            processed_query = self.bm25.preprocess_func(query)
            scores = self.bm25.vectorizer.get_scores(processed_query)
            docs = list(self.bm25.docs)

            sparse_pairs = [(doc, float(score)) for doc, score in zip(docs, scores)]
            sparse_pairs.sort(key=lambda x: x[1], reverse=True)
            sparse_pairs = sparse_pairs[: self.bm25.k]

            print("\n--- Sparse : _bm25_with_scores, sorted, not normalized  ---")
            for i, (doc, score) in enumerate(sparse_pairs,  start=1):
                print(f"Document {i} | Score: {score:.4f}  | "
                f"paragraph={doc.metadata.get('paragraph')} | "
                f"absatz={doc.metadata.get('absatz')} | nummer={doc.metadata.get('nummer')}"
                )
            print("----------------------\n")
        
            # 3. Fusion & Min-Max Normalisierung
            score_map = defaultdict(lambda: {"doc": None, "dense": None, "sparse": None})

            for doc, score in dense_pairs:
                key = doc.page_content
                score_map[key]["doc"] = doc
                score_map[key]["dense"] = score

            for doc, score in sparse_pairs:
                key = doc.page_content
                score_map[key]["doc"] = doc
                score_map[key]["sparse"] = score

            dense_scores = [v["dense"] for v in score_map.values() if v["dense"] is not None]
            sparse_scores = [v["sparse"] for v in score_map.values() if v["sparse"] is not None]

            dense_minmax = {}
            sparse_minmax = {}

            if dense_scores:
                d_arr = np.array(dense_scores, dtype=float)
                d_mn, d_mx = d_arr.min(), d_arr.max()
                for key, v in score_map.items():
                    if v["dense"] is not None:
                        dense_minmax[key] = 1.0 if d_mx == d_mn else (v["dense"] - d_mn) / (d_mx - d_mn)

            if sparse_scores:
                s_arr = np.array(sparse_scores, dtype=float)
                s_mn, s_mx = s_arr.min(), s_arr.max()
                for key, v in score_map.items():
                    if v["sparse"] is not None:
                        sparse_minmax[key] = 1.0 if s_mx == s_mn else (v["sparse"] - s_mn) / (s_mx - s_mn)

            blended = []
            for key, v in score_map.items():
                d = dense_minmax.get(key, 0.0)
                s = sparse_minmax.get(key, 0.0)
                final = 0.5 * d + 0.5 * s

                doc = v["doc"]
                meta = dict(doc.metadata)
                meta["dense_score"] = round(float(d), 4)
                meta["bm25_score"] = round(float(s), 4)
                meta["score"] = round(float(final), 4)

                blended.append(Document(page_content=doc.page_content, metadata=meta))

            blended.sort(key=lambda d: d.metadata["score"], reverse=True)
            print_hybrid_result("Hybrid Result (blended=dense*0.5 + bm25*0.5)-> sorted by blended score, normalized",  blended)
            
            return blended[:self.retrieval_k]

    return CustomHybridScoreRetriever(
        v_store=vector_store,
        bm25=bm25_retriever,
        retrieval_k=k,
        filter_kwargs=db_filter
    )


def print_hybrid_result(titel: str, docs):
    print()
    print(f"  >>>  {titel} <<< ")
    print("=" * 40 + f" docs: {titel}: score " + "=" * 40)

    for i, doc in enumerate(docs, start=1):
        md = doc.metadata
        print(
            f"Document {i} | "
            f"dense={md.get('dense_score', 0.0):.4f} | "
            f"bm25={md.get('bm25_score', 0.0):.4f} | "
            f"blended={md.get('score', 0.0):.4f} | "
            f"paragraph={md.get('paragraph', '')} | "
            f"absatz={md.get('absatz', '')} | "
            f"nummer={md.get('nummer', '')} | "
            f"thema={md.get('thema', '')}"
        )

    print("=" * 80 + "\n")
    





    
        