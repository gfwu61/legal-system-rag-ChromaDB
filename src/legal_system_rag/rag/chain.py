from typing import List, Optional

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from legal_system_rag.rag.prompts import (
    EnrichmentOutput,
    QueryExtraction,
    build_answer_prompt,
    build_query_prompt,
)


# ============================================================
# LLM ENRICHMENT RETRY
# ============================================================

@retry(
    retry=retry_if_exception_type(
        (TimeoutError, ConnectionError)
    ),
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

    The original legal text is the sole legal source.
    Generated fields must not introduce legal information that is not
    contained in the original text.
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
        # Run the LLM call with automatic retry logic.
        return _invoke_enrichment_with_retry(prompt, structured_llm)

    except Exception as e:
        print(
            f"  ⚠️ LLM-Enrichment endgültig fehlgeschlagen nach Retries: {e}"
        )

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
# BUILD PAGE CONTENT
# ============================================================

def build_page_content(
    gesetz: str,
    paragraph: str,
    paragraph_title: str,
    absatz: str,
    nummer: Optional[str],
    references: List[str],
    enrichment: EnrichmentOutput,
    original_text: str,
) -> str:
    """
    Build the final text representation stored in the vector database.

    The original legal text is intentionally included prominently
    because it is the authoritative source for answer generation.
    """

    nummer_text = nummer if nummer else "-"
    references_text = ", ".join(references) if references else "-"
    user_questions = "\n".join([f"- {q}" for q in enrichment.user_questions])
    keywords = ", ".join(enrichment.keywords)

    return f"""GESETZ: {gesetz}
PARAGRAPH: §{paragraph}
PARAGRAPH_TITEL: {paragraph_title}
ABSATZ: {absatz}
NUMMER: {nummer_text}
REFERENZEN: {references_text}

ORIGINALTEXT:
{original_text}

THEMA: {enrichment.topic}
KEYWORDS: {keywords}
TYPISCHE NUTZERFRAGEN:
{user_questions}
KLARTEXT: {enrichment.plain_language_summary}"""


# ============================================================
# DOCUMENT RETRIEVAL
# ============================================================
def retrieve_documents(input_data: dict, query_chain, vector_store):
    question = (
        input_data["question"] if isinstance(input_data, dict) else input_data
    )

    # ============================================================
    # QUERY EXTRACTION
    # ============================================================

    try:
        extracted = query_chain.invoke({"question": question})

        search_phrase = (
            extracted.search_query
            if extracted.search_query
            else question
        )

        p_filters = (
            extracted.paragraph_filter
            if extracted.paragraph_filter
            else []
        )

    except Exception as e:
        print(f"  ⚠️ Query Extraction fehlgeschlagen: {e}")

        search_phrase = question
        p_filters = []

    print(
        f"\n🔍 [Query Analyse] "
        f"Suchphrase: '{search_phrase}' | "
        f"Aktive Filter-§: {p_filters}"
    )

    # ============================================================
    # SEARCH CONFIGURATION
    # ============================================================

    search_kwargs = {"k": 3}
    has_filter = False

    if p_filters:
        clean_filters = [
            str(p).replace("§", "").strip()
            for p in p_filters
            if p
        ]

        if clean_filters:
            has_filter = True

            if len(clean_filters) == 1:
                search_kwargs["filter"] = {
                    "paragraph": clean_filters[0]
                }

            else:
                search_kwargs["filter"] = {
                    "$or": [
                        {"paragraph": p}
                        for p in clean_filters
                    ]
                }

    # ============================================================
    # RETRIEVAL
    # ============================================================

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    docs = retriever.invoke(search_phrase)

    # ============================================================
    # RETRIEVAL LOG
    # ============================================================

    print("\n" + "=" * 80)
    print("RETRIEVAL SEARCH")
    print("=" * 80)

    print(f"search_phrase={search_phrase}")
    print(f"p_filters={p_filters}")
    print(f"search_kwargs={search_kwargs}")

    print("\nRETRIEVED DOCUMENTS:")

    for i, doc in enumerate(docs, start=1):
        print(
            f"Document {i} | "
            f"paragraph={doc.metadata.get('paragraph')} | "
            f"absatz={doc.metadata.get('absatz')} | "
            f"nummer={doc.metadata.get('nummer')} | "
            f"thema={doc.metadata.get('thema')}"
        )

    print("=" * 80 + "\n")

    # ============================================================
    # FALLBACK
    # ============================================================

    if not docs and has_filter:
        print(
            "  🔀 [Fallback] Keine Dokumente mit "
            "Metadaten-Filter gefunden. "
            "Starte ungefilterte Vektorsuche..."
        )

        fallback_kwargs = {"k": 3}

        fallback_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=fallback_kwargs,
        )

        docs = fallback_retriever.invoke(search_phrase)

    return {
        "question": question,
        "docs": docs,
    }

    
def retrieve_documents1(
    input_data: dict,
    query_chain,
    vector_store,
):
    """
    Extract the search query, optionally apply paragraph filters,
    and retrieve relevant documents from the vector store.
    """

    question = (
        input_data["question"] if isinstance(input_data, dict) else input_data
    )

    # --------------------------------------------------------
    # Query extraction
    # --------------------------------------------------------

    try:
        extracted = query_chain.invoke({"question": question})
        search_phrase = (
            extracted.search_query if extracted.search_query else question
        )

        p_filters = (
            extracted.paragraph_filter if extracted.paragraph_filter else []
        )

    except Exception as e:
        print(f"  ⚠️ Query Extraction fehlgeschlagen: {e}")
        search_phrase = question
        p_filters = []

    print(
        f"\n🔍 [Query Analyse] "
        f"Suchphrase: '{search_phrase}' "
        f"| Aktive Filter-§: {p_filters}"
    )

    # --------------------------------------------------------
    # Search configuration
    # --------------------------------------------------------

    search_kwargs = {"k": 3} #5
    has_filter = False

    if p_filters:

        clean_filters = [
            str(paragraph)
            .replace("§", "")
            .strip()
            for paragraph in p_filters
            if paragraph
        ]

        if clean_filters:

            has_filter = True

            if len(clean_filters) == 1:

                search_kwargs["filter"] = {
                    "paragraph": str(clean_filters[0])
                }

            else:

                search_kwargs["filter"] = {
                    "$or": [
                        {
                            "paragraph": str(paragraph)
                        }
                        for paragraph in clean_filters
                    ]
                }

    # --------------------------------------------------------
    # MMR retrieval
    # --------------------------------------------------------

    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs=search_kwargs
    )

    docs = retriever.invoke(search_phrase)

    # --------------------------------------------------------
    # Retrieval logging
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("RETRIEVAL SEARCH")
    print("=" * 80)
    print(f"search_phrase={search_phrase}")
    print(f"p_filters={p_filters}")
    print(f"search_kwargs={search_kwargs}")

    print("\nRETRIEVED DOCUMENTS:")
    for i, doc in enumerate(docs, start=1):
        print(
            f"Document {i} | "
            f"paragraph={doc.metadata.get('paragraph')} | "
            f"absatz={doc.metadata.get('absatz')} | "
            f"nummer={doc.metadata.get('nummer')} | "
            f"thema={doc.metadata.get('thema')}"
        )

    print("=" * 80 + "\n")

    # --------------------------------------------------------
    # Fallback retrieval
    # --------------------------------------------------------

    if not docs and has_filter:

        print(
            "  🔀 [Fallback] Keine Dokumente mit Metadaten-Filter gefunden. Starte ungefilterte Vektorsuche..."
        )
        fallback_kwargs = {"k": 3}
        fallback_retriever = vector_store.as_retriever(
            #search_type="mmr", search_kwargs=fallback_kwargs
            search_type="similarity", search_kwargs=fallback_kwargs
        )
        docs = fallback_retriever.invoke(search_phrase)

    return {"question": question, "docs": docs}


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(
    input_data: dict,
    llm_answer,
    answer_prompt,
):
    """
    Generate the final answer exclusively from retrieved documents.
    """

    if (
        not isinstance(input_data, dict)
        or "docs" not in input_data
    ):
        return {
            "answer": "Fehler in der Verarbeitungskette.",
            "docs": [],
        }

    question = input_data["question"]
    docs = input_data["docs"]

    # --------------------------------------------------------
    # No documents
    # --------------------------------------------------------

    if not docs:

        return {
            "answer": (
                "Ich konnte keine passenden Dokumente finden."
            ),
            "docs": [],
        }

    context_str = "\n\n".join([doc.page_content for doc in docs])

    # --------------------------------------------------------
    # Context logging
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("ANSWER CONTEXT")
    print("=" * 80)

    print(context_str)

    print("=" * 80 + "\n")

    # --------------------------------------------------------
    # Answer generation
    # --------------------------------------------------------

    prompt_value = answer_prompt.invoke(
        {
            "context": context_str,
            "question": question,
        }
    )

    response = llm_answer.invoke(
        prompt_value
    )

    return {
        "answer": response.content,
        "docs": docs,
    }


# ============================================================
# BUILD RAG CHAIN
# ============================================================

def build_rag_chain(
    llm_query,
    llm_answer,
    vector_store,
):
    """
    Build the complete RAG pipeline:

        User question
            ↓
        Query extraction
            ↓
        Paragraph filter
            ↓
        MMR retrieval
            ↓
        Retrieved documents
            ↓
        Answer generation
    """

    # --------------------------------------------------------
    # Query chain
    # --------------------------------------------------------

    query_prompt = build_query_prompt()

    query_chain = (
        query_prompt
        | llm_query.with_structured_output(
            QueryExtraction
        )
    )

    # --------------------------------------------------------
    # Answer prompt
    # --------------------------------------------------------

    answer_prompt = build_answer_prompt()

    # --------------------------------------------------------
    # RAG pipeline
    # --------------------------------------------------------

    return (
        {"question": RunnablePassthrough()}

        | RunnableLambda(
            lambda x: retrieve_documents(
                x,
                query_chain,
                vector_store,
            )
        )

        | RunnableLambda(
            lambda x: generate_answer(
                x,
                llm_answer,
                answer_prompt,
            )
        )
    )