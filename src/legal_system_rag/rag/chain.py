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

@retry(
    retry=retry_if_exception_type(
        (TimeoutError, ConnectionError)
    ),
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(
        multiplier=1, min=1,
        max=8,
    ),
    reraise=True,
)

def _invoke_enrichment_with_retry(
    prompt: str,
    structured_llm,
) -> EnrichmentOutput:
    return structured_llm.invoke(prompt)




def enrich_chunk(text: str, structured_llm) -> EnrichmentOutput:
    prompt = f"""Du bist ein juristischer KI-Assistent.
Analysiere den folgenden Gesetzestext.
Erzeuge ein strukturiertes JSON mit den geforderten Feldern.

Regeln für die Generierung:
- Generiere MINDESTENS 3 und MAXIMAL 5 typische Nutzerfragen.
- Die Fragen sollen abdecken, was Bürger in der Praxis wissen wollen.

Gesetzestext:
{text}"""

    try:
        # Führt den Aufruf mit automatischer Retry-Logik aus
        return _invoke_enrichment_with_retry(prompt, structured_llm)
    except Exception as e:
        print(f"  ⚠️ LLM-Enrichment endgültig fehlgeschlagen nach Retries: {e}")

    return EnrichmentOutput(
        topic="Unbekannt",
        plain_language_summary=text,
        user_questions=[
            "Welche Regelung gilt hier?",
            "Was steht in diesem Paragraphen?",
            "Welche Rechte ergeben sich daraus?",
        ],
        keywords=[],
    )


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
THEMA: {enrichment.topic}
KEYWORDS: {keywords}
TYPISCHE NUTZERFRAGEN:
{user_questions}
KLARTEXT: {enrichment.plain_language_summary}
ORIGINALTEXT: {original_text}"""


def retrieve_documents(input_data: dict, query_chain, vector_store):
    question = (
        input_data["question"] if isinstance(input_data, dict) else input_data
    )
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
        f"\n🔍 [Query Analyse] Suchphrase: '{search_phrase}' | Aktive Filter-§: {p_filters}"
    )
    search_kwargs = {"k": 5}
    has_filter = False

    if p_filters:
        clean_filters = [
            str(p).replace("§", "").strip() for p in p_filters if p
        ]
        if clean_filters:
            has_filter = True
            if len(clean_filters) == 1:
                search_kwargs["filter"] = {"paragraph": str(clean_filters[0])}
            else:
                search_kwargs["filter"] = {
                    "$or": [{"paragraph": str(p)} for p in clean_filters]
                }

    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs=search_kwargs
    )
    docs = retriever.invoke(search_phrase)

    if not docs and has_filter:
        print(
            "  🔀 [Fallback] Keine Dokumente mit Metadaten-Filter gefunden. Starte ungefilterte Vektorsuche..."
        )
        fallback_kwargs = {"k": 3}
        fallback_retriever = vector_store.as_retriever(
            search_type="mmr", search_kwargs=fallback_kwargs
        )
        docs = fallback_retriever.invoke(search_phrase)

    return {"question": question, "docs": docs}


def generate_answer(input_data: dict, llm_answer, answer_prompt):
    if not isinstance(input_data, dict) or "docs" not in input_data:
        return {"answer": "Fehler in der Verarbeitungskette.", "docs": []}

    question = input_data["question"]
    docs = input_data["docs"]

    if not docs:
        return {
            "answer": "Ich konnte keine passenden Dokumente finden.",
            "docs": [],
        }

    context_str = "\n\n".join([doc.page_content for doc in docs])

    prompt_value = answer_prompt.invoke(
        {"context": context_str, "question": question}
    )
    response = llm_answer.invoke(prompt_value)

    return {"answer": response.content, "docs": docs}


def build_rag_chain(llm_query, llm_answer, vector_store):
    query_prompt = build_query_prompt()
    query_chain = query_prompt | llm_query.with_structured_output(
        QueryExtraction
    )
    answer_prompt = build_answer_prompt()

    return (
        {"question": RunnablePassthrough()}
        | RunnableLambda(
            lambda x: retrieve_documents(x, query_chain, vector_store)
        )
        | RunnableLambda(
            lambda x: generate_answer(x, llm_answer, answer_prompt)
        )
    )