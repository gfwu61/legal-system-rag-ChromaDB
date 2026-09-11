# pipelines/agentic_rag.py

# pipelines/agentic_rag.py
# Die Pipeline nimmt die Anfrage entgegen, ruft den Agent-Loop auf und nutzt llm_answer für das finale Ergebnis.
# Ist die End-to-End Pipeline, die von der UI (app.py) aufgerufen wird. 
# Sie startet den Agenten aus agent.py und erzeugt am Ende mit llm_answer die finale Antwort.


from typing import Any, List
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from legal_system_rag.rag.agent import run_agentic_retrieval_loop
from .utils import print_result



def format_context_documents(docs: List[Document]) -> str:
    """
    Formats retrieved documents into a clean context string for the prompt.
    """
    formatted_chunks = []
    for idx, doc in enumerate(docs, start=1):
        paragraph = doc.metadata.get("paragraph", "N/A")
        content = doc.page_content.strip()
        formatted_chunks.append(
            f"--- CHUNK {idx} (Norm: § {paragraph}) ---\n{content}"
        )
    return "\n\n".join(formatted_chunks)


def generate_final_answer(
    question: str, 
    docs: List[Document], 
    llm: BaseChatModel
) -> str:
    """
    Synthesizes a precise legal response in German based on retrieved document chunks.
    """
    system_prompt = (
        "Du bist ein präziser KI-Assistent für deutsches Mietrecht.\n\n"
        "Beantworte die Nutzerfrage primär auf Basis des bereitgestellten Kontexts.\n\n"
        "LEITLINIEN:\n"
        "1. **Originaltext als Basis:** Der ORIGINALTEXT ist deine primäre Quelle. "
        "Nutze ZUSATZINFORMATIONEN (z. B. THEMA, KLARTEXT) als Verständnis- und Erklärungshilfe.\n"
        "2. **Praktische Einordnung & Logik:** Wende allgemeine Gesetzesbestimmungen logisch "
        "auf die konkrete Nutzerfrage an. Wandle juristische Formulierungen (z. B. Zeitspannen, "
        "Berechnungsformeln) in die praktische Bedeutung um und rechne Fristen oder Beträge bei Bedarf aus.\n"
        "3. **Kontextuelle Zuordnung:** Allgemeine Regeln ohne spezifische Parteinennung gelten "
        "grundsätzlich für das Mietverhältnis. Parteispezifische Sonderregeln dürfen nicht "
        "fälschlicherweise auf die andere Vertragspartei übertragen werden.\n"
        "4. **Zitation & Grenzen:** Gib Gesetzesquellen (Paragraph, Absatz, Satz) präzise an, "
        "sobald sie im Kontext stehen. Fehlen benötigte Informationen im Kontext vollständig, weise kurz darauf hin.\n"
        "5. **Form:** Antworte direkt, laienverständlich und ohne anschließende Folgefragen.\n\n"
        "KONTEXT:\n{context}"
    )

    formatted_context = format_context_documents(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "context": formatted_context,
        "question": question
    })
    
    return response.content.strip()


def run_agentic_rag_pipeline(
    question: str, 
    llm_query: BaseChatModel,       
    llm_answer: BaseChatModel,      
    vector_store: Any,    
    global_bm25: Any
) -> dict:
    """
    Executes the complete end-to-end Agentic RAG pipeline.
    
    1. Runs agentic retrieval loop (routing, candidate retrieval, reranking, evaluation & query rewriting).
    2. Synthesizes final response using the answer LLM on relevant chunks.
    """
    # 1. Execute retrieval via the Agentic Evaluation Loop
    retrieval_result = run_agentic_retrieval_loop(
        question=question,
        llm_query=llm_query,
        vector_store=vector_store,
        global_bm25=global_bm25,
        max_retries=2
    )
    
    docs = retrieval_result.get("docs", [])
    print_result("docs: Self-Corrective RAG: select_diverse_top_k+sorted", docs)   
    
    # 2. Synthesize final answer using the answer LLM
    if docs:
        answer = generate_final_answer(
            question=question, 
            docs=docs, 
            llm=llm_answer
        )
    else:
        answer = "No relevant information could be found in the provided legal documents."

    return {
        "answer": answer,
        "source_documents": docs,
        "attempts": retrieval_result.get("attempts"),
        "route_used": retrieval_result.get("route_used")
    }




    