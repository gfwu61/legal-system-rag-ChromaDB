from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel


class RelevanceGrade(BaseModel):
    binary_score: str = Field(
        description="Relevance of the documents to the question: 'yes' or 'no'"
    )


def evaluate_document_relevance(
    question: str, 
    docs: list, 
    llm: BaseChatModel
) -> str:
    """
    Evaluates whether the retrieved document chunks contain sufficient information
    to adequately answer the user's question.
    """
    if not docs:
        return "no"
        
    evaluator = llm.with_structured_output(RelevanceGrade)

    system_prompt = (
        "You are a legal auditor. Assess whether the provided document chunks contain "
        "the required information to adequately answer the question.\n"
        "Respond with 'yes' if at least one chunk is helpful, otherwise respond with 'no'."
    )
    
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", f"Question: {question}\n\nContext Chunks:\n{context}")
    ])
    
    chain = prompt | evaluator
    res = chain.invoke({})
    return res.binary_score

def rewrite_search_query(
    question: str, 
    current_query: str, 
    llm: BaseChatModel
) -> str:
    """
    Führt ein Fallback-Query-Rewriting inklusive Query Expansion durch,
    falls der erste Retrieval-Durchlauf nicht genügend relevante Chunks geliefert hat.
    """
    system_prompt = (
        "Du bist ein Experte für Informationsbeschaffung im deutschen Mietrecht.\n"
        "Die bisherige Suchanfrage hat keine ausreichenden Ergebnisse in der Datenbank geliefert.\n\n"
        "DEINE AUFGABE:\n"
        "Erstelle eine erweiterte, optimierte Suchanfrage (Query Expansion) für eine Hybridsuche (BM25 + Vektorsuche).\n\n"
        "REGELN FÜR DIE QUERY EXPANSION:\n"
        "1. **Fachbegriffe & Synonyme:** Ersetze Alltagssprache durch präzise juristische Terminologie (z. B. 'rauskommen' -> 'Kündigung', 'Aufhebung', 'Frist').\n"
        "2. **Anker-Erweiterung:** Ergänze verwandte mietrechtliche Kernbegriffe und Gesetzesanker (z. B. BGB-Mietrecht, Sonderkündigung, Aufhebungsvertrag).\n"
        "3. **Fokus:** Halte die Anfrage auf maximal 5-7 relevante Fachbegriffe beschränkt (keine Sätze, keine Füllwörter).\n"
        "4. **Ausgabe:** Gib AUSSCHLIESSLICH die neue Suchphrase zurück."
    )

    human_prompt = (
        f"Originalfrage des Nutzers: '{question}'\n"
        f"Bisherige (gescheiterte) Suchanfrage: '{current_query}'\n\n"
        f"Optimierte Suchphrase für 2. Versuch:"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])

    chain = prompt | llm
    res = chain.invoke({})
    
    # Sicherstellen, dass Anführungszeichen oder Zeilenumbrüche bereinigt werden
    clean_query = res.content.strip().replace('"', '').replace("'", "")
    return clean_query





    