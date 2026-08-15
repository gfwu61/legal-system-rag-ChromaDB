from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

class EnrichmentOutput(BaseModel):
    topic: str = Field(description="Juristisches Hauptthema")
    plain_language_summary: str = Field(description="Einfache Erklärung")
    user_questions: List[str] = Field(
        description="Generiere mindestens 3 und maximal 5 typische, unterschiedliche Nutzerfragen, die dieser Text beantwortet."
    )
    keywords: List[str] = Field(
        description="3 bis 6 relevante juristische Keywords"
    )



class QueryExtraction(BaseModel):
    search_query: str = Field(description="Optimierte, suchbare juristische Kernphrase.")
    paragraph_filter: List[str] = Field(default_factory=list, description="Liste reiner Paragraphen-Nummern.")

def build_answer_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "Beantworte die juristische Frage des Nutzers AUSSCHLIESSLICH basierend auf dem bereitgestellten Kontext. "
            "Wenn der Kontext die Antwort nicht hergibt, sage das sachlich. "
            "Gehe strukturiert vor, nenne immer die exakte Quelle (§, Absatz, Nummer) und bleibe rechtssicher."
            "keine weitere Folgefrage soll am Ende der Antwort zugefügt werden.\n\n"
            "Kontext:\n{context}"
        ),
        ("human", "{question}")
    ])

def build_query_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "Du bist ein präziser juristischer Such-Assistent. "
            "Extrahiere die suchbare Kernphrase.\n"
            "Erkennst du konkrete Paragraphen-Nennungen (z.B. '§ 556d' oder 'Paragraph 573'), "
            "extrahiere NUR die reine Nummer/Ziffer (z.B. '556d', '573') in die Liste paragraph_filter."
        ),
        ("human", "{question}")
    ])