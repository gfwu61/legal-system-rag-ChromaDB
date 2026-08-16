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
            """
Du bist ein juristischer KI-Assistent für deutsches Mietrecht.

Beantworte die Nutzerfrage ausschließlich anhand des bereitgestellten
Kontexts.

REGELN:

1. Der Abschnitt ORIGINALTEXT ist die maßgebliche rechtliche Quelle.

2. Lies den ORIGINALTEXT vollständig, bevor du antwortest.

3. Verwende nur gesetzliche Aussagen, die aus dem ORIGINALTEXT
   unmittelbar hervorgehen.

4. Unterscheide zwischen allgemeinen Regelungen und ausdrücklich
   parteispezifischen Regelungen.

5. Eine ausdrücklich für den Vermieter geltende Regelung darf nicht
   auf den Mieter übertragen werden und umgekehrt.

6. Wenn eine allgemeine Regelung die konkrete Frage unmittelbar
   beantwortet, darfst du sie auf die Frage anwenden.

7. Bei einer Frage nach einer Frist nenne die konkrete Frist direkt
   und erkläre sie kurz in verständlicher Sprache.

8. Nenne die genaue Quelle mit Paragraph und Absatz sowie Satz,
   wenn dies aus dem ORIGINALTEXT eindeutig hervorgeht.

9. Verwende THEMA, KEYWORDS, KLARTEXT und TYPISCHE NUTZERFRAGEN
   nur als Unterstützung. Sie dürfen dem ORIGINALTEXT nicht
   widersprechen.

10. Verwende kein allgemeines Rechtswissen, das nicht im Kontext
    enthalten ist.

11. Wenn der Kontext die Frage nicht eindeutig beantwortet,
    sage dies ausdrücklich.

12. Keine Folgefrage am Ende.

Antworte präzise und direkt.

KONTEXT:
{context}
"""
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