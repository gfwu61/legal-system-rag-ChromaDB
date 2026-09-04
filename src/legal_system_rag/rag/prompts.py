from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


from pydantic import BaseModel, Field
from typing import List

class EnrichmentOutput(BaseModel):
    # Nutze 'thema' direkt oder setze ein Alias, damit Pydantic und Metadaten übereinstimmen
    topic: str = Field(
        description=(
            "Kompakter Begriff für das juristische Hauptthema (max. 2-4 Wörter). "
            "Beispiele: 'Kündigungsfristen bei Wohnraum', 'Sonderkündigungsrecht', 'Betriebskostenabrechnung'."
        )
    )
    plain_language_summary: str = Field(
        description="Einfache Erklärung des Paragraphen in Laiensprache."
    )
    user_questions: List[str] = Field(
        description="3 bis 4 typische Nutzerfragen aus Mieter- und Vermietersicht."
    )
    keywords: List[str] = Field(
        description="3 bis 6 relevante Keywords."
    )

    

class QueryExtraction(BaseModel):
    search_query: str = Field(
        description="Optimierte, suchbare juristische Kernphrase."
    )
    paragraph_filter: List[str] = Field(
        default_factory=list, 
        description="Liste reiner Paragraphen-Nummern als Strings OHNE Paragraphen-Zeichen '§' oder Wörtern (z. B. ['12', '13a', '5']). Beispiel: Bei '§ 12 Abs. 1' nur '12' eintragen."
    )

    
def build_answer_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """Du bist ein präziser KI-Assistent für deutsches Mietrecht.

Beantworte die Nutzerfrage primär auf Basis des bereitgestellten Kontexts.

LEITLINIEN:
1. **Originaltext als Basis:** Der ORIGINALTEXT ist deine primäre Quelle. Nutze THEMA, KLARTEXT und NUTZERFRAGEN als Verständnis- und Erklärungshilfe.
2. **Praktische Einordnung & Logik:** Wende allgemeine Gesetzesbestimmungen logisch auf die konkrete Nutzerfrage an. Wandle juristische Formulierungen (z. B. Zeitspannen, Berechnungsformeln) in die praktische Bedeutung um und rechne Fristen oder Beträge bei Bedarf aus.
3. **Kontextuelle Zuordnung:** Allgemeine Regeln ohne spezifische Parteinennung gelten grundsätzlich für das Mietverhältnis. Parteispezifische Sonderregeln dürfen nicht fälschlicherweise auf die andere Vertragspartei übertragen werden.
4. **Zitation & Grenzen:** Gib Gesetzesquellen (Paragraph, Absatz, Satz) präzise an, sobald sie im Kontext stehen. Fehlen benötigte Informationen im Kontext vollständig, weise kurz darauf hin.
5. **Form:** Antworte direkt, laienverständlich und ohne anschließende Folgefragen.

KONTEXT:
{context}"""
        ),
        ("human", "{question}")
    ])
    
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

def build_query_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "Du bist ein juristischer Such-Assistent für deutsches Mietrecht.\n"
            "Extrahierte Fachbegriffe (search_query) und Paragraphen (paragraph_filter) aus der Nutzerfrage.\n\n"

            "REGELN:\n"
            "1. search_query: 2-4 präzise juristische Schlagwörter (keine Füllwörter).\n"
            "2. Bei Zeit/Frist-Fragen ('wie lange', 'wann'): 'Kündigungsfrist' oder 'Frist' MUSS an 1. Stelle stehen.\n"
            "3. Vermeide vage Oberbegriffe (z.B. 'Kündigungsschutz', 'Wohnraummiete').\n"
            "4. paragraph_filter: Nur explizit genannte Paragraphen als Ziffer (z.B. '573c'). Sonst [].\n\n"

            "BEISPIELE:\n"
            "Frage: 'wie lang ist die Kündigungsfrist für den Vermieter?'\n"
            "search_query: 'Kündigungsfrist ordentliche Kündigung Vermieter'\n"
            "paragraph_filter: []\n\n"

            "Frage: 'Darf der Vermieter einfach die Miete erhöhen?'\n"
            "search_query: 'Mieterhöhung Zustimmung Kappungsgrenze'\n"
            "paragraph_filter: []\n\n"

            "Frage: 'Wann kriege ich meine Kaution laut § 551 zurück?'\n"
            "search_query: 'Rückzahlung Mietkaution Fälligkeit'\n"
            "paragraph_filter: ['551']"
        ),
        ("human", "{question}")
    ])


    