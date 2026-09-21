import pytest
import re
from langchain_openai import ChatOpenAI
# Ersetzen Sie 'your_module' mit dem Namen Ihrer Python-Datei, in der die Funktion liegt
from legal_system_rag.rag.router import build_llm_router, RouterDecision
from legal_system_rag.config.settings import LLM_QUERY_MODEL

from legal_system_rag.network.client_factory import (
    create_http_client,
    get_proxy_url,
)


# pytest tests/unit/test_router.py -v

@pytest.fixture(scope="module")
def router_chain():
    """Initialisiert die Router-Chain einmal für alle Tests."""
    # Verwenden Sie ein schnelles, günstiges Modell für das Routing (z.B. gpt-4o-mini)
    proxy_url = get_proxy_url()
    _sync_client= create_http_client(
        proxy_url=proxy_url,
        ignore_ssl=True,
    )

    llm_router = ChatOpenAI(
        model=LLM_QUERY_MODEL,
        temperature=0.1,
        http_client=_sync_client,
    )
    return build_llm_router(llm_router)

# Definition der Testfälle: (Nutzerfrage, Erwartete Route, Erwartete Paragraphen oder None)
TEST_CASES = [
    # Fälle für specific_norm
    ("Was steht in § 535 BGB?", "specific_norm", ["535"]),
    ("Zeige mir den Wortlaut von Paragraph 573 BGB.", "specific_norm", ["573"]),
    ("Ich möchte § 573c und § 556d sehen", "specific_norm", ["573c", "556d"]),
    
    # Fälle für concept_search (selbst wenn Paragraphen genannt werden)
    ("Wie läuft eine Eigenbedarfskündigung ab?", "concept_search", None),
    ("Darf der Vermieter die Miete um 20% erhöhen laut § 558?", "concept_search", None),
    ("Schimmel in der Küche, Mietminderung möglich?", "concept_search", None),
    ("Was muss ich bei einer Eigenbedarfskündigung nach § 573 beachten?", "concept_search", None),
]



# (Die Fixture 'router_chain' und 'TEST_CASES' bleiben hier exakt gleich)

@pytest.mark.parametrize("question, expected_route, expected_paragraphs", TEST_CASES)
def test_router_decision(router_chain, question, expected_route, expected_paragraphs):
    """Testet, ob das LLM für verschiedene Fragen die richtige Route, Filter und Suchphrasen wählt."""
    
    # Ausführen der Chain
    result: RouterDecision = router_chain.invoke({"question": question})
    
    # 1. Prüfen, ob die Route stimmt
    assert result.route == expected_route, f"Falsche Route für Frage: '{question}'. Erwartet: {expected_route}, Erhalten: {result.route}"
    
    # 2. Prüfen der Paragraphen-Filter
    if expected_paragraphs is not None:
        # Sortieren für den Vergleich, falls die Reihenfolge abweicht
        assert result.paragraph_filters is not None, "Paragraph_filters sollte nicht None sein."
        assert sorted(result.paragraph_filters) == sorted(expected_paragraphs), \
            f"Falsche Paragraphen für: '{question}'. Erwartet: {expected_paragraphs}, Erhalten: {result.paragraph_filters}"
    else:
        # Bei concept_search sollte die Liste leer oder None sein
        assert result.paragraph_filters is None or len(result.paragraph_filters) == 0, \
            f"Es wurden fälschlicherweise Paragraphen extrahiert für: '{question}'"
            
    # 3. Prüfen, ob eine Suchanfrage generiert wurde
    assert len(result.search_query.strip()) > 0, "Die search_query darf nicht leer sein."

    # 4. NEU: Prüfen, ob relevante Paragraphenzahlen in der search_query bei concept_search erhalten bleiben
    if result.route == "concept_search":
        # Findet alle Zahlen (optional mit Suffix wie 556g) nach einem §-Zeichen oder dem Wort 'Paragraph'
        found_numbers = re.findall(r'(?:§|Paragraph)\s*(\d+[a-z]*)', question, re.IGNORECASE)
        
        for num in found_numbers:
            assert num in result.search_query, \
                f"Die search_query sollte die Paragraphennummer '{num}' enthalten! " \
                f"Frage: '{question}' -> Generierte search_query: '{result.search_query}'"

    
    
