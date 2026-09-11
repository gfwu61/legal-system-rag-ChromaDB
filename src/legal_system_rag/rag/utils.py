from __future__ import annotations
import re  
from typing import Any, List
import numpy as np
from langchain_core.documents import Document


def create_structured_body(subsection_text: str | None, nummer_text: str | None) -> str:
    """Fügt Absatz- und Nummer-Texte zusammen, ignoriert Leere- und Platzhalterwerte."""
    body_lines = [p.strip() for p in [subsection_text, nummer_text] if p and p.strip() != "-"]
    return "\n".join(body_lines)

def extract_text_from_page_content(search_text: str, page_content: str) -> str:
    pattern = rf"{re.escape(search_text)}:\s*(.*?)(?=\n[A-Z_]+:|$)"
    match = re.search(pattern, page_content, re.DOTALL)
    return match.group(1).strip() if match else ""



def min_max_normalize(scores: List[float] | np.ndarray) -> np.ndarray:
    """Strictly normalizes a score array to the range [0.0, 1.0]."""
    # 1. Edge case: Handle empty array
    if len(scores) == 0:
        return np.array([])

    scores_arr = np.array(scores, dtype=float)
    min_val = float(np.min(scores_arr))
    max_val = float(np.max(scores_arr))

    # 2. Edge case: Handle identical values (min_val == max_val) to prevent division by zero
    if np.isclose(min_val, max_val):
        if np.isclose(min_val, 0.0):
            return np.zeros_like(scores_arr)  # If all values are 0 -> return zeros
        return np.ones_like(scores_arr)       # If all values are e.g. 5 -> return ones
        
    # 3. Standard case: Values are different -> perform min-max normalization
    diff = max_val - min_val
    return (scores_arr - min_val) / diff





def temperature_softmax_normalize(scores: List[float], temp: float = 0.2) -> np.ndarray:
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
    #return min_max_normalize(softmax_scores.tolist())
    return min_max_normalize(softmax_scores )



def build_page_content_for_rerank_from_page_content(page_content: str) -> str:
    paragraph = extract_text_from_page_content("PARAGRAPH", page_content)
    paragraph_title = extract_text_from_page_content("PARAGRAPH_TITEL", page_content)
    original_text = extract_text_from_page_content("ORIGINALTEXT", page_content)

    # Baut eine natürliche Überschrift: "§ 573 Ordentliche Kündigung des Vermieters"
    header = f"{paragraph} {paragraph_title}".strip()
    if not header.startswith("§"):
        header = f"§ {header}"

    return f"{header}\n{original_text}".strip()


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

    

def print_result(title: str, docs: list[Document]) -> None:
    print()
    print(f"  >>>  {title} <<< ")
    print("=" * 40 + f" docs: {title} " + "=" * 40)

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


    


def print_hybrid_result(title: str, docs):
    print()
    print(f"  >>>  {title} <<< ")
    print("=" * 40 + f" docs: {title}: score " + "=" * 40)

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
    






    