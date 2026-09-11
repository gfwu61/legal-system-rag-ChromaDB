# rag/reranker.py (oder in rag/retrieval.py)

"""
example:
    rerank_results: sorted by score, but the id shows the identity of docs.
    [
    {
        "id": 2,
        "text": "Dies ist der Text der am besten zur Suchanfrage passt...",
        "meta": {"source": "BGB.txt", ...},
        "score": 0.9823
    },
    {
        "id": 0,
        "text": "Dies ist der Text der am zweitbesten passt...",
        "meta": {"source": "BGB.txt", ...},
        "score": 0.7412
    }, 
    {
        "id": 1,
        "text": "Dieser Text hat kaum noch Relevanz für die Suchanfrage...",
        "meta": {"source": "BGB.txt",...},
        "score": 0.1256
    },...
    ]
"""


from __future__ import annotations

import os
import time
import httpx
from typing import List
import numpy as np
from langchain_core.documents import Document
from huggingface_hub import InferenceClient

from legal_system_rag.config import TOP_K, HF_TOKEN, IGNORE_SSL
from .utils import (
    build_page_content_for_rerank_from_page_content,
    print_result,
    select_diverse_top_k,
    min_max_normalize,
    temperature_softmax_normalize,
)
from legal_system_rag.network.client_factory import (
    create_http_client,
    get_proxy_url,
)


# ---------------------------------------------------------------------------
# 1. HF InferenceClient Initialisierung
# ---------------------------------------------------------------------------
proxy_url = get_proxy_url()

_HF_CLIENT = InferenceClient(
    model="BAAI/bge-reranker-v2-m3",
    token=HF_TOKEN,
    proxies=proxy_url,  # Pass string or dict: {"http": proxy_url, "https": proxy_url}
)


# ---------------------------------------------------------------------------
# 2. Unified Scoring Function über Hugging Face API
# ---------------------------------------------------------------------------

def compute_rerank_scores_api(query: str, docs: list, max_length: int = 1000) -> list[float]:
    """
    Berechnet Reranking-Scores über die Hugging Face Serverless Router API.
    Gibt garantiert eine Liste der Länge len(docs) zurück.
    """
    HF_TOKEN = os.getenv("HF_TOKEN")
    num_docs = len(docs) if docs else 0

    if not HF_TOKEN or num_docs == 0:
        return [0.0] * num_docs

    # Map von doc-Index im übergebenen Array zu gereinigtem Payload
    # Wenn ein Chunk ungültig ist, bleibt sein Platzhalter im Payload ein leerer String oder gekürzter String
    payload_pairs = []
    for doc in docs:
        content = getattr(doc, "page_content", "") or ""
        payload_pairs.append({
            "text": query,
            "text_pair": content[:max_length].strip()
        })

    payload = {"inputs": payload_pairs}

    url = "https://router.huggingface.co/hf-inference/models/BAAI/bge-reranker-v2-m3"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    for attempt in range(3):
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, headers=headers, json=payload)
                
                if response.status_code in [503, 504] and attempt < 2:
                    time.sleep(5)
                    continue

                if response.status_code == 400:
                    print("HF API 400 Bad Request Details:", response.text)

                response.raise_for_status()
                results = response.json()

                # Robustes Auslesen der Ergebnisse
                scores = []
                
                # Falls API ein einzelnes verschachteltes Ergebnis-Objekt zurückgibt
                if isinstance(results, dict) and "scores" in results:
                    results = results["scores"]

                if isinstance(results, list):
                    # Wenn das Ergebnis eine doppelt verschachtelte Liste ist (z.B. [[0.8], [0.1]])
                    if len(results) == 1 and isinstance(results[0], list) and len(results[0]) == num_docs:
                        results = results[0]

                    for item in results:
                        # Fall 1: Dict -> {"score": 0.85}
                        if isinstance(item, dict):
                            scores.append(float(item.get("score", 0.0)))
                        # Fall 2: Liste -> [0.85]
                        elif isinstance(item, list) and len(item) > 0:
                            elem = item[0]
                            if isinstance(elem, dict):
                                scores.append(float(elem.get("score", 0.0)))
                            else:
                                scores.append(float(elem))
                        # Fall 3: Zahl / String
                        elif isinstance(item, (int, float, str)):
                            scores.append(float(item))
                        else:
                            scores.append(0.0)

                # Sicherheitscheck: Länge MUSS exakt mit len(docs) übereinstimmen
                if len(scores) == num_docs:
                    return scores
                else:
                    print(f"Warnung: HF API gab {len(scores)} Scores für {num_docs} Dokumente zurück.")
                    # Auffüllen oder Abschneiden auf genau num_docs
                    if len(scores) < num_docs:
                        scores.extend([0.0] * (num_docs - len(scores)))
                    return scores[:num_docs]

        except Exception as e:
            if attempt == 2:
                print(f"Reranking fehlgeschlagen: {e}")
                return [0.0] * num_docs
            time.sleep(3)

    return [0.0] * num_docs


    
            
# ---------------------------------------------------------------------------
# 3. Hauptfunktion rerank_documents
# ---------------------------------------------------------------------------
def rerank_documents(
    query: str,
    docs_hybrid: list[Document],
    weight_hybrid: float = 0.2,
    weight_reranker: float = 0.8,
    temperature: float = 0.2,
    top_k: int = TOP_K,
) -> list[Document]:
    """
    Rerankt eine Liste von LangChain Document-Objekten mithilfe des
    Hugging Face Inference API Cross-Encoders und wählt die top_k diversesten Ergebnisse aus.
    """
    if not docs_hybrid:
        return []

    # 1. Hybrid Scores extrahieren und min-max normalisieren
    raw_hybrid_scores = [
        float(doc.metadata.get("score", 0.0)) for doc in docs_hybrid
    ]
    norm_hybrid_scores = min_max_normalize(raw_hybrid_scores)

    # 2. Raw Rerank-Scores über HF API abrufen
    raw_rerank_scores = compute_rerank_scores_api(query, docs_hybrid)

    # 3. Softmax-Normalisierung anwenden
    norm_rerank_scores = temperature_softmax_normalize(
        raw_rerank_scores, 
        temp=temperature
    )

    # 4. Neue Document-Objekte mit Blended Scores erstellen
    blended_docs: List[Document] = []
    for idx, doc in enumerate(docs_hybrid):
        h_score = norm_hybrid_scores[idx]
        r_score = norm_rerank_scores[idx]
        final_score = (weight_hybrid * h_score) + (weight_reranker * r_score)

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

    # 5. Nach Score sortieren & Diversität anwenden
    blended_docs.sort(key=lambda d: d.metadata["blended_score"], reverse=True)
    print_result("docs: rerank_retriever(): sorted", blended_docs)

    return select_diverse_top_k(blended_docs, max_k=top_k)



    





    