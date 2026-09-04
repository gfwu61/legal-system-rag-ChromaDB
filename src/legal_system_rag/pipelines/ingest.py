import hashlib
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
import textwrap

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from legal_system_rag.config.settings import (
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    IGNORE_SSL,
    LLM_ENRICHMENT_MODEL,
    OPENAI_API_KEY,
    PERSIST_DIRECTORY,
    BATCH_SIZE,
)
from legal_system_rag.network.client_factory import (
    create_http_client,
    get_proxy_url,
)
from legal_system_rag.parser.legal_parser import (
    extract_references,
    split_absaetze,
    split_nummern,
    split_paragraphs,
)
from legal_system_rag.rag.chain import (
    build_page_content,
    enrich_chunk,
    create_structured_body,
)
from legal_system_rag.rag.prompts import EnrichmentOutput


# ============================================================
# LOGGING
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[3]
LOG_FILE = PROJECT_DIR / "ingest.log"

logger = logging.getLogger("legal_rag.ingest")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(
        LOG_FILE,
        mode="a",
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

logger.propagate = False


# ============================================================
# DOCUMENT ID: deterministic with this function!!! very important
# ============================================================
""" (.*?): ? (Non-Greedy / Faul), letter for letter check by (.*?)  and (?=  \n[A-Z_]+:  |  $  )
(?=  \n[A-Z_]+:  |  $  )
 ▲   └────┬────┘ │  ▲
 │        │      │  └─ 3. Oder: Ende des Gesamtextes
 │        │      └──── 2. Oder-Operator
 │        └─────────── 1. Erste Stopp-Option (Header)
 └──────────────────── Sonderfunktion: Lookahead (prüfen ohne mitzunehmen)
"""

def extract_original_text(page_content: str) -> str:
    # Liest ab "ORIGINALTEXT:" bis zum nächsten Wort, das komplett GROSSGESCHRIEBEN ist und mit ":" endet
    # oder bis zum Ende des Strings ($)
    pattern = r"ORIGINALTEXT:\s*(.*?)(?=\n[A-Z_]+:|$)"
    match = re.search(pattern, page_content, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    return page_content

def extract_original_text(page_content: str) -> str:
    # Liest ab "ORIGINALTEXT:" bis zum nächsten Wort, das komplett GROSSGESCHRIEBEN ist und mit ":" endet
    # oder bis zum Ende des Strings ($)
    pattern = r"ORIGINALTEXT:\s*(.*?)(?=\n[A-Z_]+:|$)"
    match = re.search(pattern, page_content, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    return page_content
    
"""
def create_chunk_id(
    gesetz: str,
    paragraph: str,
    absatz: str | None,
    nummer: str | None,
) -> str:
    return (
        f"{gesetz}"
        f"_{paragraph}"
        f"_{absatz or 'none'}"
        f"_{nummer or 'none'}"
    )
"""


def create_document_id(doc: Document) -> str:
    """
    Create a stable ID for a document.
    """
    metadata = doc.metadata

    gesetz = str(metadata.get("gesetz", "unknown"))
    paragraph = str(metadata.get("paragraph", "unknown"))
    absatz = str(metadata.get("absatz", "unknown"))
    nummer = str(metadata.get("nummer", "none"))

    # Extrahiere original_text direkt aus page_content
    """
    original_text = extract_original_text(doc.page_content)

    content_hash = hashlib.sha256(
    original_text.encode("utf-8")
    ).hexdigest()[:16]
    """
    
    raw_id = (
        f"{gesetz}"
        f"_{paragraph}"
        f"_{absatz}"
        f"_{nummer}"
    )

    print("ID:", raw_id)
    print("Paragraph:", paragraph)
    print("Absatz:", absatz)
    print("Nummer:", nummer)
    #print("Content:", original_text[:600])
    print("-" * 80)

    return (
        raw_id
        .replace(" ", "_")
        .replace(".", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


# ============================================================
# ID VALIDATION
# ============================================================

def validate_unique_ids(ids: list[str]) -> None:
    """ Check whether all document IDs are unique. """
    counts = Counter(ids)
    duplicate_ids = sorted(
        doc_id
        for doc_id, count in counts.items()
        if count > 1
    )

    if duplicate_ids:
        raise ValueError(
            "Duplicate document IDs found: "
            f"{duplicate_ids}"
        )


# ============================================================
# DOCUMENT CREATION
# ============================================================

def process_and_create_doc(
    gesetz_name: str,
    paragraph_number: str,
    paragraph_title: str,
    absatz_number: str,
    nummer: str | None,
    subsection_text: str,
    nummer_text: str | None,
    filename: str,
    structured_llm,
) -> Document:
    """Create a Document from a section of a law."""
    structured_body = create_structured_body(subsection_text, nummer_text)

    enrichment_text = textwrap.dedent(f"""\
        § {paragraph_number} {paragraph_title}
        
        {structured_body}
    """).strip()

    t0 = time.time()
    enrichment = enrich_chunk(enrichment_text, structured_llm)
    print(f"  ⏱ Enrichment Dauer: {time.time() - t0:.2f} Sekunden")

    references = extract_references(structured_body)

    page_content = build_page_content(
        gesetz=gesetz_name,
        paragraph=paragraph_number,
        paragraph_title=paragraph_title,
        absatz=absatz_number,
        nummer=nummer,
        subsection_text=subsection_text,
        nummer_text=nummer_text,
        references=references,
        enrichment=enrichment,
    )
    original_text = extract_original_text(page_content)
    print(" === > process_and_create_doc: ")
    print("Content:", original_text[:600])
    content_hash = hashlib.sha256(
        original_text.encode("utf-8")
    ).hexdigest()[:16]


    metadata = {
        "gesetz": gesetz_name,
        "rechtsgebiet": "Mietrecht",
        "thema": enrichment.topic,
         
        # Chunk hierarchy
        "paragraph": str(paragraph_number),
        "paragraph_titel": paragraph_title if paragraph_title else "-",
        "absatz": str(absatz_number) if absatz_number else "-",
        "nummer": str(nummer) if nummer else "-",
       
        
        # Source
        "source": "BGB.xml",
        "file": filename,
    # Incremental ingestion
        "content_hash": content_hash,
        #"enrichment_version": "v1",
    }

# Dynamisches Überschreiben für den spezifischsten Chunk-Typ
    if paragraph_number:
        metadata["chunk_type"] = "paragraph"
    if absatz_number:
        metadata["chunk_type"] = "absatz"
    if nummer:
        metadata["chunk_type"] = "nummer"

    return Document(
        page_content=page_content,
        metadata=metadata,
    )


# ============================================================
# DOCUMENT LOGGING
# ============================================================

def log_documents(
    documents: list[Document],
    ids: list[str],
) -> None:
    """ Write all generated Document objects to ingest.log. """
    logger.info("=" * 100)
    logger.info("BATCH_DOCUMENTS: %d Dokumente", len(documents))
    logger.info("=" * 100)

    for index, (document, document_id) in enumerate(
        zip(documents, ids),
        start=1,
    ):
        logger.info("")
        logger.info("-" * 100)
        logger.info("Dokument %d von %d", index, len(documents))
        logger.info("ID: %s", document_id)
        logger.info("METADATA: %s", document.metadata)
        logger.info("PAGE_CONTENT: %s", document.page_content)
        logger.info("-" * 100)


# ============================================================
# INGESTION PIPELINE
# ============================================================

def run_ingestion() -> None:
    """Read TXT files, create document chunks, enrich them and store in Chroma with batching."""

    if not (OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")):
        raise ValueError("OPENAI_API_KEY fehlt.")

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

    proxy_url = get_proxy_url()
    sync_client = create_http_client(
        proxy_url=proxy_url,
        ignore_ssl=IGNORE_SSL,
    )

    logger.info(
        "Ingestion HTTP client created | proxy=%s | ignore_ssl=%s",
        proxy_url,
        IGNORE_SSL,
    )

    try:
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            http_client=sync_client,
        )

        vector_store = Chroma(
            persist_directory=str(PERSIST_DIRECTORY),
            embedding_function=embeddings,
        )

        llm_enrichment = ChatOpenAI(
            model=LLM_ENRICHMENT_MODEL,
            temperature=0.2,
            http_client=sync_client,
        )

        structured_llm = llm_enrichment.with_structured_output(EnrichmentOutput)

        all_documents: list[Document] = []
        # BATCH_SIZE = 20
        total_processed_count = 0
        batch_counter = 1

        txt_files = sorted(DOCUMENTS_DIR.glob("*.txt"))

        if not txt_files:
            print(f"⚠️ Keine .txt-Dateien in '{DOCUMENTS_DIR}' gefunden.")
            logger.warning("Keine Quelltexte im Verzeichnis %s vorhanden.", DOCUMENTS_DIR)
            return

        count_before = vector_store._collection.count()
        print(f"\n📊 Anzahl der Chunks in Chroma VOR dem Speichern: {count_before}")

        for filepath in txt_files:
            filename = filepath.name
            gesetz_name = filepath.stem.upper()

            print(f"\n📄 Verarbeite Datei: {filename}")
            logger.info("Verarbeite Datei: %s", filename)

            full_text = filepath.read_text(encoding="utf-8")
            paragraphs = split_paragraphs(full_text)

            for para in paragraphs:
                paragraph_number = para["paragraph"]
                paragraph_title = para["title"]
                paragraph_content = para["content"]

                print(f"  ➡ Analysiere §{paragraph_number} {paragraph_title}")

                absaetze = split_absaetze(paragraph_content)

                for absatz_item in absaetze:
                    absatz_number = absatz_item["absatz"]
                    absatz_content = absatz_item["content"]

                    nummern_list = split_nummern(absatz_content)

                    # Fall 1: Keine Unter-Nummerierung
                    if not nummern_list:
                        doc = process_and_create_doc(
                            gesetz_name=gesetz_name,
                            paragraph_number=paragraph_number,
                            paragraph_title=paragraph_title,
                            absatz_number=absatz_number,
                            nummer=None,
                            subsection_text=absatz_content,
                            nummer_text=None,
                            filename=filename,
                            structured_llm=structured_llm,
                        )
                        all_documents.append(doc)
                        total_processed_count += 1
                        continue

                    # Fall 2: Mit Unter-Nummerierung
                    for nummer_item in nummern_list:
                        doc_nummer = process_and_create_doc(
                            gesetz_name=gesetz_name,
                            paragraph_number=paragraph_number,
                            paragraph_title=paragraph_title,
                            absatz_number=absatz_number,
                            nummer=nummer_item["nummer"],
                            subsection_text=nummer_item["subsection_text"],
                            nummer_text=nummer_item["content"],
                            filename=filename,
                            structured_llm=structured_llm,
                        )
                        all_documents.append(doc_nummer)
                        total_processed_count += 1

                # PRÜFUNG UNTER DEM PARAGRAPHEN
                if len(all_documents) >= BATCH_SIZE:
                    batch_ids = [create_document_id(doc) for doc in all_documents]
                    validate_unique_ids(batch_ids)
                    log_documents(documents=all_documents, ids=batch_ids)

                    print(f"\n💾 Speichere Batch {batch_counter} mit {len(all_documents)} Chunks...")
                    vector_store.add_documents(
                        documents=all_documents,
                        ids=batch_ids,
                    )
                    print(f"📦 Batch {batch_counter} erfolgreich gespeichert & RAM geleert.")
                    
                    # Liste zurücksetzen = RAM freigeben
                    all_documents.clear()
                    batch_counter += 1

        # ====================================================
        # RESTLICHE DOKUMENTE SPEICHERN (Falls len < 20 am Ende)
        # ====================================================
        if all_documents:
            batch_ids = [create_document_id(doc) for doc in all_documents]
            validate_unique_ids(batch_ids)
            log_documents(documents=all_documents, ids=batch_ids)

            print(f"\n💾 Speichere finalen Batch {batch_counter} mit {len(all_documents)} Chunks...")
            vector_store.add_documents(
                documents=all_documents,
                ids=batch_ids,
            )
            print(f"📦 Letzter Batch {batch_counter} gespeichert & RAM geleert.")
            all_documents.clear()

        if total_processed_count == 0:
            print("⚠️ Es wurden keine Dokument-Chunks erzeugt.")
            logger.info("Keine Dokument-Chunks generiert.")
            return

        count_after = vector_store._collection.count()
        print(f"\n📊 Anzahl der Chunks in Chroma NACH dem Speichern: {count_after}")
        print(f"➕ Differenz: +{count_after - count_before} Chunks")

        logger.info("Ingestion erfolgreich abgeschlossen.")
        print("✅ Ingestion erfolgreich abgeschlossen.")

    except Exception as e:
        logger.exception("Schwerwiegender Fehler während der Ingestion:")
        print(f"❌ Fehler während der Ingestion: {e}")
        raise

    finally:
        sync_client.close()


if __name__ == "__main__":
    run_ingestion()