import hashlib
import os
import logging
from pathlib import Path

from collections import Counter

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from legal_system_rag.config.settings import (
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    LLM_ENRICHMENT_MODEL,
    OPENAI_API_KEY,
    PERSIST_DIRECTORY,
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
)
from legal_system_rag.rag.prompts import EnrichmentOutput


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





def create_document_id(doc: Document) -> str:
    """
    Erstellt eine stabile ID für ein Dokument.

    Die ID besteht aus:
    - Quelldatei
    - Paragraph
    - Absatz
    - Nummer
    - SHA-256-Hash des Originaltexts
    """

    metadata = doc.metadata

    source = str(metadata.get("source", "unknown"))
    paragraph = str(metadata.get("paragraph", "unknown"))
    absatz = str(metadata.get("absatz", "unknown"))
    nummer = str(metadata.get("nummer", "none"))

    original_text = str(
        metadata.get(
            "original_text",
            doc.page_content,
        )
    )

    content_hash = hashlib.sha256(
        original_text.encode("utf-8")
    ).hexdigest()[:16]

    raw_id = (
        f"{source}_"
        f"P{paragraph}_"
        f"A{absatz}_"
        f"N{nummer}_"
        f"H{content_hash}"
    )

    print("ID:", raw_id)
    print("Paragraph:", paragraph)
    print("Absatz:", absatz)
    print("Nummer:", nummer)
    print("Hash:", content_hash)
    print("Content:", original_text[:300])
    print("-" * 80)

    return (
        raw_id
        .replace(" ", "_")
        .replace(".", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def validate_unique_ids(ids: list[str]) -> None:
    """
    Prüft, ob alle Dokument-IDs eindeutig sind.
    """

    counts = Counter(ids)

    duplicate_ids = sorted(
        doc_id
        for doc_id, count in counts.items()
        if count > 1
    )

    if duplicate_ids:
        raise ValueError(
            "Doppelte Dokument-IDs gefunden: "
            f"{duplicate_ids}"
        )


def process_and_create_doc(
    gesetz_name: str,
    paragraph_number: str,
    paragraph_title: str,
    absatz_number: str,
    nummer: str | None,
    original_text: str,
    filename: str,
    structured_llm,
) -> Document:
    """
    Erstellt aus einem Gesetzesabschnitt ein Dokument.
    """

    references = extract_references(original_text)

    enrichment = enrich_chunk(
        original_text,
        structured_llm,
    )

    page_content = build_page_content(
        gesetz=gesetz_name,
        paragraph=paragraph_number,
        paragraph_title=paragraph_title,
        absatz=absatz_number,
        nummer=nummer,
        references=references,
        enrichment=enrichment,
        original_text=original_text,
    )

    metadata = {
        "gesetz": gesetz_name,
        "paragraph": str(paragraph_number),
        "absatz": str(absatz_number),
        "nummer": str(nummer) if nummer else "none",
        "source": filename,
        "original_text": original_text,
    }

    return Document(
        page_content=page_content,
        metadata=metadata,
    )


def log_documents(
    documents: list[Document],
    ids: list[str],
) -> None:
    """
    Schreibt alle erzeugten Document-Objekte
    inklusive IDs, page_content und Metadaten
    in ingest.log.
    """

    logger.info("=" * 100)
    logger.info(
        "ALL_DOCUMENTS: %d Dokumente",
        len(documents),
    )
    logger.info("=" * 100)

    for index, (document, document_id) in enumerate(
        zip(documents, ids),
        start=1,
    ):
        logger.info("")
        logger.info("-" * 100)
        logger.info(
            "Dokument %d von %d",
            index,
            len(documents),
        )
        logger.info("ID: %s", document_id)

        logger.info("METADATA:")
        logger.info("%s", document.metadata)

        logger.info("PAGE_CONTENT:")
        logger.info("%s", document.page_content)

        logger.info("-" * 100)

    logger.info(
        "ALL_DOCUMENTS erfolgreich in %s geschrieben",
        LOG_FILE,
    )


    

def run_ingestion() -> None:
    """
    Liest TXT-Dateien ein, erzeugt Dokument-Chunks,
    reichert sie per LLM an und speichert sie in Chroma.
    """

    if not (
        OPENAI_API_KEY
        or os.environ.get("OPENAI_API_KEY")
    ):
        raise ValueError("OPENAI_API_KEY fehlt.")

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PERSIST_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    proxy_url = get_proxy_url()

    sync_client = create_http_client(
        proxy_url=proxy_url,
    )
    # langchain uses OPENAI_API_KEY as standard env, api_key=OPENAI_API_KEY(as variable) is optional
    try:
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            #api_key=OPENAI_API_KEY, 
            http_client=sync_client,
           
        )

        vector_store = Chroma(
            persist_directory=str(PERSIST_DIRECTORY),
            embedding_function=embeddings,
        )

        llm_enrichment = ChatOpenAI(
            model=LLM_ENRICHMENT_MODEL,
            # api_key=OPENAI_API_KEY,
            temperature=0.2,
            http_client=sync_client,

        )

        structured_llm = (
            llm_enrichment.with_structured_output(
                EnrichmentOutput
            )
        )

        all_documents: list[Document] = []

        txt_files = sorted(
            DOCUMENTS_DIR.glob("*.txt")
        )

        if not txt_files:
            print(
                f"⚠️ Keine .txt-Dateien in "
                f"'{DOCUMENTS_DIR}' gefunden."
            )
            return

        for filepath in txt_files:
            filename = filepath.name

            print(
                f"\n📄 Verarbeite Datei: {filename}"
            )

            full_text = filepath.read_text(
                encoding="utf-8"
            )

            gesetz_name = filepath.stem.upper()

            paragraphs = split_paragraphs(
                full_text
            )

            for para in paragraphs:
                paragraph_number = para["paragraph"]
                paragraph_title = para["title"]
                paragraph_content = para["content"]

                print(
                    f"  ➡ Analysiere "
                    f"§{paragraph_number} "
                    f"{paragraph_title}"
                )

                absaetze = split_absaetze(
                    paragraph_content
                )

                for absatz_item in absaetze:
                    absatz_number = absatz_item["absatz"]
                    absatz_content = absatz_item["content"]

                    nummern = split_nummern(
                        absatz_content
                    )

                    if not nummern:
                        doc = process_and_create_doc(
                            gesetz_name=gesetz_name,
                            paragraph_number=paragraph_number,
                            paragraph_title=paragraph_title,
                            absatz_number=absatz_number,
                            nummer=None,
                            original_text=absatz_content,
                            filename=filename,
                            structured_llm=structured_llm,
                        )

                        all_documents.append(doc)
                        continue

                    intro_text = nummern[0].get(
                        "intro_isolated",
                        "",
                    )

                    if intro_text and len(intro_text) > 10:
                        doc_intro = process_and_create_doc(
                            gesetz_name=gesetz_name,
                            paragraph_number=paragraph_number,
                            paragraph_title=paragraph_title,
                            absatz_number=absatz_number,
                            nummer="Einleitung",
                            original_text=intro_text,
                            filename=filename,
                            structured_llm=structured_llm,
                        )

                        all_documents.append(doc_intro)

                    for nummer_item in nummern:
                        doc_nummer = process_and_create_doc(
                            gesetz_name=gesetz_name,
                            paragraph_number=paragraph_number,
                            paragraph_title=paragraph_title,
                            absatz_number=absatz_number,
                            nummer=nummer_item["nummer"],
                            original_text=nummer_item["content"],
                            filename=filename,
                            structured_llm=structured_llm,
                        )

                        all_documents.append(doc_nummer)

        if not all_documents:
            print(
                "⚠️ Es wurden keine "
                "Dokument-Chunks erzeugt."
            )
            return

        ids = [
            create_document_id(doc)
            for doc in all_documents
        ]

        validate_unique_ids(ids)

        logger.info(
            "Speichere %d Dokument-Chunks in Chroma",
        len(all_documents),
        )

        log_documents(
        documents=all_documents,
        ids=ids,
        )



        print(
            f"\n💾 Speichere {len(all_documents)} "
            "Dokument-Chunks in Chroma..."
        )

        vector_store.add_documents(
            documents=all_documents,
            ids=ids,
        )

        print(
            "✅ Ingestion erfolgreich "
            "abgeschlossen."
        )

    finally:
        sync_client.close()

        # Der AsyncClient wird hier nicht synchron geschlossen.
        # In einer vollständig asynchronen Pipeline:
        #
        # await async_client.aclose()