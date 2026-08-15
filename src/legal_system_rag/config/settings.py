import os
from pathlib import Path

import certifi
import yaml
from dotenv import load_dotenv

# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

# ============================================================
# PFAD-ERMITTLUNG
# ============================================================

CONFIG_DIR = Path(__file__).resolve().parent
BASE_DIR = CONFIG_DIR.parent.parent.parent
CONFIG_PATH = CONFIG_DIR / "rag_config.yaml"

# ============================================================
# ZERTIFIKATE
# ============================================================

CERTS_DIR = BASE_DIR / "certs"
COMPANY_ROOT = CERTS_DIR / "RB-RootCA-RSA-G01.crt"
COMPANY_PROXY = CERTS_DIR / "RB-Proxy-TLS-CA.crt"
CERT_BUNDLE_PATH = CERTS_DIR / "Company_Internet_Kombi.crt"


def create_combined_cert() -> Path:
    """Erstellt ein CA-Bundle aus certifi + Firmen-Zertifikaten."""
    if not COMPANY_ROOT.exists() or not COMPANY_PROXY.exists():
        return Path(certifi.where())

    try:
        CERTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(CERT_BUNDLE_PATH, "w", encoding="utf-8") as out_f:
            with open(certifi.where(), "r", encoding="utf-8") as certifi_f:
                out_f.write(certifi_f.read())
                out_f.write("\n")

            with open(COMPANY_PROXY, "r", encoding="utf-8") as proxy_f:
                out_f.write(proxy_f.read())
                out_f.write("\n")

            with open(COMPANY_ROOT, "r", encoding="utf-8") as root_f:
                out_f.write(root_f.read())
                out_f.write("\n")

        return CERT_BUNDLE_PATH

    except (OSError, UnicodeError) as e:
        print(f"Warnung: Kombiniertes CA-Bundle konnte nicht erstellt werden: {e}")
        return Path(certifi.where())


CERT_FILE = create_combined_cert()

if CERT_FILE and CERT_FILE.exists():
    os.environ["SSL_CERT_FILE"] = str(CERT_FILE)
    os.environ["REQUESTS_CA_BUNDLE"] = str(CERT_FILE)

# ============================================================
# YAML CONFIG LOAD
# ============================================================

def load_yaml_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Konfigurationsdatei '{config_path}' wurde nicht gefunden."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_config = load_yaml_config(CONFIG_PATH)

# ============================================================
# HELPER: SECRETS / ENV FALLBACK
# ============================================================

def get_secret_or_env(key: str) -> str | None:
    value = os.getenv(key)
    if value:
        return value

    try:
        import streamlit as st
        if key in st.secrets:
            value = st.secrets[key]
            os.environ[key] = str(value)
            return str(value)
    except Exception:
        pass

    return None

# ============================================================
# API KEYS & SENSIBLE DATEN
# ============================================================

OPENAI_API_KEY = get_secret_or_env("OPENAI_API_KEY")
PINECONE_API_KEY = get_secret_or_env("PINECONE_API_KEY")
PINECONE_API_KEY2 = get_secret_or_env("PINECONE_API_KEY2")
COMPANY_PROXY_URL = get_secret_or_env("COMPANY_PROXY_URL")

# ============================================================
# NETWORK & PROXY SETTINGS (NEU AUS YAML)
# ============================================================

_network = _config.get("network", {})
PX_HOST: str = str(_network.get("px_host", "127.0.0.1"))
PX_PORT: int = int(_network.get("px_port", 3128))
TIMEOUT: float = float(_network.get("timeout", 60.0))
RETRIES: int = int(_network.get("retries", 3))

# ============================================================
# PFADE
# ============================================================

_paths = _config.get("paths", {})

DOCUMENTS_DIR = Path(
    os.getenv(
        "DOCUMENTS_DIR",
        str(BASE_DIR / _paths.get("documents", "data/dokumente_mietrecht")),
    )
)

PERSIST_DIRECTORY = Path(
    os.getenv(
        "PERSIST_DIRECTORY",
        str(BASE_DIR / _paths.get("rag_vector_db", "chroma_legal_rag")),
    )
)

# ============================================================
# LLM- & EMBEDDING-KONFIGURATION
# ============================================================

_llm = _config.get("llm", {})
LLM_ENRICHMENT_MODEL = _llm.get("llm_enrichment", "gpt-5.4-mini")
LLM_QUERY_MODEL = _llm.get("llm_query", "gpt-5.4-nano")
LLM_ANSWER_MODEL = _llm.get("llm_answer", "gpt-5.4-mini")

_embedding = _config.get("embedding", {})
EMBEDDING_MODEL = _embedding.get("embedding_model", "text-embedding-3-small")

_chunking = _config.get("chunking", {})
CHUNK_SIZE = int(_chunking.get("size", 1000))
CHUNK_OVERLAP = int(_chunking.get("overlap", 200))

# ============================================================
# LANGCHAIN / NETWORK SETTINGS
# ============================================================

os.environ["LANGCHAIN_OPENAI_TCP_KEEPALIVE"] = "0"

if COMPANY_PROXY_URL:
    os.environ["HTTP_PROXY"] = COMPANY_PROXY_URL
    os.environ["HTTPS_PROXY"] = COMPANY_PROXY_URL

EDITABLE_TEST_MARKER = "Der -e Modus funktioniert!"