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
# PATH CONFIGURATION
# ============================================================

CONFIG_DIR = Path(__file__).resolve().parent
BASE_DIR = CONFIG_DIR.parent.parent.parent
CONFIG_PATH = CONFIG_DIR / "rag_config.yaml"

# ============================================================
# CERTIFICATES
# ============================================================

CERTS_DIR = BASE_DIR / "certs"
COMPANY_ROOT = CERTS_DIR / "RB-RootCA-RSA-G01.crt"
COMPANY_PROXY = CERTS_DIR / "RB-Proxy-TLS-CA.crt"
CERT_BUNDLE_PATH = CERTS_DIR / "Company_Internet_Kombi.crt"

def create_combined_cert() -> Path:
    """Create a CA bundle from certifi and company certificates."""

    if not COMPANY_ROOT.exists() or not COMPANY_PROXY.exists():
        if CERT_BUNDLE_PATH.exists():
            try:
                CERT_BUNDLE_PATH.unlink()
                print(
                    "Info: Existing company CA bundle removed because "
                    "required company certificates are missing."
                )
            except OSError as e:
                print(
                    f"Warning: Could not remove old CA bundle: {e}"
                )

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
        print(f"Warning: Could not create combined CA bundle: {e}")
        return Path(certifi.where())

        


# ============================================================
# SSL CERTIFICATE CONFIGURATION
# ============================================================

CERT_FILE = create_combined_cert()

if CERT_FILE and CERT_FILE.exists():
    os.environ["SSL_CERT_FILE"] = str(CERT_FILE)
    os.environ["REQUESTS_CA_BUNDLE"] = str(CERT_FILE)

# ============================================================
# YAML CONFIGURATION
# ============================================================

def load_yaml_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file '{config_path}' was not found."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_config = load_yaml_config(CONFIG_PATH)

# ============================================================
# SECRETS / ENVIRONMENT FALLBACK
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
# API KEYS & SENSITIVE DATA
# ============================================================

OPENAI_API_KEY = get_secret_or_env("OPENAI_API_KEY")
PINECONE_API_KEY = get_secret_or_env("PINECONE_API_KEY")
PINECONE_API_KEY2 = get_secret_or_env("PINECONE_API_KEY2")
COMPANY_PROXY_URL = get_secret_or_env("COMPANY_PROXY_URL")

# ============================================================
# NETWORK & PROXY SETTINGS
# ============================================================

_network = _config.get("network", {})

PX_HOST: str = str(_network.get("px_host", "127.0.0.1"))
PX_PORT: int = int(_network.get("px_port", 3128))
TIMEOUT: float = float(_network.get("timeout", 60.0))
RETRIES: int = int(_network.get("retries", 3))

# ============================================================
# SSL VERIFICATION
# ============================================================

IGNORE_SSL: bool = (
    True
    if not COMPANY_ROOT.exists() or not COMPANY_PROXY.exists()
    else bool(_network.get("ignore_ssl", False))
)


# ============================================================
# PATHS
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
# LLM & EMBEDDING CONFIGURATION
# ============================================================

_llm = _config.get("llm", {})

LLM_ENRICHMENT_MODEL = _llm.get(
    "llm_enrichment",
    "gpt-5.4-mini",
)

LLM_QUERY_MODEL = _llm.get(
    "llm_query",
    "gpt-5.4-nano",
)

LLM_ANSWER_MODEL = _llm.get(
    "llm_answer",
    "gpt-5.4-mini",
)

_embedding = _config.get("embedding", {})

EMBEDDING_MODEL = _embedding.get(
    "embedding_model",
    "text-embedding-3-small",
)

# ============================================================
# CHUNKING CONFIGURATION
# ============================================================

_chunking = _config.get("chunking", {})

CHUNK_SIZE: int = int(_chunking.get("size", 1000))
CHUNK_OVERLAP: int = int(_chunking.get("overlap", 200))

# ============================================================
# LANGCHAIN / NETWORK ENVIRONMENT
# ============================================================

os.environ["LANGCHAIN_OPENAI_TCP_KEEPALIVE"] = "0"

if COMPANY_PROXY_URL:
    os.environ["HTTP_PROXY"] = COMPANY_PROXY_URL
    os.environ["HTTPS_PROXY"] = COMPANY_PROXY_URL

# ============================================================
# TEST MARKER
# ============================================================

EDITABLE_TEST_MARKER = "Der -e Modus funktioniert!"