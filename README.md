# Legal RAG System

An end-to-end Retrieval-Augmented Generation (RAG) system for German tenancy law (*Mietrecht*), implemented as a modular Python application.

The project demonstrates a production-oriented RAG architecture with:

- Structured legal document parsing and enrichment
- Persistent ChromaDB vector storage
- Dense vector retrieval
- Sparse BM25 retrieval
- Hybrid retrieval
- Cross-encoder reranking
- LLM-based query routing
- Agentic retrieval with relevance evaluation
- Automatic query rewriting for self-correction
- Grounded answer generation
- Source-document transparency
- Streamlit chat UI
- Pytest unit, integration, and end-to-end test structure
- Configurable corporate proxy / PX networking
- Python packaging and command-line entry points

The system is designed as a technical demonstration and learning project rather than as legal advice or a production legal service.

---

## 1. Project Overview

The application answers questions about a curated collection of German tenancy-law documents.

The central design principle is separation of concerns:

```text
                         User Question
                              │
                              ▼
                    ┌───────────────────┐
                    │   RAG Application  │
                    │    Streamlit UI    │
                    └─────────┬─────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
        Standard RAG Mode          Agentic RAG Mode
                │                           │
                │                    ┌──────▼──────┐
                │                    │ LLM Router  │
                │                    └──────┬──────┘
                │                           │
                │                    Search Strategy
                │                           │
                └─────────────┬─────────────┘
                              ▼
                    Hybrid Retrieval
                   ┌──────────┴──────────┐
                   │                     │
                   ▼                     ▼
              Dense Search           BM25 Search
               ChromaDB             Sparse Search
                   │                     │
                   └──────────┬──────────┘
                              ▼
                       Candidate Chunks
                              │
                              ▼
                       Cross-Encoder
                         Reranking
                              │
                              ▼
                    Top-K Relevant Chunks
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
             Standard Mode       Agentic Mode
                    │             Relevance Evaluation
                    │                   │
                    │              ┌────┴────┐
                    │              │         │
                    │             Yes        No
                    │              │         │
                    │              │    Query Rewrite
                    │              │         │
                    └──────────────┴─────────┘
                              │
                              ▼
                         Answer LLM
                              │
                              ▼
                    Grounded Legal Answer
                              │
                              ▼
                     Sources + Metadata
```

---

# 2. Main Features

## 2.1 Two RAG Execution Modes

The Streamlit application provides two retrieval modes.

### Mode 1 — Standard RAG

A conventional linear RAG pipeline:

```text
Question
   │
   ▼
Query Processing
   │
   ▼
Hybrid Retrieval
   │
   ├── Dense / ChromaDB
   └── Sparse / BM25
   │
   ▼
Reranking
   │
   ▼
Top-K Documents
   │
   ▼
Answer LLM
   │
   ▼
Answer
```

This mode is useful as a strong baseline and makes it possible to compare a conventional RAG pipeline with the agentic approach.

### Mode 2 — Agentic / Self-Corrective RAG

The agentic mode adds an LLM-based routing and evaluation loop:

```text
User Question
      │
      ▼
   LLM Router
      │
      ├── specific_norm
      │
      └── concept_search
      │
      ▼
Search Query + Filters
      │
      ▼
Hybrid Retrieval
      │
      ▼
Candidate Documents
      │
      ▼
Cross-Encoder Reranking
      │
      ▼
Top-K Documents
      │
      ▼
LLM Relevance Evaluation
      │
      ├── Relevant ───────────────┐
      │                          │
      └── Not relevant           │
              │                  │
              ▼                  │
        Query Rewriting          │
              │                  │
              └──► Retry ────────┘
                                 │
                                 ▼
                         Final Answer LLM
```

The current implementation performs up to two retrieval attempts.

---

# 3. Agentic Retrieval

The agentic retrieval loop is implemented in:

```text
src/legal_system_rag/rag/agent.py
```

The loop performs the following steps.

### Step 1 — LLM Routing

The router classifies the user's question into one of two retrieval strategies:

```text
specific_norm
concept_search
```

`RouterDecision` is a Pydantic structured-output model.

For example:

```text
Question:
"Zeige mir den Wortlaut von § 573 BGB."

Route:
specific_norm

Paragraph filter:
["573"]
```

Whereas a question such as:

```text
"Was muss ich bei einer Eigenbedarfskündigung beachten?"
```

is treated as a conceptual legal question rather than a request for the exact wording of one paragraph.

This distinction allows the retriever to use paragraph-level filtering when appropriate.

---

## 3.1 Router Output

The router produces:

```python
RouterDecision(
    route="specific_norm",
    search_query="...",
    paragraph_filters=["573"],
)
```

The schema is defined in:

```text
src/legal_system_rag/rag/router.py
```

The router uses LLM structured output to produce predictable, typed decisions instead of parsing free-form LLM text.

---

# 4. Hybrid Retrieval

The retrieval layer is implemented primarily in:

```text
src/legal_system_rag/rag/retrieval.py
src/legal_system_rag/rag/chain.py
```

The system combines two complementary retrieval methods.

## Dense Retrieval

Semantic search is performed with:

```text
OpenAI text-embedding-3-small
        │
        ▼
     ChromaDB
```

Dense retrieval is useful when the user's wording differs from the wording of the legal text.

## Sparse Retrieval

BM25 provides lexical retrieval:

```text
User Query
    │
    ▼
 BM25 Index
    │
    ▼
Keyword / term matching
```

BM25 is particularly useful for legal terminology, paragraph identifiers, names, and exact words appearing in the source documents.

---

## 4.1 Hybrid Retrieval Strategy

The two retrieval approaches are combined:

```text
                 Query
                   │
          ┌────────┴────────┐
          ▼                 ▼
     ChromaDB              BM25
    Dense Search       Sparse Search
          │                 │
          └────────┬────────┘
                   ▼
             Hybrid Results
```

The implementation supports both the standard LangChain ensemble approach and a score-aware hybrid approach.

The score-aware implementation can normalize dense and BM25 scores and combine them before reranking.

---

# 5. Cross-Encoder Reranking

The reranking implementation is located in:

```text
src/legal_system_rag/rag/reranker.py
```

The project uses:

```text
BAAI/bge-reranker-v2-m3
```

through the Hugging Face inference service.

The purpose of reranking is to improve the ordering of candidate documents after the initial retrieval stage.

```text
Hybrid Retrieval
      │
      ▼
Candidate Documents
      │
      ▼
Cross-Encoder
      │
      ▼
Relevance Scores
      │
      ▼
Sorted Documents
      │
      ▼
Top-K Selection
```

The cross-encoder evaluates the query and document together rather than embedding them independently.

This allows a second-stage relevance assessment after the faster dense/sparse retrieval stage.

---

# 6. Self-Corrective Retrieval

The agentic pipeline adds an additional evaluation step after retrieval and reranking.

The implementation is in:

```text
src/legal_system_rag/rag/evaluators.py
```

The evaluation process asks an LLM whether the retrieved documents are sufficiently relevant to the original question.

```text
Retrieved Top-K
       │
       ▼
Relevance Evaluator
       │
       ├── yes
       │    │
       │    ▼
       │  Continue
       │
       └── no
            │
            ▼
      Query Rewriting
            │
            ▼
       New Search
```

The query-rewriting component generates an improved search query for the next retrieval attempt.

This is a form of self-corrective RAG:

```text
Retrieve
   ↓
Evaluate
   ↓
Rewrite
   ↓
Retrieve again
```

The current maximum number of attempts is:

```python
max_retries=2
```

---

# 7. Final Answer Generation

The end-to-end agentic pipeline is implemented in:

```text
src/legal_system_rag/rag/agentic_rag.py
```

After retrieval has selected the relevant chunks, the answer LLM receives:

```text
Original Question
        +
Retrieved Legal Context
        │
        ▼
     Answer LLM
        │
        ▼
Grounded Legal Answer
```

The prompt explicitly prioritizes the original legal text.

Additional enriched information such as:

- topic
- plain-language explanation
- extracted references

is used as supporting context.

The final answer is generated in German.

---

# 8. Legal Document Ingestion

The ingestion entry point is:

```text
src/legal_system_rag/ingest_documents.py
```

The main ingestion implementation is:

```text
src/legal_system_rag/pipelines/ingest.py
```

The pipeline is:

```text
Legal TXT Documents
        │
        ▼
Text Normalization
        │
        ▼
Paragraph Detection
        │
        ▼
Absatz Detection
        │
        ▼
Nummer Detection
        │
        ▼
Reference Extraction
        │
        ▼
LLM Enrichment
        │
        ▼
Structured Document
        │
        ▼
OpenAI Embedding
        │
        ▼
ChromaDB
```

Documents are inserted in batches.

The current configuration uses:

```yaml
chunking:
  size: 1000
  overlap: 200
  batch_size: 20
```

The legal parser, however, primarily respects the legal hierarchy rather than treating the source text as arbitrary prose.

---

# 9. Legal-Aware Chunking

The parser is implemented in:

```text
src/legal_system_rag/parser/legal_parser.py
```

Important functions include:

```python
normalize_text()
split_paragraphs()
split_absaetze()
split_nummern()
extract_references()
```

The parser recognizes the hierarchical structure of German legal documents.

```text
Gesetz
  │
  └── Paragraph (§)
        │
        └── Absatz
              │
              └── Nummer
```

For example:

```text
§ 573c Fristen der ordentlichen Kündigung

(1) Die Kündigung ist spätestens am dritten Werktag ...

(2) Bei Wohnraum, der nur zum vorübergehenden Gebrauch ...

(3) Bei Wohnraum nach § 549 Abs. 2 Nr. 2 ...
```

is represented using structured metadata such as:

```text
paragraph = 573c
absatz    = 1
nummer    = -
```

or:

```text
paragraph = 573
absatz    = 3
nummer    = 2
```

depending on the source structure.

---

# 10. Legal Number Detection

A key parser requirement is distinguishing legal list numbers from ordinary numbers.

For example:

```text
§ 549 Abs. 2 Nr. 2
```

contains legal references.

However:

```text
am 15. eines Monats
```

contains a date-related number and must not accidentally be interpreted as:

```text
Nummer = 15
```

The parser therefore recognizes numbered legal items only when they occur as standalone numbered list elements.

This prevents incorrect chunk boundaries.

---

# 11. Document Enrichment

During ingestion, each legal chunk is enriched by an LLM.

The enrichment information includes fields such as:

```text
THEMA
KEYWORDS
TYPISCHE NUTZERFRAGEN
KLARTEXT
```

The original legal text remains explicitly available.

Conceptually:

```text
Original Legal Text
        │
        ▼
LLM Enrichment
        │
        ├── Topic
        ├── Keywords
        ├── Typical user questions
        └── Plain-language explanation
        │
        ▼
Structured RAG Document
```

The goal is to improve retrieval and interpretation without replacing the original source text.

---

# 12. Document Metadata

Each Chroma document contains structured metadata.

Typical fields include:

```text
gesetz
rechtsgebiet
thema
paragraph
paragraph_titel
absatz
nummer
source
file
content_hash
chunk_type
```

Example:

```python
{
    "gesetz": "BGB",
    "rechtsgebiet": "Mietrecht",
    "thema": "...",
    "paragraph": "573c",
    "paragraph_titel": "Fristen der ordentlichen Kündigung",
    "absatz": "1",
    "nummer": "-",
    "source": "BGB.xml",
    "file": "BGB.txt",
    "content_hash": "...",
    "chunk_type": "absatz",
}
```

This metadata enables targeted filtering during retrieval.

---

# 13. Vector Database

The current vector database is:

```text
ChromaDB
```

with a persistent local directory:

```text
chroma_legal_rag/
```

The application loads the database through:

```python
Chroma(
    persist_directory=str(PERSIST_DIRECTORY),
    embedding_function=embeddings,
)
```

The configuration also contains the conceptual names for a future Pinecone setup:

```yaml
vector_DB:
  vector_store: 1
  legal_index: German_Law
  namespace_Mietrecht: Mietrecht
  namespace_Steuerrecht: Steuerrecht
```

The current implementation uses:

```text
vector_store = 1
```

which means ChromaDB.

---

# 14. LLM Architecture

The project separates LLM responsibilities.

The current configuration is:

```yaml
llm:
  llm_enrichment: gpt-5.4-nano
  llm_query: gpt-5.4-nano
  llm_answer: gpt-5.4-mini
```

## Enrichment LLM

Used during document ingestion to create structured enrichment information.

## Query LLM

Used for:

- query processing
- routing
- relevance evaluation
- query rewriting

## Answer LLM

Used for final answer synthesis.

This separation allows different models and temperatures to be used for different tasks.

---

# 15. Network and Corporate Proxy Support

Network handling is centralized in:

```text
src/legal_system_rag/network/client_factory.py
src/legal_system_rag/config/settings.py
```

The system supports:

- HTTPX
- HTTP/HTTPS proxy configuration
- PX proxy
- configurable timeouts
- configurable retries
- custom CA certificates
- SSL verification configuration

Example configuration:

```yaml
network:
  px_host: "127.0.0.1"
  px_port: 3128
  timeout: 60.0
  retries: 3
  ignore_ssl: False
```

The application can also use:

```text
COMPANY_PROXY_URL
```

from the environment.

---

# 16. Certificate Handling

The configuration module can build a combined CA bundle from:

```text
certifi
+
company proxy CA
+
company root CA
```

The generated bundle is:

```text
certs/Company_Internet_Kombi.crt
```

If the required company certificates are unavailable, the implementation falls back to the standard `certifi` CA bundle and adjusts SSL behavior according to the configured environment.

This functionality is primarily intended for corporate development environments.

---

# 17. Streamlit Application

The Streamlit entry point is:

```text
src/legal_system_rag/app.py
```

The application provides:

- Chat interface
- Example questions
- Pipeline selection
- Retriever selection
- Conversation history
- Retrieved source display
- Error handling
- Cached resources

The UI exposes the retrieval architecture rather than hiding it completely.

This makes the application useful for demonstrating and debugging the RAG pipeline.

---

# 18. Application Resource Lifecycle

The application caches expensive resources using Streamlit resource caching.

The initialization flow is approximately:

```text
Streamlit
    │
    ▼
get_proxy_url()
    │
    ▼
HTTPX Client
    │
    ├── OpenAI Embeddings
    ├── Query LLM
    ├── Answer LLM
    └── ChromaDB
             │
             ▼
       Global BM25 Index
             │
             ▼
         RAG Chain
```

The global BM25 index is created once for the loaded Chroma collection and reused during the application session.

---

# 19. Project Structure

The relevant source structure is:

```text
legal-system-rag/
│
├── src/
│   ├── __init__.py
│   │
│   └── legal_system_rag/
│       ├── __init__.py
│       ├── app.py
│       ├── ingest_documents.py
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   ├── settings.py
│       │   └── rag_config.yaml
│       │
│       ├── network/
│       │   ├── __init__.py
│       │   ├── client_factory.py
│       │   └── ReadMe_Certificate.md
│       │
│       ├── parser/
│       │   ├── __init__.py
│       │   └── legal_parser.py
│       │
│       ├── pipelines/
│       │   ├── __init__.py
│       │   └── ingest.py
│       │
│       └── rag/
│           ├── __init__.py
│           ├── agent.py
│           ├── agentic_rag.py
│           ├── chain.py
│           ├── evaluators.py
│           ├── prompts.py
│           ├── reranker.py
│           ├── retrieval.py
│           ├── router.py
│           └── utils.py
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_EmptyRetriever.py
│   │   ├── test_legal_parser.py
│   │   └── test_router.py
│   │
│   ├── integration/
│   │   └── test_network.py
│   │
│   ├── e2e/
│   ├── compare_chroma.py
│   ├── debug_chroma.py
│   └── inspect_chroma.py
│
├── data/
│   └── dokumente_mietrecht/
│
├── chroma_legal_rag/
│
├── certs/
│
├── activate.ps1
├── createDB.ps1
├── start.ps1
├── .env.example
├── .gitignore
├── pyproject.toml
├── Quick_Start.md
├── README.md
└── runtime.txt
```

Generated directories such as:

```text
.venv/
__pycache__/
.ipynb_checkpoints/
```

are development artifacts and are not part of the conceptual architecture.

---

# 20. Important Modules

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit UI and application orchestration |
| `ingest_documents.py` | CLI entry point for ingestion |
| `pipelines/ingest.py` | Document ingestion pipeline |
| `parser/legal_parser.py` | Legal-aware document parsing |
| `rag/router.py` | LLM-based retrieval routing |
| `rag/retrieval.py` | Routed dense + sparse retrieval |
| `rag/reranker.py` | Cross-encoder reranking |
| `rag/evaluators.py` | Relevance evaluation and query rewriting |
| `rag/agent.py` | Agentic retrieval loop |
| `rag/agentic_rag.py` | End-to-end agentic RAG pipeline |
| `rag/chain.py` | Standard RAG chain and retrieval utilities |
| `rag/prompts.py` | Query and answer prompts |
| `rag/utils.py` | Retrieval and document utilities |
| `config/settings.py` | Central configuration and environment handling |
| `network/client_factory.py` | HTTPX / proxy / SSL handling |

---

# 21. Configuration

The main RAG configuration is:

```text
src/legal_system_rag/config/rag_config.yaml
```

Current relevant settings:

```yaml
llm:
  llm_enrichment: gpt-5.4-nano
  llm_query: gpt-5.4-nano
  llm_answer: gpt-5.4-mini

embedding:
  embedding_model: text-embedding-3-small

vector_DB:
  vector_store: 1
  legal_index: German_Law
  namespace_Mietrecht: Mietrecht
  namespace_Steuerrecht: Steuerrecht

retriever:
  retriever_mode: 2
  weight_hybrid: 0.4
  top_k: 8
  top_bm25_k: 30

chunking:
  size: 1000
  overlap: 200
  batch_size: 20
```

The current retriever configuration means:

```text
retriever_mode = 2
weight_hybrid = 0.4
top_k = 8
top_bm25_k = 30
```

The reranker weight is derived as:

```python
WEIGHT_RERANKER = 1.0 - WEIGHT_HYBRID
```

so the current configuration corresponds to:

```text
Hybrid score:    0.4
Reranker score:  0.6
```

---

# 22. Environment Variables

Sensitive values are read from environment variables or Streamlit secrets.

Typical variables include:

```text
OPENAI_API_KEY
HF_TOKEN
PINECONE_API_KEY
PINECONE_API_KEY2
COMPANY_PROXY_URL
```

Create a local environment file from:

```text
.env.example
```

Do not commit real credentials.

---

# 23. Installation

The project requires:

```text
Python >= 3.12
```

Create and activate the virtual environment:

```powershell
.\activate.ps1
```

Install the package:

```powershell
pip install -e ".[dev]"
```

For local FlashRank experimentation:

```powershell
pip install -e ".[local-rerank]"
```

For development with both optional packages:

```powershell
pip install -e ".[dev,local-rerank,pdf]"
```

The package is configured through `pyproject.toml`.

---

# 24. Quick Start

For Windows PowerShell:

```powershell
.\activate.ps1
.\createDB.ps1
.\start.ps1
```

The scripts have clearly separated responsibilities.

### `activate.ps1`

Creates/checks and activates the Python virtual environment.

### `createDB.ps1`

Creates the ChromaDB database if it does not already exist.

### `start.ps1`

Starts the Streamlit application.

---

# 25. Create the Vector Database

The ingestion entry point is:

```powershell
legal-ingest
```

or:

```powershell
python -m legal_system_rag.ingest_documents
```

The ingestion process reads:

```text
data/dokumente_mietrecht/
```

and writes the persistent vector database to:

```text
chroma_legal_rag/
```

The ingestion process:

1. Loads source documents
2. Parses legal structure
3. Creates legal chunks
4. Enriches chunks
5. Generates embeddings
6. Stores documents in ChromaDB
7. Uses deterministic document IDs
8. Processes documents in batches

---

# 26. Start the Application

The Streamlit application can be started with:

```powershell
.\start.ps1
```

or:

```powershell
streamlit run src/legal_system_rag/app.py
```

The package also defines:

```text
legal-app
```

as a console entry point.

After startup, Streamlit normally provides a local address such as:

```text
http://localhost:8501
```

---

# 27. Testing

The project uses Pytest.

Run the complete test suite:

```powershell
python -m pytest
```

The configured Pytest options include:

```text
-v
--import-mode=importlib
--cov=src
--cov-branch
--cov-report=term-missing
--cov-report=html
```

The project defines the following markers:

```text
unit
integration
e2e
proxy
```

Examples:

```powershell
pytest -m unit
```

```powershell
pytest -m integration
```

```powershell
pytest -m e2e
```

Tests requiring a local corporate PX proxy can be separated using:

```powershell
pytest -m proxy
```

---

# 28. Test Structure

Current tests include:

```text
tests/
├── unit/
│   ├── test_EmptyRetriever.py
│   ├── test_legal_parser.py
│   └── test_router.py
│
├── integration/
│   └── test_network.py
│
└── e2e/
```

The unit tests cover isolated components such as:

- Legal parser behavior
- Router decisions
- Empty retriever behavior

Integration tests cover external infrastructure such as network access.

The project is configured for branch coverage reporting.

---

# 29. Dependency Overview

The core stack includes:

| Technology | Purpose |
|---|---|
| Python 3.12+ | Application language |
| LangChain | RAG orchestration |
| LangChain Core | Runnable and retriever abstractions |
| LangChain OpenAI | OpenAI model integration |
| ChromaDB | Persistent vector database |
| BM25 | Sparse lexical retrieval |
| OpenAI Embeddings | Dense vector representation |
| OpenAI Chat Models | Query processing and answer generation |
| Hugging Face | Cross-encoder inference |
| HTTPX | Network client |
| Pydantic | Structured LLM output and data validation |
| PyYAML | Configuration |
| python-dotenv | Environment configuration |
| Streamlit | User interface |
| Pytest | Testing |
| Tenacity | Retry support |

---

# 30. Design Principles

The implementation follows several software-engineering principles.

## Separation of Concerns

Different responsibilities are isolated:

```text
Configuration
     │
     ├── Network
     │
     ├── Parser
     │
     ├── Ingestion
     │
     ├── Retrieval
     │
     ├── Reranking
     │
     ├── Evaluation
     │
     ├── Generation
     │
     └── UI
```

## Dependency Injection

Models, vector stores, retrievers, and clients are passed into functions rather than being recreated throughout the application.

For example, the agentic loop receives:

```python
question
llm_query
vector_store
global_bm25
```

This makes the components easier to test and replace.

## Configuration-Driven Behavior

Important parameters are stored in YAML rather than hard-coded throughout the application.

## Typed Structured Output

The router uses Pydantic:

```python
class RouterDecision(BaseModel):
    ...
```

This gives the LLM routing decision a defined schema.

## Resource Reuse

Expensive resources such as:

- HTTP clients
- embeddings
- LLM instances
- ChromaDB
- global BM25 index

are reused through Streamlit resource caching.

---

# 31. RAG Architecture at a Glance

The complete system can be summarized as:

```text
                         ┌─────────────────┐
                         │  Legal Sources  │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Legal Parser    │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ LLM Enrichment  │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   Embeddings    │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    ChromaDB     │
                         └────────┬────────┘
                                  │
                                  │
              ┌───────────────────┘
              │
              ▼
        ┌─────────────┐
        │ User Query  │
        └──────┬──────┘
               │
               ▼
        ┌───────────────┐
        │  LLM Router   │
        └──────┬────────┘
               │
               ▼
        Search Query
        + Filters
               │
        ┌──────┴──────┐
        ▼             ▼
     Chroma          BM25
     Dense           Sparse
        │             │
        └──────┬──────┘
               ▼
       Hybrid Candidates
               │
               ▼
       Cross-Encoder
         Reranking
               │
               ▼
             Top-K
               │
               ▼
       Relevance Check
               │
          ┌────┴────┐
          │         │
         Yes        No
          │         │
          │      Query Rewrite
          │         │
          │         └──────► Retry
          │
          ▼
       Answer LLM
          │
          ▼
    Grounded Answer
          │
          ▼
   Sources + Metadata
```

---

# 32. Retrieval Strategy in More Detail

The current agentic retrieval strategy deliberately uses multiple retrieval stages.

```text
Stage 1
───────
LLM Router
    │
    ▼
Search Strategy
```

```text
Stage 2
───────
Dense + Sparse Retrieval
    │
    ├── ChromaDB
    └── BM25
    │
    ▼
Large Candidate Set
```

```text
Stage 3
───────
Cross-Encoder Reranking
    │
    ▼
High-quality Top-K
```

```text
Stage 4
───────
LLM Relevance Evaluation
    │
    ├── Relevant → answer
    └── Not relevant → rewrite and retry
```

This multi-stage design separates:

```text
Recall
  ↓
Ranking
  ↓
Evaluation
  ↓
Generation
```

rather than expecting one retrieval mechanism to perform all tasks.

---

# 33. Why Hybrid Retrieval?

Dense and sparse retrieval have different strengths.

### Dense Retrieval

Strong for:

- semantic similarity
- paraphrases
- different wording
- concept-level questions

### BM25

Strong for:

- exact terminology
- legal identifiers
- names
- specific words
- lexical matches

Combining them improves retrieval robustness.

```text
                 Query
                   │
          ┌────────┴────────┐
          ▼                 ▼
       Semantic          Lexical
       Retrieval         Retrieval
          │                 │
          └────────┬────────┘
                   ▼
             Hybrid Ranking
```

---

# 34. Why Reranking?

Initial retrieval is optimized for finding a sufficiently broad candidate set.

The reranker is optimized for deciding which candidates are most relevant to the exact query.

Therefore:

```text
Retriever
   =
high recall

Reranker
   =
high precision
```

The architecture uses:

```text
Retrieve many
      ↓
Rerank candidates
      ↓
Keep Top-K
```

This is especially useful for legal questions where several chunks may concern similar concepts but only a subset directly answers the question.

---

# 35. Why Agentic Retrieval?

The standard RAG mode follows a fixed pipeline.

The agentic mode allows the retrieval process to react to its own intermediate result.

```text
Fixed RAG:

Question
  ↓
Retrieve
  ↓
Answer
```

versus:

```text
Agentic RAG:

Question
  ↓
Route
  ↓
Retrieve
  ↓
Rerank
  ↓
Evaluate
  ↓
 ┌───────────────┐
 │ Is retrieval  │
 │ sufficient?   │
 └───────┬───────┘
         │
    ┌────┴────┐
    ▼         ▼
   Yes        No
    │         │
    │      Rewrite
    │         │
    │      Retrieve
    │         │
    └────┬────┘
         ▼
       Answer
```

This makes retrieval adaptive rather than completely static.

---

# 36. Limitations

This project is intentionally a technical demonstration.

It is **not** a production legal service and does not provide legal advice.

Important limitations include:

- The legal corpus is curated and limited.
- The system does not represent the complete German legal system.
- Retrieval quality depends on the indexed documents.
- LLM-generated enrichment can contain errors.
- LLM-based relevance evaluation can itself be imperfect.
- Cross-encoder inference currently uses an external Hugging Face service.
- The application does not replace professional legal review.
- No claim is made that generated answers are legally complete or authoritative.
- Production deployment would require substantially more evaluation, monitoring, security, access control, and corpus management.

---

# 37. Security and Secrets

Never commit:

```text
.env
```

or other files containing:

```text
OPENAI_API_KEY
HF_TOKEN
PINECONE_API_KEY
```

Use:

```text
.env.example
```

as the template for local configuration.

The local Chroma database should normally remain outside Git:

```text
chroma_legal_rag/
```

Likewise, local certificates and development artifacts should be handled according to the target environment's security policy.

---

# 38. Project Status

Current capabilities:

- [x] Modular Python package
- [x] Legal document parser
- [x] Paragraph extraction
- [x] Absatz extraction
- [x] Legal number detection
- [x] Legal reference extraction
- [x] Structured metadata
- [x] LLM document enrichment
- [x] Batch ingestion
- [x] OpenAI embeddings
- [x] Persistent ChromaDB
- [x] Dense retrieval
- [x] Sparse BM25 retrieval
- [x] Hybrid retrieval
- [x] LLM query router
- [x] Paragraph-aware routing
- [x] Cross-encoder reranking
- [x] Score-aware retrieval support
- [x] Relevance evaluation
- [x] Query rewriting
- [x] Agentic/self-corrective retrieval loop
- [x] Grounded answer generation
- [x] Source document display
- [x] Streamlit chat application
- [x] Configurable model selection
- [x] Proxy/PX support
- [x] Custom certificate handling
- [x] HTTPX client handling
- [x] Pytest unit tests
- [x] Integration tests
- [x] E2E test structure
- [x] Coverage reporting
- [x] Python packaging
- [x] CLI entry points

---

# 39. Future Extensions

Potential future extensions include:

```text
Multi-source retrieval
        │
        ├── German law
        ├── Legal commentaries
        ├── Legal reference material
        └── Web search
                │
                ▼
          Source Routing
                │
                ▼
          Evidence Fusion
```

Other possible extensions:

- Knowledge Graph / GraphRAG
- Multi-query retrieval
- HyDE
- Query expansion
- More advanced reranking
- Retrieval evaluation datasets
- Automated RAG evaluation
- Citation validation
- Confidence estimation
- Long-term conversation memory
- Additional legal domains
- Pinecone-based deployment
- Production observability
- Retrieval and generation tracing
- Guardrails and policy enforcement

These are possible architectural extensions rather than requirements of the current implementation.

---

# 40. Development Philosophy

The project is intentionally implemented as a real Python software system rather than as a single RAG notebook.

The architecture separates:

```text
                 Software Architecture
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
     Ingestion        Retrieval        UI
        │                │                │
        ▼                ▼                ▼
      Parser          Reranker       Streamlit
        │                │
        ▼                ▼
     ChromaDB       Evaluation
                         │
                         ▼
                    Generation
```

This makes individual components replaceable.

For example:

- ChromaDB can be replaced by another vector store.
- BM25 can be replaced or supplemented by another sparse retriever.
- The reranker can be replaced by another cross-encoder.
- The LLM can be changed independently of the parser.
- The router can evolve without redesigning the UI.
- The network layer can be adapted to another environment.
- Unit tests can validate individual components independently.

---

# 41. Example End-to-End Flow

For a question such as:

```text
"Welche Kündigungsfrist gilt für einen Mieter?"
```

the agentic system can conceptually execute:

```text
1. User Question
       │
       ▼
2. LLM Router
       │
       └── concept_search
       │
       ▼
3. Search Query
       │
       ▼
4. ChromaDB + BM25
       │
       ▼
5. Candidate Documents
       │
       ▼
6. Cross-Encoder Reranking
       │
       ▼
7. Top-K Selection
       │
       ▼
8. LLM Relevance Evaluation
       │
       ├── relevant
       │
       ▼
9. Answer LLM
       │
       ▼
10. Grounded Answer
       │
       ▼
11. Retrieved Sources
```

If the relevance evaluation determines that the retrieved documents are insufficient:

```text
Relevance = no
      │
      ▼
Query Rewrite
      │
      ▼
New Retrieval
      │
      ▼
Reranking
      │
      ▼
Relevance Evaluation
      │
      ▼
Final Answer
```

---

# 42. Command Reference

### Environment

```powershell
.\activate.ps1
```

### Install

```powershell
pip install -e ".[dev]"
```

### Ingest documents

```powershell
legal-ingest
```

or:

```powershell
python -m legal_system_rag.ingest_documents
```

### Run tests

```powershell
python -m pytest
```

### Start application

```powershell
legal-app
```

or:

```powershell
streamlit run src/legal_system_rag/app.py
```

### Complete Windows startup

```powershell
.\activate.ps1
.\createDB.ps1
.\start.ps1
```

---

# 43. Repository

GitHub:

```text
https://github.com/gfwu61/legal-system-rag-ChromaDB
```

---

# 44. License

This project is licensed under the MIT License.

See:

```text
LICENSE
```

---

# 45. Summary

This project implements a modular, end-to-end RAG system for German tenancy law.

Its architecture combines:

```text
Legal-aware Parsing
        +
LLM Enrichment
        +
OpenAI Embeddings
        +
ChromaDB
        +
BM25
        +
Hybrid Retrieval
        +
Cross-Encoder Reranking
        +
LLM Routing
        +
Relevance Evaluation
        +
Query Rewriting
        +
Agentic Retrieval
        +
Grounded Generation
        +
Source Transparency
        +
Automated Testing
        +
Streamlit
```

The resulting system demonstrates the evolution from a conventional linear RAG pipeline toward a more adaptive, self-corrective RAG architecture while keeping the individual components modular and testable.

The project is intended to demonstrate practical RAG engineering capabilities, software architecture, retrieval techniques, LLM orchestration, and integration of AI components into a structured Python application.
