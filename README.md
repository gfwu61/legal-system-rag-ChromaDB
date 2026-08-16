
# Legal RAG System

A professional Retrieval-Augmented Generation (RAG) system for answering questions about German tenancy law (Mietrecht).

The system combines:

- Python
- LangChain
- OpenAI LLMs
- OpenAI Embeddings
- ChromaDB
- Streamlit
- HTTPX
- pytest
- Corporate proxy / PX support

The application retrieves relevant legal documents from a persistent ChromaDB vector store and uses large language models to generate answers grounded in the retrieved legal content.

---

# 1. Project Overview

An end-to-end RAG pipeline demonstrating professional AI engineering practices using a curated set of German tenancy law texts

The system follows this general architecture:

```text
                    ┌─────────────────────┐
                    │      User Query     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Query Extraction  │
                    │      / Enrichment   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Vector Retrieval  │
                    │      ChromaDB       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Retrieved Legal   │
                    │     Documents       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Answer LLM      │
                    │   Grounded Answer   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Streamlit UI        │
                    │ Answer + Sources    │
                    └─────────────────────┘
````

The application is designed so that document ingestion, parsing, retrieval, generation, networking, and user interface are separated into independent modules.

---

# 2. Main Features

## RAG

The system uses Retrieval-Augmented Generation to combine:

1. User question
2. Query extraction
3. Vector search
4. Legal document retrieval
5. LLM-based answer generation

The generated answer is based on retrieved legal documents rather than relying exclusively on the LLM's internal knowledge.

## Legal Document Parsing

Legal documents are parsed into structured components such as:

* Paragraph
* Absatz
* Nummer
* References
* Original legal text

For example:

```text
§ 573c Fristen der ordentlichen Kündigung

(1) Die Kündigung ist spätestens am dritten Werktag ...
(2) Bei Wohnraum, der nur zum vorübergehenden Gebrauch ...
(3) Bei Wohnraum nach § 549 Abs. 2 Nr. 2 ...
(4) Eine zum Nachteil des Mieters ...
```

The parser correctly distinguishes legal numbers from references and ordinary numbers.

For example:

```text
§ 549 Abs. 2 Nr. 2
```

contains legal references, while:

```text
am 15. eines Monats
```

contains a date-related number that must not incorrectly become a legal `Nummer`.

---

# 3. Project Structure

```text
legal-system-rag/
│
├── src/
│   ├── __init__.py
│   │
│   └── legal_system_rag/
│       │
│       ├── __init__.py
│       ├── app.py
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       │
│       ├── network/
│       │   ├── __init__.py
│       │   └── client_factory.py
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
│           ├── chain.py
│           └── prompts.py
│
├── tests/
│   ├── unit/
│   │   └── test_legal_parser.py
│   │
│   └── integration/
│       └── test_network.py
│
├── config/
│   └── rag_config.yaml
│
├── data/
│   └── dokumente_mietrecht/
│
├── chroma_legal_rag/
│
├── certs/
│
├── .env
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── runtime.txt
```

---

# 4. Application: `app.py`

`app.py` is the Streamlit entry point of the application.

Its responsibility is to initialize the RAG system and provide the interactive user interface.

## Application Pipeline

```text
Streamlit starts
       │
       ▼
Configure Streamlit
       │
       ▼
Determine proxy configuration
       │
       ▼
Create HTTPX client
       │
       ▼
Initialize OpenAI embeddings
       │
       ▼
Initialize query LLM
       │
       ▼
Initialize answer LLM
       │
       ▼
Open persistent ChromaDB
       │
       ▼
Build RAG chain
       │
       ▼
Display chat history
       │
       ▼
Receive user question
       │
       ▼
Invoke RAG chain
       │
       ├──────────────► Query processing
       │
       ├──────────────► Vector retrieval
       │
       └──────────────► Answer generation
       │
       ▼
Display answer
       │
       ▼
Display retrieved sources
       │
       ▼
Store conversation in session state
```

---

# 5. `app.py` Function Descriptions

## `load_http_client()`

```text
Function description:
Creates and caches a synchronous HTTPX client.

Pipeline:

get_proxy_url()
        ↓
create_http_client()
        ↓
HTTPX client
        ↓
Streamlit resource cache
```

The client is reused between Streamlit reruns.

---

## `load_resources()`

```text
Function description:
Initializes the OpenAI embedding model, query LLM,
answer LLM and persistent ChromaDB vector store.

Pipeline:

HTTPX client
     │
     ├──► OpenAIEmbeddings
     │
     ├──► Query ChatOpenAI
     │
     ├──► Answer ChatOpenAI
     │
     └──► ChromaDB
```

The function also verifies that the configured ChromaDB directory exists and is not empty.

---

## `load_rag_chain()`

```text
Function description:
Builds and caches the complete RAG chain.

Pipeline:

Query LLM
     │
Answer LLM
     │
Vector Store
     │
     ▼
build_rag_chain()
     │
     ▼
RAG Chain
```

---

## `render_sources()`

```text
Function description:
Displays the legal documents retrieved by the RAG system.

Pipeline:

Retrieved Documents
        │
        ▼
Read metadata
        │
        ├──► Paragraph
        └──► Source file
        │
        ▼
Display document content
```

The sources are displayed in a Streamlit expander.

---

## `render_chat_history()`

```text
Function description:
Displays the previous conversation stored in Streamlit session state.

Pipeline:

st.session_state.messages
        │
        ▼
Iterate messages
        │
        ├──► User message
        │
        └──► Assistant message
                    │
                    └──► Retrieved sources
```

---

## `initialize_resources()`

```text
Function description:
Initializes all resources required by the RAG application.

Pipeline:

get_proxy_url()
       │
       ▼
load_http_client()
       │
       ▼
load_resources()
       │
       ▼
load_rag_chain()
       │
       ▼
Ready-to-use RAG chain
```

---

## `main()`

```text
Function description:
Runs the complete Streamlit application.

Pipeline:

Start application
       │
       ▼
Initialize session state
       │
       ▼
Initialize RAG resources
       │
       ▼
Render chat history
       │
       ▼
Wait for user input
       │
       ▼
User question
       │
       ▼
rag_chain.invoke()
       │
       ▼
Generated answer
       │
       ├──► Display answer
       │
       └──► Display sources
       │
       ▼
Store conversation
```

---

# 6. RAG Pipeline

The core RAG pipeline is implemented in:

```text
src/legal_system_rag/rag/chain.py
```

The pipeline conceptually consists of the following stages:

```text
User Question
      │
      ▼
Query Extraction
      │
      ├── search query
      └── paragraph filter
      │
      ▼
Retriever
      │
      ▼
ChromaDB
      │
      ▼
Relevant Legal Documents
      │
      ▼
Answer Prompt
      │
      ▼
Answer LLM
      │
      ▼
Final Answer
```

The system returns both:

```python
{
    "answer": "...",
    "docs": [...]
}
```

The `answer` contains the generated response.

The `docs` contain the retrieved source documents used by the application.

---

# 7. Document Ingestion Pipeline

Legal source documents are processed before they become available to the RAG application.

The ingestion pipeline is implemented in:

```text
src/legal_system_rag/pipelines/ingest.py
```

The general pipeline is:

```text
Legal TXT documents
        │
        ▼
Text loading
        │
        ▼
Legal parsing
        │
        ▼
Paragraph extraction
        │
        ▼
Number extraction
        │
        ▼
Reference extraction
        │
        ▼
Metadata generation
        │
        ▼
Text enrichment
        │
        ▼
Chunking
        │
        ▼
OpenAI Embeddings
        │
        ▼
ChromaDB
```

---

# 8. Legal Parser

The parser is implemented in:

```text
src/legal_system_rag/parser/legal_parser.py
```

The parser provides functions such as:

```python
normalize_text()
split_paragraphs()
split_absaetze()
split_nummern()
extract_references()
```

The parser separates the structure of German legal text.

Example:

```text
§ 573c

(3) Bei Wohnraum nach § 549 Abs. 2 Nr. 2
ist die Kündigung spätestens am 15. eines Monats
zum Ablauf dieses Monats zulässig.
```

is represented approximately as:

```text
paragraph = 573c
absatz   = 3
```

The legal reference:

```text
§ 549 Abs. 2 Nr. 2
```

is preserved as a reference.

The number:

```text
15
```

is not incorrectly interpreted as legal `Nummer 15`.

---

# 9. ChromaDB

The application uses ChromaDB as its persistent vector store.

Current database:

```text
chroma_legal_rag/
```

Collection:

```text
langchain
```

Example metadata:

```python
{
    "source": "Mietrecht_kuendigung.txt",
    "paragraph": "573c",
    "absatz": "3",
    "nummer": "none",
    "original_text": "(3) Bei Wohnraum nach § 549 Abs. 2 Nr. 2 ..."
}
```

The database is persistent and therefore does not need to be recreated every time the Streamlit application starts.

---

# 10. Example Retrieved Document

For example, a query such as:

```text
Bis wann kann bei Wohnraum nach § 549 Abs. 2 Nr. 2 gekündigt werden?
```

can retrieve:

```text
§ 573c Abs. 3 BGB

(3) Bei Wohnraum nach § 549 Abs. 2 Nr. 2 ist die Kündigung
spätestens am 15. eines Monats zum Ablauf dieses Monats zulässig.
```

The generated answer can then be grounded in this retrieved source.

---

# 11. Networking and Proxy

The networking layer is implemented in:

```text
src/legal_system_rag/network/client_factory.py
```

The system supports:

1. Local PX proxy
2. Corporate proxy
3. Direct connection

The proxy selection logic is:

```text
Is PX running?
      │
      ├── YES ──► Use PX proxy
      │
      └── NO
           │
           ▼
      Corporate proxy configured?
           │
           ├── YES ──► Use corporate proxy
           │
           └── NO ──► Direct connection
```

The functions include:

```python
check_if_px_is_running()
create_ssl_context()
get_proxy_url()
create_http_client()
create_async_http_client()
```

The HTTPX clients use:

```python
trust_env=False
```

to prevent unexpected proxy/environment configuration from interfering with the explicitly configured networking setup.

---

# 12. SSL Configuration

The application uses a configured CA certificate bundle.

The SSL context is created by:

```python
create_ssl_context()
```

Normal operation:

```text
Certificate verification = enabled
Hostname verification    = enabled
```

For testing only:

```python
create_ssl_context(ignore_ssl=True)
```

disables certificate verification.

This mode should not be used in production.

---

# 13. Streamlit Application

The user interface is implemented in:

```text
src/legal_system_rag/app.py
```

The application provides:

* Chat interface
* User questions
* Generated answers
* Retrieved legal sources
* Conversation history
* Error handling
* Cached resources

Streamlit resources are cached using:

```python
@st.cache_resource
```

This prevents expensive resources such as:

* HTTP clients
* LLMs
* Embeddings
* ChromaDB
* RAG chain

from being recreated unnecessarily on every Streamlit rerun.

---

# 14. Configuration

Configuration is centralized in:

```text
src/legal_system_rag/config/settings.py
```

Important configuration values include:

```text
EMBEDDING_MODEL
LLM_QUERY_MODEL
LLM_ANSWER_MODEL
PERSIST_DIRECTORY
CERT_FILE
COMPANY_PROXY_URL
PX_HOST
PX_PORT
RETRIES
TIMEOUT
```

Additional RAG configuration is stored in:

```text
config/rag_config.yaml
```

Sensitive configuration such as API keys belongs in:

```text
.env
```

The `.env` file must not be committed to Git.

---

# 15. Environment Variables

Create a local `.env` file based on:

```text
.env.example
```

Example:

```text
OPENAI_API_KEY=your_api_key
```

Additional corporate network configuration may be required depending on the environment.

Never commit the real API key to GitHub.

---

# 16. Installation

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project:

```powershell
pip install -e .
```

Install development dependencies:

```powershell
pip install -e ".[dev]"
```

---

# 17. Build the Python Package

The project uses `pyproject.toml` for packaging.

Build the package with:

```powershell
python -m build
```

The generated files are placed in:

```text
dist/
```

Typically:

```text
dist/
├── legal_system_rag-0.1.0-py3-none-any.whl
└── legal_system_rag-0.1.0.tar.gz
```

---

# 18. Running the Tests

Run the complete test suite:

```powershell
python -m pytest -v
```

Run only unit tests:

```powershell
python -m pytest -v tests/unit/
```

Run only integration tests:

```powershell
python -m pytest -v tests/integration/
```

Run tests marked as integration tests:

```powershell
python -m pytest -v -m integration
```

---

# 19. Test Architecture

The project separates tests into:

```text
tests/
├── unit/
│   └── test_legal_parser.py
│
└── integration/
    └── test_network.py
```

## Unit Tests

Unit tests verify individual functions without requiring the complete system.

Example:

```text
test_573c_15_is_not_a_legal_number
test_573_real_legal_numbers_are_detected
```

These tests verify the legal parser behavior.

## Integration Tests

Integration tests verify interaction with external infrastructure such as:

* PX proxy
* SSL configuration
* HTTPX
* Network connectivity

The network tests include:

```text
test_px
test_ssl_default
test_ssl_ignore
test_network_client
test_async_network_client
```

---

# 20. Code Coverage

Coverage can be generated with:

```powershell
python -m pytest -v --cov=legal_system_rag --cov-report=html
```

The HTML report is generated in:

```text
htmlcov/
```

Open:

```text
htmlcov/index.html
```

Coverage should be interpreted per module.

A low overall percentage does not necessarily mean that the tested functionality is poorly tested. For example, Streamlit application code and full RAG pipelines may not be executed by unit tests.

---

# 21. Running the Application

From the project root:

```powershell
streamlit run src/legal_system_rag/app.py
```

Streamlit starts the application and displays a local URL.

Typically:

```text
Local URL: http://localhost:8501
```

Open the URL in a browser.

---

# 22. Application Startup Pipeline

When the application starts:

```text
streamlit run src/legal_system_rag/app.py
             │
             ▼
          main()
             │
             ▼
      initialize_resources()
             │
             ▼
       get_proxy_url()
             │
             ▼
      load_http_client()
             │
             ▼
        load_resources()
             │
             ├── OpenAI Embeddings
             ├── Query LLM
             ├── Answer LLM
             └── ChromaDB
             │
             ▼
       load_rag_chain()
             │
             ▼
       RAG application ready
```

---

# 23. User Query Pipeline

When the user enters a question:

```text
User
 │
 ▼
Streamlit chat_input()
 │
 ▼
RAG Chain
 │
 ▼
Query Extraction
 │
 ▼
Retriever
 │
 ▼
ChromaDB similarity search
 │
 ▼
Relevant legal documents
 │
 ▼
Answer prompt
 │
 ▼
Answer LLM
 │
 ▼
Generated answer
 │
 ├──► Display answer
 │
 └──► Display sources
```

---

# 24. Error Handling

The application handles initialization errors separately from query-processing errors.

Initialization errors:

```python
FileNotFoundError
```

are shown as warnings when the ChromaDB directory is missing or empty.

Other initialization errors are reported as:

```text
Initialization failed
```

Query errors are reported as:

```text
Error during processing
```

The error type and message are displayed to facilitate debugging.

---

# 25. Data Flow

The complete system can be viewed as two major pipelines.

## Offline Pipeline

```text
Legal Documents
      │
      ▼
Parser
      │
      ▼
Structured Legal Data
      │
      ▼
Chunking
      │
      ▼
Embeddings
      │
      ▼
ChromaDB
```

## Online Pipeline

```text
User Question
      │
      ▼
Query LLM
      │
      ▼
Search Query
      │
      ▼
ChromaDB
      │
      ▼
Retrieved Documents
      │
      ▼
Answer LLM
      │
      ▼
Answer + Sources
```

This separation is important because document ingestion does not need to be performed every time a user asks a question.

---

# 26. Technology Stack

| Component            | Technology           |
| -------------------- | -------------------- |
| Programming Language | Python 3.12+         |
| UI                   | Streamlit            |
| RAG Framework        | LangChain            |
| LLM                  | OpenAI               |
| Embeddings           | OpenAI Embeddings    |
| Vector Database      | ChromaDB             |
| HTTP Client          | HTTPX                |
| Configuration        | YAML / `.env`        |
| Testing              | pytest               |
| Coverage             | pytest-cov           |
| Packaging            | `pyproject.toml`     |
| Network Proxy        | PX / Corporate Proxy |

---

# 27. Design Principles

The project follows several software engineering principles.

## Separation of Concerns

Different responsibilities are separated:

```text
parser/
    Legal document parsing

network/
    HTTP and proxy configuration

rag/
    Retrieval and generation

pipelines/
    Document ingestion

config/
    Configuration

app.py
    User interface and application orchestration

tests/
    Verification
```

## Dependency Injection

The RAG chain receives its dependencies explicitly:

```python
build_rag_chain(
    llm_query,
    llm_answer,
    vector_store,
)
```

This makes the architecture easier to test and maintain.

## Resource Caching

Expensive resources are cached using:

```python
@st.cache_resource
```

This avoids unnecessary recreation of clients, models and vector stores.

---

# 28. Git

The project can be version-controlled using Git.

Initialize the repository:

```powershell
git init
```

Check the repository:

```powershell
git status
```

Add files:

```powershell
git add .
```

Create the first commit:

```powershell
git commit -m "Initial commit"
```

Add the GitHub remote:

```powershell
git remote add origin <YOUR_GITHUB_REPOSITORY>
```

Push:

```powershell
git branch -M main
git push -u origin main
```

---

# 29. Important Files That Must Not Be Committed

The following should normally be excluded from Git:

```text
.env
.venv/
chroma_legal_rag/
certs/*.crt
.streamlit/secrets.toml
__pycache__/
.pytest_cache/
htmlcov/
```

The exact exclusions are defined in:

```text
.gitignore
```

The OpenAI API key must never be committed.

---

# 30. Streamlit Cloud Deployment

The application can be deployed using Streamlit Cloud.

The GitHub repository should contain the source code and configuration files, but not secrets.

The OpenAI API key should be configured through Streamlit Cloud Secrets.

The deployment entry point is:

```text
src/legal_system_rag/app.py
```

The application also requires access to the required ChromaDB data.

For cloud deployment, the persistent local ChromaDB design may need to be adapted depending on how the vector database is hosted and persisted.

---

# 31. Current ChromaDB Requirement

The application expects:

```text
chroma_legal_rag/
```

to exist and contain the indexed documents.

At startup, `app.py` checks:

1. Does the directory exist?
2. Is the directory empty?

If either condition fails, the application stops and asks the user to index the documents first.

---

# 32. Example Question

Example user question:

```text
Bis wann kann bei Wohnraum nach § 549 Abs. 2 Nr. 2 gekündigt werden?
```

Expected retrieval:

```text
§ 573c Abs. 3 BGB
```

Relevant legal text:

```text
(3) Bei Wohnraum nach § 549 Abs. 2 Nr. 2 ist die Kündigung
spätestens am 15. eines Monats zum Ablauf dieses Monats zulässig.
```

The RAG system uses this retrieved document to generate the answer.

---

# 33. Project Status

The current implementation includes:

* [x] Legal document parser
* [x] Paragraph extraction
* [x] Absatz extraction
* [x] Legal number detection
* [x] Legal reference extraction
* [x] Document ingestion pipeline
* [x] OpenAI embeddings
* [x] Persistent ChromaDB
* [x] Query extraction
* [x] Vector retrieval
* [x] LLM answer generation
* [x] Source document display
* [x] Streamlit UI
* [x] PX proxy support
* [x] Corporate proxy support
* [x] SSL configuration
* [x] Synchronous HTTPX client
* [x] Asynchronous HTTPX client
* [x] Unit tests
* [x] Integration tests
* [x] pytest coverage
* [x] Python packaging
* [x] Git/GitHub support

---

# 34. Development Philosophy

The system is designed as a modular RAG application rather than as a single monolithic script.

The main architectural idea is:

```text
              Legal Documents
                    │
                    ▼
              Parsing Layer
                    │
                    ▼
             Ingestion Layer
                    │
                    ▼
               ChromaDB
                    │
                    ▼
             Retrieval Layer
                    │
                    ▼
            Generation Layer
                    │
                    ▼
             Streamlit App
```

This structure makes it possible to modify one layer without unnecessarily changing the others.

For example:

* The parser can be refactored without changing the Streamlit UI.
* The vector database can be replaced without redesigning the UI.
* The LLM can be changed without changing the document parser.
* The network implementation can be changed without modifying the RAG logic.
* Tests can verify individual components independently.

---

# 35. Quick Start

```powershell
# 1. Activate virtual environment
.venv\Scripts\Activate.ps1

# 2. Install package
pip install -e ".[dev]"

# 3. Run tests
python -m pytest -v

# 4. Start Streamlit application
streamlit run src/legal_system_rag/app.py
```

Then open the Streamlit application in the browser.

---

# 36. Summary

An end-to-end RAG pipeline demonstrating professional AI engineering practices using a curated set of German tenancy law texts

The architecture separates:

```text
Configuration
      │
      ├── Network
      │
      ├── Parsing
      │
      ├── Ingestion
      │
      ├── Vector Storage
      │
      ├── Retrieval
      │
      ├── LLM Generation
      │
      ├── Testing
      │
      └── User Interface
```

The result is a modular Python application that combines legal document processing, vector retrieval, LLM-based generation, source transparency, automated testing, corporate network support, and Streamlit deployment.

````
