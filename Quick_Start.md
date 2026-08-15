# Quick Start

## Legal RAG System

This document describes how to set up and start the Legal RAG System on Windows using the provided PowerShell scripts.

---

## 1. Project Structure

The relevant scripts are located in the project root directory:

```text
legal-system-rag/
│
├── activate.ps1
├── createDB.ps1
├── start.ps1
│
├── .venv/
├── chroma_legal_rag/
│
├── src/
│   └── legal_system_rag/
│       ├── app.py
│       ├── ingest_documents.py
│       └── ...
│
├── pyproject.toml
└── README.md
```

The scripts have different responsibilities:

| Script         | Purpose                                                     |
| -------------- | ----------------------------------------------------------- |
| `activate.ps1` | Creates/checks and activates the Python virtual environment |
| `createDB.ps1` | Creates the ChromaDB database if it does not exist          |
| `start.ps1`    | Starts the Streamlit application                            |

---

# 2. Start the Virtual Environment

Use:

```powershell
.\activate.ps1
```

The `activate.ps1` script prepares the Python virtual environment and activates it.

After successful activation, the PowerShell prompt should show:

```text
(.venv) PS C:\...\legal-system-rag>
```

You can verify the Python installation with:

```powershell
python --version
```

and:

```powershell
where.exe python
```

The Python executable should point to:

```text
legal-system-rag\.venv\Scripts\python.exe
```

### Purpose

The virtual environment isolates the project's Python dependencies from the system-wide Python installation.

---

# 3. Create the ChromaDB Database

After activating the virtual environment, use:

```powershell
.\createDB.ps1
```

The script checks whether the ChromaDB database already exists.

The expected database directory is:

```text
chroma_legal_rag/
```

### If the database does not exist

The script starts the document ingestion process:

```powershell
python -m legal_system_rag.ingest_documents
```

The documents are processed and stored in ChromaDB.

### If the database already exists

The script does **not** run the ingestion process again.

It simply reports:

```text
ChromaDB already exists.

Nothing to do.
```

This prevents unnecessary re-ingestion of the documents.

---

# 4. Start the Application

Use:

```powershell
.\start.ps1
```

The `start.ps1` script starts the Streamlit application.

It uses the Python executable from the project's virtual environment:

```text
.venv\Scripts\python.exe
```

and starts:

```text
src\legal_system_rag\app.py
```

using:

```powershell
python -m streamlit run src\legal_system_rag\app.py
```

After startup, Streamlit normally displays a local URL such as:

```text
Local URL: http://localhost:8501
```

Open this URL in a web browser.

---

# 5. Recommended Startup Sequence

For a new installation, use the following sequence:

### Step 1 — Activate the environment

```powershell
.\activate.ps1
```

### Step 2 — Create the database

```powershell
.\createDB.ps1
```

If the database already exists, nothing is changed.

### Step 3 — Start the application

```powershell
.\start.ps1
```

---

# 6. Normal Daily Startup

Once the environment and database have already been created, normally only these commands are required:

```powershell
.\activate.ps1
.\start.ps1
```

You only need to run:

```powershell
.\createDB.ps1
```

when the ChromaDB database does not exist, for example after:

* cloning the repository on a new computer
* deleting the local ChromaDB
* creating a new database
* rebuilding the vector database

---

# 7. Script Responsibilities

The three scripts deliberately have separate responsibilities.

## `activate.ps1`

```text
Python environment
       │
       ▼
    .venv
       │
       ▼
   Activated
```

It prepares and activates the project's Python environment.

---

## `createDB.ps1`

```text
Documents
    │
    ▼
Document ingestion
    │
    ▼
Embeddings
    │
    ▼
ChromaDB
```

It creates the local vector database **only when it does not already exist**.

---

## `start.ps1`

```text
ChromaDB
    │
    ▼
RAG application
    │
    ▼
Streamlit
    │
    ▼
Web Browser
```

It starts the Legal RAG application.

---

# 8. Important Files

The following files/directories are local and should normally not be committed to Git:

```text
.venv/
chroma_legal_rag/
.env
.streamlit/secrets.toml
```

The ChromaDB directory is excluded through `.gitignore`, for example:

```gitignore
chroma_legal_rag*/
```

This allows the vector database to remain local while the source code and configuration are stored in Git.

---

# 9. Troubleshooting

## PowerShell does not allow the script to run

If PowerShell reports that script execution is disabled, check the current execution policy:

```powershell
Get-ExecutionPolicy
```

For a user-level configuration, you can use:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then try again:

```powershell
.\activate.ps1
```

---

## Virtual environment not found

Run:

```powershell
.\activate.ps1
```

The script should create `.venv` if it does not already exist.

---

## ChromaDB does not exist

Run:

```powershell
.\createDB.ps1
```

The ingestion process should create:

```text
chroma_legal_rag/
```

---

## Streamlit does not start

First make sure the virtual environment is active:

```powershell
.\activate.ps1
```

Then start the application:

```powershell
.\start.ps1
```

If necessary, verify Streamlit:

```powershell
python -m streamlit --version
```

---

# 10. Quick Reference

| Task                        | Command            |
| --------------------------- | ------------------ |
| Activate/create `.venv`     | `.\activate.ps1`   |
| Create ChromaDB if missing  | `.\createDB.ps1`   |
| Start Streamlit application | `.\start.ps1`      |
| Check Python                | `python --version` |
| Check Git status            | `git status`       |

## Complete startup

```powershell
.\activate.ps1
.\createDB.ps1
.\start.ps1
```

## Normal daily startup

```powershell
.\activate.ps1
.\start.ps1
```

---

## Summary

The Legal RAG System uses three PowerShell scripts with clearly separated responsibilities:

```text
activate.ps1
     │
     ▼
  .venv
     │
     ▼
createDB.ps1
     │
     ▼
 ChromaDB
     │
     ▼
 start.ps1
     │
     ▼
 Streamlit
     │
     ▼
 Legal RAG Application
```

This separation ensures that environment setup, database creation, and application startup can be managed independently.
