# ============================================================
# start.ps1
# Starts the Legal RAG application
#
# Pipeline:
# 1. Set project root
# 2. Create/check .venv
# 3. Check Python executable
# 4. Check ChromaDB
# 5. Run legal-ingest if ChromaDB is missing
# 6. Check ChromaDB after ingest
# 7. Check Streamlit app
# 8. Start Streamlit
# ============================================================

$ErrorActionPreference = "Stop"


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"

$ChromaPath = Join-Path $ProjectRoot "chroma_legal_rag"

$AppPath = Join-Path $ProjectRoot "src\legal_system_rag\app.py"


# ============================================================
# 1. PROJECT INFORMATION
# ============================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor DarkGray
Write-Host " Legal RAG System" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor DarkGray
Write-Host ""

Write-Host "Project: $ProjectRoot" -ForegroundColor Cyan


# ============================================================
# 2. CHECK / CREATE VIRTUAL ENVIRONMENT
# ============================================================

if (-not (Test-Path $VenvPath)) {

    Write-Host ""
    Write-Host "Virtual environment not found." -ForegroundColor Yellow
    Write-Host "Creating .venv ..." -ForegroundColor Yellow

    python -m venv $VenvPath

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment." -ForegroundColor Red
        Exit 1
    }

    Write-Host ".venv created successfully." -ForegroundColor Green
}


# ============================================================
# 3. CHECK PYTHON EXECUTABLE
# ============================================================

if (-not (Test-Path $PythonExe)) {

    Write-Host ""
    Write-Host "ERROR: Python executable not found:" -ForegroundColor Red
    Write-Host $PythonExe -ForegroundColor Red

    Exit 1
}

Write-Host "Python: $PythonExe" -ForegroundColor Green


# ============================================================
# 4. CHECK PYTHON PACKAGE
# ============================================================

Write-Host ""
Write-Host "Checking Python package ..." -ForegroundColor Cyan

& $PythonExe -c "import legal_system_rag"

if ($LASTEXITCODE -ne 0) {

    Write-Host ""
    Write-Host "ERROR: legal_system_rag package cannot be imported." -ForegroundColor Red
    Write-Host ""
    Write-Host "The project package is probably not installed in .venv." -ForegroundColor Yellow
    Write-Host "Run:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "    $PythonExe -m pip install -e ." -ForegroundColor Cyan
    Write-Host ""

    Exit 1
}

Write-Host "legal_system_rag package found." -ForegroundColor Green


# ============================================================
# 5. CHECK CHROMA VECTOR DATABASE
# ============================================================

Write-Host ""
Write-Host "Checking ChromaDB ..." -ForegroundColor Cyan

if (-not (Test-Path $ChromaPath)) {

    Write-Host "ChromaDB not found." -ForegroundColor Yellow
    Write-Host "Running legal-ingest ..." -ForegroundColor Yellow
    Write-Host ""

    & $PythonExe -m legal_system_rag.ingest_documents

    if ($LASTEXITCODE -ne 0) {

        Write-Host ""
        Write-Host "ERROR: legal-ingest failed." -ForegroundColor Red

        Exit 1
    }

    Write-Host ""
    Write-Host "legal-ingest completed successfully." -ForegroundColor Green
}
else {

    Write-Host "ChromaDB found:" -ForegroundColor Green
    Write-Host $ChromaPath -ForegroundColor DarkGray
}


# ============================================================
# 6. CHECK CHROMA DATABASE AFTER INGEST
# ============================================================

if (-not (Test-Path $ChromaPath)) {

    Write-Host ""
    Write-Host "ERROR: ChromaDB was not created." -ForegroundColor Red
    Write-Host "Expected directory:" -ForegroundColor Yellow
    Write-Host $ChromaPath -ForegroundColor Yellow

    Exit 1
}


# ============================================================
# 7. CHECK STREAMLIT APP
# ============================================================

if (-not (Test-Path $AppPath)) {

    Write-Host ""
    Write-Host "ERROR: Streamlit application not found:" -ForegroundColor Red
    Write-Host $AppPath -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please check your project structure." -ForegroundColor Yellow

    Exit 1
}


# ============================================================
# 8. START STREAMLIT
# ============================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor DarkGray
Write-Host " Starting Streamlit application ..." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor DarkGray
Write-Host ""

& $PythonExe -m streamlit run $AppPath


# ============================================================
# 9. HANDLE STREAMLIT EXIT
# ============================================================

if ($LASTEXITCODE -ne 0) {

    Write-Host ""
    Write-Host "ERROR: Streamlit terminated with exit code $LASTEXITCODE." -ForegroundColor Red

    Exit $LASTEXITCODE
}