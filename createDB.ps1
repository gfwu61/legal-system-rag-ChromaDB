# ============================================================
# create_database.ps1
# Creates the ChromaDB database if it does not exist.
#
# Pipeline:
# 1. Set project root
# 2. Check virtual environment
# 3. Check Python executable
# 4. Check ChromaDB
# 5. Create ChromaDB if it does not exist
# 6. Verify database creation
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


# ============================================================
# 1. PROJECT INFORMATION
# ============================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor DarkGray
Write-Host " Legal RAG - Database Check" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor DarkGray
Write-Host ""

Write-Host "Project: $ProjectRoot" -ForegroundColor Cyan


# ============================================================
# 2. CHECK VIRTUAL ENVIRONMENT
# ============================================================

if (-not (Test-Path $VenvPath)) {

    Write-Host ""
    Write-Host "ERROR: Virtual environment not found." -ForegroundColor Red
    Write-Host ""
    Write-Host "Expected:" -ForegroundColor Yellow
    Write-Host $VenvPath -ForegroundColor Yellow
    Write-Host ""

    Exit 1
}


# ============================================================
# 3. CHECK PYTHON EXECUTABLE
# ============================================================

if (-not (Test-Path $PythonExe)) {

    Write-Host ""
    Write-Host "ERROR: Python executable not found." -ForegroundColor Red
    Write-Host ""
    Write-Host "Expected:" -ForegroundColor Yellow
    Write-Host $PythonExe -ForegroundColor Yellow
    Write-Host ""

    Exit 1
}

Write-Host "Python: $PythonExe" -ForegroundColor Green


# ============================================================
# 4. CHECK CHROMA DATABASE
# ============================================================

Write-Host ""
Write-Host "Checking ChromaDB ..." -ForegroundColor Cyan

if (Test-Path $ChromaPath) {

    Write-Host ""
    Write-Host "ChromaDB already exists." -ForegroundColor Green
    Write-Host "Database:" -ForegroundColor DarkGray
    Write-Host $ChromaPath -ForegroundColor DarkGray
    Write-Host ""

    Write-Host "Nothing to do." -ForegroundColor Green
    Write-Host ""

    Exit 0
}


# ============================================================
# 5. CREATE CHROMA DATABASE
# ============================================================

Write-Host ""
Write-Host "ChromaDB does not exist." -ForegroundColor Yellow
Write-Host "Creating database ..." -ForegroundColor Yellow
Write-Host ""

& $PythonExe -m legal_system_rag.ingest_documents

if ($LASTEXITCODE -ne 0) {

    Write-Host ""
    Write-Host "ERROR: Database creation failed." -ForegroundColor Red
    Write-Host ""

    Exit 1
}


# ============================================================
# 6. VERIFY DATABASE CREATION
# ============================================================

Write-Host ""
Write-Host "Verifying ChromaDB ..." -ForegroundColor Cyan

if (-not (Test-Path $ChromaPath)) {

    Write-Host ""
    Write-Host "ERROR: ChromaDB was not created." -ForegroundColor Red
    Write-Host ""
    Write-Host "Expected directory:" -ForegroundColor Yellow
    Write-Host $ChromaPath -ForegroundColor Yellow
    Write-Host ""

    Exit 1
}


# ============================================================
# 7. SUCCESS
# ============================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor DarkGray
Write-Host " Database created successfully." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor DarkGray
Write-Host ""

Write-Host "ChromaDB:" -ForegroundColor Green
Write-Host $ChromaPath -ForegroundColor DarkGray
Write-Host ""