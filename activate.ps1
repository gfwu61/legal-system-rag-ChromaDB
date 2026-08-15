# ============================================================
# activate.ps1
# Aktiviert die virtuelle Umgebung des Projekts
# ============================================================

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $ProjectRoot ".venv"

# Prüfen, ob .venv existiert
if (-not (Test-Path $VenvPath)) {
    Write-Host "Virtuelle Umgebung nicht gefunden." -ForegroundColor Yellow
    Write-Host "Erstelle .venv ..." -ForegroundColor Yellow

    python -m venv $VenvPath

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Fehler beim Erstellen der virtuellen Umgebung." -ForegroundColor Red
        exit 1
    }
}

# Aktivierung
$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"

if (Test-Path $ActivateScript) {
    & $ActivateScript
}
else {
    Write-Host "Activate.ps1 wurde nicht gefunden:" -ForegroundColor Red
    Write-Host $ActivateScript
    exit 1
}