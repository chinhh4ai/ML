$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path '.venv\Scripts\python.exe')) { throw "Run scripts\setup.ps1 first." }
if (-not (Test-Path 'frontend\dist\index.html')) { throw "Build frontend first: cd frontend; npm ci; npm run build" }
Write-Host 'Riverlens: http://127.0.0.1:5076 (Ctrl+C to stop)'
& .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 5076
exit $LASTEXITCODE
