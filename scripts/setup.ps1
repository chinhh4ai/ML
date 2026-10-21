$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PIP_CACHE_DIR = Join-Path (Get-Location) '.data\pip-cache'
$env:npm_config_cache = Join-Path (Get-Location) '.data\npm-cache'
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue) -or -not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    throw "Install FFmpeg first (e.g. winget install Gyan.FFmpeg), reopen PowerShell, and retry."
}
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Python 3.12 is required. Install it from python.org." }
}
& .\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed." }
& .\.venv\Scripts\python.exe scripts\verify_models.py
if ($LASTEXITCODE -ne 0) { throw "Bundled model verification failed." }
if (-not (Test-Path '.env')) { Copy-Item '.env.example' '.env' }
if (-not (Test-Path '.data\models\yolo11n-seg.onnx')) {
    Write-Host 'Exporting official model using CPU-only PyTorch (one-time download).'
    if (-not (Test-Path '.model-venv\Scripts\python.exe')) { py -3.12 -m venv .model-venv }
    & .\.model-venv\Scripts\python.exe -m pip install -r requirements-export.txt
    if ($LASTEXITCODE -ne 0) { throw "Model export dependency installation failed." }
    & .\.model-venv\Scripts\python.exe scripts\export_model.py
    if ($LASTEXITCODE -ne 0) { throw "Model export failed." }
}
if (-not (Test-Path 'frontend\dist\index.html')) {
    Push-Location frontend
    try {
        npm ci
        if ($LASTEXITCODE -ne 0) { throw "npm ci failed." }
        npm run build
        if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
    } finally { Pop-Location }
}
Write-Host 'Setup complete. Edit .env, then run: powershell -ExecutionPolicy Bypass -File scripts\start.ps1'
