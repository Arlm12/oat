Write-Host "🚀 Starting OpenAgentTrace on Windows..." -ForegroundColor Green

# Check location
if (-not (Test-Path "pyproject.toml")) {
    Write-Error "Please run this script from the OpenAgentTrace root directory"
    exit 1
}

# Create .oat dir
if (-not (Test-Path ".oat")) {
    New-Item -ItemType Directory -Force -Path .oat | Out-Null
}

Write-Host "`nStep 1: Installing Python dependencies..." -ForegroundColor Cyan
pip install -e ".[server]"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install Python dependencies."
    exit 1
}

Write-Host "`nStep 2: Installing UI dependencies..." -ForegroundColor Cyan
Push-Location ui
npm install
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install UI dependencies."
    Pop-Location
    exit 1
}
Pop-Location

Write-Host "`nStep 3: Starting Backend Server..." -ForegroundColor Cyan
# Start Uvicorn in a new persistent window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd server; uvicorn main:app --port 8787 --host 0.0.0.0 --reload"

Write-Host "`nStep 4: Starting Frontend Dashboard..." -ForegroundColor Cyan
# Start NPM in a new persistent window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd ui; npm run dev"

Write-Host "`n✅ Services started in new windows!" -ForegroundColor Green
Write-Host "Dashboard: http://localhost:3000"
Write-Host "API:       http://localhost:8787"
Write-Host "`nTo stop the services, close the newly opened PowerShell windows."
