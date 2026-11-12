# Starts the backend and the frontend for local development, each in its own window.
#
#   .\start-dev.ps1          (or double-click start-dev.cmd)
#
# Close a window, or press Ctrl+C in it, to stop that server.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "frontend"
$python = Join-Path $root ".venv\Scripts\python.exe"
$shell = (Get-Process -Id $PID).Path   # same PowerShell that runs this script

# --- Checks -----------------------------------------------------------------
if (-not (Test-Path $python)) {
    Write-Host "Python virtual environment not found at $python" -ForegroundColor Red
    Write-Host "Create it first (see README > Setup > Backend)."
    exit 1
}
if (-not (Test-Path (Join-Path $backend ".env"))) {
    Write-Host "backend\.env not found. Copy backend\.env.example to backend\.env and set GROQ_API_KEY and ADMIN_API_KEY." -ForegroundColor Red
    exit 1
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "npm not found. Install Node.js from nodejs.org, then open a new PowerShell window." -ForegroundColor Red
    exit 1
}

# Backend port: PORT in backend\.env, else 8731 (same default as backend/app/config.py).
$port = 8731
$portLine = Get-Content (Join-Path $backend ".env") | Where-Object { $_ -match '^\s*PORT\s*=\s*(\d+)\s*$' } | Select-Object -Last 1
if ($portLine -match '(\d+)') { $port = [int]$Matches[1] }

# --- Backend ----------------------------------------------------------------
Write-Host "Starting backend on port $port ..." -ForegroundColor Cyan
Start-Process $shell -WorkingDirectory $backend -ArgumentList @(
    "-NoExit", "-Command",
    "`$Host.UI.RawUI.WindowTitle = 'ragbased backend'; & '$python' run.py"
)

# Wait until the backend answers (the first start loads the embedding model).
$healthUrl = "http://localhost:$port/api/health"
$ready = $false
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 2
    try {
        Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2 | Out-Null
        $ready = $true
        break
    } catch { Write-Host "." -NoNewline }
}
Write-Host ""
if ($ready) {
    Write-Host "Backend is ready at http://localhost:$port" -ForegroundColor Green
} else {
    Write-Host "Backend did not answer at $healthUrl after 3 minutes. Check the 'ragbased backend' window for errors." -ForegroundColor Yellow
    Write-Host "Starting the frontend anyway."
}

# --- Frontend ---------------------------------------------------------------
if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Write-Host "Installing frontend packages (first run only) ..." -ForegroundColor Cyan
    Push-Location $frontend
    try { npm install } finally { Pop-Location }
}

Write-Host "Starting frontend ..." -ForegroundColor Cyan
Start-Process $shell -WorkingDirectory $frontend -ArgumentList @(
    "-NoExit", "-Command",
    "`$Host.UI.RawUI.WindowTitle = 'ragbased frontend'; npm run dev"
)

Write-Host ""
Write-Host "Done. Open the 'Local:' URL shown in the 'ragbased frontend' window (http://localhost:5175 by default)." -ForegroundColor Green
