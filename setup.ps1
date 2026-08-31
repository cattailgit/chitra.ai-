# setup.ps1  — Chitra.ai environment bootstrap (Windows / PowerShell)
#
# Run ONCE after installing Python 3.11 from https://python.org/downloads/
# Usage:  .\setup.ps1
#
# What it does
#   1. Locates the py.exe launcher for Python 3.11
#   2. Creates .venv311\ using that interpreter
#   3. Installs PyTorch (CPU) + all project dependencies
#   4. Prints activation and run instructions

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# 1. Find Python 3.11 via the Windows Launcher
# ---------------------------------------------------------------------------
Write-Host "`n[1/4] Checking for Python 3.11..." -ForegroundColor Cyan

$py311 = $null
try {
    $py311 = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
} catch {}

if (-not $py311) {
    Write-Host @"

  ERROR: Python 3.11 not found via the Windows Python Launcher (py.exe).

  PyTorch 2.x supports Python 3.9 - 3.12.
  You are currently running Python 3.14, which is NOT yet supported.

  Fix:
    1. Download Python 3.11.x from  https://www.python.org/downloads/release/python-3119/
       (Windows installer 64-bit: python-3.11.9-amd64.exe)
    2. During install check  [x] Add Python to PATH
                             [x] Install launcher for all users
    3. Re-run this script:   .\setup.ps1

"@ -ForegroundColor Red
    exit 1
}

Write-Host "  Found: $py311" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 2. Create virtual environment .venv311\
# ---------------------------------------------------------------------------
Write-Host "`n[2/4] Creating virtual environment .venv311\ ..." -ForegroundColor Cyan

if (Test-Path ".venv311") {
    Write-Host "  .venv311\ already exists — skipping creation." -ForegroundColor Yellow
} else {
    & $py311 -m venv .venv311
    Write-Host "  Created .venv311\" -ForegroundColor Green
}

$pip = ".venv311\Scripts\pip.exe"
$python = ".venv311\Scripts\python.exe"

# Upgrade pip silently
& $pip install --upgrade pip --quiet

# ---------------------------------------------------------------------------
# 3. Install dependencies
# ---------------------------------------------------------------------------
Write-Host "`n[3/4] Installing dependencies..." -ForegroundColor Cyan

# PyTorch CPU build (avoids ~2 GB CUDA download; swap URL for GPU if needed)
Write-Host "  Installing PyTorch (CPU)..."
& $pip install torch torchvision `
    --index-url https://download.pytorch.org/whl/cpu `
    --quiet

# Remaining requirements
Write-Host "  Installing project requirements..."
& $pip install -r requirements.txt --quiet

Write-Host "  All packages installed." -ForegroundColor Green

# ---------------------------------------------------------------------------
# 4. Print activation and run instructions
# ---------------------------------------------------------------------------
Write-Host @"

[4/4] Setup complete.

To activate the environment:
    .venv311\Scripts\Activate.ps1

To start the backend:
    .venv311\Scripts\python.exe -m backend.app

To verify PyTorch:
    .venv311\Scripts\python.exe -c "import torch; print(torch.__version__)"

NOTE: Set WATSONX_APIKEY and WATSONX_PROJECT_ID before starting the server
      if you want live IBM Granite inference (optional — fallback works without them).
"@ -ForegroundColor Cyan
