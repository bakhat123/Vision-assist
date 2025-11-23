#!/usr/bin/env pwsh
# Simple helper to activate venv and run the server on Windows PowerShell

if (-Not (Test-Path -Path .\venv\Scripts\Activate.ps1)) {
    Write-Host "Virtual environment not found. Creating one..." -ForegroundColor Yellow
    python -m venv venv
}

Write-Host "Activating virtual environment..."
. .\venv\Scripts\Activate.ps1

Write-Host "Installing requirements (if needed)..."
pip install -r requirements.txt

Write-Host "Starting Uvicorn server on 0.0.0.0:8000..."
uvicorn vision_server.main:app --host 0.0.0.0 --port 8000
