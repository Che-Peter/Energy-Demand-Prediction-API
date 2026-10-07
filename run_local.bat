@echo off
rem Runs the backend on http://localhost:8000 with its data in .\local_data
rem Then open the app with ?api=http://localhost:8000 to use it.
cd /d "%~dp0"
if not exist .venv (
    python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt
)
set DATA_DIR=local_data
.venv\Scripts\python -m uvicorn main:app --port 8000
