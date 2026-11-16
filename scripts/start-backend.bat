@echo off
REM Start the FastAPI backend for the KNN classification platform.
setlocal
cd /d "%~dp0..\backend"
if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating the virtual environment...
    python -m venv .venv
    .venv\Scripts\pip install -q -r requirements.txt
)
if not exist ".env" (
    echo [setup] Creating .env from .env.example
    copy /y ".env.example" ".env" >nul
)
echo [run] API      -> http://127.0.0.1:8000
echo [run] Swagger -> http://127.0.0.1:8000/docs
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
