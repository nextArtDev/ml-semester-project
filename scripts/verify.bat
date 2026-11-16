@echo off
REM Verify the whole stack: pgvector, database, API and frontend.
setlocal
set PGHOST=localhost
set PGPORT=5432
set PGUSER=postgres
set PGPASSWORD=994038
set PGDATABASE=knn_knn_platform
set PSQL="C:\Program Files\PostgreSQL\18\bin\psql.exe"

echo.
echo === 1. pgvector extension ===
%PSQL% -tAc "SELECT extversion FROM pg_extension WHERE extname='vector'" 2>nul
if errorlevel 1 (
    echo [FAIL] Cannot connect to PostgreSQL. Is the service running?
    exit /b 1
)

echo.
echo === 2. API health ===
curl.exe -s http://127.0.0.1:8000/api/health
echo.

echo.
echo === 3. Frontend ===
curl.exe -s -o nul -w "HTTP %%{http_code}\n" http://localhost:3000/en

echo.
echo === 4. Backend tests ===
cd /d "%~dp0..\backend"
.venv\Scripts\python.exe -m pytest
