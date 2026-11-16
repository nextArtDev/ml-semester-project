@echo off
REM Start the Next.js frontend for the KNN classification platform.
setlocal
cd /d "%~dp0..\frontend"
if not exist "node_modules" (
    echo [setup] Installing dependencies with bun...
    bun install
)
if not exist ".env.local" (
    echo [setup] Creating .env.local from .env.example
    copy /y ".env.example" ".env.local" >nul
)
echo [run] Web -> http://localhost:3000
bun run dev
