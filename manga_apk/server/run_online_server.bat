@echo off
chcp 65001 > nul
title AI Manga Tarjimon - Onlayn Server (Cloudflare Tunnel)

echo ============================================================
echo   AI Manga Tarjimon Server (FastAPI + Cloudflare Tunnel)
echo   Telefon uchun 100%% Bepul Dunyo Bo'ylab Onlayn Aloqa
echo ============================================================
echo.

cd /d "%~dp0"

set "PYTHON_EXE="
if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else if exist "..\..\mashinali oqitish\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\..\mashinali oqitish\.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

set "MANGA_ALLOW_UNAUTHENTICATED=true"
set "MANGA_PDF_ENABLED=true"

echo [1/2] Mahalliy FastAPI serveri ishga tushirilmoqda...
start /b "" "%PYTHON_EXE%" -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

timeout /t 3 >nul

echo [2/2] Bepul xalqaro HTTPS tunnel (Cloudflare) ulanmoqda...
echo.
echo ============================================================
echo DIQQAT: Quyida "https://....trycloudflare.com" ko'rinishidagi
echo havola paydo bo'ladi.
echo.
echo Ushbu havolani nusxalab, telefoningizdagi AI Manga Tarjimon
echo ilovasi Sozlamalaridagi "Manga OCR Server URL" ga kiriting!
echo ============================================================
echo.

cloudflared.exe tunnel --protocol http2 --url http://127.0.0.1:8000
pause
