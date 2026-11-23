@echo off
setlocal
cd /d "%~dp0"
chcp 65001 > nul
title Manga & Manhwa Ultra-Fast Go Server (v2.0.0)

echo ============================================================
echo   Manga & Manhwa Ultra-Fast Go Server (v2.0.0)
echo   Operativ xotira (RAM): ~15-25 MB (PyTorch 2.5 GB o'rniga)
echo ============================================================
echo.

if "%PORT%"=="" set PORT=8000
if "%MANGA_BIND_HOST%"=="" set MANGA_BIND_HOST=0.0.0.0
if "%MANGA_ALLOW_UNAUTHENTICATED%"=="" set MANGA_ALLOW_UNAUTHENTICATED=false

if not exist manga-server.exe (
    echo [1/2] Go server kompilyatsiya qilinmoqda...
    where go >nul 2>&1
    if errorlevel 1 (
        echo [XATO] Go dasturlash tili topilmadi! Iltimos, Go o'rnating yoki PATH ga qo'shing.
        pause
        exit /b 1
    )
    go build -o manga-server.exe main.go
    if errorlevel 1 (
        echo [XATO] Go serverni kompilyatsiya qilishda xatolik yuz berdi!
        pause
        exit /b 1
    )
)

echo [2/2] Server ishga tushirilmoqda: http://127.0.0.1:%PORT%
manga-server.exe
if errorlevel 1 (
    echo [XATO] Server xatolik bilan to'xtadi!
    pause
    exit /b %errorlevel%
)
