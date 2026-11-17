@echo off
chcp 65001 > nul
title Manga & Manhwa Ultra-Fast Go Server (v2.0.0)

echo ============================================================
echo   Manga & Manhwa Ultra-Fast Go Server (v2.0.0)
echo   Operativ xotira (RAM): ~15-25 MB (PyTorch 2.5 GB o'rniga)
echo ============================================================
echo.

set PORT=8000
set MANGA_BIND_HOST=0.0.0.0
set MANGA_ALLOW_UNAUTHENTICATED=true

if not exist manga-server.exe (
    echo [1/2] Go server kompilyatsiya qilinmoqda...
    go build -o manga-server.exe main.go
)

echo [2/2] Server ishga tushirilmoqda: http://127.0.0.1:8000
manga-server.exe
pause

