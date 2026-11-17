@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0manga_apk\server"
echo ========================================================
echo   Manga Translator serverini Modal.com ga deploy qilish
echo ========================================================
modal deploy modal_app.py
