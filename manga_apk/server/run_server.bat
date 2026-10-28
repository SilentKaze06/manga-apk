@echo off
chcp 65001 > nul
echo ============================================================
echo   AI Manga ^& Kitob PDF Tarjimon Serveri (FastAPI)
echo   ML Nutq bulutlarini aniqlash, Inpainting va Tarjima
echo ============================================================
echo.

cd /d "%~dp0"

set "PYTHON_EXE="

if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
    goto :found_python
)

where python >nul 2>nul
if errorlevel 1 (
    echo [XATOLIK] Python tizimda topilmadi! Iltimos, Python 3.10+ o'rnating.
    pause
    exit /b 1
)

echo [1/3] Virtual muhit yaratilmoqda...
python -m venv venv
if errorlevel 1 exit /b 1
set "PYTHON_EXE=venv\Scripts\python.exe"

:found_python
rem Use one complete server environment, not the separate training environment.
"%PYTHON_EXE%" -c "import fastapi, uvicorn, torch, torchvision, easyocr, paddleocr, pymupdf, wordninja" >nul 2>nul
if errorlevel 1 (
    echo [SETUP] Server kutubxonalari to'ldirilmoqda...
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 exit /b 1
    "%PYTHON_EXE%" -c "import fastapi, uvicorn, torch, torchvision, easyocr, paddleocr, pymupdf, wordninja" >nul 2>nul
    if errorlevel 1 (
        echo [XATOLIK] Server kutubxonalari yuklanmadi. Python muhiti va requirements.txt mosligini tekshiring.
        exit /b 1
    )
)
if not defined MANGA_BIND_HOST set "MANGA_BIND_HOST=127.0.0.1"
if "%MANGA_BIND_HOST%"=="127.0.0.1" (
    if not defined MANGA_SERVER_API_KEY set "MANGA_ALLOW_UNAUTHENTICATED=true"
) else (
    if not defined MANGA_SERVER_API_KEY (
        echo [XATOLIK] Tarmoqqa ochish uchun MANGA_SERVER_API_KEY belgilang.
        exit /b 1
    )
)
echo [SERVER] Ishlatilayotgan Python: %PYTHON_EXE%
set "HOST_IP="
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4" 2^>nul') do (
    if not defined HOST_IP (
        for /f "tokens=1" %%b in ("%%a") do set "HOST_IP=%%b"
    )
)
if not defined HOST_IP set "HOST_IP=KOMPYUTER_IP"

echo ============================================================
echo   SERVER ISHGA TUSHMOQDA!
echo   Lokal manzil: http://127.0.0.1:8000
echo   Android Emulator uchun: http://10.0.2.2:8000
echo   Telefon uchun: MANGA_BIND_HOST=0.0.0.0 va MANGA_SERVER_API_KEY sozlang.
echo ============================================================
echo.

"%PYTHON_EXE%" -m uvicorn main:app --host "%MANGA_BIND_HOST%" --port 8000
pause
