@echo off
chcp 65001 >nul
title AI Manga ^& Comic Studio
cd /d "%~dp0"

echo ====================================================================
echo        AI Manga ^& Comic Studio - Ishga tushirish tizimi
echo ====================================================================
echo.

set "PYTHON_CMD="
set "PYTHON_ARGS="

REM 1. Virtual muhit mavjudligini tekshirish
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_CMD=%~dp0.venv\Scripts\python.exe"
    goto :PYTHON_FOUND
)
if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON_CMD=%~dp0venv\Scripts\python.exe"
    goto :PYTHON_FOUND
)
if exist "%~dp0..\manga_apk\server\venv\Scripts\python.exe" (
    set "PYTHON_CMD=%~dp0..\manga_apk\server\venv\Scripts\python.exe"
    goto :PYTHON_FOUND
)

REM 2. Tizimdagi 'python' buyrug'ini tekshirish
python --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto :PYTHON_FOUND
)

REM 3. 'py' launcher tekshirish
py -3 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py"
    set "PYTHON_ARGS=-3"
    goto :PYTHON_FOUND
)

REM 4. Standart Python manzillarini tekshirish
if exist "C:\Program Files\Python314\python.exe" (
    set "PYTHON_CMD=C:\Program Files\Python314\python.exe"
    goto :PYTHON_FOUND
)
if exist "C:\Program Files\Python312\python.exe" (
    set "PYTHON_CMD=C:\Program Files\Python312\python.exe"
    goto :PYTHON_FOUND
)
if exist "C:\Program Files\Python311\python.exe" (
    set "PYTHON_CMD=C:\Program Files\Python311\python.exe"
    goto :PYTHON_FOUND
)
if exist "C:\Program Files\Python310\python.exe" (
    set "PYTHON_CMD=C:\Program Files\Python310\python.exe"
    goto :PYTHON_FOUND
)
if exist "%LocalAppData%\Programs\Python\Python314\python.exe" (
    set "PYTHON_CMD=%LocalAppData%\Programs\Python\Python314\python.exe"
    goto :PYTHON_FOUND
)
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
    set "PYTHON_CMD=%LocalAppData%\Programs\Python\Python312\python.exe"
    goto :PYTHON_FOUND
)
if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
    set "PYTHON_CMD=%LocalAppData%\Programs\Python\Python311\python.exe"
    goto :PYTHON_FOUND
)
if exist "%LocalAppData%\Programs\Python\Python310\python.exe" (
    set "PYTHON_CMD=%LocalAppData%\Programs\Python\Python310\python.exe"
    goto :PYTHON_FOUND
)

echo [XATO] Kompyuteringizda Python topilmadi!
echo.
echo Iltimos, Python o'rnating: https://www.python.org/downloads/
echo (O'rnatishda "Add Python to PATH" belgisini yoqing)
echo.
pause
exit /b 1

:PYTHON_FOUND
echo [OK] Python aniqlandi: %PYTHON_CMD% %PYTHON_ARGS%
echo.

REM Kutubxonalarni tekshirish (gradio va torch)
"%PYTHON_CMD%" %PYTHON_ARGS% -c "import gradio, torch" >nul 2>&1
if errorlevel 1 goto :INSTALL_DEPS
goto :LAUNCH_APP

:INSTALL_DEPS
echo [DIQQAT] Kerakli kutubxonalar topilmadi.
echo Kutubxonalar avtomatik o'rnatilmoqda [requirements.txt]...
echo.
"%PYTHON_CMD%" %PYTHON_ARGS% -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [XATO] Kutubxonalarni o'rnatishda xatolik yuz berdi.
    pause
    exit /b 1
)
echo.
echo [OK] Barcha kutubxonalar muvaffaqiyatli o'rnatildi!
echo.

:LAUNCH_APP
echo ====================================================================
echo  AI Manga Studio ishga tushmoqda...
echo  Manzil: http://127.0.0.1:7860
echo  Brauzeringiz avtomatik ravishda ochiladi.
echo.
echo  Dasturni to'xtatish uchun shu konsol oynasida Ctrl+C bosing.
echo ====================================================================
echo.

"%PYTHON_CMD%" %PYTHON_ARGS% app.py %*

if errorlevel 1 (
    echo.
    echo [XATO] Dastur xatolik bilan to'xtadi.
    echo Yuqoridagi xabarlarni tekshiring.
    echo.
    pause
    exit /b 1
)

pause
