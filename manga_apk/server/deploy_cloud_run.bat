@echo off
chcp 65001 > nul
echo ============================================================
echo   Manga OCR Serverni Google Cloud Run-ga yuklash (Deploy)
echo ============================================================
echo.

cd /d "%~dp0"

rem The current in-process PDF queue requires a persistent single-host disk.
rem Cloud Run deploys OCR endpoints only until a durable shared job backend exists.
if not defined MANGA_SERVER_SECRET (
    echo [XATOLIK] MANGA_SERVER_SECRET ni Secret Manager nomi:versiyasi qilib belgilang.
    echo Masalan: manga-server-key:latest. Kalit qiymatini skriptga yozmang.
    exit /b 1
)
set "MANGA_DEPLOY_CONTEXT="
for /f "delims=" %%i in ('python prepare_deploy.py') do set "MANGA_DEPLOY_CONTEXT=%%i"
if not defined MANGA_DEPLOY_CONTEXT (
    echo [XATOLIK] Xavfsiz deploy paketi yaratilmadi.
    exit /b 1
)
echo Paket: %MANGA_DEPLOY_CONTEXT%
echo Cloud Run rejimida PDF vazifalari o'chirilgan; lokal serverda PDF ishlaydi.

echo [1/3] Google Cloud autentifikatsiyasini tekshirish...
call gcloud auth print-access-token >nul 2>nul
if errorlevel 1 (
    echo [DIQQAT] Google Cloud tizimiga kirish kerak. Brauzer ochilmoqda...
    call gcloud auth login
    if errorlevel 1 (
        echo [XATOLIK] Tizimga kirish bekor qilindi.
        pause
        exit /b 1
    )
)

echo [2/3] Faol Google Cloud loyihasi:
for /f "tokens=*" %%i in ('gcloud config get-value project 2^>nul') do set CURRENT_PROJECT=%%i
if "%CURRENT_PROJECT%"=="" set CURRENT_PROJECT=nimble-furnace-494010-p6
echo Ishlatilayotgan loyiha: %CURRENT_PROJECT%

echo.
echo [3/3] Cloud Run-ga deploy qilish boshlanmoqda...
echo Bu jarayon bir necha daqiqa vaqt olishi mumkin (Docker image quriladi va ishga tushiriladi).
echo.

call gcloud run deploy manga-ocr-server ^
  --source "%MANGA_DEPLOY_CONTEXT%" ^
  --project "%CURRENT_PROJECT%" ^
  --platform managed ^
  --region asia-south1 ^
  --memory 8Gi ^
  --cpu 2 ^
  --concurrency 1 ^
  --max 1 ^
  --timeout 300 ^
  --set-secrets "MANGA_SERVER_API_KEY=%MANGA_SERVER_SECRET%" ^
  --set-env-vars "MANGA_ALLOW_UNAUTHENTICATED=false,MANGA_PDF_ENABLED=false" ^
  --allow-unauthenticated

if errorlevel 1 (
    echo.
    echo [XATOLIK] Deploy amalga oshmadi. Yuqoridagi xabarni tekshiring.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   MUVAFFAQIN Bajarildi!
echo   Yuqorida berilgan Service URL manzilini (https://...)
echo   ScreenTranslator ilovasining "Server va API" sozlamalariga kiriting!
echo ============================================================
pause
