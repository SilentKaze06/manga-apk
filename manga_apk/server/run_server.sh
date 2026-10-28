#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "============================================================"
echo "  Manga & Manhwa OCR Translation Server (FastAPI)"
echo "  Yapon (Manga-OCR) + Koreys & Xitoy (PaddleOCR)"
echo "============================================================"

if ! command -v python3 &> /dev/null; then
    echo "[XATOLIK] Python3 tizimda topilmadi!"
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "[1/3] Virtual muhit (venv) yaratilmoqda..."
    python3 -m venv venv
fi

echo "[2/3] Virtual muhit faollashtirilmoqda..."
source venv/bin/activate

echo "[3/3] Kerakli kutubxonalar o'rnatilmoqda..."
pip install -r requirements.txt
python -c "import fastapi, uvicorn, torch, torchvision, easyocr, paddleocr, pymupdf, wordninja"

MANGA_BIND_HOST="${MANGA_BIND_HOST:-127.0.0.1}"
if [ "$MANGA_BIND_HOST" = "127.0.0.1" ]; then
    if [ -z "${MANGA_SERVER_API_KEY:-}" ]; then
        export MANGA_ALLOW_UNAUTHENTICATED=true
    fi
elif [ -z "${MANGA_SERVER_API_KEY:-}" ]; then
    echo "Tarmoqqa ochish uchun MANGA_SERVER_API_KEY belgilang."
    exit 1
fi
echo "SERVER ISHGA TUSHMOQDA: http://${MANGA_BIND_HOST}:8000"
exec python -m uvicorn main:app --host "$MANGA_BIND_HOST" --port 8000
