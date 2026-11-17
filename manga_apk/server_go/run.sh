#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "============================================================"
echo "  Manga & Manhwa Ultra-Fast Go Server (v2.0.0)"
echo "  RAM sarfi: ~15-25 MB (PyTorch 2.5GB o'rniga)"
echo "  Hech qanday Python yoki og'ir kutubxona kerak emas!"
echo "============================================================"

export PORT="${PORT:-8000}"
export MANGA_BIND_HOST="${MANGA_BIND_HOST:-0.0.0.0}"
export MANGA_ALLOW_UNAUTHENTICATED="${MANGA_ALLOW_UNAUTHENTICATED:-true}"

ARCH="$(uname -m)"
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    BINARY="./manga-server-linux-arm64"
else
    BINARY="./manga-server-linux"
fi

if [ ! -f "$BINARY" ]; then
    echo "Kompilyatsiya qilinmoqda ($ARCH)..."
    go build -o "$BINARY" main.go
fi

chmod +x "$BINARY"
echo "Server ishga tushmoqda: http://${MANGA_BIND_HOST}:${PORT}"
exec "$BINARY"

