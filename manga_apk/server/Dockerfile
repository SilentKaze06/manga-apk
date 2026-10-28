# Build only the reviewed context created by prepare_deploy.py.
FROM python:3.10-slim

# PaddleOCR va rasm kutubxonalari uchun tizim paketlari
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home manga

WORKDIR /opt/manga

# Kutubxonalarni o'rnatish
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Explicit copies keep private PDF jobs and training feedback out of images.
COPY main.py pdf_jobs.py ./
COPY ml/ ./ml/

# Google Cloud Run va boshqa muhitlar uchun PORT muhit o'zgaruvchisi (Cloud Run odatda 8080 beradi)
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    MANGA_ML_DIR=/opt/manga/ml \
    MANGA_EASYOCR_MODEL_DIR=/opt/manga/ml/easyocr_models \
    MANGA_PDF_JOB_DIR=/var/lib/manga/pdf_jobs \
    MANGA_ALLOW_UNAUTHENTICATED=false
RUN mkdir -p /var/lib/manga/pdf_jobs \
    && chown -R manga:manga /opt/manga /var/lib/manga
USER manga
EXPOSE 8080

# Serverni ishga tushirish
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
