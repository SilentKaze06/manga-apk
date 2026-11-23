---
title: Dosjanmanga
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
python_version: "3.10"
app_file: app.py
pinned: false
suggested_hardware: "zero-a10g"
---

# Manga & Manhwa OCR Translation Server (ZeroGPU / Secondary Fallback)

FastAPI + Gradio backend for AI Manga Tarjimon Android app.

> **Note:** High-performance production workloads, heavy inpainting (SimpleLama), MangaTextCleaner, and batch PDF translation are hosted on the **Modal GPU Server (NVIDIA L4)** with NVMe-baked model weights. This Hugging Face Space operates as the secondary / zero-gpu fallback service.

### Endpoints
- `GET /` — Interactive Gradio UI
- `GET /api/info` — API metadata and health info
- `POST /translate-manga` — Manga dialogue OCR & translation
- `POST /ocr-manga` — Fast OCR text extraction
- `POST /detect-and-translate` — Bubble detection + OCR + translation
- `POST /translate-pdf` — Multi-page PDF translation
- `GET /pdf-jobs/{job_id}` — Job progress status
