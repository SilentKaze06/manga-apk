# Build version: 2026-11-11-v85-maximum-heavy-models
import os
import sys
from pathlib import Path
import modal

SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parent.parent
ML_DIR = REPO_ROOT / "mashinali oqitish"

app = modal.App("manga-translator")

# Persistent cloud volume for PDFs, jobs, and downloaded model weights
volume = modal.Volume.from_name("manga-pdf-storage", create_if_missing=True)

# Container image with Debian 12 + Python 3.11 + PyTorch + all OCR/ML libraries
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libgomp1")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
    )
    .pip_install(
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "python-multipart>=0.0.6",
        "python-dotenv>=1.0.0",
        "onnxruntime>=1.16.0",
        "deep-translator>=1.11.4",
        "pymupdf>=1.24",
        "pymupdf-fonts>=1.0.0",
        "pillow>=9.5.0",
        "opencv-python-headless>=4.8.0",
        "wordninja>=2.0.0",
        "arabic-reshaper>=3.0.0",
        "python-bidi>=0.4.2",
        "psutil>=5.9.0",
    )
    .pip_install(
        "manga-ocr>=0.1.11",
        "easyocr>=1.7.1",
        "paddlepaddle>=2.5.0,<3.0",
        "paddleocr>=2.7.0,<3.0",
        "segmentation-models-pytorch>=0.3.3",
        "timm>=0.9.0",
        "albumentations>=1.4.0",
        "huggingface_hub>=0.25.0",
        "hf_xet>=1.8.0",
    )
    .env({
        "SERVER_CODE_VERSION": "v85-maximum-heavy-models",
        "MANGA_PDF_JOB_DIR": "/data/pdf_jobs",
        "MANGA_PDF_ENABLED": "true",
        "MANGA_PDF_PAGE_WORKERS": "8",
        "MANGA_USE_HEAVY_MODELS": "true",
        "MANGA_DISABLE_PADDLE": "false",
        "MANGA_PDF_OCR_DPI": "220",
        "MANGA_ML_DIR": "/root/mashinali_oqitish",
        "HF_HOME": "/data/cache/huggingface",
        "TORCH_HOME": "/data/cache/torch",
        "EASYOCR_MODULE_PATH": "/data/cache/easyocr",
    })
    .add_local_dir(
        str(ML_DIR),
        "/root/mashinali_oqitish",
        copy=True,
        ignore=[
            "manga_ml_datasets",
            "extracted_data",
            "eda_charts",
            "training_metrics",
            "reports",
            "__pycache__",
            ".vscode",
            ".agents",
            "tests",
            "*.zip",
            "self_learning_data",
            "recent_samples",
            "tools",
        ]
    )
    .add_local_dir(
        str(SERVER_DIR),
        "/root/server",
        copy=True,
        ignore=[
            ".env",
            "venv",
            "__pycache__",
            ".pytest_cache",
            "pdf_job_data",
            "cloudflared.exe",
            "tests",
            "deployment_tests",
        ]
    )
)


@app.function(
    image=image,
    cpu=8.0,
    memory=16384,
    volumes={"/data": volume},
    timeout=900,
    scaledown_window=300,
    max_containers=1,
)
@modal.asgi_app()
def fastapi_app():
    # Model cache persistence on high-speed volume
    for d in ["/data/cache/huggingface", "/data/cache/torch", "/data/cache/easyocr", "/data/cache/paddleocr", "/data/pdf_jobs"]:
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    # Ensure PaddleOCR writes directly to persistent volume
    try:
        root_paddle = Path("/root/.paddleocr")
        if not root_paddle.exists() and not root_paddle.is_symlink():
            root_paddle.symlink_to(Path("/data/cache/paddleocr"))
    except Exception:
        pass

    os.environ["HF_HOME"] = "/data/cache/huggingface"
    os.environ["TORCH_HOME"] = "/data/cache/torch"
    os.environ["EASYOCR_MODULE_PATH"] = "/data/cache/easyocr"
    os.environ["PADDLEOCR_HOME"] = "/data/cache/paddleocr"

    # API and server settings - read from environment, modal secret, or .env
    try:
        from dotenv import load_dotenv
        load_dotenv("/root/server/.env")
        load_dotenv(Path(__file__).resolve().parent / ".env")
        load_dotenv(SERVER_DIR / ".env")
    except Exception:
        pass

    os.environ.setdefault("MANGA_ALLOW_UNAUTHENTICATED", "false")
    os.environ.setdefault("MANGA_PDF_ENABLED", "true")
    os.environ.setdefault("MANGA_PDF_PAGE_WORKERS", "4")
    os.environ["MANGA_PDF_JOB_DIR"] = "/data/pdf_jobs"
    os.environ["MANGA_ML_DIR"] = "/root/mashinali_oqitish"
    os.environ.setdefault("MANGA_USE_HEAVY_MODELS", "true")
    os.environ["MANGA_DISABLE_PADDLE"] = "true"
    os.environ.setdefault("MANGA_USE_MYMEMORY_FALLBACK", "true")
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    os.environ.setdefault("GEMINI_API_KEY_FILE", "/data/gemini_api_key.txt")

    if "/root/server" not in sys.path:
        sys.path.insert(0, "/root/server")
    if "/root/mashinali_oqitish" not in sys.path:
        sys.path.insert(0, "/root/mashinali_oqitish")

    from main import app as web_app, resume_pdf_jobs
    resume_pdf_jobs()
    return web_app
