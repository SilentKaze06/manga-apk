# Build version: 2026-11-21-v86-modal-gpu-l4
import os
import sys
from pathlib import Path
import modal

SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parent.parent
ML_DIR = REPO_ROOT / "mashinali oqitish"

app = modal.App("manga-translator")

# Persistent cloud volume for PDFs and persistent job state
volume = modal.Volume.from_name("manga-pdf-storage", create_if_missing=True)


def bake_all_models():
    """
    Build-time pre-warming function executed during container image build.
    Bakes heavy neural models directly to fast container NVMe SSD (/opt/models/...)
    so containers start instantly without downloading weights at runtime.
    """
    import os
    import shutil
    from pathlib import Path

    # Set cache dirs directly to container NVMe SSD storage
    os.environ["HF_HOME"] = "/opt/models/huggingface"
    os.environ["TORCH_HOME"] = "/opt/models/torch"
    os.environ["EASYOCR_MODULE_PATH"] = "/opt/models/easyocr"
    os.environ["MANGA_EASYOCR_MODEL_DIR"] = "/opt/models/easyocr"
    os.environ["PADDLEOCR_HOME"] = "/opt/models/paddleocr"

    cleaner_dir = Path("/opt/models/manga_text_cleaner")
    cleaner_dir.mkdir(parents=True, exist_ok=True)
    Path("/opt/models/huggingface").mkdir(parents=True, exist_ok=True)
    Path("/opt/models/torch").mkdir(parents=True, exist_ok=True)
    Path("/opt/models/easyocr").mkdir(parents=True, exist_ok=True)
    Path("/opt/models/paddleocr").mkdir(parents=True, exist_ok=True)

    # a) Download a-b-c-x-y-z/Manga-Text-Segmentation-2025 (model.pth) via huggingface_hub.hf_hub_download(...)
    #    AND copy directly to /opt/models/manga_text_cleaner/model.pth.
    print("[bake_all_models] Downloading Manga-Text-Segmentation-2025 weights...")
    import huggingface_hub

    downloaded_path = huggingface_hub.hf_hub_download(
        repo_id="a-b-c-x-y-z/Manga-Text-Segmentation-2025",
        filename="model.pth",
        cache_dir="/opt/models/huggingface",
    )
    target_cleaner_path = cleaner_dir / "model.pth"
    shutil.copyfile(downloaded_path, target_cleaner_path)
    print(f"[bake_all_models] Saved segmentation model to {target_cleaner_path}")

    # b) Instantiate MangaOcr() once to cache kha-white/manga-ocr-base.
    print("[bake_all_models] Caching MangaOcr (kha-white/manga-ocr-base)...")
    from manga_ocr import MangaOcr

    _ = MangaOcr()

    # c) Instantiate easyocr.Reader for ["ja", "en"], ["en"], ["ko", "en"], ["ru", "en"] (gpu=False, download_enabled=True).
    print("[bake_all_models] Caching EasyOCR detection and recognition models...")
    import easyocr

    for lang_list in [["ja", "en"], ["en"], ["ko", "en"], ["ru", "en"]]:
        _ = easyocr.Reader(
            lang_list,
            gpu=False,
            download_enabled=True,
            model_storage_directory=os.environ.get("EASYOCR_MODULE_PATH", "/opt/models/easyocr"),
        )

    # d) Instantiate PaddleOCR(lang="en", use_angle_cls=False, show_log=False) (plus "japan", "korean", "ru").
    print("[bake_all_models] Caching PaddleOCR detection and recognition models...")
    from paddleocr import PaddleOCR

    for p_lang in ["en", "japan", "korean", "ru"]:
        try:
            _ = PaddleOCR(lang=p_lang, use_angle_cls=False, show_log=False)
        except Exception as e:
            print(f"[bake_all_models] PaddleOCR prewarm warning ({p_lang}): {e}")

    # e) Instantiate SimpleLama() once to bake big-lama.pt.
    print("[bake_all_models] Caching SimpleLama (big-lama.pt)...")
    from simple_lama_inpainting import SimpleLama

    _ = SimpleLama(device="cpu")

    print("[bake_all_models] All models successfully baked into container image!")


# Container image with Debian 12 + Python 3.11 + PyTorch + all OCR/ML libraries
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
        "tesseract-ocr",
        "tesseract-ocr-eng",
        "tesseract-ocr-jpn",
        "tesseract-ocr-kor",
        "tesseract-ocr-chi-sim",
        "tesseract-ocr-rus",
        "fonts-noto-cjk",
    )
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
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
        "pytesseract>=0.3.10",
        "simple-lama-inpainting>=0.1.2",
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
    .run_function(bake_all_models)
    .env({
        "SERVER_CODE_VERSION": "v86-modal-gpu-l4",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": "/opt/models/huggingface",
        "TORCH_HOME": "/opt/models/torch",
        "EASYOCR_MODULE_PATH": "/opt/models/easyocr",
        "MANGA_EASYOCR_MODEL_DIR": "/opt/models/easyocr",
        "PADDLEOCR_HOME": "/opt/models/paddleocr",
        "MANGA_TEXT_CLEANER_WEIGHTS": "/opt/models/manga_text_cleaner/model.pth",
        "MANGA_USE_HEAVY_MODELS": "true",
        "MANGA_DISABLE_PADDLE": "false",
        "MANGA_PDF_PAGE_WORKERS": "2",
        "MANGA_PDF_JOB_DIR": "/data/pdf_jobs",
        "MANGA_PDF_ENABLED": "true",
        "MANGA_PDF_OCR_DPI": "220",
        "MANGA_ML_DIR": "/root/mashinali_oqitish",
    })
    .add_local_dir(
        str(ML_DIR),
        "/root/mashinali_oqitish",
        copy=True,
        ignore=[
            "onnx_models",
            ".pytest_cache",
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
        ],
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
        ],
    )
)


@app.function(
    image=image,
    gpu="T4",
    cpu=4.0,
    memory=16384,
    volumes={"/data": volume},
    timeout=900,
    scaledown_window=300,
    max_containers=1,
    secrets=[modal.Secret.from_name("manga-translator-secrets")],
)
@modal.asgi_app()
def fastapi_app():
    # Ensure PDF jobs directory exists on high-speed persistent volume
    try:
        Path("/data/pdf_jobs").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Ensure PaddleOCR writes / looks up in baked NVMe directory
    try:
        root_paddle = Path("/root/.paddleocr")
        if not root_paddle.exists() and not root_paddle.is_symlink():
            root_paddle.symlink_to(Path("/opt/models/paddleocr"))
    except Exception:
        pass

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = "/opt/models/huggingface"
    os.environ["TORCH_HOME"] = "/opt/models/torch"
    os.environ["EASYOCR_MODULE_PATH"] = "/opt/models/easyocr"
    os.environ["MANGA_EASYOCR_MODEL_DIR"] = "/opt/models/easyocr"
    os.environ["PADDLEOCR_HOME"] = "/opt/models/paddleocr"
    os.environ["MANGA_TEXT_CLEANER_WEIGHTS"] = "/opt/models/manga_text_cleaner/model.pth"

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
    os.environ["MANGA_PDF_PAGE_WORKERS"] = "4"
    os.environ["MANGA_PDF_JOB_DIR"] = "/data/pdf_jobs"
    os.environ["MANGA_ML_DIR"] = "/root/mashinali_oqitish"
    os.environ["MANGA_USE_HEAVY_MODELS"] = "true"
    os.environ["MANGA_DISABLE_PADDLE"] = "false"
    os.environ.setdefault("MANGA_USE_MYMEMORY_FALLBACK", "true")
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    os.environ.setdefault("GEMINI_API_KEY_FILE", "/data/gemini_api_key.txt")

    if "/root/server" not in sys.path:
        sys.path.insert(0, "/root/server")
    if "/root/mashinali_oqitish" not in sys.path:
        sys.path.insert(0, "/root/mashinali_oqitish")

    from main import app as web_app

    return web_app
