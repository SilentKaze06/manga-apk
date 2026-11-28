# Build version: 2026-11-26-v90-modal-t4-ctd-qwen35-9b
import os
import sys
from pathlib import Path
import modal

SERVER_DIR = Path(__file__).resolve().parent

app = modal.App("manga-translator")

# Persistent cloud volume for PDFs and persistent job state
volume = modal.Volume.from_name("manga-pdf-storage", create_if_missing=True)


def bake_all_models():
    """
    Build-time pre-warming function executed during container image build.
    Bakes heavy neural models directly to fast container NVMe SSD (/opt/models/...)
    so containers start instantly without downloading weights at runtime.
    CPU build-time only — do NOT instantiate 4-bit CUDA models during build.
    """
    import os
    import shutil
    from pathlib import Path
    import huggingface_hub

    # Set cache dirs directly to container NVMe SSD storage
    os.environ["HF_HOME"] = "/opt/models/huggingface"
    os.environ["TORCH_HOME"] = "/opt/models/torch"

    cleaner_dir = Path("/opt/models/manga_text_cleaner")
    ctd_dir = Path("/opt/models/comic_text_detector")
    qwen_dir = Path("/opt/models/qwen3_5_9b")
    cleaner_dir.mkdir(parents=True, exist_ok=True)
    ctd_dir.mkdir(parents=True, exist_ok=True)
    qwen_dir.mkdir(parents=True, exist_ok=True)
    Path("/opt/models/huggingface").mkdir(parents=True, exist_ok=True)
    Path("/opt/models/torch").mkdir(parents=True, exist_ok=True)

    # a) Download a-b-c-x-y-z/Manga-Text-Segmentation-2025 (model.pth)
    print("[bake_all_models] Downloading Manga-Text-Segmentation-2025 weights...")
    downloaded_path = huggingface_hub.hf_hub_download(
        repo_id="a-b-c-x-y-z/Manga-Text-Segmentation-2025",
        filename="model.pth",
        cache_dir="/opt/models/huggingface",
    )
    target_cleaner_path = cleaner_dir / "model.pth"
    shutil.copyfile(downloaded_path, target_cleaner_path)
    print(f"[bake_all_models] Saved segmentation model to {target_cleaner_path}")

    # b) Download mayocream/comic-text-detector (YOLOv5 + UNet + DBNet)
    print("[bake_all_models] Downloading mayocream/comic-text-detector weights...")
    huggingface_hub.snapshot_download(
        repo_id="mayocream/comic-text-detector",
        local_dir=str(ctd_dir),
        cache_dir="/opt/models/huggingface",
    )
    print(f"[bake_all_models] Saved comic-text-detector to {ctd_dir}")

    # c) Download Qwen/Qwen3.5-9B weights (loaded in 4-bit NF4 at runtime on T4)
    print("[bake_all_models] Downloading Qwen/Qwen3.5-9B weights...")
    huggingface_hub.snapshot_download(
        repo_id="Qwen/Qwen3.5-9B",
        local_dir=str(qwen_dir),
        cache_dir="/opt/models/huggingface",
    )
    print(f"[bake_all_models] Saved Qwen3.5-9B to {qwen_dir}")

    # d) Download Manga/Anime fine-tuned Big-LaMa (anime-manga-big-lama.pt) and validate SimpleLama
    lama_dir = Path("/opt/models/lama_manga")
    lama_dir.mkdir(parents=True, exist_ok=True)
    lama_pt_path = lama_dir / "anime-manga-big-lama.pt"
    lama_url = "https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt"
    min_lama_bytes = 150 * 1024 * 1024

    if not lama_pt_path.is_file() or lama_pt_path.stat().st_size <= min_lama_bytes:
        print(f"[bake_all_models] Downloading anime-manga-big-lama.pt from {lama_url}...")
        downloaded_ok = False
        try:
            import httpx
            tmp_lama_path = lama_dir / "anime-manga-big-lama.pt.tmp"
            with httpx.stream("GET", lama_url, follow_redirects=True, timeout=300.0) as resp:
                resp.raise_for_status()
                with open(tmp_lama_path, "wb") as f_out:
                    for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                        if chunk:
                            f_out.write(chunk)
            if tmp_lama_path.is_file() and tmp_lama_path.stat().st_size > min_lama_bytes:
                tmp_lama_path.replace(lama_pt_path)
                downloaded_ok = True
            else:
                size_got = tmp_lama_path.stat().st_size if tmp_lama_path.is_file() else 0
                raise RuntimeError(f"Downloaded anime-manga-big-lama.pt too small ({size_got} bytes)")
        except Exception as dl_err:
            print(f"[bake_all_models] Primary GitHub download failed ({dl_err}); trying fallback mayocream/lama-manga...")
            fallback_path = huggingface_hub.hf_hub_download(
                repo_id="mayocream/lama-manga",
                filename="lama-manga.safetensors",
                local_dir=str(lama_dir),
                cache_dir="/opt/models/huggingface",
            )
            print(f"[bake_all_models] Saved fallback weights to {fallback_path}")

        if downloaded_ok:
            import torch
            print(f"[bake_all_models] Verifying TorchScript load for {lama_pt_path} ({lama_pt_path.stat().st_size / (1024 * 1024):.1f} MB)...")
            _jit_check = torch.jit.load(str(lama_pt_path), map_location="cpu")
            del _jit_check

    if lama_pt_path.is_file() and lama_pt_path.stat().st_size > min_lama_bytes:
        os.environ["LAMA_MODEL"] = str(lama_pt_path)
        os.environ["MANGA_LAMA_WEIGHTS"] = str(lama_pt_path)
        print(f"[bake_all_models] Validating SimpleLama(device='cpu') with LAMA_MODEL={lama_pt_path}...")
        from simple_lama_inpainting import SimpleLama
        _ = SimpleLama(device="cpu")
    else:
        print("[bake_all_models] WARNING: anime-manga-big-lama.pt unavailable; falling back to default SimpleLama...")
        from simple_lama_inpainting import SimpleLama
        _ = SimpleLama(device="cpu")

    # e) Download ogkalu/comic-text-and-bubble-detector (RT-DETR-v2 instance detector)
    rtdetr_dir = Path("/opt/models/rtdetr_bubble_detector")
    rtdetr_dir.mkdir(parents=True, exist_ok=True)
    print("[bake_all_models] Downloading ogkalu/comic-text-and-bubble-detector weights...")
    huggingface_hub.snapshot_download(
        repo_id="ogkalu/comic-text-and-bubble-detector",
        local_dir=str(rtdetr_dir),
        cache_dir="/opt/models/huggingface",
    )
    print(f"[bake_all_models] Saved comic-text-and-bubble-detector to {rtdetr_dir}")
    print("[bake_all_models] Validating AutoImageProcessor & AutoModelForObjectDetection...")
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    _ = AutoImageProcessor.from_pretrained(str(rtdetr_dir), local_files_only=True)
    _ = AutoModelForObjectDetection.from_pretrained(str(rtdetr_dir), local_files_only=True)

    print("[bake_all_models] All models successfully baked into container image!")


# Container image with Debian 12 + Python 3.11 + PyTorch + comic-text-detector + Qwen3.5-9B
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
        "torch>=2.6.0,<3.0.0",
        "torchvision>=0.21.0,<0.30.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy>=1.24.0,<2.0.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "python-multipart>=0.0.6",
        "python-dotenv>=1.0.0",
        "pymupdf>=1.24,<3.0",
        "pymupdf-fonts>=1.0.0",
        "pillow>=9.5.0",
        "opencv-python-headless>=4.8.0",
        "wordninja>=2.0.0,<3.0.0",
        "arabic-reshaper>=3.0.0",
        "python-bidi>=0.4.2",
        "psutil>=5.9.0",
        "pytesseract>=0.3.10",
        "simple-lama-inpainting>=0.1.2",
        "httpx>=0.24.0",
        "pyclipper>=1.3.0",
        "shapely>=2.0.0",
    )
    .pip_install(
        "transformers>=4.57.0",
        "accelerate>=1.1.0",
        "bitsandbytes>=0.45.0",
        "qwen-vl-utils>=0.0.14",
        "safetensors>=0.8.0",
        "segmentation-models-pytorch>=0.3.3",
        "timm>=0.9.0",
        "albumentations>=1.4.0,<2.0.0",
        "huggingface_hub>=0.25.0",
        "hf_xet>=1.8.0",
    )
    .run_function(bake_all_models)
    .env({
        "SERVER_CODE_VERSION": "v95-modal-t4-rtdetr-ctd-lama",
        "MANGA_USE_FP16": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": "/opt/models/huggingface",
        "TORCH_HOME": "/opt/models/torch",
        "LAMA_MODEL": "/opt/models/lama_manga/anime-manga-big-lama.pt",
        "MANGA_LAMA_WEIGHTS": "/opt/models/lama_manga/anime-manga-big-lama.pt",
        "MANGA_TEXT_CLEANER_WEIGHTS": "/opt/models/manga_text_cleaner/model.pth",
        "MANGA_CTD_MODEL_DIR": "/opt/models/comic_text_detector",
        "MANGA_RTDETR_DIR": "/opt/models/rtdetr_bubble_detector",
        "MANGA_QWEN_MODEL_DIR": "/opt/models/qwen3_5_9b",
        "MANGA_QWEN_4BIT": "1",
        "MANGA_USE_HEAVY_MODELS": "true",
        "MANGA_PDF_PAGE_WORKERS": "2",
        "MANGA_PDF_JOB_DIR": "/data/pdf_jobs",
        "MANGA_PDF_ENABLED": "true",
        "MANGA_PDF_OCR_DPI": "220",
        "MANGA_ML_DIR": "/root/server/ml",
    })
    .add_local_dir(
        str(SERVER_DIR),
        "/root/server",
        copy=True,
        ignore=[
            ".env",
            "venv",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "pdf_job_data",
            "cloudflared.exe",
            "tests",
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
    # Enable FP16/TF32 for comic-text-detector, Qwen3.5-9B, SimpleLama, and segmentation models when CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            os.environ["MANGA_USE_FP16"] = "1"
    except Exception:
        pass

    # Ensure PDF jobs directory exists on high-speed persistent volume
    try:
        Path("/data/pdf_jobs").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = "/opt/models/huggingface"
    os.environ["TORCH_HOME"] = "/opt/models/torch"
    os.environ["MANGA_TEXT_CLEANER_WEIGHTS"] = "/opt/models/manga_text_cleaner/model.pth"
    os.environ["MANGA_CTD_MODEL_DIR"] = "/opt/models/comic_text_detector"
    if Path("/opt/models/rtdetr_bubble_detector").is_dir():
        os.environ["MANGA_RTDETR_DIR"] = "/opt/models/rtdetr_bubble_detector"
    os.environ["MANGA_QWEN_MODEL_DIR"] = "/opt/models/qwen3_5_9b"
    os.environ["MANGA_QWEN_4BIT"] = "1"
    if Path("/opt/models/lama_manga/anime-manga-big-lama.pt").is_file():
        os.environ["LAMA_MODEL"] = "/opt/models/lama_manga/anime-manga-big-lama.pt"
        os.environ["MANGA_LAMA_WEIGHTS"] = "/opt/models/lama_manga/anime-manga-big-lama.pt"

    # API and server settings - read from environment, modal secret, or .env
    try:
        from dotenv import load_dotenv

        load_dotenv("/root/server/.env", override=False)
        load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
        load_dotenv(SERVER_DIR / ".env", override=False)
    except Exception:
        pass

    os.environ.setdefault("MANGA_ALLOW_UNAUTHENTICATED", "false")
    os.environ.setdefault("MANGA_PDF_ENABLED", "true")
    os.environ.setdefault("MANGA_PDF_PAGE_WORKERS", "2")
    os.environ.setdefault("MANGA_PDF_JOB_DIR", "/data/pdf_jobs")
    os.environ.setdefault("MANGA_ML_DIR", "/root/server/ml")
    os.environ.setdefault("MANGA_USE_HEAVY_MODELS", "true")
    os.environ.setdefault("MANGA_USE_FP16", "1")
    os.environ.setdefault("MANGA_USE_MYMEMORY_FALLBACK", "true")
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    os.environ.setdefault("GEMINI_API_KEY_FILE", "/data/gemini_api_key.txt")

    if "/root/server" not in sys.path:
        sys.path.insert(0, "/root/server")
    if "/root/server/ml" not in sys.path:
        sys.path.insert(0, "/root/server/ml")

    from main import app as web_app

    return web_app


@app.function(
    image=image,
    gpu="T4",
    cpu=4.0,
    memory=16384,
    timeout=300,
    scaledown_window=300,
    max_containers=1,
)
def detect_only_remote(image_bytes: bytes) -> tuple[bytes, list[dict]]:
    """
    Fast GPU detection-only endpoint loading ONLY ComicTextDetector + RT-DETR-v2
    (never Qwen3.5-9B or SimpleLama). Returns (annotated_png_bytes, blocks_json).
    """
    import io
    import sys
    from PIL import Image

    if "/root/server" not in sys.path:
        sys.path.insert(0, "/root/server")
    if "/root/server/ml" not in sys.path:
        sys.path.insert(0, "/root/server/ml")

    from comic_detector import get_shared_comic_detector
    from bubble_rtdetr import annotate_detection_boxes

    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    det = get_shared_comic_detector()
    blocks, _ = det.detect(pil_img)
    annotated = annotate_detection_boxes(pil_img, blocks)
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    return buf.getvalue(), blocks


@app.local_entrypoint()
def test_detect(image: str, out: str = "debug_detect.png"):
    from pathlib import Path

    img_path = Path(image)
    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found: {image}")
    print(f"[test_detect] Running remote GPU detection on {img_path} ({img_path.stat().st_size} bytes)...")
    png_bytes, blocks = detect_only_remote.remote(img_path.read_bytes())
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(png_bytes)
    print(f"[test_detect] Success! Saved overlay to {out_path} ({len(png_bytes)} bytes)")
    print(f"[test_detect] Detected {len(blocks)} blocks:")
    for b in blocks:
        print(f"  [{b.get('block_id')}] {b.get('category')} box={b.get('box')} text_box={b.get('text_box')}")


@app.function(
    image=image,
    gpu="T4",
    cpu=4.0,
    memory=16384,
    timeout=300,
    scaledown_window=300,
    max_containers=1,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("manga-translator-secrets")],
)
async def translate_page_remote(image_bytes: bytes, target_lang: str = "uz", lang: str = "auto") -> bytes:
    """
    Runs the full end-to-end /translate-page-image pipeline (detection, LaMa cleaning,
    Qwen3.5-9B OCR + translation, and typesetting) on Modal T4 GPU.
    """
    import io
    import sys
    from starlette.datastructures import UploadFile

    if "/root/server" not in sys.path:
        sys.path.insert(0, "/root/server")
    if "/root/server/ml" not in sys.path:
        sys.path.insert(0, "/root/server/ml")

    import main as server_main

    upload = UploadFile(filename="page.jpg", file=io.BytesIO(image_bytes))
    resp = await server_main.translate_page_image(
        file=upload,
        lang=lang,
        target_lang=target_lang,
        _authorized=None,
    )
    return resp.body

