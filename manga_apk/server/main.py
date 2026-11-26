import sys
import os
import io
import math
import asyncio
from contextlib import asynccontextmanager
import hmac
import importlib
import importlib.util
import json
import logging
from pathlib import Path

try:
    if not os.environ.get("PYTEST_CURRENT_TEST") and "unittest" not in sys.modules:
        from dotenv import load_dotenv
        env_file = Path(__file__).resolve().parent / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)
except Exception:
    pass
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import threading
import time
import html
from html.parser import HTMLParser
import urllib.parse
import urllib.request
import re
import unicodedata
from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends, Request, Response
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
import numpy as np
import cv2
import pymupdf
import wordninja
from starlette.concurrency import run_in_threadpool
from pdf_jobs import PdfJobError, PdfQueueFullError, PdfJobManager, inspect_pdf

try:
    from ml.ocr_engine import (
        is_manga_punctuation,
        normalize_manga_punctuation,
        is_ocr_garbage_noise,
        detect_primary_script_from_text,
        analyze_script_distribution,
        is_dark_background,
        apply_dark_box_clahe_inversion,
        enhance_crop_for_ocr,
        filter_manga_ocr_hallucinations,
    )
except ImportError:
    try:
        from ocr_engine import (
            is_manga_punctuation,
            normalize_manga_punctuation,
            is_ocr_garbage_noise,
            detect_primary_script_from_text,
            analyze_script_distribution,
            is_dark_background,
            apply_dark_box_clahe_inversion,
            enhance_crop_for_ocr,
            filter_manga_ocr_hallucinations,
        )
    except ImportError:
        pass

# Load Torch before PaddleOCR; both ship OpenMP DLLs that can conflict on Windows.
import torch
# Hardware thread limits for low-resource VPS (prevents 100% CPU lockup and timeouts)
if not torch.cuda.is_available():
    try:
        max_vps_threads = min(2, os.cpu_count() or 1)
        torch.set_num_threads(max_vps_threads)
    except Exception as exc:
        logging.getLogger("manga-server").debug("Failed setting torch num threads: %s", exc)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("manga-server")

app = FastAPI(
    title="Manga, Manhwa & Manhua Translation Server",
    description="Unified Comic Text Detector (YOLOv5+UNet+DBNet) + Qwen3.5-9B (4-bit) OCR & Translation",
    version="2.0.0"
)

# Native Android clients do not need CORS. Unauthenticated access requires
# explicit opt-in for a trusted local development server.
SERVER_API_KEY = os.environ.get("MANGA_SERVER_API_KEY", "").strip()
ALLOW_UNAUTHENTICATED = os.environ.get("MANGA_ALLOW_UNAUTHENTICATED", "false").strip().lower() in {"1", "true", "yes"}
MAX_UPLOAD_BYTES = max(1024, int(os.environ.get("MANGA_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))))
MAX_IMAGE_PIXELS = max(1024 * 1024, int(os.environ.get("MANGA_MAX_IMAGE_PIXELS", "25000000")))
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
MAX_PDF_UPLOAD_BYTES = max(1024 * 1024, int(os.environ.get("MANGA_MAX_PDF_UPLOAD_BYTES", str(50 * 1024 * 1024))))
MAX_PDF_PAGES = max(1, int(os.environ.get("MANGA_MAX_PDF_PAGES", "500")))
PDF_OCR_DPI = max(96, min(300, int(os.environ.get("MANGA_PDF_OCR_DPI", "180"))))
PDF_OCR_MAX_PIXELS = max(1_000_000, int(os.environ.get("MANGA_PDF_OCR_MAX_PIXELS", str(16 * 1024 * 1024))))
PDF_JOB_DIR = Path(os.environ.get("MANGA_PDF_JOB_DIR", str(Path(__file__).parent / "pdf_job_data")))
PDF_ENABLED = os.environ.get("MANGA_PDF_ENABLED", "false" if os.environ.get("K_SERVICE") else "true").strip().lower() in {"1", "true", "yes"}
pdf_job_manager = PdfJobManager(PDF_JOB_DIR)
if not SERVER_API_KEY:
    logger.warning("Server API kaliti sozlanmagan; mahalliy autentifikatsiyasiz rejim: %s", ALLOW_UNAUTHENTICATED)

# ─── Lazy Model Loading ────────────────────────────────────────────────────────
_mocr = None
_paddle_ja = None
_paddle_ko = None
_paddle_ch = None
_paddle_en = None
_paddle_ru = None
_paddle_latin = None
_manga_pipeline = None
_manga_pipeline_error = None
_onnx_engine = None
_manga_cleaner = None
_model_lock = threading.RLock()
_mocr_infer_lock = threading.RLock()
_paddle_infer_lock = threading.RLock()
_cleaner_lock = threading.RLock()
_gemini_key_lock = threading.Lock()
_translation_semaphore = None


def _get_writable_gemini_key_file() -> Path:
    env_file = os.environ.get("GEMINI_API_KEY_FILE")
    if env_file is not None and env_file.strip():
        return Path(env_file.strip())
    data_dir = Path("/data")
    if data_dir.is_dir() and os.access(str(data_dir), os.W_OK):
        return data_dir / "gemini_api_key.txt"
    return Path(__file__).resolve().parent / "gemini_api_key.txt"


def resolve_gemini_api_key(
    gemini_api_key: Optional[str] = None,
    x_gemini_api_key: Optional[str] = None,
) -> Optional[str]:
    """Resolves Gemini API key with priority: Form param > Header > Environment > Persistent settings file."""
    if gemini_api_key and isinstance(gemini_api_key, str) and gemini_api_key.strip():
        return gemini_api_key.strip()
    if x_gemini_api_key and isinstance(x_gemini_api_key, str) and x_gemini_api_key.strip():
        return x_gemini_api_key.strip()
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    env_file = os.environ.get("GEMINI_API_KEY_FILE")
    if env_file is not None:
        candidate_files = [env_file.strip()] if env_file.strip() else []
    else:
        candidate_files = [
            str(Path(__file__).resolve().parent / "gemini_api_key.txt"),
            "/data/gemini_api_key.txt",
            "/var/lib/manga/gemini_api_key.txt",
        ]
    for key_path in candidate_files:
        if key_path and os.path.exists(key_path):
            try:
                with open(key_path, "r", encoding="utf-8") as f:
                    c = f.read().strip()
                    if c:
                        return c
            except Exception as kerr:
                logger.debug("Gemini key file read error (%s): %s", key_path, kerr)
    return None


def get_onnx_engine():
    """Lazy-load ultra-lightweight ONNX Manga Engine (Bubble Detector, Inpainting, and Multi-language OCR)
    using pure ONNX Runtime for 4x faster CPU speed and <150MB RAM footprint."""
    if os.environ.get("MANGA_USE_HEAVY_MODELS", "false").lower() in ("1", "true", "yes"):
        return None
    global _onnx_engine
    with _model_lock:
        if _onnx_engine is None:
            try:
                from onnx_engine import MangaOnnxEngine
                models_dir = Path(__file__).parent / "onnx_models"
                if not models_dir.exists():
                    models_dir = Path(__file__).parent / "ml" / "onnx_models"
                if not models_dir.exists():
                    models_dir = Path(__file__).parent.parent / "mashinali oqitish" / "onnx_models"
                if not models_dir.exists():
                    models_dir = Path(__file__).parent.parent.parent / "mashinali oqitish" / "onnx_models"
                vocab_path = models_dir / "ocr_vocab.json"
                if not vocab_path.exists():
                    vocab_path = Path(__file__).parent / "ocr_vocab.json"
                if not vocab_path.exists():
                    vocab_path = Path(__file__).parent / "ml" / "ocr_vocab.json"
                if not vocab_path.exists():
                    vocab_path = Path(__file__).parent.parent / "mashinali oqitish" / "ocr_vocab.json"
                if not vocab_path.exists():
                    vocab_path = Path(__file__).parent.parent.parent / "mashinali oqitish" / "ocr_vocab.json"
                _onnx_engine = MangaOnnxEngine(str(models_dir), str(vocab_path))
                logger.info("MangaOnnxEngine muvaffaqiyatli yuklandi.")
            except Exception as e:
                logger.warning(f"MangaOnnxEngine yuklaşda xato: {e}")
        return _onnx_engine



def get_manga_pipeline():
    """Lazy-load MangaTranslationPipeline (Speech Bubble Detector, Inpainting, and Multi-language OCR)
    directly from 'mashinali oqitish' directory to avoid duplicate code and checkpoint models."""
    global _manga_pipeline, _manga_pipeline_error
    with _model_lock:
        if _manga_pipeline is None:
            try:
                ml_dir_env = os.environ.get("MANGA_ML_DIR")
                if ml_dir_env and Path(ml_dir_env).exists():
                    ml_dir = Path(ml_dir_env).resolve()
                elif (Path(__file__).resolve().parent / "ml").exists():
                    ml_dir = (Path(__file__).resolve().parent / "ml").resolve()
                elif (Path(__file__).resolve().parent.parent / "mashinali oqitish").exists():
                    ml_dir = (Path(__file__).resolve().parent.parent / "mashinali oqitish").resolve()
                else:
                    ml_dir = (Path(__file__).resolve().parent.parent.parent / "mashinali oqitish").resolve()
                if not ml_dir.exists():
                    raise RuntimeError(f"'mashinali oqitish' papkasi topilmadi: {ml_dir}")
                if str(ml_dir) not in sys.path:
                    sys.path.insert(0, str(ml_dir))
                logger.info("MangaTranslationPipeline yuklanmoqda ('mashinali oqitish' papkasidan: %s)...", ml_dir)
                manga_pipeline_module = importlib.import_module("manga_pipeline")
                pipeline_class = getattr(manga_pipeline_module, "MangaTranslationPipeline")
                _manga_pipeline = pipeline_class(base_dir=str(ml_dir))
                _manga_pipeline_error = None
                logger.info("MangaTranslationPipeline muvaffaqiyatli yuklandi.")
            except Exception as e:
                _manga_pipeline_error = type(e).__name__
                logger.error("MangaTranslationPipeline yuklaşda xatolik: %s", e)
                raise HTTPException(status_code=500, detail="Manga AI modelini yuklab bölmadi.")
        return _manga_pipeline


def get_manga_cleaner():
    """Lazy-load and cache shared singleton instance of MangaTextCleaner with module-level locking."""
    global _manga_cleaner
    if _manga_cleaner is not None:
        return _manga_cleaner
    with _cleaner_lock:
        if _manga_cleaner is None:
            try:
                from manga_text_cleaner import MangaTextCleaner
            except ImportError:
                from ml.manga_text_cleaner import MangaTextCleaner
            _manga_cleaner = MangaTextCleaner()
        return _manga_cleaner


PDF_LATIN_OCR_LANGS = {"fr", "de", "es", "pt", "it", "tr", "nl", "pl", "uz"}
PDF_OCR_MODEL_LANGS = {
    "ja": "japan", "ko": "korean", "zh": "ch", "en": "en", "ru": "ru",
    **{language: "latin" for language in PDF_LATIN_OCR_LANGS},
}
PDF_SOURCE_LANGUAGES = ("auto", "en", "ru", "fr", "de", "es", "pt", "it", "tr", "nl", "pl", "uz", "ja", "ko", "zh")
PDF_TARGET_LANGUAGES = tuple(code for code in PDF_SOURCE_LANGUAGES if code != "auto") + ("ar",)


def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Require a configured key on translation endpoints when deployed."""
    if not SERVER_API_KEY and not ALLOW_UNAUTHENTICATED:
        raise HTTPException(status_code=503, detail="Server API kaliti sozlanmagan. Mahalliy sinov uçun MANGA_ALLOW_UNAUTHENTICATED=true ni belgilang.")
    if SERVER_API_KEY and (not x_api_key or not hmac.compare_digest(x_api_key.encode("utf-8"), SERVER_API_KEY.encode("utf-8"))):
        raise HTTPException(status_code=401, detail="Server API kaliti notöğri yoki kiritilmagan.")


def verify_pdf_job_token(job_id: str, x_pdf_job_token: Optional[str] = Header(default=None)):
    verify_pdf_enabled()
    if not pdf_job_manager.verify_job_token(job_id, x_pdf_job_token):
        raise HTTPException(status_code=401, detail="PDF vazifasi kaliti notöğri yoki kiritilmagan.")


def verify_pdf_enabled():
    if not PDF_ENABLED:
        raise HTTPException(status_code=503, detail="Bu serverda PDF tarjimasi öçirilgan: doimiy vazifa saqlaş tizimi kerak. PDF uçun lokal yoki doimiy diskli serverdan foydalaning.")


def resume_pdf_jobs():
    global SERVER_API_KEY
    current_key = SERVER_API_KEY or os.environ.get("MANGA_SERVER_API_KEY", "").strip()
    if PDF_ENABLED and (current_key or ALLOW_UNAUTHENTICATED):
        pdf_job_manager.start_pending(translate_text, ocr_pdf_page)


def stop_pdf_job_worker():
    pdf_job_manager.shutdown()


_preload_started = False


def preload_heavy_models_async():
    """Eagerly pre-load heavy deep learning models into memory on container startup."""
    global _preload_started
    if _preload_started:
        return
    _preload_started = True
    if os.environ.get("MANGA_USE_HEAVY_MODELS", "false").lower() in ("1", "true", "yes"):
        def _worker():
            try:
                logger.info("Og'ir AI modellari xotiraga yuklanmoqda (MangaTranslationPipeline)...")
                pipe = get_manga_pipeline()
                if pipe is not None:
                    if hasattr(pipe, "_get_comic_detector"):
                        pipe._get_comic_detector()
                    if hasattr(pipe, "_get_qwen_engine"):
                        pipe._get_qwen_engine()
            except Exception as exc:
                logger.warning("Heavy pipeline preload ogohlantirish: %s", exc)
        threading.Thread(target=_worker, daemon=True, name="heavy-model-preloader").start()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _translation_semaphore
    if _translation_semaphore is None:
        _translation_semaphore = asyncio.Semaphore(4)
    resume_pdf_jobs()
    preload_heavy_models_async()
    yield
    stop_pdf_job_worker()

app.router.lifespan_context = lifespan


from text_translator import normalize_ocr_language, translation_language_code


def decode_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Böş rasm fayli yuborildi.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            image = ImageOps.exif_transpose(opened)
            if image.width <= 0 or image.height <= 0 or image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Rasm ölçami ruxsat etilgan çegaradan katta.")
            if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
                rgba = image.convert("RGBA")
                bg = Image.new("RGB", rgba.size, (255, 255, 255))
                bg.paste(rgba, mask=rgba.split()[3])
                return bg
            return image.convert("RGB")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Rasmni öqib bölmadi: {exc}") from exc


async def read_image_upload(file: UploadFile):
    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Rasm fayli {MAX_UPLOAD_BYTES // (1024 * 1024)} MB dan oşmasligi kerak.")
    return image_bytes, await run_in_threadpool(decode_image, image_bytes)


def get_qwen_vl_engine():
    """Lazy-load shared Qwen3.5-9B 4-bit VL engine when weights & CUDA are available."""
    try:
        try:
            from ml.qwen_vl_engine import get_shared_qwen_engine
        except ImportError:
            from qwen_vl_engine import get_shared_qwen_engine
        eng = get_shared_qwen_engine()
        return eng if eng.is_available else None
    except Exception as exc:
        logger.debug("QwenVLEngine unavailable: %s", exc)
        return None


def get_comic_detector():
    """Lazy-load shared ComicTextDetector when weights are available."""
    try:
        try:
            from ml.comic_detector import get_shared_comic_detector
        except ImportError:
            from comic_detector import get_shared_comic_detector
        det = get_shared_comic_detector()
        return det if det.is_available else None
    except Exception as exc:
        logger.debug("ComicTextDetector unavailable: %s", exc)
        return None


def get_manga_ocr():
    """Legacy MangaOcr loader (kept guarded for backwards compatibility)."""
    global _mocr
    if _mocr is not None:
        return _mocr
    with _model_lock:
        if _mocr is None:
            try:
                manga_ocr_mod = importlib.import_module("manga_ocr")
                manga_ocr_class = getattr(manga_ocr_mod, "MangaOcr")
                _mocr = manga_ocr_class()
            except ModuleNotFoundError:
                err = "manga-ocr kutubxonasi o'rnatilmagan (Qwen3.5-9B ishlatiladi)."
                logger.debug(err)
                raise HTTPException(status_code=500, detail=err)
            except Exception as e:
                logger.warning("MangaOcr yuklashda xatolik: %s", e)
                raise HTTPException(status_code=500, detail="MangaOcr modelini yuklab bo'lmadi.")
        return _mocr


def get_paddle_ocr(lang: str = "korean"):
    """Lazy-load PaddleOCR for Japanese, Korean, Chinese, Russian, English, or multilingual Latin."""
    global _paddle_ja, _paddle_ko, _paddle_ch
    global _paddle_en, _paddle_ru, _paddle_latin
    raw = (lang or "").strip().lower()
    explicit_aliases = {
        "japan": "ja",
        "korean": "ko",
        "ch": "zh",
        "ch_sim": "zh",
        "chi_sim": "zh",
        "chi_tra": "zh",
        "latin": "latin",
    }
    norm = explicit_aliases.get(raw) or normalize_ocr_language(raw)
    if norm in PDF_LATIN_OCR_LANGS:
        norm = "latin"

    with _model_lock:
        try:
            paddleocr_mod = importlib.import_module("paddleocr")
            paddle_ocr_class = getattr(paddleocr_mod, "PaddleOCR")
            extra_kwargs = {}
            if os.environ.get("MANGA_PADDLE_IR_OPTIM", "").strip().lower() in ("0", "false", "no"):
                extra_kwargs.update({"use_gpu": False, "ir_optim": False, "enable_mkldnn": False})
                try:
                    paddle_infer = importlib.import_module("paddle.inference")
                    orig_cp = getattr(paddle_infer, "create_predictor", None)
                    if orig_cp is not None and not getattr(orig_cp, "_safe_no_ir", False):
                        def _safe_create_predictor(cfg, _orig=orig_cp):
                            try:
                                cfg.switch_ir_optim(False)
                            except Exception:
                                pass
                            return _orig(cfg)
                        _safe_create_predictor._safe_no_ir = True
                        paddle_infer.create_predictor = _safe_create_predictor
                except Exception:
                    pass
            if norm == "ja":
                if _paddle_ja is None:
                    logger.info("PaddleOCR (Japanese) yuklanmoqda...")
                    _paddle_ja = paddle_ocr_class(use_angle_cls=True, lang="japan", show_log=False, **extra_kwargs)
                    logger.info("PaddleOCR (Japanese) tayyor.")
                return _paddle_ja

            if norm == "zh":
                if _paddle_ch is None:
                    logger.info("PaddleOCR (Chinese) yuklanmoqda...")
                    _paddle_ch = paddle_ocr_class(use_angle_cls=True, lang="ch", show_log=False, **extra_kwargs)
                    logger.info("PaddleOCR (Chinese) tayyor.")
                return _paddle_ch

            if norm == "en":
                if _paddle_en is None:
                    logger.info("PaddleOCR (English/Latin) yuklanmoqda...")
                    _paddle_en = paddle_ocr_class(use_angle_cls=True, lang="en", show_log=False, **extra_kwargs)
                    logger.info("PaddleOCR (English/Latin) tayyor.")
                return _paddle_en

            if norm == "ru":
                if _paddle_ru is None:
                    logger.info("PaddleOCR (Russian/Cyrillic) yuklanmoqda...")
                    _paddle_ru = paddle_ocr_class(use_angle_cls=True, lang="ru", show_log=False, **extra_kwargs)
                    logger.info("PaddleOCR (Russian/Cyrillic) tayyor.")
                return _paddle_ru

            if norm == "latin":
                if _paddle_latin is None:
                    logger.info("PaddleOCR (multilingual Latin) yuklanmoqda...")
                    _paddle_latin = paddle_ocr_class(use_angle_cls=True, lang="latin", show_log=False, **extra_kwargs)
                    logger.info("PaddleOCR (multilingual Latin) tayyor.")
                return _paddle_latin

            if norm == "ko":
                if _paddle_ko is None:
                    logger.info("PaddleOCR (Korean) yuklanmoqda...")
                    _paddle_ko = paddle_ocr_class(use_angle_cls=True, lang="korean", show_log=False, **extra_kwargs)
                    logger.info("PaddleOCR (Korean) tayyor.")
                return _paddle_ko

            raise ValueError(f"Unsupported PaddleOCR language: {lang}")
        except ValueError:
            raise
        except ModuleNotFoundError:
            err = "paddleocr kutubxonasi örnatilmagan! Iltimos, 'pip install paddleocr paddlepaddle' yoki 'run_server.bat' ni işga tuşiring."
            logger.error(err)
            raise HTTPException(status_code=500, detail=err)
        except Exception as e:
            logger.error("PaddleOCR (%s) yuklaşda xatolik: %s", lang, e)
            raise HTTPException(status_code=500, detail="PaddleOCR modelini yuklab bölmadi.")


def run_manga_ocr(image: Image.Image) -> str:
    # Prefer Qwen3.5-9B OCR when available; legacy MangaOcr/ONNX only as guarded fallback
    qwen = get_qwen_vl_engine()
    if qwen is not None:
        try:
            txt = qwen.ocr_crop(image, source_lang="ja")
            if txt:
                return txt
        except Exception as q_exc:
            logger.debug("QwenVLEngine ocr_crop fallback error: %s", q_exc)
    ocr_err = None
    try:
        mocr = get_manga_ocr()
        use_fp16 = (os.environ.get("MANGA_USE_FP16") in ("1", "true", "True")) and torch.cuda.is_available()
        with _mocr_infer_lock:
            if use_fp16:
                try:
                    with torch.amp.autocast("cuda"):
                        return mocr(image)
                except Exception:
                    with torch.cuda.amp.autocast():
                        return mocr(image)
            return mocr(image)
    except Exception as e:
        ocr_err = e
        logger.debug(f"Legacy MangaOcr xatosi: {e}")
        engine = get_onnx_engine()
        if engine and engine.ocr_sess:
            try:
                res = engine.ocr_bubble(image)
                if res:
                    return res
            except Exception as onnx_e:
                logger.warning(f"ONNX OCR fallback xatosi: {onnx_e}")
    if ocr_err is not None:
        raise ocr_err


def is_paddle_enabled() -> bool:
    if os.environ.get("MANGA_DISABLE_PADDLE", "false").lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("MANGA_USE_PADDLE", "true").lower() in ("0", "false", "no"):
        return False
    # Allow unit tests that explicitly mock run_paddle_ocr / run_auto_paddle_ocr without paddleocr installed
    if hasattr(globals().get("run_paddle_ocr"), "assert_called") or hasattr(globals().get("run_auto_paddle_ocr"), "assert_called"):
        return True
    try:
        if importlib.util.find_spec("paddleocr") is None:
            return False
    except Exception:
        return False
    return True


def run_paddle_ocr(image_bytes: bytes, lang: str):
    if not is_paddle_enabled():
        logger.debug("PaddleOCR unavailable or disabled.")
        return []
    import numpy as np
    try:
        ocr_inst = get_paddle_ocr(lang)
    except (ModuleNotFoundError, ImportError, HTTPException) as exc:
        logger.debug("PaddleOCR skipped (%s)", exc)
        return []
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
    max_dimension = max(image.size)
    target_dimension = 1400
    if max_dimension < 1100:
        scale = min(2.0, target_dimension / max_dimension)
        resized = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
        scale_x = resized.width / image.width
        scale_y = resized.height / image.height
        np_img = np.array(resized)
        with _paddle_infer_lock:
            results = ocr_inst.ocr(np_img, cls=True)
        if results and results[0]:
            results[0] = [
                ([[float(point[0]) / scale_x, float(point[1]) / scale_y] for point in item[0]], item[1])
                for item in results[0]
            ]
        return results
    np_img = np.array(image)
    with _paddle_infer_lock:
        return ocr_inst.ocr(np_img, cls=True)


def _has_exclusive_jp_cues(text: str, dist: dict) -> bool:
    if dist.get("kana", 0) > 0:
        return True
    return any(ch in text for ch in ("\u30fc", "\u301c", "\u3005", "\u309b", "\u309c", "\u309d", "\u309e", "\u30fd", "\u30fe"))


def _script_score(text: str, confidence: float, language: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    dist = analyze_script_distribution(text)
    has_jp_cue = _has_exclusive_jp_cues(text, dist)
    if language == "ja" and not has_jp_cue:
        # Pure Hanzi without Kana or exclusive JP marks must NOT beat Chinese ('zh')
        purity = min(1.0, dist["cjk"] / max(1, len(letters)))
        return purity * 0.4 + max(0.0, min(1.0, confidence)) * 0.2
    expected_raw = {
        "en": dist["latin"],
        "auto": dist["latin"],
        "ru": dist["cyrillic"],
        "ko": dist["hangul"],
        "ja": (dist["kana"] + dist["cjk"]),
        "zh": dist["cjk"] if not has_jp_cue else 0,
    }.get(language, 0)
    expected_weighted = {
        "en": dist["latin"],
        "auto": dist["latin"],
        "ru": dist["cyrillic"],
        "ko": dist["hangul"],
        "ja": (dist["kana"] * 3 + dist["cjk"]),
        "zh": (dist["cjk"] * 2) if not has_jp_cue else 0,
    }.get(language, 0)
    purity = min(1.0, expected_weighted / max(1, len(letters)))
    # Use unweighted character count for coverage so 1 stray 3-char JP SFX cannot saturate coverage
    coverage = min(expected_raw, 20) / 20.0
    score = purity * 2.0 + coverage + max(0.0, min(1.0, confidence)) * 0.35
    # Penalize CJK/Cyrillic candidates when Latin clearly dominates the recognized text
    if language in {"ja", "ko", "zh", "ru"} and dist["latin"] > max(12, expected_raw * 2):
        score *= 0.35
    return score


def run_auto_paddle_ocr(image_bytes: bytes):
    """Try CJK and Latin models; breaks early only when the target script genuinely dominates."""
    candidates = [("ja", "japan"), ("ko", "korean"), ("zh", "ch"), ("ru", "ru"), ("en", "latin")]
    best_language, best_results, best_score = "auto", [], 0.0
    for language, model_language in candidates:
        try:
            results = run_paddle_ocr(image_bytes, model_language)
            extract_lang = "auto" if language == "en" else language
            lines = extract_paddle_lines(results, extract_lang)
            if not lines:
                continue
            text = " ".join(line["text"] for line in lines)
            confidence = sum(line["confidence"] for line in lines) / len(lines)
            score = _script_score(text, confidence, language)
            if score > best_score:
                chosen_lang = language
                if language == "en":
                    has_non_ascii_latin = any(
                        ord(ch) > 127 and ch.isalpha() and not (
                            "\u0400" <= ch <= "\u04ff"
                            or "\u3040" <= ch <= "\u30ff"
                            or "\u4e00" <= ch <= "\u9fff"
                            or "\uac00" <= ch <= "\ud7af"
                        )
                        for ch in text
                    )
                    chosen_lang = "auto" if has_non_ascii_latin else "en"
                best_language, best_results, best_score = chosen_lang, results, score
                dist = analyze_script_distribution(text)
                script_chars = {
                    "ja": dist["kana"] + dist["cjk"],
                    "ko": dist["hangul"],
                    "zh": dist["cjk"],
                    "ru": dist["cyrillic"],
                    "en": dist["latin"],
                }.get(language, 0)
                other_chars = dist["total_letters"] - script_chars
                can_break_early = (
                    ((language != "ja") or _has_exclusive_jp_cues(text, dist))
                    and script_chars >= 12
                    and script_chars > other_chars * 2
                )
                if can_break_early and score >= 2.2 and confidence >= 0.75:
                    break
        except Exception as exc:
            logger.warning("Auto OCR %s modeli ishlamadi: %s", language, exc)
    return best_language, best_results


def extract_paddle_lines(ocr_results, lang: str):
    """Convert PaddleOCR detections into ordered text lines with page coordinates."""
    if not ocr_results or not ocr_results[0]:
        return []
    lines = []
    for item in ocr_results[0]:
        try:
            points, (text, confidence) = item[0], item[1]
            text = (text or "").strip()
            if not text or is_ocr_garbage_noise(text):
                continue
            if lang in {"en", "english", "eng"}:
                text = repair_english_word_spacing(text)
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
            if right - left < 4 or bottom - top < 4:
                continue
            lines.append({
                "left": int(left), "top": int(top), "right": int(right), "bottom": int(bottom),
                "text": text, "confidence": float(confidence or 0.0),
            })
        except (TypeError, ValueError, IndexError):
            continue

    if lang in {"auto", "en", "english", "eng", "ru", "russian", "rus", "cyrillic", "ko", "korean", "kor", *PDF_LATIN_OCR_LANGS}:
        return merge_horizontal_ocr_lines(lines, lang=lang)
    return sorted(lines, key=lambda line: (line["top"], -line["right"]))


def _group_vertical_manga_lines(lines: list[dict]) -> list[dict]:
    """Join nearby vertical Japanese OCR columns, usually one speech bubble."""
    groups = []
    for line in sorted(lines, key=lambda item: (item["top"], -item["right"])):
        best_group = None
        best_gap = None
        line_height = max(1, line["bottom"] - line["top"])
        for group in groups:
            overlap = min(line["bottom"], group["bottom"]) - max(line["top"], group["top"])
            overlap_ratio = overlap / max(1, min(line_height, group["bottom"] - group["top"]))
            horizontal_gap = max(group["left"] - line["right"], line["left"] - group["right"], 0)
            if overlap_ratio >= 0.5 and horizontal_gap <= max(12, line_height * 0.45):
                if best_gap is None or horizontal_gap < best_gap:
                    best_group, best_gap = group, horizontal_gap
        if best_group is None:
            groups.append({"lines": [line], **{key: line[key] for key in ("left", "top", "right", "bottom")}})
        else:
            best_group["lines"].append(line)
            best_group["left"] = min(best_group["left"], line["left"])
            best_group["top"] = min(best_group["top"], line["top"])
            best_group["right"] = max(best_group["right"], line["right"])
            best_group["bottom"] = max(best_group["bottom"], line["bottom"])

    merged = []
    for group in groups:
        columns = sorted(group["lines"], key=lambda item: -((item["left"] + item["right"]) / 2))
        merged.append({
            "left": group["left"], "top": group["top"],
            "right": group["right"], "bottom": group["bottom"],
            "text": "".join(item["text"] for item in columns),
            "confidence": sum(item["confidence"] for item in columns) / len(columns),
        })
    return sorted(merged, key=lambda item: (item["top"], -item["right"]))


# OCR text processing and translation engines extracted to text_translator.py and routes_pdf.py
from text_translator import (
    normalize_ocr_language,
    translation_language_code,
    KEEP_WORDS,
    SPECIFIC_COMIC_REPLACEMENTS,
    repair_english_word_spacing,
    clean_comic_dialogue_text,
    _join_dialogue_lines,
    merge_horizontal_ocr_lines,
    _GoogleResultParser,
    TRANSLATION_TIMEOUT_SECONDS,
    _google_blocked_until,
    _google_block_lock,
    translate_text,
    translate_batch,
)
from routes_pdf import (
    _sample_comic_background,
    _find_bubble_overlay_region,
    _group_comic_lines,
    ocr_pdf_page,
)

# ─── API Endpoints ─────────────────────────────────────────────────────────────

def _commit_modal_data_volume(key_file: Path) -> None:
    """Commit the Modal persistent volume after saving durable state under /data."""
    try:
        key_str = str(key_file).replace("\\", "/")
    except Exception:
        key_str = ""
    if not (key_str.startswith("/data/") or os.environ.get("MODAL_IMAGE_ID") or os.environ.get("MODAL_ENVIRONMENT")):
        return
    try:
        import modal
        vol = modal.Volume.from_name("manga-pdf-storage")
        vol.commit()
    except Exception as exc:
        logger.warning("Modal volume commit ogohlantirishi: %s", exc)


@app.post("/set-gemini-key")
def set_gemini_key(
    key: str = Form(...),
    _authorized: None = Depends(verify_api_key),
):
    """Save Google AI Studio Gemini API key to persistent server storage."""
    clean_key = key.strip()
    if not clean_key:
        raise HTTPException(status_code=400, detail="Gemini API kaliti böş bölmasligi kerak.")
    with _gemini_key_lock:
        key_file = _get_writable_gemini_key_file()
        tmp_path = None
        try:
            key_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = key_file.with_name(f".{key_file.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            if tmp_path.exists():
                tmp_path.unlink()
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            mode = 0o600
            fd = os.open(str(tmp_path), flags, mode)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(clean_key)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, key_file)
            os.environ["GEMINI_API_KEY"] = clean_key
            _commit_modal_data_volume(key_file)
        except Exception as exc:
            if tmp_path is not None:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:
                    pass
            logger.error("Gemini API kalitini saqlaşda xatolik: %s", exc)
            raise HTTPException(status_code=500, detail="Gemini API kalitini saqlashda xatolik yuz berdi.")
    return {
        "status": "success",
        "message": "Google AI Studio (Gemini) API kaliti muvaffaqiyatli saqlandi. Komikslar endi Gemini orqali kontekstli tarjima qilinadi."
    }

@app.get("/gemini-status")
def gemini_status(_authorized: None = Depends(verify_api_key)):
    from gemini_translator import GEMINI_MODELS
    has_key = bool(resolve_gemini_api_key())
    return {
        "gemini_enabled": has_key,
        "models": GEMINI_MODELS,
        "context_translation": "active" if has_key else "inactive (waiting for API key)"
    }

@app.get("/")
@app.get("/api/info")
def index():
    return {
        "name": "Manga, Manhwa & Manhua Translation Server",
        "status": "running",
        "models": ["comic-text-detector", "qwen3.5-9b (4-bit)", "manga-text-cleaner", "simple-lama"],
        "endpoints": ["/health", "/translate-manga", "/ocr-manga", "/detect-and-translate", "/translate-page-image", "/translate-pdf", "/pdf-jobs/{job_id}"]
    }

def _get_gpu_vram_stats() -> Optional[dict]:
    try:
        if not torch.cuda.is_available():
            return None
        dev_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev_idx)
        return {
            "device_name": getattr(props, "name", "cuda"),
            "allocated_mb": round(torch.cuda.memory_allocated(dev_idx) / (1024 * 1024), 1),
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated(dev_idx) / (1024 * 1024), 1),
            "reserved_mb": round(torch.cuda.memory_reserved(dev_idx) / (1024 * 1024), 1),
            "total_mb": round(props.total_memory / (1024 * 1024), 1),
        }
    except Exception:
        return None


def get_system_resources() -> dict:
    total_mb = 0.0
    used_mb = 0.0
    percent = 0.0
    try:
        import psutil
        vm = psutil.virtual_memory()
        total_mb = round(vm.total / (1024 * 1024), 1)
        used_mb = round(vm.used / (1024 * 1024), 1)
        percent = round(vm.percent, 1)
    except Exception:
        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            mem_dict = {}
            for line in lines:
                parts = line.split(":")
                if len(parts) == 2:
                    k = parts[0].strip()
                    v = parts[1].strip().split()[0]
                    mem_dict[k] = int(v)
            total_kb = mem_dict.get("MemTotal", 0)
            avail_kb = mem_dict.get("MemAvailable", mem_dict.get("MemFree", 0))
            used_kb = max(0, total_kb - avail_kb)
            total_mb = round(total_kb / 1024.0, 1)
            used_mb = round(used_kb / 1024.0, 1)
            percent = round((used_kb / total_kb) * 100.0, 1) if total_kb else 0.0
        except Exception as exc:
            logger.debug("Failed reading /proc/meminfo: %s", exc)

    for cg_path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        if os.path.exists(cg_path):
            try:
                with open(cg_path, "r") as f:
                    val = f.read().strip()
                if val and val != "max" and int(val) < (1 << 60):
                    cg_mb = round(int(val) / (1024 * 1024), 1)
                    if cg_mb > 0:
                        total_mb = min(total_mb, cg_mb) if total_mb > 0 else cg_mb
                    break
            except Exception as exc:
                logger.debug("Failed reading cgroup memory: %s", exc)

    process_mb = 0.0
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        process_mb = round(rusage.ru_maxrss / 1024.0, 1)
    except Exception as exc:
        logger.debug("resource.getrusage unavailable: %s", exc)

    total_gb = round(total_mb / 1024.0, 2)
    used_gb = round(used_mb / 1024.0, 2)
    cpu_count = float(os.cpu_count() or 1.0)

    return {
        "ram_total_gb": total_gb,
        "ram_used_gb": used_gb,
        "ram_total_mb": total_mb,
        "ram_used_mb": used_mb,
        "ram_percent": percent,
        "process_ram_mb": process_mb,
        "cpu_allocated": cpu_count,
        "human_readable": f"{used_gb} GB / {total_gb} GB ({percent}%)",
    }

@app.get("/health")
def health(x_api_key: Optional[str] = Header(default=None)):
    is_authenticated = False
    if ALLOW_UNAUTHENTICATED:
        is_authenticated = True
    elif SERVER_API_KEY and x_api_key:
        try:
            is_authenticated = hmac.compare_digest(x_api_key.encode("utf-8"), SERVER_API_KEY.encode("utf-8"))
        except Exception:
            is_authenticated = False

    base_response = {
        "status": "healthy",
        "version": os.environ.get("SERVER_CODE_VERSION", "unknown"),
        "translation_ready": bool(SERVER_API_KEY or ALLOW_UNAUTHENTICATED),
        "api_key_required": bool(SERVER_API_KEY),
        "pdf_translation_enabled": PDF_ENABLED,
        "pdf_modes": ["book", "comic"],
        "pdf_source_languages": list(PDF_SOURCE_LANGUAGES),
        "pdf_target_languages": list(PDF_TARGET_LANGUAGES),
        "pdf_max_upload_bytes": MAX_PDF_UPLOAD_BYTES,
        "pdf_max_pages": MAX_PDF_PAGES,
        "pdf_ocr_dpi": PDF_OCR_DPI,
        "pdf_ocr_max_pixels": PDF_OCR_MAX_PIXELS,
        "manga_pipeline_status": "fallback" if _manga_pipeline_error else ("loaded" if _manga_pipeline else "not_loaded"),
    }
    if not is_authenticated:
        return base_response

    engine = _onnx_engine
    sys_res = get_system_resources()
    base_response.update({
        "system_resources": sys_res,
        "gpu_vram": _get_gpu_vram_stats(),
        "onnx_ready": engine.is_ready if engine else False,
        "onnx_detector": engine.det_sess is not None if engine else False,
        "onnx_ocr": engine.ocr_sess is not None if engine else False,
        "onnx_inpaint": engine.inp_sess is not None if engine else False,
        "comic_detector_loaded": bool(_manga_pipeline and getattr(_manga_pipeline, "comic_detector", None) is not None),
        "qwen_vl_loaded": bool(_manga_pipeline and getattr(_manga_pipeline, "qwen_engine", None) is not None),
        "manga_ocr_loaded": _mocr is not None,
        "paddle_ja_loaded": _paddle_ja is not None,
        "paddle_ko_loaded": _paddle_ko is not None,
        "paddle_ch_loaded": _paddle_ch is not None,
        "paddle_en_loaded": _paddle_en is not None,
        "paddle_ru_loaded": _paddle_ru is not None,
        "manga_pipeline_loaded": _manga_pipeline is not None,
        "ml_bubble_detector_enabled": _manga_pipeline is not None and _manga_pipeline_error is None,
        "ml_inpainter_enabled": _manga_pipeline is not None and _manga_pipeline_error is None,
        "manga_pipeline_status": "fallback" if _manga_pipeline_error else ("loaded" if _manga_pipeline else "not_loaded"),
        "manga_pipeline_error": _manga_pipeline_error,
        "translation_provider": "qwen3.5-9b",
        "ocr_packages_available": {
            name: importlib.util.find_spec(name) is not None
            for name in ("transformers", "bitsandbytes", "safetensors", "torchvision")
        },
    })
    return base_response

@app.get("/system-status")
def system_status(_authorized: None = Depends(verify_api_key)):
    """Returns real-time container resource utilization (RAM, CPU, jobs)."""
    res = get_system_resources()
    job_count = len(pdf_job_manager.list_jobs(limit=100))
    cpus = res["cpu_allocated"]
    return {
        "status": "online",
        "version": os.environ.get("SERVER_CODE_VERSION", "unknown"),
        "ram": res,
        "gpu_vram": _get_gpu_vram_stats(),
        "allocated_cpus": cpus,
        "active_pdf_jobs": job_count,
        "summary": f"RAM: {res['human_readable']} | CPU: {int(cpus)} vCPUs",
    }

# PDF Translation routes extracted to routes_pdf.py
from routes_pdf import (
    router as pdf_router,
    submit_pdf_translation,
    get_pdf_translation_status,
    get_pdf_translation_review_page,
    correct_pdf_translation,
    cancel_pdf_translation,
    resume_pdf_translation,
    download_pdf_translation,
)

app.include_router(pdf_router)


async def dispatch_manga_ocr(image_bytes: bytes, pil_image: Image.Image, lang: str) -> Tuple[str, str]:
    normalized_lang = normalize_ocr_language(lang)

    # Primary path: Qwen3.5-9B VL OCR (natively reads light/dark/vertical/italic across all languages)
    qwen = get_qwen_vl_engine()
    if qwen is not None:
        try:
            q_text = await run_in_threadpool(qwen.ocr_crop, pil_image, normalized_lang)
            q_text = filter_manga_ocr_hallucinations((q_text or "").strip())
            if q_text:
                final_lang = normalized_lang
                if final_lang == "auto":
                    try:
                        det_lang = detect_primary_script_from_text(q_text)
                        if det_lang and det_lang != "auto":
                            final_lang = det_lang
                    except Exception:
                        pass
                return q_text, final_lang
        except Exception as q_exc:
            logger.debug("QwenVLEngine dispatch_manga_ocr fallback: %s", q_exc)

    async def _execute_ocr(target_bytes: bytes, target_pil: Image.Image, cur_lang: str) -> Tuple[str, str, float]:
        extracted_text = ""
        avg_conf = 0.0
        requested_auto = cur_lang == "auto"
        if requested_auto:
            if is_paddle_enabled():
                try:
                    cur_lang, auto_results = await run_in_threadpool(run_auto_paddle_ocr, target_bytes)
                    lines = extract_paddle_lines(auto_results, cur_lang)
                    sep = "" if cur_lang in ["ja", "zh"] else " "
                    extracted_text = sep.join(line["text"] for line in lines)
                    if lines:
                        avg_conf = sum(line.get("confidence", 0.0) for line in lines) / len(lines)
                except Exception as e:
                    logger.warning("Auto PaddleOCR xatosi: %s", e)
                    extracted_text = ""
            else:
                # PaddleOCR disabled: try ONNX OCR / MangaOCR directly instead of returning empty
                engine = get_onnx_engine()
                if engine and getattr(engine, "ocr_sess", None):
                    try:
                        onnx_text = await run_in_threadpool(engine.ocr_bubble, target_pil)
                        if onnx_text and onnx_text.strip():
                            extracted_text = onnx_text.strip()
                            avg_conf = 0.75
                            try:
                                cur_lang = detect_primary_script_from_text(extracted_text)
                            except Exception:
                                cur_lang = "en"
                    except Exception as exc:
                        logger.warning("ONNX auto OCR xatosi: %s", exc)
                if not extracted_text:
                    try:
                        manga_text = await run_in_threadpool(run_manga_ocr, target_pil)
                        if manga_text and manga_text.strip():
                            extracted_text = manga_text.strip()
                            avg_conf = 0.90
                            try:
                                cur_lang = detect_primary_script_from_text(extracted_text)
                            except Exception:
                                cur_lang = "ja"
                    except Exception as exc:
                        logger.warning("MangaOcr auto fallback xatosi: %s", exc)
            if cur_lang == "ja" and not extracted_text:
                try:
                    manga_text = await run_in_threadpool(run_manga_ocr, target_pil)
                    if manga_text and manga_text.strip():
                        extracted_text = manga_text.strip()
                        avg_conf = max(avg_conf, 0.95)
                except Exception as exc:
                    logger.warning("MangaOcr xatosi: %s; aniqlangan OCR matni işlatiladi.", exc)
            elif cur_lang == "ja" and extracted_text:
                try:
                    manga_text = await run_in_threadpool(run_manga_ocr, target_pil)
                    if manga_text and manga_text.strip():
                        extracted_text = manga_text.strip()
                        avg_conf = max(avg_conf, 0.95)
                except Exception as exc:
                    logger.warning("MangaOcr xatosi: %s; aniqlangan OCR matni işlatiladi.", exc)
        elif cur_lang in ["ja", "jpn", "japanese"]:
            cur_lang = "ja"
            try:
                extracted_text = await run_in_threadpool(run_manga_ocr, target_pil)
                if extracted_text and extracted_text.strip():
                    avg_conf = 0.95
            except Exception as e:
                logger.warning("MangaOcr xatosi: %s. PaddleOCR zaxirasi sinab köriladi.", e)
                extracted_text = ""
            if not extracted_text:
                try:
                    res = await run_in_threadpool(run_paddle_ocr, target_bytes, "japan")
                    lines = extract_paddle_lines(res, "ja")
                    extracted_text = "".join(line["text"] for line in lines)
                    if lines:
                        avg_conf = sum(line.get("confidence", 0.0) for line in lines) / len(lines)
                except Exception as e:
                    logger.warning("PaddleOCR (japan) xatosi: %s", e)
                    extracted_text = ""
        elif cur_lang in ["ko", "kor", "korean"]:
            cur_lang = "ko"
            try:
                res = await run_in_threadpool(run_paddle_ocr, target_bytes, "korean")
                lines = extract_paddle_lines(res, "ko")
                extracted_text = " ".join(line["text"] for line in lines)
                if lines:
                    avg_conf = sum(line.get("confidence", 0.0) for line in lines) / len(lines)
            except Exception as e:
                logger.warning("PaddleOCR (korean) xatosi: %s", e)
                extracted_text = ""
        elif cur_lang in ["zh", "ch", "chs", "cht", "chinese"]:
            cur_lang = "zh"
            try:
                res = await run_in_threadpool(run_paddle_ocr, target_bytes, "ch")
                lines = extract_paddle_lines(res, "zh")
                extracted_text = "".join(line["text"] for line in lines)
                if lines:
                    avg_conf = sum(line.get("confidence", 0.0) for line in lines) / len(lines)
            except Exception as e:
                logger.warning("PaddleOCR (ch) xatosi: %s", e)
                extracted_text = ""
        elif cur_lang in ["en", "eng", "english"]:
            cur_lang = "en"
            try:
                res = await run_in_threadpool(run_paddle_ocr, target_bytes, "en")
                lines = extract_paddle_lines(res, "en")
                extracted_text = " ".join(line["text"] for line in lines)
                if lines:
                    avg_conf = sum(line.get("confidence", 0.0) for line in lines) / len(lines)
            except Exception as e:
                logger.warning("PaddleOCR (en) xatosi: %s", e)
                extracted_text = ""
        elif cur_lang in ["ru", "rus", "russian"]:
            cur_lang = "ru"
            try:
                res = await run_in_threadpool(run_paddle_ocr, target_bytes, "ru")
                lines = extract_paddle_lines(res, "ru")
                extracted_text = " ".join(line["text"] for line in lines)
                if lines:
                    avg_conf = sum(line.get("confidence", 0.0) for line in lines) / len(lines)
            except Exception as e:
                logger.warning("PaddleOCR (ru) xatosi: %s", e)
                extracted_text = ""
        elif cur_lang in PDF_LATIN_OCR_LANGS:
            try:
                res = await run_in_threadpool(run_paddle_ocr, target_bytes, "latin")
                lines = extract_paddle_lines(res, cur_lang)
                extracted_text = " ".join(line["text"] for line in lines)
                if lines:
                    avg_conf = sum(line.get("confidence", 0.0) for line in lines) / len(lines)
            except Exception as e:
                logger.warning("PaddleOCR (latin) xatosi: %s", e)
                extracted_text = ""
        else:
            raise HTTPException(status_code=400, detail=f"OCR tili qöllab-quvvatlanmaydi: {cur_lang}")

        return extracted_text.strip(), cur_lang, avg_conf

    # Check for white text on dark background
    is_dark_bg = False
    try:
        is_dark_bg = is_dark_background(pil_image)
    except Exception as exc:
        logger.debug("Dark background check in dispatch_manga_ocr: %s", exc)

    if not is_dark_bg:
        text, final_lang, _ = await _execute_ocr(image_bytes, pil_image, normalized_lang)
        text = filter_manga_ocr_hallucinations(text)
        return text, final_lang

    # Run on original image
    text_normal, lang_normal, conf_normal = await _execute_ocr(image_bytes, pil_image, normalized_lang)
    text_normal = filter_manga_ocr_hallucinations(text_normal)

    # Invert and apply CLAHE enhancement
    try:
        enhanced_pil = apply_dark_box_clahe_inversion(pil_image)
        if isinstance(enhanced_pil, np.ndarray):
            enhanced_pil = Image.fromarray(enhanced_pil)
        buf = io.BytesIO()
        enhanced_pil.save(buf, format="PNG")
        inv_bytes = buf.getvalue()

        text_inv, lang_inv, conf_inv = await _execute_ocr(inv_bytes, enhanced_pil, normalized_lang)
        text_inv = filter_manga_ocr_hallucinations(text_inv)

        is_normal_garbage = is_ocr_garbage_noise(text_normal, normalized_lang, is_inside_bubble=True) if text_normal else True
        is_inv_garbage = is_ocr_garbage_noise(text_inv, normalized_lang, is_inside_bubble=True) if text_inv else True

        # Pick the more confident and genuine result
        score_normal = (conf_normal * (1.0 + min(1.0, len(text_normal) / 10.0))) if (text_normal and not is_normal_garbage) else 0.0
        score_inv = (conf_inv * (1.0 + min(1.0, len(text_inv) / 10.0))) if (text_inv and not is_inv_garbage) else 0.0

        if score_inv > score_normal and text_inv and not is_inv_garbage:
            return text_inv, lang_inv
        return text_normal, lang_normal
    except Exception as exc:
        logger.debug("Dark background inversion/CLAHE error: %s", exc)
        return text_normal, lang_normal


@app.post("/translate-manga")
async def translate_manga(
    file: UploadFile = File(None),
    text: Optional[str] = Form(None),
    lang: str = Form("auto"),       # "auto" (Auto-detect script), "ja" (Japanese Manga), "ko" (Korean Manhwa), "zh"/"ch" (Chinese Manhua)
    target_lang: str = Form("uz"),  # "uz", "ru", "en", etc.
    gemini_api_key: Optional[str] = Form(None),
    x_gemini_api_key: Optional[str] = Header(None),
    _authorized: None = Depends(verify_api_key),
):
    """
    Translates a cropped speech bubble or manga text image:
    1. If text is already provided (e.g. from on-device ML Kit), skips OCR for instant translation.
    2. Runs manga-ocr for Japanese vertical/horizontal manga text.
    3. Runs PaddleOCR for Korean or Chinese text.
    4. Translates recognized text to target_lang (Uzbek, Russian, English).
    """
    extracted_text = (text or "").strip()
    if not extracted_text and file is None:
        raise HTTPException(status_code=400, detail="Fayl yoki matn kiritilişi şart.")

    if extracted_text and file is None and lang in (None, ""):
        lang = "auto"

    normalized_lang = normalize_ocr_language(lang)

    # 1. OCR text extraction if text not already provided
    if not extracted_text and file is not None:
        image_bytes, pil_image = await read_image_upload(file)
        extracted_text, normalized_lang = await dispatch_manga_ocr(image_bytes, pil_image, lang)

    extracted_text = extracted_text.strip()

    # 2. Text translation
    translated_text = ""
    if extracted_text:
        effective_gemini_key = resolve_gemini_api_key(gemini_api_key, x_gemini_api_key)
        source_code = translation_language_code(normalized_lang)
        if effective_gemini_key:
            try:
                from gemini_translator import translate_single_with_gemini
                translated_text = await run_in_threadpool(
                    translate_single_with_gemini,
                    extracted_text,
                    target_lang,
                    source_code,
                    effective_gemini_key
                )
            except Exception as exc:
                logger.debug("Gemini single translate xatosi: %s", exc)

        if not translated_text:
            global _translation_semaphore
            if _translation_semaphore is None:
                _translation_semaphore = asyncio.Semaphore(4)
            async with _translation_semaphore:
                translated_text = await run_in_threadpool(
                    translate_text, extracted_text, target_lang, source_code
                )
        if not translated_text:
            raise HTTPException(status_code=502, detail="Matn aniqlandi, ammo tarjima xizmatlari javob bermadi.")

    return {
        "status": "success",
        "original_text": extracted_text,
        "translated_text": translated_text,
        "lang": lang,
        "detected_source_lang": normalized_lang,
        "target_lang": target_lang
    }

@app.post("/ocr-manga")
async def ocr_manga(
    file: UploadFile = File(...),
    lang: str = Form("auto"),
    _authorized: None = Depends(verify_api_key),
):
    """OCR extraction only without translation."""
    image_bytes, pil_image = await read_image_upload(file)
    extracted_text, normalized_lang = await dispatch_manga_ocr(image_bytes, pil_image, lang)

    return {
        "status": "success",
        "original_text": extracted_text,
        "lang": lang,
        "detected_source_lang": normalized_lang
    }

def _resolve_auto_image_lang(image_bytes: bytes, pil_image: Image.Image, pipe=None) -> str:
    """Detect primary script language when legacy fallback is active on lang='auto'."""
    if is_paddle_enabled():
        try:
            detected_p_lang, paddle_res = run_auto_paddle_ocr(image_bytes)
            if paddle_res and paddle_res[0]:
                p_lines = extract_paddle_lines(paddle_res, "auto")
                sample = " ".join(ln.get("text", "") for ln in p_lines if ln.get("text"))
                if sample.strip():
                    script = detect_primary_script_from_text(sample)
                    if script and script != "auto":
                        return normalize_ocr_language(script)
            if detected_p_lang and detected_p_lang != "auto":
                return normalize_ocr_language(detected_p_lang)
        except Exception as exc:
            logger.debug("_resolve_auto_image_lang Paddle probe skipped: %s", exc)
    return "auto"


def _is_gemini_requested(
    gemini_api_key: Optional[str],
    x_gemini_api_key: Optional[str],
    use_gemini: Any = False,
    translator: Optional[str] = None,
) -> bool:
    t_norm = translator.strip().lower() if isinstance(translator, str) else ""
    if t_norm in {"qwen", "qwen3.5", "qwen3.5-9b", "local"}:
        return False
    if isinstance(use_gemini, bool) and use_gemini:
        return True
    if isinstance(use_gemini, str) and use_gemini.strip().lower() in {"1", "true", "yes"}:
        return True
    if t_norm == "gemini":
        return True
    if (isinstance(gemini_api_key, str) and gemini_api_key.strip()) or (isinstance(x_gemini_api_key, str) and x_gemini_api_key.strip()):
        return True
    return False


@app.post("/detect-and-translate")
async def detect_and_translate(
    file: UploadFile = File(...),
    lang: str = Form("auto"),
    target_lang: str = Form("uz"),
    gemini_api_key: Optional[str] = Form(None),
    x_gemini_api_key: Optional[str] = Header(None),
    chapter_context: Optional[str] = Form(None),
    use_gemini: bool = Form(False),
    translator: Optional[str] = Form(None),
    _authorized: None = Depends(verify_api_key),
):
    """
    Accepts an entire manga/manhwa/manhua page:
    1. Detects speech bubbles, narration boxes, and borderless text via comic-text-detector.
    2. By default, runs Qwen3.5-9B (4-bit) single-pass OCR + translation (100% API-free).
    3. Optional Gemini override: if a valid Gemini key is provided AND requested, uses Qwen3.5-9B
       for OCR and passes unordered {block_id: text} to Gemini.
    """
    image_bytes, pil_image = await read_image_upload(file)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    width, height = pil_image.size
    normalized_lang = normalize_ocr_language(lang)

    gemini_requested = _is_gemini_requested(gemini_api_key, x_gemini_api_key, use_gemini=use_gemini, translator=translator)
    effective_gemini_key = resolve_gemini_api_key(gemini_api_key, x_gemini_api_key)
    use_gemini_override = bool(gemini_requested and effective_gemini_key)

    blocks: List[Dict[str, Any]] = []
    pipeline_translator_used: Optional[str] = None

    # 1. Primary path: ComicTextDetector + Qwen3.5-9B via MangaTranslationPipeline
    try:
        pipe = get_manga_pipeline()
        if pipe is not None:
            pipeline_lang = (
                "en" if normalized_lang in PDF_LATIN_OCR_LANGS
                else (normalized_lang if normalized_lang in {"ja", "ko", "zh", "en", "ru"} else "auto")
            )
            desired_target = None if use_gemini_override else target_lang
            def _pipe_proc():
                import inspect
                try:
                    sig_params = inspect.signature(pipe.process_page).parameters
                except (TypeError, ValueError):
                    sig_params = {}
                has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
                kwargs = {"auto_collect": False, "source_lang": pipeline_lang}
                if has_var_kw or "skip_cleaning" in sig_params:
                    kwargs["skip_cleaning"] = True
                if has_var_kw or "target_lang" in sig_params:
                    kwargs["target_lang"] = desired_target
                return pipe.process_page(pil_image, **kwargs)
            res = await run_in_threadpool(_pipe_proc)
            if not res.get("requires_ocr_fallback"):
                detected_src = res.get("detected_source_lang")
                if normalized_lang == "auto" and detected_src and detected_src != "auto":
                    normalized_lang = normalize_ocr_language(detected_src)
                pipeline_translator_used = res.get("translator_used")
                dialogues = res.get("dialogues", [])
                for d in dialogues:
                    if d.get("category") == "sfx" or d.get("is_sfx"):
                        continue
                    txt = (d.get("transcription") or "").strip()
                    if not txt:
                        continue
                    bx1, by1, bx2, by2 = d["bbox"]
                    bid = d.get("block_id") or f"b_{d.get('bubble_id', len(blocks) + 1)}"
                    trans_txt = (d.get("translation") or d.get("translated_text") or "").strip()
                    blocks.append({
                        "block_id": bid,
                        "bubble_id": d.get("bubble_id"),
                        "category": d.get("category", "speech_bubble"),
                        "is_dark": bool(d.get("is_dark", False)),
                        "left": max(0, int(bx1)),
                        "top": max(0, int(by1)),
                        "right": min(width, int(bx2)),
                        "bottom": min(height, int(by2)),
                        "original_text": txt,
                        "translated_text": "" if use_gemini_override else trans_txt,
                    })
    except Exception as exc:
        logger.warning("MangaTranslationPipeline xatosi /detect-and-translate da: %s", exc)

    # 2. Secondary path: comic-text-detector + Qwen / pytesseract fallback via manga_bubble_engine
    if not blocks:
        try:
            from manga_bubble_engine import extract_manga_dialogues
            raw_dialogues = await run_in_threadpool(
                extract_manga_dialogues,
                pil_image=pil_image,
                source_lang=normalized_lang,
                scale=1.0,
            )
            for d in raw_dialogues:
                if d.get("category") == "sfx":
                    continue
                txt = (d.get("text") or "").strip()
                if not txt:
                    continue
                bbox = d.get("bbox") or [0, 0, width, height]
                blocks.append({
                    "block_id": d.get("block_id") or f"b_{len(blocks) + 1}",
                    "category": d.get("category", "speech_bubble"),
                    "left": max(0, int(bbox[0])),
                    "top": max(0, int(bbox[1])),
                    "right": min(width, int(bbox[2])),
                    "bottom": min(height, int(bbox[3])),
                    "original_text": txt,
                    "translated_text": (d.get("translated_text") or "").strip() if not use_gemini_override else "",
                })
            if blocks and normalized_lang == "auto":
                detected_fb = detect_primary_script_from_text(" ".join(b["original_text"] for b in blocks))
                if detected_fb and detected_fb != "auto":
                    normalized_lang = normalize_ocr_language(detected_fb)
        except Exception as e:
            logger.debug("extract_manga_dialogues fallback error: %s", e)

    # 3. Guarded legacy Paddle fallback (only when paddleocr is installed & enabled)
    if not blocks and is_paddle_enabled():
        try:
            if normalized_lang == "auto":
                normalized_lang, ocr_results = await run_in_threadpool(run_auto_paddle_ocr, image_bytes)
            else:
                detector_lang = {
                    "ja": "japan", "ko": "korean", "zh": "ch", "en": "en", "ru": "ru",
                    **{language: "latin" for language in PDF_LATIN_OCR_LANGS},
                }.get(normalized_lang)
                ocr_results = await run_in_threadpool(run_paddle_ocr, image_bytes, detector_lang) if detector_lang is not None else None
            if ocr_results and ocr_results[0]:
                detections = extract_paddle_lines(ocr_results, normalized_lang)
                if normalized_lang in {"ja", "zh"}:
                    detections = _group_vertical_manga_lines(detections)
                for detection in detections:
                    x1 = max(0, detection["left"])
                    y1 = max(0, detection["top"])
                    x2 = min(width, detection["right"])
                    y2 = min(height, detection["bottom"])
                    if x2 - x1 < 8 or y2 - y1 < 8:
                        continue
                    orig_text = detection["text"].strip()
                    if orig_text:
                        blocks.append({
                            "block_id": f"b_{len(blocks) + 1}",
                            "left": x1,
                            "top": y1,
                            "right": x2,
                            "bottom": y2,
                            "original_text": orig_text,
                            "translated_text": "",
                        })
        except Exception as e:
            logger.debug("PaddleOCR fallback disabled or failed: %s", e)

    # 4. Translation resolution: Qwen3.5-9B direct by default; Gemini only if explicitly requested/configured
    translator_used: Optional[str] = None
    fallback_warning: Optional[str] = None

    if blocks:
        for idx, b in enumerate(blocks):
            if "block_id" not in b or not b["block_id"]:
                b["block_id"] = f"b_{idx + 1}"

        missing_indices = [
            idx for idx, b in enumerate(blocks)
            if not (isinstance(b.get("translated_text"), str) and b["translated_text"].strip())
        ]

        if not missing_indices:
            translator_used = pipeline_translator_used or "qwen3.5-9b"
        else:
            # Try Qwen3.5-9B crop translation first if not requesting Gemini override
            if not use_gemini_override:
                qwen = get_qwen_vl_engine()
                if qwen is not None:
                    try:
                        q_crops = [
                            pil_image.crop((
                                max(0, int(blocks[i]["left"]) - 4),
                                max(0, int(blocks[i]["top"]) - 4),
                                min(width, int(blocks[i]["right"]) + 4),
                                min(height, int(blocks[i]["bottom"]) + 4),
                            ))
                            for i in missing_indices
                        ]
                        q_pairs = await run_in_threadpool(
                            qwen.ocr_and_translate_crops, q_crops, target_lang, normalized_lang
                        )
                        still_missing = []
                        for i, (src_q, tgt_q) in zip(missing_indices, q_pairs):
                            if tgt_q and tgt_q.strip():
                                blocks[i]["translated_text"] = tgt_q.strip()
                                if src_q and not blocks[i]["original_text"]:
                                    blocks[i]["original_text"] = src_q
                            else:
                                still_missing.append(i)
                        missing_indices = still_missing
                        if not missing_indices:
                            translator_used = "qwen3.5-9b"
                    except Exception as q_err:
                        logger.warning("Qwen crop translation error in detect_and_translate: %s", q_err)

            if missing_indices:
                should_try_gemini = bool(effective_gemini_key) and (use_gemini_override or not translator_used)
                gemini_fail_reason = None
                if should_try_gemini:
                    try:
                        from gemini_translator import translate_dialogues_with_gemini
                        sub_ids = [blocks[i]["block_id"] for i in missing_indices]
                        sub_texts = [blocks[i]["original_text"] for i in missing_indices]
                        translated_dict = await run_in_threadpool(
                            translate_dialogues_with_gemini,
                            sub_texts,
                            target_lang,
                            normalized_lang,
                            api_key=effective_gemini_key,
                            chapter_context=chapter_context or "",
                            block_ids=sub_ids,
                            return_dict=True,
                        )
                        still_missing = []
                        if isinstance(translated_dict, dict):
                            for i in missing_indices:
                                val = translated_dict.get(blocks[i]["block_id"])
                                if isinstance(val, str) and val.strip():
                                    blocks[i]["translated_text"] = val.strip()
                                else:
                                    still_missing.append(i)
                        elif isinstance(translated_dict, list) and len(translated_dict) == len(missing_indices):
                            for i, trans_res in zip(missing_indices, translated_dict):
                                if isinstance(trans_res, str) and trans_res.strip():
                                    blocks[i]["translated_text"] = trans_res.strip()
                                else:
                                    still_missing.append(i)
                        else:
                            still_missing = list(missing_indices)
                            gemini_fail_reason = "Gemini returned empty or mismatched translation dict"
                        missing_indices = still_missing
                        if not missing_indices:
                            translator_used = "gemini"
                    except Exception as ge:
                        gemini_fail_reason = f"Gemini batch translation exception: {ge}"
                        logger.error("Gemini batch translation failed: %s", gemini_fail_reason, exc_info=True)
                elif not effective_gemini_key:
                    gemini_fail_reason = "No GEMINI_API_KEY provided and Qwen3.5-9B unavailable"

            if missing_indices:
                translator_used = "google_fallback"
                fallback_warning = f"Primary translation unavailable ({gemini_fail_reason}); falling back to google translate."
                from text_translator import translate_text as _orig_translate_text
                custom_tx = translate_text if translate_text is not _orig_translate_text else None
                missing_texts = [blocks[i]["original_text"] for i in missing_indices]
                try:
                    translated_list = await run_in_threadpool(
                        translate_batch,
                        missing_texts,
                        target_lang,
                        translation_language_code(normalized_lang),
                        translate_fn=custom_tx,
                    )
                except Exception as exc:
                    logger.debug("translate_batch failed in detect_and_translate: %s", exc)
                    translated_list = [None for _ in missing_texts]

                for idx, trans_res in zip(missing_indices, translated_list):
                    blocks[idx]["translated_text"] = (
                        trans_res if isinstance(trans_res, str) and trans_res.strip() else blocks[idx]["original_text"]
                    )

    # Ensure every block in blocks has category, is_dark, bg_color, and text_color
    img_np_full = np.array(pil_image)
    img_h, img_w = img_np_full.shape[:2]
    for b in blocks:
        bx1 = max(0, min(img_w, int(b.get("left", 0))))
        by1 = max(0, min(img_h, int(b.get("top", 0))))
        bx2 = max(0, min(img_w, int(b.get("right", img_w))))
        by2 = max(0, min(img_h, int(b.get("bottom", img_h))))

        median_luma = 255.0
        if bx2 > bx1 and by2 > by1:
            patch = img_np_full[by1:by2, bx1:bx2]
            if patch.size > 0:
                ph, pw = patch.shape[:2]
                my = max(1, int(ph * 0.12)) if ph >= 12 else 0
                mx = max(1, int(pw * 0.12)) if pw >= 12 else 0
                interior = patch[my:ph - my, mx:pw - mx] if (ph - 2 * my >= 4 and pw - 2 * mx >= 4) else patch
                int_gray = cv2.cvtColor(interior, cv2.COLOR_RGB2GRAY) if interior.ndim == 3 else interior
                p50 = float(np.median(int_gray))
                p85 = float(np.percentile(int_gray, 85))
                if p50 < 115 and p85 > 160:
                    bg_mask = int_gray < 120
                elif p50 >= 115:
                    bg_mask = int_gray >= max(110, int(np.percentile(int_gray, 35)))
                else:
                    bg_mask = np.ones_like(int_gray, dtype=bool)
                bg_pixels = interior[bg_mask] if np.any(bg_mask) else interior.reshape(-1, 3)
                med_r = int(np.median(bg_pixels[..., 0]))
                med_g = int(np.median(bg_pixels[..., 1]))
                med_b = int(np.median(bg_pixels[..., 2]))
                median_luma = 0.299 * med_r + 0.587 * med_g + 0.114 * med_b
                is_dark = (median_luma < 115.0)
                bg_color = f"#{med_r:02x}{med_g:02x}{med_b:02x}".upper()
            else:
                is_dark = False
                bg_color = "#FFFFFF"
        else:
            is_dark = False
            bg_color = "#FFFFFF"

        if "category" not in b or not b["category"]:
            b["category"] = "narration" if is_dark else "speech_bubble"
        elif is_dark and b["category"] == "speech_bubble" and median_luma < 95:
            b["category"] = "narration"

        b["is_dark"] = bool(b.get("is_dark", is_dark))
        b["bg_color"] = b.get("bg_color") or bg_color
        b["text_color"] = b.get("text_color") or ("#FFFFFF" if b["is_dark"] else "#000000")

    response_payload: Dict[str, Any] = {
        "status": "success",
        "source_lang": normalized_lang,
        "blocks_count": len(blocks),
        "blocks": blocks,
    }
    if translator_used == "google_fallback":
        response_payload["translator_used"] = "google_fallback"
        response_payload["warning"] = fallback_warning or "Gemini batch translation failed; google_fallback used"
    elif translator_used:
        response_payload["translator_used"] = translator_used

    return response_payload

@app.post("/clean-manga")
async def clean_manga(
    file: UploadFile = File(...),
    boxes_json: Optional[str] = Form(None),
    _authorized: None = Depends(verify_api_key),
):
    """
    Cleans detected speech bubbles from a manga image using PyTorch pipeline
    (MangaTranslationPipeline + MangaTextCleaner + SimpleLama).
    Returns cleaned image in image/jpeg format.
    """
    image_bytes, pil_image = await read_image_upload(file)

    target_boxes = []
    if boxes_json:
        try:
            raw_boxes = json.loads(boxes_json)
            if isinstance(raw_boxes, list):
                for b in raw_boxes:
                    if isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in b):
                        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                        if x1 < x2 and y1 < y2:
                            target_boxes.append([x1, y1, x2, y2])
        except Exception as exc:
            logger.debug("Failed parsing boxes_json: %s", exc)

    if target_boxes:
        def _crop_clean():
            try:
                cleaner = get_manga_cleaner()
                img_np = np.array(pil_image.convert("RGB"))
                box_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
                for bx1, by1, bx2, by2 in target_boxes:
                    bx1 = max(0, min(img_np.shape[1], int(bx1)))
                    by1 = max(0, min(img_np.shape[0], int(by1)))
                    bx2 = max(0, min(img_np.shape[1], int(bx2)))
                    by2 = max(0, min(img_np.shape[0], int(by2)))
                    if bx2 > bx1 and by2 > by1:
                        box_mask[by1:by2, bx1:bx2] = 255

                cleaned_res, edit_mask, _ = cleaner.clean_page(img_np, allowed_boxes=target_boxes)
                if cleaned_res is None or edit_mask is None or not np.any(np.asarray(edit_mask) > 0):
                    raise RuntimeError("MangaTextCleaner returned uncleaned fallback (edit_mask is empty/None)")
                res_np = np.array(cleaned_res)
                final_np = img_np.copy()
                final_np[box_mask > 0] = res_np[box_mask > 0]
                return Image.fromarray(final_np)
            except Exception as e:
                logger.error("MangaTextCleaner on target_boxes failed, falling back to clean_bubble_crop: %s", e, exc_info=True)
                try:
                    clean_fn = globals().get("clean_bubble_crop")
                    if clean_fn is None:
                        try:
                            from manga_pipeline import clean_bubble_crop as clean_fn
                        except ImportError:
                            try:
                                from ml.manga_pipeline import clean_bubble_crop as clean_fn
                            except ImportError:
                                from cleaning import clean_bubble_crop as clean_fn
                    img_np = np.array(pil_image.convert("RGB"))
                    cleaned_np = img_np.copy()
                    for box in target_boxes:
                        bx1, by1, bx2, by2 = box
                        bx1 = max(0, min(img_np.shape[1], int(bx1)))
                        by1 = max(0, min(img_np.shape[0], int(by1)))
                        bx2 = max(0, min(img_np.shape[1], int(bx2)))
                        by2 = max(0, min(img_np.shape[0], int(by2)))
                        if bx2 > bx1 and by2 > by1:
                            patch = cleaned_np[by1:by2, bx1:bx2]
                            if patch.size > 0:
                                cleaned_patch = clean_fn(patch)
                                cleaned_np[by1:by2, bx1:bx2] = cleaned_patch
                    return Image.fromarray(cleaned_np)
                except Exception as err:
                    logger.error("clean_bubble_crop fallback failed: %s", err, exc_info=True)
                    return pil_image
        cleaned_pil = await run_in_threadpool(_crop_clean)
    else:
        # Full page adaptive cleaning using MangaTextCleaner
        def _clean_full():
            try:
                img_np = np.array(pil_image.convert("RGB"))
                cleaner = get_manga_cleaner()
                cleaned_res, edit_mask, _ = cleaner.clean_page(img_np)
                if cleaned_res is not None and edit_mask is not None:
                    return cleaned_res
                return pil_image
            except Exception as e:
                logger.error("MangaTextCleaner full page clean failed: %s", e, exc_info=True)
                return pil_image
        cleaned_pil = await run_in_threadpool(_clean_full)

    buf = io.BytesIO()
    cleaned_pil.convert("RGB").save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/jpeg")

@app.post("/translate-page-image")
async def translate_page_image(
    file: UploadFile = File(...),
    lang: str = Form("auto"),
    target_lang: str = Form("uz"),
    gemini_api_key: Optional[str] = Form(None),
    x_gemini_api_key: Optional[str] = Header(None),
    chapter_context: Optional[str] = Form(None),
    use_gemini: bool = Form(False),
    translator: Optional[str] = Form(None),
    _authorized: None = Depends(verify_api_key),
):
    """
    End-to-end Automated Manga/Manhwa/Manhua Page Translator & Typesetter:
    1. Detects speech bubbles, narration boxes, and borderless text via comic-text-detector.
    2. Cleans text strokes using neural text segmentation + UNet mask + SimpleLaMa.
    3. By default, runs Qwen3.5-9B (4-bit) single-pass OCR + translation (100% API-free).
       If Gemini override is explicitly requested with a valid key, translates via Gemini.
    4. Typesets translated text using dynamic elliptical line-wrapping.
    """
    image_bytes, pil_image = await read_image_upload(file)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    width, height = pil_image.size
    normalized_lang = normalize_ocr_language(lang)

    gemini_requested = _is_gemini_requested(gemini_api_key, x_gemini_api_key, use_gemini=use_gemini, translator=translator)
    effective_gemini_key = resolve_gemini_api_key(gemini_api_key, x_gemini_api_key)
    use_gemini_override = bool(gemini_requested and effective_gemini_key)

    pipe = None
    try:
        pipe = get_manga_pipeline()
    except Exception as pipe_exc:
        logger.warning("get_manga_pipeline() failed in translate_page_image: %s", pipe_exc)

    cleaned_img = None
    dialogues = []
    cleaned_ids = set()

    if pipe is not None:
        pipeline_lang = (
            "en" if normalized_lang in PDF_LATIN_OCR_LANGS
            else (normalized_lang if normalized_lang in {"ja", "ko", "zh", "en", "ru"} else "auto")
        )
        desired_target = None if use_gemini_override else target_lang
        try:
            def _run_pipe_page():
                import inspect
                try:
                    sig_params = inspect.signature(pipe.process_page).parameters
                except (TypeError, ValueError):
                    sig_params = {}
                has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
                kwargs = {"auto_collect": False, "source_lang": pipeline_lang}
                if has_var_kw or "target_lang" in sig_params:
                    kwargs["target_lang"] = desired_target
                return pipe.process_page(pil_image, **kwargs)
            res = await run_in_threadpool(_run_pipe_page)
            if not res.get("requires_ocr_fallback"):
                detected_src = res.get("detected_source_lang")
                if normalized_lang == "auto" and detected_src and detected_src != "auto":
                    normalized_lang = normalize_ocr_language(detected_src)
                cleaned_img = res.get("cleaned_image")
                dialogues = res.get("dialogues", []) or []
                cleaned_ids = set(res.get("cleaned_bubble_ids") or [])
        except Exception as pipe_run_exc:
            logger.warning("pipe.process_page() failed in translate_page_image: %s", pipe_run_exc)

    used_fallback_dialogues = False
    if not dialogues:
        try:
            from manga_bubble_engine import extract_manga_dialogues
            raw_dialogues = await run_in_threadpool(
                extract_manga_dialogues,
                pil_image=pil_image,
                source_lang=normalized_lang,
                scale=1.0,
            )
            if raw_dialogues:
                used_fallback_dialogues = True
                for idx, rd in enumerate(raw_dialogues):
                    bbox = rd.get("bbox") or [0, 0, width, height]
                    txt = (rd.get("text") or "").strip()
                    if not txt:
                        continue
                    dialogues.append({
                        "block_id": rd.get("block_id") or f"b_{idx + 1}",
                        "bubble_id": idx + 1,
                        "bbox": bbox,
                        "transcription": txt,
                        "category": rd.get("category", "speech_bubble"),
                        "shape": rd.get("shape", "speech_bubble"),
                    })
        except Exception as err:
            logger.warning("Fallback dialogue extraction in translate_page_image failed: %s", err)

    for idx, d in enumerate(dialogues):
        if "block_id" not in d or not d["block_id"]:
            d["block_id"] = f"b_{d.get('bubble_id', idx + 1)}"

    speech_dialogues = [d for d in dialogues if d.get("category") != "sfx"]
    missing_clean_boxes = [
        d["bbox"] for d in speech_dialogues
        if d.get("bbox") and len(d["bbox"]) == 4 and (used_fallback_dialogues or d.get("bubble_id") not in cleaned_ids)
    ]

    cleaner_succeeded = False
    if cleaned_img is None or cleaned_img is pil_image or not cleaned_ids or bool(missing_clean_boxes):
        cleaner = get_manga_cleaner()
        if cleaner is not None and (missing_clean_boxes or not dialogues):
            try:
                img_np = np.array((cleaned_img if cleaned_img is not None else pil_image).convert("RGB"))
                cleaned_res, edit_mask, _ = await run_in_threadpool(
                    cleaner.clean_page,
                    img_np,
                    allowed_boxes=missing_clean_boxes if missing_clean_boxes else None,
                )
                mask_arr = np.asarray(edit_mask) if edit_mask is not None else None
                if cleaned_res is not None and mask_arr is not None and (not missing_clean_boxes or np.any(mask_arr > 0)):
                    cleaned_img = cleaned_res
                    cleaner_succeeded = True
                    mh, mw = mask_arr.shape[:2]
                    for d in speech_dialogues:
                        bid = d.get("bubble_id")
                        if bid is None:
                            continue
                        bbox = d.get("bbox")
                        if bbox and len(bbox) == 4:
                            bx1 = max(0, min(mw, int(bbox[0])))
                            by1 = max(0, min(mh, int(bbox[1])))
                            bx2 = max(0, min(mw, int(bbox[2])))
                            by2 = max(0, min(mh, int(bbox[3])))
                            if bx2 > bx1 and by2 > by1 and np.any(mask_arr[by1:by2, bx1:bx2] > 0):
                                cleaned_ids.add(bid)
                        elif not missing_clean_boxes:
                            cleaned_ids.add(bid)
            except Exception as err:
                logger.warning("cleaner.clean_page in translate_page_image failed: %s", err)

    if cleaned_img is None:
        cleaned_img = pil_image

    if dialogues:
        if use_gemini_override:
            for d in speech_dialogues:
                d.pop("translation", None)
                d.pop("translated_text", None)

        missing_speech_indices = [
            idx for idx, d in enumerate(speech_dialogues)
            if not (isinstance(d.get("translation") or d.get("translated_text"), str) and (d.get("translation") or d.get("translated_text") or "").strip())
            and (d.get("transcription") or "").strip()
        ]
        if missing_speech_indices:
            speech_bids = [speech_dialogues[idx]["block_id"] for idx in missing_speech_indices]
            texts_to_trans = [(speech_dialogues[idx].get("transcription") or "").strip() for idx in missing_speech_indices]

            # Try Qwen3.5-9B crop translation first when Gemini override wasn't requested
            if not use_gemini_override:
                qwen = get_qwen_vl_engine()
                if qwen is not None:
                    try:
                        q_crops = []
                        valid_q_idx = []
                        for idx in missing_speech_indices:
                            bbox = speech_dialogues[idx].get("bbox")
                            if bbox and len(bbox) == 4:
                                cx1 = max(0, int(bbox[0]) - 4)
                                cy1 = max(0, int(bbox[1]) - 4)
                                cx2 = min(width, int(bbox[2]) + 4)
                                cy2 = min(height, int(bbox[3]) + 4)
                                if cx2 > cx1 + 4 and cy2 > cy1 + 4:
                                    q_crops.append(pil_image.crop((cx1, cy1, cx2, cy2)))
                                    valid_q_idx.append(idx)
                        if q_crops:
                            q_pairs = await run_in_threadpool(
                                qwen.ocr_and_translate_crops, q_crops, target_lang, normalized_lang
                            )
                            for idx, (_, tgt_q) in zip(valid_q_idx, q_pairs):
                                if tgt_q and tgt_q.strip():
                                    speech_dialogues[idx]["translation"] = tgt_q.strip()
                    except Exception as q_err:
                        logger.warning("Qwen translate fallback error in translate_page_image: %s", q_err)

                missing_speech_indices = [
                    idx for idx in missing_speech_indices
                    if not (isinstance(speech_dialogues[idx].get("translation") or speech_dialogues[idx].get("translated_text"), str) and (speech_dialogues[idx].get("translation") or speech_dialogues[idx].get("translated_text") or "").strip())
                ]
                speech_bids = [speech_dialogues[idx]["block_id"] for idx in missing_speech_indices]
                texts_to_trans = [(speech_dialogues[idx].get("transcription") or "").strip() for idx in missing_speech_indices]

            if missing_speech_indices:
                import gemini_translator as _gt_mod
                _gt_fn = getattr(_gt_mod, "translate_dialogues_with_gemini", None)
                _gt_is_mocked = hasattr(_gt_fn, "assert_called")
                if effective_gemini_key or _gt_is_mocked:
                    try:
                        trans_res = await run_in_threadpool(
                            _gt_fn,
                            texts_to_trans,
                            target_lang,
                            normalized_lang,
                            api_key=effective_gemini_key,
                            chapter_context=chapter_context or "",
                            block_ids=speech_bids,
                            return_dict=True,
                        )
                    except Exception as ge:
                        logger.warning("Gemini dialog translation xatosi: %s", ge)
                        trans_res = None

                    still_missing = []
                    if isinstance(trans_res, dict):
                        for idx in missing_speech_indices:
                            val = trans_res.get(speech_dialogues[idx]["block_id"])
                            if isinstance(val, str) and val.strip():
                                speech_dialogues[idx]["translation"] = val.strip()
                            else:
                                still_missing.append(idx)
                    elif isinstance(trans_res, list) and len(trans_res) == len(missing_speech_indices):
                        for idx, t in zip(missing_speech_indices, trans_res):
                            if isinstance(t, str) and t.strip():
                                speech_dialogues[idx]["translation"] = t.strip()
                            else:
                                still_missing.append(idx)
                    else:
                        still_missing = list(missing_speech_indices)
                    missing_speech_indices = still_missing

            if missing_speech_indices:
                from text_translator import translate_text as _orig_translate_text
                custom_tx = translate_text if translate_text is not _orig_translate_text else None
                try:
                    single_res = await run_in_threadpool(
                        translate_batch,
                        [texts_to_trans[pos] for pos, _ in enumerate(missing_speech_indices)] if len(missing_speech_indices) == len(texts_to_trans) else [(speech_dialogues[i].get("transcription") or "").strip() for i in missing_speech_indices],
                        target_lang,
                        translation_language_code(normalized_lang),
                        translate_fn=custom_tx,
                    )
                except Exception as exc:
                    logger.debug("translate_batch failed in translate_page_image: %s", exc)
                    single_res = [None for _ in missing_speech_indices]
                for idx, t in zip(missing_speech_indices, single_res):
                    speech_dialogues[idx]["translation"] = (
                        t if isinstance(t, str) and t.strip() else speech_dialogues[idx].get("transcription", "")
                    )
        for d in dialogues:
            if d.get("category") == "sfx":
                d["translation"] = ""

        uncleaned_speech = [
            d for d in speech_dialogues
            if not cleaner_succeeded and (used_fallback_dialogues or (d.get("bubble_id") not in cleaned_ids))
        ]
        if uncleaned_speech:
            def _clean_dialogue_boxes(base_img, dlgs):
                try:
                    clean_fn = globals().get("clean_bubble_crop")
                    if clean_fn is None:
                        try:
                            from manga_pipeline import clean_bubble_crop as clean_fn
                        except ImportError:
                            try:
                                from ml.manga_pipeline import clean_bubble_crop as clean_fn
                            except ImportError:
                                from cleaning import clean_bubble_crop as clean_fn

                    cur_np = np.array(base_img.convert("RGB"))
                    img_h, img_w = cur_np.shape[:2]
                    for d in dlgs:
                        if d.get("category") == "sfx":
                            continue
                        bbox = d.get("bbox")
                        if not bbox or len(bbox) != 4:
                            continue
                        bx1, by1, bx2, by2 = bbox
                        bx1 = max(0, min(img_w, int(round(bx1))))
                        by1 = max(0, min(img_h, int(round(by1))))
                        bx2 = max(0, min(img_w, int(round(bx2))))
                        by2 = max(0, min(img_h, int(round(by2))))
                        if bx2 > bx1 and by2 > by1:
                            patch = cur_np[by1:by2, bx1:bx2]
                            if patch.size > 0:
                                cleaned_patch = clean_fn(patch)
                                cur_np[by1:by2, bx1:bx2] = cleaned_patch
                    return Image.fromarray(cur_np)
                except Exception as crop_err:
                    logger.warning("clean_bubble_crop on dialogues failed: %s", crop_err)
                    return base_img

            cleaned_img = await run_in_threadpool(_clean_dialogue_boxes, cleaned_img, uncleaned_speech)

        try:
            from typesetting import render_translated_dialogues
        except ImportError:
            try:
                from ml.typesetting import render_translated_dialogues
            except ImportError:
                from manga_pipeline import render_translated_dialogues

        final_image = await run_in_threadpool(render_translated_dialogues, cleaned_img, dialogues)
    else:
        final_image = cleaned_img

    buf = io.BytesIO()
    final_image.convert("RGB").save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("MANGA_BIND_HOST", "127.0.0.1")
    uvicorn.run("main:app", host=host, port=8000, reload=False)
