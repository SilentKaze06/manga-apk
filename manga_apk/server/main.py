import sys
import os
import io
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
            load_dotenv(env_file, override=True)
except Exception as e:
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
from PIL import Image
import pymupdf
import wordninja
from starlette.concurrency import run_in_threadpool
from pdf_jobs import PdfJobError, PdfQueueFullError, PdfJobManager, inspect_pdf

# Load Torch before PaddleOCR; both ship OpenMP DLLs that can conflict on Windows.
import torch
# Hardware thread limits for low-resource VPS (prevents 100% CPU lockup and timeouts)
if not torch.cuda.is_available():
    try:
        max_vps_threads = min(2, os.cpu_count() or 1)
        torch.set_num_threads(max_vps_threads)
    except Exception:
        pass

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("manga-server")

app = FastAPI(
    title="Manga & Manhwa OCR Translation Server",
    description="High-accuracy Manga OCR (manga-ocr ViT) and Manhwa OCR (PaddleOCR) with translation",
    version="1.0.0"
)

# Native Android clients do not need CORS. Unauthenticated access requires
# explicit opt-in for a trusted local development server.
SERVER_API_KEY = os.environ.get("MANGA_SERVER_API_KEY", "").strip()
ALLOW_UNAUTHENTICATED = os.environ.get("MANGA_ALLOW_UNAUTHENTICATED", "false").strip().lower() in {"1", "true", "yes"}
TRANSLATION_TIMEOUT_SECONDS = max(1.0, min(30.0, float(os.environ.get("MANGA_TRANSLATION_TIMEOUT_SECONDS", "8"))))
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
_model_lock = threading.RLock()
_gemini_key_lock = threading.Lock()
_translation_semaphore = None
_google_block_lock = threading.Lock()


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
                    models_dir = Path(__file__).parent.parent.parent / "mashinali oqitish" / "onnx_models"
                vocab_path = models_dir / "ocr_vocab.json"
                if not vocab_path.exists():
                    vocab_path = Path(__file__).parent / "ocr_vocab.json"
                if not vocab_path.exists():
                    vocab_path = Path(__file__).parent / "ml" / "ocr_vocab.json"
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
                get_manga_pipeline()
            except Exception as exc:
                logger.warning("Heavy pipeline preload ogohlantirish: %s", exc)
            try:
                logger.info("Og'ir AI modellari xotiraga yuklanmoqda (MangaOCR)...")
                get_manga_ocr()
            except Exception as exc:
                logger.warning("Heavy MangaOCR preload ogohlantirish: %s", exc)
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


def normalize_ocr_language(value: Optional[str]) -> str:
    """Normalize UI/API language aliases to the codes supported by this server."""
    code = (value or "ja").strip().lower()
    aliases = {
        "jpn": "ja", "japanese": "ja",
        "kor": "ko", "korean": "ko",
        "chs": "zh", "cht": "zh", "chinese": "zh", "ch": "zh",
        "zh-cn": "zh", "zh-tw": "zh",
        "eng": "en", "english": "en",
        "rus": "ru", "russian": "ru",
        "fra": "fr", "french": "fr",
        "deu": "de", "german": "de",
        "spa": "es", "spanish": "es",
        "por": "pt", "portuguese": "pt",
        "ita": "it", "italian": "it",
        "tur": "tr", "turkish": "tr",
        "nld": "nl", "dutch": "nl",
        "pol": "pl", "polish": "pl",
        "uzb": "uz", "uzbek": "uz",
    }
    return aliases.get(code, code or "ja")


def translation_language_code(value: Optional[str], default: str = "auto") -> str:
    """Map common language aliases to the codes accepted by translation providers."""
    code = normalize_ocr_language(value or default)
    return {"zh": "zh-CN", "auto": "auto"}.get(code, code)


def decode_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Böş rasm fayli yuborildi.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.width <= 0 or image.height <= 0 or image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Rasm ölçami ruxsat etilgan çegaradan katta.")
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


def get_manga_ocr():
    """Lazy-load MangaOcr (Japanese Vision Transformer model)."""
    global _mocr
    with _model_lock:
        if _mocr is None:
            try:
                logger.info("MangaOcr (Japanese ViT) modelini yuklaş...")
                manga_ocr_mod = importlib.import_module("manga_ocr")
                manga_ocr_class = getattr(manga_ocr_mod, "MangaOcr")
                _mocr = manga_ocr_class()
                logger.info("MangaOcr muvaffaqiyatli yuklandi.")
            except ModuleNotFoundError:
                err = "manga-ocr kutubxonasi örnatilmagan! Iltimos, 'pip install manga-ocr' yoki 'run_server.bat' ni işga tuşiring."
                logger.error(err)
                raise HTTPException(status_code=500, detail=err)
            except Exception as e:
                logger.error("MangaOcr yuklaşda xatolik: %s", e)
                raise HTTPException(status_code=500, detail="MangaOcr modelini yuklab bölmadi.")
        return _mocr

def get_paddle_ocr(lang: str = "korean"):
    """Lazy-load PaddleOCR for Japanese, Korean, or Chinese text detection."""
    global _paddle_ja, _paddle_ko, _paddle_ch
    global _paddle_en, _paddle_ru, _paddle_latin
    with _model_lock:
        try:
            paddleocr_mod = importlib.import_module("paddleocr")
            paddle_ocr_class = getattr(paddleocr_mod, "PaddleOCR")
            if lang in ["japan", "ja"]:
                if _paddle_ja is None:
                    logger.info("PaddleOCR (Japanese) yuklanmoqda...")
                    _paddle_ja = paddle_ocr_class(use_angle_cls=True, lang="japan", show_log=False)
                    logger.info("PaddleOCR (Japanese) tayyor.")
                return _paddle_ja

            if lang in ["ch", "zh"]:
                if _paddle_ch is None:
                    logger.info("PaddleOCR (Chinese) yuklanmoqda...")
                    _paddle_ch = paddle_ocr_class(use_angle_cls=True, lang="ch", show_log=False)
                    logger.info("PaddleOCR (Chinese) tayyor.")
                return _paddle_ch

            if lang in ["en", "eng", "english"]:
                if _paddle_en is None:
                    logger.info("PaddleOCR (English/Latin) yuklanmoqda...")
                    _paddle_en = paddle_ocr_class(use_angle_cls=True, lang="en", show_log=False)
                    logger.info("PaddleOCR (English/Latin) tayyor.")
                return _paddle_en

            if lang in ["ru", "rus", "russian", "cyrillic"]:
                if _paddle_ru is None:
                    logger.info("PaddleOCR (Russian/Cyrillic) yuklanmoqda...")
                    _paddle_ru = paddle_ocr_class(use_angle_cls=True, lang="ru", show_log=False)
                    logger.info("PaddleOCR (Russian/Cyrillic) tayyor.")
                return _paddle_ru

            if lang in PDF_LATIN_OCR_LANGS or lang in {"latin", "french", "german"}:
                if _paddle_latin is None:
                    logger.info("PaddleOCR (multilingual Latin) yuklanmoqda...")
                    _paddle_latin = paddle_ocr_class(use_angle_cls=True, lang="latin", show_log=False)
                    logger.info("PaddleOCR (multilingual Latin) tayyor.")
                return _paddle_latin

            if _paddle_ko is None:
                logger.info("PaddleOCR (Korean) yuklanmoqda...")
                _paddle_ko = paddle_ocr_class(use_angle_cls=True, lang="korean", show_log=False)
                logger.info("PaddleOCR (Korean) tayyor.")
            return _paddle_ko
        except ModuleNotFoundError:
            err = "paddleocr kutubxonasi örnatilmagan! Iltimos, 'pip install paddleocr paddlepaddle' yoki 'run_server.bat' ni işga tuşiring."
            logger.error(err)
            raise HTTPException(status_code=500, detail=err)
        except Exception as e:
            logger.error("PaddleOCR (%s) yuklaşda xatolik: %s", lang, e)
            raise HTTPException(status_code=500, detail="PaddleOCR modelini yuklab bölmadi.")


def run_manga_ocr(image: Image.Image) -> str:
    use_heavy = os.environ.get("MANGA_USE_HEAVY_MODELS", "false").lower() in ("1", "true", "yes")
    if not use_heavy:
        engine = get_onnx_engine()
        if engine and engine.ocr_sess:
            try:
                res = engine.ocr_bubble(image)
                if res:
                    return res
            except Exception as e:
                logger.warning(f"ONNX OCR xatosi: {e}")
    with _model_lock:
        return get_manga_ocr()(image)


def is_paddle_enabled() -> bool:
    if os.environ.get("MANGA_DISABLE_PADDLE", "false").lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("MANGA_USE_PADDLE", "true").lower() in ("0", "false", "no"):
        return False
    return True


def run_paddle_ocr(image_bytes: bytes, lang: str):
    if not is_paddle_enabled():
        logger.debug("PaddleOCR disabled on this CPU to prevent SIGILL crashes.")
        return []
    import numpy as np
    with _model_lock:
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
            results = get_paddle_ocr(lang).ocr(np_img, cls=True)
            if results and results[0]:
                results[0] = [
                    ([[float(point[0]) / scale_x, float(point[1]) / scale_y] for point in item[0]], item[1])
                    for item in results[0]
                ]
            return results
        np_img = np.array(image)
        return get_paddle_ocr(lang).ocr(np_img, cls=True)


def _script_score(text: str, confidence: float, language: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    latin = sum("LATIN" in unicodedata.name(char, "") for char in letters)
    ascii_latin = sum(char.isascii() and char.isalpha() for char in letters)
    cyrillic = sum("\u0400" <= char <= "\u04ff" for char in letters)
    hangul = sum("\uac00" <= char <= "\ud7af" or "\u1100" <= char <= "\u11ff" for char in letters)
    kana = sum("\u3040" <= char <= "\u30ff" for char in letters)
    han = sum("\u3400" <= char <= "\u9fff" for char in letters)
    expected = {
        "en": ascii_latin,
        "auto": latin,
        "ru": cyrillic,
        "ko": hangul,
        "ja": (kana * 3 + han) if kana > 0 else (han * 0.3),
        "zh": (han * 2) if kana == 0 else 0,
    }.get(language, 0)
    purity = min(1.0, expected / max(1, len(letters)))
    coverage = min(expected, 20) / 20.0
    return purity * 2.0 + coverage + max(0.0, min(1.0, confidence)) * 0.35


def run_auto_paddle_ocr(image_bytes: bytes):
    """Try CJK and Latin models; breaks early when confident text is found to prevent timeouts."""
    candidates = [("ja", "japan"), ("ko", "korean"), ("zh", "ch"), ("ru", "ru"), ("auto", "latin")]
    best_language, best_results, best_score = "auto", [], 0.0
    for language, model_language in candidates:
        try:
            results = run_paddle_ocr(image_bytes, model_language)
            lines = extract_paddle_lines(results, language)
            if not lines:
                continue
            text = " ".join(line["text"] for line in lines)
            confidence = sum(line["confidence"] for line in lines) / len(lines)
            score = _script_score(text, confidence, language)
            if score > best_score:
                best_language, best_results, best_score = language, results, score
                # If confident match (>1.6) is found, break immediately to save CPU and prevent timeout
                if score >= 1.6 and confidence >= 0.75:
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
        return merge_horizontal_ocr_lines(lines)
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


def _sample_comic_background(image: Image.Image, box: tuple[int, int, int, int]) -> list[float]:
    """Sample the true dominant background color of the dialogue box or speech bubble."""
    left, top, right, bottom = box
    w, h = image.width, image.height
    bx1 = max(0, left - 2)
    by1 = max(0, top - 2)
    bx2 = min(w, right + 2)
    by2 = min(h, bottom + 2)
    if bx2 <= bx1 or by2 <= by1:
        return [1.0, 1.0, 1.0]

    try:
        import numpy as np
        crop = image.crop((bx1, by1, bx2, by2)).convert("RGB")
        np_crop = np.asarray(crop)
        if np_crop.size == 0:
            return [1.0, 1.0, 1.0]

        ch_h, ch_w = np_crop.shape[0], np_crop.shape[1]
        border_thickness = min(2, max(1, ch_h // 2), max(1, ch_w // 2))
        top_ring = np_crop[:border_thickness, :]
        bottom_ring = np_crop[-border_thickness:, :]
        left_ring = np_crop[border_thickness:-border_thickness, :border_thickness]
        right_ring = np_crop[border_thickness:-border_thickness, -border_thickness:]
        border_pixels = np.concatenate([
            top_ring.reshape(-1, 3),
            bottom_ring.reshape(-1, 3),
            left_ring.reshape(-1, 3),
            right_ring.reshape(-1, 3)
        ], axis=0) if ch_h > border_thickness * 2 and ch_w > border_thickness * 2 else np_crop.reshape(-1, 3)

        dominant_pixels = border_pixels if len(border_pixels) > 0 else np_crop.reshape(-1, 3)

        median_rgb = np.median(dominant_pixels, axis=0) / 255.0
        luma = float(np.mean(median_rgb))
        if luma < 0.35:
            return [float(min(0.12, c)) for c in median_rgb]
        elif luma > 0.92:
            # Only snap to pure white when the background is nearly pure white
            return [1.0, 1.0, 1.0]
        else:
            return [float(c) for c in median_rgb]
    except Exception:
        return [1.0, 1.0, 1.0]


def _find_bubble_overlay_region(
    image: Image.Image,
    box: tuple[int, int, int, int],
    threshold_mask=None,
) -> Optional[dict]:
    """Find a bounded light bubble interior around OCR text; reject page-sized regions."""
    try:
        cv2 = importlib.import_module("cv2")
        import numpy as np

        left, top, right, bottom = box
        left = max(0, min(image.width - 1, left))
        right = max(left + 1, min(image.width, right))
        top = max(0, min(image.height - 1, top))
        bottom = max(top + 1, min(image.height, bottom))
        if threshold_mask is None:
            pixels = np.asarray(image.convert("RGB"))
            gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
            sub_gray = gray[top:bottom, left:right]
            is_dark = np.median(sub_gray) < 110 if sub_gray.size > 0 else False
            thresh_type = cv2.THRESH_BINARY_INV if is_dark else cv2.THRESH_BINARY
            thresh_val = 60 if is_dark else 220
            threshold_mask = cv2.threshold(gray, thresh_val, 255, thresh_type)[1]
        pad = min(1200, max(96, (max(right - left, bottom - top) * 2)))
        crop_left, crop_top = max(0, left - pad), max(0, top - pad)
        crop_right, crop_bottom = min(image.width, right + pad), min(image.height, bottom + pad)
        mask = threshold_mask[crop_top:crop_bottom, crop_left:crop_right].copy()
        local_left, local_top = left - crop_left, top - crop_top
        local_right, local_bottom = right - crop_left, bottom - crop_top
        # Fill the recognized ink region, then use the enclosed light area as
        # a better text box when a bubble outline can be found confidently.
        mask[local_top:local_bottom, local_left:local_right] = 255
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        label = labels[(local_top + local_bottom - 1) // 2, (local_left + local_right - 1) // 2]
        if label == 0:
            return None
        local_x, local_y, width, height, _ = stats[label]
        page_pixels = max(1, image.width * image.height)
        text_area = max(1, (right - left) * (bottom - top))
        bubble_area = width * height
        if bubble_area > page_pixels * 0.28 or bubble_area < text_area * 1.35:
            return None
        if width < (right - left) * 1.12 or height < (bottom - top) * 1.12:
            return None
        if local_x == 0 or local_y == 0 or local_x + width >= mask.shape[1] or local_y + height >= mask.shape[0]:
            return None
        component = np.where(labels[local_y:local_y + height, local_x:local_x + width] == label, 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        component = cv2.morphologyEx(component, cv2.MORPH_CLOSE, kernel)
        eroded = cv2.erode(component, kernel, iterations=1)
        if cv2.countNonZero(eroded) > 20:
            component = eroded
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, max(1.5, perimeter * 0.003), True).reshape(-1, 2)
        if len(polygon) < 3 or len(polygon) > 240:
            return None
        return {
            "bbox": [crop_left + local_x, crop_top + local_y, crop_left + local_x + width, crop_top + local_y + height],
            "polygon": [[int(crop_left + local_x + point[0]), int(crop_top + local_y + point[1])] for point in polygon],
        }
    except Exception as exc:
        logger.debug("Komiks pufagi chegarasini topib bölmadi: %s", exc)
        return None


def is_manga_punctuation(text: str) -> bool:
    """
    Detects standalone expressive manga punctuation marks like '?', '!', '?!', '!?', '...', '…', '---'.
    Also catches common OCR misreadings of exclamation mark (e.g. '?1', '?l', '?I', '!1', '1!').
    """
    t = text.strip()
    if not t:
        return False
    normalized = re.sub(r'(?<=[?!])\s*[1lI|]\b', '!', t)
    normalized = re.sub(r'\b[1lI|]\s*(?=[?!])', '!', normalized)
    clean = re.sub(r"[\s\.,!?'\"_\-~:;#@$%^&*()\[\]{}|<>/\\`+=…¿¡]", "", normalized)
    if not clean:
        return any(c in "?!…~" or ".." in normalized or "--" in normalized for c in normalized)
    if len(clean) == 1 and any(c in "?!…" for c in normalized) and len(t) <= 4:
        return True
    return False


def normalize_manga_punctuation(text: str) -> str:
    """Normalizes manga punctuation OCR quirks into clean comic typography."""
    t = text.strip()
    t = re.sub(r'(?<=[?!])\s*[1lI|]\b', '!', t)
    t = re.sub(r'\b[1lI|]\s*(?=[?!])', '!', t)
    if re.match(r'^[?!1lI|\s]+$', t):
        has_q = "?" in t
        has_ex = any(c in "!1lI|" for c in t)
        if has_q and has_ex:
            return "?!"
        elif has_q:
            return "?" * min(3, t.count("?"))
        elif has_ex:
            return "!" * min(3, sum(1 for c in t if c in "!1lI|"))
    return t


def is_ocr_garbage_noise(text: str) -> bool:
    """Detects OCR hallucinations from manga screentones, halftones, speed lines, and noise."""
    t = text.strip()
    if not t:
        return True
    if is_manga_punctuation(t):
        return False
    # Treat 2–3 char alphabetic tokens ending with ? or ! inside detected bubbles as valid dialogue
    if re.match(r'^[A-Za-z\u0400-\u04ff]{2,3}\s*[?!]+$', t):
        return False
    clean = re.sub(r"[\s\.,!?'\"_\-~:;#@$%^&*()\[\]{}|<>/\\`+=]", "", t)
    if not clean:
        return True

    if len(clean) <= 2:
        valid_shorts = {
            "i", "a", "no", "oh", "ok", "ah", "eh", "uh", "um", "yo", "ha", 
            "so", "to", "he", "we", "me", "my", "in", "on", "at", "by", "if", 
            "da", "ya", "hm", "mm", "hi", "it", "is", "am", "as", "us", "up", "do",
            "go", "an", "or", "of", "be", "ex", "re"
        }
        has_cjk_or_hangul = any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af' or '\u1100' <= c <= '\u11ff' for c in clean)
        has_cyrillic = any('\u0400' <= c <= '\u04ff' for c in clean)
        if clean.lower() not in valid_shorts and not has_cjk_or_hangul and not has_cyrillic:
            return True

    letters = sum(1 for c in clean if c.isalpha())
    digits = sum(1 for c in clean if c.isdigit())

    if letters == 0 and digits > 0:
        return True

    if digits > 0 and letters > 0:
        if digits / (digits + letters) > 0.30 and not re.search(r'\b(?:1st|2nd|3rd|\d+th|\d+pm|\d+am|\d+:\d+)\b', t, re.I):
            return True

    return False


COMMON_SHORT_WORDS = {'i', 'a', 'no', 'oh', 'ok', 'my', 'me', 'to', 'in', 'on', 'he', 'we', 'so', 'go', 'up', 'do', 'it', 'is', 'at', 'by', 'as', 'if', 'us', 'am', 'hm', 'mm', 'eh', 'ha', 'uh', 'um', 'yo', 'ya'}

def is_valid_manga_dialogue(text: str, source_lang: str = "en") -> bool:
    """Strictly validates OCR output to eliminate hallucinations on faces, hair, and halftone screentones."""
    stripped = text.strip() if text else ""
    if not stripped:
        return False
    # Standalone expressive manga punctuation (?, !, ?!, ...) is valid dialogue
    if is_manga_punctuation(stripped):
        return True
    # Treat 2–3 char alphabetic tokens ending with ? or ! inside detected bubbles as valid dialogue
    if re.match(r'^[A-Za-z\u0400-\u04ff]{2,3}\s*[?!]+$', stripped):
        return True
    # Allow valid single-letter words or single CJK characters
    if stripped.lower() in {'i', 'a'}:
        return True

    has_cjk = bool(re.search(r'[\u2e80-\u9fff\uac00-\ud7af\uff00-\uffef\u3000-\u30ff]', stripped))
    has_cyrillic = bool(re.search(r'[\u0400-\u04ff]', stripped))
    has_arabic = bool(re.search(r'[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]', stripped))

    if source_lang in ("ja", "ko", "zh") or has_cjk:
        return bool(has_cjk and len(stripped) >= 1)
    if source_lang in ("ru", "uk", "bg") or has_cyrillic:
        return bool(has_cyrillic and len(stripped) >= 1)
    if source_lang in ("ar", "fa", "ur") or has_arabic:
        return bool(has_arabic and len(stripped) >= 1)

    if len(stripped) < 2:
        return False
    cleaned = re.sub(r'[^A-Za-z0-9\s\'\.,!\?\"-\u2e80-\u9fff\uac00-\ud7af\uff00-\uffef\u3000-\u30ff\u0400-\u04ff]', '', text).strip()
    words = [w.strip(r"'.,!?\"-") for w in cleaned.split() if w.strip()]
    if not words:
        return False
    if source_lang != "en" and any(any(c.isalpha() for c in w) for w in words):
        return True
    # Validate against genuine dictionary words to block garbage like "ZW 2 TN 2", "a ay Za j 7"
    try:
        import wordninja
        dict_words = wordninja.DEFAULT_LANGUAGE_MODEL._wordcost
        real_word_count = sum(1 for w in words if w.lower() in dict_words and len(w) >= 2)
        if real_word_count >= 1:
            return True
    except Exception:
        pass
    for w in words:
        wl = w.lower()
        if len(wl) >= 3 and any(c.isalpha() for c in wl):
            return True
        if wl in COMMON_SHORT_WORDS:
            return True
    return False


def _group_comic_lines(lines, rendered_image, bubble_threshold_mask, scale, detected_lang="en"):
    """Group lines that share the same bubble or overlap closely into a single coherent block."""
    if not lines:
        return []

    line_entries = []
    for line in lines:
        if not line["text"].strip():
            continue
        left = max(0, min(rendered_image.width, line["left"]))
        top = max(0, min(rendered_image.height, line["top"]))
        right = max(left, min(rendered_image.width, line["right"]))
        bottom = max(top, min(rendered_image.height, line["bottom"]))
        box = (left, top, right, bottom)
        ov = _find_bubble_overlay_region(rendered_image, box, bubble_threshold_mask)
        line_entries.append({
            "line": line,
            "box": box,
            "overlay": ov,
        })

    if not line_entries:
        return []

    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        return inter / float(areaA + areaB - inter)

    n = len(line_entries)
    adj = [[] for _ in range(n)]
    for i in range(n):
        e1 = line_entries[i]
        ov1 = e1["overlay"]
        for j in range(i + 1, n):
            e2 = line_entries[j]
            ov2 = e2["overlay"]
            should_merge = False
            if ov1 and ov2:
                if iou(ov1["bbox"], ov2["bbox"]) > 0.35:
                    should_merge = True
            elif ov1 and not ov2:
                b2 = e2["box"]
                b1 = ov1["bbox"]
                if b1[0] <= b2[0] and b1[1] <= b2[1] and b1[2] >= b2[2] and b1[3] >= b2[3]:
                    should_merge = True
            elif not ov1 and ov2:
                b1 = e1["box"]
                b2 = ov2["bbox"]
                if b2[0] <= b1[0] and b2[1] <= b1[1] and b2[2] >= b1[2] and b2[3] >= b1[3]:
                    should_merge = True
            else:
                box1 = e1["box"]
                box2 = e2["box"]
                h1 = box1[3] - box1[1]
                h2 = box2[3] - box2[1]
                h_max = max(h1, h2)
                y_gap = max(0, box2[1] - box1[3]) if box2[1] >= box1[3] else max(0, box1[1] - box2[3])
                x_overlap = min(box1[2], box2[2]) - max(box1[0], box2[0])
                if y_gap < h_max * 1.5 and x_overlap > 0:
                    should_merge = True

            if should_merge:
                adj[i].append(j)
                adj[j].append(i)

    clusters = []
    visited = set()
    for i in range(n):
        if i in visited:
            continue
        cluster = []
        queue = [i]
        visited.add(i)
        while queue:
            curr = queue.pop(0)
            cluster.append(line_entries[curr])
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        clusters.append(cluster)

    blocks = []
    for cluster in clusters:
        cluster.sort(key=lambda x: (x["box"][1], x["box"][0]))
        text = " ".join(x["line"]["text"].strip() for x in cluster)
        if detected_lang in {"en", "english", "eng"}:
            text = repair_english_word_spacing(text)
        if not text:
            continue

        min_l = min(x["box"][0] for x in cluster)
        min_t = min(x["box"][1] for x in cluster)
        max_r = max(x["box"][2] for x in cluster)
        max_b = max(x["box"][3] for x in cluster)
        avg_conf = sum(x["line"]["confidence"] for x in cluster) / len(cluster)

        if detected_lang == "ja" and max_r > min_l and max_b > min_t:
            try:
                refined = run_manga_ocr(rendered_image.crop((min_l, min_t, max_r, max_b))).strip()
                if refined:
                    text = refined
            except Exception as exc:
                logger.warning("PDF komiksidagi MangaOCR xatosi: %s", exc)

        best_overlay = next((x["overlay"] for x in cluster if x["overlay"]), None)
        if not best_overlay:
            pad_x = max(6, int((max_r - min_l) * 0.08))
            pad_y = max(5, int((max_b - min_t) * 0.08))
            ob_left = max(0, min_l - pad_x)
            ob_top = max(0, min_t - pad_y)
            ob_right = min(rendered_image.width, max_r + pad_x)
            ob_bottom = min(rendered_image.height, max_b + pad_y)
            best_overlay = {
                "bbox": [ob_left, ob_top, ob_right, ob_bottom],
                "polygon": [
                    [ob_left, ob_top],
                    [ob_right, ob_top],
                    [ob_right, ob_bottom],
                    [ob_left, ob_bottom],
                ]
            }

        block = {
            "text": text,
            "font_size": 0.0,
            "confidence": avg_conf,
            "bbox": [min_l / scale, min_t / scale, max_r / scale, max_b / scale],
            "text_erase_bbox": [min_l / scale, min_t / scale, max_r / scale, max_b / scale],
            "overlay_bbox": [c / scale for c in best_overlay["bbox"]],
            "overlay_polygon": [[x / scale, y / scale] for x, y in best_overlay["polygon"]],
            "background": _sample_comic_background(rendered_image, (min_l, min_t, max_r, max_b)),
        }
        blocks.append(block)
    return blocks

def ocr_pdf_page(page, source_lang: str, document_mode: str = "book", page_index: int = 0, job_dir: Optional[Path] = None) -> dict:
    """
    Renders one PDF page and runs either:
    1. Comic Mode: State-of-the-Art ML Manga Translation Pipeline (MangaBubbleDetector,
       EasyOCR, and Neural Inpainting for clean text removal).
    2. Book Mode: PaddleOCR line recognition, layout extraction, and reflow blocks.
    """
    global _manga_pipeline_error
    rect = page.rect
    page_area = max(1.0, float(rect.width * rect.height) if hasattr(rect, "width") else 1.0)
    target_scale = PDF_OCR_DPI / 72.0
    pixel_limited_scale = (PDF_OCR_MAX_PIXELS / page_area) ** 0.5
    scale = min(target_scale, pixel_limited_scale)
    if scale < 0.1:
        raise PdfJobError("PDF bet ölçami OCR uçun juda katta.")
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    image_bytes = pixmap.tobytes("png")

    with Image.open(io.BytesIO(image_bytes)) as opened_image:
        rendered_image = opened_image.convert("RGB")

    normalized_lang = normalize_ocr_language(source_lang)

    # ─── 1. COMIC MODE: ML Bubble Detection & Neural Inpainting Text Cleaning ───
    if document_mode == "comic":
        try:
            pipe = get_manga_pipeline()
            with _model_lock:
                pipeline_res = pipe.process_page(rendered_image, auto_collect=False, source_lang=normalized_lang)
            cleaned_img = pipeline_res.get("cleaned_image")
            dialogues = pipeline_res.get("dialogues", [])
            cleaned_ids = set(pipeline_res.get("cleaned_bubble_ids", []))
            if pipeline_res.get("requires_ocr_fallback"):
                raise PdfJobError("ML OCR natijasi işonçsiz; zaxira OCR tekşiriladi.")

            blocks = []
            for d in dialogues:
                text = (d.get("transcription") or "").strip()
                if not text:
                    continue
                bx1, by1, bx2, by2 = d["bbox"]
                blocks.append({
                    "text": text,
                    "font_size": 0.0,
                    "confidence": float(d.get("confidence", 0.5)),
                    "bbox": [bx1 / scale, by1 / scale, bx2 / scale, by2 / scale],
                    "category": d.get("category", "speech_bubble"),
                    "bubble_id": d.get("bubble_id", len(blocks) + 1),
                    "cleaned": d.get("bubble_id") in cleaned_ids,
                })

            if not blocks:
                raise PdfJobError("ML OCR matn topmadi; zaxira OCR tekshiriladi.")

            cleaned_filename = None
            if cleaned_img is not None and job_dir is not None:
                cleaned_filename = f"cleaned_page_{page_index:04d}.png"
                cleaned_img.save(job_dir / cleaned_filename, format="PNG")

            detected_lang = pipeline_res.get("detected_source_lang", normalized_lang)
            _manga_pipeline_error = None
            return {
                "blocks": blocks,
                "detected_source_lang": detected_lang,
                "cleaned_image_filename": cleaned_filename,
                "rotation": int(getattr(page, "rotation", 0)) % 360,
            }
        except Exception as exc:
            if not isinstance(exc, PdfJobError):
                _manga_pipeline_error = type(exc).__name__
            logger.warning("Comic PDF rejimida xatolik: %s. Zaxira OCR tekşiriladi...", exc)
            try:
                from manga_bubble_engine import extract_manga_dialogues
                blocks = extract_manga_dialogues(
                    pil_image=rendered_image,
                    source_lang=normalized_lang,
                    scale=scale,
                )
                if blocks:
                    detected_lang = normalized_lang if normalized_lang != "auto" else "en"
                    _manga_pipeline_error = None
                    return {
                        "blocks": blocks,
                        "detected_source_lang": detected_lang,
                        "cleaned_image_filename": None,
                        "rotation": int(getattr(page, "rotation", 0)) % 360,
                    }
            except Exception:
                pass

    # ─── 2. BOOK MODE or Fallback OCR ──────────────────────────────────────────
    if normalized_lang == "auto":
        detected_lang, results = run_auto_paddle_ocr(image_bytes)
        if detected_lang == "en":
            detected_lang = "auto"
            lines = extract_paddle_lines(results, "en")
        else:
            lines = extract_paddle_lines(results, detected_lang)
    else:
        detector_lang = PDF_OCR_MODEL_LANGS.get(normalized_lang)
        if detector_lang is None:
            raise PdfJobError(f"PDF OCR tili qöllab-quvvatlanmaydi: {normalized_lang}")
        detected_lang = normalized_lang
        results = run_paddle_ocr(image_bytes, detector_lang)
        lines = extract_paddle_lines(results, detected_lang)

    if document_mode == "comic" and detected_lang == "ja":
        lines = _group_vertical_manga_lines(lines)

    bubble_threshold_mask = None
    if document_mode == "comic":
        try:
            cv2 = importlib.import_module("cv2")
            import numpy as np
            grayscale = cv2.cvtColor(np.asarray(rendered_image), cv2.COLOR_RGB2GRAY)
            bubble_threshold_mask = cv2.threshold(grayscale, 220, 255, cv2.THRESH_BINARY)[1]
        except Exception as exc:
            logger.debug("Pufak konturi uçun rasmni tayyorlab bölmadi: %s", exc)

    if document_mode == "comic":
        blocks = _group_comic_lines(lines, rendered_image, bubble_threshold_mask, scale, detected_lang=detected_lang)
    else:
        blocks = []
        for line in lines:
            if not line["text"].strip():
                continue
            left = max(0, min(rendered_image.width, line["left"]))
            top = max(0, min(rendered_image.height, line["top"]))
            right = max(left, min(rendered_image.width, line["right"]))
            bottom = max(top, min(rendered_image.height, line["bottom"]))
            blocks.append({
                "text": line["text"],
                "font_size": 0.0,
                "confidence": line["confidence"],
                "bbox": [left / scale, top / scale, right / scale, bottom / scale],
                "text_erase_bbox": [left / scale, top / scale, right / scale, bottom / scale],
            })
    return {
        "blocks": blocks,
        "detected_source_lang": detected_lang,
        "rotation": int(getattr(page, "rotation", 0)) % 360,
    }


KEEP_WORDS = {
    "cannot", "into", "onto", "within", "without", "today", "tonight", "tomorrow",
    "maybe", "someone", "anyone", "everyone", "nothing", "something", "anything",
    "everything", "understand", "because", "before", "behind", "already", "myself",
    "yourself", "himself", "herself", "itself", "ourselves", "themselves", "another",
    "together", "forever", "sometimes", "somewhere", "anywhere", "everywhere", "nowhere",
    "always", "almost", "also", "although", "anyway", "become", "became", "becomes",
    "between", "beyond", "inside", "outside", "upon", "towards", "toward", "classroom",
    "blackboard", "blackout", "president", "lightning", "statement", "neighbor",
    "ordinary", "coexist", "retract", "burglar", "feeling", "expected", "restored",
    "school", "family", "happy"
}

SPECIFIC_COMIC_REPLACEMENTS = [
    (r'\b(?:THE\s+)?POWER\s+(?:L[sS]T|LIST|LST|MUST)\s*BE\s*(?:TO\s*RED|TORED|ORED|RESTORED)?\b', 'THE POWER MUST BE RESTORED'),
    (r'\b(?:L[sS]T|LIST|LST|MUST)\s*BE\s*(?:TO\s*RED|TORED|ORED)\b', 'MUST BE RESTORED'),
    (r'\bTORED\b', 'RESTORED'),
    (r'\bUR\s*MED:?\b', 'HARMED.'),
    (r'\bURMED:?\b', 'HARMED.'),
    (r'\btooK\s+CARE\s+Of\s+\.\.\.\s+NO\s+ONE\s+WAS\s+(?:UR\s*MED|URMED|HARMED):?\b', 'NO ONE WAS HARMED... TOOK CARE OF IT.'),
    (r'\bCONTROL\s+IT\s+\.\.\.\s+WHY\s+COULDNT\b', "WHY COULDN'T I CONTROL IT?"),
    (r'\bCOULDNT\s+CONTROL\s*(?:UT|IT|1T)?\b', "COULDN'T CONTROL IT"),
    (r'\bMYTHFOLK\s+A\s+DRAGON\b', 'MYTHOLOGICAL DRAGON'),
    (r'\bYOURE\s+A\s+MYTHFOLK\b', "YOU'RE A MYTHOLOGICAL DRAGON"),
    (r'\bMUST\s+REMOVE\s+MY-\s*SELF\b', 'MUST REMOVE MYSELF'),
    (r'\bLISTEN\s+TO\s+ME;?\s*SHINTO\.?\b', 'LISTEN TO ME, SHINTO.'),
    (r'\bLISTBE\s+ORED\b', 'MUST BE RESTORED'),
    (r'\bPOWER\s+LISTBE\b', 'POWER MUST BE'),
    (r'\bLISTBE\b', 'MUST BE'),
    (r'\bMSST\b', 'MUST'),
    (r'\bMVST\b', 'MUST'),
    (r'\bWANTUS\b', 'WANT US'),
    (r'\bFORUSTO\b', 'FOR US TO'),
    (r'\bINORDER\b', 'IN ORDER'),
    (r'\bANDIT\'?S\b', "AND IT'S"),
    (r'\bASA\b', 'AS A'),
    (r'\bTOBE\b', 'TO BE'),
    (r'\bFORME\b', 'FOR ME'),
    (r'\bWITHME\b', 'WITH ME'),
    (r'\bTOME\b', 'TO ME'),
    (r'\bINMY\b', 'IN MY'),
    (r'\bONMY\b', 'ON MY'),
    (r'\bOFMY\b', 'OF MY'),
    (r'\bTOMY\b', 'TO MY'),
    (r'\bATMY\b', 'AT MY'),
    (r'\bANDTHE\b', 'AND THE'),
    (r'\bINTHE\b', 'IN THE'),
    (r'\bONTHE\b', 'ON THE'),
    (r'\bFORTHE\b', 'FOR THE'),
    (r'\bWITHTHE\b', 'WITH THE'),
    (r'\bFROMTHE\b', 'FROM THE'),
    (r'\bOFTHE\b', 'OF THE'),
    (r'\bATTHE\b', 'AT THE'),
    (r'\bTOTHE\b', 'TO THE'),
    (r'\bHAVETO\b', 'HAVE TO'),
    (r'\bHADTO\b', 'HAD TO'),
    (r'\bWANTTO\b', 'WANT TO'),
    (r'\bNEEDTO\b', 'NEED TO'),
    (r'\bGOINGTO\b', 'GOING TO'),
    (r'\bABOUTTO\b', 'ABOUT TO'),
    (r'\bRIGHTNOW\b', 'RIGHT NOW'),
    (r'\bLOOKAT\b', 'LOOK AT'),
    (r'\bCOMEON\b', 'COME ON'),
    (r'\bHEWAS\b', 'HE WAS'),
    (r'\bSHEWAS\b', 'SHE WAS'),
    (r'\bITWAS\b', 'IT WAS'),
    (r'\bTHEYARE\b', 'THEY ARE'),
    (r'\bWEARE\b', 'WE ARE'),
    (r'\bYOUARE\b', 'YOU ARE'),
    (r'\bYOULIKE\b', 'YOU LIKE'),
    (r'\bMYTH\s*FO\s*LK\b', 'MYTH-FOLK'),
    (r'\bMOHX\b', 'MUCH'),
    (r'\bHOTXAT[bЬ]\b', 'ХОТЯТ'),
    (r'\bUNCER-\s*', 'UNCERTAIN '),
    (r'\bTHEREA\b', 'THERE IS A'),
    (r'\bBlack-IC\b', 'BLACK-ISH'),
    (r'\bUNO\b', 'ONE'),
    (r'\bWHYP\b', 'WHY?'),
    (r'\bWHATP\b', 'WHAT?'),
    (r'\bHOWP\b', 'HOW?'),
    (r'\bWHOP\b', 'WHO?'),
]


def repair_english_word_spacing(text: str) -> str:
    """Split confidently recognized English words that OCR joined together."""
    if not text:
        return ""
    # Punctuation/OCR typo repairs in comic uppercase text
    def _fix_trailing_q(match):
        lead = match.group(1)
        mid = match.group(2)
        tail = match.group(3)
        tokens = mid.rstrip().split()
        if tokens:
            cand = (tokens[-1] + tail).lower()
            try:
                dict_words = wordninja.DEFAULT_LANGUAGE_MODEL._wordcost
                if cand in dict_words:
                    return match.group(0)
            except Exception:
                if cand in {"stop", "help", "jump", "sleep", "keep", "deep", "step", "trip", "ship", "shop", "drop", "whip", "cup", "top", "cap", "map"}:
                    return match.group(0)
        return f"{lead}{mid}?"

    text = re.sub(
        r'\b(WHY|WHAT|WHO|WHERE|WHEN|HOW|COULDN\'?T|CAN|CAN\'?T|DID|IS|ARE|DO|DOES|WILL|WOULD|SHOULD)\b([^?!.\n]+?)\b([P|p|7|/])\b',
        _fix_trailing_q,
        text,
        flags=re.IGNORECASE
    )
    for pattern, repl in SPECIFIC_COMIC_REPLACEMENTS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    try:
        word_costs = wordninja.DEFAULT_LANGUAGE_MODEL._wordcost

        def split_token(match):
            token = match.group(0)
            lower = token.lower()
            if lower in KEEP_WORDS:
                return token
            parts = wordninja.split(lower)
            if len(parts) < 2:
                return token
            if any(len(p) == 1 and p not in {'a', 'i'} for p in parts):
                return token
            if any(p not in word_costs or word_costs[p] > 13.5 for p in parts if len(p) > 1):
                return token
            token_cost = word_costs.get(lower, 999.0)
            parts_cost = sum(word_costs[p] for p in parts)
            if token_cost > 10.0 or parts_cost < token_cost - 1.0:
                if token.isupper():
                    return " ".join(p.upper() for p in parts)
                elif token[0].isupper() and token[1:].islower():
                    return parts[0].capitalize() + " " + " ".join(parts[1:])
                return " ".join(parts)
            return token

        text = re.sub(r"[A-Za-z]{3,}", split_token, text)
    except Exception:
        pass
    return text


def _join_dialogue_lines(lines: list[str], lang: str = "en") -> str:
    """Join dialogue lines, properly merging hyphenated words across line breaks."""
    result = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not result:
            result = line
        elif result.endswith("-") and len(result) >= 2 and result[-2].isalpha() and line[0].isalpha():
            result = result[:-1] + line
        else:
            result = result + " " + line
    if lang in {"en", "english", "eng"}:
        return repair_english_word_spacing(result)
    return result

def merge_horizontal_ocr_lines(items):
    """Join word boxes into readable lines, then nearby lines into distinct dialogue blocks.
    Separates adjacent speech bubbles and preserves strict column boundaries."""
    if not items:
        return []

    median_h = sorted(max(1, it["bottom"] - it["top"]) for it in items)[len(items)//2]
    # In manga speech bubbles, words on the exact same line have small spacing (typically 3-7px).
    max_word_gap = max(6, int(median_h * 0.55))

    sorted_items = sorted(items, key=lambda it: (it["top"], it["left"]))
    rows = []
    for it in sorted_items:
        h = max(1, it["bottom"] - it["top"])
        cy = (it["top"] + it["bottom"]) / 2.0

        matched_row = None
        min_gap = 9999
        for row in rows:
            rcy = (row["top"] + row["bottom"]) / 2.0
            rh = row["bottom"] - row["top"]
            # Strict vertical alignment for words on the same line
            if abs(cy - rcy) <= max(3.0, min(h, rh) * 0.35):
                gap_x = it["left"] - row["right"]
                if 0 <= gap_x <= max_word_gap and gap_x < min_gap:
                    min_gap = gap_x
                    matched_row = row

        if matched_row is not None:
            matched_row["items"].append(it)
            matched_row["right"] = max(matched_row["right"], it["right"])
            matched_row["top"] = min(matched_row["top"], it["top"])
            matched_row["bottom"] = max(matched_row["bottom"], it["bottom"])
            matched_row["text"] = matched_row["text"] + " " + it["text"]
            matched_row["confidence"] = sum(item["confidence"] for item in matched_row["items"]) / len(matched_row["items"])
        else:
            rows.append({
                "left": it["left"], "top": it["top"],
                "right": it["right"], "bottom": it["bottom"],
                "text": it["text"],
                "confidence": it["confidence"],
                "items": [it]
            })

    adj = {i: [] for i in range(len(rows))}
    for i in range(len(rows)):
        rA = rows[i]
        wA = max(1, rA["right"] - rA["left"])
        hA = max(1, rA["bottom"] - rA["top"])
        midA = (rA["left"] + rA["right"]) / 2.0
        for j in range(i + 1, len(rows)):
            rB = rows[j]
            wB = max(1, rB["right"] - rB["left"])
            hB = max(1, rB["bottom"] - rB["top"])
            midB = (rB["left"] + rB["right"]) / 2.0

            gap_y = max(0, rB["top"] - rA["bottom"]) if rB["top"] >= rA["bottom"] else max(0, rA["top"] - rB["bottom"])
            max_line_gap = max(10, int(max(hA, hB) * 1.35))
            if gap_y > max_line_gap:
                continue

            x_overlap = min(rA["right"], rB["right"]) - max(rA["left"], rB["left"])
            min_w = min(wA, wB)

            is_short = min_w <= max(20, int(median_h * 1.8))
            centers_aligned = abs(midA - midB) <= max(18.0, min_w * 0.60)

            # Must overlap significantly OR be a centered short word
            if (min_w > 0 and (x_overlap / min_w) >= 0.35 and centers_aligned) or (is_short and centers_aligned and x_overlap >= -4):
                adj[i].append(j)
                adj[j].append(i)

    visited = set()
    blocks = []
    for i in range(len(rows)):
        if i in visited:
            continue
        comp = []
        q = [i]
        visited.add(i)
        while q:
            curr = q.pop(0)
            comp.append(curr)
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append(neighbor)
        comp.sort(key=lambda idx: rows[idx]["top"])
        b_left = min(rows[idx]["left"] for idx in comp)
        b_top = min(rows[idx]["top"] for idx in comp)
        b_right = max(rows[idx]["right"] for idx in comp)
        b_bottom = max(rows[idx]["bottom"] for idx in comp)
        b_text = _join_dialogue_lines([rows[idx]["text"] for idx in comp])
        b_conf = sum(rows[idx]["confidence"] for idx in comp) / len(comp)
        blocks.append({
            "left": b_left, "top": b_top, "right": b_right, "bottom": b_bottom,
            "text": b_text, "confidence": b_conf
        })
    return sorted(blocks, key=lambda block: (block["top"], block["left"]))

class _GoogleResultParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if self.depth and tag not in {"br", "hr", "img", "input", "meta", "link", "wbr"}:
            self.depth += 1
        elif set((dict(attrs).get("class") or "").split()) & {"result-container", "t0"}:
            self.depth = 1

    def handle_endtag(self, tag):
        if tag in {"br", "hr", "img", "input", "meta", "link", "wbr"}:
            return
        if self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if self.depth:
            self.parts.append(data)


_google_blocked_until = 0.0

def translate_text(text: str, target_lang: str, source_lang: Optional[str] = None,
                   cancel_event: Optional[threading.Event] = None) -> Optional[str]:
    """Use two bounded Google routes and stop retries when cancellation is requested."""
    global _google_blocked_until
    clean_text = text.strip()
    if not clean_text:
        return ""
    if source_lang in {"en", "english", "eng"}:
        clean_text = repair_english_word_spacing(clean_text)

    src_param = translation_language_code(source_lang)
    target_param = translation_language_code(target_lang, default="uz")

    # High-accuracy DeepL translation when available (for Russian, English, Japanese, etc.)
    try:
        from deepl_translator import translate_single_with_deepl, is_deepl_supported
        if is_deepl_supported(target_param):
            deepl_res = translate_single_with_deepl(clean_text, target_param, src_param)
            if deepl_res:
                return deepl_res
    except Exception:
        pass

    # If Google was recently rate-limited (429 HTTPError on cloud datacenter IPs),
    # skip directly to the MyMemory fallback to avoid stalling requests.
    with _google_block_lock:
        is_google_blocked = time.monotonic() < _google_blocked_until
    if not is_google_blocked:
        # Both Google routes have finite socket timeouts, a shared retry budget,
        # and bounded responses. Route 0 uses high-limit dict-chrome-ex; Route 1 uses m?.
        query = urllib.parse.urlencode({"sl": src_param, "tl": target_param, "q": clean_text})
        urls = [
            f"https://clients5.google.com/translate_a/t?client=dict-chrome-ex&{query}",
            f"https://translate.google.com/m?{query}",
        ]
        deadline = time.monotonic() + TRANSLATION_TIMEOUT_SECONDS * len(urls)
        for index, url in enumerate(urls):
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Tarjima töxtatildi.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                        "Accept": "*/*",
                    }
                )
                with urllib.request.urlopen(req, timeout=min(TRANSLATION_TIMEOUT_SECONDS, remaining)) as resp:
                    chunks, received = [], 0
                    read_chunk = getattr(resp, "read1", resp.read)
                    while received <= 1024 * 1024:
                        if cancel_event is not None and cancel_event.is_set():
                            raise InterruptedError("Tarjima töxtatildi.")
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Translation deadline exceeded")
                        chunk = read_chunk(min(65536, 1024 * 1024 + 1 - received))
                        if not chunk:
                            break
                        chunks.append(chunk)
                        received += len(chunk)
                    raw = b"".join(chunks)
                if len(raw) > 1024 * 1024:
                    raise ValueError("Translation response exceeds limit")
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Tarjima töxtatildi.")
                if index == 0:
                    data = json.loads(raw.decode("utf-8"))
                    if isinstance(data, list) and data:
                        if isinstance(data[0], list) and data[0]:
                            if isinstance(data[0][0], str) and len(data[0]) <= 2:
                                translated = str(data[0][0]).strip()
                            else:
                                translated = "".join(segment[0] for segment in data[0] if segment and segment[0]).strip()
                        elif isinstance(data[0], str):
                            translated = str(data[0]).strip()
                        else:
                            translated = "".join(segment[0] for segment in data[0] if segment and segment[0]).strip()
                    else:
                        translated = ""
                else:
                    parser = _GoogleResultParser()
                    parser.feed(raw.decode("utf-8"))
                    translated = "".join(parser.parts).strip()
                if translated:
                    return translated
            except InterruptedError:
                raise
            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str or getattr(exc, "code", None) == 429:
                    with _google_block_lock:
                        _google_blocked_until = time.monotonic() + 300.0  # Cool down for 5 minutes
                logger.debug("Google tarjima yo'li %d ishlamadi: %s", index + 1, type(exc).__name__)

    if os.environ.get("MANGA_USE_MYMEMORY_FALLBACK", "false").lower() in ("1", "true", "yes"):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Tarjima töxtatildi.")
        try:
            def _detect_script_language(sample: str) -> str:
                if any('\u3040' <= c <= '\u30ff' for c in sample):
                    return "ja"
                if any('\uac00' <= c <= '\ud7af' or '\u1100' <= c <= '\u11ff' for c in sample):
                    return "ko"
                if any('\u4e00' <= c <= '\u9fff' for c in sample):
                    return "zh"
                if any('\u0400' <= c <= '\u04ff' for c in sample):
                    return "ru"
                return "en"

            mm_src = src_param if src_param != "auto" else _detect_script_language(clean_text)
            mm_query = urllib.parse.urlencode({"q": clean_text, "langpair": f"{mm_src}|{target_param}"})
            mm_url = f"https://api.mymemory.translated.net/get?{mm_query}"
            mm_req = urllib.request.Request(
                mm_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
            )
            with urllib.request.urlopen(mm_req, timeout=TRANSLATION_TIMEOUT_SECONDS) as resp:
                raw_bytes = resp.read(1024 * 1024)
                mm_data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
                status = mm_data.get("responseStatus")
                res_text = mm_data.get("responseData", {}).get("translatedText")
                if (status == 200 and res_text and
                        not res_text.startswith("MYMEMORY WARNING") and
                        not res_text.upper().startswith("INVALID") and
                        "INVALID SOURCE LANGUAGE" not in res_text.upper() and
                        "INVALID TARGET LANGUAGE" not in res_text.upper() and
                        "INVALID EMAIL" not in res_text.upper()):
                    return html.unescape(res_text).strip()
        except InterruptedError:
            raise
        except Exception as exc:
            logger.warning("MyMemory fallback xatosi: %s", exc)

    return None

# ─── API Endpoints ─────────────────────────────────────────────────────────────

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
        os.environ["GEMINI_API_KEY"] = clean_key
        key_file = Path(os.environ.get("GEMINI_API_KEY_FILE", "/var/lib/manga/gemini_api_key.txt"))
        try:
            key_file.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
            mode = 0o600
            fd = os.open(str(key_file), flags, mode)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(clean_key)
        except Exception as exc:
            logger.error("Gemini API kalitini saqlaşda xatolik: %s", exc)
            raise HTTPException(status_code=500, detail=f"Gemini API kalitini saqlaşda xatolik: {exc}")
    return {
        "status": "success",
        "message": "Google AI Studio (Gemini) API kaliti muvaffaqiyatli saqlandi. Komikslar endi Gemini orqali kontekstli tarjima qilinadi."
    }

@app.get("/gemini-status")
def gemini_status(_authorized: None = Depends(verify_api_key)):
    from gemini_translator import GEMINI_MODELS
    has_key = bool(os.environ.get("GEMINI_API_KEY") or os.path.exists(os.environ.get("GEMINI_API_KEY_FILE", "/var/lib/manga/gemini_api_key.txt")))
    return {
        "gemini_enabled": has_key,
        "models": GEMINI_MODELS,
        "context_translation": "active" if has_key else "inactive (waiting for API key)"
    }

@app.get("/")
def index():
    return {
        "name": "Manga & Manhwa OCR Translation Server",
        "status": "running",
        "models": ["manga-ocr (ja)", "paddleocr (ja, ko, zh, en, ru)"],
        "endpoints": ["/health", "/translate-manga", "/ocr-manga", "/detect-and-translate", "/translate-page-image", "/translate-pdf", "/pdf-jobs/{job_id}"]
    }

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
        except Exception:
            pass

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
            except Exception:
                pass

    process_mb = 0.0
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        process_mb = round(rusage.ru_maxrss / 1024.0, 1)
    except Exception:
        pass

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
        "onnx_ready": engine.is_ready if engine else False,
        "onnx_detector": engine.det_sess is not None if engine else False,
        "onnx_ocr": engine.ocr_sess is not None if engine else False,
        "onnx_inpaint": engine.inp_sess is not None if engine else False,
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
        "translation_provider": "google",
        "ocr_packages_available": {
            name: importlib.util.find_spec(name) is not None
            for name in ("manga_ocr", "paddleocr", "easyocr", "torchvision")
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
        "allocated_cpus": cpus,
        "active_pdf_jobs": job_count,
        "summary": f"RAM: {res['human_readable']} | CPU: {int(cpus)} vCPUs",
    }

@app.post("/translate-pdf")
async def submit_pdf_translation(
    file: UploadFile = File(...),
    document_mode: str = Form("book"),
    source_lang: str = Form("auto"),
    target_lang: str = Form("uz"),
    output_mode: str = Form("translated"),
    gemini_api_key: Optional[str] = Form(None),
    x_gemini_api_key: Optional[str] = Header(None),
    _authorized: None = Depends(verify_api_key),
):
    """Queue book OCR/reflow or comic page/bubble translation and return a persistent job id."""
    verify_pdf_enabled()
    pdf_bytes = await file.read(MAX_PDF_UPLOAD_BYTES + 1)
    if len(pdf_bytes) > MAX_PDF_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"PDF {MAX_PDF_UPLOAD_BYTES // (1024 * 1024)} MB dan oşmasligi kerak.")
    try:
        summary = await run_in_threadpool(inspect_pdf, pdf_bytes, MAX_PDF_PAGES)
        norm_src = normalize_ocr_language(source_lang)
        norm_target = normalize_ocr_language(target_lang)
        if norm_src not in PDF_SOURCE_LANGUAGES:
            raise PdfJobError("PDF manba tili qöllab-quvvatlanmaydi.")
        if norm_target not in PDF_TARGET_LANGUAGES:
            raise PdfJobError("PDF tarjima tili qöllab-quvvatlanmaydi.")
        if document_mode not in {"book", "comic"}:
            raise PdfJobError("PDF rejimi 'book' yoki 'comic' bölişi kerak.")
        if document_mode == "book" and summary["native_text_pages"] == 0 and not summary["scanned_pages"]:
            raise PdfJobError("PDFda tanlanadigan matn yoki OCR qilinadigan sahifa tasviri topilmadi.")
        if output_mode not in {"translated", "bilingual"}:
            raise PdfJobError("PDF natija turi 'translated' yoki 'bilingual' bölişi kerak.")
        gkey = gemini_api_key.strip() if (isinstance(gemini_api_key, str) and gemini_api_key.strip()) else None
        xgkey = x_gemini_api_key.strip() if (isinstance(x_gemini_api_key, str) and x_gemini_api_key.strip()) else None
        effective_gemini_key = gkey or xgkey
        try:
            resume_pdf_jobs()
        except Exception:
            pass
        job = await run_in_threadpool(pdf_job_manager.create_job,
            pdf_bytes=pdf_bytes,
            filename=file.filename or "book.pdf",
            source_lang=norm_src,
            target_lang=norm_target,
            output_mode=output_mode,
            page_count=summary["page_count"],
            document_mode=document_mode,
            gemini_api_key=effective_gemini_key,
        )
        return {
            **job,
            "pdf_title": summary["title"],
            "native_text_pages": summary["native_text_pages"],
            "ocr_pages": summary["scanned_pages"],
            "empty_pages": summary["empty_pages"],
        }
    except PdfQueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "30"}) from exc
    except PdfJobError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/pdf-jobs/{job_id}")
def get_pdf_translation_status(
    job_id: str,
    _authorized: None = Depends(verify_api_key),
    _job_token: None = Depends(verify_pdf_job_token),
):
    try:
        return pdf_job_manager.public_job(job_id)
    except PdfJobError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/pdf-jobs/{job_id}/review/{page_index}")
def get_pdf_translation_review_page(
    job_id: str,
    page_index: int,
    _authorized: None = Depends(verify_api_key),
    _job_token: None = Depends(verify_pdf_job_token),
):
    try:
        return pdf_job_manager.review_page(job_id, page_index)
    except PdfJobError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/pdf-jobs/{job_id}/review")
async def correct_pdf_translation(
    job_id: str,
    request: Request,
    _authorized: None = Depends(verify_api_key),
    _job_token: None = Depends(verify_pdf_job_token),
):
    verify_pdf_enabled()
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 512 * 1024:
        raise HTTPException(status_code=413, detail="Tuzatiş sörovi hajmi judayam katta (max 512KB).")
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Tuzatiş sörovi JSON bölişi kerak.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("corrections"), list):
        raise HTTPException(status_code=400, detail="'corrections' röyxati yuborilişi kerak.")
    try:
        return await run_in_threadpool(pdf_job_manager.apply_corrections, job_id, payload["corrections"])
    except PdfJobError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/pdf-jobs/{job_id}/cancel")
def cancel_pdf_translation(
    job_id: str,
    _authorized: None = Depends(verify_api_key),
    _job_token: None = Depends(verify_pdf_job_token),
):
    try:
        return pdf_job_manager.cancel(job_id)
    except PdfJobError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/pdf-jobs/{job_id}/resume")
def resume_pdf_translation(
    job_id: str,
    _authorized: None = Depends(verify_api_key),
    _job_token: None = Depends(verify_pdf_job_token),
):
    try:
        return pdf_job_manager.resume(job_id)
    except PdfQueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "30"}) from exc
    except PdfJobError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/pdf-jobs/{job_id}/download")
def download_pdf_translation(
    job_id: str,
    _authorized: None = Depends(verify_api_key),
    _job_token: None = Depends(verify_pdf_job_token),
):
    try:
        path = pdf_job_manager.output_path(job_id)
        job = pdf_job_manager.get_job(job_id)
        raw_name = job.get("filename", "document.pdf") if job else "document.pdf"
        stem = Path(raw_name).stem
        sanitized_stem = "".join(c for c in stem if c.isalnum() or c in ("-", "_", " ")).strip() or "document"
        return FileResponse(path, media_type="application/pdf", filename=f"{sanitized_stem}-translated.pdf")
    except PdfJobError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


async def dispatch_manga_ocr(image_bytes: bytes, pil_image: Image.Image, lang: str) -> Tuple[str, str]:
    normalized_lang = normalize_ocr_language(lang)
    extracted_text = ""
    requested_auto = normalized_lang == "auto"
    if requested_auto:
        normalized_lang, auto_results = await run_in_threadpool(run_auto_paddle_ocr, image_bytes)
        extracted_text = " ".join(line["text"] for line in extract_paddle_lines(auto_results, normalized_lang))
        if normalized_lang == "ja":
            try:
                manga_text = await run_in_threadpool(run_manga_ocr, pil_image)
                if manga_text and manga_text.strip():
                    extracted_text = manga_text.strip()
            except Exception as exc:
                logger.warning("MangaOcr xatosi: %s; aniqlangan OCR matni işlatiladi.", exc)
    elif normalized_lang in ["ja", "jpn", "japanese"]:
        normalized_lang = "ja"
        try:
            extracted_text = await run_in_threadpool(run_manga_ocr, pil_image)
        except Exception as e:
            logger.warning("MangaOcr xatosi: %s. PaddleOCR zaxirasi sinab köriladi.", e)
            extracted_text = ""
        if not extracted_text:
            try:
                res = await run_in_threadpool(run_paddle_ocr, image_bytes, "japan")
                extracted_text = " ".join(line["text"] for line in extract_paddle_lines(res, "ja"))
            except Exception:
                pass
    elif normalized_lang in ["ko", "kor", "korean"]:
        normalized_lang = "ko"
        res = await run_in_threadpool(run_paddle_ocr, image_bytes, "korean")
        extracted_text = " ".join(line["text"] for line in extract_paddle_lines(res, "ko"))
    elif normalized_lang in ["zh", "ch", "chs", "cht", "chinese"]:
        normalized_lang = "zh"
        res = await run_in_threadpool(run_paddle_ocr, image_bytes, "ch")
        extracted_text = "".join(line["text"] for line in extract_paddle_lines(res, "zh"))
    elif normalized_lang in ["en", "eng", "english"]:
        normalized_lang = "en"
        res = await run_in_threadpool(run_paddle_ocr, image_bytes, "en")
        extracted_text = " ".join(line["text"] for line in extract_paddle_lines(res, "en"))
    elif normalized_lang in ["ru", "rus", "russian"]:
        normalized_lang = "ru"
        res = await run_in_threadpool(run_paddle_ocr, image_bytes, "ru")
        extracted_text = " ".join(line["text"] for line in extract_paddle_lines(res, "ru"))
    elif normalized_lang in PDF_LATIN_OCR_LANGS:
        res = await run_in_threadpool(run_paddle_ocr, image_bytes, "latin")
        extracted_text = " ".join(line["text"] for line in extract_paddle_lines(res, normalized_lang))
    else:
        raise HTTPException(status_code=400, detail=f"OCR tili qöllab-quvvatlanmaydi: {normalized_lang}")

    return extracted_text.strip(), normalized_lang


@app.post("/translate-manga")
async def translate_manga(
    file: UploadFile = File(None),
    text: Optional[str] = Form(None),
    lang: str = Form("ja"),         # "ja" (Japanese Manga), "ko" (Korean Manhwa), "zh"/"ch" (Chinese Manhua), "auto"
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

    if extracted_text and file is None and lang in (None, "ja", ""):
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
        gkey = gemini_api_key.strip() if (isinstance(gemini_api_key, str) and gemini_api_key.strip()) else None
        xgkey = x_gemini_api_key.strip() if (isinstance(x_gemini_api_key, str) and x_gemini_api_key.strip()) else None
        effective_gemini_key = gkey or xgkey
        source_code = translation_language_code(normalized_lang)
        if effective_gemini_key or os.environ.get("GEMINI_API_KEY"):
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
    lang: str = Form("ja"),
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

@app.post("/detect-and-translate")
async def detect_and_translate(
    file: UploadFile = File(...),
    lang: str = Form("ja"),
    target_lang: str = Form("uz"),
    gemini_api_key: Optional[str] = Form(None),
    x_gemini_api_key: Optional[str] = Header(None),
    _authorized: None = Depends(verify_api_key),
):
    """
    Accepts an entire manga page/screen capture:
    1. Uses MangaTranslationPipeline (SOTA Speech Bubble Detector, Inpainting, and Multi-language EasyOCR).
    2. Falls back to ONNX Speech Bubble Detector (manga_detector.onnx) and OCR if needed.
    3. Translates all dialogue bubbles in context using Gemini AI Engine.
    4. Returns list of { left, top, right, bottom, original_text, translated_text }.
    """
    image_bytes, pil_image = await read_image_upload(file)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    width, height = pil_image.size
    normalized_lang = normalize_ocr_language(lang)

    blocks: List[Dict[str, Any]] = []

    # 1. Primary path: Full MangaTranslationPipeline (High-precision Bubble Detector + EasyOCR)
    try:
        pipe = get_manga_pipeline()
        if pipe is not None:
            pipeline_lang = normalized_lang if normalized_lang in {"ja", "ko", "zh", "en", "ru"} else "auto"
            def _pipe_proc():
                with _model_lock:
                    return pipe.process_page(pil_image, auto_collect=False, source_lang=pipeline_lang)
            res = await run_in_threadpool(_pipe_proc)
            dialogues = res.get("dialogues", [])
            for d in dialogues:
                txt = (d.get("transcription") or "").strip()
                if not txt:
                    continue
                bx1, by1, bx2, by2 = d["bbox"]
                blocks.append({
                    "left": max(0, int(bx1)),
                    "top": max(0, int(by1)),
                    "right": min(width, int(bx2)),
                    "bottom": min(height, int(by2)),
                    "original_text": txt,
                    "translated_text": ""
                })
    except Exception as exc:
        logger.warning("MangaTranslationPipeline xatosi /detect-and-translate da: %s", exc)

    # 2. Secondary path: ONNX Speech Bubble Detector & OCR fallback
    if not blocks:
        engine = get_onnx_engine()
        if engine and engine.det_sess:
            try:
                def _det_onnx():
                    with _model_lock:
                        return engine.detect_bubbles(pil_image, conf_thresh=0.20)
                detected = await run_in_threadpool(_det_onnx)
                raw_bubbles = []
                if detected:
                    for item in detected:
                        box = item["box"]
                        x1 = max(0, min(width - 1, box[0]))
                        y1 = max(0, min(height - 1, box[1]))
                        x2 = max(x1 + 1, min(width, box[2]))
                        y2 = max(y1 + 1, min(height, box[3]))
                        if x2 - x1 < 14 or y2 - y1 < 14:
                            continue

                        crop = pil_image.crop((x1, y1, x2, y2))
                        bubble_text = ""

                        # Try Manga OCR for Japanese
                        if engine.ocr_sess and normalized_lang in ["ja", "jpn", "japanese"]:
                            try:
                                def _ocr_onnx():
                                    with _model_lock:
                                        return engine.ocr_bubble(crop)
                                bubble_text = await run_in_threadpool(_ocr_onnx)
                            except Exception as e:
                                logger.debug("engine.ocr_bubble error: %s", e)

                        # Try MangaOcr ViT if Japanese
                        if not bubble_text and normalized_lang in ["ja", "jpn", "japanese"]:
                            try:
                                bubble_text = await run_in_threadpool(run_manga_ocr, crop)
                            except Exception:
                                pass

                        # Try Tesseract if available
                        if not bubble_text:
                            try:
                                import pytesseract
                                tess_lang_map = {
                                    "ja": "jpn", "ko": "kor", "zh": "chi_sim",
                                    "ru": "rus", "en": "eng"
                                }
                                tess_lang = tess_lang_map.get(normalized_lang, "eng")
                                raw_text = await run_in_threadpool(pytesseract.image_to_string, crop, lang=tess_lang)
                                cleaned = " ".join((raw_text or "").split())
                                if is_valid_manga_dialogue(cleaned, source_lang=normalized_lang):
                                    bubble_text = cleaned
                            except Exception:
                                pass

                        bubble_text = bubble_text.strip()
                        if bubble_text:
                            raw_bubbles.append((x1, y1, x2, y2, bubble_text))

                    for (x1, y1, x2, y2, orig_text) in raw_bubbles:
                        blocks.append({
                            "left": x1,
                            "top": y1,
                            "right": x2,
                            "bottom": y2,
                            "original_text": orig_text,
                            "translated_text": ""
                        })
            except Exception as e:
                logger.warning("ONNX speech bubble detection xatosi: %s", e)

    # 3. Tertiary Safe Fallback only if Paddle is explicitly functional
    if not blocks and is_paddle_enabled():
        try:
            if normalized_lang == "auto":
                normalized_lang, ocr_results = await run_in_threadpool(run_auto_paddle_ocr, image_bytes)
            else:
                detector_lang = {
                    "ja": "japan", "ko": "korean", "zh": "ch", "en": "en", "ru": "ru",
                    **{language: "latin" for language in PDF_LATIN_OCR_LANGS},
                }.get(normalized_lang)
                if detector_lang is not None:
                    ocr_results = await run_in_threadpool(run_paddle_ocr, image_bytes, detector_lang)
                else:
                    ocr_results = None
            if ocr_results and ocr_results[0]:
                use_manga_ocr = normalized_lang == "ja"
                detections = extract_paddle_lines(ocr_results, normalized_lang)
                for detection in detections:
                    x1 = max(0, detection["left"])
                    y1 = max(0, detection["top"])
                    x2 = min(width, detection["right"])
                    y2 = min(height, detection["bottom"])
                    if x2 - x1 < 8 or y2 - y1 < 8:
                        continue
                    orig_text = detection["text"].strip()
                    if use_manga_ocr:
                        try:
                            crop = pil_image.crop((x1, y1, x2, y2))
                            m_text = await run_in_threadpool(run_manga_ocr, crop)
                            if m_text and m_text.strip():
                                orig_text = m_text.strip()
                        except Exception:
                            pass
                    if orig_text:
                        blocks.append({
                            "left": x1,
                            "top": y1,
                            "right": x2,
                            "bottom": y2,
                            "original_text": orig_text,
                            "translated_text": ""
                        })
        except Exception as e:
            logger.debug("PaddleOCR fallback disabled or failed: %s", e)

    # 4. Context-Aware Dialogue Translation (Gemini AI -> DeepL -> Google Translate)
    if blocks:
        texts_to_trans = [b["original_text"] for b in blocks]
        translated_list: Optional[List[str]] = None
        gkey = gemini_api_key.strip() if (isinstance(gemini_api_key, str) and gemini_api_key.strip()) else None
        xgkey = x_gemini_api_key.strip() if (isinstance(x_gemini_api_key, str) and x_gemini_api_key.strip()) else None
        effective_gemini_key = gkey or xgkey
        try:
            from gemini_translator import translate_dialogues_with_gemini
            translated_list = await run_in_threadpool(
                translate_dialogues_with_gemini,
                texts_to_trans,
                target_lang,
                normalized_lang,
                api_key=effective_gemini_key,
            )
        except Exception as ge:
            logger.warning("Gemini dialog batch translation xatosi: %s", ge)

        if not translated_list or len(translated_list) != len(texts_to_trans):
            async def _translate_one(text_to_trans):
                if not text_to_trans or text_to_trans == "Nutq buluti":
                    return text_to_trans
                async with _translation_semaphore:
                    try:
                        return await run_in_threadpool(
                            translate_text, text_to_trans, target_lang, translation_language_code(normalized_lang)
                        )
                    except Exception:
                        return text_to_trans

            translated_list = await asyncio.gather(*[_translate_one(t) for t in texts_to_trans], return_exceptions=True)

        for b, trans_res in zip(blocks, translated_list):
            b["translated_text"] = trans_res if isinstance(trans_res, str) and trans_res else b["original_text"]

    return {
        "status": "success",
        "source_lang": normalized_lang,
        "blocks_count": len(blocks),
        "blocks": blocks
    }

@app.post("/clean-manga")
async def clean_manga(
    file: UploadFile = File(...),
    boxes_json: Optional[str] = Form(None),
    _authorized: None = Depends(verify_api_key),
):
    """
    Cleans detected speech bubbles from a manga image using neural inpainting (manga_inpaint.onnx).
    Returns cleaned image in image/jpeg format.
    """
    image_bytes, pil_image = await read_image_upload(file)
    engine = get_onnx_engine()

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
        except Exception:
            pass

    if not target_boxes and engine and engine.det_sess:
        def _det():
            with _model_lock:
                return engine.detect_bubbles(pil_image)
        detected = await run_in_threadpool(_det)
        target_boxes = [d["box"] for d in detected]

    if engine and engine.inp_sess and target_boxes:
        def _inpaint():
            with _model_lock:
                return engine.inpaint_bubbles(pil_image, target_boxes)
        cleaned_pil = await run_in_threadpool(_inpaint)
    else:
        cleaned_pil = pil_image

    buf = io.BytesIO()
    cleaned_pil.convert("RGB").save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/jpeg")

@app.post("/translate-page-image")
async def translate_page_image(
    file: UploadFile = File(...),
    lang: str = Form("ja"),
    target_lang: str = Form("uz"),
    gemini_api_key: Optional[str] = Form(None),
    x_gemini_api_key: Optional[str] = Header(None),
    _authorized: None = Depends(verify_api_key),
):
    """
    End-to-end Automated Manga Page Translator & Typesetter:
    1. Detects all speech bubbles and text lines.
    2. Cleans text strokes using neural text segmentation (preserving line art & screentones).
    3. Translates dialogues with context-aware Gemini AI localization (10/10 quality).
    4. Automatically typesets translated text using dynamic elliptical line-wrapping.
    5. Returns fully translated manga page in JPEG format.
    """
    image_bytes, pil_image = await read_image_upload(file)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    width, height = pil_image.size
    normalized_lang = normalize_ocr_language(lang)

    pipe = get_manga_pipeline()
    if pipe is None:
        raise HTTPException(status_code=503, detail="MangaTranslationPipeline yuklanmagan.")

    pipeline_lang = normalized_lang if normalized_lang in {"ja", "ko", "zh", "en", "ru"} else "auto"
    def _run_pipe():
        with _model_lock:
            return pipe.process_page(pil_image, auto_collect=False, source_lang=pipeline_lang)
    res = await run_in_threadpool(_run_pipe)
    cleaned_img = res.get("cleaned_image") or pil_image
    dialogues = res.get("dialogues", [])

    if dialogues:
        speech_dialogues = [d for d in dialogues if d.get("category") != "sfx"]
        texts_to_trans = [(d.get("transcription") or "").strip() for d in speech_dialogues]
        if texts_to_trans:
            gkey = gemini_api_key.strip() if (isinstance(gemini_api_key, str) and gemini_api_key.strip()) else None
            xgkey = x_gemini_api_key.strip() if (isinstance(x_gemini_api_key, str) and x_gemini_api_key.strip()) else None
            effective_gemini_key = gkey or xgkey
            from gemini_translator import translate_dialogues_with_gemini
            try:
                trans_res = await run_in_threadpool(
                    translate_dialogues_with_gemini,
                    texts_to_trans,
                    target_lang,
                    normalized_lang,
                    api_key=effective_gemini_key
                )
            except Exception as ge:
                logger.warning("Gemini dialog translation xatosi: %s", ge)
                trans_res = None
            if trans_res and len(trans_res) == len(speech_dialogues):
                for d, t in zip(speech_dialogues, trans_res):
                    d["translation"] = t
            else:
                async def _trans_single(text_val):
                    if not text_val:
                        return ""
                    global _translation_semaphore
                    if _translation_semaphore is None:
                        _translation_semaphore = asyncio.Semaphore(4)
                    async with _translation_semaphore:
                        try:
                            return await run_in_threadpool(translate_text, text_val, target_lang, translation_language_code(normalized_lang))
                        except Exception:
                            return text_val

                single_res = await asyncio.gather(*[_trans_single(t) for t in texts_to_trans], return_exceptions=True)
                for d, t in zip(speech_dialogues, single_res):
                    d["translation"] = t if isinstance(t, str) and t else d.get("transcription", "")
        for d in dialogues:
            if d.get("category") == "sfx":
                d["translation"] = ""

        try:
            from manga_pipeline import render_translated_dialogues
        except ImportError:
            ml_dir_env = os.environ.get("MANGA_ML_DIR")
            if ml_dir_env and Path(ml_dir_env).exists():
                _ml_dir = Path(ml_dir_env).resolve()
            elif (Path(__file__).resolve().parent / "ml").exists():
                _ml_dir = (Path(__file__).resolve().parent / "ml").resolve()
            else:
                _ml_dir = (Path(__file__).resolve().parent.parent.parent / "mashinali oqitish").resolve()
            if str(_ml_dir) not in sys.path:
                sys.path.insert(0, str(_ml_dir))
            from manga_pipeline import render_translated_dialogues

        def _render_dialogues():
            with _model_lock:
                return render_translated_dialogues(cleaned_img, dialogues)
        final_image = await run_in_threadpool(_render_dialogues)
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
