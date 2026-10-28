import io
import hmac
import importlib
import json
import logging
import os
import sys
import threading
import time
from html.parser import HTMLParser
import urllib.parse
import urllib.request
import re
import unicodedata
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends, Request
from fastapi.responses import FileResponse
from PIL import Image
import pymupdf
import wordninja
from starlette.concurrency import run_in_threadpool
from pathlib import Path
from pdf_jobs import PdfJobError, PdfQueueFullError, PdfJobManager, inspect_pdf

# Load Torch before PaddleOCR; both ship OpenMP DLLs that can conflict on Windows.
import torch

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
ALLOW_UNAUTHENTICATED = os.environ.get("MANGA_ALLOW_UNAUTHENTICATED", "").strip().lower() in {"1", "true", "yes"}
TRANSLATION_TIMEOUT_SECONDS = max(1.0, min(30.0, float(os.environ.get("MANGA_TRANSLATION_TIMEOUT_SECONDS", "8"))))
MAX_UPLOAD_BYTES = max(1024, int(os.environ.get("MANGA_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))))
MAX_IMAGE_PIXELS = max(1024 * 1024, int(os.environ.get("MANGA_MAX_IMAGE_PIXELS", "25000000")))
MAX_PDF_UPLOAD_BYTES = max(1024 * 1024, int(os.environ.get("MANGA_MAX_PDF_UPLOAD_BYTES", str(50 * 1024 * 1024))))
MAX_PDF_PAGES = max(1, int(os.environ.get("MANGA_MAX_PDF_PAGES", "500")))
PDF_OCR_DPI = max(96, min(300, int(os.environ.get("MANGA_PDF_OCR_DPI", "180"))))
PDF_OCR_MAX_PIXELS = max(1_000_000, int(os.environ.get("MANGA_PDF_OCR_MAX_PIXELS", str(16 * 1024 * 1024))))
PDF_JOB_DIR = Path(os.environ.get("MANGA_PDF_JOB_DIR", str(Path(__file__).parent / "pdf_job_data")))
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
_model_lock = threading.RLock()


def get_manga_pipeline():
    """Lazy-load MangaTranslationPipeline (Speech Bubble Detector, Inpainting, and Multi-language OCR)
    directly from 'mashinali oqitish' directory to avoid duplicate code and checkpoint models."""
    global _manga_pipeline, _manga_pipeline_error
    with _model_lock:
        if _manga_pipeline is None:
            try:
                ml_dir = Path(os.environ.get("MANGA_ML_DIR", str(Path(__file__).resolve().parent.parent.parent / "mashinali oqitish"))).resolve()
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
                logger.error(f"MangaTranslationPipeline yuklashda xatolik: {e}")
                raise HTTPException(status_code=500, detail=f"Manga AI modelini yuklab bo'lmadi: {e}")
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
        raise HTTPException(status_code=503, detail="Server API kaliti sozlanmagan. Mahalliy sinov uchun MANGA_ALLOW_UNAUTHENTICATED=true ni belgilang.")
    if SERVER_API_KEY and (not x_api_key or not hmac.compare_digest(x_api_key, SERVER_API_KEY)):
        raise HTTPException(status_code=401, detail="Server API kaliti noto'g'ri yoki kiritilmagan.")


def verify_pdf_job_token(job_id: str, x_pdf_job_token: Optional[str] = Header(default=None)):
    if not pdf_job_manager.verify_job_token(job_id, x_pdf_job_token):
        raise HTTPException(status_code=401, detail="PDF vazifasi kaliti noto'g'ri yoki kiritilmagan.")


@app.on_event("startup")
def resume_pdf_jobs():
    if SERVER_API_KEY or ALLOW_UNAUTHENTICATED:
        pdf_job_manager.start_pending(translate_text, ocr_pdf_page)


@app.on_event("shutdown")
def stop_pdf_job_worker():
    pdf_job_manager.shutdown()


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
        raise HTTPException(status_code=400, detail="Bo'sh rasm fayli yuborildi.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.width <= 0 or image.height <= 0 or image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Rasm o'lchami ruxsat etilgan chegaradan katta.")
            return image.convert("RGB")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Rasmni o'qib bo'lmadi: {exc}") from exc


async def read_image_upload(file: UploadFile):
    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Rasm fayli {MAX_UPLOAD_BYTES // (1024 * 1024)} MB dan oshmasligi kerak.")
    return image_bytes, decode_image(image_bytes)


def get_manga_ocr():
    """Lazy-load MangaOcr (Japanese Vision Transformer model)."""
    global _mocr
    with _model_lock:
        if _mocr is None:
            try:
                logger.info("MangaOcr (Japanese ViT) modelini yuklash...")
                manga_ocr_mod = importlib.import_module("manga_ocr")
                manga_ocr_class = getattr(manga_ocr_mod, "MangaOcr")
                _mocr = manga_ocr_class()
                logger.info("MangaOcr muvaffaqiyatli yuklandi.")
            except ModuleNotFoundError:
                err = "manga-ocr kutubxonasi o'rnatilmagan! Iltimos, 'pip install manga-ocr' yoki 'run_server.bat' ni ishga tushiring."
                logger.error(err)
                raise HTTPException(status_code=500, detail=err)
            except Exception as e:
                logger.error(f"MangaOcr yuklashda xatolik: {e}")
                raise HTTPException(status_code=500, detail=f"MangaOcr yuklab bo'lmadi: {str(e)}")
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
            err = f"paddleocr kutubxonasi o'rnatilmagan! Iltimos, 'pip install paddleocr paddlepaddle' yoki 'run_server.bat' ni ishga tushiring."
            logger.error(err)
            raise HTTPException(status_code=500, detail=err)
        except Exception as e:
            logger.error(f"PaddleOCR ({lang}) yuklashda xatolik: {e}")
            raise HTTPException(status_code=500, detail=f"PaddleOCR yuklab bo'lmadi: {str(e)}")


def run_manga_ocr(image: Image.Image) -> str:
    with _model_lock:
        return get_manga_ocr()(image)


def run_paddle_ocr(image_bytes: bytes, lang: str):
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
            buffer = io.BytesIO()
            resized.save(buffer, format="PNG")
            results = get_paddle_ocr(lang).ocr(buffer.getvalue(), cls=True)
            if results and results[0]:
                results[0] = [
                    ([[float(point[0]) / scale_x, float(point[1]) / scale_y] for point in item[0]], item[1])
                    for item in results[0]
                ]
            return results
        return get_paddle_ocr(lang).ocr(image_bytes, cls=True)


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
        "ja": kana * 3 + han,
        "zh": han if kana == 0 else 0,
    }.get(language, 0)
    purity = min(1.0, expected / max(1, len(letters)))
    coverage = min(expected, 20) / 20.0
    return purity * 2.0 + coverage + max(0.0, min(1.0, confidence)) * 0.35


def run_auto_paddle_ocr(image_bytes: bytes):
    """Try Latin and non-Latin models; leave actual Latin language detection to translation."""
    candidates = [("auto", "latin"), ("en", "en"), ("ru", "ru"), ("ko", "korean"), ("ja", "japan"), ("zh", "ch")]
    best_language, best_results, best_score = "en", [], 0.0
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
            if not text:
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
    left, top, right, bottom = box
    pad = max(3, min(12, int(min(right - left, bottom - top) * 0.15)))
    x0, y0 = max(0, left - pad), max(0, top - pad)
    x1, y1 = min(image.width, right + pad), min(image.height, bottom + pad)
    pixels = image.load()
    samples = []
    for x in range(x0, x1, 3):
        for y in (range(y0, min(top, y1), 3), range(max(bottom, y0), y1, 3)):
            for py in y:
                red, green, blue = pixels[x, py][:3]
                if (red + green + blue) / 3 >= 170:
                    samples.append((red, green, blue))
    for y in range(y0, y1, 3):
        for x in (range(x0, min(left, x1), 3), range(max(right, x0), x1, 3)):
            for px in x:
                red, green, blue = pixels[px, y][:3]
                if (red + green + blue) / 3 >= 170:
                    samples.append((red, green, blue))
    if not samples:
        return [1.0, 1.0, 1.0]
    samples.sort(key=lambda pixel: sum(pixel))
    median = samples[len(samples) // 2]
    return [max(0.72, channel / 255.0) for channel in median]


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
            threshold_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]
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
        logger.debug("Komiks pufagi chegarasini topib bo'lmadi: %s", exc)
        return None


def ocr_pdf_page(page, source_lang: str, document_mode: str = "book", page_index: int = 0, job_dir: Optional[Path] = None) -> dict:
    """
    Renders one PDF page and runs either:
    1. Comic Mode: State-of-the-Art ML Manga Translation Pipeline (MangaBubbleDetector,
       EasyOCR, and Neural Inpainting for clean text removal).
    2. Book Mode: PaddleOCR line recognition, layout extraction, and reflow blocks.
    """
    global _manga_pipeline_error
    page_area = max(1.0, page.rect.get_area())
    target_scale = PDF_OCR_DPI / 72.0
    pixel_limited_scale = (PDF_OCR_MAX_PIXELS / page_area) ** 0.5
    scale = min(target_scale, pixel_limited_scale)
    if scale < 0.1:
        raise PdfJobError("PDF bet o'lchami OCR uchun juda katta.")
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    image_bytes = pixmap.tobytes("png")

    with Image.open(io.BytesIO(image_bytes)) as opened_image:
        rendered_image = opened_image.convert("RGB")

    normalized_lang = normalize_ocr_language(source_lang)

    # ─── 1. COMIC MODE: ML Bubble Detection & Neural Inpainting Text Cleaning ───
    if document_mode == "comic":
        try:
            pipe = get_manga_pipeline()
            pipeline_res = pipe.process_page(rendered_image, auto_collect=False, source_lang=normalized_lang)
            cleaned_img = pipeline_res.get("cleaned_image")
            dialogues = pipeline_res.get("dialogues", [])
            cleaned_ids = set(pipeline_res.get("cleaned_bubble_ids", []))
            if pipeline_res.get("requires_ocr_fallback"):
                raise PdfJobError("ML OCR natijasi ishonchsiz; zaxira OCR tekshiriladi.")

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
            if job_dir is not None and cleaned_img is not None:
                cleaned_filename = f"page_{page_index}_cleaned.png"
                cleaned_img.save(job_dir / cleaned_filename, format="PNG")
            detected_lang = normalize_ocr_language(pipeline_res.get("detected_source_lang") or normalized_lang)
            if normalized_lang != "auto":
                detected_lang = normalized_lang
            if detected_lang not in PDF_SOURCE_LANGUAGES:
                detected_lang = "auto"
            _manga_pipeline_error = None
            return {
                "blocks": blocks,
                "detected_source_lang": detected_lang,
                "cleaned_image_filename": cleaned_filename
            }
        except Exception as exc:
            _manga_pipeline_error = type(exc).__name__
            logger.warning("ML Manga Pipeline ishlovida xatolik: %s. Zaxira OCR rejimiga o'tilmoqda...", exc)

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
            raise PdfJobError(f"PDF OCR tili qo'llab-quvvatlanmaydi: {normalized_lang}")
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
            logger.debug("Pufak konturi uchun rasmni tayyorlab bo'lmadi: %s", exc)

    blocks = []
    for line in lines:
        if not line["text"].strip():
            continue
        left = max(0, min(rendered_image.width, line["left"]))
        top = max(0, min(rendered_image.height, line["top"]))
        right = max(left, min(rendered_image.width, line["right"]))
        bottom = max(top, min(rendered_image.height, line["bottom"]))
        original_text = line["text"]
        if document_mode == "comic" and detected_lang == "ja" and right > left and bottom > top:
            try:
                refined = run_manga_ocr(rendered_image.crop((left, top, right, bottom))).strip()
                if refined:
                    original_text = refined
            except Exception as exc:
                logger.warning("PDF komiksidagi MangaOCR xatosi: %s", exc)
        block = {
            "text": original_text,
            "font_size": 0.0,
            "confidence": line["confidence"],
            "bbox": [left / scale, top / scale, right / scale, bottom / scale],
        }
        if document_mode == "comic":
            overlay_region = _find_bubble_overlay_region(
                rendered_image,
                (left, top, right, bottom),
                bubble_threshold_mask,
            )
            if overlay_region:
                block["overlay_bbox"] = [coordinate / scale for coordinate in overlay_region["bbox"]]
                block["overlay_polygon"] = [
                    [x / scale, y / scale] for x, y in overlay_region["polygon"]
                ]
            block["background"] = _sample_comic_background(rendered_image, (left, top, right, bottom))
        blocks.append(block)
    return {"blocks": blocks, "detected_source_lang": detected_lang}


def repair_english_word_spacing(text: str) -> str:
    """Split confidently recognized English words that OCR joined together."""
    word_costs = wordninja.DEFAULT_LANGUAGE_MODEL._wordcost

    def split_token(match):
        token = match.group(0)
        if token.lower() in word_costs:
            return token
        parts = wordninja.split(token)
        if len(parts) < 2:
            return token
        normalized_parts = [part.lower().strip("'") for part in parts]
        if any(not part or (len(part) == 1 and part not in {"a", "i"}) for part in normalized_parts):
            return token
        if any(part not in word_costs for part in normalized_parts):
            return token
        return " ".join(parts)

    return re.sub(r"[A-Za-z]{5,}", split_token, text)


def merge_horizontal_ocr_lines(items):
    """Join word boxes into readable lines, then nearby lines into dialogue blocks."""
    if not items:
        return []

    median_height = sorted(max(1, item["bottom"] - item["top"]) for item in items)[len(items) // 2]
    row_tolerance = max(5.0, median_height * 0.65)
    rows = []
    for item in sorted(items, key=lambda value: ((value["top"] + value["bottom"]) / 2, value["left"])):
        center_y = (item["top"] + item["bottom"]) / 2
        matching_rows = [
            row for row in rows
            if abs(center_y - row["center_y"]) <= max(row_tolerance, row["height"] * 0.35)
            and max(0, item["left"] - max(entry["right"] for entry in row["items"]),
                    min(entry["left"] for entry in row["items"]) - item["right"])
            <= max(24, median_height * 2.2)
        ]
        if not matching_rows:
            rows.append({"items": [item], "center_y": center_y, "height": item["bottom"] - item["top"]})
        else:
            row = min(matching_rows, key=lambda candidate: abs(center_y - candidate["center_y"]))
            row["items"].append(item)
            row["top"] = min(entry["top"] for entry in row["items"])
            row["bottom"] = max(entry["bottom"] for entry in row["items"])
            row["center_y"] = (row["top"] + row["bottom"]) / 2
            row["height"] = row["bottom"] - row["top"]

    for row in rows:
        ordered = sorted(row["items"], key=lambda item: item["left"])
        row.update({
            "left": min(item["left"] for item in ordered),
            "top": min(item["top"] for item in ordered),
            "right": max(item["right"] for item in ordered),
            "bottom": max(item["bottom"] for item in ordered),
            "text": " ".join(item["text"] for item in ordered),
            "confidence": sum(item["confidence"] for item in ordered) / len(ordered),
        })

    rows.sort(key=lambda row: (row["top"], row["left"]))
    adjacency = [[] for _ in rows]
    for i, first in enumerate(rows):
        for j in range(i + 1, len(rows)):
            second = rows[j]
            vertical_gap = max(0, second["top"] - first["bottom"])
            max_gap = max(12, median_height * 1.8)
            horizontal_overlap = min(first["right"], second["right"]) - max(first["left"], second["left"])
            centers_close = abs((first["left"] + first["right"]) / 2 - (second["left"] + second["right"]) / 2) <= max(30, median_height * 2.7)
            aligned = horizontal_overlap >= -max(14, median_height * 1.25) or centers_close
            if vertical_gap <= max_gap and aligned:
                adjacency[i].append(j)
                adjacency[j].append(i)

    blocks = []
    visited = set()
    for start in range(len(rows)):
        if start in visited:
            continue
        stack, component = [start], []
        visited.add(start)
        while stack:
            index = stack.pop()
            component.append(rows[index])
            for neighbor in adjacency[index]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        component.sort(key=lambda row: (row["top"], row["left"]))
        blocks.append({
            "left": min(row["left"] for row in component),
            "top": min(row["top"] for row in component),
            "right": max(row["right"] for row in component),
            "bottom": max(row["bottom"] for row in component),
            "text": " ".join(row["text"] for row in component).strip(),
            "confidence": sum(row["confidence"] for row in component) / len(component),
        })
    return sorted(blocks, key=lambda block: (block["top"], block["left"]))

class _GoogleResultParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if self.depth:
            self.depth += 1
        elif set(dict(attrs).get("class", "").split()) & {"result-container", "t0"}:
            self.depth = 1

    def handle_endtag(self, tag):
        if self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if self.depth:
            self.parts.append(data)


def translate_text(text: str, target_lang: str, source_lang: Optional[str] = None,
                   cancel_event: Optional[threading.Event] = None) -> Optional[str]:
    """Translates text using built-in Google Translate API (zero external dependencies),
    with automatic fallbacks."""
    clean_text = text.strip()
    if not clean_text:
        return ""

    src_param = translation_language_code(source_lang)
    target_param = translation_language_code(target_lang, default="uz")

    # Both Google routes have finite socket timeouts, a shared retry budget,
    # and bounded responses. Avoid deep-translator's unbounded requests.get.
    query = urllib.parse.urlencode({"sl": src_param, "tl": target_param, "q": clean_text})
    urls = [
        f"https://translate.googleapis.com/translate_a/single?client=gtx&dt=t&{query}",
        f"https://translate.google.com/m?{query}",
    ]
    deadline = time.monotonic() + TRANSLATION_TIMEOUT_SECONDS * len(urls)
    for index, url in enumerate(urls):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Tarjima to'xtatildi.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=min(TRANSLATION_TIMEOUT_SECONDS, remaining)) as resp:
                chunks, received = [], 0
                # read1 returns available bytes so a slowly trickling response
                # cannot reset the shared deadline indefinitely.
                read_chunk = getattr(resp, "read1", resp.read)
                while received <= 1024 * 1024:
                    if cancel_event is not None and cancel_event.is_set():
                        raise InterruptedError("Tarjima to'xtatildi.")
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
                raise InterruptedError("Tarjima to'xtatildi.")
            if index == 0:
                data = json.loads(raw.decode("utf-8"))
                translated = "".join(segment[0] for segment in data[0] if segment and segment[0]).strip()
            else:
                parser = _GoogleResultParser()
                parser.feed(raw.decode("utf-8"))
                translated = "".join(parser.parts).strip()
            if translated:
                return translated
        except InterruptedError:
            raise
        except Exception as exc:
            logger.warning("Google tarjima yo'li %d ishlamadi: %s", index + 1, type(exc).__name__)

    return None

# ─── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return {
        "name": "Manga & Manhwa OCR Translation Server",
        "status": "running",
        "models": ["manga-ocr (ja)", "paddleocr (ja, ko, zh, en, ru)"],
        "endpoints": ["/health", "/translate-manga", "/ocr-manga", "/detect-and-translate", "/translate-pdf", "/pdf-jobs/{job_id}"]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
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
        "translation_ready": bool(SERVER_API_KEY or ALLOW_UNAUTHENTICATED),
        "api_key_required": bool(SERVER_API_KEY),
        "pdf_translation_enabled": True,
        "pdf_modes": ["book", "comic"],
        "pdf_source_languages": list(PDF_SOURCE_LANGUAGES),
        "pdf_target_languages": list(PDF_TARGET_LANGUAGES),
        "pdf_max_upload_bytes": MAX_PDF_UPLOAD_BYTES,
        "pdf_max_pages": MAX_PDF_PAGES,
        "pdf_ocr_dpi": PDF_OCR_DPI,
        "pdf_ocr_max_pixels": PDF_OCR_MAX_PIXELS,
    }

@app.post("/translate-pdf")
async def submit_pdf_translation(
    file: UploadFile = File(...),
    document_mode: str = Form("book"),
    source_lang: str = Form("auto"),
    target_lang: str = Form("uz"),
    output_mode: str = Form("translated"),
    _authorized: None = Depends(verify_api_key),
):
    """Queue book OCR/reflow or comic page/bubble translation and return a persistent job id."""
    pdf_bytes = await file.read(MAX_PDF_UPLOAD_BYTES + 1)
    if len(pdf_bytes) > MAX_PDF_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"PDF {MAX_PDF_UPLOAD_BYTES // (1024 * 1024)} MB dan oshmasligi kerak.")
    try:
        summary = await run_in_threadpool(inspect_pdf, pdf_bytes, MAX_PDF_PAGES)
        if normalize_ocr_language(source_lang) not in PDF_SOURCE_LANGUAGES:
            raise PdfJobError("PDF manba tili qo'llab-quvvatlanmaydi.")
        if normalize_ocr_language(target_lang) not in PDF_TARGET_LANGUAGES:
            raise PdfJobError("PDF tarjima tili qo'llab-quvvatlanmaydi.")
        if document_mode not in {"book", "comic"}:
            raise PdfJobError("PDF rejimi 'book' yoki 'comic' bo'lishi kerak.")
        if document_mode == "book" and summary["native_text_pages"] == 0 and not summary["scanned_pages"]:
            raise PdfJobError("PDFda tanlanadigan matn yoki OCR qilinadigan sahifa tasviri topilmadi.")
        if output_mode not in {"translated", "bilingual"}:
            raise PdfJobError("PDF natija turi 'translated' yoki 'bilingual' bo'lishi kerak.")
        job = pdf_job_manager.create_job(
            pdf_bytes=pdf_bytes,
            filename=file.filename or "book.pdf",
            source_lang=translation_language_code(source_lang),
            target_lang=translation_language_code(target_lang, default="uz"),
            output_mode=output_mode,
            page_count=summary["page_count"],
            document_mode=document_mode,
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
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Tuzatish so‘rovi JSON bo‘lishi kerak.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("corrections"), list):
        raise HTTPException(status_code=400, detail="'corrections' ro‘yxati yuborilishi kerak.")
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
        return FileResponse(path, media_type="application/pdf", filename=f"{path.parent.name}-translated.pdf")
    except PdfJobError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/translate-manga")
async def translate_manga(
    file: UploadFile = File(...),
    lang: str = Form("ja"),         # "ja" (Japanese Manga), "ko" (Korean Manhwa), "zh"/"ch" (Chinese Manhua), "auto"
    target_lang: str = Form("uz"),  # "uz", "ru", "en", etc.
    _authorized: None = Depends(verify_api_key),
):
    """
    Translates a cropped speech bubble or manga text image:
    1. Runs manga-ocr for Japanese vertical/horizontal manga text.
    2. Runs PaddleOCR for Korean or Chinese text.
    3. Translates recognized text to target_lang (Uzbek, Russian, English).
    """
    image_bytes, pil_image = await read_image_upload(file)

    extracted_text = ""
    normalized_lang = normalize_ocr_language(lang)

    # 1. OCR text extraction
    requested_auto = normalized_lang == "auto"
    if requested_auto:
        normalized_lang, auto_results = await run_in_threadpool(run_auto_paddle_ocr, image_bytes)
        extracted_text = " ".join(line["text"] for line in extract_paddle_lines(auto_results, normalized_lang))
        if normalized_lang == "ja":
            try:
                extracted_text = await run_in_threadpool(run_manga_ocr, pil_image)
            except Exception as exc:
                logger.warning("MangaOcr xatosi: %s; aniqlangan OCR matni ishlatiladi.", exc)
    elif normalized_lang in ["ja", "jpn", "japanese"]:
        try:
            extracted_text = await run_in_threadpool(run_manga_ocr, pil_image)
        except Exception as e:
            logger.warning("MangaOcr xatosi: %s. PaddleOCR zaxirasi sinab ko'riladi.", e)
            try:
                res = await run_in_threadpool(run_paddle_ocr, image_bytes, "japan")
                extracted_text = " ".join(line["text"] for line in extract_paddle_lines(res, "ja"))
            except Exception:
                pass
    elif normalized_lang in ["ko", "kor", "korean"]:
        res = await run_in_threadpool(run_paddle_ocr, image_bytes, "korean")
        extracted_text = " ".join(line["text"] for line in extract_paddle_lines(res, "ko"))
    elif normalized_lang in ["zh", "ch", "chs", "cht", "chinese"]:
        res = await run_in_threadpool(run_paddle_ocr, image_bytes, "ch")
        extracted_text = "".join(line["text"] for line in extract_paddle_lines(res, "zh"))
    elif normalized_lang in ["en", "eng", "english"]:
        res = await run_in_threadpool(run_paddle_ocr, image_bytes, "en")
        extracted_text = " ".join(line["text"] for line in extract_paddle_lines(res, "en"))
    elif normalized_lang in ["ru", "rus", "russian"]:
        res = await run_in_threadpool(run_paddle_ocr, image_bytes, "ru")
        extracted_text = " ".join(line["text"] for line in extract_paddle_lines(res, "ru"))
    else:
        raise HTTPException(status_code=400, detail=f"OCR tili qo'llab-quvvatlanmaydi: {normalized_lang}")

    extracted_text = extracted_text.strip()

    # 2. Text translation
    translated_text = ""
    if extracted_text:
        source_code = translation_language_code(normalized_lang)
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
    extracted_text = ""
    normalized_lang = normalize_ocr_language(lang)

    if normalized_lang == "auto":
        normalized_lang, results = await run_in_threadpool(run_auto_paddle_ocr, image_bytes)
        extracted_text = " ".join(line["text"] for line in extract_paddle_lines(results, normalized_lang))
    elif normalized_lang in ["ja", "jpn", "japanese"]:
        extracted_text = await run_in_threadpool(run_manga_ocr, pil_image)
    elif normalized_lang in ["ko", "kor", "korean"]:
        res = await run_in_threadpool(run_paddle_ocr, image_bytes, "korean")
        extracted_text = " ".join(line["text"] for line in extract_paddle_lines(res, "ko"))
    elif normalized_lang in ["zh", "ch", "chinese"]:
        res = await run_in_threadpool(run_paddle_ocr, image_bytes, "ch")
        extracted_text = "".join(line["text"] for line in extract_paddle_lines(res, "zh"))
    elif normalized_lang in ["en", "eng", "english"]:
        res = await run_in_threadpool(run_paddle_ocr, image_bytes, "en")
        extracted_text = " ".join(line["text"] for line in extract_paddle_lines(res, "en"))
    elif normalized_lang in ["ru", "rus", "russian"]:
        res = await run_in_threadpool(run_paddle_ocr, image_bytes, "ru")
        extracted_text = " ".join(line["text"] for line in extract_paddle_lines(res, "ru"))

    return {
        "status": "success",
        "original_text": extracted_text.strip(),
        "lang": lang,
        "detected_source_lang": normalized_lang
    }

@app.post("/detect-and-translate")
async def detect_and_translate(
    file: UploadFile = File(...),
    lang: str = Form("ja"),
    target_lang: str = Form("uz"),
    _authorized: None = Depends(verify_api_key),
):
    """
    Accepts an entire manga page/screen capture:
    1. Uses PaddleOCR text detector to find speech bubble / text bounding boxes.
    2. Runs manga-ocr on each detected crop for Japanese, or PaddleOCR for Korean/Chinese.
    3. Translates each detected bubble into target_lang.
    4. Returns list of { bbox: [x1, y1, x2, y2], original_text, translated_text }.
    """
    image_bytes, pil_image = await read_image_upload(file)
    width, height = pil_image.size

    # Text box detection with PaddleOCR
    normalized_lang = normalize_ocr_language(lang)
    if normalized_lang == "auto":
        normalized_lang, ocr_results = await run_in_threadpool(run_auto_paddle_ocr, image_bytes)
    else:
        detector_lang = {
            "ja": "japan", "ko": "korean", "zh": "ch", "en": "en", "ru": "ru",
        }.get(normalized_lang)
        if detector_lang is None:
            raise HTTPException(status_code=400, detail=f"OCR tili qo'llab-quvvatlanmaydi: {normalized_lang}")
        ocr_results = await run_in_threadpool(run_paddle_ocr, image_bytes, detector_lang)

    blocks: List[Dict[str, Any]] = []

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

            box_crop = pil_image.crop((x1, y1, x2, y2))
            orig_text = detection["text"]
            if use_manga_ocr:
                try:
                    orig_text = await run_in_threadpool(run_manga_ocr, box_crop)
                except Exception:
                    pass

            orig_text = orig_text.strip()
            if not orig_text:
                continue

            trans_text = await run_in_threadpool(
                translate_text, orig_text, target_lang, translation_language_code(normalized_lang)
            )
            if not trans_text:
                raise HTTPException(status_code=502, detail="Matn aniqlandi, ammo tarjima xizmatlari javob bermadi.")

            blocks.append({
                "left": x1,
                "top": y1,
                "right": x2,
                "bottom": y2,
                "original_text": orig_text,
                "translated_text": trans_text
            })

    return {
        "status": "success",
        "source_lang": normalized_lang,
        "blocks_count": len(blocks),
        "blocks": blocks
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
