"""PDF Translation routes and comic/book OCR processing engine."""

import io
import json
import logging
import math
import os
import re
import time
import hmac
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import cv2
import numpy as np
import pymupdf
from PIL import Image

from fastapi import APIRouter, Depends, Form, File, Header, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from pdf_jobs import (
    PdfJobError,
    PdfQueueFullError,
    inspect_pdf,
)

def resume_pdf_jobs():
    try:
        import main
        if hasattr(main, "resume_pdf_jobs"):
            return main.resume_pdf_jobs()
    except Exception:
        pass


class _PdfJobManagerProxy:
    def __getattr__(self, name):
        import main
        return getattr(main.pdf_job_manager, name)

pdf_job_manager = _PdfJobManagerProxy()
from text_translator import (
    normalize_ocr_language,
    repair_english_word_spacing,
    merge_horizontal_ocr_lines,
    _join_dialogue_lines,
    translate_text,
    translate_batch,
)

logger = logging.getLogger("manga-server")

router = APIRouter()

MAX_PDF_UPLOAD_BYTES = max(1024 * 1024, int(os.environ.get("MANGA_MAX_PDF_UPLOAD_BYTES", str(50 * 1024 * 1024))))
MAX_PDF_PAGES = max(1, int(os.environ.get("MANGA_MAX_PDF_PAGES", "500")))
PDF_OCR_DPI = max(96, min(300, int(os.environ.get("MANGA_PDF_OCR_DPI", "180"))))
PDF_OCR_MAX_PIXELS = max(1_000_000, int(os.environ.get("MANGA_PDF_OCR_MAX_PIXELS", str(16 * 1024 * 1024))))
PDF_ENABLED = os.environ.get("MANGA_PDF_ENABLED", "false" if os.environ.get("K_SERVICE") else "true").strip().lower() in {"1", "true", "yes"}

PDF_LATIN_OCR_LANGS = {"fr", "de", "es", "pt", "it", "tr", "nl", "pl", "uz"}
PDF_OCR_MODEL_LANGS = {
    "ja": "japan", "ko": "korean", "zh": "ch", "en": "en", "ru": "ru",
    **{language: "latin" for language in PDF_LATIN_OCR_LANGS},
}
PDF_SOURCE_LANGUAGES = ("auto", "en", "ru", "fr", "de", "es", "pt", "it", "tr", "nl", "pl", "uz", "ja", "ko", "zh")
PDF_TARGET_LANGUAGES = tuple(code for code in PDF_SOURCE_LANGUAGES if code != "auto") + ("ar",)


def is_pdf_enabled() -> bool:
    try:
        import main
        return bool(getattr(main, "PDF_ENABLED", PDF_ENABLED))
    except Exception:
        return PDF_ENABLED


def verify_pdf_enabled():
    if not is_pdf_enabled():
        raise HTTPException(status_code=503, detail="Bu serverda PDF tarjimasi öçirilgan: doimiy vazifa saqlaş tizimi kerak. PDF uçun lokal yoki doimiy diskli serverdan foydalaning.")


def verify_pdf_job_token(job_id: str, x_pdf_job_token: Optional[str] = Header(default=None)):
    verify_pdf_enabled()
    if not pdf_job_manager.verify_job_token(job_id, x_pdf_job_token):
        raise HTTPException(status_code=401, detail="PDF vazifasi kaliti notöğri yoki kiritilmagan.")


def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Require a configured key on translation endpoints when deployed."""
    try:
        import main
        server_key = getattr(main, "SERVER_API_KEY", "")
        allow_unauth = getattr(main, "ALLOW_UNAUTHENTICATED", False)
    except Exception:
        server_key = os.environ.get("MANGA_SERVER_API_KEY", "").strip()
        allow_unauth = False

    if not server_key and not allow_unauth:
        raise HTTPException(status_code=503, detail="Server API kaliti sozlanmagan. Mahalliy sinov uçun MANGA_ALLOW_UNAUTHENTICATED=true ni belgilang.")
    if server_key and (not x_api_key or not hmac.compare_digest(x_api_key.encode("utf-8"), server_key.encode("utf-8"))):
        raise HTTPException(status_code=401, detail="Server API kaliti notöğri yoki kiritilmagan.")


def resolve_gemini_api_key(gemini_api_key: Optional[str] = None, x_gemini_api_key: Optional[str] = None) -> Optional[str]:
    try:
        import main
        if hasattr(main, "resolve_gemini_api_key"):
            return main.resolve_gemini_api_key(gemini_api_key, x_gemini_api_key)
    except Exception:
        pass
    return (gemini_api_key or "").strip() or (x_gemini_api_key or "").strip() or os.environ.get("GEMINI_API_KEY", "").strip() or None


def get_manga_pipeline():
    import main
    return main.get_manga_pipeline()


def run_paddle_ocr(*args, **kwargs):
    import main
    return main.run_paddle_ocr(*args, **kwargs)


def run_auto_paddle_ocr(*args, **kwargs):
    import main
    return main.run_auto_paddle_ocr(*args, **kwargs)


def run_manga_ocr(*args, **kwargs):
    import main
    return main.run_manga_ocr(*args, **kwargs)


def extract_paddle_lines(*args, **kwargs):
    import main
    return main.extract_paddle_lines(*args, **kwargs)


def is_paddle_enabled():
    try:
        import main
        return main.is_paddle_enabled()
    except Exception:
        return True


def _set_main_pipeline_error(err):
    try:
        import main
        main._manga_pipeline_error = err
    except Exception:
        pass


class _ModelLockProxy:
    def __enter__(self):
        import main
        if hasattr(main, "_model_lock"):
            return main._model_lock.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import main
        if hasattr(main, "_model_lock"):
            return main._model_lock.__exit__(exc_type, exc_val, exc_tb)

_model_lock = _ModelLockProxy()


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
        if min(float(c) for c in median_rgb) >= 0.97:
            return [1.0, 1.0, 1.0]
        return [max(0.0, min(1.0, float(c))) for c in median_rgb]
    except Exception:
        return [1.0, 1.0, 1.0]


def _group_vertical_manga_lines(*args, **kwargs):
    import main
    return main._group_vertical_manga_lines(*args, **kwargs)


def _find_bubble_overlay_region(
    image: Image.Image,
    box: tuple[int, int, int, int],
    threshold_mask=None,
    gray=None,
) -> Optional[dict]:
    """Find a bounded light bubble interior around OCR text; reject page-sized regions."""
    try:
        import cv2
        import numpy as np

        left, top, right, bottom = box
        left = max(0, min(image.width - 1, left))
        right = max(left + 1, min(image.width, right))
        top = max(0, min(image.height - 1, top))
        bottom = max(top + 1, min(image.height, bottom))
        if gray is None:
            if threshold_mask is None:
                pixels = np.asarray(image.convert("RGB"))
                gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        if gray is not None:
            sub_gray = gray[top:bottom, left:right]
            is_dark = np.median(sub_gray) < 110 if sub_gray.size > 0 else False
            if is_dark:
                threshold_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)[1]
            elif threshold_mask is None:
                threshold_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]
        elif threshold_mask is None:
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


try:
    from ocr_engine import (
        is_manga_punctuation,
        normalize_manga_punctuation,
        is_ocr_garbage_noise,
        is_dark_background,
        apply_dark_box_clahe_inversion,
        filter_manga_ocr_hallucinations,
    )
except ImportError:
    try:
        from ml.ocr_engine import (
            is_manga_punctuation,
            normalize_manga_punctuation,
            is_ocr_garbage_noise,
            is_dark_background,
            apply_dark_box_clahe_inversion,
            filter_manga_ocr_hallucinations,
        )
    except ImportError:
        from manga_pipeline import (
            is_manga_punctuation,
            normalize_manga_punctuation,
            is_ocr_garbage_noise,
            is_dark_background,
            apply_dark_box_clahe_inversion,
            filter_manga_ocr_hallucinations,
        )


from manga_bubble_engine import is_valid_manga_dialogue


def _group_comic_lines(
    lines,
    rendered_image,
    bubble_threshold_mask,
    scale,
    detected_lang="en",
    origin_x: float = 0.0,
    origin_y: float = 0.0,
):
    """Group lines that share the same bubble or overlap closely into a single coherent block."""
    if not lines:
        return []

    try:
        page_gray = cv2.cvtColor(np.asarray(rendered_image), cv2.COLOR_RGB2GRAY)
    except Exception:
        page_gray = None

    line_entries = []
    for line in lines:
        if not line["text"].strip():
            continue
        left = max(0, min(rendered_image.width, line["left"]))
        top = max(0, min(rendered_image.height, line["top"]))
        right = max(left, min(rendered_image.width, line["right"]))
        bottom = max(top, min(rendered_image.height, line["bottom"]))
        box = (left, top, right, bottom)
        ov = _find_bubble_overlay_region(rendered_image, box, bubble_threshold_mask, gray=page_gray)
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
    is_cjk_rtl = detected_lang in {"ja", "japan", "japanese", "zh", "ch", "chinese"}
    for cluster in clusters:
        is_vertical_text = is_cjk_rtl and any((x["box"][3] - x["box"][1]) > 1.25 * (x["box"][2] - x["box"][0]) for x in cluster)
        if is_vertical_text:
            cluster.sort(key=lambda x: -x["box"][2])
        else:
            cluster.sort(key=lambda x: (x["box"][1], x["box"][0]))
        joiner = "" if is_cjk_rtl else " "
        text = joiner.join(x["line"]["text"].strip() for x in cluster)
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
                crop = rendered_image.crop((min_l, min_t, max_r, max_b))
                if is_dark_background(crop):
                    crop = apply_dark_box_clahe_inversion(crop)
                refined = run_manga_ocr(crop).strip()
                refined = filter_manga_ocr_hallucinations(refined)
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

        ov_b = best_overlay["bbox"]
        block = {
            "text": text,
            "font_size": 0.0,
            "confidence": avg_conf,
            "bbox": [origin_x + min_l / scale, origin_y + min_t / scale, origin_x + max_r / scale, origin_y + max_b / scale],
            "text_erase_bbox": [origin_x + min_l / scale, origin_y + min_t / scale, origin_x + max_r / scale, origin_y + max_b / scale],
            "overlay_bbox": [origin_x + ov_b[0] / scale, origin_y + ov_b[1] / scale, origin_x + ov_b[2] / scale, origin_y + ov_b[3] / scale],
            "overlay_polygon": [[origin_x + x / scale, origin_y + y / scale] for x, y in best_overlay["polygon"]],
            "background": _sample_comic_background(rendered_image, (min_l, min_t, max_r, max_b)),
        }
        blocks.append(block)

    if blocks:
        page_h = max(1.0, float(rendered_image.height))
        band_h = max(40.0, page_h * 0.12)
        if is_cjk_rtl:
            blocks.sort(key=lambda b: (int((b["bbox"][1] - origin_y) * scale / band_h), -b["bbox"][2]))
        else:
            blocks.sort(key=lambda b: (int((b["bbox"][1] - origin_y) * scale / band_h), b["bbox"][0]))

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
    origin_x = float(getattr(rect, "x0", 0.0) or 0.0)
    origin_y = float(getattr(rect, "y0", 0.0) or 0.0)
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
    cleaned_filename = None

    try:
        # ─── 1. COMIC MODE: ML Bubble Detection & Neural Inpainting Text Cleaning ───
        if document_mode == "comic":
            try:
                pipe = get_manga_pipeline()
                pipe_lang = "en" if normalized_lang in PDF_LATIN_OCR_LANGS else normalized_lang
                pipeline_res = pipe.process_page(rendered_image, auto_collect=False, source_lang=pipe_lang)
                cleaned_img = pipeline_res.get("cleaned_image")
                dialogues = pipeline_res.get("dialogues", [])
                cleaned_ids = set(pipeline_res.get("cleaned_bubble_ids", []))

                if pipeline_res.get("requires_ocr_fallback"):
                    raise PdfJobError("ML OCR natijasi işonçsiz; zaxira OCR tekşiriladi.")

                blocks = []
                for d in dialogues:
                    if d.get("category") == "sfx" or d.get("is_sfx"):
                        continue
                    text = (d.get("transcription") or "").strip()
                    if not text:
                        continue
                    bx1, by1, bx2, by2 = d["bbox"]
                    scaled_bbox = [
                        origin_x + bx1 / scale,
                        origin_y + by1 / scale,
                        origin_x + bx2 / scale,
                        origin_y + by2 / scale,
                    ]
                    if d.get("text_erase_bbox") and len(d["text_erase_bbox"]) == 4:
                        eb = d["text_erase_bbox"]
                        text_erase = [
                            origin_x + eb[0] / scale,
                            origin_y + eb[1] / scale,
                            origin_x + eb[2] / scale,
                            origin_y + eb[3] / scale,
                        ]
                    else:
                        text_erase = scaled_bbox
                    if d.get("overlay_bbox") and len(d["overlay_bbox"]) == 4:
                        ob = d["overlay_bbox"]
                        overlay_b = [
                            origin_x + ob[0] / scale,
                            origin_y + ob[1] / scale,
                            origin_x + ob[2] / scale,
                            origin_y + ob[3] / scale,
                        ]
                    else:
                        overlay_b = scaled_bbox
                    if d.get("overlay_polygon") and isinstance(d["overlay_polygon"], list):
                        poly = [[origin_x + p[0] / scale, origin_y + p[1] / scale] for p in d["overlay_polygon"]]
                    else:
                        poly = [
                            [scaled_bbox[0], scaled_bbox[1]],
                            [scaled_bbox[2], scaled_bbox[1]],
                            [scaled_bbox[2], scaled_bbox[3]],
                            [scaled_bbox[0], scaled_bbox[3]],
                        ]
                    bg = d.get("background")
                    if not isinstance(bg, (list, tuple)) or len(bg) < 3:
                        bg = _sample_comic_background(rendered_image, (int(bx1), int(by1), int(bx2), int(by2)))
                    else:
                        bg = list(bg)

                    blocks.append({
                        "text": text,
                        "font_size": 0.0,
                        "confidence": float(d.get("confidence", 0.5)),
                        "bbox": scaled_bbox,
                        "text_erase_bbox": text_erase,
                        "overlay_bbox": overlay_b,
                        "overlay_polygon": poly,
                        "background": bg,
                        "category": d.get("category", "speech_bubble"),
                        "bubble_id": d.get("bubble_id", len(blocks) + 1),
                        "cleaned": bool(cleaned_img is not None and (d.get("bubble_id") in cleaned_ids)),
                    })

                if not blocks:
                    raise PdfJobError("ML OCR matn topmadi; zaxira OCR tekshiriladi.")

                if cleaned_img is not None and job_dir is not None and any(b.get("cleaned") for b in blocks):
                    cleaned_filename = f"cleaned_page_{page_index:04d}.png"
                    cleaned_img.save(job_dir / cleaned_filename, format="PNG")

                if normalized_lang in PDF_LATIN_OCR_LANGS:
                    detected_lang = normalized_lang
                else:
                    detected_lang = pipeline_res.get("detected_source_lang", normalized_lang)
                _set_main_pipeline_error(None)
                return {
                    "blocks": blocks,
                    "detected_source_lang": detected_lang,
                    "cleaned_image_filename": cleaned_filename,
                    "rotation": int(getattr(page, "rotation", 0)) % 360,
                }
            except Exception as exc:
                cleaned_filename = None
                _set_main_pipeline_error(type(exc).__name__ if not isinstance(exc, PdfJobError) else "fallback")
                logger.warning("Comic PDF rejimida xatolik: %s. Zaxira OCR tekşiriladi...", exc)
                try:
                    from manga_bubble_engine import extract_manga_dialogues
                    blocks = extract_manga_dialogues(
                        pil_image=rendered_image,
                        source_lang=normalized_lang,
                        scale=scale,
                    )
                    if blocks:
                        if origin_x != 0.0 or origin_y != 0.0:
                            for b in blocks:
                                for bk in ("bbox", "text_erase_bbox", "overlay_bbox"):
                                    if isinstance(b.get(bk), list) and len(b[bk]) == 4:
                                        b[bk] = [b[bk][0] + origin_x, b[bk][1] + origin_y, b[bk][2] + origin_x, b[bk][3] + origin_y]
                                if isinstance(b.get("overlay_polygon"), list):
                                    b["overlay_polygon"] = [[pt[0] + origin_x, pt[1] + origin_y] for pt in b["overlay_polygon"]]
                        detected_lang = normalized_lang if normalized_lang != "auto" else "en"
                        _set_main_pipeline_error(None)
                        if job_dir is not None:
                            try:
                                try:
                                    from manga_pipeline import clean_bubble_crop
                                except ImportError:
                                    try:
                                        from ml.manga_pipeline import clean_bubble_crop
                                    except ImportError:
                                        from cleaning import clean_bubble_crop
                                img_np = np.array(rendered_image.convert("RGB"))
                                cleaned_np = img_np.copy()
                                any_cleaned = False
                                for b in blocks:
                                    if b.get("category") == "sfx":
                                        b["cleaned"] = False
                                        continue
                                    bbox = b.get("bbox")
                                    if bbox and len(bbox) == 4:
                                        bx1 = max(0, min(img_np.shape[1], int(round((bbox[0] - origin_x) * scale))))
                                        by1 = max(0, min(img_np.shape[0], int(round((bbox[1] - origin_y) * scale))))
                                        bx2 = max(0, min(img_np.shape[1], int(round((bbox[2] - origin_x) * scale))))
                                        by2 = max(0, min(img_np.shape[0], int(round((bbox[3] - origin_y) * scale))))
                                        if bx2 > bx1 and by2 > by1:
                                            patch = cleaned_np[by1:by2, bx1:bx2]
                                            if patch.size > 0:
                                                cleaned_patch = clean_bubble_crop(patch)
                                                cleaned_np[by1:by2, bx1:bx2] = cleaned_patch
                                                b["cleaned"] = True
                                                any_cleaned = True
                                if any_cleaned:
                                    cleaned_fb = Image.fromarray(cleaned_np)
                                    cleaned_filename = f"cleaned_page_{page_index:04d}.png"
                                    cleaned_fb.save(job_dir / cleaned_filename, format="PNG")
                                else:
                                    cleaned_filename = None
                            except Exception as cl_exc:
                                cleaned_filename = None
                                for b in blocks:
                                    b["cleaned"] = False
                                logger.debug("Fallback clean in PDF page extract failed: %s", cl_exc)
                        else:
                            for b in blocks:
                                b["cleaned"] = False
                        return {
                            "blocks": blocks,
                            "detected_source_lang": detected_lang,
                            "cleaned_image_filename": cleaned_filename,
                            "rotation": int(getattr(page, "rotation", 0)) % 360,
                        }
                except Exception as fb_exc:
                    cleaned_filename = None
                    logger.warning("extract_manga_dialogues fallback xatosi: %s", fb_exc)

        # ─── 2. BOOK MODE or Fallback OCR ──────────────────────────────────────────
        cleaned_filename = None
        if normalized_lang == "auto":
            detected_lang, results = run_auto_paddle_ocr(image_bytes)
            if detected_lang == "en":
                lines = extract_paddle_lines(results, "en")
                if document_mode == "book":
                    detected_lang = "auto"
            else:
                lines = extract_paddle_lines(results, detected_lang)
        else:
            detector_lang = PDF_OCR_MODEL_LANGS.get(normalized_lang)
            if detector_lang is None:
                raise PdfJobError(f"PDF OCR tili qöllab-quvvatlanmaydi: {normalized_lang}")
            detected_lang = normalized_lang
            results = run_paddle_ocr(image_bytes, detector_lang)
            lines = extract_paddle_lines(results, detected_lang)

        if document_mode == "comic" and detected_lang in {"ja", "zh"}:
            lines = _group_vertical_manga_lines(lines)

        bubble_threshold_mask = None
        if document_mode == "comic":
            try:
                grayscale = cv2.cvtColor(np.asarray(rendered_image), cv2.COLOR_RGB2GRAY)
                bubble_threshold_mask = cv2.threshold(grayscale, 220, 255, cv2.THRESH_BINARY)[1]
            except Exception as exc:
                logger.debug("Pufak konturi uçun rasmni tayyorlab bölmadi: %s", exc)

        if document_mode == "comic":
            blocks = _group_comic_lines(
                lines,
                rendered_image,
                bubble_threshold_mask,
                scale,
                detected_lang=detected_lang,
                origin_x=origin_x,
                origin_y=origin_y,
            )
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
                    "bbox": [origin_x + left / scale, origin_y + top / scale, origin_x + right / scale, origin_y + bottom / scale],
                    "text_erase_bbox": [origin_x + left / scale, origin_y + top / scale, origin_x + right / scale, origin_y + bottom / scale],
                })
        return {
            "blocks": blocks,
            "detected_source_lang": detected_lang,
            "cleaned_image_filename": None,
            "rotation": int(getattr(page, "rotation", 0)) % 360,
        }
    finally:
        try:
            rendered_image.close()
        except Exception:
            pass

@router.post("/translate-pdf")
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
        effective_gemini_key = resolve_gemini_api_key(gemini_api_key, x_gemini_api_key)
        try:
            resume_pdf_jobs()
        except Exception as exc:
            logger.debug("Failed resuming pdf jobs: %s", exc)
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


@router.get("/pdf-jobs/{job_id}")
def get_pdf_translation_status(
    job_id: str,
    _authorized: None = Depends(verify_api_key),
    _job_token: None = Depends(verify_pdf_job_token),
):
    try:
        return pdf_job_manager.public_job(job_id)
    except PdfJobError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/pdf-jobs/{job_id}/review/{page_index}")
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


@router.post("/pdf-jobs/{job_id}/review")
async def correct_pdf_translation(
    job_id: str,
    request: Request,
    _authorized: None = Depends(verify_api_key),
    _job_token: None = Depends(verify_pdf_job_token),
):
    verify_pdf_enabled()
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > 512 * 1024:
                raise HTTPException(status_code=413, detail="Tuzatiş sörovi hajmi judayam katta (max 512KB).")
        except ValueError:
            raise HTTPException(status_code=400, detail="Noto'g'ri Content-Length qiymati.")

    max_bytes = 512 * 1024
    received = 0
    body_chunks = []
    async for chunk in request.stream():
        received += len(chunk)
        if received > max_bytes:
            raise HTTPException(status_code=413, detail="Tuzatiş sörovi hajmi judayam katta (max 512KB).")
        body_chunks.append(chunk)
    body = b"".join(body_chunks)
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Tuzatiş sörovi JSON bölişi kerak.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("corrections"), list):
        raise HTTPException(status_code=400, detail="'corrections' röyxati yuborilişi kerak.")
    try:
        return await run_in_threadpool(pdf_job_manager.apply_corrections, job_id, payload["corrections"])
    except PdfJobError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/pdf-jobs/{job_id}/cancel")
def cancel_pdf_translation(
    job_id: str,
    _authorized: None = Depends(verify_api_key),
    _job_token: None = Depends(verify_pdf_job_token),
):
    try:
        return pdf_job_manager.cancel(job_id)
    except PdfJobError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/pdf-jobs/{job_id}/resume")
def resume_pdf_translation(
    job_id: str,
    _authorized: None = Depends(verify_api_key),
    _job_token: None = Depends(verify_pdf_job_token),
):
    verify_pdf_enabled()
    try:
        return pdf_job_manager.resume(job_id)
    except PdfQueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "30"}) from exc
    except PdfJobError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/pdf-jobs/{job_id}/download")
def download_pdf_translation(
    job_id: str,
    _authorized: None = Depends(verify_api_key),
    _job_token: None = Depends(verify_pdf_job_token),
):
    safe_job_id = re.sub(r"[^a-zA-Z0-9_-]", "", job_id)
    if not safe_job_id:
        raise HTTPException(status_code=400, detail="Noto'g'ri job_id formati.")
    try:
        path = pdf_job_manager.output_path(safe_job_id)
        job = pdf_job_manager.get_job(safe_job_id)
        raw_name = job.get("filename", "document.pdf") if job else "document.pdf"
        stem = Path(raw_name).stem
        sanitized_stem = "".join(c for c in stem if c.isalnum() or c in ("-", "_", " ")).strip() or "document"
        return FileResponse(path, media_type="application/pdf", filename=f"{sanitized_stem}-translated.pdf")
    except PdfJobError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
