"""
End-to-End AI Manga Translation & Processing System.
Orchestrates speech bubble detection, multi-language OCR, border-protected inpainting,
and dynamic dialogue typesetting.
"""
import os
import sys
import logging
import threading
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    from utils import get_best_font, repair_comic_text
except (ImportError, ValueError):
    from .utils import get_best_font, repair_comic_text

try:
    from ocr_engine import (
        is_manga_punctuation,
        normalize_manga_punctuation,
        is_sfx_text,
        normalize_cyrillic_ocr,
        is_ocr_garbage_noise,
        try_read_vertical_strip,
        detect_primary_script_from_text,
        analyze_script_distribution,
        is_dark_background,
        apply_dark_box_clahe_inversion,
        enhance_crop_for_ocr,
        filter_manga_ocr_hallucinations,
    )
except (ImportError, ValueError):
    from .ocr_engine import (
        is_manga_punctuation,
        normalize_manga_punctuation,
        is_sfx_text,
        normalize_cyrillic_ocr,
        is_ocr_garbage_noise,
        try_read_vertical_strip,
        detect_primary_script_from_text,
        analyze_script_distribution,
        is_dark_background,
        apply_dark_box_clahe_inversion,
        enhance_crop_for_ocr,
        filter_manga_ocr_hallucinations,
    )

try:
    from detection import (
        _crop_reading_key,
        deduplicate_text_detections,
        sort_detections_reading_order,
    )
except (ImportError, ValueError):
    from .detection import (
        _crop_reading_key,
        deduplicate_text_detections,
        sort_detections_reading_order,
    )

try:
    from cleaning import (
        DEVICE,
        generate_ink_stroke_mask,
        get_shared_simple_lama,
        clean_bubble_crop,
        clean_page_blocks,
    )
except (ImportError, ValueError):
    from .cleaning import (
        DEVICE,
        generate_ink_stroke_mask,
        get_shared_simple_lama,
        clean_bubble_crop,
        clean_page_blocks,
    )

try:
    from typesetting import (
        hyphenate_uzbek_word,
        wrap_text_elliptical,
        render_translated_dialogues,
        resolve_wildwords_font,
    )
except (ImportError, ValueError):
    from .typesetting import (
        hyphenate_uzbek_word,
        wrap_text_elliptical,
        render_translated_dialogues,
        resolve_wildwords_font,
    )

logger = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

__all__ = [
    # OCR Engine
    "is_manga_punctuation",
    "normalize_manga_punctuation",
    "is_sfx_text",
    "normalize_cyrillic_ocr",
    "is_ocr_garbage_noise",
    "try_read_vertical_strip",
    "detect_primary_script_from_text",
    "analyze_script_distribution",
    "is_dark_background",
    "apply_dark_box_clahe_inversion",
    "enhance_crop_for_ocr",
    "filter_manga_ocr_hallucinations",
    # Detection
    "_crop_reading_key",
    "deduplicate_text_detections",
    "sort_detections_reading_order",
    # Cleaning
    "DEVICE",
    "generate_ink_stroke_mask",
    "get_shared_simple_lama",
    "clean_bubble_crop",
    "clean_page_blocks",
    # Typesetting
    "hyphenate_uzbek_word",
    "wrap_text_elliptical",
    "render_translated_dialogues",
    "resolve_wildwords_font",
    # Pipeline Orchestrator
    "MangaTranslationPipeline",
    "BASE_DIR",
]


def _deduplicate_overlapping_detections(detections, iou_thresh: float = 0.45, contain_thresh: float = 0.62):
    """Suppress heavily overlapping duplicate detections before OCR, cleaning, and typesetting."""
    if not detections or len(detections) <= 1:
        return detections

    def _box_of(det):
        b = det.get("box") or det.get("bbox") or det.get("text_box") or [0, 0, 0, 0]
        return float(b[0]), float(b[1]), float(b[2]), float(b[3])

    def _score_of(det):
        conf = float(det.get("avg_conf", det.get("conf", det.get("confidence", 0.5))))
        cat = det.get("category", "speech_bubble")
        cat_bonus = 0.05 if cat in ("speech_bubble", "narration") else 0.0
        x1, y1, x2, y2 = _box_of(det)
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return (conf + cat_bonus, area)

    indexed = list(enumerate(detections))
    indexed.sort(key=lambda item: _score_of(item[1]), reverse=True)

    kept = []
    for orig_idx, det in indexed:
        ax1, ay1, ax2, ay2 = _box_of(det)
        area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        is_dup = False
        for _, kdet in kept:
            bx1, by1, bx2, by2 = _box_of(kdet)
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
            iou = inter / max(1.0, area_a + area_b - inter)
            contain = inter / min(area_a, area_b)
            if iou >= iou_thresh or contain >= contain_thresh:
                is_dup = True
                break
        if not is_dup:
            kept.append((orig_idx, det))

    kept.sort(key=lambda item: item[0])
    return [det for _, det in kept]


class MangaTranslationPipeline:
    """
    End-to-End AI Manga Translation & Processing System (3-stage neural pipeline):
    Step 1: RT-DETR-v2 (`ogkalu/comic-text-and-bubble-detector`) blocks + CTD UNet character mask
    Step 2: Reading-order-aware Multi-Language OCR & Translation (Qwen3.5-9B 4-bit NF4)
    Step 3: 2-path Background Cleaning (fast solid-bubble fill + Manga-LaMa neural inpainting)
    Step 4: Dynamic Dialogue Typesetting
    """
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or BASE_DIR
        self.model_lock = threading.RLock()
        self.cleaner = None
        self.comic_detector = None
        self.qwen_engine = None

    def _get_comic_detector(self):
        with self.model_lock:
            if self.comic_detector is None:
                try:
                    try:
                        from .comic_detector import get_shared_comic_detector
                    except (ImportError, ValueError):
                        try:
                            from ml.comic_detector import get_shared_comic_detector
                        except ImportError:
                            from comic_detector import get_shared_comic_detector
                    det = get_shared_comic_detector()
                    if det.is_available:
                        self.comic_detector = det
                except Exception as exc:
                    logger.warning("ComicTextDetector unavailable: %s", exc)
            return self.comic_detector

    def _get_qwen_engine(self):
        with self.model_lock:
            if self.qwen_engine is None:
                try:
                    try:
                        from .qwen_vl_engine import get_shared_qwen_engine
                    except (ImportError, ValueError):
                        try:
                            from ml.qwen_vl_engine import get_shared_qwen_engine
                        except ImportError:
                            from qwen_vl_engine import get_shared_qwen_engine
                    eng = get_shared_qwen_engine()
                    if eng.is_available:
                        self.qwen_engine = eng
                except Exception as exc:
                    logger.warning("QwenVLEngine unavailable: %s", exc)
            return self.qwen_engine

    def process_page(
        self,
        image_input,
        known_bubbles=None,
        auto_collect=True,
        is_verified=False,
        progress_callback=None,
        source_lang="auto",
        reading_direction="rtl",
        skip_cleaning: bool = False,
        target_lang: str = None,
    ):
        """
        Processes a manga/manhwa/manhua page image using comic-text-detector + Qwen3.5-9B.
        """
        if progress_callback:
            progress_callback(0.08, "Sahifa tahlilga tayyorlanmoqda...")

        if isinstance(image_input, str):
            orig_img = Image.open(image_input).convert("RGB")
        else:
            orig_img = image_input.convert("RGB")

        source_lang = (source_lang or "auto").strip().lower()
        source_lang = {
            "ch": "zh", "ch_sim": "zh", "chs": "zh", "cht": "zh", "chinese": "zh", "zh-cn": "zh", "zh-tw": "zh",
            "jp": "ja", "jpn": "ja", "japanese": "ja",
            "kor": "ko", "korean": "ko",
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
        }.get(source_lang, source_lang)
        supported_langs = {"auto", "ja", "ko", "zh", "en", "ru", "fr", "de", "es", "pt", "it", "tr", "nl", "pl", "uz", "latin"}
        if source_lang not in supported_langs:
            raise ValueError(f"Unsupported OCR source language: {source_lang}")

        w, h = orig_img.size
        ctd_mask = None

        # 1. Text Block Detection via comic-text-detector
        detections = []
        if known_bubbles:
            detections = [dict(b) for b in known_bubbles]
        else:
            if progress_callback:
                progress_callback(0.25, "ComicTextDetector matn bloklarini aniqlamoqda...")
            comic_det = self._get_comic_detector()
            if comic_det is not None:
                try:
                    detections, ctd_mask = comic_det.detect(orig_img)
                except Exception as exc:
                    logger.warning("ComicTextDetector detect failed: %s", exc)
                    detections, ctd_mask = [], None

            if not detections:
                return {
                    "original_image": orig_img,
                    "annotated_image": orig_img.copy(),
                    "mask_image": Image.new("L", orig_img.size, 0),
                    "cleaned_image": orig_img.copy(),
                    "dialogues": [],
                    "learning_info": None,
                    "cleaned_bubble_ids": [],
                    "detected_source_lang": source_lang,
                    "requires_ocr_fallback": (comic_det is None),
                }

        detections = _deduplicate_overlapping_detections(detections)
        detections = sort_detections_reading_order(detections, reading_direction=reading_direction)

        # 2. Reading-Order-Preserving OCR (and direct Translation) via Qwen3.5-9B
        if progress_callback:
            progress_callback(0.55, "Qwen3.5-9B matnlarni öqimoqda va tarjima qilmoqda...")

        qwen = self._get_qwen_engine()
        need_qwen_indices = []
        need_qwen_crops = []
        for idx, det in enumerate(detections):
            existing_src = (det.get("text") or det.get("transcription") or "").strip()
            existing_tgt = (det.get("translation") or det.get("translated_text") or "").strip()
            needs_ocr = not existing_src
            needs_trans = bool(target_lang) and not existing_tgt
            if (needs_ocr or needs_trans) and qwen is not None:
                tbox = det.get("text_box") or det.get("box") or det.get("bbox") or [0, 0, 0, 0]
                tx1, ty1, tx2, ty2 = int(tbox[0]), int(tbox[1]), int(tbox[2]), int(tbox[3])
                cx1 = max(0, tx1 - 5)
                cy1 = max(0, ty1 - 5)
                cx2 = min(w, tx2 + 5)
                cy2 = min(h, ty2 + 5)
                # Clamp against neighboring detections' text_box so adjacent bubbles do not bleed into Qwen's crop
                for n_idx, n_det in enumerate(detections):
                    if n_idx == idx:
                        continue
                    ntb = n_det.get("text_box") or n_det.get("box") or n_det.get("bbox")
                    if not ntb or len(ntb) < 4:
                        continue
                    nx1, ny1, nx2, ny2 = int(ntb[0]), int(ntb[1]), int(ntb[2]), int(ntb[3])
                    if min(cy2, ny2) > max(cy1, ny1):
                        if nx2 <= tx1 and nx2 > cx1:
                            cx1 = max(cx1, (nx2 + tx1) // 2 + 1)
                        elif nx1 >= tx2 and nx1 < cx2:
                            cx2 = min(cx2, (tx2 + nx1) // 2)
                    if min(cx2, nx2) > max(cx1, nx1):
                        if ny2 <= ty1 and ny2 > cy1:
                            cy1 = max(cy1, (ny2 + ty1) // 2 + 1)
                        elif ny1 >= ty2 and ny1 < cy2:
                            cy2 = min(cy2, (ty2 + ny1) // 2)
                if cx2 > cx1 + 4 and cy2 > cy1 + 4:
                    need_qwen_indices.append(idx)
                    need_qwen_crops.append(orig_img.crop((cx1, cy1, cx2, cy2)))

        translator_used = None
        if need_qwen_crops and qwen is not None:
            if target_lang:
                pairs = qwen.ocr_and_translate_crops(
                    need_qwen_crops, target_lang=target_lang, source_lang=source_lang, batch_size=4
                )
                for idx, (src_txt, tgt_txt) in zip(need_qwen_indices, pairs):
                    if not (detections[idx].get("text") or detections[idx].get("transcription") or "").strip():
                        detections[idx]["text"] = src_txt
                    if tgt_txt and not (detections[idx].get("translation") or detections[idx].get("translated_text") or "").strip():
                        detections[idx]["translation"] = tgt_txt
                translator_used = "qwen3.5-9b"
            else:
                ocr_texts = qwen.ocr_crops(need_qwen_crops, source_lang=source_lang, batch_size=4)
                for idx, src_txt in zip(need_qwen_indices, ocr_texts):
                    if not (detections[idx].get("text") or detections[idx].get("transcription") or "").strip():
                        detections[idx]["text"] = src_txt

        if source_lang == "auto" and detections:
            sample_txt = " ".join(
                (d.get("text") or d.get("transcription") or "").strip()
                for d in detections
                if (d.get("text") or d.get("transcription") or "").strip()
            )
            if sample_txt:
                inferred = detect_primary_script_from_text(sample_txt)
                if inferred and inferred != "auto":
                    source_lang = inferred

        annotated_img = orig_img.copy()
        draw_annotated = ImageDraw.Draw(annotated_img)

        mask_np = np.zeros((h, w), dtype=np.uint8)
        orig_np_arr = np.array(orig_img)
        gray_orig = cv2.cvtColor(orig_np_arr, cv2.COLOR_RGB2GRAY)

        dialogue_records = []
        font_badge = get_best_font(size=20, bold=True)

        for idx, det in enumerate(detections):
            box = det.get("box") or det.get("bbox") or det.get("text_box") or [0, 0, 0, 0]
            cat = det.get("category", "speech_bubble")
            x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2])), min(h, int(box[3]))
            if x2 <= x1 + 4 or y2 <= y1 + 4:
                continue

            transcription = (det.get("text") or det.get("transcription") or "").strip()
            translation = (det.get("translation") or det.get("translated_text") or "").strip()
            conf_val = float(det.get("avg_conf", det.get("conf", det.get("confidence", 0.95))))

            if not transcription:
                continue

            if is_manga_punctuation(transcription):
                transcription = normalize_manga_punctuation(transcription)
                if not translation and target_lang:
                    translation = transcription
                ocr_trusted = True
            else:
                ocr_trusted = not is_ocr_garbage_noise(
                    transcription, source_lang, is_inside_bubble=(cat in ("speech_bubble", "narration"))
                )

            is_sfx_item = (cat == "sfx") or (cat != "narration" and is_sfx_text(transcription))
            effective_cat = "sfx" if is_sfx_item else cat
            stroke_box = det.get("text_box") or det.get("bbox") or det.get("box") or [x1, y1, x2, y2]
            stroke_box = [
                max(0, int(stroke_box[0])),
                max(0, int(stroke_box[1])),
                min(w, int(stroke_box[2])),
                min(h, int(stroke_box[3])),
            ]

            rec = {
                "bubble_id": len(dialogue_records) + 1,
                "block_id": det.get("block_id") or f"b_{len(dialogue_records) + 1}",
                "category": effective_cat,
                "label": det.get("label") or ("text_bubble" if effective_cat in ("speech_bubble", "narration") else "text_free"),
                "is_dark": bool(det.get("is_dark", False)),
                "bbox": [x1, y1, x2, y2],
                "text_box": stroke_box,
                "transcription": transcription,
                "confidence": conf_val,
                "ocr_trusted": ocr_trusted,
                "is_sfx": is_sfx_item,
                "cleaned": False,
            }
            if det.get("bubble_box"):
                rec["bubble_box"] = [int(v) for v in det["bubble_box"]]
            if translation and not is_sfx_item:
                rec["translation"] = translation
            dialogue_records.append(rec)

            color = (255, 0, 0)
            draw_annotated.rectangle([x1, y1, x2, y2], outline=color, width=4)
            bid_text = f"#{rec['bubble_id']}"
            badge_w = max(42, int(len(bid_text) * 14 + 10))
            draw_annotated.rectangle([x1, max(0, y1 - 24), min(w, x1 + badge_w), max(0, y1)], fill=(255, 255, 255))
            draw_annotated.text((x1 + 4, max(0, y1 - 23)), bid_text, fill=color, font=font_badge)

        if source_lang == "auto" and dialogue_records:
            sample_txt = " ".join(d["transcription"] for d in dialogue_records if d.get("transcription"))
            inferred = detect_primary_script_from_text(sample_txt)
            if inferred and inferred != "auto":
                source_lang = inferred

        has_any_trusted = any(
            d.get("ocr_trusted", False) and d.get("transcription") and not d.get("is_sfx", False) and d.get("category") != "sfx"
            for d in dialogue_records
        )

        # Skip cleaning when skip_cleaning=True or when NO dialogue has trusted OCR
        if skip_cleaning or not has_any_trusted:
            mask_img = Image.fromarray(mask_np)
            res_dict = {
                "original_image": orig_img,
                "annotated_image": annotated_img,
                "mask_image": mask_img,
                "cleaned_image": orig_img.copy(),
                "dialogues": dialogue_records,
                "learning_info": None,
                "cleaned_bubble_ids": [],
                "detected_source_lang": source_lang,
                "requires_ocr_fallback": (not has_any_trusted),
            }
            if translator_used:
                res_dict["translator_used"] = translator_used
            return res_dict

        # 3. 2-Path Background Cleaning (Fast solid-bubble fill + Manga-LaMa neural inpainting)
        if progress_callback:
            progress_callback(0.85, "Fon tozalanmoqda (Inpainting)...")

        cleanable_records = [
            d for d in dialogue_records
            if d.get("ocr_trusted", False) and d.get("transcription") and not d.get("is_sfx", False) and d.get("category") != "sfx"
        ]

        with self.model_lock:
            cleaned_np, applied_mask = clean_page_blocks(
                orig_np_arr,
                cleanable_records,
                page_mask=ctd_mask,
                device=DEVICE,
            )

        if np.any(applied_mask > 0):
            mask_np = np.maximum(mask_np, applied_mask)
        if ctd_mask is not None and ctd_mask.shape[:2] == (h, w):
            for d in cleanable_records:
                tb = d["text_box"]
                tx1, ty1, tx2, ty2 = max(0, int(tb[0])), max(0, int(tb[1])), min(w, int(tb[2])), min(h, int(tb[3]))
                if tx2 > tx1 and ty2 > ty1:
                    mask_np[ty1:ty2, tx1:tx2] = np.maximum(mask_np[ty1:ty2, tx1:tx2], ctd_mask[ty1:ty2, tx1:tx2])

        cleaned_bubble_ids = [
            d["bubble_id"] for d in cleanable_records if d.get("cleaned") and "bubble_id" in d
        ]
        mask_img = Image.fromarray(mask_np)
        cleaned_page = Image.fromarray(cleaned_np)

        learning_info = None

        if progress_callback:
            progress_callback(1.0, "Jarayon muvaffaqiyatli yakunlandi!")

        res_out = {
            "original_image": orig_img,
            "annotated_image": annotated_img,
            "mask_image": mask_img,
            "cleaned_image": cleaned_page,
            "dialogues": dialogue_records,
            "learning_info": learning_info,
            "cleaned_bubble_ids": cleaned_bubble_ids,
            "detected_source_lang": source_lang,
            "requires_ocr_fallback": (not any(d.get("ocr_trusted", False) for d in dialogue_records)),
        }
        if translator_used:
            res_out["translator_used"] = translator_used
        return res_out
