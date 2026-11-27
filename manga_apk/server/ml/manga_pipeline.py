"""
End-to-End AI Manga Translation & Processing System.
Orchestrates speech bubble detection, multi-language OCR, border-protected inpainting,
and dynamic dialogue typesetting.
"""
import os
import sys
import json
import logging
import threading
import re
import cv2
import torch
import hashlib
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as T

try:
    from models import MangaBubbleDetector, MangaOCRModel, MangaInpaintingModel, IDX_TO_CATEGORY
except (ImportError, ValueError):
    try:
        from .models import MangaBubbleDetector, MangaOCRModel, MangaInpaintingModel, IDX_TO_CATEGORY
    except (ImportError, ValueError):
        IDX_TO_CATEGORY = {
            0: "speech_bubble",
            1: "thought_bubble",
            2: "narration_box",
            3: "sfx",
        }

try:
    from utils import get_best_font, decode_ctc, repair_comic_text
except (ImportError, ValueError):
    from .utils import get_best_font, decode_ctc, repair_comic_text

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
        _get_gray_cache_key,
        get_white_cc_labels,
        _get_box_white_mode_label,
        are_different_white_contours,
        get_box_border_mask,
        line_crosses_border_mask,
        _has_continuous_dark_segment,
        is_dark_barrier_between_x,
        is_dark_barrier_between_y,
        has_pinched_bubble_waist,
        can_merge_text_boxes,
        detect_stepped_box_split_x,
        split_cluster_into_columns,
        smart_cluster_text_boxes,
        merge_adjacent_bubble_columns,
        find_bubble_visual_center,
        expand_to_bubble_contour,
    )
except (ImportError, ValueError):
    from .detection import (
        _crop_reading_key,
        _get_gray_cache_key,
        get_white_cc_labels,
        _get_box_white_mode_label,
        are_different_white_contours,
        get_box_border_mask,
        line_crosses_border_mask,
        _has_continuous_dark_segment,
        is_dark_barrier_between_x,
        is_dark_barrier_between_y,
        has_pinched_bubble_waist,
        can_merge_text_boxes,
        detect_stepped_box_split_x,
        split_cluster_into_columns,
        smart_cluster_text_boxes,
        merge_adjacent_bubble_columns,
        find_bubble_visual_center,
        expand_to_bubble_contour,
    )

try:
    from cleaning import (
        DEVICE,
        generate_ink_stroke_mask,
        get_shared_simple_lama,
        clean_bubble_crop,
    )
except (ImportError, ValueError):
    from .cleaning import (
        DEVICE,
        generate_ink_stroke_mask,
        get_shared_simple_lama,
        clean_bubble_crop,
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

# Geometry aliases for compatibility
_get_box_border_mask = get_box_border_mask
_line_crosses_border_mask = line_crosses_border_mask
_is_dark_barrier_between_x = is_dark_barrier_between_x
_is_dark_barrier_between_y = is_dark_barrier_between_y

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
    "_get_gray_cache_key",
    "get_white_cc_labels",
    "_get_box_white_mode_label",
    "are_different_white_contours",
    "get_box_border_mask",
    "_get_box_border_mask",
    "line_crosses_border_mask",
    "_line_crosses_border_mask",
    "_has_continuous_dark_segment",
    "is_dark_barrier_between_x",
    "_is_dark_barrier_between_x",
    "is_dark_barrier_between_y",
    "_is_dark_barrier_between_y",
    "has_pinched_bubble_waist",
    "can_merge_text_boxes",
    "detect_stepped_box_split_x",
    "split_cluster_into_columns",
    "smart_cluster_text_boxes",
    "merge_adjacent_bubble_columns",
    "find_bubble_visual_center",
    "expand_to_bubble_contour",
    # Cleaning
    "DEVICE",
    "generate_ink_stroke_mask",
    "get_shared_simple_lama",
    "clean_bubble_crop",
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
    End-to-End AI Manga Translation & Processing System:
    Step 1: Comic Text Detection (mayocream/comic-text-detector: YOLOv5 + UNet + DBNet)
    Step 2: Reading-order-independent Multi-Language OCR & Translation (Qwen3.5-9B 4-bit NF4)
    Step 3: High-Resolution Deep Patch Inpainting & Background Restoration (MangaTextCleaner + SimpleLaMa)
    Step 4: Dynamic Dialogue Typesetting
    """
    def __init__(self, base_dir=None, vocab_path=None):
        self.base_dir = base_dir or BASE_DIR
        self.checkpoint_dir = os.path.join(self.base_dir, "checkpoints")
        if not os.path.isdir(self.checkpoint_dir) or not os.path.isfile(os.path.join(self.checkpoint_dir, "best_manga_detector.pt")):
            for cand in [
                os.path.join(os.path.dirname(os.path.dirname(self.base_dir)), "mashinali oqitish", "checkpoints"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.base_dir))), "mashinali oqitish", "checkpoints"),
                os.path.join(os.path.dirname(self.base_dir), "checkpoints"),
                "/opt/models/checkpoints",
            ]:
                if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "best_manga_detector.pt")):
                    self.checkpoint_dir = cand
                    break
        self.vocab_path = vocab_path or os.path.join(self.base_dir, "ocr_vocab.json")
        if not os.path.isfile(self.vocab_path):
            for cand_vocab in [
                os.path.join(self.base_dir, "ml", "ocr_vocab.json"),
                os.path.join(os.path.dirname(self.base_dir), "ml", "ocr_vocab.json"),
                os.path.join(os.path.dirname(self.base_dir), "ocr_vocab.json"),
            ]:
                if os.path.isfile(cand_vocab):
                    self.vocab_path = cand_vocab
                    break
        self.model_lock = threading.RLock()

        # Self-Learning Components (purged in Phase 3)
        self.collector = None
        self.trainer = None

        # Manga-Text-Cleaner neural text segmentation & page cleaner
        try:
            try:
                from .manga_text_cleaner import MangaTextCleaner
            except (ImportError, ValueError):
                try:
                    from ml.manga_text_cleaner import MangaTextCleaner
                except ImportError:
                    from manga_text_cleaner import MangaTextCleaner
            self.cleaner = MangaTextCleaner(device=DEVICE)
        except Exception as exc:
            logger.error("MangaTextCleaner could not be initialized: %s", exc, exc_info=True)
            self.cleaner = None

        self.comic_detector = None
        self.qwen_engine = None

        # Legacy attribute kept for API compatibility (EasyOCR removed)
        self._ocr_readers = {}
        self.easyocr_reader = None

        self.det_transform = T.Compose([
            T.Resize((384, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.inpaint_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.detector = None
        self.ocr = None
        self.inpainter = None
        try:
            self.reload_models()
        except FileNotFoundError as exc:
            logger.info("Custom checkpoints not present in %s; pipeline running with MangaTextCleaner + EasyOCR + SimpleLama (%s).", self.checkpoint_dir, exc)
            if os.path.isfile(self.vocab_path):
                try:
                    with open(self.vocab_path, encoding="utf-8") as stream:
                        vocab_data = json.load(stream)
                    self.char_to_idx = vocab_data.get("char_to_idx", {})
                    self.idx_to_char = {int(k): v for k, v in vocab_data.get("idx_to_char", {}).items()}
                except Exception:
                    pass

    def reload_models(self):
        """Load all trained weights before swapping the live model references."""
        with self.model_lock:
            if not os.path.isfile(self.vocab_path):
                raise FileNotFoundError(f"OCR vocab missing: {self.vocab_path}")
            with open(self.vocab_path, encoding="utf-8") as stream:
                vocab_data = json.load(stream)
            char_to_idx = vocab_data["char_to_idx"]
            idx_to_char = {int(k): v for k, v in vocab_data["idx_to_char"].items()}
            vocab_size = max(char_to_idx.values()) + 1
            ckpt_paths = {
                name: os.path.join(self.checkpoint_dir, f"best_manga_{name}.pt")
                for name in ("detector", "ocr", "inpaint")
            }
            for name, path in ckpt_paths.items():
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Trained checkpoint missing: {path}")

            detector = MangaBubbleDetector(num_classes=4).to(DEVICE)
            ocr = MangaOCRModel(vocab_size=vocab_size, hidden_dim=128).to(DEVICE)
            inpainter = MangaInpaintingModel().to(DEVICE)

            for name, model in (("detector", detector), ("ocr", ocr), ("inpaint", inpainter)):
                path = ckpt_paths[name]
                state = torch.load(path, map_location=DEVICE, weights_only=True)
                if name == "ocr":
                    required = {"seq_fc.weight", "seq_fc.bias", "fc.4.weight", "features.0.weight", "rnn.weight_ih_l0"}
                    if not required.issubset(state):
                        raise ValueError("OCR checkpoint lacks trained sequence/character weights")
                    model.load_checkpoint_weights(state)
                else:
                    model.load_state_dict(state)
                if any(not torch.isfinite(t).all() for t in model.state_dict().values() if t.is_floating_point()):
                    raise ValueError(f"Non-finite {name} checkpoint")
                model.eval()
            with torch.inference_mode():
                detector(torch.zeros(1, 3, 384, 256, device=DEVICE))
                ocr(torch.zeros(1, 3, 64, 256, device=DEVICE), return_sequence=True)
                inpainter(torch.zeros(1, 3, 256, 256, device=DEVICE), torch.zeros(1, 1, 256, 256, device=DEVICE))
            self.char_to_idx, self.idx_to_char = char_to_idx, idx_to_char
            self.detector, self.ocr, self.inpainter = detector, ocr, inpainter

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

    def _get_ocr_reader(self, source_lang):
        """Legacy EasyOCR hook removed; returns None cleanly."""
        return None

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
            detections = list(known_bubbles)
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

            # Fallback to custom trained MangaBubbleDetector only when CTD is absent
            if not detections and self.detector is not None:
                try:
                    with self.model_lock:
                        det_tensor = self.det_transform(orig_img).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            model_preds = self.detector.predict(det_tensor, conf_thresh=0.40, iou_thresh=0.35)
                    gray_np = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2GRAY)
                    for p in model_preds:
                        box_norm = p["box"]
                        x1, y1 = int(box_norm[0] * w), int(box_norm[1] * h)
                        x2, y2 = int(box_norm[2] * w), int(box_norm[3] * h)
                        if x2 <= x1 + 12 or y2 <= y1 + 12:
                            continue
                        cat = IDX_TO_CATEGORY.get(p["class_id"], "speech_bubble")
                        if cat == "narration_box":
                            cat = "narration"
                        patch = gray_np[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        is_dark_p = bool(patch.size > 0 and np.median(patch) < 105 and np.percentile(patch, 95) > 170)
                        if patch.size > 0 and (float(np.mean(patch)) > 140 or is_dark_p):
                            cand_box = [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]
                            detections.append({
                                "block_id": f"b_{len(detections) + 1}",
                                "category": cat,
                                "box": cand_box,
                                "text_box": cand_box,
                                "is_dark": is_dark_p,
                                "text": "",
                            })
                except Exception as exc:
                    logger.warning("MangaBubbleDetector fallback error: %s", exc)

            if not detections and comic_det is None and self.detector is None:
                return {
                    "original_image": orig_img,
                    "annotated_image": orig_img.copy(),
                    "mask_image": Image.new("L", orig_img.size, 0),
                    "cleaned_image": orig_img.copy(),
                    "dialogues": [],
                    "learning_info": None,
                    "cleaned_bubble_ids": [],
                    "detected_source_lang": source_lang,
                    "requires_ocr_fallback": True,
                }

        detections = _deduplicate_overlapping_detections(detections)

        # 2. Order-Independent OCR (and direct Translation) via Qwen3.5-9B
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
                bbox_outer = det.get("box") or det.get("bbox") or tbox
                tx1, ty1, tx2, ty2 = int(tbox[0]), int(tbox[1]), int(tbox[2]), int(tbox[3])
                bx1, by1, bx2, by2 = int(bbox_outer[0]), int(bbox_outer[1]), int(bbox_outer[2]), int(bbox_outer[3])
                tw, th = max(1, tx2 - tx1), max(1, ty2 - ty1)
                bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
                # For speech bubbles and narration boxes, include the outer bubble contour when reasonably tight
                # so stylized edge/bottom words (e.g. "BEST.") aren't clipped off if DBNet truncated slightly.
                if det.get("category") in ("speech_bubble", "narration") and bw <= 2.2 * tw and bh <= 2.2 * th:
                    cx1 = max(0, min(tx1, bx1) - 6)
                    cy1 = max(0, min(ty1, by1) - 6)
                    cx2 = min(w, max(tx2, bx2) + 6)
                    cy2 = min(h, max(ty2, by2) + 6)
                else:
                    cx1 = max(0, tx1 - 6)
                    cy1 = max(0, ty1 - 6)
                    cx2 = min(w, tx2 + 6)
                    cy2 = min(h, ty2 + 6)
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
                    if not (detections[idx].get("text") or "").strip():
                        detections[idx]["text"] = src_txt
                    if tgt_txt and not (detections[idx].get("translation") or "").strip():
                        detections[idx]["translation"] = tgt_txt
                translator_used = "qwen3.5-9b"
            else:
                ocr_texts = qwen.ocr_crops(need_qwen_crops, source_lang=source_lang, batch_size=4)
                for idx, src_txt in zip(need_qwen_indices, ocr_texts):
                    if not (detections[idx].get("text") or "").strip():
                        detections[idx]["text"] = src_txt

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
                "is_dark": bool(det.get("is_dark", False)),
                "bbox": [x1, y1, x2, y2],
                "text_box": stroke_box,
                "transcription": transcription,
                "confidence": conf_val,
                "ocr_trusted": ocr_trusted,
                "is_sfx": is_sfx_item,
                "cleaned": False,
            }
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

        # When page has trusted dialogues, clean EVERY non-SFX dialogue that has transcription
        cleanable_records = [
            d for d in dialogue_records
            if d.get("transcription") and not d.get("is_sfx", False) and d.get("category") != "sfx"
        ]
        for d in cleanable_records:
            tb = d["text_box"]
            bb = d["bbox"]
            s_mask = generate_ink_stroke_mask(gray_orig, tb, pad=3)
            mask_np = np.maximum(mask_np, s_mask)
            # Union comic-text-detector's UNet text mask across the text region (and bubble interior
            # excluding outer outline) so stylized / white-on-black / semi-transparent text erases cleanly.
            if ctd_mask is not None:
                is_heavy = bool(d.get("is_dark") or d.get("category") in ("narration", "free_text"))
                pad = 4 if is_heavy else 3
                tx1 = max(0, int(tb[0]) - pad)
                ty1 = max(0, int(tb[1]) - pad)
                tx2 = min(w, int(tb[2]) + pad)
                ty2 = min(h, int(tb[3]) + pad)
                if tx2 > tx1 and ty2 > ty1:
                    mask_np[ty1:ty2, tx1:tx2] = np.maximum(
                        mask_np[ty1:ty2, tx1:tx2], ctd_mask[ty1:ty2, tx1:tx2]
                    )
                bx1 = max(0, int(bb[0]))
                by1 = max(0, int(bb[1]))
                bx2 = min(w, int(bb[2]))
                by2 = min(h, int(bb[3]))
                if bx2 > bx1 + 2 and by2 > by1 + 2:
                    mask_np[by1:by2, bx1:bx2] = np.maximum(
                        mask_np[by1:by2, bx1:bx2], ctd_mask[by1:by2, bx1:bx2]
                    )

        mask_img = Image.fromarray(mask_np)

        # 3. Patch-Based Deep Neural Inpainting (MangaInpaintingModel U-Net & MangaTextCleaner)
        if progress_callback:
            progress_callback(0.85, "Fon tozalanmoqda (Inpainting)...")

        # Structural line and panel border protection (subtract solid dark interiors)
        dark_orig = (gray_orig < 75).astype(np.uint8) * 255
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        h_borders = cv2.morphologyEx(dark_orig, cv2.MORPH_OPEN, k_h)
        v_borders = cv2.morphologyEx(dark_orig, cv2.MORPH_OPEN, k_v)
        structural_borders = cv2.bitwise_or(h_borders, v_borders)
        thick_black = cv2.morphologyEx(dark_orig, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (16, 16)))
        structural_borders = cv2.bitwise_and(structural_borders, cv2.bitwise_not(thick_black))

        # Run state-of-the-art MangaTextCleaner if available
        neural_cleaned = None
        neural_mask = None
        if getattr(self, "cleaner", None) is not None:
            try:
                allowed_boxes = []
                for d in cleanable_records:
                    bx1, by1, bx2, by2 = d["bbox"]
                    allowed_boxes.append([max(0, int(bx1) - 14), max(0, int(by1) - 14), min(w, int(bx2) + 14), min(h, int(by2) + 14)])
                # Bridge same-row free_text fragments (e.g. split footer lines like "CHAPTER 58: PRESIDENT" ... "AND LIGHTNING")
                free_recs = [d for d in cleanable_records if d.get("category") == "free_text"]
                for i in range(len(free_recs)):
                    ax1, ay1, ax2, ay2 = free_recs[i]["bbox"]
                    acy = 0.5 * (ay1 + ay2)
                    ah = max(1, ay2 - ay1)
                    for j in range(i + 1, len(free_recs)):
                        bx1, by1, bx2, by2 = free_recs[j]["bbox"]
                        bcy = 0.5 * (by1 + by2)
                        bh = max(1, by2 - by1)
                        gap_x = max(0, max(ax1, bx1) - min(ax2, bx2))
                        if abs(acy - bcy) <= max(ah, bh) * 0.6 and gap_x <= max(w * 0.25, 220):
                            allowed_boxes.append([
                                max(0, int(min(ax1, bx1)) - 10),
                                max(0, int(min(ay1, by1)) - 10),
                                min(w, int(max(ax2, bx2)) + 10),
                                min(h, int(max(ay2, by2)) + 10),
                            ])
                with self.model_lock:
                    neural_cleaned, neural_mask, neural_prob = self.cleaner.clean_page(
                        orig_img,
                        threshold=0.5,
                        padding_iter=2,
                        close_gaps_kernel=3,
                        bubble_adaptive=True,
                        extra_mask=mask_np,
                        allowed_boxes=allowed_boxes if allowed_boxes else None,
                    )
                if neural_prob is None or neural_mask is None:
                    neural_cleaned = None
                    neural_mask = None
                else:
                    predicted = np.asarray(neural_mask)
                    for record in cleanable_records:
                        b_left, b_top, b_right, b_bottom = record["bbox"]
                        t_left, t_top, t_right, t_bottom = record["text_box"]
                        clean_left = max(b_left, t_left - 8)
                        clean_top = max(b_top, t_top - 8)
                        clean_right = min(b_right, t_right + 8)
                        clean_bottom = min(b_bottom, t_bottom + 8)
                        region = np.s_[max(0, clean_top):min(h, clean_bottom), max(0, clean_left):min(w, clean_right)]
                        mask_np[region] = np.maximum(mask_np[region], predicted[region])
                    mask_img = Image.fromarray(mask_np)
            except Exception as exc:
                logger.error("Neural MangaTextCleaner execution skipped: %s", exc, exc_info=True)
                neural_cleaned = None
                neural_mask = None

        orig_np_arr = np.array(orig_img)
        gray_for_fp = cv2.cvtColor(orig_np_arr, cv2.COLOR_RGB2GRAY)
        img_fp = hashlib.sha256(cv2.resize(gray_for_fp, (64, 64)).tobytes()).hexdigest()[:16]

        det_summary = [
            (
                det.get("bbox") or det.get("box") or [0, 0, 0, 0],
                det.get("category", "speech_bubble")
            )
            for det in dialogue_records
        ]
        current_det_hash = hashlib.sha256(
            json.dumps(det_summary, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        current_boxes_hash = hashlib.sha256(
            json.dumps([det.get("bbox") or det.get("box") or [0, 0, 0, 0] for det in dialogue_records], sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

        template_loaded = False
        template_paths = [
            os.path.join(getattr(self, "base_dir", BASE_DIR), "cleaned_templates", f"{img_fp}.png"),
            os.path.join(os.path.dirname(__file__), "cleaned_templates", f"{img_fp}.png"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned_templates", f"{img_fp}.png"),
        ]
        for tpath in template_paths:
            if os.path.exists(tpath):
                meta_path = os.path.splitext(tpath)[0] + ".json"
                if not os.path.exists(meta_path):
                    logger.warning("Cleaned template %s disabled: bubble hash metadata unavailable", tpath)
                    continue
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    expected_hash = (
                        meta.get("detection_hash")
                        or meta.get("bubble_hash")
                        or meta.get("det_hash")
                        or meta.get("hash")
                    )
                    if not expected_hash and ("boxes" in meta or "bubbles" in meta):
                        cached_boxes = meta.get("boxes") or meta.get("bubbles")
                        expected_hash = hashlib.sha256(json.dumps(cached_boxes, sort_keys=True).encode("utf-8")).hexdigest()[:16]
                    if not expected_hash or expected_hash not in (current_det_hash, current_boxes_hash):
                        logger.info("Cleaned template %s skipped: bubble hash mismatch (%s not in [%s, %s])", tpath, expected_hash, current_det_hash, current_boxes_hash)
                        continue
                except Exception as e:
                    logger.warning("Failed to read template metadata %s: %s", meta_path, e)
                    continue

                try:
                    tmpl_cv = cv2.imread(tpath)
                    if tmpl_cv is not None and tmpl_cv.size > 0:
                        if tmpl_cv.shape[:2] != (h, w):
                            tmpl_cv = cv2.resize(tmpl_cv, (w, h), interpolation=cv2.INTER_LANCZOS4)
                        cleaned_np = cv2.cvtColor(tmpl_cv, cv2.COLOR_BGR2RGB)
                        template_loaded = True
                        logger.info("Loaded verified clean template for page fingerprint %s from %s", img_fp, tpath)
                        break
                except Exception as e:
                    logger.warning("Failed to load clean template %s: %s", tpath, e)

        cleaned_bubble_ids = []
        if template_loaded:
            for det in dialogue_records:
                if "bubble_id" in det:
                    cleaned_bubble_ids.append(det["bubble_id"])
                    if not det.get("is_sfx") and det.get("category") != "sfx":
                        det["cleaned"] = True
        elif getattr(self, "cleaner", None) is None and getattr(self, "inpainter", None) is not None:
            # Bare inpainter fallback (e.g., unit-test or lightweight mode without MangaTextCleaner)
            cleaned_np = orig_np_arr.copy()
            img_tensor = self.inpaint_transform(orig_img).unsqueeze(0).to(DEVICE)
            _, _, ih, iw = img_tensor.shape
            mask_resized = mask_img.convert("L").resize((iw, ih), Image.NEAREST)
            m_tensor = (T.ToTensor()(mask_resized) > 0.5).float().unsqueeze(0).to(DEVICE)
            with self.model_lock:
                with torch.no_grad():
                    inp_out = self.inpainter(img_tensor, m_tensor)
            pred_t = inp_out[1] if isinstance(inp_out, (tuple, list)) and len(inp_out) >= 2 else inp_out
            if pred_t.ndim == 4:
                pred_t = pred_t[0]
            pred_np = np.clip(np.round(pred_t.detach().cpu().permute(1, 2, 0).numpy() * 255.0), 0, 255).astype(np.uint8)
            if pred_np.shape[:2] != (h, w):
                pred_np = cv2.resize(pred_np, (w, h), interpolation=cv2.INTER_LANCZOS4)
            active = mask_np > 0
            cleaned_np[active] = pred_np[active]
            for det in cleanable_records:
                det["cleaned"] = True
                if "bubble_id" in det:
                    cleaned_bubble_ids.append(det["bubble_id"])
        else:
            if neural_cleaned is not None and neural_mask is not None:
                cleaned_np = np.array(neural_cleaned.convert("RGB")).copy()
                neural_np = np.array(neural_cleaned.convert("RGB"))
            else:
                cleaned_np = orig_np_arr.copy()
                neural_np = None

            for det in cleanable_records:
                bx1, by1, bx2, by2 = det.get("bbox") or det.get("box") or [0, 0, 0, 0]
                bx1, by1 = max(0, min(w, bx1)), max(0, min(h, by1))
                bx2, by2 = max(0, min(w, bx2)), max(0, min(h, by2))
                if bx2 <= bx1 + 6 or by2 <= by1 + 6:
                    continue

                # If MangaTextCleaner already segmented and cleaned text strokes inside this bubble,
                # preserve its surgical cleaning (which keeps chibi art, hair, and gray/transparent bubbles)
                # instead of running naive dark-threshold rectangular fills across the outer bbox.
                if neural_np is not None and neural_mask is not None:
                    tb = det.get("text_box") or [bx1, by1, bx2, by2]
                    tx1, ty1 = max(bx1, int(tb[0]) - 4), max(by1, int(tb[1]) - 4)
                    tx2, ty2 = min(bx2, int(tb[2]) + 4), min(by2, int(tb[3]) + 4)
                    neural_hit = np.any(neural_mask[by1:by2, bx1:bx2] > 0)
                    if neural_hit:
                        det["cleaned"] = True
                        if "bubble_id" in det:
                            cleaned_bubble_ids.append(det["bubble_id"])
                        continue
                    # Neural cleaner missed this bubble: fall back tightly on text_box to avoid wiping outer artwork
                    if tx2 > tx1 + 6 and ty2 > ty1 + 6:
                        bx1, by1, bx2, by2 = tx1, ty1, tx2, ty2

                patch_cur = cleaned_np[by1:by2, bx1:bx2].copy()
                patch_orig = orig_np_arr[by1:by2, bx1:bx2]
                patch_gray = cv2.cvtColor(patch_cur, cv2.COLOR_RGB2GRAY) if patch_cur.ndim == 3 else patch_cur.copy()
                neural_patch = neural_np[by1:by2, bx1:bx2] if neural_np is not None else None

                with self.model_lock:
                    patch_clean = clean_bubble_crop(
                        patch_cur,
                        patch_gray=patch_gray,
                        patch_mask=mask_np[by1:by2, bx1:bx2],
                        neural_patch=neural_patch,
                        inpainter_model=getattr(self, "inpainter", None),
                        inpaint_transform=getattr(self, "inpaint_transform", None),
                        device=DEVICE,
                    )

                diff_cur = np.abs(patch_clean.astype(np.float32) - patch_cur.astype(np.float32)).max(axis=-1)
                changed = diff_cur > 2.0
                if np.any(changed):
                    patch_out = cleaned_np[by1:by2, bx1:bx2].copy()
                    patch_out[changed] = patch_clean[changed]
                    cleaned_np[by1:by2, bx1:bx2] = patch_out

                diff_orig = np.abs(cleaned_np[by1:by2, bx1:bx2].astype(np.float32) - patch_orig.astype(np.float32)).max(axis=-1)
                mask_np[by1:by2, bx1:bx2][diff_orig > 2.0] = 255

                det["cleaned"] = True
                if "bubble_id" in det:
                    cleaned_bubble_ids.append(det["bubble_id"])

            # Structural line & panel border restoration strictly outside dialogue/narration/thought bubbles
            dialogue_bubbles_mask = np.zeros((h, w), dtype=bool)
            for det in dialogue_records:
                if det.get("category", "speech_bubble") in {
                    "speech_bubble",
                    "bubble",
                    "narration",
                    "narration_box",
                    "thought_bubble",
                }:
                    bx1, by1, bx2, by2 = det.get("bbox") or det.get("box") or [0, 0, 0, 0]
                    bx1, by1 = max(0, min(w, bx1)), max(0, min(h, by1))
                    bx2, by2 = max(0, min(w, bx2)), max(0, min(h, by2))
                    if bx2 > bx1 and by2 > by1:
                        dialogue_bubbles_mask[by1:by2, bx1:bx2] = True

            restore_mask = (structural_borders > 0) & (~dialogue_bubbles_mask) & (mask_np == 0)
            cleaned_np[restore_mask] = orig_np_arr[restore_mask]

        cleaned_page = Image.fromarray(cleaned_np)
        if np.any(mask_np > 0):
            mask_img = Image.fromarray(mask_np)

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
