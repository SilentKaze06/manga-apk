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


class MangaTranslationPipeline:
    """
    End-to-End AI Manga Translation & Processing System:
    Step 1: Text-Guided Speech Bubble Detection (EasyOCR + MangaBubbleDetector ensemble with smart clustering)
    Step 2: Multi-Language OCR (Japanese / Korean / Chinese / English)
    Step 3: High-Resolution Deep Patch Inpainting & Background Restoration (U-Net)
    Step 4: Continuous Self-Learning & Dynamic Human Feedback Loop
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

        # Readers are selected per request. Never download weights implicitly.
        self._ocr_readers = {}
        self.easyocr_reader = None  # retained for callers inspecting availability

        self.det_transform = T.Compose([
            T.Resize((384, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.inpaint_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])
        self.reload_models()

    def reload_models(self):
        """Load all trained weights before swapping the live model references."""
        with self.model_lock:
            if not os.path.isfile(self.vocab_path):
                logger.warning("ocr_vocab.json not found at %s; relying on neural segmentation and EasyOCR.", self.vocab_path)
                return
            with open(self.vocab_path, encoding="utf-8") as stream:
                vocab_data = json.load(stream)
            char_to_idx = vocab_data["char_to_idx"]
            idx_to_char = {int(k): v for k, v in vocab_data["idx_to_char"].items()}
            vocab_size = max(char_to_idx.values()) + 1
            detector = MangaBubbleDetector(num_classes=4).to(DEVICE)
            ocr = MangaOCRModel(vocab_size=vocab_size, hidden_dim=128).to(DEVICE)
            inpainter = MangaInpaintingModel().to(DEVICE)
            
            # Check if custom checkpoints are present
            has_checkpoints = True
            for name in ("detector", "ocr", "inpaint"):
                path = os.path.join(self.checkpoint_dir, f"best_manga_{name}.pt")
                if not os.path.isfile(path):
                    has_checkpoints = False
                    break
            
            if not has_checkpoints:
                logger.info("Custom checkpoints not present in %s; pipeline running with MangaTextCleaner + EasyOCR + SimpleLama.", self.checkpoint_dir)
                self.char_to_idx, self.idx_to_char = char_to_idx, idx_to_char
                self.detector, self.ocr, self.inpainter = detector, ocr, inpainter
                return

            for name, model in (("detector", detector), ("ocr", ocr), ("inpaint", inpainter)):
                path = os.path.join(self.checkpoint_dir, f"best_manga_{name}.pt")
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

    def _get_ocr_reader(self, source_lang):
        languages = {
            "ja": ["ja", "en"],
            "ko": ["ko", "en"],
            "zh": ["ch_sim", "en"],
            "en": ["en"],
            "ru": ["ru", "en"],
            "auto": ["ja", "en"],
        }
        lang_list = languages.get(source_lang, ["ja", "en"] if source_lang == "auto" else ["en"])
        key = tuple(lang_list)
        with self.model_lock:
            if key not in self._ocr_readers:
                try:
                    import easyocr
                    options = {}
                    model_dir = os.environ.get("MANGA_EASYOCR_MODEL_DIR") or os.environ.get("EASYOCR_MODULE_PATH")
                    if model_dir:
                        options["model_storage_directory"] = model_dir
                        options["download_enabled"] = False
                    else:
                        options["download_enabled"] = (os.environ.get("HF_HUB_OFFLINE") != "1")
                    self._ocr_readers[key] = easyocr.Reader(
                        lang_list, gpu=torch.cuda.is_available(),
                        verbose=False, **options)
                except Exception as exc:
                    logger.warning("Local EasyOCR %s unavailable: %s", key, exc)
                    return None
            reader = self._ocr_readers.get(key)
            self.easyocr_reader = reader
            return reader

    def process_page(self, image_input, known_bubbles=None, auto_collect=True, is_verified=False, progress_callback=None, source_lang="auto", reading_direction="rtl"):
        """
        Processes a single real manga page image with high-precision bubble detection.
        """
        if progress_callback:
            progress_callback(0.08, "Sahifa tahlilga tayyorlanmoqda...")

        if isinstance(image_input, str):
            orig_img = Image.open(image_input).convert("RGB")
        else:
            orig_img = image_input.convert("RGB")

        source_lang = (source_lang or "auto").strip().lower()
        source_lang = {"ch": "zh", "ch_sim": "zh", "jp": "ja"}.get(source_lang, source_lang)
        if source_lang not in {"auto", "ja", "ko", "zh", "en", "ru"}:
            raise ValueError(f"Unsupported OCR source language: {source_lang}")

        w, h = orig_img.size
        reader = self._get_ocr_reader(source_lang)
        self.easyocr_reader = reader

        # 1. Text-Guided Speech Bubble Detection & Clustering
        detections = []
        if known_bubbles:
            detections = known_bubbles
        else:
            raw_boxes = []

            # Step A: Detect text lines via OCR
            if reader is not None:
                if progress_callback:
                    progress_callback(0.25, "EasyOCR butun sahifa matnlarini öqimoqda...")
                try:
                    orig_np = np.array(orig_img)
                    with self.model_lock:
                        if source_lang == "auto":
                            # Detect primary script on the page (Kana/Kanji -> ja, Hangul -> ko, Cyrillic -> ru, Hanzi -> ch/zh, Latin -> en)
                            reader_en = self._get_ocr_reader("en")
                            raw_ocr = []
                            if reader_en is not None:
                                raw_ocr = list(reader_en.readtext(orig_np, canvas_size=2000, mag_ratio=1.8))

                            combined_sample = " ".join(it[1] for it in raw_ocr if it[2] >= 0.20)
                            detected_script = detect_primary_script_from_text(combined_sample)

                            confident_en = sum(1 for it in raw_ocr if it[2] >= 0.35 and any(c.isascii() and c.isalpha() for c in it[1]))
                            if confident_en >= 2 and detected_script == "en":
                                reader = reader_en
                                source_lang = "en"
                            else:
                                target_lang = detected_script if detected_script in ("ja", "ko", "ru", "zh") else "ja"
                                script_reader = self._get_ocr_reader(target_lang)
                                if script_reader is not None:
                                    reader = script_reader
                                    raw_ocr = list(reader.readtext(orig_np, canvas_size=2000, mag_ratio=1.8))
                                    source_lang = target_lang
                                elif reader is not None:
                                    raw_ocr = list(reader.readtext(orig_np, canvas_size=2000, mag_ratio=1.8))
                        else:
                            raw_ocr = list(reader.readtext(orig_np, canvas_size=2000, mag_ratio=1.8))

                    # Dual polarity: If page has dark areas, run OCR pass on inverted image
                    gray_full_a = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
                    has_dark_areas = (float(np.mean(gray_full_a < 95)) > 0.03) or (float(np.percentile(gray_full_a, 20)) < 85)
                    if has_dark_areas:
                        inv_np = apply_dark_box_clahe_inversion(orig_np)
                        if isinstance(inv_np, Image.Image):
                            inv_np = np.array(inv_np)
                        with self.model_lock:
                            inv_raw_ocr = reader.readtext(inv_np, canvas_size=2000, mag_ratio=1.8)
                        for inv_item in inv_raw_ocr:
                            ibbox, itext, iconf = inv_item
                            ixs = [p[0] for p in ibbox]
                            iys = [p[1] for p in ibbox]
                            ix1, iy1 = int(min(ixs)), int(min(iys))
                            ix2, iy2 = int(max(ixs)), int(max(iys))
                            patch_orig = gray_full_a[max(0, iy1):min(h, iy2), max(0, ix1):min(w, ix2)]
                            if patch_orig.size > 0 and (np.median(patch_orig) < 115 or np.percentile(patch_orig, 85) < 150):
                                overlap_idx = None
                                for idx_ex, ex in enumerate(raw_ocr):
                                    ex_xs = [p[0] for p in ex[0]]
                                    ex_ys = [p[1] for p in ex[0]]
                                    ex1, ey1 = int(min(ex_xs)), int(min(ex_ys))
                                    ex2, ey2 = int(max(ex_xs)), int(max(ex_ys))
                                    if not (ix2 < ex1 or ix1 > ex2 or iy2 < ey1 or iy1 > ey2):
                                        overlap_idx = idx_ex
                                        break
                                if overlap_idx is None:
                                    raw_ocr.append(inv_item)
                                elif iconf > raw_ocr[overlap_idx][2] + 0.05:
                                    raw_ocr[overlap_idx] = inv_item

                    for item in raw_ocr:
                        bbox, text, conf = item
                        text = text.strip()
                        xs = [p[0] for p in bbox]
                        ys = [p[1] for p in bbox]
                        # 3px safety padding around detected OCR text boxes so edge letters (like H in HARMED) are not clipped
                        bx1, by1 = max(0, int(min(xs)) - 3), max(0, int(min(ys)) - 3)
                        bx2, by2 = min(w, int(max(xs)) + 3), min(h, int(max(ys)) + 3)
                        bw = bx2 - bx1
                        bh = by2 - by1

                        # Intercept vertically stacked characters (e.g. S-O-R-R-Y, N-O, W-H-A-T)
                        if not is_manga_punctuation(text) and bh >= 1.6 * bw and bh >= 26 and (len(text) <= 2 or text in {"1", "l", "I", "|", "!", "/", "i", ";", ":"} or conf < 0.60):
                            strip_crop = orig_np[by1:by2, bx1:bx2]
                            with self.model_lock:
                                v_res = try_read_vertical_strip(strip_crop, reader)
                            if v_res:
                                v_text, v_conf = v_res
                                if len(v_text) >= 2:
                                    text = v_text
                                    conf = max(conf, float(v_conf), 0.75)

                        text = normalize_cyrillic_ocr(text)
                        text = text.strip("_~-=+*`\\/| \t\n")
                        if conf < 0.06 or not text:
                            continue

                        det_patch = orig_np[by1:by2, bx1:bx2]
                        is_on_light_bubble = False
                        if det_patch.size > 0:
                            det_gray = cv2.cvtColor(det_patch, cv2.COLOR_RGB2GRAY)
                            bg_brightness = float(np.percentile(det_gray, 85))
                            is_on_light_bubble = (bg_brightness >= 170)

                        if is_ocr_garbage_noise(text, source_lang=source_lang, is_inside_bubble=is_on_light_bubble):
                            if is_on_light_bubble and any(c in text for c in "?!.:;%"):
                                text = "..." if "%" in text else text
                            else:
                                continue

                        # Screentone noise protection: filter out low-confidence stray artifacts on artwork/clothing
                        if not is_on_light_bubble:
                            det_gray = cv2.cvtColor(det_patch, cv2.COLOR_RGB2GRAY) if det_patch.size > 0 else np.array([])
                            med_det = float(np.median(det_gray)) if det_gray.size > 0 else 255.0
                            p95_det = float(np.percentile(det_gray, 95)) if det_gray.size > 0 else 0.0
                            is_dark_frame = (med_det < 115 and p95_det > 165)

                            clean_tok = re.sub(r'[^\w]', '', text).lower()
                            is_known_sfx = clean_tok in {"топ", "тап", "тук", "пик", "бум", "бах", "тмп", "хлоп", "шурх"}
                            valid_short_words = {"i", "my", "it", "no", "to", "me"}
                            is_valid_short = is_dark_frame and (clean_tok in valid_short_words or is_known_sfx)

                            if is_dark_frame:
                                if conf < 0.15 and not is_known_sfx:
                                    continue
                                if len(clean_tok) <= 2 and not is_known_sfx and not is_valid_short:
                                    continue
                            else:
                                if conf < 0.40 and not is_known_sfx:
                                    continue
                                if len(clean_tok) <= 2 and not is_known_sfx:
                                    continue
                        if any(c.isascii() and c.isalpha() for c in text):
                            try:
                                text = repair_comic_text(text)
                            except Exception as rec_exc:
                                logger.warning("repair_comic_text error: %s", rec_exc)
                        raw_boxes.append({
                            "box": [bx1, by1, bx2, by2],
                            "text": text,
                            "conf": float(conf)
                        })
                except Exception as e:
                    logger.warning("EasyOCR readtext exception: %s", e)

            # Step B: Boundary-Aware Dialogue Grouping & Bubble Contour Snapping
            gray_full = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2GRAY)
            grouped_clusters = smart_cluster_text_boxes(raw_boxes, gray_full, reading_direction=reading_direction)

            clusters = []
            all_tboxes = [c["text_box"] for c in grouped_clusters]
            for idx, c in enumerate(grouped_clusters):
                tbox = c["text_box"]
                other_tboxes = [b for j, b in enumerate(all_tboxes) if j != idx]
                bubble_box = expand_to_bubble_contour(
                    gray_full, tbox, other_tboxes,
                    limit_left=c.get("limit_left"),
                    limit_right=c.get("limit_right")
                )
                t_patch = gray_full[max(0, tbox[1]):min(h, tbox[3]), max(0, tbox[0]):min(w, tbox[2])]
                p85_t = float(np.percentile(t_patch, 85)) if t_patch.size > 0 else 255.0
                words = c["text"].split()
                clean_txt = re.sub(r'[^\w]', '', c["text"]).lower()
                is_known_sfx_word = is_sfx_text(c["text"])

                b_patch = gray_full[max(0, bubble_box[1]):min(h, bubble_box[3]), max(0, bubble_box[0]):min(w, bubble_box[2])]
                med_b = float(np.median(b_patch)) if b_patch.size > 0 else 255.0
                med_t = float(np.median(t_patch)) if t_patch.size > 0 else 255.0
                p95_b = float(np.percentile(b_patch, 95)) if b_patch.size > 0 else 0.0
                p95_t = float(np.percentile(t_patch, 95)) if t_patch.size > 0 else 0.0
                is_dark_narration = (c.get("category") == "narration") or ((med_b < 95 or med_t < 95) and (p95_b > 180 or p95_t > 180))

                if is_dark_narration and not is_known_sfx_word:
                    # Dark narration frame with bright text: never classify as sfx
                    is_art = False
                    category = "narration"
                    if reader is not None:
                        try:
                            inv_bx1, inv_by1 = max(0, bubble_box[0]), max(0, bubble_box[1])
                            inv_bx2, inv_by2 = min(w, bubble_box[2]), min(h, bubble_box[3])
                            crop_dark = gray_full[inv_by1:inv_by2, inv_bx1:inv_bx2]
                            if crop_dark.size > 0 and (np.median(crop_dark) < 110 or med_t < 95):
                                inv_rgb = apply_dark_box_clahe_inversion(crop_dark)
                                if isinstance(inv_rgb, Image.Image):
                                    inv_rgb = np.array(inv_rgb)
                                with self.model_lock:
                                    inv_ocr = reader.readtext(inv_rgb, mag_ratio=1.5)
                                if inv_ocr:
                                    valid_inv = [r for r in inv_ocr if r[2] >= 0.15 and r[1].strip()]
                                    if valid_inv:
                                        inv_text = " ".join(r[1].strip() for r in valid_inv)
                                        if any(ch.isascii() and ch.isalpha() for ch in inv_text):
                                            try:
                                                inv_text = repair_comic_text(inv_text)
                                            except Exception as e_repair:
                                                logger.debug("repair_comic_text on inverted crop error: %s", e_repair)
                                        if len(inv_text) >= len(c["text"]):
                                            c["text"] = inv_text
                                            c["conf"] = max(r[2] for r in valid_inv)
                        except Exception as e_inv:
                            logger.warning("Inverted dark narration OCR error: %s", e_inv)
                else:
                    is_art = (p85_t < 170 and len(words) <= 2) or is_known_sfx_word
                    category = "sfx" if is_art else "speech_bubble"

                clusters.append({
                    "block_id": c.get("block_id", f"b_{len(clusters) + 1}"),
                    "category": category,
                    "box": bubble_box,
                    "text_box": tbox,
                    "text": c["text"],
                    "conf": c.get("conf", 1.0),
                    "avg_conf": c.get("avg_conf", c.get("conf", 1.0)),
                    "is_on_art": is_art,
                    "is_sfx": is_art,
                    "limit_left": c.get("limit_left"),
                    "limit_right": c.get("limit_right"),
                })

            # Step C: Complement with MangaBubbleDetector (Only if proposals have high whiteness / bubble shape)
            try:
                with self.model_lock:
                    det_tensor = self.det_transform(orig_img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        model_preds = self.detector.predict(det_tensor, conf_thresh=0.60, iou_thresh=0.35)

                gray_np = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2GRAY)
                for p in model_preds:
                    box_norm = p["box"]
                    x1, y1 = int(box_norm[0] * w), int(box_norm[1] * h)
                    x2, y2 = int(box_norm[2] * w), int(box_norm[3] * h)
                    if x2 <= x1 + 25 or y2 <= y1 + 25:
                        continue
                    patch = gray_np[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    cat = IDX_TO_CATEGORY.get(p["class_id"], "speech_bubble")
                    valid_proposal = False
                    if patch.size > 0:
                        if cat in ("narration", "narration_box"):
                            med_p = float(np.median(patch))
                            p95_p = float(np.percentile(patch, 95))
                            valid_proposal = (med_p < 115 and p95_p > 165) or (float(np.mean(patch)) > 200)
                        else:
                            valid_proposal = float(np.mean(patch)) > 205

                    if valid_proposal:
                        cand_box = [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]
                        overlap = False
                        for c in clusters:
                            cb = c["box"]
                            ixA = max(cand_box[0], cb[0])
                            iyA = max(cand_box[1], cb[1])
                            ixB = min(cand_box[2], cb[2])
                            iyB = min(cand_box[3], cb[3])
                            inter = max(0, ixB - ixA) * max(0, iyB - iyA)
                            areaA = max(0, cand_box[2] - cand_box[0]) * max(0, cand_box[3] - cand_box[1])
                            areaB = max(0, cb[2] - cb[0]) * max(0, cb[3] - cb[1])
                            union = areaA + areaB - inter
                            iou = inter / union if union > 0 else 0.0
                            if iou >= 0.40:
                                overlap = True
                                break
                        if not overlap:
                            clusters.append({
                                "block_id": f"b_{len(clusters) + 1}",
                                "category": cat,
                                "box": cand_box,
                                "text": ""
                            })
            except Exception as e:
                logger.warning("MangaBubbleDetector verification error: %s", e)

            # Step D: Final filtering - eliminate dark false positives (bushes, dresses, legs, sky)
            gray_np = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2GRAY)
            for c in clusters:
                bx = c["box"]
                text = c.get("text", "").strip()
                patch = gray_np[bx[1]:bx[3], bx[0]:bx[2]]
                mean_lum = np.mean(patch) if patch.size > 0 else 0
                # Must have confirmed dialogue text OR be a bright speech bubble OR narration frame
                if text or mean_lum > 215 or c.get("category") == "narration":
                    detections.append(c)

        # 2. OCR and Inpainting Mask Generation
        if progress_callback:
            progress_callback(0.65, "Nutq bulutlari va matnlar klasterlanmoqda...")

        annotated_img = orig_img.copy()
        draw_annotated = ImageDraw.Draw(annotated_img)

        mask_np = np.zeros((h, w), dtype=np.uint8)
        orig_np_arr = np.array(orig_img)
        gray_orig = cv2.cvtColor(orig_np_arr, cv2.COLOR_RGB2GRAY)

        dialogue_records = []
        font_badge = get_best_font(size=20, bold=True)

        with self.model_lock:
            for idx, det in enumerate(detections):
                box = det.get("box") or det.get("bbox") or det.get("text_box") or [0, 0, 0, 0]  # [x1, y1, x2, y2]
                cat = det.get("category", "speech_bubble")
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])

                # Use detected text or run OCR crop recognition
                transcription = det.get("text", "").strip()
                conf_val = float(det.get("avg_conf", det.get("conf", 1.0)))
                if is_manga_punctuation(transcription):
                    transcription = normalize_manga_punctuation(transcription)
                    ocr_trusted = (conf_val >= 0.20) and not is_ocr_garbage_noise(transcription, source_lang, is_inside_bubble=(cat == "speech_bubble"))
                else:
                    ocr_trusted = bool(transcription) and (conf_val >= 0.18 or not is_ocr_garbage_noise(transcription, source_lang, is_inside_bubble=(cat == "speech_bubble")))

                # Crop OCR & High-Resolution Standalone Character Refinement
                # Do not skip "narration": if background is dark, invert crop and re-OCR
                if reader is not None and cat in ("speech_bubble", "narration", "narration_box") and x2 > x1 + 15 and y2 > y1 + 15:
                    try:
                        crop_cv = orig_np_arr[y1:y2, x1:x2]
                        if crop_cv.size > 0:
                            is_dark_bg = is_dark_background(crop_cv)
                            crop_for_ocr = apply_dark_box_clahe_inversion(crop_cv) if is_dark_bg else crop_cv
                            if isinstance(crop_for_ocr, Image.Image):
                                crop_for_ocr = np.array(crop_for_ocr)
                            crop_2x = cv2.resize(crop_for_ocr, (crop_for_ocr.shape[1] * 2, crop_for_ocr.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
                            # Light contrast normalization/sharpening on crop_2x
                            if crop_2x.shape[0] > 4 and crop_2x.shape[1] > 4 and (int(np.max(crop_2x)) - int(np.min(crop_2x))) > 15:
                                norm_2x = cv2.normalize(crop_2x, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                                blurred_2x = cv2.GaussianBlur(norm_2x, (0, 0), 1.2)
                                crop_2x = cv2.addWeighted(norm_2x, 1.25, blurred_2x, -0.25, 0)

                            res_2x = reader.readtext(crop_2x)
                            valid_items = [r for r in res_2x if r[2] >= 0.15 and r[1].strip()]
                            if valid_items:
                                col_map = {id(it): 0 for it in valid_items}
                                if len(valid_items) > 2:
                                    sorted_by_cx = sorted(valid_items, key=lambda it: sum(pt[0] / 2.0 for pt in it[0]) / len(it[0]))
                                    best_split = -1
                                    max_gap = -9999
                                    for i in range(len(sorted_by_cx) - 1):
                                        lg = sorted_by_cx[:i+1]
                                        rg = sorted_by_cx[i+1:]
                                        l_xmax = max(max(pt[0] / 2.0 for pt in it[0]) for it in lg)
                                        r_xmin = min(min(pt[0] / 2.0 for pt in it[0]) for it in rg)
                                        l_ymin = min(min(pt[1] / 2.0 for pt in it[0]) for it in lg)
                                        l_ymax = max(max(pt[1] / 2.0 for pt in it[0]) for it in lg)
                                        r_ymin = min(min(pt[1] / 2.0 for pt in it[0]) for it in rg)
                                        r_ymax = max(max(pt[1] / 2.0 for pt in it[0]) for it in rg)
                                        y_ov = min(l_ymax, r_ymax) - max(l_ymin, r_ymin)
                                        min_h = min(l_ymax - l_ymin, r_ymax - r_ymin)
                                        mid_l = sum(sum(pt[0] / 2.0 for pt in it[0]) / len(it[0]) for it in lg) / len(lg)
                                        mid_r = sum(sum(pt[0] / 2.0 for pt in it[0]) / len(it[0]) for it in rg) / len(rg)
                                        gap = r_xmin - l_xmax
                                        if min_h > 0 and y_ov > 0.35 * min_h and (mid_r - mid_l) > 25 and gap > 0:
                                            if gap > max_gap:
                                                max_gap = gap
                                                best_split = i
                                    if best_split != -1 and max_gap > 0:
                                        is_cjk_crop = any(
                                            any('\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u30ff' for ch in it[1])
                                            for it in valid_items
                                        )
                                        left_ids = {id(it) for it in sorted_by_cx[:best_split+1]}
                                        for it in valid_items:
                                            is_left = id(it) in left_ids
                                            col_map[id(it)] = (1 if is_left else 0) if (reading_direction == "rtl" or is_cjk_crop) else (0 if is_left else 1)

                                # Group items into lines per column by vertical overlap
                                col_items = {}
                                for it in valid_items:
                                    c = col_map.get(id(it), 0)
                                    col_items.setdefault(c, []).append(it)

                                line_idx_map = {}
                                sorted_items = []
                                global_line_counter = 0
                                for c in sorted(col_items.keys()):
                                    items_c = col_items[c]
                                    sorted_by_yc = sorted(items_c, key=lambda it: sum(pt[1] / 2.0 for pt in it[0]) / len(it[0]))
                                    c_lines = []
                                    for it in sorted_by_yc:
                                        ys_it = [pt[1] / 2.0 for pt in it[0]]
                                        h_it = max(ys_it) - min(ys_it)
                                        y1_it = min(ys_it)
                                        y2_it = max(ys_it)
                                        matched_line = None
                                        for line in c_lines:
                                            l_y1 = min(min(pt[1] / 2.0 for pt in x[0]) for x in line)
                                            l_y2 = max(max(pt[1] / 2.0 for pt in x[0]) for x in line)
                                            l_min_h = min(max(pt[1] / 2.0 for pt in x[0]) - min(pt[1] / 2.0 for pt in x[0]) for x in line)
                                            y_ov = min(y2_it, l_y2) - max(y1_it, l_y1)
                                            if y_ov >= 0.45 * min(h_it, l_min_h):
                                                matched_line = line
                                                break
                                        if matched_line is not None:
                                            matched_line.append(it)
                                        else:
                                            c_lines.append([it])

                                    # Sort each line by x1 (left to right)
                                    for line in c_lines:
                                        line.sort(key=lambda it: min(pt[0] / 2.0 for pt in it[0]))
                                    # Sort lines top-to-bottom
                                    c_lines.sort(key=lambda line: min(min(pt[1] / 2.0 for pt in it[0]) for it in line))

                                    for line in c_lines:
                                        for it in line:
                                            line_idx_map[id(it)] = global_line_counter
                                        global_line_counter += 1
                                        sorted_items.append(line)

                                valid_items.sort(key=lambda r: _crop_reading_key(r, reading_direction=reading_direction, col_map=col_map, line_idx_map=line_idx_map))

                                min_crop_y = float(h)
                                min_crop_x = float(w)
                                max_crop_x = 0.0
                                max_crop_y = 0.0
                                line_words_list = []
                                for line in sorted_items:
                                    lw = []
                                    for r in line:
                                        t = r[1].strip()
                                        t = normalize_cyrillic_ocr(t)
                                        if is_ocr_garbage_noise(t, source_lang=source_lang, is_inside_bubble=True):
                                            continue
                                        if t in {"l", "|"}:
                                            t = "I"
                                        t = re.sub(r'(?<![\d\-])\b1\b(?!\s*(?:st|nd|rd|th)\b)(?![\d\-])', 'I', t, flags=re.I)
                                        t = re.sub(r'\b1(?!(?:st|nd|rd|th)\b)(?=[A-Za-z])', 'I', t, flags=re.I)
                                        lw.append(normalize_manga_punctuation(t) if is_manga_punctuation(t) else t)
                                        ys_r = [pt[1] / 2.0 for pt in r[0]]
                                        xs_r = [pt[0] / 2.0 for pt in r[0]]
                                        min_crop_y = min(min_crop_y, y1 + min(ys_r))
                                        max_crop_y = max(max_crop_y, y1 + max(ys_r))
                                        min_crop_x = min(min_crop_x, x1 + min(xs_r))
                                        max_crop_x = max(max_crop_x, x1 + max(xs_r))
                                    if lw:
                                        line_words_list.append(lw)

                                final_words = []
                                for l_idx, lw in enumerate(line_words_list):
                                    if l_idx > 0 and final_words and re.search(r'[A-Za-z]+-$', final_words[-1]) and re.match(r'^[A-Za-z\']+', lw[0]):
                                        combined = re.sub(r'-$', '', final_words[-1]) + lw[0]
                                        final_words[-1] = combined
                                        final_words.extend(lw[1:])
                                    else:
                                        final_words.extend(lw)

                                crop_txt = " ".join(final_words)
                                crop_txt = re.sub(r'(\b[A-Za-z]{2,})-\s+([A-Za-z\']{2,})', r'\1\2', crop_txt)
                                if any(c.isascii() and c.isalpha() for c in crop_txt):
                                    try:
                                        crop_txt = repair_comic_text(crop_txt)
                                    except Exception as rec_exc:
                                        logger.warning("repair_comic_text error: %s", rec_exc)
                                crop_txt = filter_manga_ocr_hallucinations(crop_txt)

                                crop_conf = float(np.mean([r[2] for r in valid_items]))
                                orig_conf = float(det.get("avg_conf", det.get("conf", 0.5 if transcription else 0.0)))

                                should_overwrite = False
                                if not transcription:
                                    should_overwrite = True
                                elif crop_conf >= orig_conf - 0.05:
                                    if len(crop_txt) > len(transcription) or crop_conf > orig_conf + 0.10 or (crop_txt.startswith("I ") and not transcription.startswith("I ")):
                                        should_overwrite = True
                                elif (crop_txt.startswith("I ") and not transcription.startswith("I ")) and crop_conf >= 0.50:
                                    should_overwrite = True

                                if should_overwrite:
                                    transcription = normalize_manga_punctuation(crop_txt) if is_manga_punctuation(crop_txt) else crop_txt
                                    if is_manga_punctuation(crop_txt):
                                        ocr_trusted = (crop_conf >= 0.20) and not is_ocr_garbage_noise(crop_txt, source_lang, is_inside_bubble=True)
                                    else:
                                        ocr_trusted = (crop_conf >= 0.50) and not is_ocr_garbage_noise(crop_txt, source_lang, is_inside_bubble=True)
                                    det["conf"] = crop_conf
                                    det["avg_conf"] = crop_conf
                                    tbox = det.get("text_box", [x1, y1, x2, y2])
                                    tbox[0] = max(x1, min(tbox[0], int(min_crop_x - 3)))
                                    tbox[1] = max(y1, min(tbox[1], int(min_crop_y - 3)))
                                    tbox[2] = min(x2, max(tbox[2], int(max_crop_x + 3)))
                                    tbox[3] = min(y2, max(tbox[3], int(max_crop_y + 3)))
                                    det["text_box"] = tbox
                    except Exception as exc:
                        logger.warning("Crop 2x OCR refinement error: %s", exc)

                if not transcription and x2 > x1 and y2 > y1:
                    crop = orig_img.crop((x1, y1, x2, y2))
                    crop_np = np.array(crop)
                    # 1. Try EasyOCR with CLAHE contrast enhancement for low-contrast/screentoned bubbles
                    if reader is not None and crop_np.size > 0 and np.std(crop_np) > 8:
                        try:
                            crop_gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                            enhanced_gray = clahe.apply(crop_gray)
                            enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
                            res = reader.readtext(enhanced_rgb, mag_ratio=1.5)
                            if not res:
                                res = reader.readtext(crop_np)
                            if not res and (cat == "narration" or np.median(crop_gray) < 110):
                                inv_rgb = apply_dark_box_clahe_inversion(crop_gray)
                                if isinstance(inv_rgb, Image.Image):
                                    inv_rgb = np.array(inv_rgb)
                                res = reader.readtext(inv_rgb, mag_ratio=1.5)
                                if res:
                                    det["category"] = "narration"
                                    cat = "narration"
                            if res:
                                valid_items = [r for r in res if r[2] >= 0.15 and r[1].strip()]
                                if valid_items:
                                    transcription = " ".join(r[1].strip() for r in valid_items)
                                    best_conf = max(r[2] for r in valid_items)
                                    if is_manga_punctuation(transcription):
                                        transcription = normalize_manga_punctuation(transcription)
                                        ocr_trusted = (best_conf >= 0.20) and not is_ocr_garbage_noise(transcription, source_lang, is_inside_bubble=True)
                                    else:
                                        ocr_trusted = bool(transcription) and (best_conf >= 0.18 or not is_ocr_garbage_noise(transcription, source_lang, is_inside_bubble=True))
                                    if any(c.isascii() and c.isalpha() for c in transcription):
                                        try:
                                            transcription = repair_comic_text(transcription)
                                        except Exception as rec_exc:
                                            logger.warning("repair_comic_text error: %s", rec_exc)
                        except Exception as e:
                            logger.warning("EasyOCR crop recognition error: %s", e)

                    # 2. If still empty, use custom MangaOCRModel sequence prediction (CRNN + CTC)
                    if not transcription and hasattr(self, "ocr") and self.ocr is not None:
                        try:
                            cw, ch = crop.size
                            target_h, target_w = 64, 256
                            scale = min(target_w / max(1, cw), target_h / max(1, ch))
                            nw = max(1, int(cw * scale))
                            nh = max(1, int(ch * scale))
                            resample_method = getattr(Image, "Resampling", Image).BILINEAR
                            resized_crop = crop.resize((nw, nh), resample_method)
                            padded_crop = Image.new("RGB", (target_w, target_h), (255, 255, 255))
                            padded_crop.paste(resized_crop, (0, 0))
                            seq_tensor = T.Compose([
                                T.ToTensor(),
                                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                            ])(padded_crop).unsqueeze(0).to(DEVICE)
                            with torch.no_grad():
                                log_probs = self.ocr(seq_tensor, return_sequence=True)
                                pred_text = decode_ctc(log_probs[:, 0, :], self.idx_to_char).strip()
                                if pred_text:
                                    transcription = pred_text
                                    ocr_trusted = not is_ocr_garbage_noise(transcription, source_lang, is_inside_bubble=(cat == "speech_bubble"))
                        except Exception as e:
                            logger.warning("Sequence OCR fallback error: %s", e)

                    # 3. If still empty, fallback to character-level slicing
                    if not transcription:
                        try:
                            gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
                            _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                            kernel = np.ones((2, 2), np.uint8)
                            bin_clean = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kernel)
                            contours, _ = cv2.findContours(bin_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            c_boxes = []
                            ch_h, ch_w = gray.shape
                            min_area = max(12, int((ch_h * ch_w) * 0.0008))
                            for cnt in contours:
                                bx_c, by_c, bw_c, bh_c = cv2.boundingRect(cnt)
                                if bw_c >= 3 and bh_c >= 5 and (bw_c * bh_c) >= min_area:
                                    if bw_c <= ch_w * 0.95 or bh_c <= ch_h * 0.95:
                                        c_boxes.append((bx_c, by_c, bw_c, bh_c))

                            if c_boxes:
                                line_thresh = max(14, int(ch_h * 0.35))
                                c_boxes = sorted(c_boxes, key=lambda b: (b[1] // line_thresh, b[0]))

                                char_transform = T.Compose([
                                    T.Resize((64, 64)),
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                ])

                                predicted_chars = []
                                with torch.no_grad():
                                    for bx_c, by_c, bw_c, bh_c in c_boxes:
                                        pad = 3
                                        px1, py1 = max(0, bx_c - pad), max(0, by_c - pad)
                                        px2, py2 = min(ch_w, bx_c + bw_c + pad), min(ch_h, by_c + bh_c + pad)
                                        char_crop = crop.crop((px1, py1, px2, py2))

                                        square_crop = Image.new("RGB", (64, 64), (255, 255, 255))
                                        cw_c, ch_c = char_crop.size
                                        scale_c = min(48.0 / max(cw_c, 1), 48.0 / max(ch_c, 1))
                                        nw_c, nh_c = max(4, int(cw_c * scale_c)), max(4, int(ch_c * scale_c))
                                        char_resized = char_crop.resize((nw_c, nh_c), Image.Resampling.BILINEAR)
                                        square_crop.paste(char_resized, ((64 - nw_c) // 2, (64 - nh_c) // 2))

                                        t_in = char_transform(square_crop).unsqueeze(0).to(DEVICE)
                                        logits = self.ocr(t_in, return_sequence=False)
                                        pred_id = int(torch.argmax(logits, dim=-1).item())
                                        char_str = self.idx_to_char.get(pred_id, "")
                                        if pred_id > 1 and char_str and char_str not in {"<blank>", "<unk>"}:
                                            predicted_chars.append(char_str)
                                transcription = "".join(predicted_chars)
                                if transcription:
                                    ocr_trusted = not is_ocr_garbage_noise(transcription, source_lang, is_inside_bubble=(cat == "speech_bubble"))
                        except Exception as e:
                            logger.warning("Character-level OCR fallback error: %s", e)

                if is_manga_punctuation(transcription):
                    transcription = normalize_manga_punctuation(transcription)
                    conf_val = float(det.get("avg_conf", det.get("conf", 1.0)))
                    ocr_trusted = (conf_val >= 0.20) and not is_ocr_garbage_noise(transcription, source_lang, is_inside_bubble=(cat == "speech_bubble"))
                is_sfx_item = (cat == "sfx") or is_sfx_text(transcription)
                effective_cat = "sfx" if is_sfx_item else cat
                stroke_box = det.get("text_box") or det.get("bbox") or det.get("box") or [x1, y1, x2, y2]
                dialogue_records.append({
                    "bubble_id": idx + 1,
                    "block_id": det.get("block_id", f"b_{idx + 1}"),
                    "category": effective_cat,
                    "bbox": [x1, y1, x2, y2],
                    "text_box": stroke_box,
                    "transcription": transcription,
                    "ocr_trusted": ocr_trusted,
                    "is_sfx": is_sfx_item,
                })

                # Draw on annotated image (Red box and #ID badge)
                color = (255, 0, 0)
                draw_annotated.rectangle([x1, y1, x2, y2], outline=color, width=4)
                bid_text = f"#{idx+1}"
                badge_w = max(42, int(len(bid_text) * 14 + 10))
                draw_annotated.rectangle([x1, max(0, y1 - 24), min(w, x1 + badge_w), max(0, y1)], fill=(255, 255, 255))
                draw_annotated.text((x1 + 4, max(0, y1 - 23)), bid_text, fill=color, font=font_badge)

                # Generate stroke-level inpainting mask (only masking text characters to preserve faces and line art)
                if transcription and ocr_trusted and not is_sfx_item and effective_cat != "sfx":
                    s_mask = generate_ink_stroke_mask(gray_orig, stroke_box, pad=3)
                    mask_np = np.maximum(mask_np, s_mask)

        mask_img = Image.fromarray(mask_np)

        # 3. Patch-Based Deep Neural Inpainting (MangaInpaintingModel U-Net & MangaTextCleaner)
        if progress_callback:
            progress_callback(0.85, "Fon tozalanmoqda (Inpainting)...")

        # Structural line and panel border protection (morphological detection: straight panel lines >= 40px)
        dark_orig = (gray_orig < 75).astype(np.uint8) * 255
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        h_borders = cv2.morphologyEx(dark_orig, cv2.MORPH_OPEN, k_h)
        v_borders = cv2.morphologyEx(dark_orig, cv2.MORPH_OPEN, k_v)
        structural_borders = cv2.bitwise_or(h_borders, v_borders)

        # Run state-of-the-art MangaTextCleaner if available
        neural_cleaned = None
        neural_mask = None
        if hasattr(self, "cleaner") and self.cleaner is not None:
            try:
                # Pass only confirmed dialogue_records boxes as allowed_boxes
                allowed_boxes = [
                    d["bbox"] for d in dialogue_records
                    if d.get("ocr_trusted", False) and d.get("transcription") and not d.get("is_sfx", False) and d.get("category") != "sfx"
                ]
                if not allowed_boxes:
                    allowed_boxes = [d.get("bbox") or d.get("box") for d in dialogue_records if (d.get("bbox") or d.get("box"))]

                with self.model_lock:
                    neural_cleaned, neural_mask, _ = self.cleaner.clean_page(
                        orig_img,
                        threshold=0.5,
                        padding_iter=2,
                        close_gaps_kernel=3,
                        bubble_adaptive=True,
                        allowed_boxes=allowed_boxes if allowed_boxes else None,
                    )
                if neural_mask is not None and any(d.get("ocr_trusted", False) for d in dialogue_records):
                    predicted = np.asarray(neural_mask)
                    for record in dialogue_records:
                        if record["ocr_trusted"] and record["transcription"] and not record.get("is_sfx", False) and record.get("category") != "sfx":
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
            cleaned_bubble_ids = [det["bubble_id"] for det in dialogue_records if "bubble_id" in det]
        else:
            if neural_cleaned is not None:
                cleaned_np = np.array(neural_cleaned.convert("RGB")).copy()
            else:
                cleaned_np = orig_np_arr.copy()
            neural_np = np.array(neural_cleaned) if neural_cleaned is not None else None

            for det in dialogue_records:
                if not det.get("transcription") or not det.get("ocr_trusted", True):
                    continue
                if det.get("is_sfx") or det.get("category") == "sfx":
                    continue

                bx1, by1, bx2, by2 = det.get("bbox") or det.get("box") or [0, 0, 0, 0]
                bid = det.get("bubble_id")

                bx1, by1 = max(0, min(w, bx1)), max(0, min(h, by1))
                bx2, by2 = max(0, min(w, bx2)), max(0, min(h, by2))
                if bx2 <= bx1 + 6 or by2 <= by1 + 6:
                    continue

                patch_orig = orig_np_arr[by1:by2, bx1:bx2].copy()
                patch_gray = gray_orig[by1:by2, bx1:bx2].copy()
                neural_patch = neural_np[by1:by2, bx1:bx2] if neural_np is not None else None

                with self.model_lock:
                    patch_clean = clean_bubble_crop(
                        patch_orig,
                        patch_gray=patch_gray,
                        patch_mask=mask_np[by1:by2, bx1:bx2],
                        neural_patch=neural_patch,
                        inpainter_model=getattr(self, "inpainter", None),
                        inpaint_transform=getattr(self, "inpaint_transform", None),
                        device=DEVICE,
                    )

                ph, pw = patch_clean.shape[:2]
                diff = np.abs(patch_clean.astype(np.float32) - patch_orig.astype(np.float32)).max(axis=-1)
                modified = diff > 2.0

                if np.any(modified):
                    target_region = cleaned_np[by1:by2, bx1:bx2].astype(np.float32)
                    cleaned_patch = cleaned_np[by1:by2, bx1:bx2].copy()

                    patch_gray_crop = cv2.cvtColor(patch_orig, cv2.COLOR_RGB2GRAY) if patch_orig.ndim == 3 else patch_orig
                    is_white_or_rect = (
                        (np.median(patch_gray_crop) >= 185)
                        or (np.percentile(patch_gray_crop, 85) >= 205)
                        or (np.mean(patch_clean >= 250) >= 0.50)
                    )

                    edge_alpha = np.ones((ph, pw), dtype=np.float32)
                    if is_white_or_rect:
                        feather_px = min(2, min(ph, pw) // 6)
                    else:
                        feather_px = min(6, min(ph, pw) // 4)

                    if feather_px > 0:
                        for f_i in range(feather_px):
                            alpha_v = (f_i + 1) / float(feather_px + 1)
                            edge_alpha[f_i, :] = np.minimum(edge_alpha[f_i, :], alpha_v)
                            edge_alpha[ph - 1 - f_i, :] = np.minimum(edge_alpha[ph - 1 - f_i, :], alpha_v)
                            edge_alpha[:, f_i] = np.minimum(edge_alpha[:, f_i], alpha_v)
                            edge_alpha[:, pw - 1 - f_i] = np.minimum(edge_alpha[:, pw - 1 - f_i], alpha_v)

                    if is_white_or_rect:
                        ksize = 3
                        edge_alpha = cv2.GaussianBlur(edge_alpha, (ksize, ksize), 0.5)
                    else:
                        ksize = 5 if min(ph, pw) >= 12 else 3
                        edge_alpha = cv2.GaussianBlur(edge_alpha, (ksize, ksize), 1.2)
                    edge_alpha = np.clip(edge_alpha, 0.0, 1.0)[..., np.newaxis]

                    blended = target_region * (1.0 - edge_alpha) + patch_clean.astype(np.float32) * edge_alpha
                    cleaned_patch[modified] = np.clip(np.round(blended[modified]), 0, 255).astype(np.uint8)
                    cleaned_np[by1:by2, bx1:bx2] = cleaned_patch

                cleaned_bubble_ids.append(det["bubble_id"])

            # 100% Guaranteed Structural Line & Panel Border Restoration strictly OUTSIDE round speech bubbles
            dialogue_bubbles_mask = np.zeros((h, w), dtype=bool)
            for det in dialogue_records:
                if det.get("category", "speech_bubble") in {"speech_bubble", "bubble"}:
                    bx1, by1, bx2, by2 = det.get("bbox") or det.get("box") or [0, 0, 0, 0]
                    bx1, by1 = max(0, min(w, bx1)), max(0, min(h, by1))
                    bx2, by2 = max(0, min(w, bx2)), max(0, min(h, by2))
                    if bx2 > bx1 and by2 > by1:
                        dialogue_bubbles_mask[by1:by2, bx1:bx2] = True

            cleaned_np[(structural_borders > 0) & (~dialogue_bubbles_mask)] = orig_np_arr[(structural_borders > 0) & (~dialogue_bubbles_mask)]
        cleaned_page = Image.fromarray(cleaned_np)
        if neural_mask is not None and any(d["ocr_trusted"] for d in dialogue_records):
            mask_img = Image.fromarray(mask_np)

        learning_info = None

        if progress_callback:
            progress_callback(1.0, "Jarayon muvaffaqiyatli yakunlandi!")

        return {
            "original_image": orig_img,
            "annotated_image": annotated_img,
            "mask_image": mask_img,
            "cleaned_image": cleaned_page,
            "dialogues": dialogue_records,
            "learning_info": learning_info,
            "cleaned_bubble_ids": cleaned_bubble_ids,
            "detected_source_lang": source_lang,
            "requires_ocr_fallback": (not any(d.get("ocr_trusted", False) for d in dialogue_records)) or (source_lang == "auto"),
        }
