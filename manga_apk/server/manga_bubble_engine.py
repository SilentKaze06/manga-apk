"""
Manga Speech Bubble & Dialogue Extraction Engine - Production Grade.
High-precision OCR, dialogue linguistic validation, bubble boundary expansion,
and 100% full text erasure overlay for seamless manga translation.
"""

import math
import logging
import re
import cv2
import numpy as np
try:
    import pytesseract
except ImportError:
    pytesseract = None
import wordninja
from PIL import Image
from typing import List, Dict, Optional

# Lazy-imported inside extract_manga_dialogues to avoid circular top-level imports with main.py
run_paddle_ocr = None
extract_paddle_lines = None

logger = logging.getLogger("manga_bubble_engine")

METADATA_REGEX = re.compile(
    r"(?:(?:chapter|>ter|chap|ch\.?)\s*\d+|translation by\b|lettering\b|scanlation\b|redrawer\b|cleaner\b|raw provider\b)",
    re.IGNORECASE
)

# Strong dialogue keywords of length >= 3 that characterize genuine manga speech
STRONG_DIALOGUE_KEYWORDS = {
    'you', "you're", 'youre', 'your', 'she', "she's", 'her', 'they', "they're", 
    'this', 'that', 'these', 'those', 'the', 'and', 'but', 'for', 'with', 'from', 
    'into', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'does', 'did', 
    "don't", "dont", "doesn't", "doesnt", "didn't", "didnt", 'can', "can't", "cant", 
    'could', "couldn't", "couldnt", 'will', "won't", "wont", 'would', "wouldn't", 
    'should', 'must', 'what', 'why', 'who', 'how', 'when', 'where', 'which', 'all', 
    'some', 'not', 'just', 'come', 'take', 'look', 'see', 'get', 'know', 'tell', 
    'want', 'think', 'said', 'here', 'there', 'out', 'down', 'like', 'well', 'right', 
    'please', 'thanks', 'okay', 'today', 'time', 'one', 'good', 'cuz', 'wanna', 
    'never', 'family', 'happy', 'care', 'took', 'remove', 'myself', 'listen', 
    'forget', 'parents', 'told', 'home', 'school', 'feeling', 'class', 'name',
    'turn', 'duty', 'need', 'order', 'expected', 'room', 'left', 'already', 'power'
}

# Single-word exclamations commonly found in manga speech bubbles
VALID_SINGLE_WORDS = {
    'oh', 'ok', 'okay', 'no', 'yes', 'bye', 'why', 'what', 'who', 'how', 
    'wait', 'stop', 'hey', 'hi', 'please', 'thanks', 'sure', 'yeah', 'nope', 
    'well', 'right', 'look', 'see', 'listen', 'help', 'ouch', 'ah', 'aha', 
    'uh', 'um', 'huh', 'wow', 'sorry', 'good'
}

VALID_SHORT_TOKENS = {
    'a', 'i', 'in', 'on', 'to', 'at', 'by', 'of', 'as', 'is', 'it', 'he', 'we', 'my', 'me', 'no', 'so', 'go', 'do', 'up', 'am', 'if', 'ok', 'an', 'be', 'or', 'us', 'oh', 'hi', 'ha',
    # Uzbek short words
    'bu', 'va', 'u', 'shu', 'ha', 'yo',
    # Russian short words
    'я', 'и', 'а', 'о', 'у', 'э', 'в', 'с', 'к', 'он', 'ты', 'мы', 'вы', 'да', 'не', 'но', 'ну', 'же', 'ли', 'бы',
}

def _calc_text_noise_ratio(text: str, source_lang: str = "auto") -> float:
    """Calculates the ratio of noise tokens (script-aware for CJK/KO/RU/UZ/EN)."""
    cleaned = clean_manga_text(text)
    if any('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' or '\uac00' <= c <= '\ud7af' for c in cleaned):
        letters = sum(1 for c in cleaned if c.isalpha())
        non_ws = sum(1 for c in cleaned if not c.isspace())
        if non_ws == 0:
            return 1.0
        return max(0.0, 1.0 - (letters / float(non_ws)))
    words = [re.sub(r'[^\w\']', '', w.lower(), flags=re.UNICODE) for w in cleaned.split()]
    words = [w for w in words if w]
    if not words:
        return 1.0
    noise_count = sum(
        1 for w in words
        if (len(w) <= 2 and w not in VALID_SHORT_TOKENS and not any('\u0400' <= c <= '\u04ff' for c in w))
        or any(c.isdigit() for c in w)
        or not any(c.isalpha() for c in w)
    )
    return float(noise_count) / float(len(words))

def clean_manga_text(text: str) -> str:
    """Cleans OCR noise symbols from manga dialogue while preserving multilingual & CJK punctuation."""
    t = re.sub(r'[^\w\s\'\.,!\?\"\-—…~～«»“”„’ʻʼ`「」『』、。・！？【】《》（）：；〜ー]', ' ', text)
    t = t.replace('_', ' ')
    return " ".join(t.split()).strip()

try:
    from ocr_engine import (
        is_manga_punctuation,
        apply_dark_box_clahe_inversion,
        is_dark_background,
        filter_manga_ocr_hallucinations,
    )
except ImportError:
    try:
        from ml.ocr_engine import (
            is_manga_punctuation,
            apply_dark_box_clahe_inversion,
            is_dark_background,
            filter_manga_ocr_hallucinations,
        )
    except ImportError:
        from manga_pipeline import (
            is_manga_punctuation,
            apply_dark_box_clahe_inversion,
            is_dark_background,
            filter_manga_ocr_hallucinations,
        )

def is_valid_manga_dialogue(text: str, source_lang: str = "en") -> bool:
    """
    Strictly validates that detected text represents genuine human dialogue,
    eliminating sound effects (SFX), line art, screentone moire, and clothing textures.
    Supports English, Japanese, Korean, Chinese, Russian, Uzbek, and European comics.
    """
    cleaned = clean_manga_text(text)
    if not cleaned:
        return False
    if is_manga_punctuation(cleaned):
        return True
    if cleaned.lower() in ('i', 'я', 'и'):
        return True

    lang = (source_lang or "auto").lower()

    # 1. Japanese (Kana or Kanji)
    if lang in ("ja", "jp", "jpn", "japanese") or any('\u3040' <= c <= '\u30ff' for c in text):
        jp_chars = sum(1 for c in text if '\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff')
        return jp_chars >= 1

    # 2. Korean (Hangul)
    if lang in ("ko", "kor", "korean") or any('\uac00' <= c <= '\ud7af' or '\u1100' <= c <= '\u11ff' for c in text):
        ko_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7af' or '\u1100' <= c <= '\u11ff' or '\u3130' <= c <= '\u318f')
        return ko_chars >= 1

    # 3. Chinese (Hanzi without Kana)
    if lang in ("zh", "ch", "chs", "cht", "zh-cn", "zh-tw", "chinese") or (any('\u4e00' <= c <= '\u9fff' for c in text) and not any('\u3040' <= c <= '\u30ff' for c in text)):
        zh_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf')
        return zh_chars >= 1

    # 4. Russian / Cyrillic (allow 1-char Russian words like "Я", "И" and >= 2 letters otherwise)
    if lang in ("ru", "rus", "russian", "cyrillic") or any('\u0400' <= c <= '\u04ff' for c in text):
        ru_letters = sum(1 for c in text if '\u0400' <= c <= '\u04ff' and c.isalpha())
        if cleaned.lower() in VALID_SHORT_TOKENS and ru_letters >= 1:
            return True
        return ru_letters >= 2

    if len(cleaned) < 2:
        return False

    # 5. Latin-based dialogue (English, Uzbek, French, German, Spanish, Turkish, etc.)
    words = [re.sub(r'[^\w\']', '', w.lower(), flags=re.UNICODE) for w in cleaned.split()]
    words = [w for w in words if w]
    if not words:
        return False

    # Single-word dialogue (allow 2-char EN/UZ words and common consonant exclamations)
    if len(words) == 1:
        w = words[0]
        if w in VALID_SINGLE_WORDS or w in VALID_SHORT_TOKENS or w in {"hmm", "shh", "psst", "tss", "tsk", "mph", "ugh"}:
            return True
        if len(w) >= 2 and any(c in 'aeiouyöüəıİöğşçаеёиоуыэюя' for c in w):
            return True
        return False

    # Check unknown short tokens and alphanumeric/digit noise BEFORE has_strong
    # Noise ratio >= 0.35 must reject (e.g. "YO MY LS FAMILY F 10 Z 4 HAPPY")
    noise_tokens = [
        w for w in words
        if (len(w) <= 2 and w not in VALID_SHORT_TOKENS)
        or any(c.isdigit() for c in w)
        or not any(c.isalpha() for c in w)
    ]
    if len(words) >= 3 and (len(noise_tokens) / len(words)) >= 0.35:
        return False

    # Check strong keywords if English
    if lang in ("en", "eng", "english", "auto"):
        has_strong = any(w in STRONG_DIALOGUE_KEYWORDS for w in words)
        if not has_strong:
            try:
                split_w = wordninja.split(cleaned.lower())
                has_strong = any(w in STRONG_DIALOGUE_KEYWORDS for w in split_w)
            except Exception as exc:
                logger.debug("wordninja split error in is_valid_manga_dialogue: %s", exc)
        if has_strong:
            return True
        return sum(1 for c in cleaned if c.isalpha()) >= 3

    # Uzbek & other Latin languages
    return sum(1 for c in cleaned if c.isalpha()) >= 2



# Geometry functions imported from ml.geometry
try:
    from ml.geometry import (
        _get_box_border_mask,
        _line_crosses_border_mask,
        _has_continuous_dark_segment,
        _is_dark_barrier_between_x,
        _is_dark_barrier_between_y,
        get_white_cc_labels,
        _get_box_white_mode_label,
        are_different_white_contours,
        get_box_border_mask,
        line_crosses_border_mask,
        is_dark_barrier_between_x,
        is_dark_barrier_between_y,
    )
except ImportError:
    try:
        from geometry import (
            _get_box_border_mask,
            _line_crosses_border_mask,
            _has_continuous_dark_segment,
            _is_dark_barrier_between_x,
            _is_dark_barrier_between_y,
            get_white_cc_labels,
            _get_box_white_mode_label,
            are_different_white_contours,
            get_box_border_mask,
            line_crosses_border_mask,
            is_dark_barrier_between_x,
            is_dark_barrier_between_y,
        )
    except ImportError:
        from manga_pipeline import (
            _get_box_border_mask,
            _line_crosses_border_mask,
            _has_continuous_dark_segment,
            _is_dark_barrier_between_x,
            _is_dark_barrier_between_y,
            get_white_cc_labels,
            _get_box_white_mode_label,
            are_different_white_contours,
            get_box_border_mask,
            line_crosses_border_mask,
            is_dark_barrier_between_x,
            is_dark_barrier_between_y,
        )


def _get_box_white_label(labels: np.ndarray, gray: np.ndarray, stats: Optional[np.ndarray], box: list, pad: int = 3) -> int:
    """Returns mode white label using geometry engine."""
    return _get_box_white_mode_label(labels, stats, gray, box, min_area=200, pad=pad)


def _can_merge_text_boxes(
    boxA: list,
    boxB: list,
    gray: np.ndarray,
    max_page_h: int = 2000,
    labels: Optional[np.ndarray] = None,
    stats: Optional[np.ndarray] = None,
    border_mask: Optional[np.ndarray] = None
) -> bool:
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB
    wA, hA = max(1, xA2 - xA1), max(1, yA2 - yA1)
    wB, hB = max(1, xB2 - xB1), max(1, yB2 - yB1)
    
    total_y_span = max(yA2, yB2) - min(yA1, yB1)
    if total_y_span > max_page_h * 0.18:
        return False

    comb_crop = gray[min(yA1, yB1):max(yA2, yB2), min(xA1, xB1):max(xA2, xB2)] if gray is not None and gray.size > 0 else np.array([])
    is_in_dark_box = False
    if comb_crop.size > 0:
        med_comb = float(np.median(comb_crop))
        if med_comb < 95 or (med_comb < 120 and float(np.mean(comb_crop < 110)) >= 0.45):
            is_in_dark_box = True

    if gray is not None and gray.size > 0 and not is_in_dark_box:
        if border_mask is None:
            border_mask = _get_box_border_mask(gray)
        if border_mask is not None and border_mask.size > 0:
            cxA = int((xA1 + xA2) / 2.0)
            cyA = int((yA1 + yA2) / 2.0)
            cxB = int((xB1 + xB2) / 2.0)
            cyB = int((yB1 + yB2) / 2.0)
            if _line_crosses_border_mask(border_mask, (cxA, cyA), (cxB, cyB), boxA=boxA, boxB=boxB):
                return False

    # White background (gray > 200) cv2.connectedComponents check:
    # If two text boxes belong to different closed white contours (separate bubble/panel outlines),
    # they must NEVER be merged.
    if not is_in_dark_box and gray is not None and gray.size > 0:
        if labels is None or stats is None:
            labels, stats = get_white_cc_labels(gray, return_stats=True)
        if labels is not None:
            lblA = _get_box_white_label(labels, gray, stats, boxA)
            lblB = _get_box_white_label(labels, gray, stats, boxB)
            if lblA > 0 and lblB > 0 and lblA != lblB:
                return False

    # Check dark barrier in X for side-by-side / diagonal / staggered boxes
    left_box = boxA if xA1 < xB1 else boxB
    right_box = boxB if xA1 < xB1 else boxA
    if _is_dark_barrier_between_x(gray, left_box, right_box):
        return False

    # 1. Words on same horizontal line
    y_overlap = min(yA2, yB2) - max(yA1, yB1)
    min_h = min(hA, hB)
    if min_h > 0 and (y_overlap / min_h) >= 0.40:
        gap_x = max(0, max(xA1, xB1) - min(xA2, xB2))
        if gap_x <= max(24, int(min_h * 1.8)):
            return True

    # 2. Consecutive lines stacked vertically in dialogue
    x_overlap = min(xA2, xB2) - max(xA1, xB1)
    min_w = min(wA, wB)
    midA = (xA1 + xA2) / 2.0
    midB = (xB1 + xB2) / 2.0
    is_aligned = (min_w > 0 and (x_overlap / min_w) >= 0.20) or abs(midA - midB) <= max(wA, wB) * 0.75
    if is_aligned:
        gap_y = max(0, max(yA1, yB1) - min(yA2, yB2))
        avg_h = (hA + hB) / 2.0
        if gap_y <= max(24, int(avg_h * 2.2)):
            top_box = boxA if yA1 < yB1 else boxB
            bot_box = boxB if yA1 < yB1 else boxA
            if is_in_dark_box or not _is_dark_barrier_between_y(gray, top_box, bot_box):
                return True

    return False

def is_dark_narration_box(gray: np.ndarray, box: Optional[list] = None) -> bool:
    """
    Identifies dark narration frames in manga:
    - median background < 95
    - bright text contrast p95 > 180
    - rectangular frame/border
    Such regions are categorized as 'narration' and must never be dropped as SFX.
    """
    if gray is None or gray.size == 0:
        return False
    H, W = gray.shape[:2]
    if box is not None:
        bx1, by1, bx2, by2 = [int(round(v)) for v in box]
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(W, bx2), min(H, by2)
        if bx2 <= bx1 or by2 <= by1:
            return False
        px1, py1 = max(0, bx1 - 6), max(0, by1 - 6)
        px2, py2 = min(W, bx2 + 6), min(H, by2 + 6)
        patch = gray[py1:py2, px1:px2]
    else:
        patch = gray
        bx1, by1, bx2, by2 = 0, 0, W, H

    if patch.size == 0:
        return False

    bg_pixels = patch[patch < 180]
    bg_median = float(np.median(bg_pixels)) if bg_pixels.size > 0 else float(np.median(patch))
    p95 = float(np.percentile(patch, 95))
    p98 = float(np.percentile(patch, 98))
    has_bright_contrast = (p95 > 180) or (p98 > 180 and np.sum(patch > 180) >= 10)
    if not has_bright_contrast and box is not None:
        t_patch = gray[by1:by2, bx1:bx2]
        if t_patch.size > 0 and (float(np.percentile(t_patch, 95)) > 180 or (float(np.percentile(t_patch, 98)) > 180 and np.sum(t_patch > 180) >= 8)):
            has_bright_contrast = True

    # 1. median fon < 95
    # 2. yorqin matn kontrasti p95 > 180
    if bg_median >= 95 or not has_bright_contrast:
        return False

    # 3. to'rtburchak chegara (rectangular border) - cropped to padded ROI for performance
    if box is not None and (H > (by2 - by1) + 12 or W > (bx2 - bx1) + 12):
        bw_b, bh_b = max(1, bx2 - bx1), max(1, by2 - by1)
        pad_roi_x = max(80, bw_b * 2)
        pad_roi_y = max(80, bh_b * 2)
        rx1, ry1 = max(0, bx1 - pad_roi_x), max(0, by1 - pad_roi_y)
        rx2, ry2 = min(W, bx2 + pad_roi_x), min(H, by2 + pad_roi_y)
        dark_roi = (gray[ry1:ry2, rx1:rx2] < 130).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(dark_roi, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cx, cy = (bx1 + bx2) / 2.0 - rx1, (by1 + by2) / 2.0 - ry1
        lbx1, lby1, lbx2, lby2 = bx1 - rx1, by1 - ry1, bx2 - rx1, by2 - ry1

        for cnt in contours:
            if cv2.pointPolygonTest(cnt, (float(cx), float(cy)), False) >= 0:
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                if rx <= lbx1 + 6 and ry <= lby1 + 6 and rx + rw >= lbx2 - 6 and ry + rh >= lby2 - 6:
                    area = cv2.contourArea(cnt)
                    rect_ratio = area / float(rw * rh) if rw * rh > 0 else 0
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                    corners = [(rx + 4, ry + 4), (rx + rw - 5, ry + 4), (rx + 4, ry + rh - 5), (rx + rw - 5, ry + rh - 5)]
                    corners_in = sum(1 for pt in corners if cv2.pointPolygonTest(cnt, pt, False) >= 0)
                    if rect_ratio >= 0.75 and (len(approx) <= 8 or corners_in >= 3):
                        return True

    dark_ratio = float(np.mean(patch < 130))
    return dark_ratio >= 0.50

_is_dark_narration_box = is_dark_narration_box


def _expand_to_dark_narration_contour(
    gray: np.ndarray,
    text_box: list,
    all_other_tboxes: Optional[List[list]] = None,
    max_pad_x: int = 40,
    max_pad_y: int = 40
) -> list:
    """Snaps a dark narration box outward until hitting light panel/page borders."""
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = text_box

    min_pad = 4
    x1 = max(0, x1 - min_pad)
    y1 = max(0, y1 - min_pad)
    x2 = min(w, x2 + min_pad)
    y2 = min(h, y2 + min_pad)

    limit_left, limit_right = 0, w - 1
    limit_top, limit_bot = 0, h - 1

    if all_other_tboxes:
        for ob in all_other_tboxes:
            ox1, oy1, ox2, oy2 = ob
            if max(y1, oy1) < min(y2, oy2):
                if ox2 <= x1:
                    limit_left = max(limit_left, (ox2 + x1) // 2 + 1)
                elif ox1 >= x2:
                    limit_right = min(limit_right, (x2 + ox1) // 2 - 1)
            if max(x1, ox1) < min(x2, ox2):
                if oy2 <= y1:
                    limit_top = max(limit_top, (oy2 + y1) // 2 + 1)
                elif oy1 >= y2:
                    limit_bot = min(limit_bot, (y2 + oy1) // 2 - 1)

    cx1 = x1
    for d in range(1, max_pad_x):
        tx = x1 - d
        if tx < limit_left: break
        col = gray[y1:y2, tx]
        if np.mean(col > 170) >= 0.40 or np.mean(col) > 150:
            cx1 = tx
            break
        cx1 = tx

    cx2 = x2
    for d in range(1, max_pad_x):
        tx = x2 + d
        if tx > limit_right: break
        col = gray[y1:y2, tx]
        if np.mean(col > 170) >= 0.40 or np.mean(col) > 150:
            cx2 = tx
            break
        cx2 = tx

    cy1 = y1
    for d in range(1, max_pad_y):
        ty = y1 - d
        if ty < limit_top: break
        row = gray[ty, cx1:cx2]
        if np.mean(row > 170) >= 0.40 or np.mean(row) > 150:
            cy1 = ty
            break
        cy1 = ty

    cy2 = y2
    for d in range(1, max_pad_y):
        ty = y2 + d
        if ty > limit_bot: break
        row = gray[ty, cx1:cx2]
        if np.mean(row > 170) >= 0.40 or np.mean(row) > 150:
            cy2 = ty
            break
        cy2 = ty

    return [cx1, cy1, cx2, cy2]


# Public aliases
is_dark_barrier_between_x = _is_dark_barrier_between_x
is_dark_barrier_between_y = _is_dark_barrier_between_y
can_merge_text_boxes = _can_merge_text_boxes
get_box_border_mask = _get_box_border_mask
line_crosses_border_mask = _line_crosses_border_mask
is_dark_narration_box = _is_dark_narration_box

def _expand_to_bubble_contour(
    gray: np.ndarray,
    text_box: list,
    all_other_tboxes: Optional[List[list]] = None,
    max_pad_x: int = 40,
    max_pad_y: int = 40
) -> list:
    """Snaps a dialogue bounding box outward to the speech bubble contour."""
    h, w = gray.shape
    x1, y1, x2, y2 = text_box

    min_pad = 6
    x1 = max(0, x1 - min_pad)
    y1 = max(0, y1 - min_pad)
    x2 = min(w, x2 + min_pad)
    y2 = min(h, y2 + min_pad)

    limit_left, limit_right = 0, w - 1
    limit_top, limit_bot = 0, h - 1

    if all_other_tboxes:
        for ob in all_other_tboxes:
            ox1, oy1, ox2, oy2 = ob
            if max(y1, oy1) < min(y2, oy2):
                if ox2 <= x1:
                    limit_left = max(limit_left, (ox2 + x1) // 2 + 1)
                elif ox1 >= x2:
                    limit_right = min(limit_right, (x2 + ox1) // 2 - 1)
            if max(x1, ox1) < min(x2, ox2):
                if oy2 <= y1:
                    limit_top = max(limit_top, (oy2 + y1) // 2 + 1)
                elif oy1 >= y2:
                    limit_bot = min(limit_bot, (y2 + oy1) // 2 - 1)

    cx1 = x1
    for d in range(1, max_pad_x):
        tx = x1 - d
        if tx < limit_left: break
        col = gray[y1:y2, tx]
        if np.mean(col < 135) >= 0.40 or np.mean(col) < 170:
            cx1 = tx
            break
        cx1 = tx

    cx2 = x2
    for d in range(1, max_pad_x):
        tx = x2 + d
        if tx > limit_right: break
        col = gray[y1:y2, tx]
        if np.mean(col < 135) >= 0.40 or np.mean(col) < 170:
            cx2 = tx
            break
        cx2 = tx

    cy1 = y1
    for d in range(1, max_pad_y):
        ty = y1 - d
        if ty < limit_top: break
        row = gray[ty, cx1:cx2]
        if np.mean(row < 135) >= 0.40 or np.mean(row) < 170:
            cy1 = ty
            break
        cy1 = ty

    cy2 = y2
    for d in range(1, max_pad_y):
        ty = y2 + d
        if ty > limit_bot: break
        row = gray[ty, cx1:cx2]
        if np.mean(row < 135) >= 0.40 or np.mean(row) < 170:
            cy2 = ty
            break
        cy2 = ty

    return [cx1, cy1, cx2, cy2]

def _generate_overlay_polygon(
    bx1: float, by1: float, bx2: float, by2: float,
    tx1: float, ty1: float, tx2: float, ty2: float,
    is_rectangular: Optional[bool] = None
) -> List[List[float]]:
    """
    Generates a high-coverage polygon that covers 96% of the bubble interior,
    guaranteeing complete coverage of all text while respecting bubble contours.
    """
    bw = bx2 - bx1
    bh = by2 - by1
    
    # Check if rectangular box (e.g. narration box where text fills corners)
    if is_rectangular is None:
        is_rectangular = (bw > 0 and bh > 0 and (tx2 - tx1) / bw > 0.72 and (ty2 - ty1) / bh > 0.72)
    
    if is_rectangular:
        r = min(bw * 0.12, bh * 0.12, 10.0)
        return [
            [bx1 + r, by1],
            [bx2 - r, by1],
            [bx2, by1 + r],
            [bx2, by2 - r],
            [bx2 - r, by2],
            [bx1 + r, by2],
            [bx1, by2 - r],
            [bx1, by1 + r],
        ]
    else:
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        rx = (bw / 2.0) * 0.96
        ry = (bh / 2.0) * 0.96
        poly = []
        for p_idx in range(16):
            th = 2.0 * math.pi * p_idx / 16.0
            poly.append([cx + rx * math.cos(th), cy + ry * math.sin(th)])
        return poly

def _sort_dialogue_line_boxes(comp: List[int], raw_boxes: List[Dict]) -> List[int]:
    """
    Sorts word bounding boxes within a dialogue cluster using y_center sorting
    and vertical line grouping with y_overlap >= 0.45 * min(h1, h2)
    to prevent words on the same line with minor vertical jitter from swapping.
    """
    if len(comp) <= 1:
        return comp

    sorted_words = sorted(
        comp,
        key=lambda idx: (
            (raw_boxes[idx]["box"][1] + raw_boxes[idx]["box"][3]) / 2.0,
            raw_boxes[idx]["box"][0]
        )
    )

    lines: List[List[int]] = []
    line_bounds: List[List[float]] = []

    for idx in sorted_words:
        box = raw_boxes[idx]["box"]
        y1, y2 = float(box[1]), float(box[3])
        h2 = max(1.0, y2 - y1)
        matched_line_idx = -1

        for l_idx in reversed(range(len(lines))):
            ly1, ly2, lh = line_bounds[l_idx]
            y_overlap = min(ly2, y2) - max(ly1, y1)
            min_h = min(lh, h2)
            if min_h > 0 and (y_overlap >= 0.45 * min_h):
                matched_line_idx = l_idx
                break

        if matched_line_idx >= 0:
            lines[matched_line_idx].append(idx)
            ly1, ly2, _ = line_bounds[matched_line_idx]
            new_ly1 = min(ly1, y1)
            new_ly2 = max(ly2, y2)
            avg_h = sum(raw_boxes[i]["box"][3] - raw_boxes[i]["box"][1] for i in lines[matched_line_idx]) / float(len(lines[matched_line_idx]))
            line_bounds[matched_line_idx] = [new_ly1, new_ly2, avg_h]
        else:
            lines.append([idx])
            line_bounds.append([y1, y2, h2])

    for line in lines:
        line.sort(key=lambda idx: raw_boxes[idx]["box"][0])

    lines.sort(key=lambda line: sum((raw_boxes[idx]["box"][1] + raw_boxes[idx]["box"][3]) / 2.0 for idx in line) / float(len(line)))

    sorted_comp = []
    for line in lines:
        sorted_comp.extend(line)
    return sorted_comp

sort_dialogue_line_boxes = _sort_dialogue_line_boxes

def extract_manga_dialogues(
    pil_image: Image.Image,
    source_lang: str = "en",
    scale: float = 1.0,
    onnx_engine=None
) -> List[Dict]:
    """
    Extracts all manga speech bubbles and dialogue text from a page image.
    Returns blocks formatted for PyMuPDF PDF typesetting.
    """
    global run_paddle_ocr, extract_paddle_lines
    if run_paddle_ocr is None or extract_paddle_lines is None:
        try:
            from main import run_paddle_ocr as _rpo, extract_paddle_lines as _epl
            if run_paddle_ocr is None:
                run_paddle_ocr = _rpo
            if extract_paddle_lines is None:
                extract_paddle_lines = _epl
        except Exception:
            pass

    if pytesseract is None and run_paddle_ocr is None:
        return []

    img_np = np.asarray(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    raw_lang = (source_lang or "auto").lower().strip()
    norm_iso_map = {
        "ko": "ko", "kor": "ko", "korean": "ko",
        "zh": "zh", "ch": "zh", "chs": "zh", "cht": "zh", "zh-cn": "zh", "zh-tw": "zh", "chinese": "zh",
        "ru": "ru", "rus": "ru", "russian": "ru", "cyrillic": "ru",
        "ja": "ja", "jp": "ja", "jpn": "ja", "japanese": "ja",
        "en": "en", "eng": "en", "english": "en",
        "fr": "latin", "fra": "latin", "french": "latin",
        "de": "latin", "deu": "latin", "german": "latin",
        "es": "latin", "spa": "latin", "spanish": "latin",
        "pt": "latin", "por": "latin", "portuguese": "latin",
        "it": "latin", "ita": "latin", "italian": "latin",
        "tr": "latin", "tur": "latin", "turkish": "latin",
        "nl": "latin", "nld": "latin", "dutch": "latin",
        "pl": "latin", "pol": "latin", "polish": "latin",
        "uz": "latin", "uzb": "latin", "uzbek": "latin",
        "latin": "latin",
        "auto": "auto",
    }
    paddle_lang = norm_iso_map.get(raw_lang, "latin" if raw_lang != "auto" else "auto")

    tess_lang_map = {
        "ko": "kor",
        "zh": "chi_tra" if raw_lang in ("cht", "zh-tw") else "chi_sim",
        "ru": "rus",
        "ja": "jpn",
        "en": "eng",
        "latin": "eng",
        "auto": "eng",
    }
    tess_lang = tess_lang_map.get(paddle_lang, "eng")
    
    # 1. Full page sparse word detection
    data = {'text': [], 'conf': [], 'left': [], 'top': [], 'width': [], 'height': []}
    try:
        if pytesseract is not None:
            data = pytesseract.image_to_data(gray, config='--psm 11', lang=tess_lang, output_type=pytesseract.Output.DICT)
            # Inverted OCR pass (255 - gray) for dark narration frames / white text on dark background
            try:
                inv_data = pytesseract.image_to_data(255 - gray, config='--psm 11', lang=tess_lang, output_type=pytesseract.Output.DICT)
                if isinstance(inv_data, dict) and 'text' in inv_data and len(inv_data['text']) > 0:
                    if isinstance(data, dict):
                        data = {k: list(v) for k, v in data.items()}
                        for k in ('text', 'conf', 'left', 'top', 'width', 'height'):
                            if k in data and k in inv_data:
                                data[k].extend(list(inv_data[k]))
            except Exception as inv_exc:
                logger.debug("pytesseract inverted OCR pass xatosi: %s", inv_exc)
    except Exception as exc:
        logger.warning("pytesseract.image_to_data xatosi: %s", exc)
    
    raw_boxes = []
    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        try:
            conf = int(float(data['conf'][i]))
        except (ValueError, TypeError):
            continue
        if conf < 15 or len(word) < 1:
            continue
        clean_w = "".join(c for c in word if c.isalnum())
        if not clean_w and not is_manga_punctuation(word):
            continue
        bx1 = max(0, data['left'][i] - 3)
        by1 = max(0, data['top'][i] - 3)
        bx2 = min(w, data['left'][i] + data['width'][i] + 3)
        by2 = min(h, data['top'][i] + data['height'][i] + 3)

        # Deduplicate overlapping word detections at identical position
        is_dup = False
        for rb in raw_boxes:
            ob = rb['box']
            if abs(ob[0] - bx1) <= 6 and abs(ob[1] - by1) <= 6 and abs(ob[2] - bx2) <= 6 and abs(ob[3] - by2) <= 6:
                if conf > rb.get('conf', 0):
                    rb['text'] = word
                    rb['conf'] = conf
                is_dup = True
                break
        if is_dup:
            continue

        raw_boxes.append({
            'box': [bx1, by1, bx2, by2],
            'text': word,
            'conf': conf
        })

    # If no raw boxes (or low-conf/non-English under auto) and paddleocr is available, use paddleocr fallback
    if not raw_boxes and callable(run_paddle_ocr) and callable(extract_paddle_lines):
        try:
            import io
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            if paddle_lang == "auto":
                try:
                    from main import run_auto_paddle_ocr
                    detected_p_lang, paddle_res = run_auto_paddle_ocr(img_bytes)
                    paddle_lines = extract_paddle_lines(paddle_res, detected_p_lang)
                except Exception:
                    paddle_res = run_paddle_ocr(img_bytes, "en")
                    paddle_lines = extract_paddle_lines(paddle_res, "en")
            else:
                paddle_res = run_paddle_ocr(img_bytes, paddle_lang)
                paddle_lines = extract_paddle_lines(paddle_res, raw_lang if raw_lang != "auto" else paddle_lang)
            for line in paddle_lines:
                txt = line.get("text", "").strip()
                if not txt:
                    continue
                left = line.get("left", 0)
                top = line.get("top", 0)
                right = line.get("right", left + line.get("width", 0))
                bottom = line.get("bottom", top + line.get("height", 0))
                raw_boxes.append({
                    'box': [left, top, right, bottom],
                    'text': txt,
                    'conf': int(line.get("confidence", 0.95) * 100) if line.get("confidence", 0.95) <= 1.0 else int(line.get("confidence", 95))
                })
        except Exception as p_exc:
            logger.debug("PaddleOCR fallback failed in extract_manga_dialogues: %s", p_exc)

    # 2. Graph Clustering into Dialogue Units
    cc_labels, cc_stats = get_white_cc_labels(gray, return_stats=True)
    border_mask = _get_box_border_mask(gray)

    n = len(raw_boxes)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if _can_merge_text_boxes(raw_boxes[i]["box"], raw_boxes[j]["box"], gray, max_page_h=h, labels=cc_labels, stats=cc_stats, border_mask=border_mask):
                adj[i].append(j)
                adj[j].append(i)

    visited = set()
    clusters = []
    for i in range(n):
        if i in visited:
            continue
        comp = []
        queue = [i]
        visited.add(i)
        while queue:
            cur = queue.pop(0)
            comp.append(cur)
            for neighbor in adj[cur]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        comp = _sort_dialogue_line_boxes(comp, raw_boxes)
        bx1 = min(raw_boxes[idx]["box"][0] for idx in comp)
        by1 = min(raw_boxes[idx]["box"][1] for idx in comp)
        bx2 = max(raw_boxes[idx]["box"][2] for idx in comp)
        by2 = max(raw_boxes[idx]["box"][3] for idx in comp)
        
        full_text = " ".join(raw_boxes[idx]["text"] for idx in comp)
        avg_conf_c = sum(raw_boxes[idx].get("conf", 95) for idx in comp) / float(len(comp)) if comp else 95.0
        clusters.append({
            "block_id": f"b_{len(clusters) + 1}",
            "box": [bx1, by1, bx2, by2],
            "text": full_text,
            "count": len(comp),
            "confidence": min(1.0, max(0.0, avg_conf_c / 100.0)),
        })

    # 3. Filter Valid Dialogue & Reject Metadata / Hair / Screentone False Positives
    all_tboxes = []
    filtered = []
    for c in clusters:
        text = clean_manga_text(c["text"])
        bx1, by1, bx2, by2 = c["box"]

        # Max bubble dimension check (allow tall vertical/large dialogue bubbles)
        if (by2 - by1) > h * 0.65 or (bx2 - bx1) > w * 0.90:
            continue

        # Metadata check (Chapter title, credits, etc.)
        if METADATA_REGEX.search(text):
            continue

        # Strictly validate dialogue text
        if not is_valid_manga_dialogue(text, source_lang):
            continue

        is_narration = is_dark_narration_box(gray, [bx1, by1, bx2, by2])

        # Speech bubble background check (allow colored/pastel bubbles while rejecting uncontrasted dark art)
        px1, py1 = max(0, bx1 - 6), max(0, by1 - 6)
        px2, py2 = min(w, bx2 + 6), min(h, by2 + 6)
        patch = gray[py1:py2, px1:px2]
        mean_val = float(np.mean(patch)) if patch.size > 0 else 0.0
        p95_val = float(np.percentile(patch, 95)) if patch.size > 0 else 0.0

        # Reject dark uncontrasted backgrounds (clothes, dark hair), plus tiny 1-3 char artwork noise outside bright bubbles
        if not is_narration and mean_val < 75 and p95_val < 150:
            continue
        clean_len = len(re.sub(r'[^\w]', '', text))
        if not is_narration and clean_len <= 3 and mean_val < 175 and not is_manga_punctuation(text):
            continue

        filtered.append({
            "block_id": c.get("block_id", f"b_{len(filtered) + 1}"),
            "text_box": [bx1, by1, bx2, by2],
            "text": text,
            "category": "narration" if is_narration else "speech_bubble",
            "is_narration": is_narration,
            "confidence": c.get("confidence", 0.85),
        })
        all_tboxes.append([bx1, by1, bx2, by2])

    # 4. Bubble Contour Snapping & Typesetting Blocks Construction
    blocks = []
    for idx, f in enumerate(filtered):
        tbox = f["text_box"]
        is_narration = f.get("is_narration", False)
        other_tboxes = [b for j, b in enumerate(all_tboxes) if j != idx]

        if is_narration:
            bubble_box = _expand_to_dark_narration_contour(gray, tbox, other_tboxes)
        else:
            bubble_box = _expand_to_bubble_contour(gray, tbox, other_tboxes)
        
        bx1, by1, bx2, by2 = bubble_box

        # Re-OCR with tight tbox (+2-3 px padding) instead of spiky bubble_box
        # to prevent border spikes from causing OCR noise
        tx1, ty1, tx2, ty2 = tbox
        ocr_pad = 3
        ox1 = max(0, tx1 - ocr_pad)
        oy1 = max(0, ty1 - ocr_pad)
        ox2 = min(w, tx2 + ocr_pad)
        oy2 = min(h, ty2 + ocr_pad)

        if is_narration:
            # Inverted CLAHE OCR pass for dark narration frames
            crop_patch = gray[oy1:oy2, ox1:ox2]
            if crop_patch.size > 0:
                inv_crop = apply_dark_box_clahe_inversion(crop_patch)
                crop_pil = Image.fromarray(inv_crop)
            else:
                crop_pil = pil_image.crop((ox1, oy1, ox2, oy2))
        else:
            crop_pil = pil_image.crop((ox1, oy1, ox2, oy2))
        
        text = f["text"]
        try:
            if pytesseract is not None:
                crop_text = pytesseract.image_to_string(crop_pil, lang=tess_lang, config='--psm 6').strip()
                crop_cleaned = clean_manga_text(crop_text)
                if is_valid_manga_dialogue(crop_cleaned, source_lang):
                    old_noise = _calc_text_noise_ratio(text, source_lang)
                    new_noise = _calc_text_noise_ratio(crop_cleaned, source_lang)
                    if new_noise < old_noise or (new_noise == old_noise and len(crop_cleaned) > len(text)):
                        text = crop_cleaned
        except Exception as exc:
            logger.debug("pytesseract crop ocr failed: %s", exc)

        if not is_valid_manga_dialogue(text, source_lang):
            continue

        tx1, ty1, tx2, ty2 = tbox
        scaled_bx1, scaled_by1 = bx1 / scale, by1 / scale
        scaled_bx2, scaled_by2 = bx2 / scale, by2 / scale
        scaled_tx1, scaled_ty1 = tx1 / scale, ty1 / scale
        scaled_tx2, scaled_ty2 = tx2 / scale, ty2 / scale

        # High coverage polygon covering 96% of the bubble interior
        poly = _generate_overlay_polygon(
            scaled_bx1, scaled_by1, scaled_bx2, scaled_by2,
            scaled_tx1, scaled_ty1, scaled_tx2, scaled_ty2,
            is_rectangular=is_narration or None
        )

        # Expanded text erase rectangle (+3pt padding) ensuring 100% of original letters are wiped clean
        erase_pad = 3.0
        text_erase_rect = [
            max(0.0, scaled_tx1 - erase_pad),
            max(0.0, scaled_ty1 - erase_pad),
            min(w / scale, scaled_tx2 + erase_pad),
            min(h / scale, scaled_ty2 + erase_pad)
        ]

        # Determine background color from the bubble interior
        px1, py1 = max(0, bx1 - 6), max(0, by1 - 6)
        px2, py2 = min(w, bx2 + 6), min(h, by2 + 6)
        bg_patch = gray[py1:py2, px1:px2]
        if bg_patch.size > 0:
            if is_narration:
                bg_pixels = bg_patch[bg_patch < 180]
                bg_luma = float(np.median(bg_pixels) if bg_pixels.size > 0 else np.median(bg_patch)) / 255.0
            else:
                bg_luma = float(np.median(bg_patch)) / 255.0
            block_bg = [bg_luma, bg_luma, bg_luma]
        else:
            block_bg = [0.15, 0.15, 0.15] if is_narration else [1.0, 1.0, 1.0]

        blocks.append({
            "text": text,
            "font_size": 0.0,
            "confidence": float(f.get("confidence", 0.85)),
            "bbox": [scaled_bx1, scaled_by1, scaled_bx2, scaled_by2],
            "overlay_bbox": [scaled_bx1, scaled_by1, scaled_bx2, scaled_by2],
            "text_erase_bbox": text_erase_rect,
            "overlay_polygon": poly,
            "category": "narration" if is_narration else "speech_bubble",
            "bubble_id": len(blocks) + 1,
            "block_id": f.get("block_id", f"b_{len(blocks) + 1}"),
            "background": block_bg,
            "cleaned": False,
        })

    return blocks
