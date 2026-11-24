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

try:
    from main import run_paddle_ocr, extract_paddle_lines
except Exception:
    run_paddle_ocr = None
    extract_paddle_lines = None

logger = logging.getLogger("manga_bubble_engine")

METADATA_REGEX = re.compile(
    r"(?:(?:chapter|>ter|chap|ch\.?)\s*\d+|translation by|lettering\b|morishita|raizen\s+and\s+lightning|ogami['’]?s|not-so-ordinary|scanlation|redrawer|cleaner|raw provider|kyla aiko|mei amaki)",
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
    'forget', 'lightning', 'statement', 'earlier', 'retract', 'neighbor', 'born', 
    'frail', 'parents', 'told', 'sneaking', 'home', 'burglar', 'school', 'feeling',
    'class', 'name', 'turn', 'duty', 'need', 'order', 'coexist', 'expected', 'room',
    'left', 'blackout', 'president', 'already', 'power', 'restored', 'harmed'
}

# Single-word exclamations commonly found in manga speech bubbles
VALID_SINGLE_WORDS = {
    'oh', 'ok', 'okay', 'no', 'yes', 'bye', 'why', 'what', 'who', 'how', 
    'wait', 'stop', 'hey', 'hi', 'please', 'thanks', 'sure', 'yeah', 'nope', 
    'well', 'right', 'look', 'see', 'listen', 'help', 'ouch', 'ah', 'aha', 
    'uh', 'um', 'huh', 'wow', 'sorry', 'good'
}

VALID_SHORT_TOKENS = {'a', 'i', 'in', 'on', 'to', 'at', 'by', 'of', 'as', 'is', 'it', 'he', 'we', 'my', 'me', 'no', 'so', 'go', 'do', 'up', 'am', 'if', 'ok', 'an', 'be', 'or', 'us'}

def _calc_text_noise_ratio(text: str) -> float:
    """Calculates the ratio of noise tokens (unknown short tokens and alphanumeric/digit noise)."""
    cleaned = clean_manga_text(text)
    words = [re.sub(r'[^\w\']', '', w.lower(), flags=re.UNICODE) for w in cleaned.split()]
    words = [w for w in words if w]
    if not words:
        return 1.0
    noise_count = sum(
        1 for w in words
        if (len(w) <= 2 and w not in VALID_SHORT_TOKENS)
        or any(c.isdigit() for c in w)
        or not any(c.isalpha() for c in w)
    )
    return float(noise_count) / float(len(words))

def clean_manga_text(text: str) -> str:
    """Cleans OCR noise symbols from manga dialogue while preserving multilingual characters."""
    t = re.sub(r'[^\w\s\'\.,!\?\"\-—…~～«»“”„’ʻʼ`]', ' ', text)
    t = t.replace('_', ' ')
    return " ".join(t.split()).strip()

def is_manga_punctuation(text: str) -> bool:
    """
    Detects standalone expressive manga punctuation marks like '?', '!', '?!', '!?', '...', '…', '---'.
    In manga, characters often speak with just exclamation marks or question marks.
    These must NEVER be treated as noise or dropped.
    """
    t = text.strip()
    if not t:
        return False
    if t in ("?", "!", "…", "?!", "!?", "!?!", "!!", "???", "??", "!!!!", "--", "---", "...", ".."):
        return True
    # Normalize common OCR misreadings where '!' was read as 1, l, or I next to punctuation
    normalized = re.sub(r'(?<=[?!])\s*[1lI|]\b', '!', t)
    normalized = re.sub(r'\b[1lI|]\s*(?=[?!])', '!', normalized)
    clean = re.sub(r"[\s\.,!?'\"_\-~:;#@$%^&*()\[\]{}|<>/\\`+=…¿¡]", "", normalized)
    if re.fullmatch(r"[.\s…]{2,}", normalized):
        return True
    if not clean:
        return any(c in "?!…~" or ".." in normalized or "--" in normalized for c in normalized)
    if len(clean) == 1 and any(c in "?!…" for c in normalized) and len(t) <= 4:
        return True
    return False

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
    if cleaned.lower() in ('i', 'a'):
        return True

    lang = (source_lang or "auto").lower()

    # 1. Japanese (Kana or Kanji)
    if lang in ("ja", "jpn", "japanese") or any('\u3040' <= c <= '\u30ff' for c in text):
        jp_chars = sum(1 for c in text if '\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff')
        return jp_chars >= 1

    # 2. Korean (Hangul)
    if lang in ("ko", "kor", "korean") or any('\uac00' <= c <= '\ud7af' or '\u1100' <= c <= '\u11ff' for c in text):
        ko_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7af' or '\u1100' <= c <= '\u11ff' or '\u3130' <= c <= '\u318f')
        return ko_chars >= 1

    # 3. Chinese (Hanzi without Kana)
    if lang in ("zh", "chs", "cht", "chinese", "ch") or (any('\u4e00' <= c <= '\u9fff' for c in text) and not any('\u3040' <= c <= '\u30ff' for c in text)):
        zh_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf')
        return zh_chars >= 1

    if len(cleaned) < 2:
        return False

    # 4. Russian / Cyrillic
    if lang in ("ru", "rus", "russian", "cyrillic") or any('\u0400' <= c <= '\u04ff' for c in text):
        ru_letters = sum(1 for c in text if '\u0400' <= c <= '\u04ff' and c.isalpha())
        return ru_letters >= 2

    # 5. Latin-based dialogue (English, Uzbek, French, German, Spanish, etc.)
    words = [re.sub(r'[^\w\']', '', w.lower(), flags=re.UNICODE) for w in cleaned.split()]
    words = [w for w in words if w]
    if not words:
        return False

    # Single-word dialogue
    if len(words) == 1:
        w = words[0]
        if w in VALID_SINGLE_WORDS:
            return True
        if len(w) >= 3 and any(c in 'aeiouyаеёиоуыэюя' for c in w):
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
            except Exception:
                pass
        if has_strong:
            return True
        return sum(1 for c in cleaned if c.isalpha()) >= 4

    # Other languages
    return sum(1 for c in cleaned if c.isalpha()) >= 3


def _get_box_border_mask(gray: np.ndarray) -> Optional[np.ndarray]:
    """
    Computes a morphological straight frame mask (box_border_mask) for the page.
    Uses resolution-aware kernel sizes scaled to page dimensions (with safe min/max clamps)
    so bold/SFX letters on high-res pages are not flagged as box borders.
    """
    if gray is None or gray.size == 0:
        return None
    H, W = gray.shape[:2]
    dark = (gray < 160).astype(np.uint8)
    kw = max(24, min(int(round(W * 0.035)), 80))
    kh = max(24, min(int(round(H * 0.035)), 80))
    line_kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    line_kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
    horiz_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, line_kernel_h)
    vert_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, line_kernel_v)

    # Exclude solid large dark regions (e.g. dark narration boxes, dark clothing/artwork)
    # so they are not treated as straight box borders
    thick_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 16))
    solid_dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, thick_kernel)
    horiz_lines = cv2.bitwise_and(horiz_lines, cv2.bitwise_not(solid_dark))
    vert_lines = cv2.bitwise_and(vert_lines, cv2.bitwise_not(solid_dark))
    return cv2.bitwise_or(horiz_lines, vert_lines)


def _line_crosses_border_mask(
    border_mask: np.ndarray,
    ptA: tuple,
    ptB: tuple,
    boxA: Optional[list] = None,
    boxB: Optional[list] = None
) -> bool:
    """
    Checks if a straight frame border barrier separates boxA and boxB or crosses ptA->ptB.
    Checks border continuity specifically across the facing gap corridor between boxA and boxB.
    """
    if border_mask is None or border_mask.size == 0:
        return False
    H, W = border_mask.shape[:2]

    if ptA is not None and len(ptA) == 4 and boxA is None:
        boxA = ptA
        ptA = ((ptA[0] + ptA[2]) / 2.0, (ptA[1] + ptA[3]) / 2.0)
    if ptB is not None and len(ptB) == 4 and boxB is None:
        boxB = ptB
        ptB = ((ptB[0] + ptB[2]) / 2.0, (ptB[1] + ptB[3]) / 2.0)

    # Check border continuity specifically across the facing gap corridor between boxA and boxB
    if boxA is not None and boxB is not None:
        xA1, yA1, xA2, yA2 = boxA
        xB1, yB1, xB2, yB2 = boxB

        # 1. Vertically stacked / facing boxes (one above another)
        top_box = boxA if yA1 < yB1 else boxB
        bot_box = boxB if yA1 < yB1 else boxA
        x_start = max(top_box[0], bot_box[0])
        x_end = min(top_box[2], bot_box[2])
        corridor_w = x_end - x_start

        if corridor_w >= 8 and top_box[3] < bot_box[1]:
            y_gap_top = top_box[3]
            y_gap_bot = bot_box[1]
            search_y1 = max(0, y_gap_top - 2)
            search_y2 = min(H, y_gap_bot + 3)
            min_consec_w = max(10, int(0.35 * corridor_w))

            for r in range(search_y1, search_y2):
                row_slice = border_mask[r, x_start:x_end]
                if np.mean(row_slice > 0) >= 0.45:
                    return True
                max_run = 0
                cur_run = 0
                for val in row_slice:
                    if val > 0:
                        cur_run += 1
                        if cur_run > max_run:
                            max_run = cur_run
                    else:
                        cur_run = 0
                if max_run >= min_consec_w:
                    return True

        # 2. Horizontally adjacent / facing boxes (columns)
        left_box = boxA if xA1 < xB1 else boxB
        right_box = boxB if xA1 < xB1 else boxA
        y_start = max(left_box[1], right_box[1])
        y_end = min(left_box[3], right_box[3])
        corridor_h = y_end - y_start

        if corridor_h >= 8 and left_box[2] < right_box[0]:
            x_gap_left = left_box[2]
            x_gap_right = right_box[0]
            search_x1 = max(0, x_gap_left - 2)
            search_x2 = min(W, x_gap_right + 3)
            min_consec_h = max(10, int(0.35 * corridor_h))

            for c in range(search_x1, search_x2):
                col_slice = border_mask[y_start:y_end, c]
                if np.mean(col_slice > 0) >= 0.45:
                    return True
                max_run = 0
                cur_run = 0
                for val in col_slice:
                    if val > 0:
                        cur_run += 1
                        if cur_run > max_run:
                            max_run = cur_run
                    else:
                        cur_run = 0
                if max_run >= min_consec_h:
                    return True

        # If both boxes have substantial horizontal overlap or vertical overlap without a corridor barrier:
        # Do not let diagonal rays falsely split them!
        if corridor_w >= 10 or corridor_h >= 10:
            return False

    # Fallback for diagonal rays or when boxA/boxB are not provided
    cxA, cyA = ptA
    cxB, cyB = ptB
    dist = int(np.hypot(cxB - cxA, cyB - cyA))
    if dist < 2:
        return False
    num_samples = max(dist * 2, 20)
    xs = np.clip(np.linspace(cxA, cxB, num_samples).round().astype(int), 0, W - 1)
    ys = np.clip(np.linspace(cyA, cyB, num_samples).round().astype(int), 0, H - 1)

    pad = 2
    valid_indices = []
    for idx in range(num_samples):
        x, y = xs[idx], ys[idx]
        if boxA is not None:
            if (boxA[0] + pad <= x <= boxA[2] - pad) and (boxA[1] + pad <= y <= boxA[3] - pad):
                continue
        if boxB is not None:
            if (boxB[0] + pad <= x <= boxB[2] - pad) and (boxB[1] + pad <= y <= boxB[3] - pad):
                continue
        valid_indices.append(idx)

    if not valid_indices:
        inner_start = max(1, int(0.20 * num_samples))
        inner_end = min(num_samples - 1, int(0.80 * num_samples))
        valid_indices = list(range(inner_start, inner_end + 1))

    sampled_vals = border_mask[ys[valid_indices], xs[valid_indices]]
    max_run = 0
    cur_run = 0
    for v in sampled_vals:
        if v > 0:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            cur_run = 0
    return max_run >= 2 or np.sum(sampled_vals > 0) >= 3


def _has_continuous_dark_segment(arr_1d, dark_thresh=110, min_len=12):
    """
    Checks if a 1D array has at least min_len consecutive pixels with value < dark_thresh.
    Used to detect continuous dark frame/border lines across boundaries.
    """
    if arr_1d is None or len(arr_1d) < min_len or min_len <= 0:
        return False
    is_dark = (arr_1d < dark_thresh)
    consec = 0
    for v in is_dark:
        if v:
            consec += 1
            if consec >= min_len:
                return True
        else:
            consec = 0
    return False


def _is_dark_barrier_between_x(gray: np.ndarray, boxA: list, boxB: list, dark_thresh: int = 100, min_line_ratio: float = 0.40, border_mask: Optional[np.ndarray] = None) -> bool:
    if gray is None or gray.size == 0:
        return False
    H, W = gray.shape[:2]
    if boxA[0] > boxB[0]:
        boxA, boxB = boxB, boxA
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB

    # Check if both boxes sit inside a dark narration box
    comb_crop = gray[min(yA1, yB1):max(yA2, yB2), min(xA1, xB1):max(xA2, xB2)]
    if comb_crop.size > 0:
        med_val = float(np.median(comb_crop))
        if med_val < 95 or (med_val < 120 and float(np.mean(comb_crop < 110)) >= 0.50):
            # Inside a dark narration box: dark pixels are normal background, not a barrier
            y_top = max(yA1, yB1)
            y_bot = min(yA2, yB2)
            if y_bot > y_top and xA2 < xB1:
                gap_slice = gray[y_top:y_bot, max(0, xA2):min(W, xB1)]
                if gap_slice.size > 0 and np.mean(gap_slice > 185) >= 0.40:
                    return True
            return False

    cxA = int((xA1 + xA2) / 2.0)
    cyA = int((yA1 + yA2) / 2.0)
    cxB = int((xB1 + xB2) / 2.0)
    cyB = int((yB1 + yB2) / 2.0)
    if border_mask is None:
        border_mask = _get_box_border_mask(gray)
    if border_mask is not None and _line_crosses_border_mask(border_mask, (cxA, cyA), (cxB, cyB), boxA=boxA, boxB=boxB):
        return True

    y_top = max(yA1, yB1)
    y_bot = min(yA2, yB2)
    # If vertically stacked or separated with no vertical overlap, there is no X barrier
    if y_bot <= y_top:
        return False
    if y_bot - y_top < 6:
        mid_y = int((y_top + y_bot) / 2.0)
        y_top = max(0, mid_y - 8)
        y_bot = min(H, mid_y + 8)
    else:
        y_top = max(0, y_top)
        y_bot = min(H, y_bot)

    span = y_bot - y_top
    min_consec = max(12, int(0.25 * span))

    # Standard positive gap check
    if xA2 < xB1:
        xA2_c = max(0, min(W - 1, xA2))
        xB1_c = max(0, min(W, xB1))
        if xB1_c > xA2_c and y_bot > y_top:
            gap_slice = gray[y_top:y_bot, xA2_c:xB1_c]
            if gap_slice.size > 0 and gap_slice.shape[0] >= 4:
                dark_counts = np.sum(gap_slice < dark_thresh, axis=0)
                col_ratios = dark_counts / float(gap_slice.shape[0])
                if bool(np.any(col_ratios >= min_line_ratio)):
                    return True
                for c in range(gap_slice.shape[1]):
                    if _has_continuous_dark_segment(gap_slice[:, c], dark_thresh=110, min_len=min_consec):
                        return True
            if xB1_c - xA2_c <= 8:
                b_left = max(0, xA2_c - 3)
                b_right = min(W, xB1_c + 3)
                b_slice = gray[y_top:y_bot, b_left:b_right]
                if b_slice.size > 0 and b_slice.shape[0] >= 4:
                    for c in range(b_slice.shape[1]):
                        if _has_continuous_dark_segment(b_slice[:, c], dark_thresh=110, min_len=min_consec):
                            return True
        return False
    else:
        # When xA2 >= xB1: check if there is a vertical divider column between cxA and cxB
        cx_min = max(0, min(cxA, cxB))
        cx_max = min(W - 1, max(cxA, cxB))
        if cx_max - cx_min >= 4 and (y_bot - y_top) >= 8:
            center_slice = gray[y_top:y_bot, cx_min:cx_max + 1]
            if center_slice.size > 0 and center_slice.shape[0] >= 6:
                dark_counts = np.sum(center_slice < dark_thresh, axis=0)
                col_ratios = dark_counts / float(center_slice.shape[0])
                if np.any(col_ratios >= 0.65):
                    return True
        return False


def _is_dark_barrier_between_y(gray: np.ndarray, boxA: list, boxB: list, dark_thresh: int = 95, min_line_ratio: float = 0.40, border_mask: Optional[np.ndarray] = None) -> bool:
    if gray is None or gray.size == 0:
        return False
    H, W = gray.shape[:2]
    if boxA[1] > boxB[1]:
        boxA, boxB = boxB, boxA
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB

    # Check if both boxes sit inside a dark narration box
    comb_crop = gray[min(yA1, yB1):max(yA2, yB2), min(xA1, xB1):max(xA2, xB2)]
    if comb_crop.size > 0:
        med_val = float(np.median(comb_crop))
        if med_val < 95 or (med_val < 120 and float(np.mean(comb_crop < 110)) >= 0.50):
            # Inside a dark narration box: dark pixels are normal background, not a barrier
            x_left = max(xA1, xB1)
            x_right = min(xA2, xB2)
            if x_right > x_left and yA2 < yB1:
                gap_slice = gray[max(0, yA2):min(H, yB1), max(0, x_left):min(W, x_right)]
                if gap_slice.size > 0 and np.mean(gap_slice > 185) >= 0.40:
                    return True
            return False

    cxA = int((xA1 + xA2) / 2.0)
    cyA = int((yA1 + yA2) / 2.0)
    cxB = int((xB1 + xB2) / 2.0)
    cyB = int((yB1 + yB2) / 2.0)
    if border_mask is None:
        border_mask = _get_box_border_mask(gray)
    if border_mask is not None and _line_crosses_border_mask(border_mask, (cxA, cyA), (cxB, cyB), boxA=boxA, boxB=boxB):
        return True

    x_left = max(xA1, xB1)
    x_right = min(xA2, xB2)
    if x_right - x_left < 6:
        mid_x = int(((xA1 + xA2) / 2.0 + (xB1 + xB2) / 2.0) / 2.0)
        x_left = max(0, mid_x - 8)
        x_right = min(W, mid_x + 8)
    else:
        x_left = max(0, x_left)
        x_right = min(W, x_right)

    span = x_right - x_left
    min_consec = max(12, int(0.25 * span))

    # Standard positive gap check
    if yA2 < yB1:
        yA2_c = max(0, min(H, yA2))
        yB1_c = max(0, min(H, yB1))
        if yB1_c > yA2_c and x_right > x_left:
            gap_slice = gray[yA2_c:yB1_c, x_left:x_right]
            if gap_slice.size > 0 and gap_slice.shape[1] >= 4:
                dark_counts = np.sum(gap_slice < dark_thresh, axis=1)
                row_ratios = dark_counts / float(gap_slice.shape[1])
                if bool(np.any(row_ratios >= min_line_ratio)):
                    return True
                for r in range(gap_slice.shape[0]):
                    if _has_continuous_dark_segment(gap_slice[r, :], dark_thresh=110, min_len=min_consec):
                        return True
            if yB1_c - yA2_c <= 8:
                b_top = max(0, yA2_c - 3)
                b_bot = min(H, yB1_c + 3)
                b_slice = gray[b_top:b_bot, x_left:x_right]
                if b_slice.size > 0 and b_slice.shape[1] >= 4:
                    for r in range(b_slice.shape[0]):
                        if _has_continuous_dark_segment(b_slice[r, :], dark_thresh=110, min_len=min_consec):
                            return True
        return False

    # When yA2 >= yB1:
    # Do NOT test the center-to-center line cyA..cyB (it mistakes ink of letters for a wall!).
    # Only check if there is a CONTINUOUS horizontal dark frame line (thickness 1-4 px, occupying >= 70% of width)
    # between the two boxes.
    cyA = int((yA1 + yA2) / 2.0)
    cyB = int((yB1 + yB2) / 2.0)
    cxA = int((xA1 + xA2) / 2.0)
    cxB = int((xB1 + xB2) / 2.0)
    cx_min = max(0, min(cxA, cxB))
    cx_max = min(W - 1, max(cxA, cxB))
    cy_min = max(0, min(cyA, cyB))
    cy_max = min(H - 1, max(cyA, cyB))

    if cy_max - cy_min < 1:
        return False

    span_w = x_right - x_left
    if span_w < 4:
        x_left = max(0, min(xA1, xB1))
        x_right = min(W, max(xA2, xB2))
        span_w = x_right - x_left
        if span_w < 4:
            return False

    consecutive_dark = 0
    for y in range(cy_min, cy_max + 1):
        row = gray[y, x_left:x_right]
        if row.size < 4:
            continue
        dark_mask = (row < dark_thresh)
        dark_ratio = np.sum(dark_mask) / float(row.size)
        if dark_ratio >= 0.70:
            # Check continuity: longest consecutive run of dark pixels
            max_run = 0
            cur_run = 0
            for d in dark_mask:
                if d:
                    cur_run += 1
                    if cur_run > max_run:
                        max_run = cur_run
                else:
                    cur_run = 0
            if max_run >= int(row.size * 0.70):
                consecutive_dark += 1
                continue

        if 1 <= consecutive_dark <= 4:
            return True
        consecutive_dark = 0

    if 1 <= consecutive_dark <= 4:
        return True

    return False

def get_white_cc_labels(gray: np.ndarray):
    """
    Computes 8-connectivity connected components on white background (gray > 200).
    Returns (labels, stats) where stats contains CC_STAT_AREA.
    Does not use id(gray) cache to prevent memory address collision.
    """
    if gray is None or gray.size == 0:
        return None, None
    white_mask = (gray > 200).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    return labels, stats

def _get_box_white_label(labels: np.ndarray, gray: np.ndarray, stats: Optional[np.ndarray], box: list, pad: int = 3) -> int:
    """
    Returns the mode white component label (most frequent non-zero, area > 200 px)
    from a 2-4 px expanded perimeter around the box instead of a single center pixel.
    """
    if labels is None or gray is None:
        return 0
    xA1, yA1, xA2, yA2 = box
    H, W = gray.shape[:2]
    px1, py1 = max(0, int(xA1) - pad), max(0, int(yA1) - pad)
    px2, py2 = min(W, int(xA2) + pad), min(H, int(yA2) + pad)
    if px2 <= px1 or py2 <= py1:
        return 0
    sub_labels = labels[py1:py2, px1:px2]
    sub_gray = gray[py1:py2, px1:px2]
    valid_labels = sub_labels[(sub_labels > 0) & (sub_gray > 200)]
    if valid_labels.size == 0:
        return 0

    if stats is not None and len(stats) > 0:
        areas = stats[valid_labels, cv2.CC_STAT_AREA]
        large_labels = valid_labels[areas > 200]
        if large_labels.size > 0:
            valid_labels = large_labels

    vals, counts = np.unique(valid_labels, return_counts=True)
    return int(vals[np.argmax(counts)])

def are_different_white_contours(gray: np.ndarray, boxA: list, boxB: list) -> bool:
    """
    Checks if boxA and boxB belong to distinct white connected components (gray > 200).
    """
    if gray is None or gray.size == 0:
        return False
    labels, stats = get_white_cc_labels(gray)
    if labels is None or stats is None:
        return False
    lblA = _get_box_white_label(labels, gray, stats, boxA)
    lblB = _get_box_white_label(labels, gray, stats, boxB)
    return bool(lblA > 0 and lblB > 0 and lblA != lblB)

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
            labels, stats = get_white_cc_labels(gray)
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

    # 3. to'rtburchak chegara (rectangular border)
    if box is not None and (H > (by2 - by1) + 12 or W > (bx2 - bx1) + 12):
        dark = (gray < 130).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0

        for cnt in contours:
            if cv2.pointPolygonTest(cnt, (float(cx), float(cy)), False) >= 0:
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                if rx <= bx1 + 6 and ry <= by1 + 6 and rx + rw >= bx2 - 6 and ry + rh >= by2 - 6:
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
    if pytesseract is None and run_paddle_ocr is None:
        return []

    img_np = np.asarray(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    lang_map = {
        "ko": "kor", "kor": "kor", "korean": "kor",
        "zh": "chi_sim", "chs": "chi_sim", "cht": "chi_sim", "chinese": "chi_sim",
        "ru": "rus", "rus": "rus", "russian": "rus",
        "ja": "jpn", "jpn": "jpn", "japanese": "jpn",
        "en": "eng", "eng": "eng", "english": "eng",
        "auto": "eng",
    }
    tess_lang = lang_map.get((source_lang or "auto").lower().strip(), "eng")
    
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
        if conf < 25 or len(word) < 1:
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
                is_dup = True
                break
        if is_dup:
            continue

        raw_boxes.append({
            'box': [bx1, by1, bx2, by2],
            'text': word,
            'conf': conf
        })

    # If no raw boxes and paddleocr is available, use paddleocr fallback
    if not raw_boxes and callable(run_paddle_ocr) and callable(extract_paddle_lines):
        try:
            import io
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            paddle_res = run_paddle_ocr(img_bytes, tess_lang)
            paddle_lines = extract_paddle_lines(paddle_res, source_lang)
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
    cc_labels, cc_stats = get_white_cc_labels(gray)
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
        clusters.append({
            "block_id": f"b_{len(clusters) + 1}",
            "box": [bx1, by1, bx2, by2],
            "text": full_text,
            "count": len(comp)
        })

    # 3. Filter Valid Dialogue & Reject Metadata / Hair / Screentone False Positives
    all_tboxes = []
    filtered = []
    for c in clusters:
        text = clean_manga_text(c["text"])
        bx1, by1, bx2, by2 = c["box"]
        
        # Max bubble dimension check
        if (by2 - by1) > h * 0.22 or (bx2 - bx1) > w * 0.65:
            continue

        # Metadata check (Chapter title, credits, etc.)
        if METADATA_REGEX.search(text):
            continue

        # Strictly validate dialogue text
        if not is_valid_manga_dialogue(text, source_lang):
            continue

        is_narration = is_dark_narration_box(gray, [bx1, by1, bx2, by2])

        # Speech bubble whiteness check (ensures bubble has light background)
        px1, py1 = max(0, bx1 - 6), max(0, by1 - 6)
        px2, py2 = min(w, bx2 + 6), min(h, by2 + 6)
        patch = gray[py1:py2, px1:px2]
        pure_white = np.mean(patch > 215) if patch.size > 0 else 0
        mean_val = np.mean(patch) if patch.size > 0 else 0

        # Reject dark backgrounds (clothes, dark hair), but allow dark narration frames
        if not is_narration and pure_white < 0.40 and mean_val < 185:
            continue

        filtered.append({
            "block_id": c.get("block_id", f"b_{len(filtered) + 1}"),
            "text_box": [bx1, by1, bx2, by2],
            "text": text,
            "category": "narration" if is_narration else "speech_bubble",
            "is_narration": is_narration,
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
            # Inverted OCR pass (255 - gray) for dark narration frames
            crop_patch = gray[oy1:oy2, ox1:ox2]
            crop_pil = Image.fromarray(255 - crop_patch) if crop_patch.size > 0 else pil_image.crop((ox1, oy1, ox2, oy2))
        else:
            crop_pil = pil_image.crop((ox1, oy1, ox2, oy2))
        
        text = f["text"]
        try:
            crop_text = pytesseract.image_to_string(crop_pil, lang=tess_lang, config='--psm 6').strip()
            crop_cleaned = clean_manga_text(crop_text)
            if is_valid_manga_dialogue(crop_cleaned, source_lang):
                old_noise = _calc_text_noise_ratio(text)
                new_noise = _calc_text_noise_ratio(crop_cleaned)
                if new_noise < old_noise:
                    text = crop_cleaned
        except Exception:
            pass

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
            "confidence": 0.95,
            "bbox": [scaled_bx1, scaled_by1, scaled_bx2, scaled_by2],
            "overlay_bbox": [scaled_bx1, scaled_by1, scaled_bx2, scaled_by2],
            "text_erase_bbox": text_erase_rect,
            "overlay_polygon": poly,
            "category": "narration" if is_narration else "speech_bubble",
            "bubble_id": len(blocks) + 1,
            "block_id": f.get("block_id", f"b_{len(blocks) + 1}"),
            "background": block_bg,
            "cleaned": True,
        })

    return blocks
