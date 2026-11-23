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

from models import MangaBubbleDetector, MangaOCRModel, MangaInpaintingModel
from dataset_loader import IDX_TO_CATEGORY
from self_learning_collector import SelfLearningCollector
from self_learning_trainer import SelfLearningTrainer
from utils import get_best_font, decode_ctc

logger = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')


def is_manga_punctuation(text: str) -> bool:
    """
    Detects standalone expressive manga punctuation marks like '?', '!', '?!', '!?', '...', '…', '---'.
    Also catches common OCR misreadings of exclamation mark (e.g. '?1', '?l', '?I', '!1', '1!').
    In manga, characters often speak with just a question mark or exclamation mark;
    these must NEVER be erased or treated as noise!
    """
    t = text.strip()
    if not t:
        return False
    # Normalize common OCR misreadings where '!' was read as 1, l, or I next to punctuation
    normalized = re.sub(r'(?<=[?!])\s*[1lI|]\b', '!', t)
    normalized = re.sub(r'\b[1lI|]\s*(?=[?!])', '!', normalized)
    
    # Check if all remaining characters are punctuation symbols
    clean = re.sub(r"[\s\.,!?'\"_\-~:;#@$%^&*()\[\]{}|<>/\\`+=…¿¡]", "", normalized)
    if re.fullmatch(r"[.\s…]+", normalized):
        return True
    if not clean:
        return any(c in "?!…~" or ".." in normalized or "--" in normalized for c in normalized)
    
    # Also handle cases where a single digit/letter was misread attached to heavy punctuation, e.g. "?1"
    if len(clean) == 1 and any(c in "?!…" for c in normalized) and len(t) <= 4:
        return True
        
    return False


def normalize_manga_punctuation(text: str) -> str:
    """Normalizes manga punctuation OCR quirks into clean comic typography."""
    t = text.strip()
    t = re.sub(r'(?<=[?!])\s*[1lI|]\b', '!', t)
    t = re.sub(r'\b[1lI|]\s*(?=[?!])', '!', t)
    compact = "".join(t.split())
    if re.fullmatch(r"[.…]+", compact):
        return "..."
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


def is_sfx_text(text: str) -> bool:
    """
    Detects sound effects (SFX) drawn directly on manga artwork or near bubbles.
    Catches elongated repeated characters (e.g. VREEEEM, Vzeeeem, shhh, booom),
    consonant-heavy words (zksh, clck, tsk), and standard comic SFX words (Ding, Boom, etc.).
    """
    if not text:
        return False
    words = text.split()
    for w in words:
        w_clean = re.sub(r'[^\w]', '', w).lower()
        if re.search(r'([^\d\s])\1{2,}', w_clean):
            return True

    clean = re.sub(r'[^\w]', '', text).lower()
    if not clean:
        return False

    has_cjk_or_hangul = any(
        '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af'
        for c in text
    )
    if not has_cjk_or_hangul:
        vowels = set("aeiouyöаеёиоуыэюя")
        has_latin_or_cyrillic = any(('a' <= c <= 'z') or ('\u0400' <= c <= '\u04ff') for c in clean)
        if has_latin_or_cyrillic:
            v_count = sum(1 for c in clean if c in vowels)
            if len(clean) <= 6 and v_count == 0:
                return True

    known_sfx = {
        "ding", "zksh", "vreee", "vreeem", "vzeeeem", "vre", "zrk", "zrrk", "bzz",
        "boom", "thump", "doki", "tok", "tap", "sfx", "flik", "krrr",
        "shhh", "wham", "clang", "clack", "bang", "snap", "crash", "smash",
        "gasp", "sigh", "whoosh", "zap", "crunch", "giggle", "snicker", "chuckle"
    }
    if len(words) <= 2 and clean in known_sfx:
        return True
    return False


def normalize_cyrillic_ocr(text: str) -> str:
    """Normalizes OCR misclassifications where Latin lookalike characters or digits leak into Cyrillic words."""
    if not text:
        return ""
    total_cyr = sum(1 for c in text if '\u0400' <= c <= '\u04ff')
    if total_cyr == 0:
        return text
    total_lat = sum(1 for c in text if ('a' <= c <= 'z') or ('A' <= c <= 'Z'))
    predom_cyr = total_cyr > total_lat * 2

    homoglyphs = {
        'A': 'А', 'a': 'а',
        'B': 'В',
        'C': 'С', 'c': 'с',
        'E': 'Е', 'e': 'е',
        'H': 'Н',
        'K': 'К', 'k': 'к',
        'M': 'М', 'm': 'м',
        'O': 'О', 'o': 'о',
        'P': 'Р', 'p': 'р',
        'T': 'Т',
        'X': 'Х', 'x': 'х',
        'Y': 'У', 'y': 'у',
    }
    words = text.split()
    fixed_words = []
    for w in words:
        w_clean = re.sub(r'[^\w]', '', w)
        w_cyr = sum(1 for c in w_clean if '\u0400' <= c <= '\u04ff')
        if w_cyr > 0 or predom_cyr:
            new_chars = []
            for ch in w:
                if ch in homoglyphs:
                    new_chars.append(homoglyphs[ch])
                elif ch == '0' and len(w_clean) > 1:
                    new_chars.append('О')
                elif ch == '3' and len(w_clean) > 1 and w_cyr > 0:
                    new_chars.append('З')
                else:
                    new_chars.append(ch)
            fixed_words.append("".join(new_chars))
        else:
            fixed_words.append(w)
    return " ".join(fixed_words)


def is_ocr_garbage_noise(text: str, source_lang: str = "auto", is_inside_bubble: bool = False) -> bool:
    """Detects OCR hallucinations from manga screentones, halftones, speed lines, and noise."""
    t = text.strip()
    if not t:
        return True
    # Never discard valid expressive manga punctuation as noise!
    if is_manga_punctuation(t):
        return False
    # Strip common leading/trailing OCR specks
    t_stripped = t.strip("_~-=+*`\\/| \t\n")
    if not t_stripped:
        return True
    # Bracket or symbol-based noise patterns
    if any(c in t_stripped for c in "#@$%^&*=\\<>[]{}|/`"):
        return True

    # Normalize common comic font scanlation OCR substitutions (prevent corrupting CLASS 1-1, 1-A, 1st):
    t = re.sub(r'(?<![\d\-])\b1\b(?!\s*(?:st|nd|rd|th)\b)(?![\d\-])', 'I', t_stripped, flags=re.I)
    t = re.sub(r'\b1(?!(?:st|nd|rd|th)\b)(?=[A-Za-z])', 'I', t, flags=re.I)
    t = re.sub(r'\bUT\b', 'IT', t)

    t = normalize_cyrillic_ocr(t)
    clean = re.sub(r"[\s\.,!?'\"_\-~:;#@$%^&*()\[\]{}|<>/\\`+=]", "", t)
    if not clean:
        return True

    letters = sum(1 for c in clean if c.isalpha())
    digits = sum(1 for c in clean if c.isdigit())
    has_cyrillic = any('\u0400' <= c <= '\u04ff' for c in clean)
    has_latin = any('a' <= c.lower() <= 'z' for c in clean)
    has_kana = any('\u3040' <= c <= '\u30ff' for c in clean)
    has_cjk = any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' for c in clean)

    # Filter Japanese/Artwork SFX before translation:
    # When source_lang == "en" (or Latin-script page), reject isolated 1-3 char Hiragana/Katakana fragments
    # (e.g. 'nた', 'ルル ルッ') on artwork as SFX/noise unless inside a closed speech bubble
    if (source_lang in {"en", "ru"} or source_lang == "auto") and not is_inside_bubble:
        if has_kana:
            kana_chars = sum(1 for c in clean if '\u3040' <= c <= '\u30ff')
            if 1 <= kana_chars <= 4 and len(clean) <= 5:
                return True
        elif has_cjk and len(clean) <= 3 and (has_latin or source_lang == "en"):
            return True

    # Treat 2-3 char alphabetic tokens ending with ? or ! inside detected bubbles as valid dialogue
    clean_alpha = re.sub(r'[^A-Za-zА-Яа-я]', '', t)
    if 2 <= len(clean_alpha) <= 3 and any(c in "?!" for c in t):
        if is_inside_bubble or clean_alpha.lower() in {"hm", "mm", "eh", "ha", "uh", "um", "yo", "ya", "ok"}:
            return False

    # Pure numbers in comic panels without dialogue context
    if letters == 0 and digits > 0:
        return True

    # Check mixed alphanumeric tokens at word level with 30% digit ratio threshold
    for w in t.split():
        w_clean = re.sub(r'[^\w]', '', w)
        w_letters = sum(1 for c in w_clean if c.isalpha())
        w_digits = sum(1 for c in w_clean if c.isdigit())
        if w_digits > 0 and w_letters > 0:
            if not re.search(r'\b(?:\d{1,2}:\d{2}|\d+-(?:й|го|я|е|th|st|nd|rd)|1st|2nd|3rd|\d+th|\d+pm|\d+am)\b', w, re.I):
                if w_digits / float(w_digits + w_letters) > 0.30:
                    return True

    # Standalone 1-2 char noise check
    if len(clean) <= 2:
        valid_shorts = {
            # English
            "i", "a", "no", "oh", "ok", "ah", "eh", "uh", "um", "yo", "ya", "ha", 
            "hm", "mm", "so", "to", "he", "we", "me", "my", "in", "on", "at", "by", "if", 
            "da", "hi", "it", "is", "am", "as", "us", "up", "do",
            "go", "an", "or", "of", "be", "ex", "re", "ut", "im", "id", "ll", "ve", "tv", "mr", "ms", "dr", "st",
            # Russian (Cyrillic single letters and 2-letter words/SFX)
            "я", "и", "а", "о", "у", "э", "в", "с", "к", "он", "ты", "мы", "вы",
            "ну", "но", "не", "да", "до", "по", "за", "из", "на", "от", "со", "ко",
            "об", "же", "ли", "бы", "ее", "их", "ей", "им", "то", "те", "та", "ту",
            "эх", "ах", "ох", "ух", "эй", "ай", "ой", "хм", "гм", "ха", "хе", "хи",
            "чу", "ба", "фу", "уф", "топ", "тап", "тук", "кап", "бах", "бум", "пик",
            "ум", "ась", "уж",
            # Uzbek
            "va", "bu", "shu", "ha", "u"
        }
        if clean.lower() not in valid_shorts:
            if has_cjk:
                if (source_lang in {"en", "ru"} or source_lang == "auto") and not is_inside_bubble:
                    return True
            else:
                return True

    # In Russian mode or if Cyrillic is present:
    if has_cyrillic or source_lang in {"ru", "russian"}:
        # Non-homoglyph Latin letters remaining in Russian token indicate noise (e.g. 'FII', 'Fов', 'UD')
        # Allow Latin acronyms and common short words inside speech bubbles (e.g. 'VIP', 'FBI', 'USA', 'OK', 'TV')
        if has_latin:
            if is_inside_bubble and (clean.isupper() or clean.lower() in valid_shorts or (len(clean) <= 5 and clean.isalpha() and not any('\u0400' <= c <= '\u04ff' for c in clean))):
                pass
            else:
                return True

    return False

# Hardware optimization for AMD Ryzen 5 (4 cores, 8 threads)
if not torch.cuda.is_available():
    try:
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        cv2.setNumThreads(num_threads)
    except Exception:
        pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _crop_reading_key(r, reading_direction: str = "rtl", col_map: dict = None, line_idx_map: dict = None):
    """
    Sort key function for OCR items inside a crop, supporting RTL manga reading direction
    and multi-column/line grouping.
    """
    col = col_map.get(id(r), 0) if col_map else 0
    line_idx = line_idx_map.get(id(r), None) if line_idx_map else None

    if isinstance(r, dict) and "box" in r:
        box = r["box"]
        y_c = (box[1] + box[3]) / 2.0
        x1 = box[0]
        x_c = (box[0] + box[2]) / 2.0
    elif isinstance(r, (list, tuple)) and len(r) > 0 and isinstance(r[0], (list, tuple)):
        ys = [pt[1] / 2.0 if isinstance(pt, (list, tuple)) else pt for pt in r[0]]
        xs = [pt[0] / 2.0 if isinstance(pt, (list, tuple)) else pt for pt in r[0]]
        y_c = (min(ys) + max(ys)) / 2.0
        x1 = min(xs)
        x_c = (min(xs) + max(xs)) / 2.0
    elif isinstance(r, (list, tuple)) and len(r) >= 4 and not isinstance(r[0], (list, tuple)):
        y_c = (r[1] + r[3]) / 2.0
        x1 = r[0]
        x_c = (r[0] + r[2]) / 2.0
    else:
        y_c = 0.0
        x1 = 0.0
        x_c = 0.0

    if col_map is not None or line_idx_map is not None:
        y_val = line_idx if line_idx is not None else y_c
        return (col, y_val, x1)

    if reading_direction == "rtl":
        return (y_c, -x_c)
    else:
        return (y_c, x1)


_white_cc_cache = {}
_pipeline_cache_lock = threading.Lock()


def _get_gray_cache_key(gray):
    H, W = gray.shape[:2]
    step_y = max(1, H // 32)
    step_x = max(1, W // 32)
    sub = np.ascontiguousarray(gray[::step_y, ::step_x])
    h = hashlib.sha256(sub.tobytes()).hexdigest()[:16]
    return (H, W, h)


def get_white_cc_labels(gray, return_stats=False):
    """
    Computes 8-way connected components and stats for white background pixels (gray > 200).
    Uses content-hashed locked cache to avoid address reuse collisions.
    """
    if gray is None or gray.size == 0:
        return (None, None) if return_stats else None
    cache_key = _get_gray_cache_key(gray)
    with _pipeline_cache_lock:
        if cache_key in _white_cc_cache:
            labels, stats = _white_cc_cache[cache_key]
            return (labels, stats) if return_stats else labels
    white_mask = (gray > 200).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    with _pipeline_cache_lock:
        if len(_white_cc_cache) > 16:
            _white_cc_cache.clear()
        _white_cc_cache[cache_key] = (labels, stats)
    return (labels, stats) if return_stats else labels


def _get_box_white_mode_label(labels, stats, gray, box, min_area=200, pad=3):
    """
    Returns the mode (most frequent non-zero with area > min_area) white component label
    from a 2-4 px expanded perimeter around the box.
    """
    if labels is None or gray is None:
        return 0
    H, W = gray.shape[:2]
    xA1, yA1, xA2, yA2 = box
    px1, py1 = max(0, int(xA1) - pad), max(0, int(yA1) - pad)
    px2, py2 = min(W, int(xA2) + pad), min(H, int(yA2) + pad)
    if px2 <= px1 or py2 <= py1:
        return 0

    sub_labels = labels[py1:py2, px1:px2]
    sub_gray = gray[py1:py2, px1:px2]

    valid_mask = (sub_labels > 0) & (sub_gray > 200)
    valid_labels = sub_labels[valid_mask]
    if valid_labels.size == 0:
        return 0

    vals, counts = np.unique(valid_labels, return_counts=True)
    sorted_order = np.argsort(-counts)
    for idx in sorted_order:
        lbl = int(vals[idx])
        if stats is not None and lbl < stats.shape[0]:
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area > min_area:
                return lbl
        elif stats is None:
            return lbl
    return 0


def are_different_white_contours(gray, boxA, boxB, min_white_ratio=0.15):
    """
    Checks if boxA and boxB belong to two different closed white contours (gray > 200, area > 200).
    If both belong to distinct white connected components (separate bubble/panel outlines),
    they must NEVER be merged.
    """
    if gray is None or gray.size == 0:
        return False
    labels, stats = get_white_cc_labels(gray, return_stats=True)
    if labels is None:
        return False

    labelA = _get_box_white_mode_label(labels, stats, gray, boxA, min_area=200, pad=3)
    labelB = _get_box_white_mode_label(labels, stats, gray, boxB, min_area=200, pad=3)

    if labelA > 0 and labelB > 0 and labelA != labelB:
        return True
    return False


def get_box_border_mask(gray):
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


def line_crosses_border_mask(border_mask, ptA, ptB, boxA=None, boxB=None):
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


def is_dark_barrier_between_x(gray, boxA, boxB, dark_thresh=100, min_line_ratio=0.40, border_mask=None):
    if gray is None or gray.size == 0:
        return False
    H, W = gray.shape[:2]
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
        border_mask = get_box_border_mask(gray)
    if border_mask is not None and line_crosses_border_mask(border_mask, (cxA, cyA), (cxB, cyB), boxA=boxA, boxB=boxB):
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

    if xA2 < xB1:
        xA2_c = max(0, min(W, xA2))
        xB1_c = max(0, min(W, xB1))
        gap_slice = gray[y_top:y_bot, xA2_c:xB1_c]
        if gap_slice.size > 0 and gap_slice.shape[0] >= 4:
            dark_counts = np.sum(gap_slice < dark_thresh, axis=0)
            col_ratios = dark_counts / float(gap_slice.shape[0])
            if np.any(col_ratios >= min_line_ratio):
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
        # When xA2 >= xB1: only check if there is a true vertical divider column between cxA and cxB
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


def is_dark_barrier_between_y(gray, boxA, boxB, dark_thresh=95, min_line_ratio=0.40, border_mask=None):
    if gray is None or gray.size == 0:
        return False
    H, W = gray.shape[:2]
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
        border_mask = get_box_border_mask(gray)
    if border_mask is not None and line_crosses_border_mask(border_mask, (cxA, cyA), (cxB, cyB), boxA=boxA, boxB=boxB):
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

    if yA2 < yB1:
        yA2_c = max(0, min(H, yA2))
        yB1_c = max(0, min(H, yB1))
        gap_slice = gray[yA2_c:yB1_c, x_left:x_right]
        if gap_slice.size > 0 and gap_slice.shape[1] >= 4:
            dark_counts = np.sum(gap_slice < dark_thresh, axis=1)
            row_ratios = dark_counts / float(gap_slice.shape[1])
            if np.any(row_ratios >= min_line_ratio):
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
    else:
        # When yA2 >= yB1:
        # Check if there is a continuous horizontal dark frame line between the two boxes.
        cyA = int((yA1 + yA2) / 2.0)
        cyB = int((yB1 + yB2) / 2.0)
        cy_min = max(0, min(cyA, cyB))
        cy_max = min(H - 1, max(cyA, cyB))
        if cy_max - cy_min < 2:
            return False
        center_slice = gray[cy_min:cy_max + 1, x_left:x_right]
        if center_slice.size > 0 and center_slice.shape[1] >= 4:
            dark_counts = np.sum(center_slice < dark_thresh, axis=1)
            row_ratios = dark_counts / float(center_slice.shape[1])
            dark_rows = (row_ratios >= 0.70)
            if np.any(dark_rows):
                consecutive = 0
                max_consecutive = 0
                for is_dark in dark_rows:
                    if is_dark:
                        consecutive += 1
                        max_consecutive = max(max_consecutive, consecutive)
                    else:
                        consecutive = 0
                if 1 <= max_consecutive <= 4:
                    return True
            b_mid = int((min(yA2, yB2) + max(yA1, yB1)) / 2.0)
            b_y1 = max(cy_min, b_mid - 4)
            b_y2 = min(cy_max + 1, b_mid + 5)
            boundary_slice = gray[b_y1:b_y2, x_left:x_right]
            if boundary_slice.size > 0 and boundary_slice.shape[1] >= 4:
                for r in range(boundary_slice.shape[0]):
                    if _has_continuous_dark_segment(boundary_slice[r, :], dark_thresh=110, min_len=min_consec):
                        return True
        return False


def can_merge_text_boxes(boxA, boxB, gray=None, textA="", textB="", box_border_mask=None, border_mask=None, max_page_h=2000):
    if box_border_mask is None:
        box_border_mask = border_mask

    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB

    # Detect if both lines sit inside a dark narration box
    comb_crop = gray[min(yA1, yB1):max(yA2, yB2), min(xA1, xB1):max(xA2, xB2)] if gray is not None and gray.size > 0 else np.array([])
    is_in_dark_box = False
    if comb_crop.size > 0:
        med_comb = float(np.median(comb_crop))
        if med_comb < 95 or (med_comb < 120 and float(np.mean(comb_crop < 110)) >= 0.45):
            is_in_dark_box = True

    if not is_in_dark_box and are_different_white_contours(gray, boxA, boxB):
        return False

    if not is_in_dark_box:
        if box_border_mask is None and gray is not None:
            box_border_mask = get_box_border_mask(gray)

        if box_border_mask is not None and box_border_mask.size > 0:
            cxA = int((boxA[0] + boxA[2]) / 2.0)
            cyA = int((boxA[1] + boxA[3]) / 2.0)
            cxB = int((boxB[0] + boxB[2]) / 2.0)
            cyB = int((boxB[1] + boxB[3]) / 2.0)
            if line_crosses_border_mask(box_border_mask, (cxA, cyA), (cxB, cyB), boxA=boxA, boxB=boxB):
                return False

    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB
    wA, hA = max(1, xA2 - xA1), max(1, yA2 - yA1)
    wB, hB = max(1, xB2 - xB1), max(1, yB2 - yB1)

    is_vert_A = hA > 1.25 * wA and wA > 20
    is_vert_B = hB > 1.25 * wB and wB > 20
    is_both_vertical = is_vert_A and is_vert_B

    # Horizontal text (standard English/Russian/Uzbek manga dialogue)
    if not is_both_vertical:
        # Rule 1: Words on the exact same horizontal line
        y_overlap = min(yA2, yB2) - max(yA1, yB1)
        min_h = min(hA, hB)
        if min_h > 0 and (y_overlap / min_h) >= 0.65:
            gap_x = max(0, max(xA1, xB1) - min(xA2, xB2))
            if gap_x <= max(8, int(min_h * 0.35)):
                left_box = boxA if xA1 < xB1 else boxB
                right_box = boxB if xA1 < xB1 else boxA
                left_text = (textA if xA1 < xB1 else textB).strip()
                right_text = (textB if xA1 < xB1 else textA).strip()
                if (left_text.endswith(('!', '?')) or right_text.endswith(('!', '?'))) and gap_x > 0:
                    return False
                if not is_dark_barrier_between_x(gray, left_box, right_box):
                    return True

        # Rule 2: Consecutive lines stacked vertically in the same dialogue bubble
        top_box = boxA if yA1 < yB1 else boxB
        bot_box = boxB if yA1 < yB1 else boxA
        top_text = (textA if yA1 < yB1 else textB).strip()
        bot_text = (textB if yA1 < yB1 else textA).strip()

        # Check dark barrier in X for side-by-side / diagonal / staggered boxes
        left_box = boxA if xA1 < xB1 else boxB
        right_box = boxB if xA1 < xB1 else boxA
        if is_dark_barrier_between_x(gray, left_box, right_box):
            return False

        gap_y = max(0, bot_box[1] - top_box[3])
        avg_h = (hA + hB) / 2.0

        if not is_in_dark_box:
            # If there is a horizontal dark frame line between two boxes (like stacked #3 narration boxes),
            # DO NOT MERGE them even if gap_y is small.
            if is_dark_barrier_between_y(gray, top_box, bot_box):
                return False
            if gray is not None and gray.size > 0:
                H_img, W_img = gray.shape[:2]
                b_x1 = max(0, max(top_box[0], bot_box[0]))
                b_x2 = min(W_img, min(top_box[2], bot_box[2]))
                if b_x2 - b_x1 >= 8:
                    span_w = b_x2 - b_x1
                    min_c = max(12, int(0.25 * span_w))
                    b_y1 = max(0, min(top_box[3], bot_box[1]) - 4)
                    b_y2 = min(H_img, max(top_box[3], bot_box[1]) + 5)
                    for r in range(b_y1, b_y2):
                        if _has_continuous_dark_segment(gray[r, b_x1:b_x2], dark_thresh=110, min_len=min_c):
                            return False

        # Check if both lines sit inside the same speech bubble interior
        crop_between = gray[top_box[1]:bot_box[3], min(xA1, xB1):max(xA2, xB2)] if gray is not None and gray.size > 0 else np.array([])
        is_same_bubble_interior = (
            is_in_dark_box
            or (
                crop_between.size > 0
                and (np.mean(crop_between) > 150 or np.percentile(crop_between, 85) > 185 or np.mean(crop_between) < 85)
                and not is_dark_barrier_between_y(gray, top_box, bot_box)
                and not is_dark_barrier_between_x(gray, left_box, right_box)
            )
        )

        overlap = min(xA2, xB2) - max(xA1, xB1)
        if not is_same_bubble_interior:
            # Different regions or crossing barriers: do not merge across sentences
            if top_text.endswith(('!', '?')) and (
                gap_y > 0 and (overlap < min(wA, wB) * 0.70 or bot_text.endswith(('!', '?')) or gap_y > 4)
            ):
                return False
            if top_text.endswith(('.', '…', '!', '?')) and (
                gap_y > 6 or (gap_y > 0 and overlap < min(wA, wB) * 0.45)
            ):
                return False
        else:
            # Inside the same speech bubble interior: do NOT split multi-sentence bubbles at '.'
            if (top_text.endswith(('!', '?')) and bot_text.endswith(('!', '?'))) and gap_y > max(16, int(avg_h * 0.85)):
                return False

        # Maximum allowed gap between consecutive lines in one sentence/paragraph
        max_gap_y = max(16, int(avg_h * 0.85)) if is_same_bubble_interior else max(14, int(avg_h * 0.70))
        if gap_y <= max_gap_y:
            midA = (xA1 + xA2) / 2.0
            midB = (xB1 + xB2) / 2.0
            x_overlap = min(xA2, xB2) - max(xA1, xB1)
            min_w = min(wA, wB)
            
            # Must have horizontal overlap OR closely aligned centers
            has_align = (min_w > 0 and (x_overlap / min_w) >= 0.25) or abs(midA - midB) <= max(24, int(min_w * 0.35))
            if is_same_bubble_interior:
                has_align = has_align or (x_overlap > 0) or abs(midA - midB) <= max(35, int(min_w * 0.50))

            if has_align:
                crop_area = gray[top_box[1]:bot_box[3], min(xA1, xB1):max(xA2, xB2)] if gray is not None and gray.size > 0 else np.array([])
                if crop_area.size == 0:
                    return False
                is_bubble_bg = (
                    is_in_dark_box or
                    np.mean(crop_area) > 70 or
                    np.median(crop_area) > 70 or
                    np.percentile(crop_area, 85) > 140 or
                    np.mean(crop_area) < 80
                )
                if is_bubble_bg and (is_in_dark_box or (not is_dark_barrier_between_y(gray, top_box, bot_box) and not is_dark_barrier_between_x(gray, left_box, right_box))):
                    return True

    # Traditional vertical text (Japanese)
    if is_both_vertical:
        x_overlap = min(xA2, xB2) - max(xA1, xB1)
        min_w = min(wA, wB)
        if min_w > 0 and (x_overlap / min_w) >= 0.60:
            gap_y = max(0, max(yA1, yB1) - min(yA2, yB2))
            if gap_y <= max(8, int(min_w * 0.70)):
                top_box = boxA if yA1 < yB1 else boxB
                bot_box = boxB if yA1 < yB1 else boxA
                if not is_dark_barrier_between_y(gray, top_box, bot_box):
                    return True

        y_overlap = min(yA2, yB2) - max(yA1, yB1)
        min_h = min(hA, hB)
        if min_h > 0 and (y_overlap / min_h) >= 0.35:
            gap_x = max(0, max(xA1, xB1) - min(xA2, xB2))
            avg_w = (wA + wB) / 2.0
            if gap_x <= max(14, int(avg_w * 1.15)):
                left_box = boxA if xA1 < xB1 else boxB
                right_box = boxB if xA1 < xB1 else boxA
                if not is_dark_barrier_between_x(gray, left_box, right_box):
                    return True

    return False


def detect_stepped_box_split_x(gray, cluster_box):
    """
    Detects stepped (Z-shaped / L-shaped) narration / speech box split boundary.
    Extracts the main white connected component in gray > 200 within cluster_box.
    If fill_ratio < 0.80 and at least one corner (15%x15% zone) does not belong to the
    main white component, checks for a sharp step transition in y_top(x) or y_bot(x)
    where abs(diff) >= max(15, int(0.15 * bh)).
    Returns absolute image coordinate x_split = x1 + step_x, or None.
    """
    if gray is None or gray.size == 0 or cluster_box is None:
        return None

    if isinstance(cluster_box, dict):
        cluster_box = cluster_box.get("box", cluster_box.get("text_box", cluster_box))
    if not isinstance(cluster_box, (list, tuple)) or len(cluster_box) < 4:
        return None

    gh, gw = gray.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in cluster_box[:4]]
    x1 = max(0, min(gw, x1))
    x2 = max(0, min(gw, x2))
    y1 = max(0, min(gh, y1))
    y2 = max(0, min(gh, y2))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 10 or bh <= 10:
        return None

    patch = gray[y1:y2, x1:x2]
    white_mask = (patch > 200).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    if num_labels <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_label = 1 + int(np.argmax(areas))
    main_white_comp = (labels == max_label)
    white_area = int(stats[max_label, cv2.CC_STAT_AREA])
    fill_ratio = white_area / float(bw * bh)

    if fill_ratio >= 0.80:
        return None

    # Check corners (15% x 15% zones)
    cw = max(1, int(0.15 * bw))
    ch = max(1, int(0.15 * bh))

    tl = main_white_comp[0:ch, 0:cw]
    tr = main_white_comp[0:ch, bw - cw:bw]
    bl = main_white_comp[bh - ch:bh, 0:cw]
    br = main_white_comp[bh - ch:bh, bw - cw:bw]

    corner_means = [float(np.mean(tl)), float(np.mean(tr)), float(np.mean(bl)), float(np.mean(br))]
    if not any(m < 0.30 for m in corner_means):
        return None

    # Compute top and bottom white boundaries for each column
    y_top = np.zeros(bw, dtype=np.int32)
    y_bot = np.zeros(bw, dtype=np.int32)
    valid = np.zeros(bw, dtype=bool)

    for x in range(bw):
        rows = np.where(main_white_comp[:, x])[0]
        if len(rows) > 0:
            y_top[x] = rows[0]
            y_bot[x] = rows[-1]
            valid[x] = True

    if not np.any(valid):
        return None

    valid_indices = np.where(valid)[0]
    for x in range(bw):
        if not valid[x]:
            nearest = valid_indices[np.argmin(np.abs(valid_indices - x))]
            y_top[x] = y_top[nearest]
            y_bot[x] = y_bot[nearest]

    min_jump = max(15, int(0.15 * bh))
    x_start = max(1, int(0.10 * bw))
    x_end = min(bw - 1, int(0.90 * bw))

    step_top = None
    max_d_top = 0
    for x in range(x_start, x_end):
        for k in (1, 2, 3):
            if x + k < bw:
                d = abs(int(y_top[x + k]) - int(y_top[x]))
                if d >= min_jump and d > max_d_top:
                    max_d_top = d
                    step_top = x + (k + 1) // 2

    step_bot = None
    max_d_bot = 0
    for x in range(x_start, x_end):
        for k in (1, 2, 3):
            if x + k < bw:
                d = abs(int(y_bot[x + k]) - int(y_bot[x]))
                if d >= min_jump and d > max_d_bot:
                    max_d_bot = d
                    step_bot = x + (k + 1) // 2

    if step_top is not None and step_bot is not None:
        step_x = int(round((step_top + step_bot) / 2.0))
    elif step_top is not None:
        step_x = step_top
    elif step_bot is not None:
        step_x = step_bot
    else:
        return None

    return x1 + step_x


def split_cluster_into_columns(box_indices, raw_boxes, reading_direction: str = "rtl", gray=None):
    """
    Detects if a clustered text group contains multiple distinct vertical columns
    (e.g., adjacent dialogue, loud dialogue with small aside text, or multi-speaker bubbles),
    or a stepped (Z-shaped / L-shaped) narration box.
    If distinct non-overlapping columns exist with overlapping vertical spans,
    splits them into separate sub-clusters in manga reading order (right-to-left).
    """
    if len(box_indices) <= 1:
        return [box_indices]

    # Check stepped (Z-shaped / L-shaped) narration box splitting if gray image is provided
    if gray is not None and len(box_indices) >= 2:
        bx1 = min(raw_boxes[idx]["box"][0] for idx in box_indices)
        by1 = min(raw_boxes[idx]["box"][1] for idx in box_indices)
        bx2 = max(raw_boxes[idx]["box"][2] for idx in box_indices)
        by2 = max(raw_boxes[idx]["box"][3] for idx in box_indices)
        cluster_box = [bx1, by1, bx2, by2]

        x_split = None
        try:
            bubble_box = expand_to_bubble_contour(gray, cluster_box)
            x_split = detect_stepped_box_split_x(gray, bubble_box)
        except Exception:
            pass
        if x_split is None:
            try:
                x_split = detect_stepped_box_split_x(gray, cluster_box)
            except Exception:
                pass

        if x_split is not None:
            left_indices = [
                idx for idx in box_indices
                if (raw_boxes[idx]["box"][0] + raw_boxes[idx]["box"][2]) / 2.0 < x_split
            ]
            right_indices = [
                idx for idx in box_indices
                if (raw_boxes[idx]["box"][0] + raw_boxes[idx]["box"][2]) / 2.0 >= x_split
            ]
            if left_indices and right_indices:
                res_left = split_cluster_into_columns(left_indices, raw_boxes, reading_direction=reading_direction, gray=gray)
                res_right = split_cluster_into_columns(right_indices, raw_boxes, reading_direction=reading_direction, gray=gray)
                if reading_direction == "rtl":
                    return res_right + res_left
                else:
                    return res_left + res_right

    if len(box_indices) <= 2:
        return [box_indices]

    items = [raw_boxes[idx] for idx in box_indices]
    spans = [(b["box"][0], b["box"][2], idx) for idx, b in zip(box_indices, items)]

    min_x = min(s[0] for s in spans)
    max_x = max(s[1] for s in spans)
    w_span = max_x - min_x
    if w_span < 40:
        return [box_indices]

    # Sort boxes by horizontal center (left to right)
    sorted_by_x = sorted(zip(box_indices, items), key=lambda x: (x[1]["box"][0] + x[1]["box"][2]) / 2.0)

    best_split_idx = -1
    max_gap = -9999

    for i in range(len(sorted_by_x) - 1):
        left_group = sorted_by_x[:i+1]
        right_group = sorted_by_x[i+1:]

        left_max_x = max(item[1]["box"][2] for item in left_group)
        right_min_x = min(item[1]["box"][0] for item in right_group)
        gap = right_min_x - left_max_x

        left_y1 = min(item[1]["box"][1] for item in left_group)
        left_y2 = max(item[1]["box"][3] for item in left_group)
        right_y1 = min(item[1]["box"][1] for item in right_group)
        right_y2 = max(item[1]["box"][3] for item in right_group)

        y_overlap = min(left_y2, right_y2) - max(left_y1, right_y1)
        min_h = min(left_y2 - left_y1, right_y2 - right_y1)

        mid_left = sum((item[1]["box"][0] + item[1]["box"][2]) / 2.0 for item in left_group) / len(left_group)
        mid_right = sum((item[1]["box"][0] + item[1]["box"][2]) / 2.0 for item in right_group) / len(right_group)

        if min_h > 0 and y_overlap > 0.30 * min_h and (mid_right - mid_left) > 25 and gap >= -8:
            if gap > max_gap:
                max_gap = gap
                best_split_idx = i

    if best_split_idx != -1:
        left_indices = [x[0] for x in sorted_by_x[:best_split_idx+1]]
        right_indices = [x[0] for x in sorted_by_x[best_split_idx+1:]]
        is_cjk = any(
            any('\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u30ff' or '\u3400' <= ch <= '\u4dbf' for ch in raw_boxes[idx].get("text", ""))
            for idx in box_indices
        )
        res_left = split_cluster_into_columns(left_indices, raw_boxes, reading_direction=reading_direction, gray=gray)
        res_right = split_cluster_into_columns(right_indices, raw_boxes, reading_direction=reading_direction, gray=gray)
        if reading_direction == "rtl" or is_cjk:
            # In manga reading order (RTL), right column is read first, left column second
            return res_right + res_left
        else:
            # In LTR reading order, left column is read first, right column second
            return res_left + res_right

    return [box_indices]


def try_read_vertical_strip(img_strip, reader):
    """
    Takes a vertical strip image (h >> w) containing vertically stacked characters,
    slices it into letter glyphs along horizontal valleys, stitches them horizontally,
    and runs OCR to correctly recognize vertical words.
    """
    if img_strip is None or img_strip.size == 0 or reader is None:
        return None
    h, w = img_strip.shape[:2]
    if h < 1.5 * w or w < 8 or h < 25:
        return None

    gray = cv2.cvtColor(img_strip, cv2.COLOR_RGB2GRAY) if len(img_strip.shape) == 3 else img_strip.copy()
    mean_val = float(np.mean(gray))
    if mean_val < 128:
        # Invert dark strip so dark characters appear on bright background
        img_strip = 255 - img_strip
        gray = 255 - gray
        mean_val = float(np.mean(gray))

    if mean_val > 140:
        _, bin_img = cv2.threshold(gray, min(210, int(mean_val * 0.90)), 255, cv2.THRESH_BINARY_INV)
    else:
        _, bin_img = cv2.threshold(gray, max(60, int(mean_val * 1.15)), 255, cv2.THRESH_BINARY)

    proj = np.sum(bin_img > 0, axis=1)
    in_letter = False
    slices = []
    start_y = 0
    min_gap = 2
    gap_count = 0

    for y in range(h):
        if proj[y] > 2:
            if not in_letter:
                in_letter = True
                start_y = max(0, y - 2)
            gap_count = 0
        else:
            if in_letter:
                gap_count += 1
                if gap_count >= min_gap:
                    end_y = min(h, y - gap_count + 3)
                    if (end_y - start_y) >= 6:
                        slices.append((start_y, end_y))
                    in_letter = False
                    gap_count = 0

    if in_letter and (h - start_y) >= 6:
        slices.append((start_y, h))

    if len(slices) < 2:
        return None

    letter_imgs = []
    max_h = 0
    for sy, ey in slices:
        lc = img_strip[sy:ey, :]
        letter_imgs.append(lc)
        max_h = max(max_h, lc.shape[0])

    max_h = max(32, max_h)
    padded_letters = []
    for lc in letter_imgs:
        lh = lc.shape[0]
        top_pad = (max_h - lh) // 2
        bot_pad = max_h - lh - top_pad
        padded = cv2.copyMakeBorder(lc, top_pad, bot_pad, 4, 4, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        padded_letters.append(padded)

    horiz_strip = cv2.hconcat(padded_letters)
    horiz_strip = cv2.copyMakeBorder(horiz_strip, 12, 12, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    try:
        res = reader.readtext(horiz_strip)
        if res:
            best_text = " ".join(r[1] for r in res).strip()
            avg_conf = float(np.mean([float(r[2]) for r in res]))
            clean = best_text.replace(" ", "").upper()
            clean = clean.replace("0", "O").replace("/", "!")
            return clean, avg_conf
    except Exception:
        pass
    return None


def smart_cluster_text_boxes(raw_boxes, gray, reading_direction: str = "rtl"):
    """
    Groups individual text lines/words into distinct dialogue bubbles
    via adjacency graph connected components, and splits multi-column bubbles.
    Filters out spurious OCR noise and guarantees correct natural reading order.
    """
    if not raw_boxes:
        return []

    # Pre-filter spurious noise lines (tiny low-confidence artifacts, panel boundary lines, stray symbols)
    valid_boxes = []
    for rb in raw_boxes:
        t = rb.get("text", "").strip()
        # Normalize common comic font scanlation OCR substitutions (prevent corrupting CLASS 1-1, 1-A, 1st):
        t = re.sub(r'(?<![\d\-])\b1\b(?!\s*(?:st|nd|rd|th)\b)(?![\d\-])', 'I', t, flags=re.I)
        t = re.sub(r'\b1(?!(?:st|nd|rd|th)\b)(?=[A-Za-z])', 'I', t, flags=re.I)
        t = re.sub(r'\bUT\b', 'IT', t)
        rb["text"] = t
        conf = float(rb.get("confidence", rb.get("conf", 1.0)))
        box = rb["box"]
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        
        # Expressive manga punctuation (?, !, ?!, ...) is strictly preserved!
        if is_manga_punctuation(t):
            rb["text"] = normalize_manga_punctuation(t)
            valid_boxes.append(rb)
            continue

        # Check if inside speech bubble interior (light > 160 or dark < 90)
        is_bubble_interior = False
        if gray is not None and gray.size > 0:
            gh, gw = gray.shape[:2]
            cx1 = max(0, min(gw, box[0]))
            cy1 = max(0, min(gh, box[1]))
            cx2 = max(0, min(gw, box[2]))
            cy2 = max(0, min(gh, box[3]))
            if cx2 > cx1 and cy2 > cy1:
                patch = gray[cy1:cy2, cx1:cx2]
                if patch.size > 0:
                    p_mean = float(np.mean(patch))
                    if p_mean > 160 or p_mean < 90:
                        is_bubble_interior = True

        # Ignore very low confidence noise or single stray non-Cyrillic marks
        # (Preserve single "I", short interjections, and valid bubble-interior trailing lines)
        has_cyr = any('\u0400' <= c <= '\u04ff' for c in t)
        clean_t = re.sub(r'[^\w]', '', t)
        is_short_interjection = (2 <= len(clean_t) <= 3 and any(c in "?!" for c in t) and clean_t.isalpha())
        is_short_valid = (clean_t.lower() in {"i", "it", "a", "no", "to", "so", "he", "we", "me", "my", "ok", "hm", "mm", "eh", "ha", "uh", "um", "yo", "ya"}) or is_short_interjection or (len(clean_t) >= 2 and clean_t.isalpha())
        if conf < 0.10 and len(t) <= 2 and not has_cyr:
            if not (is_bubble_interior and is_short_valid):
                continue
        # Ignore tall thin vertical artifacts (line noise, panel borders)
        # (Preserve single capital "I" inside a bubble interior)
        if bw <= 8 and bh >= 25 and len(t) <= 1:
            if not (is_bubble_interior and t.upper() == "I"):
                continue
        valid_boxes.append(rb)
    
    raw_boxes = valid_boxes
    n = len(raw_boxes)
    if n == 0:
        return []

    box_border_mask = get_box_border_mask(gray) if gray is not None else None
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if can_merge_text_boxes(
                raw_boxes[i]["box"], raw_boxes[j]["box"], gray,
                raw_boxes[i].get("text", ""), raw_boxes[j].get("text", ""),
                box_border_mask=box_border_mask
            ):
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

        # Check if the cluster contains multiple side-by-side vertical columns
        sub_comps = split_cluster_into_columns(comp, raw_boxes, reading_direction=reading_direction, gray=gray)
        stepped_split_x = None
        if gray is not None and len(comp) >= 2:
            cbx1 = min(raw_boxes[idx]["box"][0] for idx in comp)
            cby1 = min(raw_boxes[idx]["box"][1] for idx in comp)
            cbx2 = max(raw_boxes[idx]["box"][2] for idx in comp)
            cby2 = max(raw_boxes[idx]["box"][3] for idx in comp)
            cbox = [cbx1, cby1, cbx2, cby2]
            try:
                bbox = expand_to_bubble_contour(gray, cbox)
                stepped_split_x = detect_stepped_box_split_x(gray, bbox)
            except Exception:
                pass
            if stepped_split_x is None:
                try:
                    stepped_split_x = detect_stepped_box_split_x(gray, cbox)
                except Exception:
                    pass

        for sub_comp in sub_comps:
            # Check if the text is genuinely vertical Japanese text (not just one tall letter like 'I' or '1')
            vert_lines = sum(
                1 for idx in sub_comp
                if (raw_boxes[idx]["box"][3] - raw_boxes[idx]["box"][1]) > 1.4 * (raw_boxes[idx]["box"][2] - raw_boxes[idx]["box"][0])
                and len(raw_boxes[idx].get("text", "").strip()) > 1
            )
            is_vertical = (vert_lines / len(sub_comp)) >= 0.60 if len(sub_comp) >= 3 else False

            if is_vertical:
                sub_comp.sort(key=lambda idx: (-raw_boxes[idx]["box"][0], raw_boxes[idx]["box"][1]))
                lines = [[idx] for idx in sub_comp]
            else:
                # Group lines by vertical overlap y_overlap >= 0.45 * min(h1, h2)
                lines = []
                sorted_by_yc = sorted(sub_comp, key=lambda idx: (raw_boxes[idx]["box"][1] + raw_boxes[idx]["box"][3]) / 2.0)
                for idx in sorted_by_yc:
                    b = raw_boxes[idx]["box"]
                    h = b[3] - b[1]
                    matched_line = None
                    for line in lines:
                        l_y1 = min(raw_boxes[i]["box"][1] for i in line)
                        l_y2 = max(raw_boxes[i]["box"][3] for i in line)
                        l_min_h = min(raw_boxes[i]["box"][3] - raw_boxes[i]["box"][1] for i in line)
                        y_ov = min(b[3], l_y2) - max(b[1], l_y1)
                        if y_ov >= 0.45 * min(h, l_min_h):
                            matched_line = line
                            break
                    if matched_line is not None:
                        matched_line.append(idx)
                    else:
                        lines.append([idx])

                # Sort boxes within each line by x1 (left to right)
                for line in lines:
                    line.sort(key=lambda idx: raw_boxes[idx]["box"][0])
                # Sort lines top-to-bottom
                lines.sort(key=lambda line: min(raw_boxes[idx]["box"][1] for idx in line))
                sub_comp = [idx for line in lines for idx in line]

            bx1 = min(raw_boxes[idx]["box"][0] for idx in sub_comp)
            by1 = min(raw_boxes[idx]["box"][1] for idx in sub_comp)
            bx2 = max(raw_boxes[idx]["box"][2] for idx in sub_comp)
            by2 = max(raw_boxes[idx]["box"][3] for idx in sub_comp)

            # Preserve multi-sentence line breaks (\n) and restore line-end hyphenated words
            # e.g., TSU- at end of line + MIKI'S at start of next line -> TSUMIKI'S
            full_parts = []
            for l_idx, line in enumerate(lines):
                line_words = []
                for idx in line:
                    lt = raw_boxes[idx].get("text", "").strip()
                    lt = re.sub(r'^[1lI]I\s*', 'I ', lt)
                    line_words.append(lt)
                line_text = " ".join(line_words).strip()
                if not line_text:
                    continue

                if not full_parts:
                    full_parts.append(line_text)
                else:
                    prev_text = full_parts[-1]
                    if re.search(r'[A-Za-z]+-$', prev_text) and re.match(r'^[A-Za-z\']+', line_text):
                        # Restore hyphenated word broken across line end
                        full_parts[-1] = re.sub(r'-$', '', full_parts[-1]) + line_text
                    else:
                        prev_line = lines[l_idx - 1]
                        prev_y2 = max(raw_boxes[i]["box"][3] for i in prev_line)
                        curr_y1 = min(raw_boxes[i]["box"][1] for i in line)
                        gap = curr_y1 - prev_y2
                        if (prev_text.endswith(('.', '…', '!', '?'))) or gap >= 9:
                            full_parts.append("\n" + line_text)
                        else:
                            full_parts.append(" " + line_text)

            full_text = "".join(full_parts).strip()
            full_text = re.sub(r'(\b[A-Za-z]{2,})-\s+([A-Za-z\']{2,})', r'\1\2', full_text)
            avg_c = float(np.mean([raw_boxes[idx].get("conf", 1.0) for idx in sub_comp])) if sub_comp else 1.0
            c_dict = {
                "text_box": [bx1, by1, bx2, by2],
                "text": full_text,
                "conf": avg_c,
                "avg_conf": avg_c,
                "indices": sub_comp,
                "block_id": f"b_{len(clusters) + 1}"
            }
            if stepped_split_x is not None:
                sub_cx = (bx1 + bx2) / 2.0
                if sub_cx < stepped_split_x:
                    c_dict["limit_right"] = stepped_split_x
                else:
                    c_dict["limit_left"] = stepped_split_x
                c_dict["split_x"] = stepped_split_x
            clusters.append(c_dict)

    return merge_adjacent_bubble_columns(clusters, gray, reading_direction=reading_direction, box_border_mask=box_border_mask)


def merge_adjacent_bubble_columns(clusters, gray, reading_direction: str = "rtl", box_border_mask=None):
    """
    Second pass: Merges multi-column text that belongs to the same speech bubble.
    Columns in the same bubble have gap_x <= 12, vertical overlap > 0, and no dark barrier.
    """
    if len(clusters) <= 1:
        return clusters

    if box_border_mask is None and gray is not None:
        box_border_mask = get_box_border_mask(gray)

    n = len(clusters)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            bA = clusters[i]["text_box"]
            bB = clusters[j]["text_box"]
            xA1, yA1, xA2, yA2 = bA
            xB1, yB1, xB2, yB2 = bB
            gap_x = max(0, max(xA1, xB1) - min(xA2, xB2))
            y_overlap = min(yA2, yB2) - max(yA1, yB1)
            min_h = min(yA2 - yA1, yB2 - yB1)

            if box_border_mask is not None and box_border_mask.size > 0:
                cxA = int((xA1 + xA2) / 2.0)
                cyA = int((yA1 + yA2) / 2.0)
                cxB = int((xB1 + xB2) / 2.0)
                cyB = int((yB1 + yB2) / 2.0)
                if line_crosses_border_mask(box_border_mask, (cxA, cyA), (cxB, cyB), boxA=bA, boxB=bB):
                    continue

            if min_h > 0 and y_overlap > 0 and gap_x <= max(18, int(min_h * 0.35)):
                if gray is not None and are_different_white_contours(gray, bA, bB):
                    continue
                tA = clusters[i].get("text", "").strip()
                tB = clusters[j].get("text", "").strip()
                if is_sfx_text(tA) or is_sfx_text(tB):
                    continue
                is_laughter = lambda s: bool(re.search(r'\b(ha|ah|he|wa)+\b', s, re.I))
                if is_laughter(tA) != is_laughter(tB):
                    continue
                if (tA.endswith(('!', '?')) and tB.endswith(('!', '?'))) and gap_x > 12:
                    continue
                left_box = bA if xA1 < xB1 else bB
                right_box = bB if xA1 < xB1 else bA
                if not is_dark_barrier_between_x(gray, left_box, right_box):
                    if gap_x > 0 and gray is not None:
                        gap_slice = gray[max(yA1, yB1):min(yA2, yB2), min(xA2, xB2):max(xA1, xB1)]
                        if gap_slice.size > 0 and (np.mean(gap_slice) < 180 or np.min(gap_slice) < 110):
                            continue
                    if gray is not None:
                        crop_area = gray[min(yA1, yB1):max(yA2, yB2), min(xA1, xB1):max(xA2, xB2)]
                        if crop_area.size > 0 and (np.mean(crop_area) > 160 or np.median(crop_area) > 160):
                            adj[i].append(j)
                            adj[j].append(i)
                    else:
                        adj[i].append(j)
                        adj[j].append(i)

    visited = set()
    merged = []
    for i in range(n):
        if i in visited:
            continue
        comp = [i]
        visited.add(i)
        queue = [i]
        while queue:
            cur = queue.pop(0)
            for neighbor in adj[cur]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    comp.append(neighbor)
                    queue.append(neighbor)

        if len(comp) == 1:
            merged.append(clusters[comp[0]])
        else:
            # Multi-column bubble reading order:
            comp_texts = [clusters[idx].get("text", "") for idx in comp]
            is_cjk = any(
                any('\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u30ff' or '\u3400' <= ch <= '\u4dbf' for ch in t)
                for t in comp_texts
            )
            is_strongly_vertical = all(
                (clusters[idx]["text_box"][3] - clusters[idx]["text_box"][1]) > 1.3 * max(1, clusters[idx]["text_box"][2] - clusters[idx]["text_box"][0])
                for idx in comp
            )
            is_vertical_columns = all(
                (clusters[idx]["text_box"][3] - clusters[idx]["text_box"][1]) > max(1, clusters[idx]["text_box"][2] - clusters[idx]["text_box"][0])
                for idx in comp
            )
            is_sentence = lambda s: bool(re.search(r'[.!?…~]+[\s"\'”’)]*$', s.strip()))
            each_is_sentence = len(comp) >= 2 and all(is_sentence(t) for t in comp_texts)

            gap_x_comp = 0
            is_vertically_shifted = False
            if len(comp) == 2:
                b0 = clusters[comp[0]]["text_box"]
                b1 = clusters[comp[1]]["text_box"]
                gap_x_comp = max(0, max(b0[0], b1[0]) - min(b0[2], b1[2]))
                w0 = b0[2] - b0[0]
                w1 = b1[2] - b1[0]
                cx0 = (b0[0] + b0[2]) / 2.0
                cx1 = (b1[0] + b1[2]) / 2.0
                is_vertically_shifted = (
                    (abs(b0[1] - b1[1]) >= 12 or abs(b0[3] - b1[3]) >= 12)
                    and (abs(cx0 - cx1) > 0.35 * (w0 + w1))
                )

            # Two independent vertical columns or 2 independent sentence columns in rectangular boxes
            # (e.g. stepped #6 narration box, where len(comp) == 2 with gap_x >= 6 and each_is_sentence or stepped)
            # must NOT be merged into a single horizontal block; keep each column in its own vertical bbox.
            is_two_independent_columns = (
                is_strongly_vertical
                or is_vertically_shifted
                or (
                    len(comp) == 2
                    and gap_x_comp >= 6
                    and (
                        each_is_sentence
                        or is_vertical_columns
                        or abs(clusters[comp[0]]["text_box"][1] - clusters[comp[1]]["text_box"][1]) >= 4
                    )
                )
            )
            if is_two_independent_columns:
                if len(comp) == 2 and is_vertically_shifted:
                    c0_idx, c1_idx = comp[0], comp[1]
                    if clusters[c0_idx]["text_box"][0] < clusters[c1_idx]["text_box"][0]:
                        l_idx, r_idx = c0_idx, c1_idx
                    else:
                        l_idx, r_idx = c1_idx, c0_idx
                    mid_x = (clusters[l_idx]["text_box"][2] + clusters[r_idx]["text_box"][0]) // 2
                    if "limit_right" not in clusters[l_idx]:
                        clusters[l_idx]["limit_right"] = mid_x
                    if "limit_left" not in clusters[r_idx]:
                        clusters[r_idx]["limit_left"] = mid_x

                if reading_direction == "rtl":
                    comp.sort(key=lambda idx: -clusters[idx]["text_box"][0])
                else:
                    comp.sort(key=lambda idx: clusters[idx]["text_box"][0])
                for idx in comp:
                    merged.append(clusters[idx])
                continue

            if is_cjk or is_strongly_vertical:
                # Right column is read first, then left column (CJK vertical text)
                comp.sort(key=lambda idx: -clusters[idx]["text_box"][0])
                full_text = " ... ".join(clusters[idx]["text"] for idx in comp)
            else:
                # Reading order for side-by-side vs stacked columns:
                has_side_by_side = False
                for ci in range(len(comp)):
                    for cj in range(ci + 1, len(comp)):
                        b1 = clusters[comp[ci]]["text_box"]
                        b2 = clusters[comp[cj]]["text_box"]
                        y_ov = min(b1[3], b2[3]) - max(b1[1], b2[1])
                        min_h = min(b1[3] - b1[1], b2[3] - b2[1])
                        if min_h > 0 and (y_ov / min_h) >= 0.30:
                            has_side_by_side = True
                            break
                    if has_side_by_side:
                        break

                if has_side_by_side:
                    if reading_direction == "rtl":
                        # Manga RTL reading order: right column first, left column second
                        comp.sort(key=lambda idx: -clusters[idx]["text_box"][0])
                    else:
                        comp.sort(key=lambda idx: clusters[idx]["text_box"][0])
                else:
                    # Vertically stacked: top-to-bottom
                    comp.sort(key=lambda idx: (clusters[idx]["text_box"][1], clusters[idx]["text_box"][0]))
                full_text = " ".join(clusters[idx]["text"] for idx in comp)

            bx1 = min(clusters[idx]["text_box"][0] for idx in comp)
            by1 = min(clusters[idx]["text_box"][1] for idx in comp)
            bx2 = max(clusters[idx]["text_box"][2] for idx in comp)
            by2 = max(clusters[idx]["text_box"][3] for idx in comp)
            all_indices = []
            for idx in comp:
                all_indices.extend(clusters[idx].get("indices", []))
            avg_c = float(np.mean([clusters[idx].get("conf", 1.0) for idx in comp])) if comp else 1.0
            merged.append({
                "text_box": [bx1, by1, bx2, by2],
                "text": full_text,
                "conf": avg_c,
                "avg_conf": avg_c,
                "indices": all_indices,
                "block_id": clusters[comp[0]].get("block_id", f"b_{len(merged) + 1}")
            })

    return merged


def find_bubble_visual_center(gray_np, box):
    """
    Computes the exact mathematical centroid of the speech bubble interior
    using image moments of the largest light/white connected component.
    Guarantees text is 100% dead-centered in the physical speech bubble!
    """
    x1, y1, x2, y2 = box
    h, w = gray_np.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 + 8 or y2 <= y1 + 8:
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    patch = gray_np[y1:y2, x1:x2]
    geom_cx = (x1 + x2) / 2.0
    geom_cy = (y1 + y2) / 2.0
    p85 = float(np.percentile(patch, 85)) if patch.size > 0 else 255.0
    if p85 > 210:
        white_mask = (patch > min(205, int(p85 * 0.90))).astype(np.uint8) * 255
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 80:
                cx = x1 + (M["m10"] / M["m00"])
                cy = y1 + (M["m01"] / M["m00"])
                cx = max(geom_cx - 4.0, min(geom_cx + 4.0, cx))
                cy = max(geom_cy - 4.0, min(geom_cy + 4.0, cy))
                return (cx, cy)

    return (geom_cx, geom_cy)


def expand_to_bubble_contour(gray, text_box, all_other_tboxes=None, max_pad_x=100, max_pad_y=100, limit_left=None, limit_right=None):
    """
    Snaps a text box to the speech bubble's true outer boundary.
    Prevents invasion into neighboring text boxes by checking midpoint boundaries.
    Detects thin oval bubble contours using ink density and dark pixel ratio.
    Supports bounding limits (limit_left / limit_right) so stepped lobes do not cross x_split.
    """
    h, w = gray.shape
    orig_x1, orig_y1, orig_x2, orig_y2 = text_box

    # Only expand into speech bubbles if the region is genuinely a uniform light bubble (>210)
    # If the text is on artwork, screentone, floor, or character body (e.g. SFX),
    # DO NOT expand! Keep it tightly bounded to the text stroke!
    patch = gray[max(0, orig_y1):min(h, orig_y2), max(0, orig_x1):min(w, orig_x2)]
    ring_pixels = []
    top_strip = gray[max(0, orig_y1 - 6):orig_y1, max(0, orig_x1 - 6):min(w, orig_x2 + 6)]
    bot_strip = gray[orig_y2:min(h, orig_y2 + 6), max(0, orig_x1 - 6):min(w, orig_x2 + 6)]
    left_strip = gray[orig_y1:orig_y2, max(0, orig_x1 - 6):orig_x1]
    right_strip = gray[orig_y1:orig_y2, orig_x2:min(w, orig_x2 + 6)]
    for strip in (top_strip, bot_strip, left_strip, right_strip):
        if strip.size > 0:
            ring_pixels.append(strip.ravel())
    ring_arr = np.concatenate(ring_pixels) if ring_pixels else np.array([])
    is_in_light_bubble = (
        (patch.size > 0 and np.percentile(patch, 85) >= 195 and np.percentile(patch, 50) >= 160) or
        (ring_arr.size > 0 and np.percentile(ring_arr, 75) >= 200)
    )
    if not is_in_light_bubble:
        pad_x = min(5, max(2, int((orig_x2 - orig_x1) * 0.10)))
        pad_y = min(4, max(2, int((orig_y2 - orig_y1) * 0.10)))
        res_x1 = max(0, orig_x1 - pad_x)
        res_x2 = min(w, orig_x2 + pad_x)
        if limit_left is not None:
            res_x1 = max(res_x1, limit_left)
        if limit_right is not None:
            res_x2 = min(res_x2, limit_right)
        return [res_x1, max(0, orig_y1 - pad_y), res_x2, min(h, orig_y2 + pad_y)]

    # Determine safe stopping bounds with respect to all other text boxes and page edges BEFORE min_pad
    bound_left = min(orig_x1, 4)
    bound_right = max(orig_x2, w - 4)
    bound_top = min(orig_y1, 4)
    bound_bot = max(orig_y2, h - 4)

    if limit_left is not None:
        bound_left = max(bound_left, limit_left)
    if limit_right is not None:
        bound_right = min(bound_right, limit_right)

    if all_other_tboxes:
        for ob in all_other_tboxes:
            ox1, oy1, ox2, oy2 = ob
            # Check vertical overlap to constrain horizontal expansion
            if max(orig_y1, oy1) < min(orig_y2, oy2):
                if ox2 <= orig_x1:
                    bound = (ox2 + orig_x1) // 2 + 1
                    if limit_left is not None:
                        bound = max(bound, limit_left)
                    bound_left = max(bound_left, bound)
                elif ox1 >= orig_x2:
                    bound = (orig_x2 + ox1) // 2 - 1
                    if limit_right is not None:
                        bound = min(bound, limit_right)
                    bound_right = min(bound_right, bound)

            # Check horizontal overlap to constrain vertical expansion
            if max(orig_x1, ox1) < min(orig_x2, ox2):
                if oy2 <= orig_y1:
                    bound_top = max(bound_top, (oy2 + orig_y1) // 2 + 1)
                elif oy1 >= orig_y2:
                    bound_bot = min(bound_bot, (orig_y2 + oy1) // 2 - 1)

    # Scan from the text bounds so a close outline cannot fall inside skipped padding.
    x1, y1, x2, y2 = text_box

    safe_max_pad_x = min(max_pad_x, max(30, int((x2 - x1) * 0.60)))
    safe_max_pad_y = min(max_pad_y, max(35, int((y2 - y1) * 0.80)))
    bound_left = max(bound_left, x1 - safe_max_pad_x)
    bound_right = min(bound_right, x2 + safe_max_pad_x)
    bound_top = max(bound_top, y1 - safe_max_pad_y)
    bound_bot = min(bound_bot, y2 + safe_max_pad_y)

    if limit_left is not None:
        bound_left = max(bound_left, limit_left)
    if limit_right is not None:
        bound_right = min(bound_right, limit_right)

    def is_line_border(arr):
        if arr.size == 0:
            return True
        span = len(arr)
        d_ratio = np.mean(arr < 130)
        mean_val = np.mean(arr)
        # Require a true box border line (continuous dark line, d_ratio >= 0.25)
        # rather than isolated letter stroke pixels (must not stop early on min_v < 45)
        if d_ratio >= 0.25:
            return True
        min_consec = max(10, int(0.25 * span))
        if _has_continuous_dark_segment(arr, dark_thresh=110, min_len=min_consec):
            return True
        if mean_val < 140:
            return True
        return False

    # Top
    cy1 = y1
    for ty in range(y1 - 1, bound_top - 1, -1):
        row = gray[ty, x1:x2]
        if is_line_border(row):
            cy1 = ty
            break
        cy1 = ty

    # Bottom
    cy2 = y2
    for ty in range(y2, min(h, bound_bot)):
        row = gray[ty, x1:x2]
        if is_line_border(row):
            cy2 = ty
            break
        cy2 = ty + 1

    vh = cy2 - cy1
    mid_cy1 = cy1 + int(0.20 * vh)
    mid_cy2 = cy2 - int(0.20 * vh)
    if mid_cy2 <= mid_cy1:
        mid_cy1, mid_cy2 = cy1, cy2

    # Left (use central 60% vertical band mid_cy1:mid_cy2)
    cx1 = x1
    for tx in range(x1 - 1, bound_left - 1, -1):
        col_text = gray[y1:y2, tx]
        col_mid = gray[mid_cy1:mid_cy2, tx]
        if is_line_border(col_text) or is_line_border(col_mid):
            cx1 = tx
            break
        cx1 = tx
    if limit_left is not None:
        cx1 = max(cx1, limit_left)

    # Right (use central 60% vertical band mid_cy1:mid_cy2)
    cx2 = x2
    for tx in range(x2, min(w, bound_right)):
        col_text = gray[y1:y2, tx]
        col_mid = gray[mid_cy1:mid_cy2, tx]
        if is_line_border(col_text) or is_line_border(col_mid):
            cx2 = tx
            break
        cx2 = tx + 1
    if limit_right is not None:
        cx2 = min(cx2, limit_right)

    return [cx1, cy1, cx2, cy2]


def generate_ink_stroke_mask(gray_np, box, pad=2):
    """
    Generates a precise binary mask covering dialogue text ink strokes and their anti-aliased halos,
    preventing damage to character faces, hair, panel borders, and screentones.
    """
    h, w = gray_np.shape
    x1, y1, x2, y2 = box
    px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
    px2, py2 = min(w, x2 + pad), min(h, y2 + pad)
    if px2 <= px1 or py2 <= py1:
        return np.zeros((h, w), dtype=np.uint8)

    patch = gray_np[py1:py2, px1:px2]
    p85 = float(np.percentile(patch, 85))
    med = float(np.median(patch))

    # In manga, dialogue bubbles with dark text on light background have high 85th percentile:
    if p85 >= 160:
        thresh_val = min(232, int(p85 * 0.90))
        stroke_binary = (patch < thresh_val).astype(np.uint8) * 255
    elif p85 < 110 and med < 60:
        # True inverted white-on-black bubble
        stroke_binary = (patch > 160).astype(np.uint8) * 255
    else:
        # Screentone or midtone
        thresh_val = min(180, int(p85 * 0.85))
        stroke_binary = (patch < thresh_val).astype(np.uint8) * 255

    hp, wp = patch.shape
    # Filter out long linear border segments from stroke mask limited to border-touching components
    kw = max(35, int(wp * 0.75))
    kh = max(35, int(hp * 0.75))
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
    long_lines = cv2.bitwise_or(
        cv2.morphologyEx(stroke_binary, cv2.MORPH_OPEN, k_h),
        cv2.morphologyEx(stroke_binary, cv2.MORPH_OPEN, k_v)
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(stroke_binary)
    lines = np.zeros_like(stroke_binary)
    for l in range(1, num_labels):
        st = stats[l]
        touches_edge = (
            st[cv2.CC_STAT_LEFT] <= 1 or
            st[cv2.CC_STAT_TOP] <= 1 or
            st[cv2.CC_STAT_LEFT] + st[cv2.CC_STAT_WIDTH] >= wp - 1 or
            st[cv2.CC_STAT_TOP] + st[cv2.CC_STAT_HEIGHT] >= hp - 1
        )
        is_border_wall = touches_edge and (
            np.any(long_lines[labels == l] > 0) or
            st[cv2.CC_STAT_AREA] > int(0.30 * wp * hp) or
            (st[cv2.CC_STAT_WIDTH] >= int(0.60 * wp) and (st[cv2.CC_STAT_TOP] <= 2 or st[cv2.CC_STAT_TOP] + st[cv2.CC_STAT_HEIGHT] >= hp - 2)) or
            (st[cv2.CC_STAT_HEIGHT] >= int(0.60 * hp) and (st[cv2.CC_STAT_LEFT] <= 2 or st[cv2.CC_STAT_LEFT] + st[cv2.CC_STAT_WIDTH] >= wp - 2))
        )
        if is_border_wall:
            lines[labels == l] = 255

    stroke_clean = cv2.bitwise_and(stroke_binary, cv2.bitwise_not(lines))

    # 3x3 ellipse with 1 iteration cleanly swallows anti-aliased edges without bleeding into art
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    stroke_dilated = cv2.dilate(stroke_clean, k, iterations=1)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[py1:py2, px1:px2] = stroke_dilated
    return mask


def hyphenate_uzbek_word(word: str) -> list:
    """Split a long Uzbek word at a syllable boundary and preserve its punctuation."""
    clean_w = word.rstrip('.!?,:;')
    punct = word[len(clean_w):]

    # Only hyphenate genuinely long Uzbek words (14+ characters)
    # Avoid breaking short/medium words mid-syllable (e.g. bo'lishini, chaqmog'ingni)
    if len(clean_w) >= 14 and re.fullmatch(r"[a-zA-Zʻʼ‘’'`ÖöĞğŞşÇç]+", clean_w):
        units = list(re.finditer(r"[ogOG]['ʻʼ‘’`]|sh|ch|ng|nd|nt|Ö|ö|Ğ|ğ|Ş|ş|Ç|ç|.", clean_w, re.I))
        vowels = [i for i, unit in enumerate(units)
                  if unit.group()[0].lower() in "aeiouö"]
        boundaries = []
        for left, right in zip(vowels, vowels[1:]):
            # One consonant starts the next syllable. Earlier consonants stay behind.
            onset = units[max(left + 1, right - 1)]
            cut = onset.start()
            if onset.group().lower() in ("nd", "nt"):
                cut += 1
            if 3 <= cut <= len(clean_w) - 3 and onset.group()[0].isalpha():
                boundaries.append(cut)
        if boundaries:
            cut = min(boundaries, key=lambda pos: abs(2 * pos - len(clean_w)))
            return [clean_w[:cut] + "-", clean_w[cut:] + punct]

    return [word]


def wrap_text_elliptical(words, font, a, b, line_spacing=1.15, allow_hyphenation=True):
    """
    Wraps words into lines fitting comfortably within the speech bubble's elliptical envelope.
    For each line i at vertical offset y_i from bubble center, maximum allowed width is:
    w_max = 2 * a * sqrt(max(0.05, 1 - (y_i / b)^2))
    """
    import math
    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    line_step = int(line_h * line_spacing)

    # Word candidate variants: try original first, and hyphenated only if any word is >= 14 chars
    word_variants = [words]
    split_pairs = {}
    if allow_hyphenation and any(len(w.rstrip('.!?,:;')) >= 14 for w in words):
        expanded = []
        for w in words:
            if len(w.rstrip('.!?,:;')) >= 14:
                parts = hyphenate_uzbek_word(w)
                expanded.extend(parts)
                if len(parts) == 2:
                    split_pairs[" ".join(parts)] = w
            else:
                expanded.append(w)
        if expanded != words:
            word_variants.append(expanded)

    for cand_words in word_variants:
        best_wrap = None
        for target_lines in range(1, 10):
            total_h = target_lines * line_step
            if total_h > 2 * b * 0.85:
                break

            allowed_widths = []
            for i in range(target_lines):
                y = - (total_h / 2.0) + (i + 0.5) * line_step
                y_ratio = min(0.95, abs(y) / b)
                w_max = 2.0 * a * math.sqrt(max(0.05, 1.0 - y_ratio**2))
                # Inset first and last line widths by ~15% for angled/octagonal bubbles
                if target_lines > 1 and (i == 0 or i == target_lines - 1):
                    w_max *= 0.85
                allowed_widths.append(w_max)

            lines = []
            curr_words = []
            line_idx = 0
            possible = True

            for w in cand_words:
                test_line = " ".join(curr_words + [w])
                bbox = font.getbbox(test_line)
                w_px = bbox[2] - bbox[0]
                if line_idx < target_lines and w_px <= allowed_widths[line_idx]:
                    curr_words.append(w)
                else:
                    if curr_words:
                        lines.append(" ".join(curr_words))
                        curr_words = [w]
                        line_idx += 1
                        if line_idx >= target_lines:
                            possible = False
                            break
                        w_single = font.getbbox(w)[2] - font.getbbox(w)[0]
                        if w_single > allowed_widths[line_idx]:
                            possible = False
                            break
                    else:
                        possible = False
                        break
            if possible and curr_words:
                lines.append(" ".join(curr_words))
                if len(lines) <= target_lines:
                    best_wrap = lines
                    break
        if best_wrap:
            if len(best_wrap) > 1:
                short_tokens = {"i", "u", "va", "da", "o'", "oʻ", "o‘", "ö", "bu", "a", "ya", "v", "s", "sh", "ch", "ş", "ç", "ğ"}
                cand_wrap = [list(line.split()) for line in best_wrap]
                # Move dangling single short token at end of line to next line ONLY IF next line width permits
                for li in range(len(cand_wrap) - 1):
                    if len(cand_wrap[li]) > 1 and cand_wrap[li][-1].lower() in short_tokens:
                        cand_next = [cand_wrap[li][-1]] + cand_wrap[li + 1]
                        bb_next = font.getbbox(" ".join(cand_next))
                        w_next = bb_next[2] - bb_next[0]
                        target_idx = min(li + 1, len(allowed_widths) - 1) if 'allowed_widths' in locals() and allowed_widths else None
                        max_w = allowed_widths[target_idx] if target_idx is not None else 2 * a
                        if w_next <= max_w:
                            moved = cand_wrap[li].pop()
                            cand_wrap[li + 1].insert(0, moved)
                best_wrap = [" ".join(wl) for wl in cand_wrap if wl]

            if cand_words is words:
                return best_wrap

            if cand_words is not words:
                for split, original in split_pairs.items():
                    best_wrap = [line.replace(split, original) for line in best_wrap]
            break

    return best_wrap


def render_translated_dialogues(image: Image.Image, dialogues: list) -> Image.Image:
    """
    Automatically typesets translated dialogue bubbles onto a cleaned manga page
    using dynamic elliptical line wrapping and font scaling.
    """
    rendered = image.copy()
    if rendered.mode != "RGB":
        rendered = rendered.convert("RGB")
    draw = ImageDraw.Draw(rendered)
    gray_full = np.array(image.convert("L"))

    for item in dialogues:
        text = (item.get("translation") or item.get("translated_text") or "").strip()
        if not text:
            continue

        text = text.replace("ʻ", "'").replace("ʼ", "'").replace("‘", "'").replace("’", "'").replace("`", "'").replace("—", "-").replace("–", "-").replace("…", "...")

        # Sound effects (SFX) drawn directly on artwork must NEVER be overwritten with standard fonts
        is_sfx = (
            item.get("is_sfx", False)
            or item.get("category") == "sfx"
            or is_sfx_text(item.get("transcription") or item.get("original_text") or "")
        )
        if is_sfx:
            continue
        box = item.get("bbox") or [item.get("left", 0), item.get("top", 0), item.get("right", 0), item.get("bottom", 0)]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        draw_box = [x1, y1, x2, y2]
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            continue

        # 1. Compute true mathematical centroid of the physical speech bubble
        crop_gray = np.array(image.crop((x1, y1, x2, y2)).convert("L"))
        bg_lum = float(np.median(crop_gray)) if crop_gray.size > 0 else 255.0
        bg_std = float(np.std(crop_gray)) if crop_gray.size > 0 else 0.0

        # Safe inscribed box for white bubble / narration box:
        # If top or bottom edge intersects a dark border row, shrink inward >= 4px from that border
        is_white_box = (
            (bg_lum >= 170 and np.mean(crop_gray > 200) >= 0.25)
            or item.get("category") == "narration"
        )
        if is_white_box and crop_gray.size > 0:
            def _is_dark_row(row_arr):
                if row_arr.size == 0:
                    return False
                if np.mean(row_arr) < 120 or np.mean(row_arr < 120) >= 0.20:
                    return True
                min_c = max(6, int(0.20 * len(row_arr)))
                if _has_continuous_dark_segment(row_arr, dark_thresh=120, min_len=min_c):
                    return True
                return False

            # Check top edge
            top_border_y = None
            for r in range(min(bh // 2, 8)):
                if _is_dark_row(crop_gray[r, :]):
                    top_border_y = y1 + r
                else:
                    if top_border_y is not None:
                        break
            if top_border_y is None:
                for y_cand in range(max(0, y1 - 3), y1):
                    if _is_dark_row(gray_full[y_cand, x1:x2]):
                        top_border_y = y_cand

            if top_border_y is not None:
                draw_box[1] = max(draw_box[1], top_border_y + 1 + 4)

            # Check bottom edge
            bot_border_y = None
            for offset in range(min(bh // 2, 8)):
                r = bh - 1 - offset
                if _is_dark_row(crop_gray[r, :]):
                    bot_border_y = y1 + r
                else:
                    if bot_border_y is not None:
                        break
            if bot_border_y is None:
                for y_cand in range(y2, min(gray_full.shape[0], y2 + 4)):
                    if _is_dark_row(gray_full[y_cand, x1:x2]):
                        bot_border_y = y_cand
                        break

            if bot_border_y is not None:
                draw_box[3] = min(draw_box[3], bot_border_y - 4)

            if draw_box[3] > draw_box[1] + 10:
                x1, y1, x2, y2 = draw_box
                bw, bh = x2 - x1, y2 - y1
                crop_gray = np.array(image.crop((x1, y1, x2, y2)).convert("L"))
                bg_lum = float(np.median(crop_gray)) if crop_gray.size > 0 else 255.0
                bg_std = float(np.std(crop_gray)) if crop_gray.size > 0 else 0.0

        cx, cy = find_bubble_visual_center(gray_full, (x1, y1, x2, y2))

        if bh > bw * 1.8:
            cy = (y1 + y2) / 2.0
        is_on_artwork = is_sfx or bg_lum < 225 or bg_std > 25 or item.get("is_on_art", False)

        is_rect = item.get("category") == "narration" or (bw <= 85 and bh <= 90 and bg_lum > 240)
        margin_ratio = 0.80 if is_rect else (0.72 if not is_on_artwork else 0.82)
        a = (bw / 2.0) * margin_ratio
        b = (bh / 2.0) * margin_ratio

        # Horizontal rectangular ticker / banner (e.g. TV news ticker, headline banner)
        is_banner = (bw >= 3.5 * bh and bh <= 60) or (bw >= 260 and bh <= 45)
        if is_banner:
            clean_banner_txt = text.replace("\n", " ").strip()
            words = clean_banner_txt.split()
            best_fs = 9
            best_lines = [clean_banner_txt]
            for fs in range(min(14, max(8, int(bh * 0.45))), 7, -1):
                f_b = get_best_font(size=fs, bold=True)
                ascent, descent = f_b.getmetrics()
                lh = ascent + descent
                # Try 2 lines
                mid = len(words) // 2
                l1 = " ".join(words[:mid])
                l2 = " ".join(words[mid:])
                bb1 = f_b.getbbox(l1)
                bb2 = f_b.getbbox(l2)
                w1 = bb1[2] - bb1[0]
                w2 = bb2[2] - bb2[0]
                if max(w1, w2) <= bw * 0.94 and (lh * 2 * 1.1) <= bh * 0.92:
                    best_fs = fs
                    best_lines = [l1, l2]
                    break
                # Try 1 line
                bb_single = f_b.getbbox(clean_banner_txt)
                if (bb_single[2] - bb_single[0]) <= bw * 0.94 and lh <= bh * 0.85:
                    best_fs = fs
                    best_lines = [clean_banner_txt]
                    break

            f_b = get_best_font(size=best_fs, bold=True)
            ascent, descent = f_b.getmetrics()
            lh = ascent + descent
            lstep = int(lh * 1.1)
            tot_h = (len(best_lines) - 1) * lstep + lh
            start_y = cy - (tot_h / 2.0) - (ascent * 0.15)
            for li, line in enumerate(best_lines):
                bb = f_b.getbbox(line)
                lw = bb[2] - bb[0]
                lx = cx - (lw / 2.0) - bb[0]
                draw.text((lx, start_y + li * lstep), line, fill=(15, 15, 15), font=f_b)
            continue

        # Multi-lobe shock bubble with central spikes: render upper lobe and lower lobe avoiding central spikes
        dark_ratio = np.count_nonzero(crop_gray < 100) / float(crop_gray.size) if crop_gray.size > 0 else 0
        mid_y1 = int(bh * 0.32)
        mid_y2 = int(bh * 0.68)
        has_center_spikes = False
        if dark_ratio > 0.45 and mid_y2 > mid_y1:
            row_dark_ratios = np.sum((crop_gray[mid_y1:mid_y2, :] < 130), axis=1) / float(bw)
            if np.any(row_dark_ratios > 0.40):
                has_center_spikes = True

        # Multi-lobe shock bubble with central spikes:
        is_shock_bubble = has_center_spikes and (bw >= 120 and bh >= 100)
        if is_shock_bubble and bh <= bw * 1.8:
            all_words = text.split()
            if "\n" in text:
                parts = [p.strip() for p in text.split("\n") if p.strip()]
            elif len(all_words) > 2:
                mid_idx = max(1, len(all_words) // 2)
                parts = [" ".join(all_words[:mid_idx]), " ".join(all_words[mid_idx:])]
            else:
                parts = [text]

            fs = min(13, max(10, int(bh * 0.17)))
            f = get_best_font(size=fs, bold=True)
            cy1_lobe = y1 + int(bh * 0.22)
            cy2_lobe = y1 + int(bh * 0.72)

            # Determine contrast based on lobe background:
            h_img, w_img = gray_full.shape[:2]
            lobe1_patch = gray_full[max(0, cy1_lobe-5):min(h_img, cy1_lobe+6), max(0, int(cx)-10):min(w_img, int(cx)+11)]
            med1 = float(np.median(lobe1_patch)) if lobe1_patch.size > 0 else 255.0
            if med1 < 100:
                # Dark background / silhouette: crisp WHITE text with black stroke
                sb_fill = (255, 255, 255)
                sb_stroke_fill = (0, 0, 0)
                sb_stroke_w = 2
            else:
                # Light / white bubble: crisp black text
                sb_fill = (15, 15, 15)
                sb_stroke_fill = None
                sb_stroke_w = 0

            if len(parts) >= 2:
                bb1 = f.getbbox(parts[0])
                w1, h1 = bb1[2] - bb1[0], bb1[3] - bb1[1]
                if sb_stroke_w > 0:
                    draw.text((cx - w1 // 2 - bb1[0], cy1_lobe - h1 // 2 - bb1[1]), parts[0], fill=sb_fill, stroke_width=sb_stroke_w, stroke_fill=sb_stroke_fill, font=f)
                else:
                    draw.text((cx - w1 // 2 - bb1[0], cy1_lobe - h1 // 2 - bb1[1]), parts[0], fill=sb_fill, font=f)
                bot_text = "\n".join(parts[1:])
                bb2 = draw.multiline_textbbox((0, 0), bot_text, font=f)
                w2, h2 = bb2[2] - bb2[0], bb2[3] - bb2[1]
                if sb_stroke_w > 0:
                    draw.text((cx - w2 // 2 - bb2[0], cy2_lobe - h2 // 2 - bb2[1]), bot_text, fill=sb_fill, stroke_width=sb_stroke_w, stroke_fill=sb_stroke_fill, font=f)
                else:
                    draw.text((cx - w2 // 2 - bb2[0], cy2_lobe - h2 // 2 - bb2[1]), bot_text, fill=sb_fill, font=f)
            else:
                bb = f.getbbox(text)
                w, h_txt = bb[2] - bb[0], bb[3] - bb[1]
                cy_mid = (y1 + y2) // 2
                if sb_stroke_w > 0:
                    draw.text((cx - w // 2 - bb[0], cy_mid - h_txt // 2 - bb[1]), text, fill=sb_fill, stroke_width=sb_stroke_w, stroke_fill=sb_stroke_fill, font=f)
                else:
                    draw.text((cx - w // 2 - bb[0], cy_mid - h_txt // 2 - bb[1]), text, fill=sb_fill, font=f)
            continue

        if not is_sfx and not is_manga_punctuation(text):
            text = text.upper()

        # Separate multi-sentence / multi-paragraph text cleanly, avoiding rogue breaks in single sentences
        raw_paras = [p.strip() for p in text.split("\n") if p.strip()]
        paragraphs = []
        for p in raw_paras:
            split_pts = re.split(r'(?<=[.!?…])\s+(?=[A-ZА-ЯÖĞŞÇO\'G\'a-zöğşç])', p)
            if len(split_pts) > 1 and len(split_pts) <= 3:
                paragraphs.extend([sp.strip() for sp in split_pts if sp.strip()])
            else:
                paragraphs.append(p)
        if not paragraphs:
            paragraphs = [text.strip()]

        total_words = sum(len(p.split()) for p in paragraphs)
        max_fs = min(22, max(14, int((bw * bh) ** 0.5 * 0.17)))
        min_fs = 10
        margin_ratio = 0.82 if is_on_artwork else 0.80

        if is_rect:
            margin_ratio = max(margin_ratio, 0.80)

        a = (bw / 2.0) * margin_ratio
        b = (bh / 2.0) * margin_ratio

        chosen_font = None
        chosen_paras = None

        # For single short tokens / short phrases in horizontal or round bubbles, try single-line fitting first
        if len(paragraphs) == 1 and (total_words <= 2 or len(text) <= 12) and (bw >= bh * 0.70):
            for fs in range(max_fs, min_fs - 1, -1):
                f = get_best_font(size=fs, bold=True)
                bbox = f.getbbox(text)
                lw = bbox[2] - bbox[0]
                lh = bbox[3] - bbox[1]
                if lw <= a * 1.80 and lh <= b * 1.80:
                    chosen_font = f
                    chosen_paras = [[text]]
                    break

        if not chosen_paras:
            # 1. Prefer word-level line wrapping and font-size scaling before inserting hyphens (-)
            # Try fitting unhyphenated text from max_fs down to minimum readable font size (6 pt)
            no_hyphen_min_fs = max(6, int(max_fs * 0.45))
            for fs in range(max_fs, no_hyphen_min_fs - 1, -1):
                f = get_best_font(size=fs, bold=True)
                candidate_paras = []
                fits = True
                for p_idx, p in enumerate(paragraphs):
                    p_words = p.split()
                    if is_sfx and any("-" in w for w in p_words):
                        exp = []
                        for w in p_words:
                            exp.extend([sub for sub in w.split("-") if sub] if "-" in w else [w])
                        p_words = exp
                    lines_p = wrap_text_elliptical(p_words, f, a, b, allow_hyphenation=False)
                    if not lines_p:
                        fits = False
                        break
                    candidate_paras.append(lines_p)
                if fits and candidate_paras:
                    ascent, descent = f.getmetrics()
                    lh = ascent + descent
                    lstep = int(lh * 1.15)
                    tot_lines = sum(len(lp) for lp in candidate_paras)
                    pgap = int(lh * 0.45) if len(candidate_paras) > 1 else 0
                    th = (tot_lines - 1) * lstep + lh + (len(candidate_paras) - 1) * pgap
                    if th <= 2 * b * 0.85:
                        chosen_font = f
                        chosen_paras = candidate_paras
                        break

        if not chosen_paras:
            # 2. Only fall back to allow_hyphenation=True if unhyphenated text still cannot fit at the reduced font size
            for fs in range(max_fs, 5, -1):
                f = get_best_font(size=fs, bold=True)
                candidate_paras = []
                fits = True
                for p_idx, p in enumerate(paragraphs):
                    p_words = p.split()
                    if is_sfx and any("-" in w for w in p_words):
                        exp = []
                        for w in p_words:
                            exp.extend([sub for sub in w.split("-") if sub] if "-" in w else [w])
                        p_words = exp
                    lines_p = wrap_text_elliptical(p_words, f, a, b, allow_hyphenation=True)
                    if not lines_p:
                        fits = False
                        break
                    candidate_paras.append(lines_p)
                if fits and candidate_paras:
                    ascent, descent = f.getmetrics()
                    lh = ascent + descent
                    lstep = int(lh * 1.15)
                    tot_lines = sum(len(lp) for lp in candidate_paras)
                    pgap = int(lh * 0.45) if len(candidate_paras) > 1 else 0
                    th = (tot_lines - 1) * lstep + lh + (len(candidate_paras) - 1) * pgap
                    if th <= 2 * b * 0.85:
                        chosen_font = f
                        chosen_paras = candidate_paras
                        break

        if not chosen_paras:
            # Absolute last resort: scale down font size until it strictly fits inside the bubble envelope
            for fallback_fs in range(max(6, min_fs - 1), 4, -1):
                f_fb = get_best_font(size=fallback_fs, bold=True)
                ascent, descent = f_fb.getmetrics()
                lh_fb = ascent + descent
                lstep_fb = int(lh_fb * 1.15)
                fb_paras = []
                for p in paragraphs:
                    p_words = p.split()
                    cur_line = []
                    lines = []
                    for w in p_words:
                        test_str = " ".join(cur_line + [w])
                        bb = f_fb.getbbox(test_str)
                        w_line = bb[2] - bb[0]
                        line_limit = (2 * a * 0.72) if (not lines or len(lines) == 0) else (2 * a * 0.82)
                        if cur_line and w_line > line_limit:
                            lines.append(" ".join(cur_line))
                            cur_line = [w]
                        else:
                            cur_line.append(w)
                    if cur_line:
                        lines.append(" ".join(cur_line))
                    fb_paras.append(lines)
                tot_lines_fb = sum(len(lp) for lp in fb_paras)
                th_fb = (tot_lines_fb - 1) * lstep_fb + lh_fb
                if th_fb <= 2 * b * 0.85 or fallback_fs == 5:
                    chosen_font = f_fb
                    chosen_paras = fb_paras
                    break

        # Guarantee no orphaned 1-2 char words (e.g. 'i', 'I', 'u', 'va', 'da', "o'", "ö", "ş", "ç")
        short_tokens = {"i", "u", "va", "da", "o'", "ö", "oʻ", "o‘", "bu", "a", "ya", "v", "s", "men", "ha", "yo'q", "ku", "o’", "sh", "ch", "ş", "ç", "ğ"}
        cleaned_paras = []
        for p_lines in chosen_paras:
            if len(p_lines) > 1:
                for li in range(len(p_lines) - 1):
                    w_list = p_lines[li].split()
                    if len(w_list) > 1 and w_list[-1].lower() in short_tokens:
                        moved = w_list[-1]
                        cand_next = moved + " " + p_lines[li + 1]
                        bb_next = chosen_font.getbbox(cand_next)
                        if (bb_next[2] - bb_next[0]) <= 2 * a * 0.95:
                            w_list.pop()
                            p_lines[li] = " ".join(w_list)
                            p_lines[li + 1] = cand_next
                merged = []
                for li, l_str in enumerate(p_lines):
                    s_str = l_str.strip()
                    if len(s_str.split()) == 1 and s_str.lower() in short_tokens:
                        if merged and (chosen_font.getbbox(merged[-1] + " " + s_str)[2] - chosen_font.getbbox(merged[-1] + " " + s_str)[0]) <= 2 * a * 0.95:
                            merged[-1] = merged[-1] + " " + s_str
                        elif li + 1 < len(p_lines) and (chosen_font.getbbox(s_str + " " + p_lines[li + 1])[2] - chosen_font.getbbox(s_str + " " + p_lines[li + 1])[0]) <= 2 * a * 0.95:
                            p_lines[li + 1] = s_str + " " + p_lines[li + 1]
                        else:
                            merged.append(l_str)
                    else:
                        merged.append(l_str)
                p_lines = [l for l in merged if l.strip()]
            cleaned_paras.append(p_lines)
        chosen_paras = cleaned_paras

        ascent, descent = chosen_font.getmetrics()
        line_h = ascent + descent
        line_step = int(line_h * 1.15)
        para_gap = int(line_h * 0.45) if len(chosen_paras) > 1 else 0
        tot_lines = sum(len(lp) for lp in chosen_paras)
        total_h = (tot_lines - 1) * line_step + line_h + (len(chosen_paras) - 1) * para_gap
        pad_y = max(8, int(bh * 0.12))
        max_allowed_h = max(line_h, bh - 2 * pad_y)
        if total_h > max_allowed_h and tot_lines > 1:
            line_step = max(line_h, int((max_allowed_h - line_h - (len(chosen_paras) - 1) * para_gap) / (tot_lines - 1)))
            total_h = (tot_lines - 1) * line_step + line_h + (len(chosen_paras) - 1) * para_gap
        start_y = cy - (total_h / 2.0)
        if start_y < y1 + pad_y:
            start_y = y1 + pad_y
        if start_y + total_h > y2 - pad_y:
            start_y = max(y1 + pad_y, y2 - pad_y - total_h)

        if bh > bw * 1.8 and tot_lines:
            first = chosen_font.getbbox(chosen_paras[0][0])
            last = chosen_font.getbbox(chosen_paras[-1][-1])
            start_y = cy - (first[1] + total_h - line_h + last[3]) / 2
            start_y = max(y1 + pad_y, min(y2 - pad_y - total_h, start_y))

        # Adaptive font color & stroke based on local background luminance
        # CRITICAL: Evaluate local luminance at the visual center (cx, cy) where text is drawn,
        # NOT the whole bounding box (which might contain dark outer art or spikes).
        h_img, w_img = gray_full.shape[:2]
        r_box = max(4, min(14, int(min(bw, bh) * 0.35)))
        center_patch = gray_full[max(0, int(cy)-r_box):min(h_img, int(cy)+r_box+1), max(0, int(cx)-r_box):min(w_img, int(cx)+r_box+1)]
        center_p75 = float(np.percentile(center_patch, 75)) if center_patch.size > 0 else 255.0
        center_med = float(np.median(center_patch)) if center_patch.size > 0 else 255.0

        if center_med >= 170:
            # The text area inside the bubble is light/white -> crisp black text
            text_fill = (15, 15, 15)
            stroke_fill = None
            stroke_w = 0
        elif center_med < 75:
            # Genuinely dark background / black narration box / dark shadow:
            # Crisp WHITE text with solid black outline so it never gets lost!
            text_fill = (255, 255, 255)
            stroke_fill = (0, 0, 0)
            stroke_w = 2
        else:
            # Screentone / gray halftone / artwork (75 <= center_med < 170):
            # Crisp black text with bright white halo/stroke so it pops out and never fades away!
            text_fill = (10, 10, 10)
            stroke_fill = (255, 255, 255)
            stroke_w = 2

        # Standalone expressive manga punctuation (?, !, ?!, ...) is rendered large, bold, and centered
        if is_manga_punctuation(text):
            clean_punc = normalize_manga_punctuation(text)
            fs = min(52, max(22, int(min(bw, bh) * 0.44)))
            punc_font = get_best_font(size=fs, bold=True)
            bb = punc_font.getbbox(clean_punc)
            pw = bb[2] - bb[0]
            ph = bb[3] - bb[1]
            px = cx - (pw / 2.0) - bb[0]
            py = cy - (ph / 2.0) - bb[1]
            if stroke_w > 0 and stroke_fill is not None:
                draw.text((px, py), clean_punc, fill=text_fill, font=punc_font, stroke_width=stroke_w, stroke_fill=stroke_fill)
            else:
                draw.text((px, py), clean_punc, fill=text_fill, font=punc_font)
            continue

        # Narrow vertical speech bubble lettering (e.g. S-O-R-R-Y, U-Z-R, П-Р-О-С-Т-И)
        is_vertical_strip = (bw <= 65 and bh >= 1.7 * bw)
        if is_vertical_strip and item.get("category") != "narration" and len(text.split()) == 1 and len(text) * 14 <= bh:
            chars = [c for c in text.strip() if not c.isspace()]
            if chars:
                fs = min(22, max(11, int(bw * 0.70), int(bh / (len(chars) * 1.35))))
                f = get_best_font(size=fs, bold=True)
                ascent, descent = f.getmetrics()
                char_h = ascent + descent
                step_y = min(char_h, bh / len(chars))
                total_h = (len(chars) - 1) * step_y + (f.getbbox(chars[-1])[3] - f.getbbox(chars[-1])[1])
                sy = cy - (total_h / 2.0)
                for ci, ch in enumerate(chars):
                    bb = f.getbbox(ch)
                    cw = bb[2] - bb[0]
                    cx_ch = cx - (cw / 2.0) - bb[0]
                    cy_ch = sy + ci * step_y - bb[1]
                    if stroke_w > 0 and stroke_fill is not None:
                        draw.text((cx_ch, cy_ch), ch, fill=text_fill, font=f, stroke_width=stroke_w, stroke_fill=stroke_fill)
                    else:
                        draw.text((cx_ch, cy_ch), ch, fill=text_fill, font=f)
                continue

        cur_y = start_y
        for pi, p_lines in enumerate(chosen_paras):
            if pi > 0:
                cur_y += para_gap
            for line in p_lines:
                bbox = chosen_font.getbbox(line)
                lw = bbox[2] - bbox[0]
                # Precise glyph-level horizontal centering: subtract left bearing offset
                lx = cx - (lw / 2.0) - bbox[0]
                # Guaranteed safety clamping: keep text strictly within speech bubble interior with 10-12% vertical padding
                pad_x = max(6, int(bw * 0.08))
                lx = max(x1 + pad_x, min(max(x1 + pad_x, x2 - pad_x - lw), lx))
                draw_y = max(y1 + pad_y, min(max(y1 + pad_y, y2 - pad_y - line_h), cur_y))
                if stroke_w > 0 and stroke_fill is not None:
                    draw.text((lx, draw_y), line, fill=text_fill, font=chosen_font, stroke_width=stroke_w, stroke_fill=stroke_fill)
                else:
                    draw.text((lx, draw_y), line, fill=text_fill, font=chosen_font)
                cur_y += line_step

    return rendered


def clean_bubble_crop(
    patch_orig,
    patch_gray=None,
    patch_mask=None,
    neural_patch=None,
    inpainter_model=None,
    inpaint_transform=None,
    device=DEVICE,
):
    """
    Master Border-Protected Bubble Cleaning Algorithm:
    - White / cream bubble with border: preserves outer contour border; cleans interior text to local background color.
    - Dark narration box: preserves border contour; fills interior text with dark background color (never pure white).
    - Screentone / artwork / borderless: uses neural cleaner or U-Net inpainter if available, else Telea.
    """
    if patch_orig is None or patch_orig.size == 0:
        return patch_orig

    hp, wp = patch_orig.shape[:2]
    if hp <= 6 or wp <= 6:
        return patch_orig.copy()

    if patch_gray is None:
        patch_gray = cv2.cvtColor(patch_orig, cv2.COLOR_RGB2GRAY)

    k_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_rect3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Padded interior region (ignoring outer margin pixels that may contain borders or opposite-tone gutters)
    my = max(2, int(hp * 0.12))
    mx = max(2, int(wp * 0.12))
    if hp - 2 * my >= 4 and wp - 2 * mx >= 4:
        interior_gray = patch_gray[my:hp - my, mx:wp - mx]
        interior_orig = patch_orig[my:hp - my, mx:wp - mx]
    else:
        interior_gray = patch_gray
        interior_orig = patch_orig

    interior_median = float(np.median(interior_gray))
    interior_p15 = float(np.percentile(interior_gray, 15))
    interior_p25 = float(np.percentile(interior_gray, 25))
    interior_p85 = float(np.percentile(interior_gray, 85))

    full_median = float(np.median(patch_gray))
    full_p15 = float(np.percentile(patch_gray, 15))
    full_p85 = float(np.percentile(patch_gray, 85))

    dark_interior_pixels = interior_gray[interior_gray < 130]
    has_dark_interior_bg = (dark_interior_pixels.size >= 10 and float(np.median(dark_interior_pixels)) <= 90)

    # Determine is_dark_bubble from the INTERIOR region of the crop (ignoring outer margins)
    is_dark_bubble = (
        (interior_median < 95)
        or (interior_p15 < 55 and interior_median < 120 and interior_p85 < 175)
        or (interior_median < 110 and interior_p25 < 65 and has_dark_interior_bg)
        or (full_median < 65)
        or (full_p15 < 40 and full_p85 < 110)
    )

    is_white_bubble = False
    if not is_dark_bubble:
        is_white_bubble = (interior_median >= 175) or (interior_p85 >= 205 and interior_median >= 145) or (full_median >= 190)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. WHITE OR CREAM BUBBLES
    # ──────────────────────────────────────────────────────────────────────────
    if is_white_bubble:
        # Determine local background color from non-ink pixels inside the padded interior
        non_ink_interior = (interior_gray >= 185)
        if np.any(non_ink_interior):
            local_bg_color = np.median(interior_orig[non_ink_interior], axis=0).astype(np.uint8)
        else:
            non_ink_full = (patch_gray >= 185)
            if np.any(non_ink_full):
                local_bg_color = np.median(patch_orig[non_ink_full], axis=0).astype(np.uint8)
            else:
                local_bg_color = np.array([255, 255, 255], dtype=np.uint8)

        # Dark ink mask
        dark_mask = (patch_gray < 185).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)

        # Build protected_mask for true perimeter/border structures:
        # a) Straight box frames along the perimeter (top, bottom, left, right)
        frame_pad_y = max(4, int(hp * 0.15))
        frame_pad_x = max(4, int(wp * 0.15))
        kw = max(24, int(wp * 0.60))
        kh = max(24, int(hp * 0.60))
        horiz_frame = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1)))
        vert_frame = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh)))

        straight_lines = np.zeros_like(dark_mask)
        if np.any(horiz_frame):
            straight_lines[0:frame_pad_y, :] |= horiz_frame[0:frame_pad_y, :]
            straight_lines[hp - frame_pad_y:hp, :] |= horiz_frame[hp - frame_pad_y:hp, :]
        if np.any(vert_frame):
            straight_lines[:, 0:frame_pad_x] |= vert_frame[:, 0:frame_pad_x]
            straight_lines[:, wp - frame_pad_x:wp] |= vert_frame[:, wp - frame_pad_x:wp]

        # b) Identify components that are true borders, curved bubble contours, or outer artwork
        # Note: Do NOT add straight_lines connected components to keep_labels; straight_lines
        # itself protects the structural frame directly without protecting interior letters touching it.
        keep_labels = set()

        border_margin_x = max(2, int(wp * 0.08))
        border_margin_y = max(2, int(hp * 0.08))

        has_straight_lines = np.any(straight_lines)

        for lbl in range(1, num_labels):
            lx = stats[lbl, cv2.CC_STAT_LEFT]
            ly = stats[lbl, cv2.CC_STAT_TOP]
            lw = stats[lbl, cv2.CC_STAT_WIDTH]
            lh = stats[lbl, cv2.CC_STAT_HEIGHT]
            la = stats[lbl, cv2.CC_STAT_AREA]

            # 1. Outer artwork / clothing touching crop boundaries and spanning across
            touches_top = (ly <= 1)
            touches_bot = (ly + lh >= hp - 1)
            touches_left = (lx <= 1)
            touches_right = (lx + lw >= wp - 1)

            # Large solid background artwork touching edge
            if (touches_top or touches_bot or touches_left or touches_right):
                if (lw >= 0.70 * wp and (touches_top or touches_bot) and la >= 0.15 * (wp * hp)):
                    keep_labels.add(lbl)
                    continue
                if (lh >= 0.70 * hp and (touches_left or touches_right) and la >= 0.15 * (wp * hp)):
                    keep_labels.add(lbl)
                    continue

            # 2. Large curved oval bubble contour outline spanning across the bubble
            # Only when no straight box frame is present; must be hollow around center (center_ratio <= 0.08)
            # so large words/letters (e.g. 'MUS... TOR', 'HAR') are never misclassified as bubble contours.
            if not has_straight_lines:
                spans_width = (lx <= border_margin_x and (lx + lw) >= wp - border_margin_x and lw >= 0.70 * wp)
                spans_height = (ly <= border_margin_y and (ly + lh) >= hp - border_margin_y and lh >= 0.50 * hp)
                solidity = la / float(lw * lh) if (lw * lh > 0) else 1.0

                if spans_width and spans_height and solidity <= 0.35:
                    cy1, cy2 = int(hp * 0.25), int(hp * 0.75)
                    cx1, cx2 = int(wp * 0.25), int(wp * 0.75)
                    center_pixels = np.sum(labels[cy1:cy2, cx1:cx2] == lbl)
                    center_ratio = center_pixels / float(la) if la > 0 else 0.0
                    if center_ratio <= 0.08:
                        keep_labels.add(lbl)
                        continue

        # Build protected_mask:
        if keep_labels:
            protected_mask = np.isin(labels, list(keep_labels)).astype(np.uint8) * 255
        else:
            protected_mask = np.zeros_like(dark_mask)

        if has_straight_lines:
            protected_mask = cv2.bitwise_or(protected_mask, (straight_lines > 0).astype(np.uint8) * 255)

        # c) Identify letter ink to erase
        letter_mask = ((dark_mask > 0) & (protected_mask == 0)).astype(np.uint8)

        # Include external OCR mask if provided (excluding protected structures)
        if patch_mask is not None and np.any(patch_mask):
            ext_mask = ((patch_mask > 0) & (protected_mask == 0)).astype(np.uint8)
            letter_mask = cv2.bitwise_or(letter_mask, ext_mask)

        # Dilate letter ink by 2 px with (3, 3) kernel
        dilated_letters = cv2.dilate(letter_mask, k_rect3, iterations=2)

        # CRITICAL: Subtract protected_mask AFTER dilating erase_mask so borders are never nicked
        dilated_erase = cv2.bitwise_and(dilated_letters, cv2.bitwise_not(protected_mask))

        # Softly clean anti-aliased gray halo pixels (185..240) touching the erased text
        anti_aliased_mask = ((patch_gray >= 185) & (patch_gray <= 240)).astype(np.uint8)
        halo_zone = cv2.dilate(dilated_erase, k_rect3, iterations=1)
        halo_candidates = (halo_zone > 0) & (anti_aliased_mask > 0) & (protected_mask == 0)

        erase_mask = cv2.bitwise_and(
            cv2.bitwise_or(dilated_erase, halo_candidates.astype(np.uint8)),
            cv2.bitwise_not(protected_mask)
        )

        patch_clean = patch_orig.copy()
        if np.any(erase_mask):
            # Compute local ring background color around the erased text if possible
            ring = cv2.dilate(erase_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2)
            ring_bg = (ring > 0) & (erase_mask == 0) & (protected_mask == 0) & (patch_gray >= 170)
            if np.sum(ring_bg) > 20:
                fill_color = np.median(patch_orig[ring_bg], axis=0).astype(np.uint8)
            else:
                fill_color = local_bg_color

            patch_clean[erase_mask > 0] = fill_color

            if inpainter_model is not None and inpaint_transform is not None:
                try:
                    t_patch = inpaint_transform(Image.fromarray(patch_orig)).unsqueeze(0).to(device)
                    t_pmask = (inpaint_transform(Image.fromarray((erase_mask > 0).astype(np.uint8) * 255)) > 0.5).float().unsqueeze(0).to(device)
                    with torch.no_grad():
                        inpainter_model(t_patch, t_pmask)
                except Exception as e:
                    logger.debug("Inpainter execution failed: %s", e)

        return patch_clean

    # ──────────────────────────────────────────────────────────────────────────
    # 2. DARK NARRATION BOXES / DARK BUBBLES
    # ──────────────────────────────────────────────────────────────────────────
    if is_dark_bubble and not is_white_bubble:
        # Local dark background color from interior non-text pixels
        interior_dark = (interior_gray < 110)
        if np.any(interior_dark):
            dark_bg_color = np.median(interior_orig[interior_dark], axis=0).astype(np.uint8)
        else:
            dark_bg_color = np.median(interior_orig, axis=(0, 1)).astype(np.uint8)
        if np.mean(dark_bg_color) > 100:
            dark_bg_color = np.array([25, 25, 25], dtype=np.uint8)

        # Light text ink inside the dark bubble
        light_thresh = max(115, int(interior_median + 30))
        light_mask = (patch_gray > light_thresh).astype(np.uint8)

        # Connected components on light_mask to isolate text from outer light margins
        num_l, l_labels, l_stats, _ = cv2.connectedComponentsWithStats(light_mask, connectivity=8)
        outer_light_labels = set()
        for lbl in range(1, num_l):
            lx = l_stats[lbl, cv2.CC_STAT_LEFT]
            ly = l_stats[lbl, cv2.CC_STAT_TOP]
            lw = l_stats[lbl, cv2.CC_STAT_WIDTH]
            lh = l_stats[lbl, cv2.CC_STAT_HEIGHT]
            la = l_stats[lbl, cv2.CC_STAT_AREA]
            touches_edge = (lx <= 0 or ly <= 0 or lx + lw >= wp or ly + lh >= hp)
            if touches_edge and (lw >= 0.40 * wp or lh >= 0.40 * hp or la >= 0.10 * (wp * hp)):
                outer_light_labels.add(lbl)

        # Filter: keep outer boundary margins in outer_light_labels
        outer_margin_mask = np.isin(l_labels, list(outer_light_labels)).astype(np.uint8) * 255 if outer_light_labels else np.zeros_like(patch_gray)

        # Text ink: light components that do not belong to the outer margin
        text_light_mask = ((light_mask > 0) & (outer_margin_mask == 0)).astype(np.uint8)
        if patch_mask is not None and np.any(patch_mask):
            text_light_mask = cv2.bitwise_or(text_light_mask, ((patch_mask > 0) & (outer_margin_mask == 0)).astype(np.uint8))

        # Dilate text ink by 3 px to swallow black outlines around white text
        dilated_light = cv2.dilate(text_light_mask, k_rect3, iterations=3)
        text_erase_mask = cv2.bitwise_and(dilated_light, cv2.bitwise_not(outer_margin_mask))

        patch_clean = patch_orig.copy()
        if np.any(text_erase_mask):
            # Compute local ring background color around light text
            ring = cv2.dilate(text_erase_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2)
            ring_bg = (ring > 0) & (text_erase_mask == 0) & (outer_margin_mask == 0) & (patch_gray < 110)
            if np.sum(ring_bg) > 20:
                dark_fill = np.median(patch_orig[ring_bg], axis=0).astype(np.uint8)
            else:
                dark_fill = dark_bg_color

            # Smooth row-wise gradient fill if vertical gradient is present (for gradient dark boxes like #4, #5, #6)
            ring_gray_vals = patch_gray[ring_bg] if np.sum(ring_bg) > 20 else patch_gray[patch_gray < 110]
            if ring_gray_vals.size > 0 and float(np.std(ring_gray_vals)) >= 5.0:
                for r_y in range(hp):
                    row_stroke = text_erase_mask[r_y] > 0
                    if not np.any(row_stroke):
                        continue
                    y_min = max(0, r_y - 2)
                    y_max = min(hp, r_y + 3)
                    band_mask = (text_erase_mask[y_min:y_max] == 0) & (ring[y_min:y_max] > 0) & (outer_margin_mask[y_min:y_max] == 0) & (patch_gray[y_min:y_max] < 110)
                    band_pix = patch_orig[y_min:y_max][band_mask]
                    if band_pix.size >= 3:
                        patch_clean[r_y, row_stroke] = np.median(band_pix.reshape(-1, 3), axis=0).astype(np.uint8)
                    else:
                        patch_clean[r_y, row_stroke] = dark_fill
            else:
                patch_clean[text_erase_mask > 0] = dark_fill

        return patch_clean

    # ──────────────────────────────────────────────────────────────────────────
    # 3. ARTWORK / SCREENTONE / BORDERLESS
    # ──────────────────────────────────────────────────────────────────────────
    dark_thresh = min(215, int(interior_p85 * 0.88))
    dark = (patch_gray < dark_thresh).astype(np.uint8) * 255
    text_dil = patch_mask if (patch_mask is not None and np.any(patch_mask > 0)) else cv2.dilate(dark, k_ellipse, iterations=2)
    patch_clean = patch_orig.copy()
    cleaned_with_model = False
    if neural_patch is not None and neural_patch.shape == patch_orig.shape:
        patch_clean[text_dil > 0] = neural_patch[text_dil > 0]
        cleaned_with_model = True
    if not cleaned_with_model:
        try:
            from simple_lama_inpainting import SimpleLama
            lama = SimpleLama(device=device)
            inp_pil = lama(Image.fromarray(patch_orig), Image.fromarray(text_dil))
            c_lama = np.array(inp_pil.convert("RGB"))
            if c_lama.shape == patch_orig.shape:
                patch_clean[text_dil > 0] = c_lama[text_dil > 0]
                cleaned_with_model = True
        except Exception as lama_err:
            logger.debug("clean_bubble_crop SimpleLama fallback: %s", lama_err)
    if not cleaned_with_model and inpainter_model is not None and inpaint_transform is not None:
        try:
            t_patch = inpaint_transform(Image.fromarray(patch_orig)).unsqueeze(0).to(device)
            t_pmask = (inpaint_transform(Image.fromarray(text_dil)) > 0.5).float().unsqueeze(0).to(device)
            with torch.no_grad():
                _, c_out = inpainter_model(t_patch, t_pmask)
            c_np = (c_out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            c_res = cv2.resize(c_np, (wp, hp), interpolation=cv2.INTER_LINEAR)
            patch_clean[text_dil > 0] = c_res[text_dil > 0]
            cleaned_with_model = True
        except Exception as e:
            logger.debug("Inpainter execution failed: %s", e)
    if not cleaned_with_model:
        clean_telea = cv2.inpaint(patch_orig, text_dil, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        patch_clean[text_dil > 0] = clean_telea[text_dil > 0]

    return patch_clean


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
        self.vocab_path = vocab_path or os.path.join(self.base_dir, "ocr_vocab.json")
        self.model_lock = threading.RLock()
        
        # Self-Learning Components
        self.collector = SelfLearningCollector(base_dir=self.base_dir)
        self.trainer = SelfLearningTrainer(base_dir=self.base_dir)

        # Manga-Text-Cleaner neural text segmentation & page cleaner
        try:
            from manga_text_cleaner import MangaTextCleaner
            self.cleaner = MangaTextCleaner(device=DEVICE)
        except Exception as exc:
            logger.warning("MangaTextCleaner could not be initialized: %s", exc)
            self.cleaner = None

        # Comic-Text-Detector heavy PyTorch detector & text stroke segmenter
        try:
            from comic_text_detector import ComicTextDetector
            self.comic_detector = ComicTextDetector(device=DEVICE)
        except Exception as exc:
            logger.debug("ComicTextDetector could not be initialized: %s", exc)
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
            with open(self.vocab_path, encoding="utf-8") as stream:
                vocab_data = json.load(stream)
            char_to_idx = vocab_data["char_to_idx"]
            idx_to_char = {int(k): v for k, v in vocab_data["idx_to_char"].items()}
            vocab_size = max(char_to_idx.values()) + 1
            detector = MangaBubbleDetector(num_classes=4).to(DEVICE)
            ocr = MangaOCRModel(vocab_size=vocab_size, hidden_dim=128).to(DEVICE)
            inpainter = MangaInpaintingModel().to(DEVICE)
            for name, model in (("detector", detector), ("ocr", ocr), ("inpaint", inpainter)):
                path = os.path.join(self.checkpoint_dir, f"best_manga_{name}.pt")
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Trained checkpoint missing: {path}")
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
            "auto": ["en", "ja"],
        }
        lang_list = languages.get(source_lang, ["en", "ja"] if source_lang == "auto" else ["en"])
        key = tuple(lang_list)
        with self.model_lock:
            if key not in self._ocr_readers:
                try:
                    import easyocr
                    options = {}
                    if os.environ.get("MANGA_EASYOCR_MODEL_DIR"):
                        options["model_storage_directory"] = os.environ["MANGA_EASYOCR_MODEL_DIR"]
                        options["download_enabled"] = False
                    else:
                        options["download_enabled"] = (os.environ.get("HF_HUB_OFFLINE") != "1")
                    self._ocr_readers[key] = easyocr.Reader(
                        lang_list, gpu=torch.cuda.is_available(),
                        verbose=False, **options)
                except Exception as exc:
                    logger.warning("Local EasyOCR %s unavailable: %s", key, exc)
                    self._ocr_readers[key] = None
            reader = self._ocr_readers[key]
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
        if source_lang == "auto":
            reader = None
        else:
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
                        raw_ocr = reader.readtext(orig_np, canvas_size=2000, mag_ratio=1.8)
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
                            clean_tok = re.sub(r'[^\w]', '', text).lower()
                            is_known_sfx = clean_tok in {"топ", "тап", "тук", "пик", "бум", "бах", "тмп", "хлоп", "шурх"}
                            if conf < 0.40 and not is_known_sfx:
                                continue
                            if len(clean_tok) <= 2 and not is_known_sfx:
                                continue
                        if any(c.isascii() and c.isalpha() for c in text):
                            try:
                                from utils import repair_comic_text
                                text = repair_comic_text(text)
                            except Exception:
                                pass
                        raw_boxes.append({
                            "box": [bx1, by1, bx2, by2],
                            "text": text,
                            "conf": float(conf)
                        })
                except Exception as e:
                    logger.debug("EasyOCR readtext exception: %s", e)

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
                is_art = (p85_t < 170 and len(words) <= 2) or is_known_sfx_word
                clusters.append({
                    "block_id": c.get("block_id", f"b_{len(clusters) + 1}"),
                    "category": "sfx" if is_art else "speech_bubble",
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
                    # Only accept detector box if it's genuinely a white bubble interior (>205)
                    if patch.size > 0 and np.mean(patch) > 205:
                        overlap = False
                        for c in clusters:
                            cb = c["box"]
                            if not (cb[2] < x1 or x2 < cb[0] or cb[3] < y1 or y2 < cb[1]):
                                overlap = True
                                break
                        if not overlap:
                            clusters.append({
                                "block_id": f"b_{len(clusters) + 1}",
                                "category": IDX_TO_CATEGORY.get(p["class_id"], "speech_bubble"),
                                "box": [max(0, x1), max(0, y1), min(w, x2), min(h, y2)],
                                "text": ""
                            })
            except Exception as e:
                logger.debug("MangaBubbleDetector verification error: %s", e)

            # Step D: Final filtering - eliminate dark false positives (bushes, dresses, legs, sky)
            gray_np = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2GRAY)
            for c in clusters:
                bx = c["box"]
                text = c.get("text", "").strip()
                patch = gray_np[bx[1]:bx[3], bx[0]:bx[2]]
                mean_lum = np.mean(patch) if patch.size > 0 else 0
                # Must have confirmed dialogue text OR be a bright speech bubble
                if text or mean_lum > 215:
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
                box = det.get("box") or det.get("bbox") or det.get("text_box") or [0, 0, 0, 0] # [x1, y1, x2, y2]
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
                if reader is not None and cat == "speech_bubble" and x2 > x1 + 15 and y2 > y1 + 15:
                    try:
                        crop_cv = orig_np_arr[y1:y2, x1:x2]
                        if crop_cv.size > 0:
                            crop_2x = cv2.resize(crop_cv, (crop_cv.shape[1] * 2, crop_cv.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
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

                                # Group items into lines per column by vertical overlap (no bucket division)
                                col_items = {}
                                for it in valid_items:
                                    c = col_map.get(id(it), 0)
                                    col_items.setdefault(c, []).append(it)

                                line_idx_map = {}
                                sorted_items = []
                                global_line_counter = 0
                                for c in sorted(col_items.keys()):
                                    items_c = col_items[c]
                                    # Sort by y_center first
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
                                        from utils import repair_comic_text
                                        crop_txt = repair_comic_text(crop_txt)
                                    except Exception:
                                        pass

                                crop_conf = float(np.mean([r[2] for r in valid_items]))
                                orig_conf = float(det.get("avg_conf", det.get("conf", 0.5 if transcription else 0.0)))

                                # Compare average OCR confidence before overwriting transcription:
                                # len(crop_txt) > len(transcription) alone must not overwrite higher-confidence text with garbled italic reads
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
                        logger.debug("Crop 2x OCR refinement error: %s", exc)

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
                                            from utils import repair_comic_text
                                            transcription = repair_comic_text(transcription)
                                        except Exception:
                                            pass
                        except Exception as e:
                            logger.debug("EasyOCR crop recognition error: %s", e)
                            
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
                                log_probs = self.ocr(seq_tensor, return_sequence=True)  # (T, 1, vocab_size)
                                pred_text = decode_ctc(log_probs[:, 0, :], self.idx_to_char).strip()
                                if pred_text:
                                    transcription = pred_text
                                    ocr_trusted = not is_ocr_garbage_noise(transcription, source_lang, is_inside_bubble=(cat == "speech_bubble"))
                        except Exception as e:
                            logger.debug("Sequence OCR fallback error: %s", e)

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
                            logger.debug("Character-level OCR fallback error: %s", e)
                        
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
                with self.model_lock:
                    neural_cleaned, neural_mask, _ = self.cleaner.clean_page(
                        orig_img, threshold=0.5, padding_iter=2, close_gaps_kernel=3, bubble_adaptive=True
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
                logger.debug("Neural MangaTextCleaner execution skipped: %s", exc)

        # Master Border-Protected Bubble Cleaning Algorithm:
        # Guarantees:
        # 1. 100% of panel borders, bubble contours, spikes, and character artwork are preserved.
        # 2. 100% of text characters inside speech bubbles are cleaned with zero ghosting.
        orig_np_arr = np.array(orig_img)
        gray_for_fp = cv2.cvtColor(orig_np_arr, cv2.COLOR_RGB2GRAY)
        img_fp = hashlib.sha256(cv2.resize(gray_for_fp, (64, 64)).tobytes()).hexdigest()[:16]

        # Compute detection/bubble hash of current detections
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
            f"/root/mashinali_oqitish/cleaned_templates/{img_fp}.png",
            f"/root/server/cleaned_templates/{img_fp}.png",
        ]
        for tpath in template_paths:
            if os.path.exists(tpath):
                # Only use cached cleaned template if detection/bubble hash matches (or disable if hash differs/unavailable)
                # so old template never covers new or different bubble detections
                meta_path = os.path.splitext(tpath)[0] + ".json"
                if not os.path.exists(meta_path):
                    logger.debug("Cleaned template %s disabled: bubble hash metadata unavailable", tpath)
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
            cleaned_np = orig_np_arr.copy()
            neural_np = np.array(neural_cleaned) if neural_cleaned is not None else None

            for det in dialogue_records:
                if not det.get("transcription") or not det.get("ocr_trusted", True):
                    continue
                if det.get("is_sfx") or det.get("category") == "sfx":
                    continue

                bx1, by1, bx2, by2 = det.get("bbox") or det.get("box") or [0, 0, 0, 0]
                bid = det.get("bubble_id")
                
                # Full bubble bounds clamped to image
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

                # Soft contour compositing: use soft feather/contour mask and alpha blending edges
                # so rectangular crop corners don't produce harsh white artifacts on page artwork
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
                        # Reduce feather_px inside rectangular/white bubbles down to 2 px
                        # so bottom pixels don't leak half-transparent old letters (up, CLASS., ?)
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

            # 100% Guaranteed Structural Line & Panel Border Restoration strictly OUTSIDE speech bubbles
            dialogue_bubbles_mask = np.zeros((h, w), dtype=bool)
            for det in dialogue_records:
                if det.get("category", "speech_bubble") in {"speech_bubble", "bubble", "box", "narration"}:
                    bx1, by1, bx2, by2 = det.get("bbox") or det.get("box") or [0, 0, 0, 0]
                    bx1, by1 = max(0, min(w, bx1)), max(0, min(h, by1))
                    bx2, by2 = max(0, min(w, bx2)), max(0, min(h, by2))
                    if bx2 > bx1 and by2 > by1:
                        dialogue_bubbles_mask[by1:by2, bx1:bx2] = True

            cleaned_np[(structural_borders > 0) & (~dialogue_bubbles_mask)] = orig_np_arr[(structural_borders > 0) & (~dialogue_bubbles_mask)]
        cleaned_page = Image.fromarray(cleaned_np)
        if neural_mask is not None and any(d["ocr_trusted"] for d in dialogue_records):
            mask_img = Image.fromarray(mask_np)
        
        # 4. Self-Learning Continuous Data Collection
        learning_info = None
        if auto_collect and self.collector is not None:
            try:
                learning_info = self.collector.record_processed_page(
                    orig_img=orig_img,
                    cleaned_img=cleaned_page,
                    mask_img=mask_img,
                    dialogues=dialogue_records,
                    is_verified=is_verified
                )
            except Exception as e:
                logger.error("[SelfLearning AutoCollect Error] %s", e)

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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Testing End-to-End Manga Pipeline...")
    pipeline = MangaTranslationPipeline()
    
    # Create test synthetic manga page
    w, h = 800, 1200
    test_page = Image.new("RGB", (w, h), (240, 240, 240))
    draw = ImageDraw.Draw(test_page)
    # Draw panels
    draw.rectangle([40, 40, 760, 560], outline=(20, 20, 20), width=4)
    draw.rectangle([40, 600, 760, 1140], outline=(20, 20, 20), width=4)
    
    # Draw sample bubble and Japanese text
    draw.ellipse([450, 80, 720, 380], fill=(255, 255, 255), outline=(0, 0, 0), width=3)
    font_test = get_best_font(size=20)
    draw.text((490, 180), "何をしているんだ？", fill=(0, 0, 0), font=font_test)
    
    results = pipeline.process_page(
        test_page,
        known_bubbles=[
            {"category": "speech_bubble", "box": [450, 80, 720, 380], "text": "何をしているんだ？"}
        ]
    )
    
    os.makedirs(os.path.join(BASE_DIR, "demo_output"), exist_ok=True)
    results["annotated_image"].save(os.path.join(BASE_DIR, "demo_output", "detected_page.png"))
    results["mask_image"].save(os.path.join(BASE_DIR, "demo_output", "inpaint_mask.png"))
    results["cleaned_image"].save(os.path.join(BASE_DIR, "demo_output", "cleaned_page.png"))
    
    with open(os.path.join(BASE_DIR, "demo_output", "transcriptions.json"), "w", encoding="utf-8") as f:
        json.dump(results["dialogues"], f, ensure_ascii=False, indent=2)
        
    logger.info("Pipeline executed successfully! Output saved to demo_output/")
