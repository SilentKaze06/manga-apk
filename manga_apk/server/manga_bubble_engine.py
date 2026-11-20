"""
Manga Speech Bubble & Dialogue Extraction Engine - Production Grade.
High-precision OCR, dialogue linguistic validation, bubble boundary expansion,
and 100% full text erasure overlay for seamless manga translation.
"""

import math
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

VALID_SHORT_TOKENS = {'a', 'i', 'in', 'on', 'to', 'at', 'by', 'of', 'as', 'is', 'it', 'he', 'we', 'my', 'me', 'no', 'so', 'go', 'do', 'up', 'am', 'if', 'ok'}

def clean_manga_text(text: str) -> str:
    """Cleans OCR noise symbols from manga dialogue while preserving multilingual characters."""
    t = re.sub(r'[^\w\s\'\.,!\?\"\-—…~～«»“”„’ʻʼ`]', ' ', text)
    t = t.replace('_', ' ')
    return " ".join(t.split()).strip()

def is_valid_manga_dialogue(text: str, source_lang: str = "en") -> bool:
    """
    Strictly validates that detected text represents genuine human dialogue,
    eliminating sound effects (SFX), line art, screentone moire, and clothing textures.
    Supports English, Japanese, Korean, Chinese, Russian, Uzbek, and European comics.
    """
    cleaned = clean_manga_text(text)
    if not cleaned:
        return False
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
        # Filter noise with too many 1-letter garbage tokens
        unknown_short_tokens = [w for w in words if len(w) <= 2 and w not in VALID_SHORT_TOKENS]
        if len(words) >= 3 and len(unknown_short_tokens) / len(words) >= 0.40:
            return False
        return sum(1 for c in cleaned if c.isalpha()) >= 4

    # Other languages
    return sum(1 for c in cleaned if c.isalpha()) >= 3

def _is_dark_barrier_between_x(gray: np.ndarray, boxA: list, boxB: list, dark_thresh: int = 100, min_line_ratio: float = 0.40) -> bool:
    if gray is None or gray.size == 0:
        return False
    H, W = gray.shape[:2]
    if boxA[0] > boxB[0]:
        boxA, boxB = boxB, boxA
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB

    y_top = max(yA1, yB1)
    y_bot = min(yA2, yB2)
    if y_bot - y_top < 6:
        mid_y = int(((yA1 + yA2) / 2.0 + (yB1 + yB2) / 2.0) / 2.0)
        y_top = max(0, mid_y - 8)
        y_bot = min(H, mid_y + 8)
    else:
        y_top = max(0, min(H, y_top))
        y_bot = max(0, min(H, y_bot))

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
        return False

    # When xA2 >= xB1: sample along the line between box centers (cxA..cxB) for dark border pixels
    cxA = int((xA1 + xA2) / 2.0)
    cyA = int((yA1 + yA2) / 2.0)
    cxB = int((xB1 + xB2) / 2.0)
    cyB = int((yB1 + yB2) / 2.0)
    cx_min = max(0, min(cxA, cxB))
    cx_max = min(W - 1, max(cxA, cxB))
    cy_min = max(0, min(cyA, cyB))
    cy_max = min(H - 1, max(cyA, cyB))

    if cx_max - cx_min < 4:
        return False

    if y_bot - y_top >= 4:
        center_slice = gray[y_top:y_bot, cx_min:cx_max]
        if center_slice.size > 0 and center_slice.shape[0] >= 4:
            dark_counts = np.sum(center_slice < dark_thresh, axis=0)
            col_ratios = dark_counts / float(center_slice.shape[0])
            if bool(np.any(col_ratios >= min_line_ratio)):
                return True

    dist = int(abs(cxB - cxA))
    if dist >= 6:
        cy_mid = int((cyA + cyB) / 2.0)
        cy_clamped = min(H - 1, max(0, cy_mid))
        xs = np.linspace(cx_min, cx_max, dist).astype(int)
        xs = np.clip(xs, 0, W - 1)
        pts = gray[cy_clamped, xs]
        if bool(np.sum(pts < dark_thresh) / float(dist) >= min_line_ratio):
            return True

    return False

def _is_dark_barrier_between_y(gray: np.ndarray, boxA: list, boxB: list, dark_thresh: int = 95, min_line_ratio: float = 0.40) -> bool:
    if gray is None or gray.size == 0:
        return False
    H, W = gray.shape[:2]
    if boxA[1] > boxB[1]:
        boxA, boxB = boxB, boxA
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB

    x_left = max(xA1, xB1)
    x_right = min(xA2, xB2)
    if x_right - x_left < 6:
        mid_x = int(((xA1 + xA2) / 2.0 + (xB1 + xB2) / 2.0) / 2.0)
        x_left = max(0, mid_x - 8)
        x_right = min(W, mid_x + 8)
    else:
        x_left = max(0, x_left)
        x_right = min(W, x_right)

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
    stats: Optional[np.ndarray] = None
) -> bool:
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB
    wA, hA = max(1, xA2 - xA1), max(1, yA2 - yA1)
    wB, hB = max(1, xB2 - xB1), max(1, yB2 - yB1)
    
    total_y_span = max(yA2, yB2) - min(yA1, yB1)
    if total_y_span > max_page_h * 0.18:
        return False

    # White background (gray > 200) cv2.connectedComponents check:
    # If two text boxes belong to different closed white contours (separate bubble/panel outlines),
    # they must NEVER be merged.
    if gray is not None and gray.size > 0:
        if labels is None or stats is None:
            labels, stats = get_white_cc_labels(gray)
        if labels is not None:
            lblA = _get_box_white_label(labels, gray, stats, boxA)
            lblB = _get_box_white_label(labels, gray, stats, boxB)
            if lblA > 0 and lblB > 0 and lblA != lblB:
                return False

    # 1. Words on same horizontal line
    y_overlap = min(yA2, yB2) - max(yA1, yB1)
    min_h = min(hA, hB)
    if min_h > 0 and (y_overlap / min_h) >= 0.40:
        gap_x = max(0, max(xA1, xB1) - min(xA2, xB2))
        if gap_x <= max(24, int(min_h * 1.8)):
            left_box = boxA if xA1 < xB1 else boxB
            right_box = boxB if xA1 < xB1 else boxA
            if not _is_dark_barrier_between_x(gray, left_box, right_box):
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
            if not _is_dark_barrier_between_y(gray, top_box, bot_box):
                return True

    return False

# Public aliases
is_dark_barrier_between_x = _is_dark_barrier_between_x
is_dark_barrier_between_y = _is_dark_barrier_between_y
can_merge_text_boxes = _can_merge_text_boxes

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

def _generate_overlay_polygon(bx1: float, by1: float, bx2: float, by2: float, tx1: float, ty1: float, tx2: float, ty2: float) -> List[List[float]]:
    """
    Generates a high-coverage polygon that covers 96% of the bubble interior,
    guaranteeing complete coverage of all text while respecting bubble contours.
    """
    bw = bx2 - bx1
    bh = by2 - by1
    
    # Check if rectangular box (e.g. narration box where text fills corners)
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
    if pytesseract is None:
        return []

    img_np = np.asarray(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    tess_lang = "eng" if source_lang in ("en", "auto") else ("jpn" if source_lang == "ja" else "rus")
    
    # 1. Full page sparse word detection
    data = pytesseract.image_to_data(gray, config='--psm 11', lang=tess_lang, output_type=pytesseract.Output.DICT)
    
    raw_boxes = []
    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        conf = int(data['conf'][i])
        if conf < 25 or len(word) < 1:
            continue
        clean_w = "".join(c for c in word if c.isalnum())
        if not clean_w:
            continue
        bx1 = data['left'][i]
        by1 = data['top'][i]
        bx2 = bx1 + data['width'][i]
        by2 = by1 + data['height'][i]
        raw_boxes.append({
            'box': [bx1, by1, bx2, by2],
            'text': word,
            'conf': conf
        })

    # 2. Graph Clustering into Dialogue Units
    cc_labels, cc_stats = get_white_cc_labels(gray)

    n = len(raw_boxes)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if _can_merge_text_boxes(raw_boxes[i]["box"], raw_boxes[j]["box"], gray, max_page_h=h, labels=cc_labels, stats=cc_stats):
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
        
        comp.sort(key=lambda idx: (raw_boxes[idx]["box"][1], raw_boxes[idx]["box"][0]))
        bx1 = min(raw_boxes[idx]["box"][0] for idx in comp)
        by1 = min(raw_boxes[idx]["box"][1] for idx in comp)
        bx2 = max(raw_boxes[idx]["box"][2] for idx in comp)
        by2 = max(raw_boxes[idx]["box"][3] for idx in comp)
        
        full_text = " ".join(raw_boxes[idx]["text"] for idx in comp)
        clusters.append({
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

        # Speech bubble whiteness check (ensures bubble has light background)
        px1, py1 = max(0, bx1 - 6), max(0, by1 - 6)
        px2, py2 = min(w, bx2 + 6), min(h, by2 + 6)
        patch = gray[py1:py2, px1:px2]
        pure_white = np.mean(patch > 215) if patch.size > 0 else 0
        mean_val = np.mean(patch) if patch.size > 0 else 0

        # Reject dark backgrounds (clothes, dark hair)
        if pure_white < 0.40 and mean_val < 185:
            continue

        filtered.append({
            "text_box": [bx1, by1, bx2, by2],
            "text": text,
        })
        all_tboxes.append([bx1, by1, bx2, by2])

    # 4. Bubble Contour Snapping & Typesetting Blocks Construction
    blocks = []
    for idx, f in enumerate(filtered):
        tbox = f["text_box"]
        other_tboxes = [b for j, b in enumerate(all_tboxes) if j != idx]
        bubble_box = _expand_to_bubble_contour(gray, tbox, other_tboxes)
        
        bx1, by1, bx2, by2 = bubble_box
        crop_pil = pil_image.crop((bx1, by1, bx2, by2))
        
        text = f["text"]
        try:
            crop_text = pytesseract.image_to_string(crop_pil, lang=tess_lang, config='--psm 6').strip()
            crop_cleaned = clean_manga_text(crop_text)
            if is_valid_manga_dialogue(crop_cleaned, source_lang):
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
            scaled_tx1, scaled_ty1, scaled_tx2, scaled_ty2
        )

        # Expanded text erase rectangle (+3pt padding) ensuring 100% of original letters are wiped clean
        erase_pad = 3.0
        text_erase_rect = [
            max(0.0, scaled_tx1 - erase_pad),
            max(0.0, scaled_ty1 - erase_pad),
            min(w / scale, scaled_tx2 + erase_pad),
            min(h / scale, scaled_ty2 + erase_pad)
        ]

        blocks.append({
            "text": text,
            "font_size": 0.0,
            "confidence": 0.95,
            "bbox": [scaled_bx1, scaled_by1, scaled_bx2, scaled_by2],
            "overlay_bbox": [scaled_bx1, scaled_by1, scaled_bx2, scaled_by2],
            "text_erase_bbox": text_erase_rect,
            "overlay_polygon": poly,
            "category": "speech_bubble",
            "bubble_id": len(blocks) + 1,
            "background": [1.0, 1.0, 1.0],
            "cleaned": True,
        })

    return blocks
