"""
OCR engine utilities, crop preparation, noise filters, and text normalization.
"""
import logging
import re
from typing import Union, Tuple, Optional, Dict, Any
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def analyze_script_distribution(text: str) -> Dict[str, Any]:
    """
    Analyzes the script character distribution across Unicode ranges:
    - hangul: Syllables, Jamo, Compatibility Jamo (\\uAC00-\\uD7AF, \\u1100-\\u11FF, \\u3130-\\u318F, \\uA960-\\uA97F, \\uD7B0-\\uD7FF)
    - cyrillic: Basic, Supplementary, Extended (\\u0400-\\u04FF, \\u0500-\\u052F, \\u2DE0-\\u2DFF, \\uA640-\\uA69F, \\u1C80-\\u1C8F)
    - kana: Hiragana, Katakana, Small Kana, Halfwidth Katakana (\\u3040-\\u309F, \\u30A0-\\u30FF, \\uFF65-\\uFF9F, \\u31F0-\\u31FF)
    - cjk: Unified Ideographs, Extension A (\\u4E00-\\u9FFF, \\u3400-\\u4DBF)
    - latin: Basic Latin, Latin-1 Supplement, Latin Extended-A/B (\\u0041-\\u005A, \\u0061-\\u007A, \\u00C0-\\u024F, \\u1E00-\\u1EFF)
    - digits: ASCII and fullwidth digits
    """
    if not text:
        return {
            "hangul": 0, "cyrillic": 0, "kana": 0, "cjk": 0, "latin": 0,
            "digits": 0, "has_jp_punct": False, "total_letters": 0
        }

    hangul = len(re.findall(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uD7B0-\uD7FF]", text))
    cyrillic = len(re.findall(r"[\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F\u1C80-\u1C8F]", text))
    kana = len(re.findall(r"[\u3040-\u309F\u30A0-\u30FF\uFF65-\uFF9F\u31F0-\u31FF]", text))
    cjk = len(re.findall(r"[\u4E00-\u9FFF\u3400-\u4DBF]", text))
    latin = len(re.findall(r"[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF]", text))
    digits = len(re.findall(r"[\d\uFF10-\uFF19]", text))

    # Japanese specific punctuation (quotations 「 」, 『 』, prolonged sound mark ー \u30FC, wave dash 〜 \u301C)
    has_jp_punct = bool(re.search(r"[\u3001-\u3003\u3008-\u3011\u3014-\u301F\u30FC]", text))

    total = hangul + cyrillic + kana + cjk + latin
    return {
        "hangul": hangul,
        "cyrillic": cyrillic,
        "kana": kana,
        "cjk": cjk,
        "latin": latin,
        "digits": digits,
        "has_jp_punct": has_jp_punct,
        "total_letters": total,
    }


def detect_primary_script_from_text(text: str) -> str:
    """
    Detects the primary script/language from text character sets:
    - Hangul -> 'ko' (Korean)
    - Cyrillic -> 'ru' (Russian / Cyrillic)
    - Kana (Hiragana / Katakana) or Japanese prolonged sound/punctuation -> 'ja' (Japanese)
    - CJK Ideographs (Hanzi / Kanji) -> 'ja' if Japanese cues/context exist, otherwise 'zh' (Chinese)
    - Latin -> 'en' (English / Latin alphabet)
    """
    if not text:
        return "ja"

    counts = analyze_script_distribution(text)
    hangul = counts["hangul"]
    cyrillic = counts["cyrillic"]
    kana = counts["kana"]
    cjk = counts["cjk"]
    latin = counts["latin"]
    has_jp_punct = counts["has_jp_punct"]

    if hangul > 0 and hangul >= max(cyrillic, kana):
        return "ko"
    if cyrillic > 0 and cyrillic >= max(kana, latin):
        return "ru"
    if kana > 0 or (cjk > 0 and has_jp_punct):
        return "ja"
    if cjk > 0:
        return "zh"
    if latin > 0 and latin >= max(cjk, kana, cyrillic, hangul):
        return "en"
    return "ja"


def filter_manga_ocr_hallucinations(text: str) -> str:
    """
    Suppresses MangaOCR / OCR model repetitive hallucination loops.
    E.g.:
    - 'ああああああああああ' -> 'あああ'
    - '「「「「「' -> '「'
    - 'wwwwwwww' -> 'www'
    - '......' -> '...'
    """
    if not text:
        return ""
    t = text.strip()
    # Collapse repetitive character loops of >= 4 to max 3
    t = re.sub(r'([^\d\s])\1{3,}', r'\1\1\1', t)
    # Collapse repetitive 2-char tokens (e.g. 'のののの' or 'ああ')
    t = re.sub(r'(.{2,}?)\1{3,}', r'\1\1', t)
    # Filter single-token loops like '「 「 「'
    t = re.sub(r'(\b\S+\b)(\s+\1){3,}', r'\1 \1', t)
    return t.strip()


def is_dark_background(
    img: Union[np.ndarray, Image.Image],
    median_thresh: float = 110.0,
    contrast_p95: float = 160.0
) -> bool:
    """
    Checks if an image patch or crop represents a dark background with bright text
    (e.g., black speech bubble, dark narration frame, or inverted shock dialogue).
    """
    if img is None:
        return False
    if isinstance(img, Image.Image):
        gray = np.array(img.convert("L"))
    elif isinstance(img, np.ndarray):
        if img.size == 0:
            return False
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
    else:
        return False

    if gray.size == 0:
        return False

    median_val = float(np.median(gray))
    p95_val = float(np.percentile(gray, 95))
    return median_val < median_thresh and p95_val > contrast_p95


def apply_dark_box_clahe_inversion(
    img: Union[np.ndarray, Image.Image],
    clip_limit: float = 2.0,
    tile_size: Tuple[int, int] = (8, 8)
) -> Union[np.ndarray, Image.Image]:
    """
    Inverts dark background with bright text into high-contrast dark text on bright background
    and applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Guarantees optimal edge definition for MangaOCR, PaddleOCR, and EasyOCR.
    """
    is_pil = isinstance(img, Image.Image)
    if is_pil:
        np_arr = np.array(img.convert("RGB"))
    else:
        np_arr = img.copy()

    if np_arr.size == 0:
        return img

    if np_arr.ndim == 3:
        gray = cv2.cvtColor(np_arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = np_arr

    # Invert so white text becomes dark text and dark background becomes bright
    inv_gray = 255 - gray
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced_gray = clahe.apply(inv_gray)

    if is_pil or (isinstance(img, np.ndarray) and img.ndim == 3):
        enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(enhanced_rgb) if is_pil else enhanced_rgb
    return enhanced_gray


def enhance_crop_for_ocr(
    crop_img: Union[np.ndarray, Image.Image],
    allow_dark_inversion: bool = True,
    scale_2x: bool = True
) -> Tuple[Union[np.ndarray, Image.Image], bool]:
    """
    Full preprocessing pipeline for text crop recognition:
    1. Checks if crop has a dark background.
    2. If dark, applies CLAHE inversion.
    3. If 2x scaling is requested, applies cubic interpolation upscaling with unsharp masking.
    Returns (processed_crop, was_dark_inverted).
    """
    if crop_img is None:
        return crop_img, False

    is_dark = is_dark_background(crop_img)
    if is_dark and allow_dark_inversion:
        enhanced = apply_dark_box_clahe_inversion(crop_img)
        was_dark = True
    else:
        enhanced = crop_img
        was_dark = False

    if not scale_2x:
        return enhanced, was_dark

    if isinstance(enhanced, Image.Image):
        w, h = enhanced.size
        if w > 0 and h > 0 and (w < 400 or h < 200):
            resample = getattr(Image, "Resampling", Image).BICUBIC
            enhanced = enhanced.resize((w * 2, h * 2), resample)
    elif isinstance(enhanced, np.ndarray) and enhanced.size > 0:
        h, w = enhanced.shape[:2]
        if w > 0 and h > 0 and (w < 400 or h < 200):
            enhanced = cv2.resize(enhanced, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            if enhanced.ndim == 2 or (enhanced.ndim == 3 and enhanced.shape[2] == 1):
                # Unsharp mask for crisp character stroke edges
                blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
                enhanced = cv2.addWeighted(enhanced, 1.25, blurred, -0.25, 0)

    return enhanced, was_dark


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
    # Repetitive single-character hallucinations (e.g. 「「「「「 or あああああああああ)
    compact_chars = re.sub(r'\s+', '', t_stripped)
    if len(compact_chars) >= 6 and len(set(compact_chars)) == 1:
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
    except Exception as exc:
        logger.debug("try_read_vertical_strip OCR error: %s", exc)
    return None
