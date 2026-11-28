"""
OCR engine utilities, crop preparation, noise filters, and text normalization.
"""
import logging
import re
from typing import Union, Tuple, Optional, Dict, Any
import cv2
import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


def analyze_script_distribution(text: str) -> Dict[str, Any]:
    """
    Analyzes the script character distribution across Unicode ranges:
    - hangul: Syllables, Jamo, Compatibility Jamo (\\uAC00-\\uD7AF, \\u1100-\\u11FF, \\u3130-\\u318F, \\uA960-\\uA97F, \\uD7B0-\\uD7FF)
    - cyrillic: Basic, Supplementary, Extended (\\u0400-\\u04FF, \\u0500-\\u052F, \\u2DE0-\\u2DFF, \\uA640-\\uA69F, \\u1C80-\\u1C8F)
    - kana: Hiragana, Katakana, Small Kana, Halfwidth Katakana (\\u3040-\\u309F, \\u30A0-\\u30FF, \\uFF65-\\uFF9F, \\u31F0-\\u31FF)
    - cjk: Unified Ideographs, Extension A/B+, Compatibility (\\u4E00-\\u9FFF, \\u3400-\\u4DBF, \\uF900-\\uFAFF)
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
    cjk = len(re.findall(r"[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF\U00020000-\U0002A6DF]", text))
    latin = len(re.findall(r"[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF]", text))
    digits = len(re.findall(r"[\d\uFF10-\uFF19]", text))

    # Exclusive Japanese cues: Kana or Japanese prolonged sound mark ー (\u30FC).
    # Shared CJK punctuation (、, 。, 「」, 【】, 《》) is used in Chinese too and must NOT force 'ja'.
    has_jp_punct = bool(re.search(r"[\u3040-\u309F\u30A0-\u30FF\uFF65-\uFF9F\u31F0-\u31FF\u30FC]", text))

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
    Detects the primary script/language from text character sets using dominant script share:
    - Hangul -> 'ko' (Korean)
    - Cyrillic -> 'ru' (Russian / Cyrillic)
    - Kana or ー (with/without CJK) -> 'ja' (Japanese)
    - CJK Ideographs without Kana/ー -> 'zh' (Chinese)
    - Latin -> 'en' (English / Latin alphabet)
    - Empty or punctuation-only -> 'auto'
    """
    if not text:
        return "auto"

    counts = analyze_script_distribution(text)
    if counts["total_letters"] == 0:
        return "auto"

    hangul = counts["hangul"]
    cyrillic = counts["cyrillic"]
    kana = counts["kana"]
    cjk = counts["cjk"]
    latin = counts["latin"]
    has_jp = (kana > 0) or counts["has_jp_punct"]

    scores = {
        "ko": hangul,
        "ru": cyrillic,
        "ja": (kana + cjk) if has_jp else 0,
        "zh": cjk if not has_jp else 0,
        "en": latin,
    }
    best_lang = max(("ja", "ko", "zh", "ru", "en"), key=lambda lang_k: scores[lang_k])
    if scores[best_lang] <= 0:
        return "auto"
    return best_lang


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
    p99_val = float(np.percentile(gray, 99))
    max_val = float(np.max(gray))
    has_bimodal_bright = (p95_val > contrast_p95) or (p99_val > 155.0 and max_val > 175.0 and (p99_val - median_val) > 50.0)
    return median_val < median_thresh and has_bimodal_bright


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
            enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1.2, percent=125, threshold=2))
    elif isinstance(enhanced, np.ndarray) and enhanced.size > 0:
        h, w = enhanced.shape[:2]
        if w > 0 and h > 0 and (w < 400 or h < 200):
            enhanced = cv2.resize(enhanced, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
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
    # Normalize common OCR misreadings where '!' was read as 1, l, or | next to punctuation (never strip 'I')
    normalized = re.sub(r'(?<=[?!])\s*[1l|]\b', '!', t)
    normalized = re.sub(r'\b[1l|]\s*(?=[?!])', '!', normalized)

    # Check if all remaining characters are punctuation symbols
    clean = re.sub(r"[\s\.,!?'\"_\-~:;#@$%^&*()\[\]{}|<>/\\`+=…¿¡]", "", normalized)
    if re.fullmatch(r"[.\s…]+", normalized):
        return True
    if not clean:
        return any(c in "?!…~" or ".." in normalized or "--" in normalized for c in normalized)

    # Also handle cases where a single digit was misread attached to heavy punctuation, e.g. "?1"
    if len(clean) == 1 and clean.isdigit() and any(c in "?!…" for c in normalized) and len(t) <= 4:
        return True

    return False


def normalize_manga_punctuation(text: str) -> str:
    """Normalizes manga punctuation OCR quirks into clean comic typography."""
    t = text.strip()
    t = re.sub(r'(?<=[?!])\s*[1l|]\b', '!', t)
    t = re.sub(r'\b[1l|]\s*(?=[?!])', '!', t)
    compact = "".join(t.split())
    if re.fullmatch(r"[.…]+", compact):
        return "..."
    if re.match(r'^[?!1l|\s]+$', t):
        has_q = "?" in t
        has_ex = any(c in "!1l|" for c in t)
        if has_q and has_ex:
            return "?!"
        elif has_q:
            return "?" * min(3, t.count("?"))
        elif has_ex:
            return "!" * min(3, sum(1 for c in t if c in "!1lI|"))
    return t


_DIALOGUE_INTERJECTIONS = {
    "no", "so", "oh", "ah", "eh", "uh", "um", "hm", "hmm", "m", "ugh", "ooh", "ahh",
    "ohh", "uhh", "hey", "why", "way", "yay", "wow", "ow", "aw", "ha", "he",
    "hi", "ok", "yes", "nah", "yea", "yeah", "wait", "stop", "what", "whoa",
    "go", "me", "my", "we", "you", "mr", "mrs", "ms", "dr", "st", "vs", "tv",
    "hp", "mp", "xp", "lv", "id", "pc", "ai", "dj", "gm", "mc", "ps", "lol",
    "brb", "omg", "wtf", "btw", "idk", "tbh", "please", "really", "sorry",
    "sh", "shh", "psst", "tss", "tsk", "mph", "hmph", "run", "yoq", "yo'q",
    "нет", "да", "не", "ну", "ой", "эй", "ай", "ах", "ох", "ух", "эх", "хм",
    "мм", "эм", "ага", "угу", "что", "а",
}


def is_sfx_text(text: str) -> bool:
    """
    Detects sound effects (SFX) drawn directly on manga artwork or near bubbles.
    Catches elongated repeated characters (e.g. VREEEEM, Vzeeeem, booom),
    consonant-heavy words (zksh, clck), and standard comic SFX words (Ding, Boom, etc.).
    """
    if not text:
        return False
    words = text.split()
    if len(words) > 2:
        return False

    clean = re.sub(r'[^\w]', '', text).lower()
    if not clean:
        return False

    known_sfx = {
        "ding", "zksh", "vreee", "vreeem", "vzeeeem", "vre", "zrk", "zrrk", "bzz",
        "boom", "thump", "doki", "tok", "tap", "sfx", "flik", "krrr",
        "wham", "clang", "clack", "bang", "snap", "crash", "smash",
        "gasp", "sigh", "whoosh", "zap", "crunch", "giggle", "snicker", "chuckle"
    }

    word_tokens = [re.sub(r'[^\w]', '', w).lower() for w in words]
    word_tokens = [w for w in word_tokens if w]
    if not word_tokens:
        return False

    def _is_dialogue_interjection(tok: str) -> bool:
        collapsed = re.sub(r'(.)\1+', r'\1', tok)
        return tok in _DIALOGUE_INTERJECTIONS or collapsed in _DIALOGUE_INTERJECTIONS

    # Do not let whitelisted dialogue interjections or elongated exclamations trigger SFX
    if any(_is_dialogue_interjection(t) for t in word_tokens):
        return False

    if len(word_tokens) == 1 and clean in known_sfx:
        return True
    if len(word_tokens) >= 2 and all(t in known_sfx for t in word_tokens):
        return True

    for w_clean in word_tokens:
        if re.search(r'([^\d\s])\1{2,}', w_clean):
            return True

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
        if re.match(r'^\d+-', w):
            fixed_words.append(w)
            continue
        w_clean = re.sub(r'[^\w]', '', w)
        w_cyr = sum(1 for c in w_clean if '\u0400' <= c <= '\u04ff')
        w_lat = sum(1 for c in w_clean if ('a' <= c <= 'z') or ('A' <= c <= 'Z'))
        # Never corrupt pure digits ("1000") or pure ASCII words/acronyms ("HP", "Dungeon")
        is_mixed_cyr_word = (w_cyr > 0)
        is_pure_homoglyph_word = (
            predom_cyr and w_cyr == 0 and w_lat >= 3
            and not w_clean.isdigit()
            and not (w_clean.isupper() and w_lat <= 4)
            and all(c in homoglyphs for c in w_clean if c.isalpha())
        )
        if is_mixed_cyr_word or is_pure_homoglyph_word:
            new_chars = []
            for ch in w:
                if ch in homoglyphs:
                    new_chars.append(homoglyphs[ch])
                elif ch == '0' and w_cyr > 0 and len(w_clean) > 1:
                    new_chars.append('О')
                elif ch == '3' and w_cyr > 0 and len(w_clean) > 1:
                    new_chars.append('З')
                else:
                    new_chars.append(ch)
            fixed_words.append("".join(new_chars))
        else:
            fixed_words.append(w)
    return " ".join(fixed_words)


_VALID_SHORT_TOKENS = {
    # English
    "i", "a", "no", "oh", "ok", "ah", "eh", "uh", "um", "yo", "ya", "ha",
    "hm", "mm", "so", "to", "he", "we", "me", "my", "in", "on", "at", "by", "if",
    "da", "hi", "it", "is", "am", "as", "us", "up", "do",
    "go", "an", "or", "of", "be", "ex", "re", "ut", "im", "id", "ll", "ve", "tv", "mr", "ms", "dr", "st",
    # Russian (Cyrillic single letters and 2-letter words/SFX)
    "я", "и", "а", "о", "у", "э", "в", "с", "к", "он", "ты", "мы", "вы",
    "ну", "но", "не", "да", "до", "по", "за", "из", "на", "от", "со", "ко",
    "об", "же", "ли", "бы", "ее", "её", "их", "ей", "им", "то", "те", "та", "ту",
    "эх", "ах", "ох", "ух", "эй", "ай", "ой", "хм", "гм", "ха", "хе", "хи",
    "чу", "ба", "фу", "уф", "топ", "тап", "тук", "кап", "бах", "бум", "пик",
    "ум", "ась", "уж",
    # Uzbek
    "va", "bu", "shu", "ha", "u",
}


def is_ocr_garbage_noise(text: str, source_lang: str = "auto", is_inside_bubble: bool = False) -> bool:
    """Detects OCR hallucinations from manga screentones, halftones, speed lines, and noise."""
    valid_shorts = _VALID_SHORT_TOKENS
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

    # Bracket or symbol-based noise patterns (allow %, $, #, = and & * / [ ] < > when paired with dialogue/stats)
    if any(c in t_stripped for c in "^\\{}|`"):
        return True
    if any(c in t_stripped for c in "&*/[]<>"):
        compact_sym = re.sub(r'\s+', '', t_stripped)
        alnum_cnt = sum(1 for ch in compact_sym if ch.isalnum())
        alnum_ratio = alnum_cnt / max(1, len(compact_sym))
        if alnum_cnt == 0 or (not is_inside_bubble and alnum_ratio < 0.50):
            return True
    if any(c in t_stripped for c in "%$#="):
        if not any(c.isalnum() for c in t_stripped):
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
    has_hangul = any('\uac00' <= c <= '\ud7af' or '\u1100' <= c <= '\u11ff' or '\u3130' <= c <= '\u318f' for c in clean)
    has_cjk = any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' for c in clean)

    # Filter Japanese/Artwork SFX before translation ONLY when explicit Latin/Cyrillic source_lang is set
    if source_lang in {"en", "ru", "fr", "de", "es", "pt", "it", "tr", "nl", "pl", "uz"} and not is_inside_bubble:
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

    # Pure numbers: keep percentages, prices, or numbers inside bubbles (e.g. "100%", "$50", "#1")
    if letters == 0 and digits > 0:
        if is_inside_bubble or any(c in t_stripped for c in "%$#"):
            return False
        return True

    # Check mixed alphanumeric tokens at word level (allow game/manga stats like Lv5, S1, AK-47)
    for w in t.split():
        w_clean = re.sub(r'[^\w]', '', w)
        w_letters = sum(1 for c in w_clean if c.isalpha())
        w_digits = sum(1 for c in w_clean if c.isdigit())
        if w_digits > 0 and w_letters > 0:
            is_valid_alnum = bool(re.search(
                r'^(?:\d{1,2}:\d{2}|\d+-(?:й|го|я|е|th|st|nd|rd)|1st|2nd|3rd|\d+th|\d+pm|\d+am|(?:lv|hp|mp|xp|sp|exp|atk|def|ch|ep|s|a|b|c|d|e|f)\d+|[A-Za-z]{1,4}\d{1,3})$',
                w_clean,
                re.I,
            )) or ('-' in w and len(w_clean) >= 3)
            if not is_valid_alnum:
                thresh = 0.60 if is_inside_bubble else 0.45
                if w_digits / float(w_digits + w_letters) > thresh:
                    return True

    # Standalone 1-2 char noise check
    if len(clean) <= 2:
        if clean.lower() not in valid_shorts:
            if has_hangul:
                if source_lang in {"en", "ru"} and not is_inside_bubble:
                    return True
            elif has_cjk:
                if source_lang in {"en", "ru"} and not is_inside_bubble:
                    return True
            else:
                return True

    # In Russian mode or if Cyrillic is present:
    # Check Latin interference per-word rather than concatenated string so "Идем в Dungeon" passes cleanly
    if has_cyrillic or source_lang in {"ru", "russian"}:
        if has_latin:
            for w in t.split():
                w_clean = re.sub(r'[^\w]', '', w)
                if not w_clean:
                    continue
                w_cyr = any('\u0400' <= c <= '\u04ff' for c in w_clean)
                w_lat = any('a' <= c.lower() <= 'z' for c in w_clean)
                if w_cyr and w_lat:
                    return True
                if w_lat and not w_cyr:
                    if is_inside_bubble or w_clean.isupper() or w_clean.lower() in valid_shorts or len(w_clean) >= 2:
                        continue
                    return True

    return False


def try_read_vertical_strip(img_strip, reader, source_lang: str = "auto"):
    """
    Takes a vertical strip image (h >> w) containing vertically stacked characters,
    slices it into letter glyphs along horizontal valleys, stitches them horizontally,
    and runs OCR to correctly recognize vertical Latin/Cyrillic words.
    """
    if img_strip is None or img_strip.size == 0 or reader is None:
        return None
    lang_norm = (source_lang or "auto").strip().lower()
    if lang_norm in {"ja", "jp", "jpn", "japanese", "zh", "ch", "chs", "cht", "chinese", "ko", "kor", "korean"}:
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
        _, bin_img = cv2.threshold(gray, max(60, int(mean_val * 0.90)), 255, cv2.THRESH_BINARY_INV)

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
            # If OCR output turns out to be CJK/Hangul, don't flatten spaces or convert '0'->'O'
            if any('\u3040' <= ch <= '\u9fff' or '\uac00' <= ch <= '\ud7af' for ch in best_text):
                return best_text, avg_conf
            clean = best_text.replace(" ", "").upper()
            clean = clean.replace("0", "O").replace("/", "!")
            return clean, avg_conf
    except Exception as exc:
        logger.debug("try_read_vertical_strip OCR error: %s", exc)
    return None


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


def clean_manga_text(text: str) -> str:
    """Cleans OCR noise symbols from manga dialogue while preserving multilingual & CJK punctuation."""
    t = re.sub(r'[^\w\s\'\.,!\?\"\-—…~～«»“”„’ʻʼ`「」『』、。・！？【】《》（）：；〜ー]', ' ', text)
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
                import wordninja
                split_w = wordninja.split(cleaned.lower())
                has_strong = any(w in STRONG_DIALOGUE_KEYWORDS for w in split_w)
            except Exception as exc:
                logger.debug("wordninja split error in is_valid_manga_dialogue: %s", exc)
        if has_strong:
            return True
        return sum(1 for c in cleaned if c.isalpha()) >= 3

    # Uzbek & other Latin languages
    return sum(1 for c in cleaned if c.isalpha()) >= 2
