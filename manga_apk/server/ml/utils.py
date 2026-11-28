import os
import json
import tempfile
import logging
from functools import lru_cache
from PIL import ImageFont

logger = logging.getLogger("manga_utils")

MAX_OCR_RENDER_LEN = 24


@lru_cache(maxsize=512)
def repair_comic_text(text: str, source_lang: str = "en") -> str:
    """
    Cleans up comic scanlation OCR artifacts, repairs joined English words,
    and corrects common letter misrecognitions in stylized all-caps comic fonts.
    Strictly gated to English (`source_lang == "en"`) so Uzbek/romaji/non-English text is never split.
    """
    if not text:
        return ""
    import re
    # Apply Cyrillic OCR homoglyph repair before the English-only guard
    text = re.sub(r'\b[HН][OО0][TТ][XХ][AА][TТ][BВЬbь]\b', 'ХОТЯТ', text, flags=re.IGNORECASE)
    lang_norm = (source_lang or "en").strip().lower()
    if lang_norm not in {"en", "eng", "english"}:
        return text
    # 1. Punctuation repairs (in stylized comic fonts, ? often becomes |)
    text = re.sub(
        r'\b(WHY|WHAT|WHO|WHERE|WHEN|HOW|COULDN\'?T|CAN|CAN\'?T|DID|IS|ARE|DO|DOES|WILL|WOULD|SHOULD)\b([^?!.\n]+?)(?:\|)(?=\s|$)',
        r'\1\2?',
        text,
        flags=re.IGNORECASE
    )
    # Common OCR mistranscriptions in all-caps manga fonts
    replacements = [
        (r'\bLISTBE\b', 'MUST BE'),
        (r'\bMSST\b', 'MUST'),
        (r'\bMVST\b', 'MUST'),
        (r'\bWANTUS\b', 'WANT US'),
        (r'\bFORUSTO\b', 'FOR US TO'),
        (r'\bINORDER\b', 'IN ORDER'),
        (r'\bANDIT\'?S\b', "AND IT'S"),
        (r'\bMYTH\s*FO\s*LK\b', 'MYTH-FOLK'),
        (r'\bMOHX\b', 'MUCH'),
        (r'\bTHEREA\b', 'THERE IS A'),
        (r'\bBlack-IC\b', 'BLACK-ISH'),
        (r'\bWHYP\b', 'WHY?'),
        (r'\bWHATP\b', 'WHAT?'),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    try:
        import wordninja
        word_costs = wordninja.DEFAULT_LANGUAGE_MODEL._wordcost

        _protected_proper = {"tsumiki", "ogami", "senpai", "sensei", "kohai", "kouhai", "kawaii"}

        def split_token(match):
            token = match.group(0)
            low = token.lower()
            if low in word_costs or low in _protected_proper:
                return token
            parts = wordninja.split(token)
            if len(parts) < 2:
                return token
            normalized_parts = [part.lower().strip("'") for part in parts]
            if any(not part or (len(part) == 1 and part not in {"a", "i"}) for part in normalized_parts):
                return token
            # Prevent splitting unknown single words / hyphen-fragments while allowing merged short English words
            _common_short_words = {
                "i", "a", "is", "it", "in", "on", "to", "of", "us", "my", "no", "we", "he",
                "be", "do", "go", "so", "am", "an", "as", "at", "by", "if", "me", "or", "up",
                "oh", "ah", "ok",
                "for", "and", "the", "are", "you", "can", "was", "not", "but", "out",
                "all", "any", "did", "far", "get", "got", "had", "has", "her", "him", "his",
                "how", "its", "let", "man", "may", "new", "now", "off", "old", "one", "our",
                "own", "put", "run", "saw", "say", "see", "she", "too", "two", "use", "way",
                "who", "why", "yes", "yet",
            }
            if any(len(p) <= 3 and p not in _common_short_words for p in normalized_parts):
                return token
            if any(part not in word_costs for part in normalized_parts if len(part) > 1):
                return token
            return " ".join(parts)

        text = re.sub(r"[A-Za-z]{4,}", split_token, text)
    except Exception:
        pass
    return text


def _has_glyph_coverage(font, test_chars="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'.,!?-"):
    if not test_chars:
        return True
    try:
        missing = font.getmask("\U0010ffff")
        for ch in set(test_chars):
            if ch.isspace():
                continue
            mask = font.getmask(ch)
            if not mask.getbbox() or (mask.size == missing.size and bytes(mask) == bytes(missing)):
                return False
            bbox = font.getbbox(ch)
            if bbox is None or (bbox[2] - bbox[0] == 0 and bbox[3] - bbox[1] == 0):
                return False
        return True
    except Exception:
        return False


def resolve_font_path(font_name_or_path: str, fonts_dir: str = None) -> str:
    """Safe router for font lookups: routes duplicate wild words aliases to animeace2_reg.ttf."""
    if not font_name_or_path:
        return font_name_or_path
    base = os.path.basename(font_name_or_path).lower().replace("-", "").replace("_", "").replace(" ", "")
    if "wildword" in base:
        dir_name = os.path.dirname(font_name_or_path) or fonts_dir
        return os.path.join(dir_name, "animeace2_reg.ttf") if dir_name else "animeace2_reg.ttf"
    return font_name_or_path


def _font_script_tag(sample_text: str) -> tuple:
    """Classify sample_text into a canonical (script_tag, representative_chars) cache key."""
    if not sample_text:
        return ("ascii", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'.,!?-")
    has_cjk = any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' for c in sample_text)
    has_hangul = any('\uac00' <= c <= '\ud7af' for c in sample_text)
    has_uz_cyr = any(c in "ЎўҚқҒғҲҳ" for c in sample_text)
    has_cyr = any('\u0400' <= c <= '\u04ff' for c in sample_text)
    has_diacritics = any(c in "ÖöĞğŞşÇçÜüİıƏə" or ord(c) > 127 for c in sample_text if c.isalpha())

    if has_cjk or has_hangul:
        non_ascii = "".join(sorted({c for c in sample_text if ord(c) > 127}))[:16]
        return (f"cjk:{non_ascii}", sample_text)
    if has_uz_cyr:
        return ("uz_cyr", "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяЎўҚқҒғҲҳ")
    if has_cyr:
        return ("cyr", "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя")
    if has_diacritics:
        extra = "".join(sorted({c for c in sample_text if ord(c) > 127 and c.isalpha()}))
        return (f"lat_ext:{extra}", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" + extra)
    return ("ascii", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'.,!?-")


@lru_cache(maxsize=64)
def _get_best_font_by_script(size: int, bold: bool, script_tag: str, test_chars: str):
    """Cached font lookup keyed strictly by (size, bold, script_tag)."""
    candidate_paths = []
    
    # 0. Bundled comic fonts in project repository (works on Windows, Linux container, Modal)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fonts_dir = os.path.join(base_dir, "fonts")
    parent_dir = os.path.dirname(base_dir)
    server_fonts_dir = os.path.join(parent_dir, "fonts")
    alt_server_fonts_dir = os.path.join(parent_dir, "manga_apk", "server", "fonts")
    root_server_fonts_dir = "/root/server/fonts"

    search_dirs = [
        fonts_dir,
        server_fonts_dir,
        alt_server_fonts_dir,
        root_server_fonts_dir,
    ]

    if bold:
        for fd in search_dirs:
            candidate_paths.extend([
                os.path.join(fd, "ComicNeue-Bold.ttf"),
                os.path.join(fd, "comicbd.ttf"),
                os.path.join(fd, "animeace2_reg.ttf"),
                os.path.join(fd, "arialbd.ttf"),
                os.path.join(fd, "komika.ttf"),
            ])
    else:
        for fd in search_dirs:
            candidate_paths.extend([
                os.path.join(fd, "comic.ttf"),
                os.path.join(fd, "arial.ttf"),
                os.path.join(fd, "ComicNeue-Regular.ttf"),
                os.path.join(fd, "animeace2_reg.ttf"),
                os.path.join(fd, "komika.ttf"),
            ])

    # 1. Windows fonts
    win_dir = os.environ.get("WINDIR", r"C:\Windows")
    fonts_win = os.path.join(win_dir, "Fonts")
    if bold:
        candidate_paths.extend([
            os.path.join(fonts_win, "comicbd.ttf"),
            os.path.join(fonts_win, "arialbd.ttf"),
            os.path.join(fonts_win, "segoeuib.ttf"),
            os.path.join(fonts_win, "msgothic.ttc"),
            os.path.join(fonts_win, "msyh.ttc"),
        ])
    else:
        candidate_paths.extend([
            os.path.join(fonts_win, "comic.ttf"),
            os.path.join(fonts_win, "arial.ttf"),
            os.path.join(fonts_win, "segoeui.ttf"),
            os.path.join(fonts_win, "msgothic.ttc"),
            os.path.join(fonts_win, "msyh.ttc"),
        ])
        
    # 2. Linux / macOS fonts
    candidate_paths.extend([
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf"
    ])
    
    for fpath in candidate_paths:
        if os.path.exists(fpath):
            try:
                f = ImageFont.truetype(fpath, size)
                if _has_glyph_coverage(f, test_chars):
                    return f
            except Exception as e:
                logger.debug("Failed loading font %s: %s", fpath, e)
                continue
                
    fallback = ImageFont.load_default(size=size)
    return fallback


def get_best_font(size=20, bold=False, sample_text=None):
    """
    Cross-platform font discovery for PIL ImageFont with script-tag LRU caching.
    Respects bold priority and verifies glyph coverage for Uzbek Latin, Turkish diacritics,
    Uzbek Cyrillic, and CJK characters, falling back to system fonts and ImageFont.load_default().
    """
    script_tag, test_chars = _font_script_tag(sample_text)
    return _get_best_font_by_script(int(size), bool(bold), script_tag, test_chars)


def atomik_json_yoz(filepath, data, indent=2, ensure_ascii=False):
    """
    Atomically writes JSON data to a file via a temporary file in the same directory,
    preventing file corruption on process termination or concurrent read.
    """
    dir_name = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(dir_name, exist_ok=True)

    temp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
            temp_name = tf.name
            json.dump(data, tf, ensure_ascii=ensure_ascii, indent=indent)
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(temp_name, filepath)
    except Exception as e:
        if temp_name and os.path.exists(temp_name):
            try:
                os.remove(temp_name)
            except Exception:
                pass
        raise e


def decode_ctc(log_probs_tb, idx_to_char):
    """
    Standard Greedy CTC decode for a single sample: log_probs_tb is (T, C).
    Takes raw argmax across classes, collapses consecutive duplicates,
    and removes blank token (0) and <unk> token (1).
    """
    preds_idx = log_probs_tb.argmax(dim=-1)
    toks = []
    prev = None
    for t in preds_idx.cpu().numpy():
        t = int(t)
        if t != prev and t > 1:
            toks.append(t)
        prev = t
    return "".join(idx_to_char.get(t, "") for t in toks)
