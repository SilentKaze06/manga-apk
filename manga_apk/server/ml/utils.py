import os
import json
import tempfile
import logging
from functools import lru_cache
from PIL import ImageFont

logger = logging.getLogger("manga_utils")

MAX_OCR_RENDER_LEN = 24


@lru_cache(maxsize=512)
def repair_comic_text(text: str) -> str:
    """
    Cleans up comic scanlation OCR artifacts, repairs joined English words,
    and corrects common letter misrecognitions in stylized all-caps comic fonts.
    """
    if not text:
        return ""
    import re
    # 1. Punctuation repairs (in stylized comic fonts, ? often becomes | or /)
    text = re.sub(
        r'\b(WHY|WHAT|WHO|WHERE|WHEN|HOW|COULDN\'?T|CAN|CAN\'?T|DID|IS|ARE|DO|DOES|WILL|WOULD|SHOULD)\b([^?!.\n]+?)(?:[|/])(?=\s|$)',
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
        (r'\b[HН][OО0][TТ][XХ][AА][TТ][BВЬbь]\b', 'ХОТЯТ'),
        (r'\bUNCER-\s*', 'UNCERTAIN '),
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

        def split_token(match):
            token = match.group(0)
            if token.lower() in word_costs:
                return token
            parts = wordninja.split(token)
            if len(parts) < 2:
                return token
            normalized_parts = [part.lower().strip("'") for part in parts]
            if any(not part or (len(part) == 1 and part not in {"a", "i", "s", "m", "d"}) for part in normalized_parts):
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


@lru_cache(maxsize=64)
def get_best_font(size=20, bold=False, sample_text=None):
    """
    Cross-platform font discovery for PIL ImageFont with LRU caching.
    Respects bold priority and verifies glyph coverage for updated Uzbek Latin,
    Cyrillic, and CJK characters, falling back to system fonts and ImageFont.load_default().
    """
    candidate_paths = []
    
    # 0. Bundled comic fonts in project repository (works on Windows, Linux container, Modal)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fonts_dir = os.path.join(base_dir, "fonts")
    parent_dir = os.path.dirname(base_dir)
    server_fonts_dir = os.path.join(parent_dir, "fonts")
    alt_server_fonts_dir = os.path.join(parent_dir, "manga_apk", "server", "fonts")
    container_fonts_dir = "/root/mashinali_oqitish/fonts"
    root_server_fonts_dir = "/root/server/fonts"

    search_dirs = [
        fonts_dir,
        server_fonts_dir,
        alt_server_fonts_dir,
        container_fonts_dir,
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
    
    default_test_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'.,!?-"
    test_chars = sample_text if sample_text is not None else default_test_chars
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

def atomik_json_yoz(filepath, data, indent=2, ensure_ascii=False):
    """
    Atomically writes JSON data to a file via a temporary file in the same directory,
    preventing file corruption on process termination or concurrent read.
    """
    dir_name = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(dir_name, exist_ok=True)
    
    # Use same directory so os.replace is atomic across the filesystem
    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
        temp_name = tf.name
        json.dump(data, tf, ensure_ascii=ensure_ascii, indent=indent)
        tf.flush()
        os.fsync(tf.fileno())
        
    try:
        os.replace(temp_name, filepath)
    except Exception as e:
        if os.path.exists(temp_name):
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
