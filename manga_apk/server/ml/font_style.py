"""
Style-aware manga font selection, Google Fonts (SIL OFL) downloader,
Cyrillic + Uzbek Latin glyph coverage verification, and proportional stroke/color styling.
"""
import argparse
import logging
import os
import re
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import ImageFont

try:
    from utils import get_best_font
except (ImportError, ValueError):
    try:
        from .utils import get_best_font
    except (ImportError, ValueError):
        from ml.utils import get_best_font

logger = logging.getLogger(__name__)

# 5 manga style buckets mapped to free SIL OFL Google Fonts (with local fallbacks)
STYLE_FONT_BUCKETS: Dict[str, List[str]] = {
    "dialogue": [
        "BalsamiqSans-Bold.ttf",
        "ShantellSans-Bold.ttf",
        "comicbd.ttf",
    ],
    "shout": [
        "ShantellSans-ExtraBold.ttf",
        "RussoOne-Regular.ttf",
        "BalsamiqSans-Bold.ttf",
    ],
    "narration": [
        "Lora-Bold.ttf",
        "PTSerif-Bold.ttf",
        "DejaVuSerif-Bold.ttf",
    ],
    "thought": [
        "BalsamiqSans-Italic.ttf",
        "BalsamiqSans-BoldItalic.ttf",
        "Caveat-Bold.ttf",
    ],
    "sfx_title": [
        "AmaticSC-Bold.ttf",
        "RussoOne-Regular.ttf",
        "BalsamiqSans-Bold.ttf",
    ],
}

# Direct raw GitHub URLs from google/fonts (SIL OFL)
FONT_DOWNLOAD_URLS: Dict[str, List[str]] = {
    "BalsamiqSans-Bold.ttf": [
        "https://raw.githubusercontent.com/google/fonts/main/ofl/balsamiqsans/BalsamiqSans-Bold.ttf",
    ],
    "BalsamiqSans-Italic.ttf": [
        "https://raw.githubusercontent.com/google/fonts/main/ofl/balsamiqsans/BalsamiqSans-Italic.ttf",
    ],
    "BalsamiqSans-BoldItalic.ttf": [
        "https://raw.githubusercontent.com/google/fonts/main/ofl/balsamiqsans/BalsamiqSans-BoldItalic.ttf",
    ],
    "ShantellSans-Bold.ttf": [
        "https://raw.githubusercontent.com/google/fonts/main/ofl/shantellsans/ShantellSans%5BBNCE%2CINFM%2CSPAC%2Cwght%5D.ttf",
    ],
    "ShantellSans-ExtraBold.ttf": [
        "https://raw.githubusercontent.com/google/fonts/main/ofl/shantellsans/ShantellSans%5BBNCE%2CINFM%2CSPAC%2Cwght%5D.ttf",
    ],
    "RussoOne-Regular.ttf": [
        "https://raw.githubusercontent.com/google/fonts/main/ofl/russoone/RussoOne-Regular.ttf",
    ],
    "Lora-Bold.ttf": [
        "https://raw.githubusercontent.com/google/fonts/main/ofl/lora/Lora%5Bwght%5D.ttf",
    ],
    "PTSerif-Bold.ttf": [
        "https://raw.githubusercontent.com/google/fonts/main/ofl/ptserif/PT_Serif-Web-Bold.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/ptserif/PTSerif-Bold.ttf",
    ],
    "Caveat-Bold.ttf": [
        "https://raw.githubusercontent.com/google/fonts/main/ofl/caveat/Caveat%5Bwght%5D.ttf",
    ],
    "AmaticSC-Bold.ttf": [
        "https://raw.githubusercontent.com/google/fonts/main/ofl/amaticsc/AmaticSC-Bold.ttf",
    ],
}

CYRILLIC_REQUIRED_CHARS = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюя"
UZBEK_LATIN_REQUIRED_CHARS = "ÖöĞğŞşÇç"


def _default_fonts_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "fonts"


def _font_search_dirs() -> List[Path]:
    dirs = [
        _default_fonts_dir(),
        Path("/root/server/fonts"),
        Path("/opt/models/fonts"),
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("C:/Windows/Fonts"),
    ]
    return [d for d in dirs if d.is_dir()]


def verify_font_glyph_coverage(font: ImageFont.FreeTypeFont, test_chars: str) -> bool:
    """
    Verify that `font` actually renders non-tofu glyphs for every character in `test_chars`.
    """
    if not test_chars:
        return True
    try:
        tofu_bytes = bytes(font.getmask("\uffff"))
    except Exception:
        tofu_bytes = b""

    for ch in test_chars:
        if ch.isspace():
            continue
        try:
            mask = font.getmask(ch)
            if mask.getbbox() is None:
                return False
            if tofu_bytes and bytes(mask) == tofu_bytes:
                return False
        except Exception:
            return False
    return True


def verify_font_file_russian_and_uzbek(font_path: Path) -> bool:
    """Check that a .ttf file loads cleanly and supports Russian Cyrillic + Uzbek Latin."""
    try:
        f = ImageFont.truetype(str(font_path), 24)
        return verify_font_glyph_coverage(f, CYRILLIC_REQUIRED_CHARS + UZBEK_LATIN_REQUIRED_CHARS)
    except Exception:
        return False


def _resolve_font_file(filename: str) -> Optional[Path]:
    for d in _font_search_dirs():
        cand = d / filename
        if cand.is_file() and cand.stat().st_size > 1024:
            return cand
    return None


@lru_cache(maxsize=256)
def _load_verified_styled_font(style: str, size: int, required_chars: str):
    bucket = STYLE_FONT_BUCKETS.get(style, STYLE_FONT_BUCKETS["dialogue"])
    check_str = (required_chars or "") + "АаЯя" + UZBEK_LATIN_REQUIRED_CHARS
    for fname in bucket:
        fpath = _resolve_font_file(fname)
        if fpath is None:
            continue
        try:
            font = ImageFont.truetype(str(fpath), int(size))
            if verify_font_glyph_coverage(font, check_str):
                return font
        except Exception as exc:
            logger.debug("Font %s failed load/coverage: %s", fpath, exc)
    return None


def get_styled_font(style: str, size: int, sample_text: Optional[str] = None) -> ImageFont.FreeTypeFont:
    """
    Resolve a PIL FreeTypeFont for `style` (`dialogue`, `shout`, `narration`, `thought`, `sfx_title`)
    verifying Russian Cyrillic & Uzbek Latin glyph coverage, with graceful fallback to utils.get_best_font.
    """
    size = max(6, int(round(size)))
    req_chars = "".join(sorted({c for c in (sample_text or "") if not c.isspace()}))
    font = _load_verified_styled_font(style or "dialogue", size, req_chars)
    if font is not None:
        return font
    is_bold = (style != "thought")
    return get_best_font(size=size, bold=is_bold, sample_text=sample_text)


_neural_font_clf = None
_neural_font_clf_attempted = False


def _try_neural_font_style(orig_crop: Optional[np.ndarray]) -> Optional[str]:
    """
    Optional neural font-classifier hook (`gaborcselle/font-identifier`),
    strictly gated behind `ENABLE_NEURAL_FONT_STYLE=1`.
    """
    global _neural_font_clf, _neural_font_clf_attempted
    if os.environ.get("ENABLE_NEURAL_FONT_STYLE", "0") != "1" or orig_crop is None:
        return None
    if not _neural_font_clf_attempted:
        _neural_font_clf_attempted = True
        try:
            from transformers import pipeline
            _neural_font_clf = pipeline("image-classification", model="gaborcselle/font-identifier")
        except Exception as exc:
            logger.debug("Optional neural font-identifier unavailable: %s", exc)
            _neural_font_clf = None
    if _neural_font_clf is None:
        return None
    try:
        from PIL import Image
        preds = _neural_font_clf(Image.fromarray(orig_crop))
        if preds:
            top = str(preds[0].get("label", "")).lower()
            if "serif" in top and "sans" not in top:
                return "narration"
            if "italic" in top or "script" in top or "hand" in top:
                return "thought"
            if "display" in top or "heavy" in top or "black" in top:
                return "shout"
    except Exception:
        pass
    return None


def classify_bubble_style(
    item: dict,
    text: str,
    bw: int,
    bh: int,
    is_on_artwork: bool,
    has_center_spikes: bool,
    orig_crop: Optional[np.ndarray] = None,
) -> str:
    """
    Classify a dialogue/caption block into one of 5 manga typography buckets:
    `dialogue`, `shout`, `narration`, `thought`, `sfx_title`.
    """
    cat = str(item.get("category", "")).lower()
    label = str(item.get("label", "")).lower()
    txt = (text or "").strip()

    if item.get("is_sfx") or cat == "sfx":
        return "sfx_title"

    if cat in ("narration", "narration_box", "box", "rect"):
        return "narration"

    if cat in ("thought", "thought_bubble"):
        return "thought"

    # Wide floating caption directly on artwork (manhua narration/title)
    if is_on_artwork and (label == "text_free" or cat == "free_text"):
        if len(txt.split()) <= 3 and ("!" in txt or bw >= 180):
            return "sfx_title"
        return "narration"

    if has_center_spikes or "?!" in txt or "!?" in txt or "!!" in txt:
        return "shout"

    # Single-word / short exclamation ending in '!'
    if txt.endswith("!") and len(txt.split()) <= 4:
        return "shout"

    # Trailing ellipsis / introspective phrasing without exclamation
    if (txt.endswith("...") or txt.endswith("…")) and "!" not in txt and "?" not in txt:
        return "thought"

    neural_hint = _try_neural_font_style(orig_crop)
    if neural_hint in STYLE_FONT_BUCKETS:
        return neural_hint

    return "dialogue"


def compute_proportional_stroke(
    luma: float,
    font_size: int,
    is_on_artwork: bool,
    style: str,
    orig_crop: Optional[np.ndarray] = None,
    clean_crop: Optional[np.ndarray] = None,
) -> Tuple[Tuple[int, int, int], Optional[Tuple[int, int, int]], int]:
    """
    Compute `(text_fill, stroke_fill, stroke_width)` scaling stroke proportionally with `font_size`:
    - On busy/colored artwork: `max(2, min(6, round(font_size * 0.15)))`
    - On dark background (`luma < 125`): white text + proportional black outline
    - On uniform light bubble (`luma >= 165` and not `is_on_artwork`): crisp black text, 0 stroke (or 1-2px for shout)
    - When `orig_crop` is provided, optionally extracts vivid original ink color.
    """
    prop_sw = max(2, min(6, int(round(font_size * 0.15))))

    fill_rgb: Tuple[int, int, int] = (255, 255, 255) if luma < 125.0 else (15, 15, 15)

    # Optional vivid original ink color extraction when orig_crop is supplied
    if orig_crop is not None and orig_crop.size > 0 and orig_crop.ndim == 3 and orig_crop.shape[2] >= 3:
        try:
            rgb = orig_crop[:, :, :3].astype(np.float32)
            sat = np.max(rgb, axis=2) - np.min(rgb, axis=2)
            vivid = sat > 65.0
            if int(np.sum(vivid)) >= 25:
                med_vivid = np.round(np.median(rgb[vivid], axis=0)).astype(int)
                vivid_luma = 0.299 * med_vivid[0] + 0.587 * med_vivid[1] + 0.114 * med_vivid[2]
                # Only adopt vivid ink if it has strong contrast against background luma
                if abs(vivid_luma - luma) >= 55.0:
                    fill_rgb = (int(med_vivid[0]), int(med_vivid[1]), int(med_vivid[2]))
        except Exception:
            pass

    if luma < 125.0:
        return fill_rgb, (0, 0, 0), prop_sw

    if is_on_artwork or luma < 165.0:
        stroke_c = (0, 0, 0) if (0.299 * fill_rgb[0] + 0.587 * fill_rgb[1] + 0.114 * fill_rgb[2]) > 140 else (255, 255, 255)
        return fill_rgb, stroke_c, prop_sw

    if style in ("shout", "sfx_title"):
        return fill_rgb, (255, 255, 255), max(1, min(3, prop_sw // 2))

    return fill_rgb, None, 0


def download_default_fonts(target_dir: Optional[str] = None) -> Dict[str, dict]:
    """
    Download SIL OFL Google Fonts into `target_dir` (defaults to `server/fonts/`)
    and verify file size (> 10 KB) + Russian Cyrillic & Uzbek Latin glyph coverage.
    """
    out_dir = Path(target_dir) if target_dir else _default_fonts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, dict] = {}

    for fname, urls in FONT_DOWNLOAD_URLS.items():
        dest = out_dir / fname
        downloaded = dest.is_file() and dest.stat().st_size > 10_240
        err_msg = None
        if not downloaded:
            for url in urls:
                try:
                    tmp = dest.with_suffix(".ttf.tmp")
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = resp.read()
                    if len(data) > 10_240:
                        tmp.write_bytes(data)
                        tmp.replace(dest)
                        downloaded = True
                        break
                except Exception as exc:
                    err_msg = str(exc)

        size_bytes = dest.stat().st_size if dest.is_file() else 0
        glyph_ok = verify_font_file_russian_and_uzbek(dest) if (downloaded and size_bytes > 10_240) else False
        report[fname] = {
            "path": str(dest),
            "downloaded": downloaded,
            "size_bytes": size_bytes,
            "cyrillic_uzbek_ok": glyph_ok,
            "error": err_msg if not downloaded else None,
        }
    _load_verified_styled_font.cache_clear()
    return report


def main():
    parser = argparse.ArgumentParser(description="Manga font style manager & Google Fonts OFL downloader")
    parser.add_argument("--download-fonts", action="store_true", help="Download SIL OFL fonts into server/fonts/")
    parser.add_argument("--target-dir", type=str, default=None, help="Optional target fonts directory")
    args = parser.parse_args()

    if args.download_fonts:
        rep = download_default_fonts(args.target_dir)
        ok_count = 0
        for fname, info in rep.items():
            status = "OK" if (info["downloaded"] and info["cyrillic_uzbek_ok"]) else "FAIL"
            if status == "OK":
                ok_count += 1
            print(f"[{status}] {fname}: {info['size_bytes']} bytes, cyr+uz={info['cyrillic_uzbek_ok']}" + (f" err={info['error']}" if info["error"] else ""))
        print(f"Downloaded & verified: {ok_count}/{len(rep)} fonts")


if __name__ == "__main__":
    main()
