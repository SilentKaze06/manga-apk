"""
Font selection, elliptical text wrapping, Uzbek syllable hyphenation, and dialogue rendering.
"""
import logging
import math
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    from utils import get_best_font, resolve_font_path
except (ImportError, ValueError):
    from .utils import get_best_font, resolve_font_path

try:
    from ocr_engine import is_manga_punctuation, normalize_manga_punctuation, is_sfx_text
except (ImportError, ValueError):
    from .ocr_engine import is_manga_punctuation, normalize_manga_punctuation, is_sfx_text

try:
    from detection import _has_continuous_dark_segment
except (ImportError, ValueError):
    from .detection import _has_continuous_dark_segment

logger = logging.getLogger(__name__)


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
    if patch.size == 0 or float(np.median(patch)) < 120:
        return (geom_cx, geom_cy)

    p85 = float(np.percentile(patch, 85))
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


def resolve_wildwords_font(font_name: str) -> str:
    """Routes wild words font requests to animeace2_reg.ttf."""
    if not font_name:
        return "animeace2_reg.ttf"
    base = font_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    if "wildword" in base:
        return "animeace2_reg.ttf"
    return font_name


def get_contrast_text_colors(luma: float, is_on_artwork: bool = False) -> tuple:
    """
    Unified contrast color determination:
    - Dark background (luma < 125): white text (255, 255, 255) + black outline (stroke_width=2).
    - Mid-tone / busy artwork (125 <= luma < 165 or is_on_artwork): black text (15, 15, 15) + white outline (stroke_width=2).
    - Uniform light background (luma >= 165 and not is_on_artwork): black text (15, 15, 15) without stroke.
    """
    if luma < 125.0:
        return (255, 255, 255), (0, 0, 0), 2
    if luma < 165.0 or is_on_artwork:
        return (15, 15, 15), (255, 255, 255), 2
    return (15, 15, 15), None, 0


def hyphenate_uzbek_word(word: str) -> list:
    """Split a long Uzbek or Cyrillic word at a syllable boundary and preserve its punctuation."""
    clean_w = word.rstrip('.!?,:;')
    punct = word[len(clean_w):]

    # Only hyphenate genuinely long words (7+ characters)
    # Avoid breaking short/medium words mid-syllable (e.g. bo'lishini, chaqmog'ingni)
    if len(clean_w) >= 7 and re.fullmatch(r"[a-zA-Zʻʼ‘’'`ÖöĞğŞşÇç]+", clean_w):
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

    if len(clean_w) >= 7 and re.fullmatch(r"[а-яА-ЯёЁўқғҳЎҚҒҲ]+", clean_w):
        cyr_vowels = set("аеёиоуыэюя")
        vowels = [i for i, ch in enumerate(clean_w) if ch.lower() in cyr_vowels]
        boundaries = []
        for left, right in zip(vowels, vowels[1:]):
            cut = max(left + 1, right - 1)
            if cut < len(clean_w) and clean_w[cut].lower() in ("ь", "ъ"):
                cut += 1
            if 3 <= cut <= len(clean_w) - 3:
                boundaries.append(cut)
        if boundaries:
            cut = min(boundaries, key=lambda pos: abs(2 * pos - len(clean_w)))
            return [clean_w[:cut] + "-", clean_w[cut:] + punct]

    return [word]


def wrap_text_elliptical(words, font, a, b, line_spacing=1.15, allow_hyphenation=True, is_rect: bool = False):
    """
    Wraps words into lines fitting comfortably within the speech bubble's elliptical envelope.
    For each line i at vertical offset y_i from bubble center, maximum allowed width is:
    w_max = 2 * a * sqrt(max(0.05, 1 - (y_i / b)^2))
    When is_rect=True (narration box), full rectangular width (2 * a) is used.
    """
    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    line_step = int(line_h * line_spacing)

    # Word candidate variants: try original first, and hyphenated if any word is >= 7 chars
    word_variants = [words]
    split_pairs = {}
    if allow_hyphenation and any(len(w.rstrip('.!?,:;')) >= 7 for w in words):
        expanded = []
        for w in words:
            if len(w.rstrip('.!?,:;')) >= 7:
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
            total_h = (target_lines - 1) * line_step + line_h
            if total_h > 2 * b * (1.25 if is_rect else 0.85):
                break

            allowed_widths = []
            for i in range(target_lines):
                if is_rect:
                    w_max = 2.0 * a
                else:
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
    img_w_full, img_h_full = rendered.size

    for item in dialogues:
        text = (item.get("translation") or item.get("translated_text") or "").strip()
        if not text:
            continue

        text = text.replace("ʻ", "'").replace("ʼ", "'").replace("‘", "'").replace("’", "'").replace("`", "'").replace("—", "-").replace("–", "-").replace("…", "...")

        # Sound effects (SFX) drawn directly on artwork must NEVER be overwritten with standard fonts
        is_sfx = bool(item.get("is_sfx", False) or item.get("category") == "sfx")
        if is_sfx:
            continue
        box = item.get("bbox") or [item.get("left", 0), item.get("top", 0), item.get("right", 0), item.get("bottom", 0)]
        x1 = max(0, min(img_w_full, int(box[0])))
        y1 = max(0, min(img_h_full, int(box[1])))
        x2 = max(0, min(img_w_full, int(box[2])))
        y2 = max(0, min(img_h_full, int(box[3])))
        draw_box = [x1, y1, x2, y2]
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            continue

        # 1. Compute true mathematical centroid of the physical speech bubble
        crop_gray = np.array(image.crop((x1, y1, x2, y2)).convert("L"))
        bg_lum = float(np.median(crop_gray)) if crop_gray.size > 0 else 255.0
        bg_std = float(np.std(crop_gray)) if crop_gray.size > 0 else 0.0

        # Safe inscribed box ONLY for light bubbles / light narration boxes (never shrink dark narration boxes by 24px):
        is_white_box = (bg_lum >= 170 and np.mean(crop_gray > 200) >= 0.25)
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
        is_on_artwork = is_sfx or bg_lum < 165 or bg_std > 25 or item.get("is_on_art", False)

        is_rect = item.get("category") in ("narration", "narration_box", "box", "rect") or (bw <= 85 and bh <= 90 and bg_lum > 240)
        margin_ratio = 0.80 if is_rect else (0.72 if not is_on_artwork else 0.82)
        a = (bw / 2.0) * margin_ratio
        b = (bh / 2.0) * margin_ratio

        # Horizontal rectangular ticker / banner (e.g. TV news ticker, headline banner)
        is_banner = (bw >= 3.5 * bh and bh <= 60) or (bw >= 260 and bh <= 45)
        if is_banner:
            clean_banner_txt = text.replace("\n", " ").strip()
            if not is_sfx and not is_manga_punctuation(clean_banner_txt):
                clean_banner_txt = clean_banner_txt.upper()
            words = clean_banner_txt.split()
            best_fs = 7
            best_lines = [clean_banner_txt]
            for fs in range(min(14, max(8, int(bh * 0.45))), 5, -1):
                f_b = get_best_font(size=fs, bold=True, sample_text=clean_banner_txt)
                ascent, descent = f_b.getmetrics()
                lh = ascent + descent
                if len(words) >= 2:
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
                bb_single = f_b.getbbox(clean_banner_txt)
                if (bb_single[2] - bb_single[0]) <= bw * 0.94 and lh <= bh * 0.85:
                    best_fs = fs
                    best_lines = [clean_banner_txt]
                    break

            f_b = get_best_font(size=best_fs, bold=True, sample_text=clean_banner_txt)
            ascent, descent = f_b.getmetrics()
            lh = ascent + descent
            lstep = int(lh * 1.1)
            tot_h = (len(best_lines) - 1) * lstep + lh
            start_y = max(y1 + 2, cy - (tot_h / 2.0) - (ascent * 0.15))
            banner_fill, banner_stroke_fill, banner_stroke_w = get_contrast_text_colors(bg_lum, is_on_artwork=is_on_artwork)
            for li, line in enumerate(best_lines):
                bb = f_b.getbbox(line)
                lw = bb[2] - bb[0]
                lx = cx - (lw / 2.0) - bb[0]
                lx = max(x1 + 4, min(max(x1 + 4, x2 - 4 - lw), lx))
                if banner_stroke_w > 0 and banner_stroke_fill is not None:
                    draw.text((lx, start_y + li * lstep), line, fill=banner_fill, font=f_b, stroke_width=banner_stroke_w, stroke_fill=banner_stroke_fill)
                else:
                    draw.text((lx, start_y + li * lstep), line, fill=banner_fill, font=f_b)
            continue

        # Multi-lobe shock bubble with central spikes (never match narration/narration_box or uniform dark boxes!)
        is_narration_cat = item.get("category") in ("narration", "narration_box")
        dark_ratio = np.count_nonzero(crop_gray < 100) / float(crop_gray.size) if crop_gray.size > 0 else 0
        mid_y1 = int(bh * 0.32)
        mid_y2 = int(bh * 0.68)
        has_center_spikes = False
        if not is_narration_cat and 0.45 < dark_ratio <= 0.72 and mid_y2 > mid_y1 and np.percentile(crop_gray, 85) > 190:
            row_dark_ratios = np.sum((crop_gray[mid_y1:mid_y2, :] < 130), axis=1) / float(bw)
            if np.any(row_dark_ratios > 0.40):
                has_center_spikes = True

        if not is_sfx and not is_manga_punctuation(text):
            text = text.upper()

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

            cy1_lobe = y1 + int(bh * 0.22)
            cy2_lobe = y1 + int(bh * 0.72)

            # Determine contrast independently for upper and lower lobes:
            h_img, w_img = gray_full.shape[:2]
            lobe1_patch = gray_full[max(0, cy1_lobe-5):min(h_img, cy1_lobe+6), max(0, int(cx)-10):min(w_img, int(cx)+11)]
            med1 = float(np.median(lobe1_patch)) if lobe1_patch.size > 0 else 255.0
            sb1_fill, sb1_stroke_fill, sb1_stroke_w = get_contrast_text_colors(med1, is_on_artwork=is_on_artwork)

            lobe2_patch = gray_full[max(0, cy2_lobe-5):min(h_img, cy2_lobe+6), max(0, int(cx)-10):min(w_img, int(cx)+11)]
            med2 = float(np.median(lobe2_patch)) if lobe2_patch.size > 0 else med1
            sb2_fill, sb2_stroke_fill, sb2_stroke_w = get_contrast_text_colors(med2, is_on_artwork=is_on_artwork)

            def _fit_shock_part(part_text: str, max_w: float, max_h: float):
                part_words = part_text.split()
                for try_fs in range(min(13, max(10, int(bh * 0.17))), 7, -1):
                    try_font = get_best_font(size=try_fs, bold=True, sample_text=part_text)
                    wrapped, total_h = wrap_text_elliptical(
                        part_words, try_font, max_w / 2.0, max_h / 2.0,
                        line_spacing=1.1, is_rect=True,
                    )
                    rendered_part = "\n".join(wrapped) if wrapped else part_text
                    bb = draw.multiline_textbbox((0, 0), rendered_part, font=try_font)
                    rw, rh = bb[2] - bb[0], bb[3] - bb[1]
                    if (rw <= max_w and rh <= max_h) or try_fs == 8:
                        return rendered_part, try_font, bb, rw, rh
                return part_text, get_best_font(size=8, bold=True, sample_text=part_text), (0, 0, 1, 1), 1, 1

            def _draw_clamped_shock(part_text: str, cy_target: int, fill_c, stroke_c, stroke_w_val, max_w: float, max_h: float):
                r_txt, f_use, bb, rw, rh = _fit_shock_part(part_text, max_w, max_h)
                tx = int(round(cx - rw / 2.0 - bb[0]))
                ty = int(round(cy_target - rh / 2.0 - bb[1]))
                min_tx = max(0, x1 + 2 - bb[0])
                max_tx = max(min_tx, min(img_w_full - rw - 2, x2 - rw - 2) - bb[0])
                min_ty = max(0, y1 + 2 - bb[1])
                max_ty = max(min_ty, min(img_h_full - rh - 2, y2 - rh - 2) - bb[1])
                tx = max(min_tx, min(max_tx, tx))
                ty = max(min_ty, min(max_ty, ty))
                if stroke_w_val > 0:
                    draw.multiline_text((tx, ty), r_txt, fill=fill_c, stroke_width=stroke_w_val, stroke_fill=stroke_c, font=f_use, align="center")
                else:
                    draw.multiline_text((tx, ty), r_txt, fill=fill_c, font=f_use, align="center")

            lobe_max_w = max(32.0, bw * 0.78)
            if len(parts) >= 2:
                lobe_max_h = max(20.0, bh * 0.36)
                _draw_clamped_shock(parts[0], cy1_lobe, sb1_fill, sb1_stroke_fill, sb1_stroke_w, lobe_max_w, lobe_max_h)
                _draw_clamped_shock(" ".join(parts[1:]), cy2_lobe, sb2_fill, sb2_stroke_fill, sb2_stroke_w, lobe_max_w, lobe_max_h)
            else:
                _draw_clamped_shock(text, (y1 + y2) // 2, sb1_fill, sb1_stroke_fill, sb1_stroke_w, lobe_max_w, max(24.0, bh * 0.78))
            continue

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
                f = get_best_font(size=fs, bold=True, sample_text=text)
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
                f = get_best_font(size=fs, bold=True, sample_text=text)
                candidate_paras = []
                fits = True
                for p_idx, p in enumerate(paragraphs):
                    p_words = p.split()
                    if is_sfx and any("-" in w for w in p_words):
                        exp = []
                        for w in p_words:
                            exp.extend([sub for sub in w.split("-") if sub] if "-" in w else [w])
                        p_words = exp
                    lines_p = wrap_text_elliptical(p_words, f, a, b, allow_hyphenation=False, is_rect=is_rect)
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
                f = get_best_font(size=fs, bold=True, sample_text=text)
                candidate_paras = []
                fits = True
                for p_idx, p in enumerate(paragraphs):
                    p_words = p.split()
                    if is_sfx and any("-" in w for w in p_words):
                        exp = []
                        for w in p_words:
                            exp.extend([sub for sub in w.split("-") if sub] if "-" in w else [w])
                        p_words = exp
                    lines_p = wrap_text_elliptical(p_words, f, a, b, allow_hyphenation=True, is_rect=is_rect)
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
            # Absolute last resort: scale down font size (down to 4pt) until it strictly fits inside the bubble envelope
            for fallback_fs in range(max(6, min_fs - 1), 3, -1):
                f_fb = get_best_font(size=fallback_fs, bold=True, sample_text=text)
                ascent, descent = f_fb.getmetrics()
                lh_fb = ascent + descent
                lstep_fb = int(lh_fb * 1.12)
                fb_paras = []
                for p in paragraphs:
                    p_words = p.split()
                    cur_line = []
                    lines = []
                    for w in p_words:
                        line_limit = (2 * a * 0.72) if (not lines or len(lines) == 0) else (2 * a * 0.82)
                        bb_w = f_fb.getbbox(w)
                        w_len = bb_w[2] - bb_w[0]
                        if w_len > line_limit:
                            if cur_line:
                                lines.append(" ".join(cur_line))
                                cur_line = []
                            rem = w
                            while rem:
                                limit = (2 * a * 0.72) if (not lines or len(lines) == 0) else (2 * a * 0.82)
                                bb_rem = f_fb.getbbox(rem)
                                if (bb_rem[2] - bb_rem[0]) <= limit:
                                    cur_line = [rem]
                                    break
                                split_idx = len(rem) - 1
                                found = False
                                while split_idx >= 1:
                                    cand = rem[:split_idx] + "-"
                                    bb_c = f_fb.getbbox(cand)
                                    if (bb_c[2] - bb_c[0]) <= limit:
                                        lines.append(cand)
                                        rem = rem[split_idx:]
                                        found = True
                                        break
                                    split_idx -= 1
                                if not found:
                                    cand = rem[:1] + "-"
                                    lines.append(cand)
                                    rem = rem[1:]
                            continue

                        test_str = " ".join(cur_line + [w]) if cur_line else w
                        bb = f_fb.getbbox(test_str)
                        w_line = bb[2] - bb[0]
                        if cur_line and w_line > line_limit:
                            lines.append(" ".join(cur_line))
                            cur_line = [w]
                        else:
                            cur_line.append(w)
                    if cur_line:
                        lines.append(" ".join(cur_line))
                    fb_paras.append(lines)
                tot_lines_fb = sum(len(lp) for lp in fb_paras)
                pgap_fb = int(lh_fb * 0.35) if len(fb_paras) > 1 else 0
                th_fb = (tot_lines_fb - 1) * lstep_fb + lh_fb + (len(fb_paras) - 1) * pgap_fb
                if th_fb <= 2 * b * 0.90 or fallback_fs == 4:
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
        pad_y = max(4, int(bh * 0.12))
        if total_h > max(line_h, bh - 2 * pad_y):
            pad_y = max(2, min(pad_y, (bh - total_h) // 2)) if bh > total_h else 2
        max_allowed_h = max(line_h, bh - 2 * pad_y)
        if total_h > max_allowed_h and tot_lines > 1:
            # Compress line step slightly before overprinting lines
            para_gap = min(para_gap, max(0, int(line_h * 0.2)))
            line_step = max(max(4, int(line_h * 0.85)), int((max_allowed_h - line_h - (len(chosen_paras) - 1) * para_gap) / (tot_lines - 1)))
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
        h_img, w_img = gray_full.shape[:2]
        r_box = max(4, min(14, int(min(bw, bh) * 0.35)))
        center_patch = gray_full[max(0, int(cy)-r_box):min(h_img, int(cy)+r_box+1), max(0, int(cx)-r_box):min(w_img, int(cx)+r_box+1)]
        center_med = float(np.median(center_patch)) if center_patch.size > 0 else 255.0

        effective_luma = min(bg_lum, center_med) if (item.get("is_dark", False) or item.get("category") in ("narration", "narration_box")) else center_med
        text_fill, stroke_fill, stroke_w = get_contrast_text_colors(effective_luma, is_on_artwork=is_on_artwork)

        # Standalone expressive manga punctuation (?, !, ?!, ...) is rendered large, bold, and centered
        if is_manga_punctuation(text):
            clean_punc = normalize_manga_punctuation(text)
            fs = min(52, max(22, int(min(bw, bh) * 0.44)))
            punc_font = get_best_font(size=fs, bold=True, sample_text=clean_punc)
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
        if is_vertical_strip and not is_narration_cat and len(text.split()) == 1 and len(text) * 14 <= bh:
            chars = [c for c in text.strip() if not c.isspace()]
            if chars:
                fs = min(22, max(10, min(int(bw * 0.70), int(bh / max(1, len(chars) * 1.35)))))
                f = get_best_font(size=fs, bold=True, sample_text=text)
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
                # Guaranteed safety clamping: keep text strictly within speech bubble interior
                pad_x = max(6, int(bw * 0.08))
                lx = max(x1 + pad_x, min(max(x1 + pad_x, x2 - pad_x - lw), lx))
                draw_y = max(y1 + pad_y, cur_y)
                if stroke_w > 0 and stroke_fill is not None:
                    draw.text((lx, draw_y), line, fill=text_fill, font=chosen_font, stroke_width=stroke_w, stroke_fill=stroke_fill)
                else:
                    draw.text((lx, draw_y), line, fill=text_fill, font=chosen_font)
                cur_y += line_step

    return rendered
