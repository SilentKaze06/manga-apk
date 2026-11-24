"""PDF Rendering and Layout Engine for Manga and Book translations."""

import html
import logging
import os
import re
import uuid
import gc
import io
import math
from pathlib import Path
from typing import Optional, Any

import numpy as np
import cv2
import pymupdf

logger = logging.getLogger("manga-server")

class PdfJobError(Exception):
    """An expected PDF validation or translation error suitable for the UI."""

PAGE_WIDTH, PAGE_HEIGHT = pymupdf.paper_rect("a4").width, pymupdf.paper_rect("a4").height
PAGE_RECT = pymupdf.Rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT)
TEXT_RECT = PAGE_RECT + (54, 54, -54, -54)
MAX_TRANSLATION_CHUNK_CHARS = 3200
COMIC_FONT_NAME = "ScreenTranslatorNoto"

try:
    import pymupdf_fonts
    COMIC_FONT_BUFFER = pymupdf_fonts.fontbuffers["notos"]()
except Exception:
    COMIC_FONT_BUFFER = pymupdf.Font("helv").buffer

try:
    COMIC_CJK_FONT_BUFFER = pymupdf.Font("cjk").buffer
except Exception:
    COMIC_CJK_FONT_BUFFER = COMIC_FONT_BUFFER

try:
    COMIC_FIGO_FONT_BUFFER = pymupdf.Font("figo").buffer
except Exception:
    COMIC_FIGO_FONT_BUFFER = COMIC_FONT_BUFFER

_CJK_CODE_PATTERN = re.compile(r"[\u2e80-\u9fff\uac00-\ud7af\uff00-\uffef\u3000-\u30ff]")
_ARABIC_CODE_PATTERN = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]")
_MANGA_PUNCTUATION_CHARS = set("?!…—–¿¡.")


def _has_renderable_content(text: str) -> bool:
    """Check if text contains alphanumeric characters or standalone manga punctuation."""
    if not text:
        return False
    if any(c.isalnum() for c in text):
        return True
    stripped = text.strip()
    return bool(stripped) and any(c in _MANGA_PUNCTUATION_CHARS for c in stripped)


def _bidi_display_line(line: str) -> str:
    """Apply bidi visual reordering per line for Arabic script."""
    if not line or not _ARABIC_CODE_PATTERN.search(line):
        return line
    try:
        from bidi.algorithm import get_display
        return get_display(line)
    except Exception:
        return line


def _bidi_display_lines(lines: list[str]) -> list[str]:
    """Apply bidi algorithm per line to a list of wrapped lines."""
    return [_bidi_display_line(line) for line in lines]


def _get_font_and_shaped_text(text: str) -> tuple[str, bytes, str]:
    """Select font and shape text for CJK, Arabic, or Latin/Cyrillic scripts."""
    # Normalize Uzbek and typographic apostrophes to standard ASCII single quote
    text_str = str(text).replace("ʻ", "'").replace("ʼ", "'").replace("‘", "'").replace("’", "'").replace("`", "'")
    if _ARABIC_CODE_PATTERN.search(text_str):
        try:
            import arabic_reshaper
            shaped = arabic_reshaper.reshape(text_str)
        except Exception:
            shaped = text_str
        return "ScreenTranslatorFigo", COMIC_FIGO_FONT_BUFFER, shaped
    elif _CJK_CODE_PATTERN.search(text_str):
        return "ScreenTranslatorCJK", COMIC_CJK_FONT_BUFFER, text_str
    return COMIC_FONT_NAME, COMIC_FONT_BUFFER, text_str


def _insert_page_font(page, font_name: str, font_buffer: bytes) -> None:
    """Insert font into page if not already inserted, preventing duplicates."""
    inserted = getattr(page, "_inserted_fonts", None)
    if inserted is None:
        inserted = set()
        try:
            for f in page.get_fonts():
                if len(f) > 4 and f[4]:
                    inserted.add(f[4])
        except Exception as exc:
            logger.debug("Failed reading page fonts: %s", exc)
        setattr(page, "_inserted_fonts", inserted)
    if font_name not in inserted:
        try:
            page.insert_font(fontname=font_name, fontbuffer=font_buffer)
        except Exception as exc:
            logger.debug("Failed inserting font %s: %s", font_name, exc)
        inserted.add(font_name)


def _translation_html(title: str, pages: list[dict], mode: str) -> str:
    pieces = [
        "<html><head><meta charset='utf-8'><style>",
        "body { font-family: sans-serif; font-size: 11pt; color: #171717; line-height: 1.42; }",
        "h1 { font-size: 22pt; margin-bottom: 22pt; }",
        "h2 { font-size: 15pt; margin: 18pt 0 8pt; }",
        "p { margin: 0 0 9pt; }",
        ".source { color: #555; font-size: 9pt; margin-top: 2pt; }",
        ".source-page { color: #888; font-size: 8pt; margin-top: 18pt; }",
        "</style></head><body>",
    ]
    if title:
        pieces.append(f"<h1>{html.escape(title)}</h1>")

    for page_number, page in enumerate(pages, 1):
        if not page.get("blocks"):
            if page.get("ocr_attempted"):
                pieces.append(f"<p class='source-page'>Manba PDF: {page_number}-bet — OCR matn topmadi.</p>")
            continue
        for index, block in enumerate(page["blocks"]):
            trans_val = block.get("translated")
            orig_val = block.get("original")
            translated = html.escape(str(trans_val if trans_val is not None else "")).replace("\n", "<br>")
            original = html.escape(str(orig_val if orig_val is not None else "")).replace("\n", "<br>")
            if not translated:
                continue
            is_heading = block.get("is_heading", False)
            tag = "h2" if is_heading else "p"
            pieces.append(f"<{tag}>{translated}</{tag}>")
            if mode == "bilingual":
                pieces.append(f"<p class='source'>{original}</p>")
        if page.get("ocr_warning"):
            pieces.append(f"<p class='source'>{html.escape(str(page['ocr_warning']))}</p>")
        pieces.append(f"<p class='source-page'>Manba PDF: {page_number}-bet</p>")
    pieces.append("</body></html>")
    return "".join(pieces)


def _write_reflowed_pdf(output_path: Path, title: str, pages: list[dict], mode: str) -> None:
    font_archive = pymupdf.Archive()
    font_css = pymupdf.css_for_pymupdf_font("notos", name="sans-serif", archive=font_archive)
    try:
        cjk_css = pymupdf.css_for_pymupdf_font("cjk", name="sans-serif", archive=font_archive)
        font_css = f"{font_css}\n{cjk_css}"
    except Exception as exc:
        logger.debug("cjk_css unavailable: %s", exc)
    try:
        figo_css = pymupdf.css_for_pymupdf_font("figo", name="sans-serif", archive=font_archive)
        font_css = f"{font_css}\n{figo_css}"
    except Exception as exc:
        logger.debug("figo_css unavailable: %s", exc)
    story = pymupdf.Story(
        _translation_html(title, pages, mode),
        user_css=font_css,
        archive=font_archive,
    )
    writer = pymupdf.DocumentWriter(str(output_path))
    device = None
    try:
        more = True
        while more:
            device = writer.begin_page(PAGE_RECT)
            more, _ = story.place(TEXT_RECT)
            story.draw(device)
            writer.end_page()
            device = None
    finally:
        if device is not None:
            del device
        writer.close()
    if "device" in locals() and device is not None:
        del device
    del writer
    gc.collect()
    if not output_path.exists() or output_path.stat().st_size < 100:
        raise PdfJobError("Tarjima PDF faylini yaratib bölmadi.")
    with pymupdf.open(output_path) as translated:
        translated.set_metadata({"title": title or "Tarjima qilingan PDF", "producer": "ScreenTranslator"})
        translated.saveIncr()
    del translated
    gc.collect()


def _wrap_for_box(
    text: str,
    font: pymupdf.Font,
    fontsize: float,
    width: float,
    allow_char_break: bool = False,
    allow_hyphenation: bool = True,
) -> Optional[list[str]]:
    lines = []
    for paragraph in str(text).splitlines() or [str(text)]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if font.text_length(candidate, fontsize=fontsize) <= width:
                current = candidate
                continue
            if current:
                lines.append(current)
                current = ""
            if font.text_length(word, fontsize=fontsize) <= width:
                current = word
            else:
                is_cjk = bool(_CJK_CODE_PATTERN.search(word))
                if not allow_char_break and not is_cjk:
                    return None
                hyphen = "-" if (allow_hyphenation and not is_cjk) else ""
                part = ""
                for character in word:
                    if font.text_length(character, fontsize=fontsize) > width:
                        return None
                    if part and font.text_length(part + character + hyphen, fontsize=fontsize) > width:
                        lines.append(part + hyphen)
                        part = character
                    else:
                        part += character
                current = part
        if current:
            lines.append(current)
    return lines


def _lines_fit_box(
    lines: list[str],
    font: pymupdf.Font,
    fontsize: float,
    lineheight: float,
    wrap_width: float,
    wrap_height: float,
    is_ellipse: bool = False,
    bubble_w: float = 0.0,
    bubble_h: float = 0.0,
) -> bool:
    if not lines:
        return False
    total_height = len(lines) * fontsize * lineheight
    if total_height > wrap_height:
        return False

    half_h = bubble_h / 2.0 if bubble_h > 0 else wrap_height / 2.0
    half_w = bubble_w / 2.0 if bubble_w > 0 else wrap_width / 2.0
    step = fontsize * lineheight

    for i, line in enumerate(lines):
        line_w = font.text_length(line, fontsize=fontsize)
        if line_w > wrap_width:
            return False
        if is_ellipse and half_h > 0 and half_w > 0:
            y_i = -(total_height / 2.0) + (i + 0.5) * step
            outer_y = abs(y_i) + 0.5 * step
            if outer_y >= half_h:
                return False
            norm_y = outer_y / half_h
            chord_w = 2.0 * half_w * math.sqrt(max(0.0, 1.0 - norm_y ** 2)) * 0.85
            if line_w > chord_w:
                return False
    return True


def _insert_fitted_text(
    page,
    rect,
    text: str,
    min_size: float,
    max_size: float,
    fill=None,
    fill_polygon: Optional[Any] = None,
    background_image: Optional[bytes] = None,
    erase_rect: Optional[list] = None,
    text_color: Optional[tuple] = None,
    bubble_shape: Optional[str] = None,
) -> bool:
    rect = pymupdf.Rect(rect)
    font_name, font_buffer, shaped_text = _get_font_and_shaped_text(text)
    if not shaped_text or not shaped_text.strip() or not _has_renderable_content(shaped_text):
        return False

    font = pymupdf.Font(fontbuffer=font_buffer)
    _insert_page_font(page, font_name, font_buffer)
    rot = int(getattr(page, "rotation", 0)) % 360
    dmat = page.derotation_matrix if rot else None
    unrot_rect = rect * dmat if dmat is not None else rect

    if bubble_shape is not None:
        is_ellipse = str(bubble_shape).lower() in ("ellipse", "oval", "round", "circle")
    elif fill_polygon is not None and fill_polygon is not False:
        is_ellipse = True
    else:
        is_ellipse = False

    # Speech bubble safe padding: oval/round speech bubbles require at least 15%-18% inset
    # (inscribed ellipse safe rect: ~0.707*w, 0.707*h) to prevent text clipping curved borders
    if is_ellipse:
        inset_x = max(4.0, rect.width * 0.16)
        inset_y = max(4.0, rect.height * 0.16)
    else:
        inset_x = max(3.0, rect.width * 0.08)
        inset_y = max(3.0, rect.height * 0.08)
    visual_text_rect = rect + (inset_x, inset_y, -inset_x, -inset_y)
    if visual_text_rect.is_empty or visual_text_rect.width < 4 or visual_text_rect.height < 4:
        # Fallback to minimal 1.5pt padding
        visual_text_rect = rect + (1.5, 1.5, -1.5, -1.5)
        if visual_text_rect.is_empty or visual_text_rect.width < 4 or visual_text_rect.height < 4:
            return False
    unrot_text_rect = visual_text_rect * dmat if dmat is not None else visual_text_rect

    # When rot in (90, 270), swap width and height for text wrapping boundaries
    wrap_width = unrot_text_rect.height if rot in (90, 270) else unrot_text_rect.width
    wrap_height = unrot_text_rect.width if rot in (90, 270) else unrot_text_rect.height
    bubble_w = unrot_rect.height if rot in (90, 270) else unrot_rect.width
    bubble_h = unrot_rect.width if rot in (90, 270) else unrot_rect.height
    if wrap_width <= 0 or wrap_height <= 0:
        return False

    # Dynamic text color: white for dark backgrounds, dark for light backgrounds
    if text_color is None:
        if fill is not None and len(fill) >= 3 and sum(fill[:3]) / 3.0 < 0.45:
            text_color = (0.95, 0.95, 0.95)
        elif background_image is not None:
            try:
                from PIL import Image
                with Image.open(io.BytesIO(background_image)) as bg_im:
                    bg_arr = np.array(bg_im.convert("L"))
                    bh, bw = bg_arr.shape[:2]
                    my = max(1, int(bh * 0.12))
                    mx = max(1, int(bw * 0.12))
                    interior = bg_arr[my:bh - my, mx:bw - mx] if bh > 2 * my and bw > 2 * mx else bg_arr
                    int_luma = float(np.median(interior)) / 255.0
                    mean_luma = float(np.mean(bg_arr)) / 255.0
                    text_color = (0.95, 0.95, 0.95) if (int_luma < 0.45 or mean_luma < 0.45) else (0.06, 0.06, 0.06)
            except Exception:
                text_color = (0.06, 0.06, 0.06)
        else:
            text_color = (0.06, 0.06, 0.06)

    chosen_lines = None
    chosen_size = None
    chosen_lh = 1.18
    chosen_rem = 0.0

    # Pass 1: Try to fit without hyphenation/char-break from max_size down to min_size
    # For space-less CJK text, allow character-level wrapping so CJK does not shrink to <= 4.5pt
    is_cjk_target = bool(_CJK_CODE_PATTERN.search(shaped_text))
    fontsize = max_size
    while fontsize >= min_size:
        test_lines = _wrap_for_box(shaped_text, font, fontsize, wrap_width, allow_char_break=is_cjk_target, allow_hyphenation=False)
        if test_lines and _lines_fit_box(test_lines, font, fontsize, 1.18, wrap_width, wrap_height, is_ellipse, bubble_w, bubble_h):
            display_lines = _bidi_display_lines(test_lines)
            test_shape = page.new_shape()
            rem = test_shape.insert_textbox(
                unrot_text_rect, "\n".join(display_lines), fontname=font_name,
                fontsize=fontsize, lineheight=1.18, rotate=rot
            )
            if rem >= 0:
                chosen_lines = display_lines
                chosen_size = fontsize
                chosen_lh = 1.18
                chosen_rem = rem
                break
        fontsize -= 0.5

    # Pass 1.5: Prefer shrinking font size slightly (-0.5 to -1.5 pt below min_size)
    # over awkward mid-word hyphenation (e.g. BO'LI-SHINI)
    if chosen_lines is None:
        shrink_limit = max(3.0, min_size - 1.5)
        fontsize = min_size - 0.5
        while fontsize >= shrink_limit:
            test_lines = _wrap_for_box(shaped_text, font, fontsize, wrap_width, allow_char_break=is_cjk_target, allow_hyphenation=False)
            if test_lines:
                lh = 1.14 if fontsize >= 4.5 else 1.10
                if _lines_fit_box(test_lines, font, fontsize, lh, wrap_width, wrap_height, is_ellipse, bubble_w, bubble_h):
                    display_lines = _bidi_display_lines(test_lines)
                    test_shape = page.new_shape()
                    rem = test_shape.insert_textbox(
                        unrot_text_rect, "\n".join(display_lines), fontname=font_name,
                        fontsize=fontsize, lineheight=lh, rotate=rot
                    )
                    if rem >= 0:
                        chosen_lines = display_lines
                        chosen_size = fontsize
                        chosen_lh = lh
                        chosen_rem = rem
                        break
            fontsize -= 0.25

    # Pass 2: If tight, allow hyphenation down to 3.5 — require total_height <= wrap_height strictly (NO 1.15 overflow)
    if chosen_lines is None:
        fontsize = max(3.5, min_size - 0.5)
        while fontsize >= 3.5:
            test_lines = _wrap_for_box(shaped_text, font, fontsize, wrap_width, allow_char_break=True, allow_hyphenation=True)
            if test_lines:
                lh = 1.12 if fontsize >= 4.5 else 1.08
                if _lines_fit_box(test_lines, font, fontsize, lh, wrap_width, wrap_height, is_ellipse, bubble_w, bubble_h):
                    display_lines = _bidi_display_lines(test_lines)
                    test_shape = page.new_shape()
                    rem = test_shape.insert_textbox(
                        unrot_text_rect, "\n".join(display_lines), fontname=font_name,
                        fontsize=fontsize, lineheight=lh, rotate=rot
                    )
                    if rem >= 0:
                        chosen_lines = display_lines
                        chosen_size = fontsize
                        chosen_lh = lh
                        chosen_rem = rem
                        break
            fontsize -= 0.25

    # Pass 3: Emergency compact down to 2.5 pt (and down to 2.2 pt or tighter lineheight = 1.08 / 1.05)
    # until total_height <= wrap_height AND every line fits strictly
    if chosen_lines is None:
        fontsize = 3.25
        while fontsize >= 2.2:
            for lh in (1.08, 1.05):
                test_lines = _wrap_for_box(shaped_text, font, fontsize, wrap_width, allow_char_break=True, allow_hyphenation=True)
                if test_lines and _lines_fit_box(test_lines, font, fontsize, lh, wrap_width, wrap_height, is_ellipse, bubble_w, bubble_h):
                    display_lines = _bidi_display_lines(test_lines)
                    test_shape = page.new_shape()
                    rem = test_shape.insert_textbox(
                        unrot_text_rect, "\n".join(display_lines), fontname=font_name,
                        fontsize=fontsize, lineheight=lh, rotate=rot
                    )
                    if rem >= 0:
                        chosen_lines = display_lines
                        chosen_size = fontsize
                        chosen_lh = lh
                        chosen_rem = rem
                        break
            if chosen_lines is not None:
                break
            fontsize -= 0.2

    # If text still cannot fit at all, DO NOT erase or clean bubble! Retain source art.
    if chosen_lines is None:
        return False

    # Mathematical vertical centering: chosen_rem is the exact unused vertical room in visual_text_rect
    vert_offset = max(0.0, chosen_rem / 2.0)
    visual_centered = pymupdf.Rect(
        visual_text_rect.x0,
        visual_text_rect.y0 + vert_offset,
        visual_text_rect.x1,
        visual_text_rect.y1,
    )
    target_rect = visual_centered * dmat if dmat is not None else visual_centered
    text_content = "\n".join(chosen_lines)

    # Inserter safety: verify insert_textbox succeeds (rem >= 0) on a scratch/trial check
    test_shape = page.new_shape()
    rem = test_shape.insert_textbox(
        target_rect,
        text_content,
        fontname=font_name,
        fontsize=chosen_size,
        lineheight=chosen_lh,
        align=pymupdf.TEXT_ALIGN_CENTER,
        rotate=rot,
    )

    if rem < 0:
        # Retry unshifted if shifted rect was tight
        target_rect = unrot_text_rect
        test_shape = page.new_shape()
        rem = test_shape.insert_textbox(
            target_rect,
            text_content,
            fontname=font_name,
            fontsize=chosen_size,
            lineheight=chosen_lh,
            align=pymupdf.TEXT_ALIGN_CENTER,
            rotate=rot,
        )

    # Step down chosen_size by 0.5 pt increments until insert_textbox returns rem >= 0
    while rem < 0 and chosen_size > 2.5:
        chosen_size = max(2.5, chosen_size - 0.5)
        cur_lines = _wrap_for_box(shaped_text, font, chosen_size, wrap_width, allow_char_break=True, allow_hyphenation=True)
        if cur_lines:
            chosen_lines = _bidi_display_lines(cur_lines)
            text_content = "\n".join(chosen_lines)
        test_shape = page.new_shape()
        rem = test_shape.insert_textbox(
            target_rect,
            text_content,
            fontname=font_name,
            fontsize=chosen_size,
            lineheight=chosen_lh,
            align=pymupdf.TEXT_ALIGN_CENTER,
            rotate=rot,
        )

    # If insert_textbox still fails at minimum size, tighten lineheight (1.05) and re-wrap to exact render_rect width minus 2 pt safety margin
    if rem < 0:
        chosen_lh = 1.05
        chosen_size = 2.5
        safety_width = max(4.0, (unrot_text_rect.height if rot in (90, 270) else unrot_text_rect.width) - 2.0)
        cur_lines = _wrap_for_box(shaped_text, font, chosen_size, safety_width, allow_char_break=True, allow_hyphenation=True)
        if cur_lines:
            chosen_lines = _bidi_display_lines(cur_lines)
            text_content = "\n".join(chosen_lines)
        test_shape = page.new_shape()
        rem = test_shape.insert_textbox(
            target_rect,
            text_content,
            fontname=font_name,
            fontsize=chosen_size,
            lineheight=chosen_lh,
            align=pymupdf.TEXT_ALIGN_CENTER,
            rotate=rot,
        )

    # Never fall back to unclipped shape.insert_text with overflowing lines when insert_textbox returns rem < 0
    if rem < 0:
        return False

    # Erase original text / draw cleaned background patch ONLY after insertion is verified
    if background_image is not None:
        page.insert_image(
            unrot_rect,
            stream=background_image,
            rotate=rot,
            keep_proportion=False,
            overlay=True,
        )
    elif fill_polygon and isinstance(fill_polygon, (list, tuple)) and len(fill_polygon) >= 3:
        unrot_poly = [pymupdf.Point(point[0], point[1]) * dmat for point in fill_polygon] if dmat is not None else [pymupdf.Point(point[0], point[1]) for point in fill_polygon]
        page.draw_polyline(
            unrot_poly,
            color=None,
            fill=fill or (1.0, 1.0, 1.0),
            closePath=True,
            overlay=True,
        )
    elif fill_polygon is True or bubble_shape == "ellipse":
        page.draw_oval(
            unrot_rect,
            color=None,
            fill=fill or (1.0, 1.0, 1.0),
            overlay=True,
        )
    elif fill is not None:
        if erase_rect:
            try:
                e_rect = pymupdf.Rect(erase_rect) & page.rect
                if not e_rect.is_empty:
                    e_padded = e_rect + (-3, -3, 3, 3) & page.rect
                    unrot_e_rect = e_padded * dmat if dmat is not None else e_padded
                    page.draw_rect(unrot_e_rect, color=None, fill=fill, overlay=True)
                else:
                    page.draw_rect(unrot_rect, color=None, fill=fill, overlay=True)
            except Exception:
                page.draw_rect(unrot_rect, color=None, fill=fill, overlay=True)
        else:
            page.draw_rect(unrot_rect, color=None, fill=fill, overlay=True)

    shape = page.new_shape()
    rem_final = shape.insert_textbox(
        target_rect,
        text_content,
        fontname=font_name,
        fontsize=chosen_size,
        lineheight=chosen_lh,
        color=text_color,
        align=pymupdf.TEXT_ALIGN_CENTER,
        rotate=rot,
    )
    if rem_final < 0:
        return False
    shape.commit(overlay=True)
    return True


def _comic_pdf_transcript_page(document, source_page: int, blocks: list[dict], include_source: bool) -> None:
    """Flow every transcript line onto as many pages as needed."""
    page_rect = pymupdf.paper_rect("a4")
    margin = 48
    page = None
    y = page_rect.height

    def add_page():
        new_page = document.new_page(width=page_rect.width, height=page_rect.height)
        header_text = f"Komiks - {source_page}-bet tarjimasi"
        h_name, h_buf, h_shaped = _get_font_and_shaped_text(header_text)
        _insert_page_font(new_page, h_name, h_buf)
        new_page.insert_text((margin, margin + 18), h_shaped,
                             fontname=h_name, fontsize=16, color=(0.05, 0.05, 0.05))
        return new_page, margin + 48

    item_index = 0
    for block in blocks:
        original = str(block.get("original", "")).strip()
        translated = str(block.get("translated", "")).strip()
        if not original and not translated:
            continue
        item_index += 1
        paras = []
        if include_source and original:
            paras.append((f"{item_index}. [Asl]: ", original))
            if translated:
                paras.append(("   [Tarjima]: ", translated))
        elif translated:
            paras.append((f"{item_index}. ", translated))
        elif original:
            paras.append((f"{item_index}. ", original))

        for prefix, para_text in paras:
            f_name, f_buf, shaped_para = _get_font_and_shaped_text(f"{prefix}{para_text}")
            font = pymupdf.Font(fontbuffer=f_buf)
            lines = _wrap_for_box(shaped_para, font, 10, page_rect.width - 2 * margin, allow_char_break=True) or [shaped_para]
            lines = _bidi_display_lines(lines)
            for line in lines:
                if page is None or y + 14 > page_rect.height - margin:
                    page, y = add_page()
                if line:
                    _insert_page_font(page, f_name, f_buf)
                    page.insert_text((margin, y), line, fontname=f_name, fontsize=10)
                y += 14
            y += 4
        y += 6


def _sample_page_background_color(src_page, rect: pymupdf.Rect) -> list[float]:
    """Sample median background color from page rect interior."""
    try:
        pix = src_page.get_pixmap(clip=rect, dpi=72)
        if pix.width > 0 and pix.height > 0:
            pix_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
            ph, pw = pix_np.shape[:2]
            my = max(1, int(ph * 0.12))
            mx = max(1, int(pw * 0.12))
            int_pix = pix_np[my:ph - my, mx:pw - mx] if ph > 2 * my and pw > 2 * mx else pix_np
            return list(np.median(int_pix[..., :3], axis=(0, 1)) / 255.0)
    except Exception as exc:
        logger.debug("Failed sampling page background color: %s", exc)
    return [1.0, 1.0, 1.0]


def _write_comic_pdf(
    input_path: Path,
    output_path: Path,
    pages: list[dict],
    include_source: bool,
) -> None:
    """Copy comic artwork, mask detected text boxes, and add fitted translations."""
    try:
        source = pymupdf.open(str(input_path))
    except Exception as exc:
        raise PdfJobError(f"Komiks PDFni qayta oçib bölmadi: {exc}") from exc

    translated = pymupdf.open()
    transcript_pages = []
    job_dir = input_path.parent
    tmp_output_path = None
    try:
        for page_index, page_data in enumerate(pages):
            if page_index >= source.page_count:
                break
            src_page = source[page_index]
            if src_page.rect.width <= 0 or src_page.rect.height <= 0:
                continue
            page_blocks = page_data.get("blocks", [])
            cleaned_fname = page_data.get("cleaned_image_filename")
            cleaned_path = (job_dir / cleaned_fname) if cleaned_fname else None

            if cleaned_path and cleaned_path.exists():
                from PIL import Image
                translated.insert_pdf(source, from_page=page_index, to_page=page_index)
                page = translated[-1]
                with Image.open(cleaned_path) as image:
                    cleaned_image = image.convert("RGB")
                overflow = []
                for block in page_blocks:
                    rect = pymupdf.Rect(block.get("bbox", [0, 0, 0, 0])) & page.rect
                    if rect.is_empty or rect.width < 5 or rect.height < 5:
                        overflow.append(block)
                        continue

                    patch_bytes = None
                    patch_np = None
                    if src_page.rect.width > 0 and src_page.rect.height > 0:
                        scale_x = cleaned_image.width / src_page.rect.width
                        scale_y = cleaned_image.height / src_page.rect.height
                        cx0 = round((rect.x0 - src_page.rect.x0) * scale_x)
                        cy0 = round((rect.y0 - src_page.rect.y0) * scale_y)
                        cx1 = round((rect.x1 - src_page.rect.x0) * scale_x)
                        cy1 = round((rect.y1 - src_page.rect.y0) * scale_y)
                        cx0 = max(0, min(cleaned_image.width - 1, cx0))
                        cy0 = max(0, min(cleaned_image.height - 1, cy0))
                        cx1 = max(cx0 + 1, min(cleaned_image.width, cx1))
                        cy1 = max(cy0 + 1, min(cleaned_image.height, cy1))
                        if cx1 > cx0 and cy1 > cy0:
                            patch = cleaned_image.crop((cx0, cy0, cx1, cy1))
                            patch_np = np.array(patch)
                            is_dark_or_gradient = False
                            if patch_np.size > 0:
                                p_gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY) if patch_np.ndim == 3 else patch_np
                                p_med = float(np.median(p_gray))
                                p_std = float(np.std(p_gray))
                                is_dark_or_gradient = (p_med < 115) or (p_std >= 5.0)
                            if block.get("cleaned", False) or is_dark_or_gradient or str(block.get("category", "")).lower() in ("narration", "box", "rect"):
                                patch_buffer = io.BytesIO()
                                patch.save(patch_buffer, format="PNG")
                                patch_bytes = patch_buffer.getvalue()

                    if patch_np is not None and patch_np.size > 0:
                        ph, pw = patch_np.shape[:2]
                        my = max(1, int(ph * 0.12))
                        mx = max(1, int(pw * 0.12))
                        int_patch = patch_np[my:ph - my, mx:pw - mx] if ph > 2 * my and pw > 2 * mx else patch_np
                        raw_bg = list(np.median(int_patch[..., :3], axis=(0, 1)) / 255.0)
                        block["background"] = raw_bg
                    else:
                        raw_bg = block.get("background")
                        if not isinstance(raw_bg, (list, tuple)) or len(raw_bg) < 3:
                            raw_bg = _sample_page_background_color(src_page, rect)
                            block["background"] = raw_bg

                    bg_brightness = float(sum(raw_bg[:3]) / 3.0)
                    if bg_brightness < 0.45:
                        inferred_text_color = (0.95, 0.95, 0.95)
                    else:
                        inferred_text_color = (0.06, 0.06, 0.06)
                        if patch_np is not None and patch_np.size > 0:
                            patch_luma = float(np.mean(patch_np[..., :3])) / 255.0
                            if patch_luma < 0.45:
                                inferred_text_color = (0.95, 0.95, 0.95)

                    trans_text = str(block.get("translated", "")).strip()
                    orig_text = str(block.get("original", "")).strip()

                    if not trans_text or not _has_renderable_content(trans_text):
                        overflow.append(block)
                        continue

                    norm_trans = "".join(ch for ch in trans_text.casefold() if ch.isalnum())
                    norm_orig = "".join(ch for ch in orig_text.casefold() if ch.isalnum())
                    is_short_or_numeric = len(orig_text.split()) <= 2 or norm_orig.isdigit()
                    if norm_orig and norm_orig == norm_trans and not block.get("cleaned", False) and not is_short_or_numeric:
                        overflow.append(block)
                        continue

                    max_size = min(16.0, max(6.5, rect.height * 0.45))
                    if patch_bytes:
                        block_fill = None
                    else:
                        if bg_brightness < 0.35:
                            block_fill = tuple(max(0.0, min(0.15, float(c))) for c in raw_bg[:3])
                        elif bg_brightness > 0.92:
                            block_fill = (1.0, 1.0, 1.0)
                        else:
                            block_fill = tuple(max(0.0, min(1.0, float(c))) for c in raw_bg[:3])

                    cat = str(block.get("category", "")).lower()
                    b_shape = str(block.get("bubble_shape", "")).lower() or str(block.get("shape", "")).lower()
                    if cat in ("narration", "box", "rect") or b_shape in ("rect", "box", "rectangle"):
                        bubble_shape = "rect"
                    else:
                        bubble_shape = "ellipse"

                    poly = block.get("overlay_polygon")
                    fill_poly = poly if (isinstance(poly, list) and len(poly) >= 3) else (True if bubble_shape == "ellipse" else None)

                    if not _insert_fitted_text(
                        page,
                        rect,
                        trans_text,
                        min_size=5.0,
                        max_size=max_size,
                        fill=block_fill,
                        fill_polygon=fill_poly,
                        background_image=patch_bytes,
                        text_color=inferred_text_color,
                        bubble_shape=bubble_shape,
                    ):
                        overflow.append(block)
                if include_source:
                    transcript_pages.append((page_index + 1, page_blocks))
                elif overflow:
                    valid_ov = [b for b in overflow if _has_renderable_content(str(b.get("translated", "")))]
                    if valid_ov:
                        transcript_pages.append((page_index + 1, valid_ov))
            else:
                translated.insert_pdf(source, from_page=page_index, to_page=page_index)
                page = translated[-1]
                overflow = []
                for block in page_blocks:
                    try:
                        trans_text = str(block.get("translated", "")).strip()
                        orig_text = str(block.get("original", "")).strip()
                        if not trans_text or not _has_renderable_content(trans_text):
                            continue
                        norm_trans = "".join(ch for ch in trans_text.casefold() if ch.isalnum())
                        norm_orig = "".join(ch for ch in orig_text.casefold() if ch.isalnum())
                        is_short_or_numeric = len(orig_text.split()) <= 2 or norm_orig.isdigit()
                        if norm_orig and norm_orig == norm_trans and not is_short_or_numeric:
                            overflow.append(block)
                            continue
                        polygon = block.get("overlay_polygon")
                        if not block.get("overlay_bbox") or not isinstance(polygon, list):
                            overflow.append(block)
                            continue
                        rect = pymupdf.Rect(block["overlay_bbox"]) & page.rect
                        if rect.is_empty or rect.width < 5 or rect.height < 5:
                            overflow.append(block)
                            continue
                        patch_bytes = None

                        raw_color = block.get("background")
                        if not isinstance(raw_color, (list, tuple)) or len(raw_color) < 3:
                            raw_color = _sample_page_background_color(src_page, rect)
                            block["background"] = raw_color
                        bg_brightness = sum(raw_color[:3]) / 3.0
                        if patch_bytes:
                            fill = None
                        elif bg_brightness < 0.35:
                            fill = tuple(max(0.0, min(0.15, float(c))) for c in raw_color[:3])
                        elif bg_brightness > 0.92:
                            fill = (1.0, 1.0, 1.0)
                        else:
                            fill = tuple(max(0.0, min(1.0, float(c))) for c in raw_color[:3])

                        inferred_text_color = (0.95, 0.95, 0.95) if bg_brightness < 0.45 else (0.06, 0.06, 0.06)
                        cat = str(block.get("category", "")).lower()
                        b_shape = str(block.get("bubble_shape", "")).lower() or str(block.get("shape", "")).lower()
                        if cat in ("narration", "box", "rect") or b_shape in ("rect", "box", "rectangle"):
                            bubble_shape = "rect"
                        else:
                            bubble_shape = "ellipse"

                        fill_poly = polygon if (isinstance(polygon, list) and len(polygon) >= 3) else (True if bubble_shape == "ellipse" else None)

                        if not _insert_fitted_text(
                            page,
                            rect,
                            trans_text,
                            min_size=5.0,
                            max_size=min(16.0, max(6.5, rect.height * 0.45)),
                            fill=fill,
                            fill_polygon=fill_poly,
                            background_image=patch_bytes,
                            text_color=inferred_text_color,
                            bubble_shape=bubble_shape,
                        ):
                            overflow.append(block)
                    except Exception as exc:
                        logger.debug("Failed rendering block: %s", exc)
                        overflow.append(block)

                if include_source:
                    transcript_pages.append((page_index + 1, page_blocks))
                elif overflow:
                    valid_ov = [b for b in overflow if _has_renderable_content(str(b.get("translated", "")))]
                    if valid_ov:
                        transcript_pages.append((page_index + 1, valid_ov))
                elif not page_blocks and page_data.get("ocr_attempted"):
                    transcript_pages.append((page_index + 1, [{"original": "", "translated": "Bu betda matn aniqlanmadi."}]))

        for source_page, blocks in transcript_pages:
            _comic_pdf_transcript_page(translated, source_page, blocks, include_source)

        translated.set_metadata({"title": f"{input_path.stem} — tarjima qilingan komiks", "producer": "ScreenTranslator"})
        tmp_output_path = output_path.with_name(f"{output_path.name}.{uuid.uuid4().hex[:8]}.tmp")
        translated.save(str(tmp_output_path), garbage=4, deflate=True)
    finally:
        try:
            translated.close()
        except Exception:
            pass
        try:
            source.close()
        except Exception:
            pass
    if tmp_output_path is None or not tmp_output_path.exists() or tmp_output_path.stat().st_size < 100:
        if tmp_output_path is not None and tmp_output_path.exists():
            try:
                tmp_output_path.unlink()
            except OSError as exc:
                logger.debug("Failed unlinking invalid tmp_output_path: %s", exc)
        raise PdfJobError("Komiks tarjima PDF faylini yaratib bölmadi.")
    os.replace(tmp_output_path, output_path)
