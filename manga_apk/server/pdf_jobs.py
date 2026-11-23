"""Persistent PDF translation jobs for selectable-text and OCR book PDFs."""

import html
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
import uuid
import gc
import inspect
import io
import copy
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import pymupdf

logger = logging.getLogger("manga-server")

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
        except Exception:
            pass
        setattr(page, "_inserted_fonts", inserted)
    if font_name not in inserted:
        try:
            page.insert_font(fontname=font_name, fontbuffer=font_buffer)
        except Exception:
            pass
        inserted.add(font_name)


PAGE_NUMBER_PATTERN = re.compile(r"^(?:page\s*)?(?:\d{1,5}|[ivxlcdm]{1,8})$", re.IGNORECASE)


class PdfJobError(Exception):
    """An expected PDF validation or translation error suitable for the UI."""


class PdfQueueFullError(PdfJobError):
    """The bounded local worker queue cannot accept another job yet."""


def _atomic_json_write(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode="w", encoding="utf-8")
    tmp_path = Path(tmp_file.name)
    try:
        json.dump(value, tmp_file, ensure_ascii=False, indent=2)
        tmp_file.flush()
        tmp_file.close()
        for attempt in range(10):
            try:
                os.replace(tmp_path, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.02)
    except Exception:
        try:
            if not tmp_file.closed:
                tmp_file.close()
        except Exception:
            pass
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def _order_page_blocks(blocks: list[dict], page_width: float) -> list[dict]:
    """Read a clear two-column page down the left column, then down the right."""
    natural_order = sorted(blocks, key=lambda block: (block["bbox"][1], block["bbox"][0]))
    if len(natural_order) < 4 or page_width <= 0:
        return natural_order

    # Keep full-width headings/footers outside the columns. If one interrupts
    # the body, prefer the original top-to-bottom order over a risky guess.
    column_blocks = []
    spanning_blocks = []
    for block in natural_order:
        width = float(block["bbox"][2]) - float(block["bbox"][0])
        (column_blocks if width < page_width * 0.72 else spanning_blocks).append(block)
    if len(column_blocks) < 4:
        return natural_order

    starts = sorted(column_blocks, key=lambda block: (float(block["bbox"][0]), float(block["bbox"][1])))
    gap_threshold = max(25.0, page_width * 0.11)
    best_gap = None
    for index in range(1, len(starts)):
        gap = float(starts[index]["bbox"][0]) - float(starts[index - 1]["bbox"][0])
        if gap < gap_threshold or index < 2 or len(starts) - index < 2:
            continue
        if best_gap is None or gap > best_gap[0]:
            best_gap = (gap, index)
    if best_gap is None:
        return natural_order

    split_x = (
        float(starts[best_gap[1] - 1]["bbox"][0]) + float(starts[best_gap[1]]["bbox"][0])
    ) / 2
    left_column = [block for block in column_blocks if float(block["bbox"][0]) < split_x]
    right_column = [block for block in column_blocks if float(block["bbox"][0]) >= split_x]
    if not left_column or not right_column:
        return natural_order
    max_left_x2 = max(float(block["bbox"][2]) for block in left_column)
    min_right_x0 = min(float(block["bbox"][0]) for block in right_column)
    if max_left_x2 > min_right_x0:
        return natural_order

    first_body_y = min(float(block["bbox"][1]) for block in column_blocks)
    last_body_y = max(float(block["bbox"][3]) for block in column_blocks)
    top_spanning = [block for block in spanning_blocks if float(block["bbox"][3]) <= first_body_y]
    bottom_spanning = [block for block in spanning_blocks if float(block["bbox"][1]) >= last_body_y]
    if len(top_spanning) + len(bottom_spanning) != len(spanning_blocks):
        return natural_order

    by_position = lambda block: (block["bbox"][1], block["bbox"][0])
    return (
        sorted(top_spanning, key=by_position)
        + sorted(left_column, key=by_position)
        + sorted(right_column, key=by_position)
        + sorted(bottom_spanning, key=by_position)
    )


def _read_page_blocks(page) -> list[dict]:
    blocks = []
    for block in page.get_text("dict", sort=True).get("blocks", []):
        if block.get("type") != 0:
            continue
        lines = []
        weighted_sizes = []
        for line in block.get("lines", []):
            line_text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            if line_text:
                lines.append(line_text)
            for span in line.get("spans", []):
                span_text = span.get("text", "").strip()
                if span_text:
                    span_sz = float(span.get("size", 0))
                    if span_sz > 0:
                        weighted_sizes.extend([span_sz] * len(span_text))
        text = " ".join(lines)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            if weighted_sizes:
                sorted_sizes = sorted(weighted_sizes)
                p75_idx = int(len(sorted_sizes) * 0.75)
                block_font_size = sorted_sizes[min(p75_idx, len(sorted_sizes) - 1)]
            else:
                block_font_size = 0.0
            blocks.append({
                "text": text,
                "font_size": block_font_size,
                "bbox": block.get("bbox", [0, 0, 0, 0]),
            })
    return _order_page_blocks(blocks, float(page.rect.width))


def _text_key(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _find_repeated_page_chrome(document) -> set[str]:
    """Find identical short header/footer blocks repeated on most pages."""
    page_count = document.page_count
    if page_count < 3:
        return set()
    threshold = max(3, (page_count * 4 + 4) // 5)
    occurrences: dict[str, set[int]] = {}
    for page_index, page in enumerate(document):
        page_height = float(page.rect.height)
        for block in _read_page_blocks(page):
            text = block["text"]
            if not text or len(text) > 100:
                continue
            y0, y1 = float(block["bbox"][1]), float(block["bbox"][3])
            if y1 <= page_height * 0.08 or y0 >= page_height * 0.92:
                occurrences.setdefault(_text_key(text), set()).add(page_index)
    return {text for text, pages in occurrences.items() if len(pages) >= threshold}


def _is_page_chrome(block: dict, page_height: float, repeated_chrome: set[str]) -> bool:
    bbox = block.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False
    y0, y1 = float(bbox[1]), float(bbox[3])
    if y1 > page_height * 0.08 and y0 < page_height * 0.92:
        return False
    text = _text_key(str(block.get("text", "")))
    return text in repeated_chrome or bool(PAGE_NUMBER_PATTERN.fullmatch(text))


def _image_coverage(page) -> float:
    """Estimate how much of a page is covered by embedded images."""
    page_area = max(1.0, page.rect.get_area())
    image_area = 0.0
    for image in page.get_images(full=True):
        try:
            for rect in page.get_image_rects(image[0]):
                image_area += pymupdf.Rect(rect).get_area()
        except Exception:
            continue
    return min(1.0, image_area / page_area)


def _page_needs_ocr(page, blocks: Optional[list[dict]] = None) -> bool:
    """Detect image-only scans and pages with only a small text overlay (for example page numbers)."""
    blocks = _read_page_blocks(page) if blocks is None else blocks
    if not page.get_images():
        return False
    text_length = sum(len(block["text"]) for block in blocks)
    coverage = _image_coverage(page)
    if not text_length:
        return coverage >= 0.05
    return text_length < 120 and coverage >= 0.35


def inspect_pdf(pdf_bytes: bytes, max_pages: int) -> dict:
    if not pdf_bytes.startswith(b"%PDF-"):
        raise PdfJobError("Tanlangan fayl haqiqiy PDF emas.")
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.needs_pass:
                raise PdfJobError("Parol bilan himoyalangan PDF hozircha ochilmaydi.")
            page_count = document.page_count
            if page_count < 1:
                raise PdfJobError("PDF ichida sahifa topilmadi.")
            if page_count > max_pages:
                raise PdfJobError(f"PDF {max_pages} betdan oşmasligi kerak.")
            title = (document.metadata or {}).get("title", "").strip()
            native_text_pages = 0
            scanned_pages = []
            empty_pages = []
            for index, page in enumerate(document):
                blocks = _read_page_blocks(page)
                if _page_needs_ocr(page, blocks):
                    scanned_pages.append(index + 1)
                elif blocks:
                    native_text_pages += 1
                else:
                    empty_pages.append(index + 1)
            return {
                "page_count": page_count,
                "title": title,
                "native_text_pages": native_text_pages,
                "scanned_pages": scanned_pages,
                "empty_pages": empty_pages,
            }
    except PdfJobError:
        raise
    except Exception as exc:
        raise PdfJobError(f"PDFni oçib bölmadi: {exc}") from exc


def _split_page_blocks(blocks: list[dict]) -> list[dict]:
    """Keep paragraphs together when possible and cap provider request size."""
    chunks = []
    current = []
    current_size = 0
    current_heading = None
    for block in blocks:
        text = block["text"]
        is_heading = block.get("font_size", 0) >= 15 and len(text) < 180
        if current and (current_heading != is_heading or (is_heading and current_heading)):
            chunks.append({"text": "\n\n".join(current), "is_heading": current_heading})
            current, current_size = [], 0
        if len(text) > MAX_TRANSLATION_CHUNK_CHARS:
            if current:
                chunks.append({"text": "\n\n".join(current), "is_heading": current_heading})
                current, current_size = [], 0
            remainder = text
            while len(remainder) > MAX_TRANSLATION_CHUNK_CHARS:
                split_at = remainder.rfind(" ", 0, MAX_TRANSLATION_CHUNK_CHARS)
                if split_at < MAX_TRANSLATION_CHUNK_CHARS // 2:
                    split_at = MAX_TRANSLATION_CHUNK_CHARS
                chunks.append({"text": remainder[:split_at].strip(), "is_heading": is_heading})
                remainder = remainder[split_at:].strip()
            if remainder:
                current, current_size = [remainder], len(remainder)
                current_heading = is_heading
            continue
        extra_size = len(text) + (2 if current else 0)
        if current and current_size + extra_size > MAX_TRANSLATION_CHUNK_CHARS:
            chunks.append({"text": "\n\n".join(current), "is_heading": current_heading})
            current, current_size = [], 0
        current.append(text)
        current_size += len(text) + (2 if len(current) > 1 else 0)
        current_heading = is_heading
    if current:
        chunks.append({"text": "\n\n".join(current), "is_heading": current_heading})
    return chunks


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
    except Exception:
        pass
    try:
        figo_css = pymupdf.css_for_pymupdf_font("figo", name="sans-serif", archive=font_archive)
        font_css = f"{font_css}\n{figo_css}"
    except Exception:
        pass
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
    # PyMuPDF's Windows file handles are released when the writer/device
    # wrappers are collected, after close() has flushed the PDF.
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


def _wrap_for_box(text: str, font: pymupdf.Font, fontsize: float, width: float, allow_char_break: bool = False) -> Optional[list[str]]:
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
                    # Single word does not fit on one line at this font size.
                    # Tell caller to decrease fontsize rather than chopping the word!
                    return None
                hyphen = "" if is_cjk else "-"
                part = ""
                for character in word:
                    if part and font.text_length(part + character + hyphen, fontsize=fontsize) > width:
                        lines.append(part + hyphen)
                        part = character
                    else:
                        part += character
                current = part
        if current:
            lines.append(current)
    return lines


def _insert_fitted_text(
    page,
    rect,
    text: str,
    min_size: float,
    max_size: float,
    fill=None,
    fill_polygon: Optional[list[list[float]]] = None,
    background_image: Optional[bytes] = None,
    erase_rect: Optional[list] = None,
    text_color: Optional[tuple] = None,
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

    # Speech bubble safe padding: 8-10% inset protects curved borders while maximizing text area
    if fill_polygon:
        inset_x = max(4.0, rect.width * 0.12)
        inset_y = max(4.0, rect.height * 0.12)
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
    if wrap_width <= 0 or wrap_height <= 0:
        return False

    # Dynamic text color: white for dark backgrounds, dark for light backgrounds
    if text_color is None:
        if fill is not None and len(fill) >= 3 and sum(fill[:3]) / 3.0 < 0.45:
            text_color = (0.95, 0.95, 0.95)
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
        test_lines = _wrap_for_box(shaped_text, font, fontsize, wrap_width, allow_char_break=is_cjk_target)
        if test_lines:
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

    # Pass 2: If tight, allow hyphenation down to 3.5
    if chosen_lines is None:
        fontsize = max(3.5, min_size - 0.5)
        while fontsize >= 3.5:
            test_lines = _wrap_for_box(shaped_text, font, fontsize, wrap_width, allow_char_break=True)
            if test_lines:
                display_lines = _bidi_display_lines(test_lines)
                lh = 1.12 if fontsize >= 4.5 else 1.08
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

    # Pass 3: Emergency compact down to 2.2 for dense text in small speech bubbles
    if chosen_lines is None:
        fontsize = 3.25
        while fontsize >= 2.2:
            test_lines = _wrap_for_box(shaped_text, font, fontsize, wrap_width, allow_char_break=True)
            if test_lines:
                display_lines = _bidi_display_lines(test_lines)
                test_shape = page.new_shape()
                rem = test_shape.insert_textbox(
                    unrot_text_rect, "\n".join(display_lines), fontname=font_name,
                    fontsize=fontsize, lineheight=1.05, rotate=rot
                )
                if rem >= 0:
                    chosen_lines = display_lines
                    chosen_size = fontsize
                    chosen_lh = 1.05
                    chosen_rem = rem
                    break
            fontsize -= 0.2

    # If text still cannot fit at all, DO NOT erase or clean bubble! Retain source art.
    if chosen_lines is None:
        return False

    # Mathematical vertical centering: chosen_rem is the exact unused vertical room in visual_text_rect
    vert_offset = chosen_rem / 2.0
    visual_centered = pymupdf.Rect(
        visual_text_rect.x0,
        visual_text_rect.y0 + vert_offset,
        visual_text_rect.x1,
        visual_text_rect.y1,
    )
    target_rect = visual_centered * dmat if dmat is not None else visual_centered
    text_content = "\n".join(chosen_lines)

    # Inserter safety: Check rem = shape.insert_textbox(...) before committing erasure; retry unshifted if rem < 0
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
    elif fill_polygon and len(fill_polygon) >= 3:
        unrot_poly = [pymupdf.Point(point[0], point[1]) * dmat for point in fill_polygon] if dmat is not None else [pymupdf.Point(point[0], point[1]) for point in fill_polygon]
        page.draw_polyline(
            unrot_poly,
            color=None,
            fill=fill or (1.0, 1.0, 1.0),
            closePath=True,
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
    shape.insert_textbox(
        target_rect,
        text_content,
        fontname=font_name,
        fontsize=chosen_size,
        lineheight=chosen_lh,
        color=text_color,
        align=pymupdf.TEXT_ALIGN_CENTER,
        rotate=rot,
    )
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
                # Retain source art. Apply each cleaned patch only when its
                # translation fits; unreadable/overflow bubbles keep the source.
                from PIL import Image
                translated.insert_pdf(source, from_page=page_index, to_page=page_index)
                page = translated[-1]
                with Image.open(cleaned_path) as image:
                    cleaned_image = image.convert("RGB")
                overflow = []
                for block in page_blocks:
                    trans_text = str(block.get("translated", "")).strip()
                    orig_text = str(block.get("original", "")).strip()
                    if not trans_text:
                        continue
                    norm_trans = "".join(ch for ch in trans_text.casefold() if ch.isalnum())
                    norm_orig = "".join(ch for ch in orig_text.casefold() if ch.isalnum())
                    is_short_or_numeric = len(orig_text.split()) <= 2 or norm_orig.isdigit()
                    if norm_orig and norm_orig == norm_trans and not block.get("cleaned", False) and not is_short_or_numeric:
                        overflow.append(block)
                        continue
                    rect = pymupdf.Rect(block.get("bbox", [0, 0, 0, 0])) & page.rect
                    if rect.is_empty or rect.width < 5 or rect.height < 5:
                        overflow.append(block)
                        continue

                    patch_bytes = None
                    if block.get("cleaned", False) and src_page.rect.width > 0 and src_page.rect.height > 0:
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
                            patch_buffer = io.BytesIO()
                            patch.save(patch_buffer, format="PNG")
                            patch_bytes = patch_buffer.getvalue()

                    max_size = min(16.0, max(6.5, rect.height * 0.45))
                    raw_bg = block.get("background")
                    if not isinstance(raw_bg, (list, tuple)) or len(raw_bg) < 3:
                        raw_bg = [1.0, 1.0, 1.0]
                    bg_brightness = sum(raw_bg[:3]) / 3.0
                    inferred_text_color = (0.95, 0.95, 0.95) if bg_brightness < 0.45 else (0.06, 0.06, 0.06)

                    # Use the block's actual background color instead of always white
                    if patch_bytes:
                        block_fill = None  # cleaned patch provides its own background
                    else:
                        if bg_brightness < 0.35:
                            block_fill = tuple(max(0.0, min(0.15, float(c))) for c in raw_bg[:3])
                        elif bg_brightness > 0.92:
                            block_fill = (1.0, 1.0, 1.0)
                        else:
                            block_fill = tuple(max(0.0, min(1.0, float(c))) for c in raw_bg[:3])
                    if not _insert_fitted_text(
                        page,
                        rect,
                        trans_text,
                        min_size=5.0,
                        max_size=max_size,
                        fill=block_fill,
                        background_image=patch_bytes,
                        text_color=inferred_text_color,
                    ):
                        overflow.append(block)
                if include_source:
                    transcript_pages.append((page_index + 1, page_blocks))
                elif overflow:
                    valid_ov = [b for b in overflow if _has_renderable_content(str(b.get("translated", "")))]
                    if valid_ov:
                        transcript_pages.append((page_index + 1, valid_ov))
            else:
                # ─── Fallback: Original PDF Vector Overlay ───
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
                        erase_bbox = block.get("text_erase_bbox")
                        if not block.get("overlay_bbox") or not isinstance(polygon, list):
                            overflow.append(block)
                            continue
                        rect = pymupdf.Rect(block["overlay_bbox"]) & page.rect
                        if rect.is_empty or rect.width < 5 or rect.height < 5:
                            overflow.append(block)
                            continue
                        overlay_rect = rect
                        raw_color = block.get("background")
                        if not isinstance(raw_color, (list, tuple)) or len(raw_color) < 3:
                            raw_color = [1.0, 1.0, 1.0]
                        # Support dark (black boxes/blackboards), pure white (bubbles), and gray (screentones)
                        bg_brightness = sum(raw_color[:3]) / 3.0
                        if bg_brightness < 0.35:
                            fill = tuple(max(0.0, min(0.15, float(c))) for c in raw_color[:3])
                        elif bg_brightness > 0.92:
                            fill = (1.0, 1.0, 1.0)
                        else:
                            fill = tuple(max(0.0, min(1.0, float(c))) for c in raw_color[:3])

                        max_size = min(14.0, max(6.0, overlay_rect.height * 0.46))
                        if not _insert_fitted_text(
                            page,
                            overlay_rect,
                            block.get("translated", ""),
                            min_size=4.0,
                            max_size=max_size,
                            fill=fill,
                            fill_polygon=polygon if isinstance(polygon, list) and len(polygon) >= 3 else None,
                            erase_rect=erase_bbox,
                        ):
                            overflow.append(block)
                    except (TypeError, ValueError, IndexError):
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
        translated.close()
        source.close()
    if not tmp_output_path.exists() or tmp_output_path.stat().st_size < 100:
        if tmp_output_path.exists():
            try:
                tmp_output_path.unlink()
            except OSError:
                pass
        raise PdfJobError("Komiks tarjima PDF faylini yaratib bölmadi.")
    os.replace(tmp_output_path, output_path)


def translate_pdf_file(
    input_path: Path,
    output_path: Path,
    page_cache_path: Path,
    title: str,
    source_lang: str,
    target_lang: str,
    output_mode: str,
    translate_text: Callable[[str, str, Optional[str]], Optional[str]],
    progress_callback: Callable[[int, int], None],
    cancel_event: threading.Event,
    ocr_page: Optional[Callable] = None,
    document_mode: str = "book",
    gemini_api_key: Optional[str] = None,
) -> None:
    """Translate native-text and scanned book pages, caching each completed page for restart/resume."""
    try:
        document = pymupdf.open(str(input_path))
    except Exception as exc:
        raise PdfJobError(f"PDFni qayta oçib bölmadi: {exc}") from exc

    with document:
        try:
            cached_pages = json.loads(page_cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached_pages = {}
        if not isinstance(cached_pages, dict):
            cached_pages = {}

        # Reuse the detected OCR language after a local worker restart.
        effective_source_lang = source_lang
        if source_lang == "auto":
            for cached in cached_pages.values():
                if isinstance(cached, dict) and cached.get("detected_source_lang"):
                    effective_source_lang = cached["detected_source_lang"]
                    break

        total_pages = document.page_count
        repeated_chrome = _find_repeated_page_chrome(document) if document_mode == "book" else set()

        cache_lock = threading.Lock()
        completed_pages = sum(1 for i in range(total_pages) if str(i) in cached_pages)
        if completed_pages > 0:
            progress_callback(completed_pages, total_pages)

        abort_event = threading.Event()

        def _is_stopped() -> bool:
            return cancel_event.is_set() or abort_event.is_set()

        # Container has 8 vCPUs dedicated; 4 concurrent workers provides the optimal 4x speedup
        page_workers = max(1, min(int(os.environ.get("MANGA_PDF_PAGE_WORKERS", "4")), max(1, total_pages)))

        _tls = threading.local()
        worker_docs = []
        worker_docs_lock = threading.Lock()

        def _get_worker_doc():
            doc = getattr(_tls, "doc", None)
            if doc is None or doc.is_closed:
                doc = pymupdf.open(str(input_path))
                _tls.doc = doc
                with worker_docs_lock:
                    worker_docs.append(doc)
            return doc

        def _close_worker_doc():
            doc = getattr(_tls, "doc", None)
            if doc is not None and not doc.is_closed:
                try:
                    doc.close()
                except Exception:
                    pass
                _tls.doc = None

        def _close_all_worker_docs():
            _close_worker_doc()
            with worker_docs_lock:
                for doc in worker_docs:
                    if doc is not None and not doc.is_closed:
                        try:
                            doc.close()
                        except Exception:
                            pass
                worker_docs.clear()
            gc.collect()

        def _process_page_worker(page_index: int) -> dict:
            if _is_stopped():
                raise InterruptedError("Vazifa foydalanuvçi tomonidan töxtatildi.")
            key = str(page_index)
            with cache_lock:
                if key in cached_pages:
                    return cached_pages[key]

            # Use thread-local document handle for multi-threaded safety
            thread_doc = _get_worker_doc()
            page = thread_doc[page_index]
            blocks = _read_page_blocks(page)
            page_result = {"blocks": []}
            page_source_lang = source_lang
            should_ocr = document_mode == "comic" or _page_needs_ocr(page, blocks)
            if should_ocr:
                if ocr_page is None:
                    raise PdfJobError("Skan PDF uçun OCR xizmati ulanmagan.")
                page_result["ocr_attempted"] = True
                try:
                    ocr_result = ocr_page(page, effective_source_lang, document_mode, page_index=page_index, job_dir=page_cache_path.parent)
                except TypeError:
                    ocr_result = ocr_page(page, effective_source_lang, document_mode)
                if isinstance(ocr_result, dict):
                    blocks = ocr_result.get("blocks", [])
                    detected_lang = ocr_result.get("detected_source_lang")
                    if "cleaned_image_filename" in ocr_result and ocr_result["cleaned_image_filename"]:
                        page_result["cleaned_image_filename"] = ocr_result["cleaned_image_filename"]
                else:
                    blocks, detected_lang = ocr_result, None
                if not isinstance(blocks, list):
                    raise PdfJobError(f"{page_index + 1}-bet OCR notöğri natija qaytardi.")
                if detected_lang:
                    page_result["detected_source_lang"] = detected_lang
                    page_source_lang = detected_lang
                elif effective_source_lang != "auto":
                    page_source_lang = effective_source_lang
                confidences = [
                    float(block.get("confidence", 1.0))
                    for block in blocks
                    if isinstance(block, dict) and block.get("text")
                ]
                if confidences and sum(confidences) / len(confidences) < 0.45:
                    page_result["ocr_warning"] = "Bu betda OCR işonçi past; tarjimani tekşirib çiqing."
            if source_lang == "auto" and effective_source_lang != "auto":
                if page_source_lang == "auto" or not page_result.get("detected_source_lang"):
                    page_source_lang = effective_source_lang
            if document_mode == "book":
                page_height = float(page.rect.height)
                blocks = [
                    block for block in blocks
                    if isinstance(block, dict) and not _is_page_chrome(block, page_height, repeated_chrome)
                ]
            if document_mode == "comic":
                norm_lang = str(page_source_lang or "").strip().lower()
                if norm_lang in {"en", "eng", "english"}:
                    try:
                        from main import repair_english_word_spacing
                    except ImportError:
                        repair_english_word_spacing = lambda t: t
                else:
                    repair_english_word_spacing = lambda t: t

                source_chunks = [
                    {"text": repair_english_word_spacing(block.get("text", "")), "is_heading": False, **{
                        key: block[key] for key in ("bbox", "overlay_bbox", "text_erase_bbox", "overlay_polygon", "confidence", "background", "cleaned") if key in block
                    }}
                    for block in blocks if isinstance(block, dict) and block.get("text", "").strip()
                ]
            else:
                source_chunks = _split_page_blocks(blocks)

            # Attempt High-Context Batch Translation with Gemini or fallback
            gemini_translations = None
            effective_gemini_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
            if effective_gemini_key and source_chunks:
                try:
                    from gemini_translator import translate_dialogues_with_gemini
                    chunk_texts = [c["text"] for c in source_chunks]

                    # Construct inter-page context and chapter glossary
                    prev_dialogues = []
                    if page_index > 0:
                        with cache_lock:
                            prev_page = cached_pages.get(str(page_index - 1))
                        if prev_page and isinstance(prev_page, dict):
                            prev_blocks = prev_page.get("blocks", [])
                            if isinstance(prev_blocks, list):
                                for b in prev_blocks:
                                    if isinstance(b, dict):
                                        orig = str(b.get("original", "")).strip()
                                        trans = str(b.get("translated", "")).strip()
                                        if orig and trans:
                                            prev_dialogues.append(f"{orig} -> {trans}")
                                        elif orig:
                                            prev_dialogues.append(orig)
                                prev_dialogues = prev_dialogues[-3:]

                    glossary = None
                    with cache_lock:
                        if isinstance(cached_pages, dict) and "glossary" in cached_pages:
                            glossary = copy.deepcopy(cached_pages["glossary"])
                    if glossary is None:
                        glossary_file = page_cache_path.parent / "glossary.json"
                        if glossary_file.is_file():
                            try:
                                glossary = json.loads(glossary_file.read_text(encoding="utf-8"))
                            except Exception:
                                glossary = None

                    context_parts = []
                    if prev_dialogues:
                        context_parts.append("Previous page context:\n" + "\n".join(prev_dialogues))
                    if glossary:
                        if isinstance(glossary, dict):
                            glossary_str = "\n".join(f"{k}: {v}" for k, v in glossary.items())
                            context_parts.append(f"Chapter glossary:\n{glossary_str}")
                        elif isinstance(glossary, list):
                            context_parts.append("Chapter glossary:\n" + "\n".join(str(g) for g in glossary))
                        else:
                            context_parts.append(f"Chapter glossary:\n{glossary}")
                    chapter_context = "\n\n".join(context_parts) if context_parts else None

                    if chapter_context:
                        try:
                            gemini_translations = translate_dialogues_with_gemini(
                                chunk_texts, target_lang, page_source_lang, api_key=effective_gemini_key, chapter_context=chapter_context
                            )
                        except TypeError:
                            gemini_translations = translate_dialogues_with_gemini(
                                chunk_texts, target_lang, page_source_lang, api_key=effective_gemini_key
                            )
                    else:
                        gemini_translations = translate_dialogues_with_gemini(
                            chunk_texts, target_lang, page_source_lang, api_key=effective_gemini_key
                        )
                except Exception as e:
                    logger.debug("Gemini batch tarjima xatosi: %s", e)

            for idx, source_chunk in enumerate(source_chunks):
                if _is_stopped():
                    raise InterruptedError("Vazifa foydalanuvçi tomonidan töxtatildi.")
                translated = None
                if gemini_translations and idx < len(gemini_translations) and gemini_translations[idx]:
                    translated = gemini_translations[idx]
                if not translated:
                    try:
                        accepts_cancel = "cancel_event" in inspect.signature(translate_text).parameters
                    except (TypeError, ValueError):
                        accepts_cancel = False
                    kwargs = {"cancel_event": cancel_event} if accepts_cancel else {}
                    translated = translate_text(source_chunk["text"], target_lang, page_source_lang, **kwargs)
                if _is_stopped():
                    raise InterruptedError("Vazifa foydalanuvçi tomonidan töxtatildi.")
                if not translated:
                    raise PdfJobError(
                        f"{page_index + 1}-bet tarjima bölmadi. Vazifani davom ettirsangiz, şu bet qayta işlanadi."
                    )
                translated_block = {
                    "original": source_chunk["text"],
                    "translated": translated,
                    "is_heading": source_chunk["is_heading"],
                }
                for chunk_key in ("bbox", "overlay_bbox", "text_erase_bbox", "overlay_polygon", "confidence", "background", "cleaned"):
                    if chunk_key in source_chunk:
                        translated_block[chunk_key] = source_chunk[chunk_key]
                page_result["blocks"].append(translated_block)

            with cache_lock:
                cached_pages[key] = page_result
                _atomic_json_write(page_cache_path, cached_pages)
                actual_completed = sum(1 for i in range(total_pages) if str(i) in cached_pages)
                nonlocal completed_pages
                if actual_completed > completed_pages:
                    completed_pages = actual_completed
                cur_completed = completed_pages
            progress_callback(cur_completed, total_pages)
            return page_result

        pages_to_process = [i for i in range(total_pages) if str(i) not in cached_pages]
        if pages_to_process:
            try:
                if page_workers > 1 and len(pages_to_process) > 1:
                    with ThreadPoolExecutor(max_workers=page_workers, thread_name_prefix="pdf-page-worker") as pool:
                        futures = {pool.submit(_process_page_worker, i): i for i in pages_to_process}
                        try:
                            for fut in as_completed(futures):
                                fut.result()
                        except Exception:
                            abort_event.set()
                            for fut in futures:
                                fut.cancel()
                            raise
                        finally:
                            try:
                                list(pool.map(lambda _: _close_worker_doc(), range(page_workers)))
                            except Exception:
                                pass
                else:
                    for i in pages_to_process:
                        _process_page_worker(i)
            finally:
                _close_all_worker_docs()

        translated_pages = [cached_pages[str(i)] for i in range(total_pages)]

        if cancel_event.is_set():
            raise InterruptedError("Vazifa foydalanuvçi tomonidan töxtatildi.")
        if document_mode == "comic":
            _write_comic_pdf(input_path, output_path, translated_pages, output_mode == "bilingual")
        else:
            _write_reflowed_pdf(output_path, title, translated_pages, output_mode)


class PdfJobManager:
    """Single-worker, restartable job queue backed by per-job JSON and PDF files."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdf-translation")
        self._jobs: dict[str, dict] = {}
        self._metadata = self._jobs
        self._events: dict[str, threading.Event] = {}
        self._translate_text = None
        self._ocr_page = None
        self._retention_seconds = max(1, int(os.environ.get("MANGA_PDF_JOB_RETENTION_HOURS", "48"))) * 3600
        self._last_cleanup = 0.0
        self._max_pending_jobs = max(1, int(os.environ.get("MANGA_PDF_MAX_PENDING_JOBS", "8")))
        self._stopping = False
        self._correction_locks: dict[str, threading.RLock] = {}
        self._job_write_locks: dict[str, threading.Lock] = {}
        self._api_keys: dict[str, str] = {}
        self._submitted_jobs: set[str] = set()
        self._load_existing_jobs()
        self._cleanup_expired(force=True)

    def _job_dir(self, job_id: str) -> Path:
        try:
            safe_id = str(uuid.UUID(job_id))
        except ValueError as exc:
            raise PdfJobError("Vazifa raqami notöğri.") from exc
        directory = (self.data_dir / safe_id).resolve()
        if directory.parent != self.data_dir:
            raise PdfJobError("Vazifa manzili notöğri.")
        return directory

    def _metadata_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job.json"

    def _get_write_lock(self, job_id: str) -> threading.Lock:
        with self._lock:
            return self._job_write_locks.setdefault(job_id, threading.Lock())

    def _get_job(self, job_id: str) -> Optional[dict]:
        with self._lock:
            in_memory = self._jobs.get(job_id)
            try:
                metadata_path = self._metadata_path(job_id)
            except PdfJobError:
                return copy.deepcopy(in_memory) if in_memory is not None else None
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if in_memory is None or float(metadata.get("updated_at", 0)) > float(in_memory.get("updated_at", 0)):
                        self._jobs[job_id] = metadata
                        return copy.deepcopy(metadata)
                except Exception as exc:
                    logger.warning("Fayldan job o'qishda xatolik (%s): %s", job_id, exc)
            return copy.deepcopy(in_memory) if in_memory is not None else None

    def get_job(self, job_id: str) -> Optional[dict]:
        """Public accessor for job metadata, returning a deepcopy or None."""
        return self._get_job(job_id)

    def _write_metadata(self, job_id: str) -> None:
        write_lock = self._get_write_lock(job_id)
        with write_lock:
            with self._lock:
                job = self._jobs.get(job_id)
                if not job:
                    return
                job_copy = copy.deepcopy(job)
                metadata_path = self._metadata_path(job_id)
            _atomic_json_write(metadata_path, job_copy)

    def _cleanup_expired(self, force: bool = False) -> None:
        now = time.time()
        with self._lock:
            if not force and time.monotonic() - self._last_cleanup < 300:
                return
            self._last_cleanup = time.monotonic()
            expired = [
                job_id for job_id, job in self._jobs.items()
                if job.get("status") in {"completed", "failed", "canceled"}
                and now - float(job.get("updated_at", now)) > self._retention_seconds
            ]
            for job_id in expired:
                corr_lock = self._correction_locks.get(job_id)
                if corr_lock is not None:
                    if getattr(corr_lock, "_is_owned", lambda: False)():
                        continue
                    if not corr_lock.acquire(blocking=False):
                        continue
                try:
                    job_dir = self._job_dir(job_id)
                    if job_dir.parent == self.data_dir and job_dir.exists():
                        shutil.rmtree(job_dir)
                    self._jobs.pop(job_id, None)
                    self._events.pop(job_id, None)
                    self._api_keys.pop(job_id, None)
                    self._correction_locks.pop(job_id, None)
                    self._job_write_locks.pop(job_id, None)
                    self._submitted_jobs.discard(job_id)
                except Exception as exc:
                    logger.warning("Eski PDF vazifasini öçirib bölmadi (%s): %s", job_id, exc)
                finally:
                    if corr_lock is not None:
                        try:
                            corr_lock.release()
                        except RuntimeError:
                            pass

    def _load_existing_jobs(self) -> None:
        for metadata_path in self.data_dir.glob("*/job.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                job_id = str(uuid.UUID(metadata["job_id"]))
                metadata.setdefault("document_mode", "book")
                if self._job_dir(job_id) != metadata_path.parent.resolve():
                    continue
                if metadata.get("status") == "processing":
                    metadata["status"] = "queued"
                    metadata["message"] = "Server qayta işga tuşdi; vazifa davom ettiriladi."
                    _atomic_json_write(metadata_path, metadata)
                self._jobs[job_id] = metadata
                if metadata.get("status") in {"queued", "processing"}:
                    self._events[job_id] = threading.Event()
            except Exception as exc:
                logger.warning("PDF vazifasi metama'lumotini öqib bölmadi (%s): %s", metadata_path, exc)

    def start_pending(self, translate_text: Callable, ocr_page: Optional[Callable] = None):
        self._translate_text = translate_text
        self._ocr_page = ocr_page
        with self._lock:
            pending_ids = [
                job_id for job_id, job in self._jobs.items()
                if job.get("status") == "queued" and job_id not in self._submitted_jobs
            ]
            for job_id in pending_ids:
                self._submitted_jobs.add(job_id)
                self._executor.submit(self._run_job, job_id)

    def create_job(
        self,
        pdf_bytes: bytes,
        filename: str,
        source_lang: str,
        target_lang: str,
        output_mode: str,
        page_count: int,
        document_mode: str = "book",
        gemini_api_key: Optional[str] = None,
    ) -> dict:
        self._cleanup_expired()
        with self._lock:
            self._check_queue_capacity()
        return self._create_job_locked(pdf_bytes, filename, source_lang, target_lang, output_mode, page_count, document_mode, gemini_api_key)

    def _check_queue_capacity(self):
        if sum(job.get("status") in {"queued", "processing"} for job in self._jobs.values()) >= self._max_pending_jobs:
            raise PdfQueueFullError("PDF navbati tölgan. Birozdan söng qayta urinib köring.")

    def _create_job_locked(self, pdf_bytes, filename, source_lang, target_lang, output_mode, page_count, document_mode, gemini_api_key=None):
        job_id = str(uuid.uuid4())
        job_token = secrets.token_urlsafe(32)
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)

        # Write input.pdf outside self._lock to avoid blocking concurrent requests on slow I/O
        try:
            (job_dir / "input.pdf").write_bytes(pdf_bytes)
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

        safe_name = Path((filename or "book.pdf").replace("\\", "/")).name
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"
        metadata = {
            "job_id": job_id,
            "filename": safe_name,
            "status": "queued",
            "progress": 0,
            "current_page": 0,
            "total_pages": page_count,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "output_mode": output_mode,
            "document_mode": document_mode,
            "job_token_sha256": hashlib.sha256(job_token.encode("utf-8")).hexdigest(),
            "message": "Navbatda",
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

        should_submit = False
        with self._lock:
            self._check_queue_capacity()
            if gemini_api_key:
                self._api_keys[job_id] = str(gemini_api_key).strip()
            self._jobs[job_id] = metadata
            self._events[job_id] = threading.Event()
            if self._translate_text is not None:
                self._submitted_jobs.add(job_id)
                should_submit = True
            else:
                logger.error("Job %s yaratildi, lekin _translate_text None! Worker ishga tushmadi!", job_id)

        self._write_metadata(job_id)
        if should_submit:
            self._executor.submit(self._run_job, job_id)

        return {**self.public_job(job_id), "job_token": job_token}

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.update(changes)
            job["updated_at"] = time.time()
        self._write_metadata(job_id)

    def _run_job(self, job_id: str) -> None:
        try:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job.get("status") != "queued":
                    return
                cancel_event = self._events.setdefault(job_id, threading.Event())
                self._update(job_id, status="processing", message="PDF sahifalari tarjima qilinmoqda…", error=None)
                snapshot = dict(self._jobs[job_id])

            job_dir = self._job_dir(job_id)
            last_reported_page = 0

            def report_progress(current_page, total_pages):
                nonlocal last_reported_page
                with self._lock:
                    if current_page < last_reported_page:
                        return
                    last_reported_page = current_page
                progress = min(95, int(current_page * 95 / max(1, total_pages)))
                self._update(
                    job_id,
                    current_page=current_page,
                    total_pages=total_pages,
                    progress=progress,
                    message=f"{current_page}/{total_pages}-bet tarjima qilindi",
                )

            try:
                translate_pdf_file(
                    input_path=job_dir / "input.pdf",
                    output_path=job_dir / "translated.pdf",
                    page_cache_path=job_dir / "page_translations.json",
                    title=Path(snapshot["filename"]).stem,
                    source_lang=snapshot["source_lang"],
                    target_lang=snapshot["target_lang"],
                    output_mode=snapshot["output_mode"],
                    translate_text=self._translate_text,
                    ocr_page=self._ocr_page,
                    progress_callback=report_progress,
                    cancel_event=cancel_event,
                    document_mode=snapshot.get("document_mode", "book"),
                    gemini_api_key=self._api_keys.get(job_id),
                )
                if cancel_event.is_set():
                    self._update(job_id, status="queued" if self._stopping else "canceled", message="Vazifa töxtatildi.")
                else:
                    self._update(job_id, status="completed", progress=100, message="Tarjima PDF tayyor.", error=None)
            except InterruptedError as exc:
                self._update(job_id, status="queued" if self._stopping else "canceled", message=str(exc))
            except Exception as exc:
                logger.exception("PDF vazifasi bajarilmadi (%s)", job_id)
                self._update(job_id, status="failed", message="Tarjima tugamadi.", error=str(exc))
        finally:
            with self._lock:
                self._submitted_jobs.discard(job_id)

    def public_job(self, job_id: str) -> dict:
        self._cleanup_expired()
        job = self._get_job(job_id)
        if job is None:
            raise PdfJobError("PDF vazifasi topilmadi.")
        res = {
            key: job[key]
            for key in (
                "job_id", "filename", "status", "progress", "current_page", "total_pages",
                "source_lang", "target_lang", "output_mode", "message", "error",
                "document_mode",
                "created_at", "updated_at",
            )
            if key in job
        }
        if "warning" in job and job["warning"]:
            res["warning"] = job["warning"]
        return res

    def review_page(self, job_id: str, page_index: int) -> dict:
        """Return editable OCR and translation text for one completed PDF page."""
        job = self.public_job(job_id)
        if job["status"] != "completed":
            raise PdfJobError("Tahrir oynasi tarjima tugagandan keyin oçiladi.")
        if page_index < 0 or page_index >= int(job["total_pages"]):
            raise PdfJobError("Bet raqami PDF çegarasidan taşqarida.")
        job_dir = self._job_dir(job_id)
        try:
            cached_pages = json.loads((job_dir / "page_translations.json").read_text(encoding="utf-8"))
            page_data = cached_pages[str(page_index)]
            blocks = page_data.get("blocks", [])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise PdfJobError("Bet tarjima ma’lumoti topilmadi.") from exc
        return {
            "page_index": page_index,
            "total_pages": int(job["total_pages"]),
            "document_mode": job.get("document_mode", "book"),
            "warning": page_data.get("ocr_warning"),
            "blocks": [
                {
                    "block_index": index,
                    "original": str(block.get("original", "")),
                    "translated": str(block.get("translated", "")),
                    "confidence": block.get("confidence"),
                }
                for index, block in enumerate(blocks)
                if isinstance(block, dict)
            ],
        }

    def apply_corrections(self, job_id: str, corrections: list[dict]) -> dict:
        """Update OCR/translation text and atomically rebuild a completed PDF."""
        with self._lock:
            correction_lock = self._correction_locks.setdefault(job_id, threading.RLock())
        with correction_lock:
            return self._apply_corrections_locked(job_id, corrections)

    def _apply_corrections_locked(self, job_id: str, corrections: list[dict]) -> dict:
        if not isinstance(corrections, list) or not corrections or len(corrections) > 100:
            raise PdfJobError("Bir sörovda 1 dan 100 tagaça tuzatiş yuboring.")
        job = self.public_job(job_id)
        if job["status"] != "completed":
            raise PdfJobError("Faqat tayyor PDF tarjimasini tahrirlaş mumkin.")
        job_dir = self._job_dir(job_id)
        cache_path = job_dir / "page_translations.json"
        updated_cache_path = job_dir / "page_translations.updated.json"
        output_path = job_dir / "translated.pdf"
        updated_output_path = job_dir / "translated.updated.pdf"
        try:
            cached_pages = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PdfJobError("Bet tarjima ma’lumoti topilmadi.") from exc

        seen = set()
        for correction in corrections:
            if not isinstance(correction, dict):
                raise PdfJobError("Tuzatiş şakli notöğri.")
            page_index, block_index = correction.get("page_index"), correction.get("block_index")
            if (not isinstance(page_index, int) or isinstance(page_index, bool)
                    or not isinstance(block_index, int) or isinstance(block_index, bool)):
                raise PdfJobError("Bet va matn raqami butun son bölişi kerak.")
            identity = (page_index, block_index)
            if identity in seen:
                raise PdfJobError("Bir matn bölagi ikki marta yuborildi.")
            seen.add(identity)
            page_data = cached_pages.get(str(page_index))
            blocks = page_data.get("blocks", []) if isinstance(page_data, dict) else []
            if page_index < 0 or page_index >= int(job["total_pages"]) or block_index < 0 or block_index >= len(blocks):
                raise PdfJobError("Tuzatiş körsatgan bet yoki matn topilmadi.")
            block = blocks[block_index]
            orig_changed = False
            for field in ("original", "translated"):
                if field not in correction:
                    continue
                value = correction[field]
                if not isinstance(value, str) or len(value) > 5000:
                    raise PdfJobError("Matn 5000 belgidan oşmasligi va matn körinişida bölişi kerak.")
                value = value.strip()
                if field == "translated" and not value:
                    raise PdfJobError("Tarjima böş bölişi mumkin emas.")
                if field == "original" and value != block.get("original"):
                    orig_changed = True
                block[field] = value

            # If original was corrected and translated was not explicitly set, re-translate
            if orig_changed and "translated" not in correction and self._translate_text is not None:
                try:
                    re_trans = self._translate_text(block["original"], job["target_lang"], job["source_lang"])
                    if re_trans:
                        block["translated"] = re_trans
                except Exception as e:
                    logger.debug("Tuzatilgan asl matnni qayta tarjima qilişda xato: %s", e)

        _atomic_json_write(updated_cache_path, cached_pages)
        try:
            translate_pdf_file(
                input_path=job_dir / "input.pdf",
                output_path=updated_output_path,
                page_cache_path=updated_cache_path,
                title=Path(job["filename"]).stem,
                source_lang=job["source_lang"],
                target_lang=job["target_lang"],
                output_mode=job["output_mode"],
                translate_text=self._translate_text or (lambda *_args: None),
                ocr_page=self._ocr_page,
                progress_callback=lambda *_args: None,
                cancel_event=threading.Event(),
                document_mode=job.get("document_mode", "book"),
                gemini_api_key=self._api_keys.get(job_id),
            )
            os.replace(updated_output_path, output_path)
            os.replace(updated_cache_path, cache_path)
        except Exception as exc:
            raise PdfJobError(f"Tuzatilgan PDFni qayta yaratib bölmadi: {exc}") from exc
        finally:
            updated_output_path.unlink(missing_ok=True)
            updated_cache_path.unlink(missing_ok=True)

        self._update(job_id, message="Tuzatişlar PDFga saqlandi.")
        return self.public_job(job_id)

    def list_jobs(self, limit: int = 20) -> list[dict]:
        self._cleanup_expired()
        with self._lock:
            job_ids = sorted(
                self._jobs,
                key=lambda item: float(self._jobs[item].get("created_at", 0)),
                reverse=True,
            )[:max(1, min(limit, 50))]
        return [self.public_job(job_id) for job_id in job_ids]

    def output_path(self, job_id: str) -> Path:
        job = self._get_job(job_id)
        if job is None:
            raise PdfJobError("PDF vazifasi topilmadi.")
        job_dir = self._job_dir(job_id)
        output_path = (job_dir / "translated.pdf").resolve()
        if output_path.parent != job_dir:
            raise PdfJobError("Natija manzili notöğri.")
        if job.get("status") != "completed" or not output_path.is_file():
            raise PdfJobError("Tarjima PDF hali tayyor emas.")
        return output_path

    def verify_job_token(self, job_id: str, token: Optional[str]) -> bool:
        job = self._get_job(job_id)
        if job is None or not token:
            return False
        expected = job.get("job_token_sha256", "")
        actual = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return bool(expected) and hmac.compare_digest(actual, expected)

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            self._get_job(job_id)
            job = self._jobs.get(job_id)
            if job is None:
                raise PdfJobError("PDF vazifasi topilmadi.")
            if job.get("status") == "queued":
                self._events.setdefault(job_id, threading.Event()).set()
                job.update(status="canceled", message="Vazifa bekor qilindi.", updated_at=time.time())
            elif job.get("status") == "processing":
                self._events.setdefault(job_id, threading.Event()).set()
                job.update(message="Töxtatiş söraldi; joriy bet tugagaç töxtaydi.", updated_at=time.time())
        self._write_metadata(job_id)
        return self.public_job(job_id)

    def resume(self, job_id: str, gemini_api_key: Optional[str] = None) -> dict:
        should_submit = False
        with self._lock:
            self._get_job(job_id)
            job = self._jobs.get(job_id)
            if job is None:
                raise PdfJobError("PDF vazifasi topilmadi.")
            if job.get("status") not in {"failed", "canceled"}:
                raise PdfJobError("Faqat xato bergan yoki töxtatilgan vazifani davom ettiriş mumkin.")
            self._check_queue_capacity()

            if gemini_api_key and str(gemini_api_key).strip():
                self._api_keys[job_id] = str(gemini_api_key).strip()
                job.pop("warning", None)
            elif not self._api_keys.get(job_id):
                env_key = os.environ.get("GEMINI_API_KEY")
                if env_key and env_key.strip():
                    self._api_keys[job_id] = env_key.strip()
                    job.pop("warning", None)
                else:
                    warning_msg = "Gemini API kaliti topilmadi; tarjima cheklangan rejimda davom ettiriladi."
                    logger.warning("PDF vazifasini davom ettirishda GEMINI_API_KEY topilmadi (%s)", job_id)
                    job["warning"] = warning_msg

            job.update(status="queued", message="Qayta navbatga qöyildi.", error=None, updated_at=time.time())
            self._events[job_id] = threading.Event()

            if self._translate_text is not None and job_id not in self._submitted_jobs:
                self._submitted_jobs.add(job_id)
                should_submit = True

        self._write_metadata(job_id)
        if should_submit:
            self._executor.submit(self._run_job, job_id)

        res = self.public_job(job_id)
        if job.get("warning"):
            res["warning"] = job["warning"]
        return res

    def shutdown(self) -> None:
        with self._lock:
            self._stopping = True
            for event in self._events.values():
                event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
