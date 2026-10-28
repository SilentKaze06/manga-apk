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
from concurrent.futures import ThreadPoolExecutor
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
    COMIC_FONT_BUFFER = pymupdf.Font("notos").buffer
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


def _get_font_and_shaped_text(text: str) -> tuple[str, bytes, str]:
    """Select font and shape text for CJK, Arabic, or Latin/Cyrillic scripts."""
    text_str = str(text)
    if _ARABIC_CODE_PATTERN.search(text_str):
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            shaped = get_display(arabic_reshaper.reshape(text_str))
        except Exception:
            shaped = text_str
        return "ScreenTranslatorFigo", COMIC_FIGO_FONT_BUFFER, shaped
    elif _CJK_CODE_PATTERN.search(text_str):
        return "ScreenTranslatorCJK", COMIC_CJK_FONT_BUFFER, text_str
    return COMIC_FONT_NAME, COMIC_FONT_BUFFER, text_str


PAGE_NUMBER_PATTERN = re.compile(r"^(?:page\s*)?(?:\d{1,5}|[ivxlcdm]{1,8})$", re.IGNORECASE)


class PdfJobError(Exception):
    """An expected PDF validation or translation error suitable for the UI."""


class PdfQueueFullError(PdfJobError):
    """The bounded local worker queue cannot accept another job yet."""


def _atomic_json_write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


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
        sizes = []
        for line in block.get("lines", []):
            line_text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            if line_text:
                lines.append(line_text)
            sizes.extend(float(span.get("size", 0)) for span in line.get("spans", []))
        text = " ".join(lines)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            blocks.append({
                "text": text,
                "font_size": max(sizes, default=0.0),
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
    if not text_length:
        return True
    return text_length < 120 and _image_coverage(page) >= 0.35


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
                raise PdfJobError(f"PDF {max_pages} betdan oshmasligi kerak.")
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
        raise PdfJobError(f"PDFni ochib bo'lmadi: {exc}") from exc


def _split_page_blocks(blocks: list[dict]) -> list[dict]:
    """Keep paragraphs together when possible and cap provider request size."""
    chunks = []
    current = []
    current_size = 0
    current_heading = None
    for block in blocks:
        text = block["text"]
        is_heading = block.get("font_size", 0) >= 15 and len(text) < 180
        if current and current_heading != is_heading:
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
            translated = html.escape(block.get("translated", "")).replace("\n", "<br>")
            original = html.escape(block.get("original", "")).replace("\n", "<br>")
            if not translated:
                continue
            is_heading = block.get("is_heading", False)
            tag = "h2" if is_heading else "p"
            pieces.append(f"<{tag}>{translated}</{tag}>")
            if mode == "bilingual":
                pieces.append(f"<p class='source'>{original}</p>")
        if page.get("ocr_warning"):
            pieces.append(f"<p class='source'>{html.escape(page['ocr_warning'])}</p>")
        pieces.append(f"<p class='source-page'>Manba PDF: {page_number}-bet</p>")
    pieces.append("</body></html>")
    return "".join(pieces)


def _write_reflowed_pdf(output_path: Path, title: str, pages: list[dict], mode: str) -> None:
    font_archive = pymupdf.Archive()
    font_css = pymupdf.css_for_pymupdf_font("notos", name="sans-serif", archive=font_archive)
    story = pymupdf.Story(
        _translation_html(title, pages, mode),
        user_css=font_css,
        archive=font_archive,
    )
    writer = pymupdf.DocumentWriter(str(output_path))
    try:
        more = True
        while more:
            device = writer.begin_page(PAGE_RECT)
            more, _ = story.place(TEXT_RECT)
            story.draw(device)
            writer.end_page()
    finally:
        writer.close()
    # PyMuPDF's Windows file handles are released when the writer/device
    # wrappers are collected, after close() has flushed the PDF.
    if "device" in locals():
        del device
    del writer
    gc.collect()
    if not output_path.exists() or output_path.stat().st_size < 100:
        raise PdfJobError("Tarjima PDF faylini yaratib bo'lmadi.")
    with pymupdf.open(output_path) as translated:
        translated.set_metadata({"title": title or "Tarjima qilingan PDF", "producer": "ScreenTranslator"})
        translated.saveIncr()
    del translated
    gc.collect()


def _wrap_for_box(text: str, font: pymupdf.Font, fontsize: float, width: float) -> list[str]:
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
                part = ""
                for character in word:
                    if part and font.text_length(part + character, fontsize=fontsize) > width:
                        lines.append(part)
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
) -> bool:
    rect = pymupdf.Rect(rect)
    if fill_polygon:
        inset_x = min(rect.width * 0.16, max(2.0, rect.width * 0.06))
        inset_y = min(rect.height * 0.16, max(2.0, rect.height * 0.08))
    else:
        inset_x = inset_y = min(2.0, max(0.5, rect.width * 0.025), max(0.5, rect.height * 0.025))
    text_rect = rect + (inset_x, inset_y, -inset_x, -inset_y)
    if text_rect.is_empty or text_rect.width < 4 or text_rect.height < 4:
        return False

    font_name, font_buffer, shaped_text = _get_font_and_shaped_text(text)
    font = pymupdf.Font(fontbuffer=font_buffer)
    fontsize = max_size
    rot = int(getattr(page, "rotation", 0)) % 360
    dmat = page.derotation_matrix if rot else None
    unrot_text_rect = text_rect * dmat if dmat is not None else text_rect
    unrot_rect = rect * dmat if dmat is not None else rect

    while fontsize >= min_size:
        lines = _wrap_for_box(shaped_text, font, fontsize, text_rect.width)
        line_height = fontsize * 1.18
        if len(lines) * line_height <= text_rect.height:
            page.insert_font(fontname=font_name, fontbuffer=font_buffer)
            shape = page.new_shape()
            remaining = shape.insert_textbox(
                unrot_text_rect, "\n".join(lines), fontname=font_name,
                fontsize=fontsize, lineheight=1.18, color=(0.06, 0.06, 0.06),
                align=pymupdf.TEXT_ALIGN_CENTER,
                rotate=rot,
            )
            if remaining < 0:
                fontsize -= 0.5
                continue
            # Only erase original artwork after text layout has succeeded.
            if background_image is not None:
                page.insert_image(
                    unrot_rect,
                    stream=background_image,
                    rotate=rot,
                    keep_proportion=False,
                    overlay=True,
                )
            if fill_polygon and len(fill_polygon) >= 3:
                unrot_poly = [pymupdf.Point(point[0], point[1]) * dmat for point in fill_polygon] if dmat is not None else [pymupdf.Point(point[0], point[1]) for point in fill_polygon]
                page.draw_polyline(
                    unrot_poly,
                    color=None,
                    fill=fill,
                    closePath=True,
                    overlay=True,
                )
            elif fill is not None:
                page.draw_rect(unrot_rect, color=None, fill=fill, overlay=True)
            shape.commit(overlay=True)
            return True
        fontsize -= 0.5
    return False


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
        new_page.insert_font(fontname=h_name, fontbuffer=h_buf)
        new_page.insert_text((margin, margin + 18), h_shaped,
                             fontname=h_name, fontsize=16, color=(0.05, 0.05, 0.05))
        return new_page, margin + 48

    for index, block in enumerate(blocks, 1):
        original = str(block.get("original", "")).strip()
        translated = str(block.get("translated", "")).strip()
        paragraphs = ([original] if include_source and original else []) + [translated]
        for paragraph in paragraphs:
            f_name, f_buf, shaped_para = _get_font_and_shaped_text(paragraph)
            font = pymupdf.Font(fontbuffer=f_buf)
            for line in _wrap_for_box(f"{index}. {shaped_para}", font, 10, page_rect.width - 2 * margin):
                if page is None or y + 3 > page_rect.height - margin:
                    page, y = add_page()
                if line:
                    page.insert_font(fontname=f_name, fontbuffer=f_buf)
                    page.insert_text((margin, y), line, fontname=f_name, fontsize=10)
                y += 14
            y += 8


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
        raise PdfJobError(f"Komiks PDFni qayta ochib bo'lmadi: {exc}") from exc

    translated = pymupdf.open()
    transcript_pages = []
    job_dir = input_path.parent
    try:
        for page_index, page_data in enumerate(pages):
            if page_index >= source.page_count:
                break
            src_page = source[page_index]
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
                    if not trans_text:
                        continue
                    if not block.get("cleaned", False):
                        overflow.append(block)
                        continue
                    rect = pymupdf.Rect(block.get("bbox", [0, 0, 0, 0])) & page.rect
                    if rect.is_empty or rect.width < 5 or rect.height < 5:
                        overflow.append(block)
                        continue
                    scale_x = cleaned_image.width / src_page.rect.width
                    scale_y = cleaned_image.height / src_page.rect.height
                    patch = cleaned_image.crop((round(rect.x0 * scale_x), round(rect.y0 * scale_y),
                                                round(rect.x1 * scale_x), round(rect.y1 * scale_y)))
                    patch_buffer = io.BytesIO()
                    patch.save(patch_buffer, format="PNG")
                    max_size = min(16.0, max(6.5, rect.height * 0.45))
                    if not _insert_fitted_text(
                        page,
                        rect,
                        trans_text,
                        min_size=5.0,
                        max_size=max_size,
                        fill=None,
                        background_image=patch_buffer.getvalue(),
                    ):
                        overflow.append(block)
                if include_source:
                    transcript_pages.append((page_index + 1, page_blocks))
                elif overflow:
                    transcript_pages.append((page_index + 1, overflow))
            else:
                # ─── Fallback: Original PDF Vector Overlay ───
                translated.insert_pdf(source, from_page=page_index, to_page=page_index)
                page = translated[-1]
                overflow = []
                for block in page_blocks:
                    try:
                        polygon = block.get("overlay_polygon")
                        if not block.get("overlay_bbox") or not isinstance(polygon, list) or len(polygon) < 3:
                            overflow.append(block)
                            continue
                        rect = pymupdf.Rect(block["overlay_bbox"]) & page.rect
                        if rect.is_empty or rect.width < 5 or rect.height < 5:
                            overflow.append(block)
                            continue
                        pad_x = min(5.0, max(1.5, rect.width * 0.08))
                        pad_y = min(4.0, max(1.0, rect.height * 0.08))
                        overlay_rect = (rect + (-pad_x, -pad_y, pad_x, pad_y)) & page.rect
                        raw_color = block.get("background", [1.0, 1.0, 1.0])
                        fill = tuple(max(0.72, min(1.0, float(channel))) for channel in raw_color[:3])
                        max_size = min(12.0, max(6.0, overlay_rect.height * 0.46))
                        if not _insert_fitted_text(
                            page,
                            overlay_rect,
                            block.get("translated", ""),
                            min_size=5.0,
                            max_size=max_size,
                            fill=fill,
                            fill_polygon=polygon,
                        ):
                            overflow.append(block)
                    except (TypeError, ValueError, IndexError):
                        overflow.append(block)

                if include_source:
                    transcript_pages.append((page_index + 1, page_blocks))
                elif overflow:
                    transcript_pages.append((page_index + 1, overflow))
                elif not page_blocks and page_data.get("ocr_attempted"):
                    transcript_pages.append((page_index + 1, [{"original": "", "translated": "Bu betda matn aniqlanmadi."}]))

        for source_page, blocks in transcript_pages:
            _comic_pdf_transcript_page(translated, source_page, blocks, include_source)

        translated.set_metadata({"title": f"{input_path.stem} — tarjima qilingan komiks", "producer": "ScreenTranslator"})
        translated.save(str(output_path), garbage=4, deflate=True)
    finally:
        translated.close()
        source.close()
    if not output_path.exists() or output_path.stat().st_size < 100:
        raise PdfJobError("Komiks tarjima PDF faylini yaratib bo'lmadi.")


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
) -> None:
    """Translate native-text and scanned book pages, caching each completed page for restart/resume."""
    try:
        document = pymupdf.open(str(input_path))
    except Exception as exc:
        raise PdfJobError(f"PDFni qayta ochib bo'lmadi: {exc}") from exc

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

        translated_pages = []
        total_pages = document.page_count
        repeated_chrome = _find_repeated_page_chrome(document) if document_mode == "book" else set()
        for page_index, page in enumerate(document):
            if cancel_event.is_set():
                raise InterruptedError("Vazifa foydalanuvchi tomonidan to'xtatildi.")
            key = str(page_index)
            if key in cached_pages:
                page_result = cached_pages[key]
            else:
                blocks = _read_page_blocks(page)
                page_result = {"blocks": []}
                page_source_lang = source_lang
                should_ocr = document_mode == "comic" or _page_needs_ocr(page, blocks)
                if should_ocr:
                    if ocr_page is None:
                        raise PdfJobError("Skan PDF uchun OCR xizmati ulanmagan.")
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
                        raise PdfJobError(f"{page_index + 1}-bet OCR noto'g'ri natija qaytardi.")
                    if detected_lang:
                        page_result["detected_source_lang"] = detected_lang
                        if source_lang == "auto":
                            effective_source_lang = detected_lang
                            page_source_lang = detected_lang
                    elif effective_source_lang != "auto":
                        page_source_lang = effective_source_lang
                    confidences = [
                        float(block.get("confidence", 1.0))
                        for block in blocks
                        if isinstance(block, dict) and block.get("text")
                    ]
                    if confidences and sum(confidences) / len(confidences) < 0.45:
                        page_result["ocr_warning"] = "Bu betda OCR ishonchi past; tarjimani tekshirib chiqing."
                elif source_lang == "auto" and effective_source_lang != "auto":
                    page_source_lang = effective_source_lang
                if document_mode == "book":
                    page_height = float(page.rect.height)
                    blocks = [
                        block for block in blocks
                        if isinstance(block, dict) and not _is_page_chrome(block, page_height, repeated_chrome)
                    ]
                if document_mode == "comic":
                    source_chunks = [
                        {"text": block.get("text", ""), "is_heading": False, **{
                            key: block[key] for key in ("bbox", "overlay_bbox", "overlay_polygon", "confidence", "background", "cleaned") if key in block
                        }}
                        for block in blocks if isinstance(block, dict) and block.get("text", "").strip()
                    ]
                else:
                    source_chunks = _split_page_blocks(blocks)
                for source_chunk in source_chunks:
                    if cancel_event.is_set():
                        raise InterruptedError("Vazifa foydalanuvchi tomonidan to'xtatildi.")
                    try:
                        accepts_cancel = "cancel_event" in inspect.signature(translate_text).parameters
                    except (TypeError, ValueError):
                        accepts_cancel = False
                    kwargs = {"cancel_event": cancel_event} if accepts_cancel else {}
                    translated = translate_text(source_chunk["text"], target_lang, page_source_lang, **kwargs)
                    if cancel_event.is_set():
                        raise InterruptedError("Vazifa foydalanuvchi tomonidan to'xtatildi.")
                    if not translated:
                        raise PdfJobError(
                            f"{page_index + 1}-bet tarjima bo'lmadi. Vazifani davom ettirsangiz, shu bet qayta ishlanadi."
                        )
                    translated_block = {
                        "original": source_chunk["text"],
                        "translated": translated,
                        "is_heading": source_chunk["is_heading"],
                    }
                    for chunk_key in ("bbox", "overlay_bbox", "overlay_polygon", "confidence", "background", "cleaned"):
                        if chunk_key in source_chunk:
                            translated_block[chunk_key] = source_chunk[chunk_key]
                    page_result["blocks"].append(translated_block)
                cached_pages[key] = page_result
                _atomic_json_write(page_cache_path, cached_pages)
            translated_pages.append(page_result)
            progress_callback(page_index + 1, total_pages)

        if cancel_event.is_set():
            raise InterruptedError("Vazifa foydalanuvchi tomonidan to'xtatildi.")
        if document_mode == "comic":
            _write_comic_pdf(input_path, output_path, translated_pages, output_mode == "bilingual")
        else:
            _write_reflowed_pdf(output_path, title, translated_pages, output_mode)


class PdfJobManager:
    """Single-worker, restartable job queue backed by per-job JSON and PDF files."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir.resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdf-translation")
        self._jobs: dict[str, dict] = {}
        self._events: dict[str, threading.Event] = {}
        self._translate_text = None
        self._ocr_page = None
        self._retention_seconds = max(1, int(os.environ.get("MANGA_PDF_JOB_RETENTION_HOURS", "48"))) * 3600
        self._last_cleanup = 0.0
        self._max_pending_jobs = max(1, int(os.environ.get("MANGA_PDF_MAX_PENDING_JOBS", "8")))
        self._stopping = False
        self._correction_locks: dict[str, threading.RLock] = {}
        self._load_existing_jobs()
        self._cleanup_expired(force=True)

    def _job_dir(self, job_id: str) -> Path:
        try:
            safe_id = str(uuid.UUID(job_id))
        except ValueError as exc:
            raise PdfJobError("Vazifa raqami noto'g'ri.") from exc
        directory = (self.data_dir / safe_id).resolve()
        if directory.parent != self.data_dir:
            raise PdfJobError("Vazifa manzili noto'g'ri.")
        return directory

    def _metadata_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job.json"

    def _write_metadata(self, job_id: str) -> None:
        _atomic_json_write(self._metadata_path(job_id), self._jobs[job_id])

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
                try:
                    job_dir = self._job_dir(job_id)
                    if job_dir.parent == self.data_dir and job_dir.exists():
                        shutil.rmtree(job_dir)
                    self._jobs.pop(job_id, None)
                    self._events.pop(job_id, None)
                except Exception as exc:
                    logger.warning("Eski PDF vazifasini o'chirib bo'lmadi (%s): %s", job_id, exc)

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
                    metadata["message"] = "Server qayta ishga tushdi; vazifa davom ettiriladi."
                    _atomic_json_write(metadata_path, metadata)
                self._jobs[job_id] = metadata
                if metadata.get("status") in {"queued", "processing"}:
                    self._events[job_id] = threading.Event()
            except Exception as exc:
                logger.warning("PDF vazifasi metama'lumotini o'qib bo'lmadi (%s): %s", metadata_path, exc)

    def start_pending(self, translate_text: Callable, ocr_page: Optional[Callable] = None):
        self._translate_text = translate_text
        self._ocr_page = ocr_page
        with self._lock:
            pending_ids = [job_id for job_id, job in self._jobs.items() if job.get("status") == "queued"]
        for job_id in pending_ids:
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
    ) -> dict:
        self._cleanup_expired()
        with self._lock:
            self._check_queue_capacity()
            return self._create_job_locked(pdf_bytes, filename, source_lang, target_lang, output_mode, page_count, document_mode)

    def _check_queue_capacity(self):
        if sum(job.get("status") in {"queued", "processing"} for job in self._jobs.values()) >= self._max_pending_jobs:
            raise PdfQueueFullError("PDF navbati to'lgan. Birozdan so'ng qayta urinib ko'ring.")

    def _create_job_locked(self, pdf_bytes, filename, source_lang, target_lang, output_mode, page_count, document_mode):
        job_id = str(uuid.uuid4())
        job_token = secrets.token_urlsafe(32)
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)
        (job_dir / "input.pdf").write_bytes(pdf_bytes)
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
        with self._lock:
            self._jobs[job_id] = metadata
            self._events[job_id] = threading.Event()
            self._write_metadata(job_id)
        if self._translate_text is not None:
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
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.get("status") != "queued":
                return
            cancel_event = self._events.setdefault(job_id, threading.Event())
            self._update(job_id, status="processing", message="PDF sahifalari tarjima qilinmoqda…", error=None)
            snapshot = dict(self._jobs[job_id])

        job_dir = self._job_dir(job_id)

        def report_progress(current_page, total_pages):
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
            )
            if cancel_event.is_set():
                self._update(job_id, status="queued" if self._stopping else "canceled", message="Vazifa to'xtatildi.")
            else:
                self._update(job_id, status="completed", progress=100, message="Tarjima PDF tayyor.", error=None)
        except InterruptedError as exc:
            self._update(job_id, status="queued" if self._stopping else "canceled", message=str(exc))
        except Exception as exc:
            logger.exception("PDF vazifasi bajarilmadi (%s)", job_id)
            self._update(job_id, status="failed", message="Tarjima tugamadi.", error=str(exc))

    def public_job(self, job_id: str) -> dict:
        self._cleanup_expired()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise PdfJobError("PDF vazifasi topilmadi.")
            return {
                key: job[key]
                for key in (
                    "job_id", "filename", "status", "progress", "current_page", "total_pages",
                    "source_lang", "target_lang", "output_mode", "message", "error",
                    "document_mode",
                    "created_at", "updated_at",
                )
            }

    def review_page(self, job_id: str, page_index: int) -> dict:
        """Return editable OCR and translation text for one completed PDF page."""
        job = self.public_job(job_id)
        if job["status"] != "completed":
            raise PdfJobError("Tahrir oynasi tarjima tugagandan keyin ochiladi.")
        if page_index < 0 or page_index >= int(job["total_pages"]):
            raise PdfJobError("Bet raqami PDF chegarasidan tashqarida.")
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
            raise PdfJobError("Bir so‘rovda 1 dan 100 tagacha tuzatish yuboring.")
        job = self.public_job(job_id)
        if job["status"] != "completed":
            raise PdfJobError("Faqat tayyor PDF tarjimasini tahrirlash mumkin.")
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
                raise PdfJobError("Tuzatish shakli noto‘g‘ri.")
            page_index, block_index = correction.get("page_index"), correction.get("block_index")
            if (not isinstance(page_index, int) or isinstance(page_index, bool)
                    or not isinstance(block_index, int) or isinstance(block_index, bool)):
                raise PdfJobError("Bet va matn raqami butun son bo‘lishi kerak.")
            identity = (page_index, block_index)
            if identity in seen:
                raise PdfJobError("Bir matn bo‘lagi ikki marta yuborildi.")
            seen.add(identity)
            page_data = cached_pages.get(str(page_index))
            blocks = page_data.get("blocks", []) if isinstance(page_data, dict) else []
            if page_index < 0 or page_index >= int(job["total_pages"]) or block_index < 0 or block_index >= len(blocks):
                raise PdfJobError("Tuzatish ko‘rsatgan bet yoki matn topilmadi.")
            block = blocks[block_index]
            for field in ("original", "translated"):
                if field not in correction:
                    continue
                value = correction[field]
                if not isinstance(value, str) or len(value) > 5000:
                    raise PdfJobError("Matn 5000 belgidan oshmasligi va matn ko‘rinishida bo‘lishi kerak.")
                value = value.strip()
                if field == "translated" and not value:
                    raise PdfJobError("Tarjima bo‘sh bo‘lishi mumkin emas.")
                block[field] = value

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
                translate_text=lambda *_args: None,
                progress_callback=lambda *_args: None,
                cancel_event=threading.Event(),
                document_mode=job.get("document_mode", "book"),
            )
            os.replace(updated_output_path, output_path)
            os.replace(updated_cache_path, cache_path)
        except Exception as exc:
            raise PdfJobError(f"Tuzatilgan PDFni qayta yaratib bo‘lmadi: {exc}") from exc
        finally:
            updated_output_path.unlink(missing_ok=True)
            updated_cache_path.unlink(missing_ok=True)

        self._update(job_id, message="Tuzatishlar PDFga saqlandi.")
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
        self.public_job(job_id)
        job_dir = self._job_dir(job_id)
        output_path = (job_dir / "translated.pdf").resolve()
        if output_path.parent != job_dir:
            raise PdfJobError("Natija manzili noto'g'ri.")
        with self._lock:
            if self._jobs[job_id].get("status") != "completed" or not output_path.is_file():
                raise PdfJobError("Tarjima PDF hali tayyor emas.")
        return output_path

    def verify_job_token(self, job_id: str, token: Optional[str]) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not token:
                return False
            expected = job.get("job_token_sha256", "")
            actual = hashlib.sha256(token.encode("utf-8")).hexdigest()
            return bool(expected) and hmac.compare_digest(actual, expected)

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise PdfJobError("PDF vazifasi topilmadi.")
            if job.get("status") == "queued":
                self._events.setdefault(job_id, threading.Event()).set()
                job.update(status="canceled", message="Vazifa bekor qilindi.", updated_at=time.time())
                self._write_metadata(job_id)
            elif job.get("status") == "processing":
                self._events.setdefault(job_id, threading.Event()).set()
                job.update(message="To'xtatish so'raldi; joriy bet tugagach to'xtaydi.", updated_at=time.time())
                self._write_metadata(job_id)
            return self.public_job(job_id)

    def resume(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise PdfJobError("PDF vazifasi topilmadi.")
            if job.get("status") not in {"failed", "canceled"}:
                raise PdfJobError("Faqat xato bergan yoki to'xtatilgan vazifani davom ettirish mumkin.")
            self._check_queue_capacity()
            job.update(status="queued", message="Qayta navbatga qo'yildi.", error=None, updated_at=time.time())
            self._events[job_id] = threading.Event()
            self._write_metadata(job_id)
            if self._translate_text is not None:
                self._executor.submit(self._run_job, job_id)
            return self.public_job(job_id)

    def shutdown(self) -> None:
        with self._lock:
            self._stopping = True
            for event in self._events.values():
                event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
