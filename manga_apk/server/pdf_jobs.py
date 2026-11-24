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
import math
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional, Any

import numpy as np
import cv2
import pymupdf

logger = logging.getLogger("manga-server")

from pdf_renderer import (
    COMIC_CJK_FONT_BUFFER,
    COMIC_FIGO_FONT_BUFFER,
    COMIC_FONT_BUFFER,
    COMIC_FONT_NAME,
    MAX_TRANSLATION_CHUNK_CHARS,
    PAGE_HEIGHT,
    PAGE_RECT,
    PAGE_WIDTH,
    TEXT_RECT,
    PdfJobError,
    _ARABIC_CODE_PATTERN,
    _CJK_CODE_PATTERN,
    _MANGA_PUNCTUATION_CHARS,
    _bidi_display_line,
    _bidi_display_lines,
    _comic_pdf_transcript_page,
    _get_font_and_shaped_text,
    _has_renderable_content,
    _insert_fitted_text,
    _insert_page_font,
    _lines_fit_box,
    _sample_page_background_color,
    _translation_html,
    _wrap_for_box,
    _write_comic_pdf,
    _write_reflowed_pdf,
)

PAGE_NUMBER_PATTERN = re.compile(r"^(?:page\s*)?(?:\d{1,5}|[ivxlcdm]{1,8})$", re.IGNORECASE)


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
        except Exception as exc:
            logger.debug("Failed closing temp file: %s", exc)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError as exc:
            logger.debug("Failed unlinking temp file: %s", exc)
        raise


def _partition_into_columns(blocks: list[dict], page_width: float) -> Optional[list[list[dict]]]:
    """Partitions blocks into 2 or 3 distinct vertical columns if a clear vertical gap separates them."""
    if len(blocks) < 4:
        return None

    by_x0 = sorted(blocks, key=lambda b: (float(b["bbox"][0]), float(b["bbox"][1])))
    gap_threshold = max(20.0, page_width * 0.08)

    potential_splits = []
    for i in range(1, len(by_x0)):
        gap = float(by_x0[i]["bbox"][0]) - float(by_x0[i - 1]["bbox"][0])
        if gap >= gap_threshold and i >= 2 and (len(by_x0) - i) >= 2:
            potential_splits.append((gap, i))

    if not potential_splits:
        return None

    potential_splits.sort(key=lambda item: item[0], reverse=True)

    for best_gap, split_idx in potential_splits:
        split_x = (float(by_x0[split_idx - 1]["bbox"][0]) + float(by_x0[split_idx]["bbox"][0])) / 2.0
        left = [b for b in blocks if float(b["bbox"][0]) < split_x]
        right = [b for b in blocks if float(b["bbox"][0]) >= split_x]
        if not left or not right:
            continue
        max_left_x2 = max(float(b["bbox"][2]) for b in left)
        min_right_x0 = min(float(b["bbox"][0]) for b in right)
        if max_left_x2 <= min_right_x0:
            if len(right) >= 4:
                sub = _partition_into_columns(right, page_width)
                if sub:
                    return [left] + sub
            return [left, right]
    return None


def _order_page_blocks(blocks: list[dict], page_width: float) -> list[dict]:
    """Read multi-column page down columns within sections, preserving spanning blocks."""
    by_position = lambda block: (float(block["bbox"][1]), float(block["bbox"][0]))
    natural_order = sorted(blocks, key=by_position)
    if len(natural_order) < 4 or page_width <= 0:
        return natural_order

    # Keep full-width headings/footers/banners outside the columns.
    column_blocks = []
    spanning_blocks = []
    for block in natural_order:
        width = float(block["bbox"][2]) - float(block["bbox"][0])
        (column_blocks if width < page_width * 0.72 else spanning_blocks).append(block)

    if not spanning_blocks:
        cols = _partition_into_columns(column_blocks, page_width)
        if cols:
            result = []
            for col in cols:
                result.extend(sorted(col, key=by_position))
            return result
        return natural_order

    sorted_spanning = sorted(spanning_blocks, key=by_position)
    first_body_y = min(float(b["bbox"][1]) for b in column_blocks) if column_blocks else 0.0
    last_body_y = max(float(b["bbox"][3]) for b in column_blocks) if column_blocks else 0.0
    top_spanning = [b for b in spanning_blocks if float(b["bbox"][3]) <= first_body_y]
    bottom_spanning = [b for b in spanning_blocks if float(b["bbox"][1]) >= last_body_y]

    if len(top_spanning) + len(bottom_spanning) == len(spanning_blocks):
        cols = _partition_into_columns(column_blocks, page_width)
        if cols:
            ordered_cols = []
            for col in cols:
                ordered_cols.extend(sorted(col, key=by_position))
            return sorted(top_spanning, key=by_position) + ordered_cols + sorted(bottom_spanning, key=by_position)
        return natural_order

    # Mid-page spanning blocks present: divide page into vertical sections
    result = []
    current_y = 0.0
    for span in sorted_spanning:
        span_top = float(span["bbox"][1])
        span_bottom = float(span["bbox"][3])
        band = [b for b in column_blocks if current_y <= (float(b["bbox"][1]) + float(b["bbox"][3])) / 2.0 < span_top]
        if band:
            cols = _partition_into_columns(band, page_width)
            if cols:
                for col in cols:
                    result.extend(sorted(col, key=by_position))
            else:
                result.extend(sorted(band, key=by_position))
        result.append(span)
        current_y = span_bottom

    band = [b for b in column_blocks if (float(b["bbox"][1]) + float(b["bbox"][3])) / 2.0 >= current_y]
    if band:
        cols = _partition_into_columns(band, page_width)
        if cols:
            for col in cols:
                result.extend(sorted(col, key=by_position))
        else:
            result.extend(sorted(band, key=by_position))

    seen = {id(b) for b in result}
    if len(seen) == len(blocks):
        return result
    return natural_order


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


# Rendering functions (_translation_html, _write_reflowed_pdf, _wrap_for_box, _lines_fit_box,
# _insert_fitted_text, _comic_pdf_transcript_page, _sample_page_background_color, _write_comic_pdf)
# are imported from pdf_renderer and re-exported above.



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
        with pymupdf.open(str(input_path)) as init_doc:
            total_pages = init_doc.page_count
            repeated_chrome = _find_repeated_page_chrome(init_doc) if document_mode == "book" else set()
    except Exception as exc:
        raise PdfJobError(f"PDFni qayta oçib bölmadi: {exc}") from exc

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

    cache_lock = threading.Lock()
    completed_pages = sum(1 for i in range(total_pages) if str(i) in cached_pages)
    if completed_pages > 0:
        progress_callback(completed_pages, total_pages)

    abort_event = threading.Event()

    def _is_stopped() -> bool:
        return cancel_event.is_set() or abort_event.is_set()

    # Container has 8 vCPUs dedicated; 4 concurrent workers provides the optimal 4x speedup
    page_workers = max(1, min(int(os.environ.get("MANGA_PDF_PAGE_WORKERS", "4")), max(1, total_pages)))

    def _process_page_worker(page_index: int) -> dict:
        if _is_stopped():
            raise InterruptedError("Vazifa foydalanuvçi tomonidan töxtatildi.")
        key = str(page_index)
        with cache_lock:
            if key in cached_pages:
                return cached_pages[key]

        # Open and close pymupdf.Document strictly on this worker thread
        thread_doc = pymupdf.open(str(input_path))
        try:
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
        finally:
            try:
                thread_doc.close()
            except Exception as exc:
                logger.debug("Failed closing thread_doc: %s", exc)

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
                {"text": repair_english_word_spacing(block.get("text", "")), "is_heading": False, "block_id": block.get("block_id") or f"b_{b_idx+1}", **{
                    key: block[key] for key in ("bbox", "overlay_bbox", "text_erase_bbox", "overlay_polygon", "confidence", "background", "cleaned", "category", "bubble_shape", "shape", "bubble_type", "bubble_id", "block_id") if key in block
                }}
                for b_idx, block in enumerate(blocks) if isinstance(block, dict) and block.get("text", "").strip()
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
                chunk_bids = []
                seen_bids = set()
                for i, c in enumerate(source_chunks):
                    raw_bid = str(c.get("block_id") or f"b_{i+1}")
                    bid = raw_bid
                    if bid in seen_bids:
                        bid = f"{raw_bid}_{i+1}"
                    seen_bids.add(bid)
                    c["block_id"] = bid
                    chunk_bids.append(bid)

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

                try:
                    gemini_translations = translate_dialogues_with_gemini(
                        chunk_texts, target_lang, page_source_lang,
                        api_key=effective_gemini_key, chapter_context=chapter_context,
                        block_ids=chunk_bids, return_dict=True
                    )
                except TypeError:
                    gemini_translations = translate_dialogues_with_gemini(
                        chunk_texts, target_lang, page_source_lang,
                        api_key=effective_gemini_key, chapter_context=chapter_context
                    )
            except Exception as e:
                logger.debug("Gemini batch tarjima xatosi: %s", e)

        # Fast batch fallback for missing blocks using translate_batch (only when default translate_text is used)
        from text_translator import translate_text as _default_translate_text
        missing_indices = []
        for idx, source_chunk in enumerate(source_chunks):
            bid = str(source_chunk.get("block_id") or f"b_{idx+1}")
            trans_val = None
            if isinstance(gemini_translations, dict):
                trans_val = gemini_translations.get(bid) or gemini_translations.get(f"[{bid}]")
            elif isinstance(gemini_translations, list) and idx < len(gemini_translations) and gemini_translations[idx]:
                trans_val = gemini_translations[idx]
            if not trans_val:
                missing_indices.append(idx)

        if (translate_text is _default_translate_text or getattr(translate_text, "__name__", "") == "translate_text") and missing_indices and len(missing_indices) > 1:
            try:
                from text_translator import translate_batch
                batch_res = translate_batch(
                    [source_chunks[i]["text"] for i in missing_indices],
                    target_lang,
                    page_source_lang,
                    cancel_event=cancel_event,
                )
                if gemini_translations is None:
                    gemini_translations = {}
                elif isinstance(gemini_translations, list):
                    gemini_translations = {
                        str(source_chunks[i].get("block_id") or f"b_{i+1}"): gemini_translations[i]
                        for i in range(len(gemini_translations)) if gemini_translations[i]
                    }
                if isinstance(gemini_translations, dict):
                    for i_pos, idx in enumerate(missing_indices):
                        if i_pos < len(batch_res) and batch_res[i_pos]:
                            bid = str(source_chunks[idx].get("block_id") or f"b_{idx+1}")
                            gemini_translations[bid] = batch_res[i_pos]
            except Exception as b_exc:
                logger.debug("PDF translate_batch optimization error: %s", b_exc)

        for idx, source_chunk in enumerate(source_chunks):
            if _is_stopped():
                raise InterruptedError("Vazifa foydalanuvçi tomonidan töxtatildi.")
            translated = None
            bid = str(source_chunk.get("block_id") or f"b_{idx+1}")
            if isinstance(gemini_translations, dict):
                translated = gemini_translations.get(bid) or gemini_translations.get(f"[{bid}]")
            elif isinstance(gemini_translations, list) and idx < len(gemini_translations) and gemini_translations[idx]:
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
                "block_id": bid,
            }
            for chunk_key in ("bbox", "overlay_bbox", "text_erase_bbox", "overlay_polygon", "confidence", "background", "cleaned", "category", "bubble_shape", "shape", "bubble_type", "bubble_id", "block_id"):
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
        else:
            for i in pages_to_process:
                _process_page_worker(i)

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
                        except RuntimeError as exc:
                            logger.debug("Correction lock already released: %s", exc)

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
