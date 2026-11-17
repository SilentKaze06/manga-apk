import asyncio
import io
import importlib.util
import json
import os
import threading
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import UploadFile, HTTPException
import atexit
import pymupdf
from PIL import Image, ImageDraw

# Importing the application must never inspect or expire real user jobs.
_test_job_directory = tempfile.TemporaryDirectory(prefix="manga_test_app_")
atexit.register(_test_job_directory.cleanup)
with patch.dict(os.environ, {"MANGA_PDF_JOB_DIR": _test_job_directory.name}):
    import main as server_main
from pdf_jobs import (PdfJobManager, PdfQueueFullError, _read_page_blocks,
                      _comic_pdf_transcript_page, _write_comic_pdf, _insert_fitted_text, inspect_pdf, translate_pdf_file)


class PdfJobPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_path = self.root / "mixed.pdf"
        self.output_path = self.root / "translated.pdf"
        self.cache_path = self.root / "page_translations.json"
        self._create_mixed_pdf()
        # Fallback tests must not download/load real neural models implicitly.
        pipeline = patch.object(server_main, "get_manga_pipeline", side_effect=RuntimeError("Isolated fallback test"))
        pipeline.start()
        self.addCleanup(pipeline.stop)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_mixed_pdf(self):
        image = Image.new("RGB", (700, 900), "white")
        ImageDraw.Draw(image).text((45, 55), "A scanned paragraph", fill="black")
        image_path = self.root / "scan.png"
        image.save(image_path)

        document = pymupdf.open()
        native = document.new_page(width=595, height=842)
        native.insert_text((60, 70), "Chapter One", fontsize=20)
        native.insert_text((60, 110), "Native paragraph for translation.", fontsize=11)
        scanned = document.new_page(width=595, height=842)
        scanned.insert_image(scanned.rect, filename=str(image_path))
        document.save(self.input_path)
        document.close()

    def _translate(self, source_text, target_lang, source_lang):
        return f"UZ {source_text}"

    def _run(self, ocr_callback, translator=None, document_mode="book", output_mode="bilingual"):
        translate_pdf_file(
            input_path=self.input_path,
            output_path=self.output_path,
            page_cache_path=self.cache_path,
            title="Sample",
            source_lang="auto",
            target_lang="uz",
            output_mode=output_mode,
            translate_text=translator or self._translate,
            progress_callback=lambda current, total: None,
            cancel_event=threading.Event(),
            ocr_page=ocr_callback,
            document_mode=document_mode,
        )

    def test_inspection_detects_native_and_scanned_pages(self):
        summary = inspect_pdf(self.input_path.read_bytes(), max_pages=10)

        self.assertEqual(summary["page_count"], 2)
        self.assertEqual(summary["native_text_pages"], 1)
        self.assertEqual(summary["scanned_pages"], [2])
        self.assertEqual(summary["empty_pages"], [])

    def test_two_column_book_is_read_down_each_column(self):
        document = pymupdf.open()
        page = document.new_page(width=600, height=800)
        page.insert_text((55, 100), "LEFT FIRST PARAGRAPH", fontsize=12)
        page.insert_text((55, 200), "LEFT SECOND PARAGRAPH", fontsize=12)
        page.insert_text((330, 100), "RIGHT FIRST PARAGRAPH", fontsize=12)
        page.insert_text((330, 200), "RIGHT SECOND PARAGRAPH", fontsize=12)
        blocks = _read_page_blocks(page)
        source_path = self.root / "two-columns.pdf"
        output_path = self.root / "two-columns-translated.pdf"
        cache_path = self.root / "two-columns-cache.json"
        document.save(source_path)
        document.close()

        texts = [block["text"] for block in blocks]
        self.assertEqual(texts, [
            "LEFT FIRST PARAGRAPH",
            "LEFT SECOND PARAGRAPH",
            "RIGHT FIRST PARAGRAPH",
            "RIGHT SECOND PARAGRAPH",
        ])
        translate_pdf_file(
            input_path=source_path,
            output_path=output_path,
            page_cache_path=cache_path,
            title="Two columns",
            source_lang="en",
            target_lang="uz",
            output_mode="translated",
            translate_text=self._translate,
            progress_callback=lambda current, total: None,
            cancel_event=threading.Event(),
        )
        with pymupdf.open(output_path) as translated:
            exported_text = "\n".join(result_page.get_text() for result_page in translated)
        positions = [exported_text.index(phrase) for phrase in texts]
        self.assertEqual(positions, sorted(positions))

    def test_repeated_book_headers_and_page_numbers_are_not_translated(self):
        source = pymupdf.open()
        for page_number in range(1, 5):
            page = source.new_page(width=595, height=842)
            page.insert_text((60, 40), "Running Header - The Silver River", fontsize=10)
            page.insert_text((60, 180), f"Unique body paragraph {page_number}.", fontsize=12)
            page.insert_text((280, 820), f"Page {page_number}", fontsize=9)
        source_path = self.root / "repeated-chrome.pdf"
        output_path = self.root / "repeated-chrome-translated.pdf"
        source.save(source_path)
        source.close()

        translate_pdf_file(
            input_path=source_path,
            output_path=output_path,
            page_cache_path=self.root / "repeated-chrome-cache.json",
            title="Novel sample",
            source_lang="en",
            target_lang="uz",
            output_mode="bilingual",
            translate_text=self._translate,
            progress_callback=lambda current, total: None,
            cancel_event=threading.Event(),
        )
        with pymupdf.open(output_path) as translated:
            exported_text = "\n".join(page.get_text() for page in translated)
        self.assertNotIn("Running Header", exported_text)
        self.assertNotIn("Page 1", exported_text)
        for page_number in range(1, 5):
            self.assertIn(f"Unique body paragraph {page_number}.", exported_text)

    def test_mixed_pdf_runs_ocr_and_exports_bilingual_text(self):
        ocr_calls = []

        def fake_ocr(page, source_lang, document_mode):
            ocr_calls.append((page.number + 1, source_lang))
            return {
                "blocks": [{"text": "Scanned source paragraph.", "confidence": 0.3}],
                "detected_source_lang": "en",
            }

        self._run(fake_ocr)

        self.assertEqual(ocr_calls, [(2, "auto")])
        with pymupdf.open(self.output_path) as result:
            exported_text = "\n".join(page.get_text() for page in result)
        self.assertIn("UZ Native paragraph for translation.", exported_text)
        self.assertIn("Scanned source paragraph.", exported_text)
        self.assertIn("UZ Scanned source paragraph.", exported_text)
        self.assertIn("OCR işonçi past", exported_text)
        self.assertIn("Manba PDF: 2-bet", exported_text)

    def test_cached_pages_resume_without_repeating_ocr_or_translation(self):
        def fake_ocr(page, source_lang, document_mode):
            return {
                "blocks": [{"text": "Scanned source paragraph.", "confidence": 0.9}],
                "detected_source_lang": "en",
            }

        self._run(fake_ocr)
        self.assertTrue(self.cache_path.exists())
        with self.cache_path.open(encoding="utf-8") as saved_cache:
            self.assertEqual(set(json.load(saved_cache)), {"0", "1"})
        self.output_path.unlink()

        def should_not_run(*args):
            self.fail("A completed page should be restored from the page cache.")

        self._run(should_not_run, translator=should_not_run)

        with pymupdf.open(self.output_path) as result:
            exported_text = "\n".join(page.get_text() for page in result)
        self.assertIn("UZ Scanned source paragraph.", exported_text)

    def test_comic_mode_keeps_art_and_places_translation_in_detected_bubble(self):
        def fake_ocr(page, source_lang, document_mode):
            self.assertEqual(document_mode, "comic")
            return {
                "blocks": [{
                    "text": f"Comic line on page {page.number + 1}.",
                    "confidence": 0.95,
                    "bbox": [100, 120, 480, 260],
                    "overlay_bbox": [90, 110, 490, 270],
                    "overlay_polygon": [[90, 110], [490, 110], [490, 270], [90, 270]],
                    "background": [1.0, 1.0, 1.0],
                }],
                "detected_source_lang": "en",
            }

        self._run(fake_ocr, document_mode="comic", output_mode="translated")

        with pymupdf.open(self.output_path) as result:
            self.assertEqual(result.page_count, 2)
            self.assertTrue(result[1].get_images())
            self.assertIn("UZ Comic line on page 2.", result[1].get_text())

    def test_comic_mode_uses_transcript_when_bubble_contour_is_unknown(self):
        def fake_ocr(page, source_lang, document_mode):
            return {
                "blocks": [{
                    "text": f"Unmasked source on page {page.number + 1}.",
                    "confidence": 0.95,
                    "bbox": [100, 120, 480, 260],
                }],
                "detected_source_lang": "en",
            }

        self._run(fake_ocr, document_mode="comic", output_mode="translated")

        with pymupdf.open(self.output_path) as result:
            self.assertEqual(result.page_count, 4)
            self.assertTrue(result[1].get_images())
            self.assertNotIn("UZ Unmasked source", result[1].get_text())
            transcript = "\n".join(page.get_text() for page in result[2:])
        self.assertIn("UZ Unmasked source on page 2.", transcript)

    def test_comic_bilingual_mode_adds_original_and_translation_transcript(self):
        def fake_ocr(page, source_lang, document_mode):
            return {
                "blocks": [{
                    "text": "Original comic words.",
                    "confidence": 0.95,
                    "bbox": [100, 120, 480, 260],
                    "background": [1.0, 1.0, 1.0],
                }],
                "detected_source_lang": "en",
            }

        self._run(fake_ocr, document_mode="comic", output_mode="bilingual")

        with pymupdf.open(self.output_path) as result:
            self.assertEqual(result.page_count, 4)
            transcript = "\n".join(page.get_text() for page in result[2:])
        self.assertIn("Original comic words.", transcript)
        self.assertIn("UZ Original comic words.", transcript)
        self.assertNotIn("�", transcript)

    def test_comic_ocr_callback_renders_bounded_page_and_returns_pdf_coordinates(self):
        def fake_ocr(image_bytes, language):
            self.assertEqual(language, "en")
            with Image.open(io.BytesIO(image_bytes)) as rendered:
                self.assertLessEqual(rendered.width * rendered.height, server_main.PDF_OCR_MAX_PIXELS)
            return []

        with pymupdf.open(self.input_path) as source:
            with patch.object(server_main, "run_paddle_ocr", side_effect=fake_ocr):
                with patch.object(server_main, "extract_paddle_lines", return_value=[{
                    "left": 100, "top": 200, "right": 300, "bottom": 400,
                    "text": "comic words", "confidence": 0.9,
                }]):
                    result = server_main.ocr_pdf_page(source[0], "en", "comic")

        self.assertEqual(result["detected_source_lang"], "en")
        self.assertEqual(result["blocks"][0]["bbox"], [40.0, 80.0, 120.0, 160.0])
        self.assertEqual(len(result["blocks"][0]["background"]), 3)

    def test_pdf_ocr_accepts_common_world_book_languages_with_latin_model(self):
        with pymupdf.open(self.input_path) as source:
            for language in ("fr", "de", "es", "pt", "it", "tr", "nl", "pl", "uz"):
                with self.subTest(language=language):
                    with patch.object(server_main, "run_paddle_ocr", return_value=[]) as paddle:
                        with patch.object(server_main, "extract_paddle_lines", return_value=[]):
                            result = server_main.ocr_pdf_page(source[0], language, "book")
                    self.assertEqual(paddle.call_args.args[1], "latin")
                    self.assertEqual(result["detected_source_lang"], language)

    def test_pdf_ocr_auto_keeps_source_detection_for_latin_translation(self):
        with pymupdf.open(self.input_path) as source:
            with patch.object(server_main, "run_auto_paddle_ocr", return_value=("en", [])):
                with patch.object(server_main, "extract_paddle_lines", return_value=[]):
                    result = server_main.ocr_pdf_page(source[0], "auto", "book")
        self.assertEqual(result["detected_source_lang"], "auto")

    def test_auto_ocr_tries_shared_latin_model_for_accented_latin_text(self):
        recognized_lines = [{"text": "maison française", "confidence": 0.9}]
        with patch.object(server_main, "run_paddle_ocr", return_value=[]) as paddle:
            with patch.object(server_main, "extract_paddle_lines", return_value=recognized_lines):
                language, _results = server_main.run_auto_paddle_ocr(b"mock-image")
        self.assertEqual(language, "auto")
        self.assertIn("latin", [call.args[1] for call in paddle.call_args_list])

    def test_common_latin_language_codes_share_one_paddle_model(self):
        paddle_ocr_class = Mock(return_value=object())
        fake_paddleocr = SimpleNamespace(PaddleOCR=paddle_ocr_class)
        with patch.object(server_main, "_paddle_latin", None):
            with patch.object(server_main.importlib, "import_module", return_value=fake_paddleocr):
                french = server_main.get_paddle_ocr("fr")
                german = server_main.get_paddle_ocr("de")
        self.assertIs(french, german)
        paddle_ocr_class.assert_called_once_with(use_angle_cls=True, lang="latin", show_log=False)

    def test_japanese_comic_ocr_groups_neighbor_columns_and_refines_with_manga_ocr(self):
        with pymupdf.open(self.input_path) as source:
            with patch.object(server_main, "run_paddle_ocr", return_value=[]) as paddle:
                with patch.object(server_main, "extract_paddle_lines", return_value=[
                    {"left": 700, "top": 300, "right": 730, "bottom": 550, "text": "右列", "confidence": 0.9},
                    {"left": 660, "top": 310, "right": 690, "bottom": 545, "text": "左列", "confidence": 0.8},
                ]):
                    with patch.object(server_main, "run_manga_ocr", return_value="漫画の台詞") as manga:
                        result = server_main.ocr_pdf_page(source[0], "ja", "comic")

        self.assertEqual(paddle.call_args[0][1], "japan")
        manga.assert_called_once()
        self.assertEqual(len(result["blocks"]), 1)
        self.assertEqual(result["blocks"][0]["text"], "漫画の台詞")

    def test_comic_overlay_box_expands_to_enclosed_speech_bubble(self):
        bubble_image = Image.new("RGB", (600, 600), "white")
        drawing = ImageDraw.Draw(bubble_image)
        drawing.ellipse((150, 150, 450, 450), outline="black", width=8)
        drawing.text((260, 280), "HELLO", fill="black")
        image_path = self.root / "bubble.png"
        bubble_image.save(image_path)
        bubble_pdf = self.root / "bubble.pdf"
        pdf = pymupdf.open()
        page = pdf.new_page(width=240, height=240)
        page.insert_image(page.rect, filename=str(image_path))
        pdf.save(bubble_pdf)
        pdf.close()

        with pymupdf.open(bubble_pdf) as source:
            with patch.object(server_main, "run_paddle_ocr", return_value=[]):
                with patch.object(server_main, "extract_paddle_lines", return_value=[{
                    "left": 260, "top": 280, "right": 340, "bottom": 320,
                    "text": "HELLO", "confidence": 0.9,
                }]):
                    result = server_main.ocr_pdf_page(source[0], "en", "comic")

        block = result["blocks"][0]
        self.assertIn("overlay_bbox", block)
        self.assertIn("overlay_polygon", block)
        self.assertGreaterEqual(len(block["overlay_polygon"]), 3)
        overlay_width = block["overlay_bbox"][2] - block["overlay_bbox"][0]
        detected_width = block["bbox"][2] - block["bbox"][0]
        self.assertGreater(overlay_width, detected_width * 1.5)
        self.assertLess(block["overlay_bbox"][2] - block["overlay_bbox"][0], 200)

    def test_api_accepts_comic_pdf_and_reports_ocr_pages(self):
        fake_job = {
            "job_id": "00000000-0000-0000-0000-000000000001",
            "job_token": "secret-token",
            "status": "queued",
        }

        async def submit():
            upload = UploadFile(filename="comic.pdf", file=io.BytesIO(self.input_path.read_bytes()))
            return await server_main.submit_pdf_translation(
                file=upload,
                document_mode="comic",
                source_lang="ja",
                target_lang="uz",
                output_mode="translated",
                _authorized=None,
            )

        with patch.object(server_main.pdf_job_manager, "create_job", return_value=fake_job) as create_job:
            response = asyncio.run(submit())

        self.assertEqual(response["ocr_pages"], [2])
        self.assertEqual(response["native_text_pages"], 1)
        self.assertEqual(create_job.call_args.kwargs["document_mode"], "comic")

    def test_job_token_and_queued_state_survive_manager_recreation(self):
        data_dir = self.root / "jobs"
        first_manager = PdfJobManager(data_dir)
        created = first_manager.create_job(
            pdf_bytes=self.input_path.read_bytes(),
            filename="mixed.pdf",
            source_lang="auto",
            target_lang="uz",
            output_mode="translated",
            page_count=2,
            document_mode="comic",
        )
        first_manager.shutdown()

        restarted_manager = PdfJobManager(data_dir)
        try:
            self.assertTrue(restarted_manager.verify_job_token(created["job_id"], created["job_token"]))
            self.assertFalse(restarted_manager.verify_job_token(created["job_id"], "incorrect"))
            restored = restarted_manager.public_job(created["job_id"])
            self.assertEqual(restored["status"], "queued")
            self.assertEqual(restored["document_mode"], "comic")
        finally:
            restarted_manager.shutdown()

    def test_completed_job_can_be_reviewed_corrected_and_reexported(self):
        source = pymupdf.open()
        page = source.new_page()
        page.insert_text((60, 90), "A simple sentence for editing.", fontsize=14)
        pdf_bytes = source.tobytes()
        source.close()

        manager = PdfJobManager(self.root / "review-jobs")
        try:
            job = manager.create_job(
                pdf_bytes=pdf_bytes,
                filename="review.pdf",
                source_lang="en",
                target_lang="uz",
                output_mode="translated",
                page_count=1,
                document_mode="book",
            )
            manager.start_pending(self._translate)
            deadline = time.monotonic() + 5
            while manager.public_job(job["job_id"])["status"] in {"queued", "processing"} and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(manager.public_job(job["job_id"])["status"], "completed")

            review = manager.review_page(job["job_id"], 0)
            self.assertEqual(review["blocks"][0]["original"], "A simple sentence for editing.")
            self.assertEqual(review["blocks"][0]["translated"], "UZ A simple sentence for editing.")

            manager.apply_corrections(job["job_id"], [{
                "page_index": 0,
                "block_index": 0,
                "original": "OCR fixed sentence.",
                "translated": "Tuzatilgan jumla.",
            }])
            updated = manager.review_page(job["job_id"], 0)
            self.assertEqual(updated["blocks"][0]["original"], "OCR fixed sentence.")
            with pymupdf.open(manager.output_path(job["job_id"])) as result:
                self.assertIn("Tuzatilgan jumla.", result[0].get_text())
        finally:
            manager.shutdown()

    def test_primary_comic_pipeline_preserves_auto_and_receives_selected_language(self):
        process = Mock(return_value={
            "dialogues": [{"transcription": "This is an English comic.", "bbox": [40, 40, 200, 120], "bubble_id": 7}],
            "cleaned_image": Image.new("RGB", (750, 1000), "white"),
            "cleaned_bubble_ids": [7], "detected_source_lang": "auto",
        })
        with pymupdf.open(self.input_path) as source, patch.object(
            server_main, "get_manga_pipeline", return_value=SimpleNamespace(process_page=process)
        ):
            result = server_main.ocr_pdf_page(source[0], "auto", "comic", job_dir=self.root)
        self.assertEqual(process.call_args.kwargs["source_lang"], "auto")
        self.assertEqual(result["detected_source_lang"], "auto")
        self.assertTrue(result["blocks"][0]["cleaned"])
        self.assertTrue((self.root / result["cleaned_image_filename"]).is_file())

    def test_primary_comic_empty_result_uses_requested_language_fallback(self):
        process = Mock(return_value={"dialogues": [], "requires_ocr_fallback": True})
        with pymupdf.open(self.input_path) as source, patch.object(
            server_main, "get_manga_pipeline", return_value=SimpleNamespace(process_page=process)
        ), patch.object(server_main, "run_paddle_ocr", return_value=[]) as paddle:
            result = server_main.ocr_pdf_page(source[0], "ko", "comic")
        self.assertEqual(process.call_args.kwargs["source_lang"], "ko")
        self.assertEqual(paddle.call_args.args[1], "korean")
        self.assertEqual(result["detected_source_lang"], "ko")
        self.assertEqual(server_main.health()["manga_pipeline_status"], "fallback")

    def test_long_comic_transcript_flows_without_losing_any_words(self):
        text = ("Review paragraph sample. " * 205)[:4900]
        document = pymupdf.open()
        try:
            _comic_pdf_transcript_page(document, 1, [{"translated": text}], False)
            exported = " ".join(page.get_text() for page in document)
            self.assertGreater(document.page_count, 1)
            self.assertEqual(exported.count("Review"), text.count("Review"))
            self.assertEqual(exported.count("sample."), text.count("sample."))
        finally:
            document.close()

    def test_cleaned_comic_overflow_keeps_original_artwork_and_full_transcript(self):
        source_path = self.root / "source-art.pdf"
        output_path = self.root / "preserved.pdf"
        with pymupdf.open() as source:
            page = source.new_page(width=300, height=400)
            page.draw_rect(pymupdf.Rect(20, 20, 70, 50), color=None, fill=(0, 0, 0))
            source.save(source_path)
        Image.new("RGB", (300, 400), "white").save(self.root / "cleaned.png")
        text = ("Overflow translation. " * 230)[:4900]
        _write_comic_pdf(source_path, output_path, [{
            "cleaned_image_filename": "cleaned.png",
            "blocks": [{"original": "source", "translated": text, "bbox": [20, 20, 70, 50], "cleaned": True}],
        }], False)
        with pymupdf.open(output_path) as result:
            self.assertEqual(result[0].get_pixmap().pixel(30, 30), (0, 0, 0))
            transcript = " ".join(page.get_text() for page in result)
            self.assertEqual(transcript.count("Overflow"), text.count("Overflow"))

    def test_cleaned_comic_patches_only_the_successfully_translated_bubble(self):
        source_path, output_path = self.root / "art.pdf", self.root / "patched.pdf"
        with pymupdf.open() as source:
            page = source.new_page(width=300, height=400)
            page.draw_rect(page.rect, color=None, fill=(0, 0, 0))
            source.save(source_path)
        Image.new("RGB", (300, 400), "white").save(self.root / "cleaned.png")
        _write_comic_pdf(source_path, output_path, [{
            "cleaned_image_filename": "cleaned.png",
            "blocks": [{"translated": "Salom dunyo", "bbox": [20, 20, 280, 120], "cleaned": True}],
        }], False)
        with pymupdf.open(output_path) as result:
            self.assertEqual(result.page_count, 1)
            self.assertIn("Salom dunyo", result[0].get_text())
            pixels = result[0].get_pixmap()
            self.assertEqual(pixels.pixel(150, 250), (0, 0, 0))
            self.assertEqual(pixels.pixel(25, 25), (255, 255, 255))

    def test_translation_routes_use_timeouts_and_html_fallback(self):
        response = io.BytesIO(b'<div class="result-container">Salom &amp; dunyo</div>')
        with patch.object(server_main.urllib.request, "urlopen", side_effect=[TimeoutError(), response]) as request:
            result = server_main.translate_text("Hello world", "uz", "en")
        self.assertEqual(result, "Salom & dunyo")
        self.assertEqual(request.call_count, 2)
        for call in request.call_args_list:
            self.assertGreater(call.kwargs["timeout"], 0)
            self.assertLessEqual(call.kwargs["timeout"], server_main.TRANSLATION_TIMEOUT_SECONDS)

    def test_translation_stops_after_two_failed_routes(self):
        with patch.object(server_main.urllib.request, "urlopen", side_effect=TimeoutError()) as request:
            self.assertIsNone(server_main.translate_text("Hello", "uz", "en"))
        self.assertEqual(request.call_count, 2)

    def test_slow_translation_response_cannot_extend_retry_budget(self):
        response = io.BytesIO(b'partial response')
        with patch.object(server_main.urllib.request, "urlopen", return_value=response) as request:
            with patch.object(server_main, "TRANSLATION_TIMEOUT_SECONDS", 8):
                with patch.object(server_main.time, "monotonic", side_effect=[0, 0, 0, 17, 17]):
                    self.assertIsNone(server_main.translate_text("Hello", "uz", "en"))
        self.assertEqual(request.call_count, 1)

    def test_cancel_during_first_translation_route_skips_fallback(self):
        canceled = threading.Event()
        def timeout_and_cancel(*args, **kwargs):
            canceled.set()
            raise TimeoutError()
        with patch.object(server_main.urllib.request, "urlopen", side_effect=timeout_and_cancel) as request:
            with self.assertRaises(InterruptedError):
                server_main.translate_text("Hello", "uz", "en", cancel_event=canceled)
        self.assertEqual(request.call_count, 1)

    def test_cancellation_between_chunks_prevents_more_translation(self):
        canceled = threading.Event()
        translator = Mock(side_effect=lambda *args: (canceled.set() or "translated"))
        with self.assertRaises(InterruptedError):
            translate_pdf_file(self.input_path, self.output_path, self.cache_path, "Sample", "en", "uz",
                               "translated", translator, lambda *args: None, canceled)
        self.assertEqual(translator.call_count, 1)
        self.assertFalse(self.output_path.exists())

    def test_queue_limit_rejects_upload_and_resume_until_capacity_is_free(self):
        with patch.dict(os.environ, {"MANGA_PDF_MAX_PENDING_JOBS": "1"}):
            manager = PdfJobManager(self.root / "bounded-jobs")
        def create():
            return manager.create_job(self.input_path.read_bytes(), "book.pdf", "en", "uz", "translated", 2)
        try:
            first = create()
            with self.assertRaises(PdfQueueFullError):
                create()
            self.assertEqual(len(list(manager.data_dir.glob("*/input.pdf"))), 1)
            manager.cancel(first["job_id"])
            second = create()
            with self.assertRaises(PdfQueueFullError):
                manager.resume(first["job_id"])
            manager.cancel(second["job_id"])
            self.assertEqual(manager.resume(first["job_id"])["status"], "queued")
        finally:
            manager.shutdown()

    def test_server_requires_key_or_explicit_local_opt_in(self):
        with patch.object(server_main, "SERVER_API_KEY", ""), patch.object(server_main, "ALLOW_UNAUTHENTICATED", False):
            with self.assertRaises(HTTPException) as error:
                server_main.verify_api_key(None)
            self.assertEqual(error.exception.status_code, 503)
        with patch.object(server_main, "SERVER_API_KEY", ""), patch.object(server_main, "ALLOW_UNAUTHENTICATED", True):
            server_main.verify_api_key(None)
        with patch.object(server_main, "SERVER_API_KEY", "test-only-key"):
            with self.assertRaises(HTTPException) as error:
                server_main.verify_api_key("wrong-key")
            self.assertEqual(error.exception.status_code, 401)
            server_main.verify_api_key("test-only-key")

    def test_server_configuration_defaults_to_authenticated_access(self):
        # Load the actual configuration in isolation; patched runtime booleans
        # alone would miss an accidental permissive environment default.
        spec = importlib.util.spec_from_file_location("server_default_config", server_main.__file__)
        isolated = importlib.util.module_from_spec(spec)
        clean_environment = {key: value for key, value in os.environ.items()
                             if key not in {"MANGA_SERVER_API_KEY", "MANGA_ALLOW_UNAUTHENTICATED"}}
        clean_environment["MANGA_PDF_JOB_DIR"] = str(self.root / "isolated-config-jobs")
        with patch.dict(os.environ, clean_environment, clear=True):
            spec.loader.exec_module(isolated)
        try:
            self.assertFalse(isolated.ALLOW_UNAUTHENTICATED)
            self.assertFalse(isolated.health()["translation_ready"])
            with self.assertRaises(HTTPException) as error:
                isolated.verify_api_key(None)
            self.assertEqual(error.exception.status_code, 503)
        finally:
            isolated.pdf_job_manager.shutdown()

    def test_full_queue_is_reported_as_retryable_http_429(self):
        async def submit():
            return await server_main.submit_pdf_translation(
                file=UploadFile(filename="comic.pdf", file=io.BytesIO(self.input_path.read_bytes())),
                document_mode="comic", source_lang="en", target_lang="uz", output_mode="translated",
                _authorized=None,
            )
        with patch.object(server_main.pdf_job_manager, "create_job", side_effect=PdfQueueFullError("Full")):
            with self.assertRaises(HTTPException) as error:
                asyncio.run(submit())
        self.assertEqual(error.exception.status_code, 429)
        self.assertEqual(error.exception.headers["Retry-After"], "30")

    def test_missing_server_auth_does_not_start_persisted_jobs(self):
        with patch.object(server_main, "SERVER_API_KEY", ""), patch.object(server_main, "ALLOW_UNAUTHENTICATED", False):
            with patch.object(server_main.pdf_job_manager, "start_pending") as start:
                server_main.resume_pdf_jobs()
        start.assert_not_called()

    def test_disabled_pdf_backend_rejects_work_without_reading_upload(self):
        class UnreadableUpload:
            async def read(self, size):
                self.fail("Disabled backend must not read uploaded documents")
        with patch.object(server_main, "PDF_ENABLED", False):
            with self.assertRaises(HTTPException) as error:
                asyncio.run(server_main.submit_pdf_translation(
                    file=UnreadableUpload(), document_mode="comic", source_lang="en",
                    target_lang="uz", output_mode="translated", _authorized=None,
                ))
            self.assertEqual(error.exception.status_code, 503)
            self.assertFalse(server_main.health()["pdf_translation_enabled"])
            with self.assertRaises(HTTPException):
                server_main.resume_pdf_translation("unused-id", None, None)
            with patch.object(server_main, "SERVER_API_KEY", "test-key"), patch.object(server_main.pdf_job_manager, "start_pending") as start:
                server_main.resume_pdf_jobs()
            start.assert_not_called()

    def test_comic_pdf_unicode_cjk_and_arabic_rendering(self):
        cjk_text = "日本語 中文 한글"
        arabic_text = "مرحبا بالعالم"

        # Test direct insertion on comic page and ensure no NUL characters (\x00)
        with pymupdf.open() as doc:
            p = doc.new_page(width=300, height=400)
            ok_cjk = _insert_fitted_text(p, pymupdf.Rect(20, 20, 280, 80), cjk_text, 6.0, 14.0)
            ok_ar = _insert_fitted_text(p, pymupdf.Rect(20, 100, 280, 160), arabic_text, 6.0, 14.0)
            self.assertTrue(ok_cjk)
            self.assertTrue(ok_ar)
            page_text = p.get_text()
            self.assertNotIn("\x00", page_text)
            self.assertIn("日本語", page_text)
            self.assertIn("中文", page_text)
            self.assertIn("한글", page_text)

        # Test comic transcript page generation
        with pymupdf.open() as tdoc:
            _comic_pdf_transcript_page(
                tdoc, 1,
                [{"original": "Orig CJK", "translated": cjk_text}, {"original": "Orig AR", "translated": arabic_text}],
                include_source=True
            )
            transcript_text = " ".join(tp.get_text() for tp in tdoc)
            self.assertNotIn("\x00", transcript_text)
            self.assertIn("日本語", transcript_text)
            self.assertIn("中文", transcript_text)
            self.assertIn("한글", transcript_text)

    def test_comic_pdf_rotated_page_coordinates_all_orientations(self):
        for rot in [0, 90, 180, 270]:
            with self.subTest(rotation=rot):
                doc = pymupdf.open()
                page = doc.new_page(width=300, height=400)
                page.set_rotation(rot)

                vis_box = pymupdf.Rect(50, 60, 150, 160)
                img = Image.new("RGB", (100, 100), color=(0, 200, 0))
                buf = io.BytesIO()
                img.save(buf, format="PNG")

                ok = _insert_fitted_text(
                    page,
                    vis_box,
                    "Rotated Text",
                    min_size=6.0,
                    max_size=12.0,
                    background_image=buf.getvalue(),
                )
                self.assertTrue(ok)

                pix = page.get_pixmap()
                green_pixels = []
                for y in range(pix.height):
                    for x in range(pix.width):
                        r, g, b = pix.pixel(x, y)
                        if g > 150 and r < 50:
                            green_pixels.append((x, y))

                self.assertTrue(len(green_pixels) > 0, f"Green patch was not drawn for rot={rot}")
                xs = [pt[0] for pt in green_pixels]
                ys = [pt[1] for pt in green_pixels]

                self.assertAlmostEqual(min(xs), 50, delta=2)
                self.assertAlmostEqual(max(xs), 149, delta=2)
                self.assertAlmostEqual(min(ys), 60, delta=2)
                self.assertAlmostEqual(max(ys), 159, delta=2)


if __name__ == "__main__":
    unittest.main()

