"""Unit tests for server API, cleaning/rendering, OCR routing, PDF jobs, and translators."""
import asyncio
import atexit
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pymupdf
from fastapi import HTTPException, UploadFile
from PIL import Image, ImageDraw

if "main" in sys.modules:
    server_main = sys.modules["main"]
else:
    _tmp_job_dir = tempfile.TemporaryDirectory(prefix="manga_test_fixes_")
    atexit.register(_tmp_job_dir.cleanup)
    with patch.dict(
        os.environ,
        {
            "MANGA_PDF_JOB_DIR": _tmp_job_dir.name,
            "DEEPL_API_KEY": "",
            "DEEPL_API_KEY_FILE": "",
            "GEMINI_API_KEY": "",
            "GEMINI_API_KEY_FILE": "",
        },
    ):
        import main as server_main

import deepl_translator
import gemini_translator
import pdf_jobs
import pdf_renderer
import routes_pdf
import text_translator


class DummyUploadFile:
    def __init__(self, data: bytes, filename: str = "page.png"):
        self.data = data
        self.filename = filename

    async def read(self, size: int = -1):
        return self.data


class ServerCleanAndRenderTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.addCleanup(self.temp_dir.cleanup)

    def test_clean_manga_falls_back_to_clean_bubble_crop_when_edit_mask_is_none(self):
        img = Image.new("RGB", (200, 200), (245, 238, 219))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        fake_cleaner = SimpleNamespace(
            clean_page=Mock(return_value=(img, None, None))
        )
        cleaned_crops = []

        def fake_clean_bubble_crop(crop, **kwargs):
            cleaned_crops.append(crop.shape)
            return np.full_like(crop, 240)

        with patch.object(server_main, "get_manga_cleaner", return_value=fake_cleaner), patch.object(
            server_main, "clean_bubble_crop", side_effect=fake_clean_bubble_crop, create=True
        ):
            resp = asyncio.run(
                server_main.clean_manga(
                    file=DummyUploadFile(buf.getvalue()),
                    boxes_json=json.dumps([[20, 20, 120, 100]]),
                    _authorized=None,
                )
            )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(fake_cleaner.clean_page.call_args.kwargs.get("allowed_boxes"), [[20, 20, 120, 100]])
        self.assertEqual(len(cleaned_crops), 1)

    def test_translate_page_image_cleans_uncleaned_non_sfx_blocks_only(self):
        img = Image.new("RGB", (240, 240), (255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        # Bubble 1 was cleaned by pipeline; Bubble 2 was NOT in cleaned_bubble_ids; Bubble 3 is SFX
        fake_pipe = SimpleNamespace(
            process_page=Mock(
                return_value={
                    "cleaned_image": img.copy(),
                    "cleaned_bubble_ids": [1],
                    "detected_source_lang": "en",
                    "dialogues": [
                        {"bubble_id": 1, "block_id": "b_1", "bbox": [10, 10, 80, 60], "transcription": "ONE", "category": "speech_bubble"},
                        {"bubble_id": 2, "block_id": "b_2", "bbox": [10, 80, 120, 140], "transcription": "TWO", "category": "speech_bubble"},
                        {"bubble_id": 3, "block_id": "sfx_3", "bbox": [130, 130, 200, 190], "transcription": "BOOM", "category": "sfx"},
                    ],
                }
            )
        )
        fake_cleaner = SimpleNamespace(clean_page=Mock(return_value=(img, None, None)))
        cleaned_shapes = []

        def spy_clean_bubble_crop(crop, **kwargs):
            cleaned_shapes.append(crop.shape)
            return crop

        with patch.object(server_main, "get_manga_pipeline", return_value=fake_pipe), patch.object(
            server_main, "get_manga_cleaner", return_value=fake_cleaner
        ), patch.object(
            server_main, "clean_bubble_crop", side_effect=spy_clean_bubble_crop, create=True
        ), patch(
            "gemini_translator.translate_dialogues_with_gemini",
            return_value={"b_1": "BIR", "b_2": "IKKI"},
        ), patch(
            "ml.typesetting.render_translated_dialogues", side_effect=lambda im, dlgs: im
        ):
            resp = asyncio.run(
                server_main.translate_page_image(
                    file=DummyUploadFile(buf.getvalue()),
                    lang="en",
                    target_lang="uz",
                    _authorized=None,
                )
            )

        self.assertEqual(resp.status_code, 200)
        # Only bubble 2 ([10, 80, 120, 140] -> 60x110) should be cleaned via clean_bubble_crop
        self.assertEqual(cleaned_shapes, [(60, 110, 3)])

    def test_translate_page_image_survives_pipeline_init_error(self):
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        fallback_dlgs = [{"bbox": [20, 20, 100, 80], "text": "HELLO", "category": "speech_bubble"}]
        cleaned_shapes = []

        def spy_clean(crop, **kwargs):
            cleaned_shapes.append(crop.shape)
            return crop

        with patch.object(server_main, "get_manga_pipeline", side_effect=HTTPException(status_code=500, detail="no weights")), patch.object(
            server_main, "get_manga_cleaner", return_value=None
        ), patch(
            "manga_bubble_engine.extract_manga_dialogues", return_value=fallback_dlgs
        ), patch.object(
            server_main, "clean_bubble_crop", side_effect=spy_clean, create=True
        ), patch(
            "gemini_translator.translate_dialogues_with_gemini", return_value={"b_1": "SALOM"}
        ), patch(
            "ml.typesetting.render_translated_dialogues", side_effect=lambda im, dlgs: im
        ):
            resp = asyncio.run(
                server_main.translate_page_image(
                    file=DummyUploadFile(buf.getvalue()),
                    lang="en",
                    target_lang="uz",
                    _authorized=None,
                )
            )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(cleaned_shapes, [(60, 80, 3)])

    def test_detect_and_translate_skips_cleaning_filters_sfx_and_preserves_bg_color(self):
        # Cream bubble (245, 232, 199) with black text inside and black border outside
        img = Image.new("RGB", (200, 200), (245, 232, 199))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 150, 90], outline=(0, 0, 0), width=4)
        draw.text((30, 40), "HI", fill=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        captured_kwargs = {}

        def fake_process_page(pil_img, **kwargs):
            captured_kwargs.update(kwargs)
            return {
                "detected_source_lang": "en",
                "requires_ocr_fallback": False,
                "dialogues": [
                    {"bubble_id": 1, "block_id": "b_1", "bbox": [10, 10, 150, 90], "transcription": "HI", "category": "speech_bubble"},
                    {"bubble_id": 2, "block_id": "sfx_1", "bbox": [10, 110, 80, 160], "transcription": "BAM", "category": "sfx"},
                ],
            }

        fake_pipe = SimpleNamespace(process_page=fake_process_page)
        with patch.object(server_main, "get_manga_pipeline", return_value=fake_pipe), patch.object(
            server_main, "resolve_gemini_api_key", return_value="dummy-key"
        ), patch(
            "gemini_translator.translate_dialogues_with_gemini", return_value={"b_1": "SALOM"}
        ):
            res = asyncio.run(
                server_main.detect_and_translate(
                    file=DummyUploadFile(buf.getvalue()),
                    lang="fr",
                    target_lang="uz",
                    _authorized=None,
                )
            )

        self.assertTrue(captured_kwargs.get("skip_cleaning"))
        self.assertEqual(captured_kwargs.get("source_lang"), "en")
        self.assertEqual(res["source_lang"], "fr")
        self.assertEqual(res["blocks_count"], 1)
        self.assertEqual(res["blocks"][0]["bg_color"], "#F5E8C7")
        self.assertFalse(res["blocks"][0]["is_dark"])

    def test_decode_image_rgba_composites_onto_white_and_applies_exif(self):
        rgba = Image.new("RGBA", (40, 20), (0, 0, 0, 0))
        # Apply EXIF orientation 6 (90 deg CW -> transposed width=20, height=40)
        exif = rgba.getexif()
        exif[274] = 6
        buf = io.BytesIO()
        rgba.save(buf, format="PNG", exif=exif)
        decoded = server_main.decode_image(buf.getvalue())
        self.assertEqual(decoded.mode, "RGB")
        self.assertEqual(decoded.size, (20, 40))
        self.assertEqual(decoded.getpixel((5, 5)), (255, 255, 255))

    def test_routes_pdf_clears_stale_cleaned_filename_on_fallback_and_handles_latin_and_cropbox(self):
        # 1. Stale cleaned_image_filename must NOT leak when requires_ocr_fallback=True
        fake_pipe = SimpleNamespace(
            process_page=Mock(
                return_value={
                    "cleaned_image": Image.new("RGB", (300, 400), "white"),
                    "cleaned_bubble_ids": [1],
                    "requires_ocr_fallback": True,
                    "dialogues": [],
                }
            )
        )
        doc = pymupdf.open()
        page = doc.new_page(width=400, height=500)
        page.set_cropbox(pymupdf.Rect(50, 100, 350, 450))
        try:
            with patch.object(server_main, "get_manga_pipeline", return_value=fake_pipe), patch(
                "manga_bubble_engine.extract_manga_dialogues", return_value=[]
            ), patch.object(server_main, "run_paddle_ocr", return_value=[]), patch.object(
                server_main,
                "extract_paddle_lines",
                return_value=[{"left": 10, "top": 20, "right": 110, "bottom": 60, "text": "Salom", "confidence": 0.9}],
            ):
                res = routes_pdf.ocr_pdf_page(page, "uz", "comic", page_index=0, job_dir=self.root)
            self.assertIsNone(res["cleaned_image_filename"])
            self.assertFalse((self.root / "cleaned_page_0000.png").exists())
            self.assertEqual(res["detected_source_lang"], "uz")
            # Normalized CropBox coords fit inside page.rect (0..300, 0..350)
            self.assertGreaterEqual(res["blocks"][0]["bbox"][0], 0.0)
            self.assertLessEqual(res["blocks"][0]["bbox"][2], 300.0)

            # Non-zero rect.x0 / rect.y0 origin offsets must also be preserved
            offset_page = SimpleNamespace(
                rect=pymupdf.Rect(50, 100, 350, 450),
                rotation=0,
                get_pixmap=page.get_pixmap,
            )
            with patch.object(server_main, "get_manga_pipeline", return_value=fake_pipe), patch(
                "manga_bubble_engine.extract_manga_dialogues", return_value=[]
            ), patch.object(server_main, "run_paddle_ocr", return_value=[]), patch.object(
                server_main,
                "extract_paddle_lines",
                return_value=[{"left": 10, "top": 20, "right": 110, "bottom": 60, "text": "Salom", "confidence": 0.9}],
            ):
                off_res = routes_pdf.ocr_pdf_page(offset_page, "uz", "comic", page_index=0, job_dir=self.root)
            self.assertGreaterEqual(off_res["blocks"][0]["bbox"][0], 50.0)
            self.assertGreaterEqual(off_res["blocks"][0]["bbox"][1], 100.0)

            # 2. Primary path for PDF_LATIN_OCR_LANGS maps cleanly to 'en' without setting _manga_pipeline_error
            ok_pipe = SimpleNamespace(
                process_page=Mock(
                    return_value={
                        "cleaned_image": Image.new("RGB", (300, 400), "white"),
                        "cleaned_bubble_ids": [1],
                        "requires_ocr_fallback": False,
                        "detected_source_lang": "en",
                        "dialogues": [
                            {"bubble_id": 1, "bbox": [10, 20, 110, 60], "transcription": "Salom", "category": "speech_bubble"}
                        ],
                    }
                )
            )
            with patch.object(server_main, "get_manga_pipeline", return_value=ok_pipe):
                ok_res = routes_pdf.ocr_pdf_page(page, "uz", "comic", page_index=1, job_dir=self.root)
            self.assertEqual(ok_pipe.process_page.call_args.kwargs["source_lang"], "en")
            self.assertEqual(ok_res["detected_source_lang"], "uz")
            self.assertIsNone(server_main._manga_pipeline_error)
        finally:
            doc.close()

    def test_routes_pdf_verify_api_key_fails_closed(self):
        with patch.dict(os.environ, {"MANGA_SERVER_API_KEY": ""}, clear=False), patch.object(
            server_main, "SERVER_API_KEY", ""
        ), patch.object(server_main, "ALLOW_UNAUTHENTICATED", False):
            with self.assertRaises(HTTPException) as ctx:
                routes_pdf.verify_api_key(None)
            self.assertEqual(ctx.exception.status_code, 503)

    def test_routes_pdf_korean_is_not_rtl_and_zh_runs_vertical_grouping(self):
        img = Image.new("RGB", (400, 400), "white")
        ko_lines = [
            {"left": 20, "top": 50, "right": 100, "bottom": 80, "text": "안녕", "confidence": 0.95},
            {"left": 200, "top": 50, "right": 280, "bottom": 80, "text": "하세요", "confidence": 0.95},
        ]
        blocks = routes_pdf._group_comic_lines(ko_lines, img, None, scale=1.0, detected_lang="ko")
        self.assertEqual([b["text"] for b in blocks], ["안녕", "하세요"])

        doc = pymupdf.open()
        page = doc.new_page(width=300, height=400)
        try:
            with patch.object(server_main, "run_paddle_ocr", return_value=[]), patch.object(
                server_main, "extract_paddle_lines", return_value=[]
            ), patch.object(
                routes_pdf, "_group_vertical_manga_lines", return_value=[]
            ) as group_vert:
                routes_pdf.ocr_pdf_page(page, "zh", "book", page_index=0)
                group_vert.assert_not_called()
                # In comic mode, "zh" should invoke _group_vertical_manga_lines
                with patch.object(server_main, "get_manga_pipeline", side_effect=RuntimeError("skip")), patch(
                    "manga_bubble_engine.extract_manga_dialogues", return_value=[]
                ):
                    routes_pdf.ocr_pdf_page(page, "zh", "comic", page_index=0)
                group_vert.assert_called_once()
        finally:
            doc.close()

    def test_pdf_renderer_never_uses_uncleaned_patch_and_preserves_colors(self):
        source_path = self.root / "src.pdf"
        output_path = self.root / "out.pdf"
        with pymupdf.open() as src:
            page = src.new_page(width=300, height=300)
            # Draw dark text "RAW" inside a pastel cream rectangle on the source PDF
            page.draw_rect(pymupdf.Rect(40, 40, 220, 140), color=None, fill=(0.96, 0.93, 0.85))
            page.insert_text((70, 95), "RAW", fontsize=24, color=(0, 0, 0))
            src.save(source_path)

        # Save an UNCLEANED image (still containing black text "RAW" so p_std >= 5.0!)
        uncleaned = Image.new("RGB", (300, 300), (245, 237, 217))
        ImageDraw.Draw(uncleaned).text((70, 75), "RAW", fill=(0, 0, 0))
        uncleaned.save(self.root / "cleaned_stale.png")

        pdf_renderer._write_comic_pdf(
            source_path,
            output_path,
            [
                {
                    "cleaned_image_filename": "cleaned_stale.png",
                    "blocks": [
                        {
                            "original": "RAW",
                            "translated": "NEW",
                            "bbox": [40, 40, 220, 140],
                            "overlay_bbox": [40, 40, 220, 140],
                            "overlay_polygon": [[40, 40], [220, 40], [220, 140], [40, 140]],
                            "background": [0.96, 0.93, 0.85],
                            "category": "narration_box",
                            "cleaned": False,
                        }
                    ],
                }
            ],
            False,
        )
        with pymupdf.open(output_path) as out_doc:
            pix = out_doc[0].get_pixmap()
            # Corner inside narration box (away from "NEW" text) must retain pastel cream, NOT pure white (255,255,255)
            r, g, b = pix.pixel(50, 50)
            self.assertLess(b, 235)
            self.assertGreater(r, 230)
            self.assertIn("NEW", out_doc[0].get_text())


class ServerOcrRoutingAndJobTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.addCleanup(self.temp_dir.cleanup)

    def test_get_paddle_ocr_aliases_and_unknown_raises(self):
        fake_cls = Mock(side_effect=lambda **kw: SimpleNamespace(kw=kw))
        fake_mod = SimpleNamespace(PaddleOCR=fake_cls)
        with patch.object(server_main, "_paddle_ja", None), patch.object(
            server_main, "_paddle_ch", None
        ), patch.object(server_main, "_paddle_ko", None), patch.object(
            server_main.importlib, "import_module", return_value=fake_mod
        ):
            ja = server_main.get_paddle_ocr("jpn")
            zh = server_main.get_paddle_ocr("chi_sim")
            self.assertEqual(ja.kw["lang"], "japan")
            self.assertEqual(zh.kw["lang"], "ch")
            with self.assertRaises(ValueError):
                server_main.get_paddle_ocr("klingon_xx")

    def test_script_score_distinguishes_chinese_with_shared_punct_and_english(self):
        zh_text = "「你好，世界。」《英雄》"
        self.assertGreater(
            server_main._script_score(zh_text, 0.9, "zh"),
            server_main._script_score(zh_text, 0.9, "ja"),
        )
        ja_text = "「こんにちは、世界。」"
        self.assertGreater(
            server_main._script_score(ja_text, 0.9, "ja"),
            server_main._script_score(ja_text, 0.9, "zh"),
        )
        with patch.object(server_main, "run_paddle_ocr", return_value=[]), patch.object(
            server_main,
            "extract_paddle_lines",
            return_value=[{"text": "HELLO WORLD", "confidence": 0.95}],
        ):
            lang, _ = server_main.run_auto_paddle_ocr(b"img")
        self.assertEqual(lang, "en")

    def test_auto_paddle_ocr_not_hijacked_by_stray_japanese_sfx_on_english_page(self):
        def _fake_paddle(_img_bytes, model_lang):
            if model_lang == "japan":
                return [[([[0, 0], [40, 0], [40, 20], [0, 20]], ("ぎゅっ", 0.92))]]
            if model_lang == "latin":
                return [[([[10, 10], [200, 10], [200, 40], [10, 40]], ("I AM JUST GOING TO KEEP MAKING MISTAKES", 0.91))]]
            return [[]]

        with patch.object(server_main, "run_paddle_ocr", side_effect=_fake_paddle):
            lang, _ = server_main.run_auto_paddle_ocr(b"img")
        self.assertEqual(lang, "en")

    def test_dispatch_auto_without_paddle_and_translate_manga_preserves_explicit_ja(self):
        img = Image.new("RGB", (60, 30), "white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        fake_engine = SimpleNamespace(ocr_sess=True, ocr_bubble=Mock(return_value="HELLO"))
        with patch.object(server_main, "is_paddle_enabled", return_value=False), patch.object(
            server_main, "get_onnx_engine", return_value=fake_engine
        ):
            txt, detected = asyncio.run(server_main.dispatch_manga_ocr(buf.getvalue(), img, "auto"))
        self.assertEqual(txt, "HELLO")
        self.assertEqual(detected, "en")

        with patch.object(server_main, "resolve_gemini_api_key", return_value=None), patch.object(
            server_main, "translate_text", return_value="SALOM"
        ):
            res = asyncio.run(
                server_main.translate_manga(
                    file=None, text="テスト", lang="ja", target_lang="uz", _authorized=None
                )
            )
        self.assertEqual(res["detected_source_lang"], "ja")

    def test_pdf_jobs_resume_keeps_auto_per_page_and_no_deadlock(self):
        doc = pymupdf.open()
        p1 = doc.new_page(width=300, height=400)
        p1.insert_image(p1.rect, stream=self._png_bytes())
        p2 = doc.new_page(width=300, height=400)
        p2.insert_image(p2.rect, stream=self._png_bytes())
        src_pdf = self.root / "two.pdf"
        doc.save(src_pdf)
        doc.close()

        cache_path = self.root / "cache.json"
        cache_path.write_text(json.dumps({"0": {"blocks": [{"original": "a", "translated": "b", "is_heading": False}], "detected_source_lang": "en"}}), encoding="utf-8")
        seen_langs = []

        def fake_ocr(page, src_lang, doc_mode, **kwargs):
            seen_langs.append(src_lang)
            return {"blocks": [{"text": "こんにちは", "confidence": 0.9}], "detected_source_lang": "ja"}

        pdf_jobs.translate_pdf_file(
            input_path=src_pdf,
            output_path=self.root / "out.pdf",
            page_cache_path=cache_path,
            title="Test",
            source_lang="auto",
            target_lang="uz",
            output_mode="translated",
            translate_text=lambda t, tl, sl: f"UZ:{t}",
            progress_callback=lambda *_: None,
            cancel_event=threading.Event(),
            ocr_page=fake_ocr,
            document_mode="book",
        )
        self.assertEqual(seen_langs, ["auto"])

        # Stress concurrent _update and _get_job / cancel without ABBA deadlock
        mgr = pdf_jobs.PdfJobManager(self.root / "mgr")
        try:
            job = mgr.create_job(src_pdf.read_bytes(), "two.pdf", "en", "uz", "translated", 2)
            jid = job["job_id"]
            errors = []

            def updater():
                try:
                    for k in range(30):
                        mgr._update(jid, message=f"m{k}")
                except Exception as exc:
                    errors.append(exc)

            def reader():
                try:
                    for _ in range(30):
                        mgr.get_job(jid)
                except Exception as exc:
                    errors.append(exc)

            t1 = threading.Thread(target=updater)
            t2 = threading.Thread(target=reader)
            t1.start()
            t2.start()
            t1.join(timeout=5)
            t2.join(timeout=5)
            self.assertFalse(t1.is_alive() or t2.is_alive(), "Deadlock detected in PdfJobManager")
            self.assertEqual(errors, [])
        finally:
            mgr.shutdown()

    def _png_bytes(self) -> bytes:
        im = Image.new("RGB", (100, 100), "white")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()


class TranslatorAndParserTests(unittest.TestCase):
    def test_dialogue_cleaning_preserves_p_words_7_senpai_and_decades(self):
        for phrase in ("WHY STOP", "CAN YOU HELP", "ARE YOU UP", "DID YOU SLEEP", "HOW DEEP", "WHY 7", "SENPAI", "TSUMIKI", "1990s"):
            self.assertEqual(text_translator.clean_comic_dialogue_text(phrase), phrase)
            self.assertEqual(text_translator.repair_english_word_spacing(phrase), phrase)
            self.assertEqual(gemini_translator.clean_comic_dialogue_text(phrase), phrase)

        self.assertEqual(text_translator.clean_comic_dialogue_text("WHYP"), "WHY?")
        self.assertEqual(text_translator.clean_comic_dialogue_text("WHY P"), "WHY?")
        self.assertEqual(text_translator.clean_comic_dialogue_text("WHAT ARE YOU DOING /"), "WHAT ARE YOU DOING?")
        self.assertEqual(text_translator.normalize_ocr_language(""), "auto")
        self.assertEqual(text_translator.translation_language_code("zh-TW"), "zh-TW")

    def test_deepl_removes_hardcoded_mithiki_and_book_mode_skips_colloquialise(self):
        sample = "Он познакомился с народными традициями 1990s."
        self.assertEqual(deepl_translator.colloquialise_manga_text(sample, "ru"), sample)

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"translations": [{"text": "Осуществлять в соответствии с 1990s."}]}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = deepl_translator.translate_dialogues_with_deepl(
                ["Carry out according to 1990s."], "ru", api_key="k:fx", document_mode="book"
            )
        self.assertEqual(res, ["Осуществлять в соответствии с 1990s."])

    def test_failed_translations_never_return_original_text(self):
        with patch.dict(os.environ, {"DEEPL_API_KEY": "", "DEEPL_API_KEY_FILE": "", "MANGA_USE_MYMEMORY_FALLBACK": "false"}), patch.object(
            text_translator, "translate_text", return_value=None
        ):
            batch_res = text_translator.translate_batch(["Hello", "World"], "uz", "en")
        self.assertEqual(batch_res, [None, None])

        gemini_translator._gemini_rate_limited_until = 0.0
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError("u", 500, "err", {}, None)), patch.object(
            text_translator, "translate_text", return_value=None
        ):
            gem_res = gemini_translator.translate_dialogues_with_gemini(
                ["STOP", "RUN"], "uz", "en", api_key="key1"
            )
        self.assertIsNone(gem_res)

    def test_gemini_fences_unbracketed_keys_system_instruction_and_sfx(self):
        self.assertFalse(gemini_translator.is_sfx_word("Nooo!!"))
        self.assertFalse(gemini_translator.is_sfx_word("Whaaat?!"))
        self.assertTrue(gemini_translator.is_sfx_word("BOOM"))

        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            resp = MagicMock()
            resp.__enter__.return_value = resp
            resp.read.return_value = json.dumps(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "```json\n{\"b_1\": \"TO'XTA\", \"b_2\": \"YUGUR\"}\n```"
                                    }
                                ]
                            }
                        }
                    ]
                }
            ).encode("utf-8")
            return resp

        gemini_translator._gemini_rate_limited_until = 0.0
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            res = gemini_translator.translate_dialogues_with_gemini(
                ["STOP", "RUN"],
                "ar",
                "en",
                api_key="key_abc",
                block_ids=["b_1", "b_2"],
                return_dict=True,
                chapter_context="X" * 5000,
            )
        self.assertEqual(res, {"b_1": "TO'XTA", "b_2": "YUGUR"})
        self.assertIn("systemInstruction", captured["payload"])
        sys_txt = captured["payload"]["systemInstruction"]["parts"][0]["text"]
        self.assertIn("Arabic", sys_txt)
        self.assertLess(len(sys_txt), 8000)

    def test_mymemory_rejects_error_strings(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {
                "responseStatus": 200,
                "responseData": {"translatedText": "QUERY LENGTH LIMIT EXCEEDED. MAX ALLOWED QUERY : 500 CHARS"},
            }
        ).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch.dict(os.environ, {"MANGA_USE_MYMEMORY_FALLBACK": "true", "DEEPL_API_KEY": "", "DEEPL_API_KEY_FILE": ""}), patch.object(
            text_translator, "_google_blocked_until", time.monotonic() + 60.0
        ), patch("urllib.request.urlopen", return_value=mock_resp):
            res = text_translator.translate_text("Hello world", "uz", "en")
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()
