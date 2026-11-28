"""Unit tests for the comic-text-detector + Qwen3.5-9B pipeline refactor."""
import asyncio
import importlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

from ml.comic_detector import ComicTextDetector, YoloV5Backbone, UnetHead, DBHead, TextDetNet
from ml.manga_pipeline import MangaTranslationPipeline
from ml.qwen_vl_engine import (
    QwenVLEngine,
    clean_qwen_output_text,
    parse_ocr_translate_json,
)
import gemini_translator


class _DummyUploadFile:
    def __init__(self, data: bytes, filename: str = "page.png"):
        self.data = data
        self.filename = filename

    async def read(self, size: int = -1):
        return self.data


class LegacyOcrRemovalImportTests(unittest.TestCase):
    def test_server_imports_cleanly_without_easyocr_paddle_or_manga_ocr(self):
        poisoned = {
            "easyocr": None,
            "paddleocr": None,
            "paddle": None,
            "paddle.inference": None,
            "manga_ocr": None,
        }
        with patch.dict(sys.modules, poisoned):
            for mod_name in (
                "ml.comic_detector",
                "ml.qwen_vl_engine",
                "ml.manga_pipeline",
                "gemini_translator",
                "routes_pdf",
                "main",
            ):
                mod = importlib.import_module(mod_name)
                self.assertIsNotNone(mod)

            import main as server_main
            # Health check reflects new stack and never requires legacy packages
            health = server_main.health(x_api_key=None)
            self.assertEqual(health["status"], "healthy")


class QwenJsonAndMarkdownTests(unittest.TestCase):
    def test_clean_qwen_output_text_strips_markdown_and_thinking(self):
        raw = "<think>Let me read this carefully.</think>\n```json\n_Salom_, *o‘zim* `akajon`—ket-\ndik!\n```"
        cleaned = clean_qwen_output_text(raw, target_lang="uz")
        self.assertNotIn("<think>", cleaned)
        self.assertNotIn("_", cleaned)
        self.assertNotIn("*", cleaned)
        self.assertNotIn("`", cleaned)
        self.assertIn("o'zim", cleaned)
        self.assertIn("ketdik!", cleaned)

    def test_parse_ocr_translate_json_handles_valid_fenced_and_malformed_json(self):
        # 1. Valid fenced JSON with markdown artifacts inside values
        raw1 = '```json\n{"src": "WHOAAA!!", "tgt": "_O‘xshadi_ *ajoyib*!!"}\n```'
        src1, tgt1 = parse_ocr_translate_json(raw1, target_lang="uz")
        self.assertEqual(src1, "WHOAAA!!")
        self.assertEqual(tgt1, "O'xshadi ajoyib!!")

        # 2. Thinking block + extra prose around JSON
        raw2 = '<think>reasoning...</think> Here is the result: {"src": "LOOK OUT!", "tgt": "ОСТОРОЖНО!"} Done.'
        src2, tgt2 = parse_ocr_translate_json(raw2, target_lang="ru")
        self.assertEqual(src2, "LOOK OUT!")
        self.assertEqual(tgt2, "ОСТОРОЖНО!")

        # 3. Empty SFX response
        src3, tgt3 = parse_ocr_translate_json('{"src": "", "tgt": ""}', target_lang="uz")
        self.assertEqual((src3, tgt3), ("", ""))

        # 4. Regex fallback when trailing brace is missing
        raw4 = '{"src": "WAIT FOR ME", "tgt": "MENI KUT"'
        src4, tgt4 = parse_ocr_translate_json(raw4, target_lang="uz")
        self.assertEqual(src4, "WAIT FOR ME")
        self.assertEqual(tgt4, "MENI KUT")


class ComicDetectorAndPipelineSchemaTests(unittest.TestCase):
    def test_ctd_network_forward_shape_cpu(self):
        import torch

        blk_det = YoloV5Backbone()
        blk_det.out_indices = [1, 3, 5, 7, 9]
        text_seg = UnetHead(act="leaky")
        text_det = DBHead(64, act="leaky")
        net = TextDetNet(blk_det, text_seg, text_det).eval()

        x = torch.zeros(1, 3, 256, 256)
        with torch.inference_mode():
            blks, mask, lines = net(x)
        self.assertEqual(blks.ndim, 3)
        self.assertEqual(blks.shape[-1], 7)  # 4 box + 1 obj + 2 classes
        self.assertEqual(tuple(mask.shape), (1, 1, 256, 256))
        self.assertEqual(tuple(lines.shape), (1, 2, 256, 256))

    def test_pipeline_uses_ctd_and_qwen_single_pass_and_unions_unet_mask(self):
        img = Image.new("RGB", (320, 240), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.ellipse([20, 20, 140, 100], outline=(0, 0, 0), width=3)
        draw.rectangle([170, 20, 300, 100], fill=(20, 20, 20), outline=(0, 0, 0), width=2)
        draw.text((185, 50), "DARK BOX", fill=(255, 255, 255))

        ctd_blocks = [
            {
                "block_id": "b_1",
                "box": [20, 20, 140, 100],
                "text_box": [40, 40, 120, 80],
                "category": "speech_bubble",
                "is_dark": False,
                "confidence": 0.94,
            },
            {
                "block_id": "b_2",
                "box": [170, 20, 300, 100],
                "text_box": [180, 40, 290, 80],
                "category": "narration",
                "is_dark": True,
                "confidence": 0.92,
            },
            {
                "block_id": "b_3",
                "box": [40, 140, 160, 200],
                "text_box": [45, 145, 155, 195],
                "category": "free_text",
                "is_dark": False,
                "confidence": 0.88,
            },
        ]
        ctd_mask = np.zeros((240, 320), dtype=np.uint8)
        ctd_mask[45:75, 185:285] = 255  # UNet stroke mask inside dark narration box

        mock_ctd = SimpleNamespace(is_available=True, detect=Mock(return_value=(ctd_blocks, ctd_mask)))
        mock_qwen = SimpleNamespace(
            is_available=True,
            ocr_and_translate_crops=Mock(
                return_value=[
                    ("HELLO!", "SALOM!"),
                    ("DARK BOX", "QORA QUTI"),
                    ("Whoaaa!!", "Vooooy!!"),
                ]
            ),
        )

        pipe = MangaTranslationPipeline.__new__(MangaTranslationPipeline)
        import threading
        pipe.model_lock = threading.RLock()
        pipe.comic_detector = mock_ctd
        pipe.qwen_engine = mock_qwen
        pipe.cleaner = None
        pipe.detector = None
        pipe.ocr = None
        pipe.inpainter = None

        res = pipe.process_page(img, source_lang="auto", target_lang="uz", reading_direction="ltr", skip_cleaning=False)
        self.assertFalse(res["requires_ocr_fallback"])
        self.assertEqual(res.get("translator_used"), "qwen3.5-9b")
        self.assertEqual(len(res["dialogues"]), 3)
        self.assertEqual([d["category"] for d in res["dialogues"]], ["speech_bubble", "narration", "free_text"])
        self.assertEqual([d["translation"] for d in res["dialogues"]], ["SALOM!", "QORA QUTI", "Vooooy!!"])

        # UNet mask inside dark narration box must be unioned into mask_image
        mask_arr = np.asarray(res["mask_image"])
        self.assertEqual(int(mask_arr[60, 220]), 255)

    def test_textless_page_returns_empty_dialogues_and_preserves_image(self):
        img = Image.new("RGB", (240, 180), (210, 225, 240))
        ImageDraw.Draw(img).rectangle([30, 30, 210, 150], fill=(120, 160, 200))

        pipe = MangaTranslationPipeline.__new__(MangaTranslationPipeline)
        pipe.model_lock = __import__("threading").RLock()
        pipe.cleaner = None
        pipe.comic_detector = SimpleNamespace(detect=Mock(return_value=([], None)))
        pipe.qwen_engine = Mock()

        res = pipe.process_page(img, source_lang="en", target_lang="uz")
        self.assertEqual(res["dialogues"], [])
        self.assertFalse(res["requires_ocr_fallback"])
        self.assertTrue(np.array_equal(np.asarray(res["cleaned_image"]), np.asarray(img)))
        pipe.qwen_engine.ocr_and_translate_crops.assert_not_called()


class ApiRoutingAndGeminiTests(unittest.TestCase):
    def test_detect_and_translate_defaults_to_qwen_without_calling_gemini(self):
        import main as server_main

        img = Image.new("RGB", (200, 200), (255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        fake_pipe = SimpleNamespace(
            process_page=Mock(
                return_value={
                    "detected_source_lang": "en",
                    "requires_ocr_fallback": False,
                    "translator_used": "qwen3.5-9b",
                    "dialogues": [
                        {
                            "bubble_id": 1,
                            "block_id": "b_1",
                            "bbox": [10, 10, 120, 80],
                            "transcription": "GO!",
                            "translation": "KETDIK!",
                            "category": "speech_bubble",
                            "is_dark": False,
                        }
                    ],
                }
            )
        )

        # Even if GEMINI_API_KEY exists in env, default call (no explicit request) uses Qwen3.5-9B with 0 Gemini calls
        with patch.object(server_main, "get_manga_pipeline", return_value=fake_pipe), patch.object(
            server_main, "resolve_gemini_api_key", return_value="env-key-should-not-fire"
        ), patch(
            "gemini_translator.translate_dialogues_with_gemini",
            side_effect=AssertionError("Gemini must not be called when Qwen3.5-9B handles default translation"),
        ):
            res = asyncio.run(
                server_main.detect_and_translate(
                    file=_DummyUploadFile(buf.getvalue()),
                    lang="auto",
                    target_lang="uz",
                    _authorized=None,
                )
            )
        self.assertEqual(res["translator_used"], "qwen3.5-9b")
        self.assertEqual(res["blocks"][0]["translated_text"], "KETDIK!")
        self.assertNotIn("warning", res)

    def test_detect_and_translate_uses_gemini_only_when_explicitly_requested(self):
        import main as server_main

        img = Image.new("RGB", (200, 200), (255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        fake_pipe = SimpleNamespace(
            process_page=Mock(
                return_value={
                    "detected_source_lang": "en",
                    "requires_ocr_fallback": False,
                    "dialogues": [
                        {
                            "bubble_id": 1,
                            "block_id": "b_1",
                            "bbox": [10, 10, 120, 80],
                            "transcription": "SO?",
                            "category": "speech_bubble",
                        }
                    ],
                }
            )
        )

        with patch.object(server_main, "get_manga_pipeline", return_value=fake_pipe), patch(
            "gemini_translator.translate_dialogues_with_gemini",
            return_value={"b_1": "XO'SH?"},
        ) as mock_gem:
            res = asyncio.run(
                server_main.detect_and_translate(
                    file=_DummyUploadFile(buf.getvalue()),
                    lang="en",
                    target_lang="uz",
                    gemini_api_key="explicit-key",
                    use_gemini=True,
                    _authorized=None,
                )
            )
        self.assertEqual(res["translator_used"], "gemini")
        self.assertEqual(res["blocks"][0]["translated_text"], "XO'SH?")
        mock_gem.assert_called_once()
        # Verify OCR-only target_lang=None was passed to pipeline
        self.assertIsNone(fake_pipe.process_page.call_args.kwargs.get("target_lang"))

    def test_gemini_prompt_is_unordered_and_strips_markdown(self):
        from unittest.mock import MagicMock
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
                                "parts": [{"text": json.dumps({"[b_1]": "_Salom_ *dunyo*!"})}]
                            }
                        }
                    ]
                }
            ).encode("utf-8")
            return resp

        gemini_translator._gemini_rate_limited_until = 0.0
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = gemini_translator.translate_dialogues_with_gemini(
                ["Hello world!"], "uz", "en", api_key="k123", block_ids=["b_1"], return_dict=True
            )
        sys_prompt = captured["payload"]["systemInstruction"]["parts"][0]["text"]
        self.assertIn("unordered comic dialogue bubbles", sys_prompt)
        self.assertEqual(out["b_1"], "Salom dunyo!")

    def test_ctd_deduplicates_nested_blocks_but_keeps_pinched_lobes(self):
        from ml.comic_detector import _deduplicate_blocks

        # Nested duplicate (like Page 5 b_7 inside b_3) + near-identical overlap (Page 2 b_4 & b_9)
        # alongside two adjacent pinched lobes (Page 6 b_2 & b_3).
        raw_blocks = [
            {
                "block_id": "b_1",
                "text_box": [195, 901, 250, 976],
                "box": [195, 901, 250, 976],
                "category": "speech_bubble",
                "is_dark": False,
                "confidence": 0.65,
            },
            {
                "block_id": "b_2",
                "text_box": [194, 896, 403, 1007],
                "box": [194, 896, 403, 1007],
                "category": "speech_bubble",
                "is_dark": False,
                "confidence": 0.90,
            },
            {
                "block_id": "b_3",
                "text_box": [663, 663, 721, 737],
                "box": [663, 663, 721, 737],
                "category": "speech_bubble",
                "is_dark": False,
                "confidence": 0.88,
            },
            {
                "block_id": "b_4",
                "text_box": [666, 663, 720, 744],
                "box": [666, 663, 720, 744],
                "category": "speech_bubble",
                "is_dark": False,
                "confidence": 0.72,
            },
            {
                "block_id": "b_5",
                "text_box": [632, 406, 744, 529],
                "box": [632, 406, 744, 529],
                "category": "speech_bubble",
                "is_dark": False,
                "confidence": 0.92,
            },
            {
                "block_id": "b_6",
                "text_box": [420, 411, 505, 503],
                "box": [420, 411, 505, 503],
                "category": "speech_bubble",
                "is_dark": False,
                "confidence": 0.91,
            },
        ]
        deduped = _deduplicate_blocks(raw_blocks)
        self.assertEqual(len(deduped), 4)
        self.assertEqual([d["block_id"] for d in deduped], ["b_1", "b_2", "b_3", "b_4"])

    def test_all_server_modules_resolve_typing_annotations_without_nameerror(self):
        import ast

        server_dir = Path(__file__).resolve().parents[1]
        for py_path in list(server_dir.glob("*.py")) + list((server_dir / "ml").glob("*.py")):
            src = py_path.read_text(encoding="utf-8")
            tree = ast.parse(src, str(py_path))
            for sym in ("Optional", "List", "Dict", "Tuple", "Any", "Union"):
                used = any(
                    isinstance(node, ast.Name) and node.id == sym
                    for node in ast.walk(tree)
                )
                if used:
                    imported = any(
                        isinstance(node, ast.ImportFrom)
                        and any((a.asname or a.name) == sym for a in node.names)
                        for node in ast.walk(tree)
                    )
                    self.assertTrue(
                        imported,
                        f"{py_path.name} references {sym} without importing it from typing (fails on Python 3.11)",
                    )

    def test_classify_dark_text_block_rejects_white_bubble_with_black_text(self):
        from ml.comic_detector import _classify_dark_text_block

        white_bubble = np.full((120, 120), 255, dtype=np.uint8)
        # Black text strokes inside white bubble (~20% pixels)
        white_bubble[30:90, 25:95:3] = 15
        self.assertFalse(_classify_dark_text_block(white_bubble, [20, 25, 100, 95]))

        dark_box = np.full((120, 120), 25, dtype=np.uint8)
        # White text strokes inside dark narration box (~18% pixels)
        dark_box[35:85, 30:90:4] = 245
        self.assertTrue(_classify_dark_text_block(dark_box, [20, 25, 100, 95]))


if __name__ == "__main__":
    unittest.main()
