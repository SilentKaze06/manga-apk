"""Regression tests for Stage 2: Core ML Pipeline Bug Fixes."""
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ml.detection import split_joined_bubble_lobes
from ml.geometry import has_pinched_bubble_waist
from ml.manga_pipeline import MangaTranslationPipeline
from ml.manga_text_cleaner import MangaTextCleaner
from ml.ocr_engine import is_manga_punctuation, is_sfx_text, normalize_cyrillic_ocr
from ml.qwen_vl_engine import clean_qwen_output_text
from ml.typesetting import render_translated_dialogues


class Stage2CoreFixesTests(unittest.TestCase):
    def test_typesetting_shock_bubble_1_2_3_lines_no_crash(self):
        # Shock bubble: tall narrow ellipse (bh >= 2.2 * bw and bw <= 110)
        for idx, txt in enumerate([
            "WHOA!",
            "HOLD ON!",
            "WHAT ARE YOU DOING?!",
        ], start=1):
            img = Image.new("RGB", (220, 320), (255, 255, 255))
            dlgs = [{
                "bbox": [60, 20, 150, 260],
                "translation": txt,
                "category": "speech_bubble",
                "is_dark": False,
            }]
            out = render_translated_dialogues(img, dlgs)
            self.assertIsInstance(out, Image.Image)
            # Verify pixels were actually drawn inside the shock bubble
            diff = np.abs(np.asarray(out, dtype=np.int16) - np.asarray(img, dtype=np.int16))
            self.assertGreater(int(np.sum(diff > 10)), 20, f"Shock bubble case {idx} drew nothing")

    def test_has_pinched_bubble_waist_bottom_half_1600px_and_touching_gap_zero(self):
        gray = np.full((1600, 500), 255, dtype=np.uint8)
        # Draw a figure-8 pinched double bubble at y > 1000
        cv2.circle(gray, (150, 1200), 80, 0, -1)
        cv2.circle(gray, (310, 1200), 80, 0, -1)
        cv2.rectangle(gray, (190, 1175), (270, 1225), 0, -1)
        # Clear interior white ovals and waist bridge
        cv2.circle(gray, (150, 1200), 75, 255, -1)
        cv2.circle(gray, (310, 1200), 75, 255, -1)
        cv2.rectangle(gray, (190, 1178), (270, 1222), 255, -1)

        # Touching split-half probes (gap_x == 0) in bottom half (y > 1000)
        box_l = [100, 1150, 230, 1250]
        box_r = [230, 1150, 360, 1250]
        self.assertTrue(has_pinched_bubble_waist(gray, box_l, box_r))

    def test_split_joined_bubble_lobes_divider_line_and_panel_gutter(self):
        # 1. Two touching bubbles separated by a thin black divider line
        gray1 = np.full((160, 260), 255, dtype=np.uint8)
        cv2.rectangle(gray1, (10, 20), (250, 140), 0, 3)
        cv2.line(gray1, (130, 20), (130, 140), 0, 3)
        lobes1 = split_joined_bubble_lobes(gray1, [10, 20, 250, 140])
        self.assertGreaterEqual(len(lobes1), 2)

        # 2. Two bubbles across a thick black panel gutter (>= 75% of bh)
        gray2 = np.full((160, 260), 255, dtype=np.uint8)
        cv2.line(gray2, (130, 0), (130, 160), 0, 6)
        lobes2 = split_joined_bubble_lobes(gray2, [10, 20, 250, 140])
        self.assertEqual(len(lobes2), 2)
        self.assertLess(lobes2[0][1][2], lobes2[1][1][2])

    def test_manga_text_cleaner_textured_dark_artwork_routes_to_lama(self):
        # Create textured dark artwork (std_bg >= 16, median <= 85) with white text
        rng = np.random.default_rng(42)
        bg = rng.integers(15, 95, size=(120, 160), dtype=np.uint8)
        img_np = np.stack([bg, bg, bg], axis=-1)
        cv2.putText(img_np, "HI", (55, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (250, 250, 250), 3)
        pil_img = Image.fromarray(img_np)

        prob = np.zeros((120, 160), dtype=np.float32)
        prob[45:80, 50:105] = 0.9

        cleaner = MangaTextCleaner.__new__(MangaTextCleaner)
        cleaner.model = object()
        cleaner.device = "cpu"

        lama_called = []

        def fake_lama(im, m):
            lama_called.append(True)
            return Image.new("RGB", im.size, (40, 40, 40))

        with patch.object(cleaner, "predict_mask", return_value=prob), patch(
            "ml.manga_text_cleaner.get_shared_simple_lama", return_value=fake_lama
        ):
            cleaned_img, _, _ = cleaner.clean_page(pil_img, threshold=0.5)

        self.assertTrue(lama_called, "Textured dark artwork should route to SimpleLaMa, not flat black fill")
        self.assertEqual(cleaned_img.size, (160, 120))

    def test_clean_qwen_output_text_preserves_em_dash_spaces(self):
        self.assertEqual(clean_qwen_output_text("Wait— what?"), "Wait- what?")
        self.assertEqual(clean_qwen_output_text("com-\nfort"), "comfort")

    def test_ocr_engine_sfx_punctuation_and_cyrillic_ordinals(self):
        self.assertFalse(is_sfx_text("Shh"))
        self.assertFalse(is_sfx_text("Snap out!"))
        self.assertFalse(is_sfx_text("Don't sigh"))
        self.assertFalse(is_sfx_text("ЧТООО?!"))
        self.assertFalse(is_sfx_text("RUNNN!"))
        self.assertFalse(is_sfx_text("YO'OOOQ!"))
        self.assertTrue(is_sfx_text("VREEEEM!"))

        self.assertFalse(is_manga_punctuation("I?!"))
        self.assertFalse(is_manga_punctuation("I!"))
        self.assertFalse(is_manga_punctuation("U?!"))
        self.assertFalse(is_manga_punctuation("何?!"))
        self.assertFalse(is_manga_punctuation("뭐?!"))
        self.assertFalse(is_manga_punctuation("Я?!"))
        self.assertTrue(is_manga_punctuation("?!"))

        self.assertEqual(normalize_cyrillic_ocr("30-го числа"), "30-го числа")

    def test_pipeline_excludes_untrusted_garbage_from_cleanable_records(self):
        img = Image.new("RGB", (300, 200), (255, 255, 255))
        pipe = MangaTranslationPipeline.__new__(MangaTranslationPipeline)
        pipe.model_lock = threading.RLock()
        pipe.cleaner = None
        pipe.comic_detector = SimpleNamespace(
            detect=Mock(return_value=(
                [
                    {"block_id": "b_1", "box": [10, 10, 120, 80], "text_box": [20, 20, 110, 70], "category": "speech_bubble"},
                    {"block_id": "b_2", "box": [150, 10, 280, 80], "text_box": [160, 20, 270, 70], "category": "speech_bubble"},
                ],
                None,
            ))
        )
        # First crop is valid dialogue, second crop is garbage noise
        pipe.qwen_engine = SimpleNamespace(
            ocr_and_translate_crops=Mock(return_value=[("Hello there!", "Salom!"), ("qq", "")]),
            ocr_crops=Mock(return_value=["Hello there!", "qq"]),
        )

        res = pipe.process_page(img, source_lang="en", target_lang="uz")
        by_id = {d["block_id"]: d for d in res["dialogues"]}
        self.assertTrue(by_id["b_1"]["ocr_trusted"])
        self.assertTrue(by_id["b_1"]["cleaned"])
        self.assertFalse(by_id["b_2"]["ocr_trusted"])
        self.assertFalse(by_id["b_2"]["cleaned"])
