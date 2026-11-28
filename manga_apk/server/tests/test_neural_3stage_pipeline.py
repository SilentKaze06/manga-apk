"""
Unit/smoke tests for the 3-stage neural manga/manhua pipeline:
RT-DETR-v2 (blocks) + CTD UNet (character mask only) + 2-path / Manga-LaMa (inpainting).
"""
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

from ml.bubble_rtdetr import BubbleRTDetr
from ml.comic_detector import ComicTextDetector
from ml.cleaning import clean_block_with_mask, clean_page_blocks
from ml.detection import deduplicate_text_detections, sort_detections_reading_order
from ml.manga_pipeline import MangaTranslationPipeline


class NeuralThreeStagePipelineTests(unittest.TestCase):
    def test_multiline_dialogue_in_single_bubble_stays_one_block_and_empty_bubble_ignored(self):
        """
        1. Long multi-line dialogue inside one speech bubble stays as 1 unified block
           (not split into multiple fragments), and empty class-0 `bubble` is not sent to OCR.
        """
        img = Image.new("RGB", (400, 400), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Bubble 1 with long multi-line dialogue
        draw.ellipse([40, 40, 260, 260], outline=(0, 0, 0), width=3)
        draw.text((70, 80), "THIS IS A VERY LONG\nMULTI-LINE DIALOGUE\nINSIDE ONE BUBBLE!", fill=(0, 0, 0))
        # Empty bubble 2 (no text_bubble inside)
        draw.ellipse([280, 280, 380, 380], outline=(0, 0, 0), width=2)

        rtdetr_dets = [
            {"box": [40, 40, 260, 260], "score": 0.96, "label": "bubble"},
            {"box": [65, 75, 235, 220], "score": 0.94, "label": "text_bubble"},
            {"box": [280, 280, 380, 380], "score": 0.91, "label": "bubble"},  # empty bubble
        ]
        mock_rtdetr = SimpleNamespace(is_available=True, detect=Mock(return_value=rtdetr_dets))

        # Mock CTD UNet returning mask pixels both inside text_bubble AND spurious artwork noise outside
        import torch
        fake_mask_logits = torch.zeros((1, 1, 1024, 1024), dtype=torch.float32)
        # Put mask inside the text_bubble region and also spurious noise at bottom-left artwork
        fake_mask_logits[0, 0, 200:500, 180:580] = 1.0
        fake_mask_logits[0, 0, 800:900, 20:100] = 1.0
        fake_net = Mock(return_value=(torch.zeros((1, 1, 7)), fake_mask_logits, torch.zeros((1, 2, 1024, 1024))))

        det = ComicTextDetector(device="cpu", half=False)
        with patch("ml.comic_detector.get_shared_bubble_rtdetr", return_value=mock_rtdetr), patch.object(
            det, "load_model", return_value=fake_net
        ):
            blocks, page_mask = det.detect(img)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["label"], "text_bubble")
        self.assertEqual(blocks[0]["text_box"], [65, 75, 235, 220])
        self.assertEqual(blocks[0]["bubble_box"], [40, 40, 260, 260])
        # Spurious CTD mask outside text_box must be zeroed by intersection
        self.assertEqual(int(page_mask[350, 20]), 0)
        self.assertGreater(int(page_mask[140, 150]), 0)

    def test_textless_page_produces_zero_blocks_and_untouched_image(self):
        """
        2. Textless pages produce 0 text blocks and preserve the original image 100%.
        """
        img = Image.new("RGB", (300, 400), (180, 200, 220))
        ImageDraw.Draw(img).rectangle([40, 40, 260, 360], fill=(90, 120, 160))

        mock_rtdetr = SimpleNamespace(is_available=True, detect=Mock(return_value=[]))
        det = ComicTextDetector(device="cpu", half=False)
        with patch("ml.comic_detector.get_shared_bubble_rtdetr", return_value=mock_rtdetr):
            blocks, page_mask = det.detect(img)

        self.assertEqual(blocks, [])
        self.assertEqual(int(np.sum(page_mask)), 0)

        pipe = MangaTranslationPipeline.__new__(MangaTranslationPipeline)
        pipe.model_lock = threading.RLock()
        pipe.cleaner = None
        pipe.comic_detector = det
        pipe.qwen_engine = Mock()

        with patch("ml.comic_detector.get_shared_bubble_rtdetr", return_value=mock_rtdetr):
            res = pipe.process_page(img, source_lang="auto", target_lang="uz")

        self.assertEqual(res["dialogues"], [])
        self.assertTrue(np.array_equal(np.asarray(res["cleaned_image"]), np.asarray(img)))
        pipe.qwen_engine.ocr_and_translate_crops.assert_not_called()

    def test_colored_gradient_manhua_routes_through_lama_without_rect_artifacts(self):
        """
        3. Colored/gradient manhua crops (`text_free` and colored `text_bubble`) route through
           Manga-LaMa without rectangular patch artifacts, while uniform solid `text_bubble`
           uses fast median fill and unmasked artwork stays 100% untouched.
        """
        h, w = 240, 320
        page_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        # Left half: smooth gradient manhua background (y-gradient red->blue)
        for y in range(h):
            page_rgb[y, :160] = [int(255 * y / h), 80, int(255 * (1 - y / h))]
        # Right half: uniform cream speech bubble [180, 30, 300, 120]
        page_rgb[30:120, 180:300] = [245, 240, 225]

        # Draw text strokes in both regions
        page_mask = np.zeros((h, w), dtype=np.uint8)
        # text_free on gradient artwork
        page_rgb[60:90, 40:120] = [255, 255, 255]
        page_mask[62:88, 42:118] = 255
        # uniform text_bubble
        page_rgb[55:95, 200:280] = [15, 15, 15]
        page_mask[57:93, 202:278] = 255

        blocks = [
            {
                "block_id": "b_1",
                "label": "text_free",
                "category": "free_text",
                "text_box": [38, 58, 122, 92],
                "box": [38, 58, 122, 92],
            },
            {
                "block_id": "b_2",
                "label": "text_bubble",
                "category": "speech_bubble",
                "text_box": [198, 53, 282, 97],
                "box": [180, 30, 300, 120],
                "bubble_box": [180, 30, 300, 120],
            },
        ]

        lama_calls = []

        def _fake_lama(img_pil, mask_pil):
            lama_calls.append((img_pil.size, np.any(np.asarray(mask_pil) > 0)))
            arr = np.asarray(img_pil).copy()
            # Assert aligned to multiple of 8
            self.assertEqual(arr.shape[0] % 8, 0)
            self.assertEqual(arr.shape[1] % 8, 0)
            # Fill masked pixels smoothly from surrounding gradient
            m = np.asarray(mask_pil) > 0
            arr[m] = [120, 80, 135]
            return Image.fromarray(arr)

        cleaned, total_mask = clean_page_blocks(page_rgb, blocks, page_mask, lama=_fake_lama)

        # Only text_free should have called LaMa (uniform solid text_bubble took fast path)
        self.assertEqual(len(lama_calls), 1)
        self.assertEqual(blocks[0]["clean_path"], "manga_lama")
        self.assertEqual(blocks[1]["clean_path"], "fast_solid_fill")
        # Uniform bubble cleaned to cream color
        self.assertTrue(np.allclose(cleaned[75, 240], [245, 240, 225], atol=3))
        # Unmasked artwork far from text boxes is 100% untouched
        self.assertTrue(np.array_equal(cleaned[180:230, 10:150], page_rgb[180:230, 10:150]))

    def test_tall_webtoon_vertical_slicing_and_reading_order(self):
        """
        Tall pages (h/w > 2.0) slice vertically with overlap, map coordinates back,
        and deduplicate across slice seams. Reading order respects RTL vs LTR.
        """
        tall_img = Image.new("RGB", (400, 1800), (255, 255, 255))
        det = BubbleRTDetr(device="cpu", half=False)
        det.model = object()
        det.processor = object()

        slice_calls = []

        def _fake_detect_single(slice_rgb):
            slice_calls.append(slice_rgb.size)
            return [
                {"box": [50, 100, 200, 180], "score": 0.92, "label": "text_bubble"},
            ]

        with patch.object(det, "_detect_single", side_effect=_fake_detect_single):
            out = det.detect(tall_img)

        self.assertGreaterEqual(len(slice_calls), 2)
        self.assertGreaterEqual(len(out), 2)

        # Reading order check: RTL (right first) vs LTR (left first) on same row
        row_blocks = [
            {"box": [20, 20, 120, 80], "text_box": [20, 20, 120, 80], "label": "text_bubble"},
            {"box": [220, 20, 320, 80], "text_box": [220, 20, 320, 80], "label": "text_bubble"},
        ]
        rtl_sorted = sort_detections_reading_order(row_blocks, reading_direction="rtl")
        ltr_sorted = sort_detections_reading_order(row_blocks, reading_direction="ltr")
        self.assertEqual(rtl_sorted[0]["box"][0], 220)
        self.assertEqual(ltr_sorted[0]["box"][0], 20)

        # Class-agnostic deduplication removes overlapping text_bubble & text_free
        dups = [
            {"box": [50, 50, 150, 120], "score": 0.90, "label": "text_bubble"},
            {"box": [52, 52, 148, 118], "score": 0.85, "label": "text_free"},
        ]
        deduped = deduplicate_text_detections(dups)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["label"], "text_bubble")


if __name__ == "__main__":
    unittest.main()
