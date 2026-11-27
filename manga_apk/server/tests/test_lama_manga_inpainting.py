"""Unit tests for anime-manga-big-lama.pt resolution, dark-box masking, and UNet extra_mask union."""
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ml import manga_text_cleaner
from ml.manga_pipeline import _deduplicate_overlapping_detections


class LamaMangaInpaintingTests(unittest.TestCase):
    def test_lama_model_resolution_prefers_manga_weights_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manga_pt = Path(tmpdir) / "anime-manga-big-lama.pt"
            manga_pt.write_bytes(b"fake-torchscript-checkpoint")
            default_pt = Path(tmpdir) / "big-lama.pt"
            default_pt.write_bytes(b"generic-places2")

            with patch.dict(
                os.environ,
                {
                    "MANGA_LAMA_WEIGHTS": str(manga_pt),
                    "LAMA_MODEL": str(default_pt),
                },
                clear=False,
            ):
                resolved = manga_text_cleaner.resolve_lama_model_path()
                self.assertEqual(resolved, str(manga_pt))

                # Also verify get_shared_simple_lama propagates resolved path to LAMA_MODEL
                captured = {}

                class _FakeSimpleLama:
                    def __init__(self, device=None):
                        captured["lama_model_env"] = os.environ.get("LAMA_MODEL")
                        captured["device"] = device

                fake_mod = types.ModuleType("simple_lama_inpainting")
                fake_mod.SimpleLama = _FakeSimpleLama

                orig_shared = manga_text_cleaner._shared_simple_lama
                orig_failed = manga_text_cleaner._shared_simple_lama_failed
                try:
                    manga_text_cleaner._shared_simple_lama = None
                    manga_text_cleaner._shared_simple_lama_failed = False
                    with patch.dict(sys.modules, {"simple_lama_inpainting": fake_mod}):
                        inst = manga_text_cleaner.get_shared_simple_lama(device="cpu")
                    self.assertIsNotNone(inst)
                    self.assertEqual(captured.get("lama_model_env"), str(manga_pt))
                finally:
                    manga_text_cleaner._shared_simple_lama = orig_shared
                    manga_text_cleaner._shared_simple_lama_failed = orig_failed

    def test_dark_box_mask_covers_bright_white_pixels_without_smearing(self):
        # Create a 200x120 image with a dark narration box (< 40) and bright white text (> 200)
        img = Image.new("RGB", (200, 120), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 180, 100], fill=(28, 28, 32))
        draw.text((40, 48), "WHY DID I...", fill=(245, 245, 245))
        img_np = np.array(img)

        # Partial extra_mask that misses part of the white letters; dark-box logic must still catch all > 180 pixels
        extra_mask = np.zeros((120, 200), dtype=np.uint8)
        extra_mask[45:65, 40:90] = 255

        cleaner = manga_text_cleaner.MangaTextCleaner(device="cpu")
        with patch.object(cleaner, "predict_mask", return_value=None):
            cleaned_pil, edit_mask, _ = cleaner.clean_page(
                img_np,
                extra_mask=extra_mask,
                allowed_boxes=[[20, 20, 180, 100]],
            )

        cleaned_np = np.array(cleaned_pil)
        # Inside the inner dark narration box, zero bright white (> 140) pixels should remain
        inner_gray = cv2.cvtColor(cleaned_np[30:90, 30:170], cv2.COLOR_RGB2GRAY)
        self.assertEqual(int(np.sum(inner_gray > 140)), 0)
        # Background stays dark (< 60), not smeared gray/white
        self.assertLess(float(np.median(inner_gray)), 50.0)
        self.assertIsNotNone(edit_mask)

    def test_union_extra_mask_preserves_structural_border_mask(self):
        # Create a 240x160 image with a long dark horizontal panel border and a translucent textured bubble
        img_np = np.full((160, 240, 3), 180, dtype=np.uint8)
        # Add screentone/texture variation so std_bg >= 18 routes to LaMa
        img_np[::2, ::2] = 140
        # Long dark horizontal panel border at y=20..24 spanning width (outside allowed_boxes at x<40 and x>200)
        img_np[20:24, 10:230] = 15

        # Translucent text region touching near the border at y=26..60
        extra_mask = np.zeros((160, 240), dtype=np.uint8)
        extra_mask[25:55, 60:180] = 255

        cleaner = manga_text_cleaner.MangaTextCleaner(device="cpu")
        fake_prob = np.zeros((160, 240), dtype=np.float32)

        def _fake_lama_inpaint(arr, mask, lama):
            out = arr.copy()
            out[mask > 0] = [170, 170, 170]
            return out

        with patch.object(cleaner, "predict_mask", return_value=fake_prob), patch.object(
            manga_text_cleaner, "get_shared_simple_lama", return_value=object()
        ), patch.object(
            manga_text_cleaner, "inpaint_mask_with_simple_lama", side_effect=_fake_lama_inpaint
        ):
            cleaned_pil, edit_mask, _ = cleaner.clean_page(
                img_np,
                extra_mask=extra_mask,
                allowed_boxes=[[40, 18, 200, 90]],
            )

        cleaned_np = np.array(cleaned_pil)
        # Structural border at y=21, x=100 must NOT be in edit_mask and must stay dark (15)
        self.assertEqual(int(edit_mask[21, 100]), 0)
        self.assertLess(int(cleaned_np[21, 100, 0]), 30)
        # Interior extra_mask region at y=40, x=100 must be in edit_mask and cleaned
        self.assertEqual(int(edit_mask[40, 100]), 255)
        self.assertEqual(int(cleaned_np[40, 100, 0]), 170)

    def test_deduplicate_overlapping_detections_suppresses_duplicates_keeps_lobes(self):
        dets = [
            {"box": [100, 20, 220, 100], "category": "narration", "confidence": 0.92},
            {"box": [104, 22, 218, 98], "category": "narration", "confidence": 0.81},  # duplicate
            {"box": [225, 20, 310, 100], "category": "speech_bubble", "confidence": 0.88},  # adjacent lobe
        ]
        deduped = _deduplicate_overlapping_detections(dets)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["confidence"], 0.92)
        self.assertEqual(deduped[1]["box"], [225, 20, 310, 100])


if __name__ == "__main__":
    unittest.main()
