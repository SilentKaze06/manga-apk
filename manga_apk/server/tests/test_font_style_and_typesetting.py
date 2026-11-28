"""Unit tests for style-aware manga fonts and upgraded typesetting."""
import unittest

import numpy as np
from PIL import Image

from ml.font_style import (
    classify_bubble_style,
    compute_proportional_stroke,
    get_styled_font,
    verify_font_glyph_coverage,
)
from ml.typesetting import _should_render_vertical_strip, render_translated_dialogues


class FontStyleAndTypesettingTests(unittest.TestCase):
    def test_classify_bubble_style_distinct_buckets(self):
        self.assertEqual(
            classify_bubble_style({"category": "narration"}, "Позднее тем вечером...", 200, 80, False, False),
            "narration",
        )
        self.assertEqual(
            classify_bubble_style({"category": "speech_bubble"}, "ЮТАКА-КУН?!", 160, 120, False, False),
            "shout",
        )
        self.assertEqual(
            classify_bubble_style({"category": "thought_bubble"}, "Неужели это правда...", 180, 120, False, False),
            "thought",
        )
        self.assertEqual(
            classify_bubble_style({"category": "speech_bubble"}, "Привет, как дела?", 180, 120, False, False),
            "dialogue",
        )

    def test_vertical_strip_bug_fixed(self):
        self.assertFalse(_should_render_vertical_strip("ДА.", 42, 95, False))
        self.assertTrue(_should_render_vertical_strip("П-Р-О-С-Т-И", 42, 180, False))

    def test_styled_fonts_support_russian_and_uzbek_glyphs(self):
        sample = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ абвгдежзийклмнопрстуфхцчшщъыьэюя ÖöĞğŞşÇç"
        for style in ("dialogue", "shout", "narration", "thought", "sfx_title"):
            f = get_styled_font(style, 20, sample_text=sample)
            self.assertTrue(verify_font_glyph_coverage(f, "АаЯяÖöĞğŞşÇç"), f"Missing glyphs for style={style}")

    def test_proportional_stroke_scales_with_font_size(self):
        _, _, sw_small = compute_proportional_stroke(150.0, 14, is_on_artwork=True, style="dialogue")
        _, _, sw_large = compute_proportional_stroke(150.0, 36, is_on_artwork=True, style="sfx_title")
        self.assertEqual(sw_small, 2)
        self.assertEqual(sw_large, 5)

    def test_render_translated_dialogues_two_arg_signature(self):
        img = Image.new("RGB", (400, 300), (255, 255, 255))
        dialogues = [
            {
                "bbox": [20, 20, 200, 140],
                "category": "speech_bubble",
                "translation": "ЮТАКА-КУН?!",
            },
            {
                "bbox": [220, 20, 380, 140],
                "category": "narration",
                "translation": "Özgacha tuyg'u va Şirin xotira",
            },
            {
                "bbox": [40, 160, 82, 255],
                "category": "speech_bubble",
                "translation": "ДА.",
            },
        ]
        out = render_translated_dialogues(img, dialogues)
        self.assertEqual(out.size, (400, 300))
        self.assertFalse(np.array_equal(np.asarray(out), np.asarray(img)))


if __name__ == "__main__":
    unittest.main()
