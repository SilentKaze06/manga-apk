"""Unit tests for RT-DETR bubble detection and horizontal column splitting."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

SERVER_DIR = Path(__file__).resolve().parent.parent
ML_DIR = SERVER_DIR / "ml"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from bubble_rtdetr import _box_iou, _class_aware_nms, BubbleRTDetr
from comic_detector import _CTDBlock, _split_horizontal_blk_columns, _expand_text_box_with_ink, ComicTextDetector
from tests.debug_detect import annotate_detection_boxes


class RTDetrBubbleSplitTests(unittest.TestCase):
    def test_class_aware_nms_preserves_distinct_adjacent_bubbles(self):
        # Two touching speech bubbles side-by-side
        dets = [
            {"box": [50, 50, 150, 150], "score": 0.88, "label": "bubble"},
            {"box": [140, 50, 240, 150], "score": 0.82, "label": "bubble"},  # touching, IoU ~ 0.05
            {"box": [52, 52, 148, 148], "score": 0.65, "label": "bubble"},   # duplicate of 1st, IoU ~ 0.90
            {"box": [70, 70, 130, 120], "score": 0.75, "label": "text_bubble"},  # text inside 1st bubble
        ]
        kept = _class_aware_nms(dets, iou_thresh=0.45)
        # Should keep both bubbles and text_bubble, suppressing only the duplicate
        labels_and_boxes = [(d["label"], d["box"]) for d in kept]
        self.assertEqual(len(kept), 3)
        bubble_boxes = [d["box"] for d in kept if d["label"] == "bubble"]
        self.assertEqual(len(bubble_boxes), 2)
        self.assertIn([50, 50, 150, 150], bubble_boxes)
        self.assertIn([140, 50, 240, 150], bubble_boxes)

    def test_split_horizontal_blk_columns_splits_side_by_side_columns(self):
        # Create a horizontal block with 4 lines: 2 in left column, 2 in right column
        lines = [
            # Left column lines
            [[20, 30], [80, 30], [80, 50], [20, 50]],
            [[22, 55], [82, 55], [82, 75], [22, 75]],
            # Right column lines (separated horizontally with zero overlap)
            [[160, 30], [220, 30], [220, 50], [160, 50]],
            [[158, 55], [218, 55], [218, 75], [158, 75]],
        ]
        blk = _CTDBlock([20, 30, 220, 75], lines=lines, language="eng", conf=0.90)

        # Create dummy gray image with white background
        gray = np.full((120, 260), 255, dtype=np.uint8)
        mask = np.zeros((120, 260), dtype=np.uint8)
        mask[30:75, 20:82] = 255
        mask[30:75, 158:220] = 255

        splitted, sub_blks = _split_horizontal_blk_columns(blk, gray=gray, mask_u8=mask)
        self.assertTrue(splitted)
        self.assertEqual(len(sub_blks), 2)

        # Left sub_blk must have the 2 left lines, right sub_blk the 2 right lines
        left_blk, right_blk = sub_blks[0], sub_blks[1]
        self.assertEqual(len(left_blk.lines), 2)
        self.assertEqual(len(right_blk.lines), 2)
        self.assertLess(left_blk.xyxy[2], right_blk.xyxy[0])

    def test_split_horizontal_blk_columns_splits_on_dark_gutter_barrier(self):
        # Two lines with slight overlap but separated by a dark gutter (gray < 100)
        lines = [
            [[30, 30], [110, 30], [110, 60], [30, 60]],
            [[120, 30], [190, 30], [190, 60], [120, 60]],
        ]
        blk = _CTDBlock([30, 30, 190, 60], lines=lines, language="eng", conf=0.92)

        # Gray image with dark vertical gutter at x=112..118
        gray = np.full((100, 220), 240, dtype=np.uint8)
        gray[:, 112:118] = 20  # dark gutter
        mask = np.zeros((100, 220), dtype=np.uint8)

        splitted, sub_blks = _split_horizontal_blk_columns(blk, gray=gray, mask_u8=mask)
        self.assertTrue(splitted)
        self.assertEqual(len(sub_blks), 2)
        self.assertLessEqual(sub_blks[0].xyxy[2], 112)
        self.assertGreaterEqual(sub_blks[1].xyxy[0], 118)

    def test_debug_detect_does_not_import_qwen_or_simple_lama(self):
        # Verify debug_detect imports cleanly even if qwen and simple_lama are poisoned in sys.modules
        poisoned = {
            "qwen_vl_engine": None,
            "simple_lama_inpainting": None,
        }
        with patch.dict(sys.modules, poisoned):
            import tests.debug_detect as dbg
            self.assertTrue(callable(dbg.annotate_detection_boxes))

    def test_annotate_detection_boxes_draws_distinct_overlay(self):
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        blocks = [
            {"block_id": "b_1", "box": [10, 10, 90, 90], "text_box": [20, 20, 80, 80], "category": "speech_bubble"},
            {"block_id": "b_2", "box": [110, 10, 190, 90], "text_box": [120, 20, 180, 80], "category": "speech_bubble"},
        ]
        annotated = annotate_detection_boxes(img, blocks)
        self.assertEqual(annotated.size, (200, 200))
        # Image should now contain colorful pixels from the palette
        ann_np = np.array(annotated)
        self.assertFalse(np.all(ann_np == 255))

    def test_split_horizontal_blk_columns_slices_single_wide_line_on_projection_valley(self):
        # Single wide DBNet line spanning two figure-8 bubble columns
        wide_line = [[20, 20], [240, 20], [240, 80], [20, 80]]
        blk = _CTDBlock([20, 20, 240, 80], lines=[wide_line], language="eng", conf=0.91)

        gray = np.full((120, 260), 255, dtype=np.uint8)
        mask = np.zeros((120, 260), dtype=np.uint8)
        mask[25:75, 25:95] = 255    # Left column ink
        mask[25:75, 165:235] = 255  # Right column ink (valley at 95..165)

        splitted, sub_blks = _split_horizontal_blk_columns(blk, gray=gray, mask_u8=mask)
        self.assertTrue(splitted)
        self.assertEqual(len(sub_blks), 2)
        self.assertLessEqual(sub_blks[0].xyxy[2], 100)
        self.assertGreaterEqual(sub_blks[1].xyxy[0], 160)

    def test_expand_text_box_with_ink_captures_all_dark_box_lines(self):
        gray = np.full((120, 120), 20, dtype=np.uint8)
        # 3 bright text lines inside dark narration box [10, 10, 110, 110]
        gray[22:32, 30:85] = 240
        gray[48:58, 35:75] = 240
        gray[75:85, 40:80] = 240

        # DBNet only detected middle line
        dbnet_tbox = [35, 48, 75, 58]
        expanded = _expand_text_box_with_ink(dbnet_tbox, [10, 10, 110, 110], gray=gray, mask_u8=None)
        self.assertLessEqual(expanded[1], 22)
        self.assertGreaterEqual(expanded[3], 85)

    def test_annotate_detection_boxes_clamps_y_to_height_not_width(self):
        # Portrait image w=200, h=600 with box below y=200
        img = Image.new("RGB", (200, 600), (255, 255, 255))
        blocks = [
            {"block_id": "b_1", "box": [20, 420, 160, 540], "text_box": [30, 430, 150, 530], "category": "speech_bubble"},
        ]
        annotated = annotate_detection_boxes(img, blocks)
        ann_np = np.array(annotated)
        # Bottom edge at y=540 must be drawn (not flattened to y=200)
        self.assertFalse(np.all(ann_np[420:545, 20:165] == 255))


if __name__ == "__main__":
    unittest.main()
