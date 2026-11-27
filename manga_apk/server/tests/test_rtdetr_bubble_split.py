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
from detection import detect_stepped_box_split_y, split_joined_bubble_lobes
from comic_detector import (
    _CTDBlock,
    _split_horizontal_blk_columns,
    _expand_text_box_with_ink,
    _merge_same_bubble_rtdetr_boxes,
    _merge_overlapping_same_bubble_results,
    ComicTextDetector,
)
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

    def test_merge_rtdetr_half_boxes_on_single_smooth_oval(self):
        # Two RT-DETR half-boxes on a single smooth white oval (contain ~ 0.20, shared text line, no dark border)
        rt_bubbles = [
            {"box": [20, 20, 140, 120], "score": 0.88, "label": "bubble"},
            {"box": [110, 22, 230, 118], "score": 0.84, "label": "bubble"},
        ]
        shared_line = np.array([[45, 60], [205, 60], [205, 82], [45, 82]], dtype=np.int32)
        gray = np.full((150, 260), 255, dtype=np.uint8)

        merged = _merge_same_bubble_rtdetr_boxes(rt_bubbles, [shared_line], gray)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["box"], [20, 20, 230, 120])

    def test_preserve_genuinely_touching_bubbles_with_border_or_pinch(self):
        # Case A: Dark vertical ink border separating two touching bubbles with separate lines
        rt_bubbles = [
            {"box": [20, 20, 130, 120], "score": 0.90, "label": "bubble"},
            {"box": [120, 20, 230, 120], "score": 0.89, "label": "bubble"},
        ]
        lines = [
            np.array([[35, 50], [105, 50], [105, 80], [35, 80]], dtype=np.int32),
            np.array([[145, 50], [215, 50], [215, 80], [145, 80]], dtype=np.int32),
        ]
        gray_border = np.full((150, 260), 255, dtype=np.uint8)
        gray_border[20:120, 124:127] = 15  # dark vertical ink stroke

        kept_border = _merge_same_bubble_rtdetr_boxes(rt_bubbles, lines, gray_border)
        self.assertEqual(len(kept_border), 2)

        # Case B: Sharp figure-8 concave pinch (white span at overlap midpoint < 68%) with separate lines
        gray_pinch = np.full((150, 260), 30, dtype=np.uint8)
        gray_pinch[20:120, 20:115] = 255
        gray_pinch[20:120, 135:230] = 255
        gray_pinch[55:85, 115:135] = 255  # narrow 30px neck out of 100px overlap height (30% < 68%)

        kept_pinch = _merge_same_bubble_rtdetr_boxes(rt_bubbles, lines, gray_pinch)
        self.assertEqual(len(kept_pinch), 2)

    def test_split_horizontal_blk_columns_ignores_narrow_word_space_but_splits_wide_valley(self):
        # Single English sentence with a 5px inter-word space in the middle must NOT split
        wide_line = [[20, 30], [220, 30], [220, 60], [20, 60]]
        blk_single = _CTDBlock([20, 30, 220, 60], lines=[wide_line], language="eng", conf=0.90)
        gray_single = np.full((100, 250), 255, dtype=np.uint8)
        mask_single = np.zeros((100, 250), dtype=np.uint8)
        mask_single[35:55, 25:115] = 255
        mask_single[35:55, 120:215] = 255  # 5px word space at 115..120
        # Thin text strokes (not a solid 67%-height black wall)
        gray_single[38:42, 25:115] = 20
        gray_single[48:52, 25:115] = 20
        gray_single[38:42, 120:215] = 20
        gray_single[48:52, 120:215] = 20

        splitted_single, sub_single = _split_horizontal_blk_columns(blk_single, gray=gray_single, mask_u8=mask_single)
        self.assertFalse(splitted_single)
        self.assertEqual(len(sub_single), 1)

        # Wide two-column figure-8 bubble with a genuine 60px valley MUST split
        blk_double = _CTDBlock([20, 20, 240, 80], lines=[[[20, 20], [240, 20], [240, 80], [20, 80]]], language="eng", conf=0.91)
        gray_double = np.full((120, 260), 255, dtype=np.uint8)
        mask_double = np.zeros((120, 260), dtype=np.uint8)
        mask_double[25:75, 25:95] = 255
        mask_double[25:75, 155:235] = 255
        gray_double[30:35, 25:95] = 20
        gray_double[55:60, 25:95] = 20
        gray_double[30:35, 155:235] = 20
        gray_double[55:60, 155:235] = 20

        splitted_double, sub_double = _split_horizontal_blk_columns(blk_double, gray=gray_double, mask_u8=mask_double)
        self.assertTrue(splitted_double)
        self.assertEqual(len(sub_double), 2)

    def test_merge_overlapping_same_bubble_results_merges_overlapping_text_boxes(self):
        gray = np.full((200, 300), 255, dtype=np.uint8)
        results = [
            {
                "block_id": "b_1",
                "box": [20, 20, 170, 120],
                "text_box": [40, 40, 150, 100],
                "category": "speech_bubble",
                "is_dark": False,
                "confidence": 0.85,
                "vertical": False,
            },
            {
                "block_id": "b_2",
                "box": [100, 22, 240, 118],
                "text_box": [125, 42, 220, 98],
                "category": "speech_bubble",
                "is_dark": False,
                "confidence": 0.88,
                "vertical": False,
            },
        ]
        merged = _merge_overlapping_same_bubble_results(results, gray)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["box"], [20, 20, 240, 120])
        self.assertEqual(merged[0]["text_box"], [40, 40, 220, 100])
        self.assertAlmostEqual(merged[0]["confidence"], 0.88)

    def test_recover_short_bubble_missed_by_dbnet_via_rtdetr_text_bubble(self):
        import torch

        img = Image.new("RGB", (200, 200), (20, 20, 20))
        # White "BUT..." pixels inside dark box [40, 40, 140, 120]
        arr = np.array(img)
        arr[65:95, 55:125] = 240
        img = Image.fromarray(arr)

        detector = ComicTextDetector(device="cpu")
        mock_net = Mock()
        # Return empty YOLO detections, zero UNet mask, dummy lines_map
        blks_raw = torch.zeros((1, 0, 6), dtype=torch.float32)
        mask_raw = torch.zeros((1, 1, 1024, 1024), dtype=torch.float32)
        lines_map = torch.zeros((1, 2, 1024, 1024), dtype=torch.float32)
        mock_net.return_value = (blks_raw, mask_raw, lines_map)

        detector.load_model = Mock(return_value=mock_net)
        detector.seg_rep = Mock(return_value=([], []))

        mock_rt = Mock()
        mock_rt.is_available = True
        mock_rt.detect.return_value = [
            {"box": [40, 40, 140, 120], "score": 0.78, "label": "bubble"},
            {"box": [55, 65, 125, 95], "score": 0.65, "label": "text_bubble"},
        ]

        with patch("comic_detector._non_max_suppression", return_value=[torch.zeros((0, 6))]), \
             patch("comic_detector.get_shared_bubble_rtdetr", return_value=mock_rt):
            blocks, _ = detector.detect(img)

        self.assertEqual(len(blocks), 1)
        tb = blocks[0]["text_box"]
        self.assertLessEqual(tb[0], 55)
        self.assertGreaterEqual(tb[2], 125)
        self.assertLessEqual(tb[1], 65)
        self.assertGreaterEqual(tb[3], 95)

    def test_split_joined_bubble_lobes_splits_vertical_and_diagonal_but_preserves_single_oval(self):
        import cv2

        # 1. Vertical figure-8 double bubble (white page bg, dark 5px bubble outline, pinched neck)
        gray_v = np.full((240, 140), 255, dtype=np.uint8)
        cv2.circle(gray_v, (70, 60), 48, 0, -1)
        cv2.circle(gray_v, (70, 180), 48, 0, -1)
        cv2.rectangle(gray_v, (54, 95), (86, 145), 0, -1)
        cv2.circle(gray_v, (70, 60), 44, 255, -1)
        cv2.circle(gray_v, (70, 180), 44, 255, -1)
        cv2.rectangle(gray_v, (58, 95), (82, 145), 255, -1)  # 24px neck (< 0.68 * 44)
        lobes_v = split_joined_bubble_lobes(gray_v, [10, 5, 130, 235])
        self.assertEqual(len(lobes_v), 2)
        ys_centers = sorted((b[1] + b[3]) / 2.0 for _, b in lobes_v)
        self.assertLess(ys_centers[0], 100)
        self.assertGreater(ys_centers[1], 140)

        # 2. Diagonal figure-8 double bubble
        gray_d = np.full((220, 220), 255, dtype=np.uint8)
        cv2.circle(gray_d, (65, 65), 48, 0, -1)
        cv2.circle(gray_d, (155, 155), 48, 0, -1)
        cv2.line(gray_d, (65, 65), (155, 155), 0, 32)
        cv2.circle(gray_d, (65, 65), 44, 255, -1)
        cv2.circle(gray_d, (155, 155), 44, 255, -1)
        cv2.line(gray_d, (65, 65), (155, 155), 255, 24)
        lobes_d = split_joined_bubble_lobes(gray_d, [10, 10, 210, 210])
        self.assertEqual(len(lobes_d), 2)

        # 3. Single elongated oval bubble must NOT over-split (saddle > 0.68 * min_peak)
        gray_oval = np.full((240, 140), 255, dtype=np.uint8)
        cv2.ellipse(gray_oval, (70, 120), (48, 98), 0, 0, 360, 0, -1)
        cv2.ellipse(gray_oval, (70, 120), (44, 94), 0, 0, 360, 255, -1)
        lobes_oval = split_joined_bubble_lobes(gray_oval, [10, 10, 130, 230])
        self.assertEqual(lobes_oval, [])

    def test_detect_stepped_box_split_y_detects_vertical_step(self):
        # Upper box [20..120, 20..90] + lower offset box [60..180, 90..160]
        gray = np.full((180, 200), 30, dtype=np.uint8)
        gray[20:90, 20:120] = 250
        gray[90:160, 60:180] = 250
        y_split = detect_stepped_box_split_y(gray, [15, 15, 185, 165])
        self.assertIsNotNone(y_split)
        self.assertTrue(80 <= y_split <= 100)


if __name__ == "__main__":
    unittest.main()
