"""
Geometry and spatial barrier utilities for manga text detection and clustering.
Contains border mask computation, ray-crossing barriers, white connected components,
and dark barrier detection.
"""

import math
import hashlib
import threading
import logging
from collections import OrderedDict
from typing import Optional, List, Tuple, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_white_cc_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
_border_mask_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_MAX_CACHE_ENTRIES = 4
_pipeline_cache_lock = threading.Lock()


def _get_gray_cache_key(gray):
    H, W = gray.shape[:2]
    buf = np.ascontiguousarray(gray)
    h = hashlib.blake2b(buf.view(np.uint8), digest_size=16).hexdigest()
    return (H, W, h)


def get_white_cc_labels(gray, return_stats=False):
    """
    Computes 8-way connected components and stats for white background pixels (gray > 200).
    Uses full-content blake2b hashed LRU cache (maxsize=4) to avoid collisions and excess RAM use.
    """
    if gray is None or gray.size == 0:
        return (None, None) if return_stats else None
    cache_key = _get_gray_cache_key(gray)
    with _pipeline_cache_lock:
        if cache_key in _white_cc_cache:
            _white_cc_cache.move_to_end(cache_key)
            labels, stats = _white_cc_cache[cache_key]
            return (labels, stats) if return_stats else labels
    white_mask = (gray > 200).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    with _pipeline_cache_lock:
        _white_cc_cache[cache_key] = (labels, stats)
        _white_cc_cache.move_to_end(cache_key)
        while len(_white_cc_cache) > _MAX_CACHE_ENTRIES:
            _white_cc_cache.popitem(last=False)
    return (labels, stats) if return_stats else labels


def _get_box_white_mode_label(labels, stats, gray, box, min_area=200, pad=3):
    """
    Returns the mode white component label inside box first (preventing outer gutter leakage),
    falling back to a small padded perimeter only if no interior white pixels exist.
    Scales min_area down for small bubbles so small text boxes are not dropped to label=0.
    """
    if labels is None or gray is None:
        return 0
    H, W = gray.shape[:2]
    xA1, yA1, xA2, yA2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    ix1, iy1 = max(0, xA1), max(0, yA1)
    ix2, iy2 = min(W, xA2), min(H, yA2)
    box_area = max(1, (ix2 - ix1) * (iy2 - iy1))
    eff_min_area = min(min_area, max(25, int(box_area * 0.25)))

    def _pick_mode(sub_labels, sub_gray):
        valid_mask = (sub_labels > 0) & (sub_gray > 200)
        valid_labels = sub_labels[valid_mask]
        if valid_labels.size == 0:
            return 0
        vals, counts = np.unique(valid_labels, return_counts=True)
        sorted_order = np.argsort(-counts)
        for idx in sorted_order:
            lbl = int(vals[idx])
            if stats is not None and lbl < stats.shape[0]:
                area = stats[lbl, cv2.CC_STAT_AREA]
                if area > eff_min_area:
                    return lbl
            elif stats is None:
                return lbl
        return 0

    if ix2 > ix1 and iy2 > iy1:
        interior_lbl = _pick_mode(labels[iy1:iy2, ix1:ix2], gray[iy1:iy2, ix1:ix2])
        if interior_lbl > 0:
            return interior_lbl

    eff_pad = min(pad, 2)
    px1, py1 = max(0, xA1 - eff_pad), max(0, yA1 - eff_pad)
    px2, py2 = min(W, xA2 + eff_pad), min(H, yA2 + eff_pad)
    if px2 <= px1 or py2 <= py1:
        return 0
    return _pick_mode(labels[py1:py2, px1:px2], gray[py1:py2, px1:px2])


def are_different_white_contours(gray, boxA, boxB, min_white_ratio=0.15):
    """
    Checks if boxA and boxB belong to two different closed white contours (gray > 200, area > 200).
    If both belong to distinct white connected components (separate bubble/panel outlines),
    they must NEVER be merged.
    """
    if gray is None or gray.size == 0:
        return False
    labels, stats = get_white_cc_labels(gray, return_stats=True)
    if labels is None:
        return False

    labelA = _get_box_white_mode_label(labels, stats, gray, boxA, min_area=200, pad=3)
    labelB = _get_box_white_mode_label(labels, stats, gray, boxB, min_area=200, pad=3)

    if labelA > 0 and labelB > 0 and labelA != labelB:
        return True
    return False


def get_box_border_mask(gray):
    """
    Computes a morphological straight frame mask (box_border_mask) for the page.
    Uses resolution-aware kernel sizes scaled to page dimensions (with safe min/max clamps)
    and caches by full-content gray key to avoid O(N^2) morphology passes.
    """
    if gray is None or gray.size == 0:
        return None
    cache_key = _get_gray_cache_key(gray)
    with _pipeline_cache_lock:
        if cache_key in _border_mask_cache:
            _border_mask_cache.move_to_end(cache_key)
            return _border_mask_cache[cache_key]

    H, W = gray.shape[:2]
    dark = (gray < 160).astype(np.uint8)
    kw = max(24, min(int(round(W * 0.035)), 80))
    kh = max(24, min(int(round(H * 0.035)), 80))
    line_kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    line_kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
    horiz_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, line_kernel_h)
    vert_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, line_kernel_v)

    # Exclude solid large dark regions (e.g. dark narration boxes, dark clothing/artwork)
    # except thin-to-medium panel gutters spanning >= 0.40 of page width/height
    thick_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 16))
    solid_dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, thick_kernel)
    long_gutter_h = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(48, int(W * 0.40)), 1)))
    long_gutter_v = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(48, int(H * 0.40)))))
    horiz_lines = cv2.bitwise_or(cv2.bitwise_and(horiz_lines, cv2.bitwise_not(solid_dark)), long_gutter_h)
    vert_lines = cv2.bitwise_or(cv2.bitwise_and(vert_lines, cv2.bitwise_not(solid_dark)), long_gutter_v)
    result = cv2.bitwise_or(horiz_lines, vert_lines)

    with _pipeline_cache_lock:
        _border_mask_cache[cache_key] = result
        _border_mask_cache.move_to_end(cache_key)
        while len(_border_mask_cache) > _MAX_CACHE_ENTRIES:
            _border_mask_cache.popitem(last=False)
    return result


def _max_true_run_1d(bool_arr: np.ndarray) -> int:
    """Vectorized maximum consecutive True run length in a 1D boolean array."""
    if bool_arr.size == 0 or not np.any(bool_arr):
        return 0
    padded = np.empty(bool_arr.size + 2, dtype=np.uint8)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = bool_arr.astype(np.uint8)
    diffs = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1)
    if starts.size == 0 or ends.size == 0:
        return 0
    return int(np.max(ends - starts))


def line_crosses_border_mask(border_mask, ptA, ptB, boxA=None, boxB=None):
    """
    Checks if a straight frame border barrier separates boxA and boxB or crosses ptA->ptB.
    Checks border continuity specifically across the facing gap corridor between boxA and boxB.
    """
    if border_mask is None or border_mask.size == 0:
        return False
    H, W = border_mask.shape[:2]

    if ptA is not None and len(ptA) == 4 and boxA is None:
        boxA = ptA
        ptA = ((ptA[0] + ptA[2]) / 2.0, (ptA[1] + ptA[3]) / 2.0)
    if ptB is not None and len(ptB) == 4 and boxB is None:
        boxB = ptB
        ptB = ((ptB[0] + ptB[2]) / 2.0, (ptB[1] + ptB[3]) / 2.0)

    # Check border continuity specifically across the facing gap corridor between boxA and boxB
    if boxA is not None and boxB is not None:
        xA1, yA1, xA2, yA2 = boxA
        xB1, yB1, xB2, yB2 = boxB

        # 1. Vertically stacked / facing boxes (one above another)
        top_box = boxA if yA1 < yB1 else boxB
        bot_box = boxB if yA1 < yB1 else boxA
        x_start = max(0, max(top_box[0], bot_box[0]))
        x_end = min(W, min(top_box[2], bot_box[2]))
        corridor_w = x_end - x_start

        if corridor_w >= 8 and (top_box[3] - bot_box[1]) <= 8:
            y_gap_top = min(top_box[3], bot_box[1])
            y_gap_bot = max(top_box[3], bot_box[1])
            search_y1 = max(0, y_gap_top - 2)
            search_y2 = min(H, y_gap_bot + 3)
            min_consec_w = max(10, int(0.35 * corridor_w))

            for r in range(search_y1, search_y2):
                row_bool = border_mask[r, x_start:x_end] > 0
                if np.mean(row_bool) >= 0.45 or _max_true_run_1d(row_bool) >= min_consec_w:
                    return True

        # 2. Horizontally adjacent / facing boxes (columns)
        left_box = boxA if xA1 < xB1 else boxB
        right_box = boxB if xA1 < xB1 else boxA
        y_start = max(0, max(left_box[1], right_box[1]))
        y_end = min(H, min(left_box[3], right_box[3]))
        corridor_h = y_end - y_start

        if corridor_h >= 8 and (left_box[2] - right_box[0]) <= 8:
            x_gap_left = min(left_box[2], right_box[0])
            x_gap_right = max(left_box[2], right_box[0])
            search_x1 = max(0, x_gap_left - 2)
            search_x2 = min(W, x_gap_right + 3)
            min_consec_h = max(10, int(0.35 * corridor_h))

            for c in range(search_x1, search_x2):
                col_bool = border_mask[y_start:y_end, c] > 0
                if np.mean(col_bool) >= 0.45 or _max_true_run_1d(col_bool) >= min_consec_h:
                    return True

        # If both boxes have substantial horizontal overlap or vertical overlap without a corridor barrier:
        # Do not let diagonal rays falsely split them!
        if corridor_w >= 10 or corridor_h >= 10:
            return False

    # Fallback for diagonal rays or when boxA/boxB are not provided
    cxA, cyA = ptA
    cxB, cyB = ptB
    dist = int(np.hypot(cxB - cxA, cyB - cyA))
    if dist < 2:
        return False
    num_samples = max(dist * 2, 20)
    xs = np.clip(np.linspace(cxA, cxB, num_samples).round().astype(int), 0, W - 1)
    ys = np.clip(np.linspace(cyA, cyB, num_samples).round().astype(int), 0, H - 1)

    pad = 2
    valid_indices = []
    for idx in range(num_samples):
        x, y = xs[idx], ys[idx]
        if boxA is not None:
            if (boxA[0] + pad <= x <= boxA[2] - pad) and (boxA[1] + pad <= y <= boxA[3] - pad):
                continue
        if boxB is not None:
            if (boxB[0] + pad <= x <= boxB[2] - pad) and (boxB[1] + pad <= y <= boxB[3] - pad):
                continue
        valid_indices.append(idx)

    if not valid_indices:
        inner_start = max(1, int(0.20 * num_samples))
        inner_end = min(num_samples - 1, int(0.80 * num_samples))
        valid_indices = list(range(inner_start, inner_end + 1))

    sampled_vals = border_mask[ys[valid_indices], xs[valid_indices]]
    max_run = 0
    cur_run = 0
    for v in sampled_vals:
        if v > 0:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            cur_run = 0
    return max_run >= 2 or np.sum(sampled_vals > 0) >= 3


def _has_continuous_dark_segment(arr_1d, dark_thresh=110, min_len=12):
    """
    Checks if a 1D array has at least min_len consecutive pixels with value < dark_thresh.
    Used to detect continuous dark frame/border lines across boundaries.
    """
    if arr_1d is None or len(arr_1d) < min_len or min_len <= 0:
        return False
    return _max_true_run_1d(np.asarray(arr_1d) < dark_thresh) >= min_len


def is_dark_barrier_between_x(gray, boxA, boxB, dark_thresh=100, min_line_ratio=0.40, border_mask=None):
    if gray is None or gray.size == 0:
        return False
    H, W = gray.shape[:2]
    left_box, right_box = (boxA, boxB) if boxA[0] <= boxB[0] else (boxB, boxA)
    xA1, yA1, xA2, yA2 = left_box
    xB1, yB1, xB2, yB2 = right_box

    # Check if both boxes sit inside a dark narration box
    comb_crop = gray[max(0, min(yA1, yB1)):min(H, max(yA2, yB2)), max(0, min(xA1, xB1)):min(W, max(xA2, xB2))]
    if comb_crop.size > 0:
        med_val = float(np.median(comb_crop))
        if med_val < 95 or (med_val < 120 and float(np.mean(comb_crop < 110)) >= 0.50):
            # Inside a dark narration box: dark pixels are normal background, not a barrier
            y_top = max(0, max(yA1, yB1))
            y_bot = min(H, min(yA2, yB2))
            if y_bot > y_top and xA2 < xB1:
                gap_slice = gray[y_top:y_bot, max(0, xA2):min(W, xB1)]
                if gap_slice.size > 0 and np.mean(gap_slice > 185) >= 0.40:
                    return True
            return False

    cxA = int((xA1 + xA2) / 2.0)
    cyA = int((yA1 + yA2) / 2.0)
    cxB = int((xB1 + xB2) / 2.0)
    cyB = int((yB1 + yB2) / 2.0)
    if border_mask is None:
        border_mask = get_box_border_mask(gray)
    if border_mask is not None and line_crosses_border_mask(border_mask, (cxA, cyA), (cxB, cyB), boxA=left_box, boxB=right_box):
        return True

    y_top = max(yA1, yB1)
    y_bot = min(yA2, yB2)
    # If vertically stacked or separated with no vertical overlap, there is no X barrier
    if y_bot <= y_top:
        return False
    if y_bot - y_top < 6:
        mid_y = int((y_top + y_bot) / 2.0)
        y_top = max(0, mid_y - 8)
        y_bot = min(H, mid_y + 8)
    else:
        y_top = max(0, y_top)
        y_bot = min(H, y_bot)

    span = y_bot - y_top
    min_consec = max(12, int(0.25 * span))

    if xA2 < xB1:
        xA2_c = max(0, min(W, xA2))
        xB1_c = max(0, min(W, xB1))
        gap_slice = gray[y_top:y_bot, xA2_c:xB1_c]
        if gap_slice.size > 0 and gap_slice.shape[0] >= 4:
            dark_counts = np.sum(gap_slice < dark_thresh, axis=0)
            col_ratios = dark_counts / float(gap_slice.shape[0])
            if np.any(col_ratios >= min_line_ratio):
                return True
            for c in range(gap_slice.shape[1]):
                if _has_continuous_dark_segment(gap_slice[:, c], dark_thresh=110, min_len=min_consec):
                    return True
        return False
    else:
        # When xA2 >= xB1: only check if there is a true vertical divider column between cxA and cxB
        cx_min = max(0, min(cxA, cxB))
        cx_max = min(W - 1, max(cxA, cxB))
        if cx_max - cx_min >= 4 and (y_bot - y_top) >= 8:
            center_slice = gray[y_top:y_bot, cx_min:cx_max + 1]
            if center_slice.size > 0 and center_slice.shape[0] >= 6:
                dark_counts = np.sum(center_slice < dark_thresh, axis=0)
                col_ratios = dark_counts / float(center_slice.shape[0])
                if np.any(col_ratios >= 0.65):
                    return True
        return False


def is_dark_barrier_between_y(gray, boxA, boxB, dark_thresh=95, min_line_ratio=0.40, border_mask=None):
    if gray is None or gray.size == 0:
        return False
    H, W = gray.shape[:2]
    top_box, bot_box = (boxA, boxB) if boxA[1] <= boxB[1] else (boxB, boxA)
    xA1, yA1, xA2, yA2 = top_box
    xB1, yB1, xB2, yB2 = bot_box

    # Check if both boxes sit inside a dark narration box
    comb_crop = gray[max(0, min(yA1, yB1)):min(H, max(yA2, yB2)), max(0, min(xA1, xB1)):min(W, max(xA2, xB2))]
    if comb_crop.size > 0:
        med_val = float(np.median(comb_crop))
        if med_val < 95 or (med_val < 120 and float(np.mean(comb_crop < 110)) >= 0.50):
            # Inside a dark narration box: dark pixels are normal background, not a barrier
            x_left = max(0, max(xA1, xB1))
            x_right = min(W, min(xA2, xB2))
            if x_right > x_left and yA2 < yB1:
                gap_slice = gray[max(0, yA2):min(H, yB1), x_left:x_right]
                if gap_slice.size > 0 and np.mean(gap_slice > 185) >= 0.40:
                    return True
            return False

    cxA = int((xA1 + xA2) / 2.0)
    cyA = int((yA1 + yA2) / 2.0)
    cxB = int((xB1 + xB2) / 2.0)
    cyB = int((yB1 + yB2) / 2.0)
    if border_mask is None:
        border_mask = get_box_border_mask(gray)
    if border_mask is not None and line_crosses_border_mask(border_mask, (cxA, cyA), (cxB, cyB), boxA=top_box, boxB=bot_box):
        return True

    x_left = max(xA1, xB1)
    x_right = min(xA2, xB2)
    if x_right - x_left < 6:
        mid_x = int(((xA1 + xA2) / 2.0 + (xB1 + xB2) / 2.0) / 2.0)
        x_left = max(0, mid_x - 8)
        x_right = min(W, mid_x + 8)
    else:
        x_left = max(0, x_left)
        x_right = min(W, x_right)

    span = x_right - x_left
    min_consec = max(12, int(0.25 * span))

    if yA2 < yB1:
        yA2_c = max(0, min(H, yA2))
        yB1_c = max(0, min(H, yB1))
        gap_slice = gray[yA2_c:yB1_c, x_left:x_right]
        if gap_slice.size > 0 and gap_slice.shape[1] >= 4:
            dark_counts = np.sum(gap_slice < dark_thresh, axis=1)
            row_ratios = dark_counts / float(gap_slice.shape[1])
            if np.any(row_ratios >= min_line_ratio):
                return True
            for r in range(gap_slice.shape[0]):
                if _has_continuous_dark_segment(gap_slice[r, :], dark_thresh=110, min_len=min_consec):
                    return True
        return False
    else:
        # When yA2 >= yB1: check if a continuous horizontal dark frame line sits at the overlap boundary
        cy_min = max(0, min(cyA, cyB))
        cy_max = min(H - 1, max(cyA, cyB))
        if cy_max - cy_min < 2:
            return False
        center_slice = gray[cy_min:cy_max + 1, x_left:x_right]
        if center_slice.size > 0 and center_slice.shape[1] >= 4:
            dark_counts = np.sum(center_slice < dark_thresh, axis=1)
            row_ratios = dark_counts / float(center_slice.shape[1])
            max_consecutive = _max_true_run_1d(row_ratios >= 0.85)
            if 1 <= max_consecutive <= 4:
                return True
            b_mid = int((min(yA2, yB2) + max(yA1, yB1)) / 2.0)
            b_y1 = max(cy_min, b_mid - 4)
            b_y2 = min(cy_max + 1, b_mid + 5)
            boundary_slice = gray[b_y1:b_y2, x_left:x_right]
            min_frame_len = max(12, int(0.60 * span))
            if boundary_slice.size > 0 and boundary_slice.shape[1] >= 4:
                for r in range(boundary_slice.shape[0]):
                    if _has_continuous_dark_segment(boundary_slice[r, :], dark_thresh=110, min_len=min_frame_len):
                        return True
        return False


def has_pinched_bubble_waist(gray, boxA, boxB, mask_u8: Optional[np.ndarray] = None) -> bool:
    """
    Detects whether two side-by-side text columns lie in separate lobes of a pinched
    (figure-8 / double-oval / scalloped double) speech bubble with inward contour cusps
    between them. Never triggers on open canvases or convex single ovals/rectangles.
    """
    if gray is None or gray.size == 0 or boxA is None or boxB is None:
        return False

    H, W = gray.shape[:2]
    left_box, right_box = (boxA, boxB) if boxA[0] <= boxB[0] else (boxB, boxA)
    gap_x = right_box[0] - left_box[2]
    y_overlap = min(left_box[3], right_box[3]) - max(left_box[1], right_box[1])
    if gap_x < 0 or y_overlap <= 0:
        return False

    comb_y1 = max(0, min(left_box[1], right_box[1]))
    comb_y2 = min(H, max(left_box[3], right_box[3]))
    comb_x1 = max(0, min(left_box[0], right_box[0]))
    comb_x2 = min(W, max(left_box[2], right_box[2]))
    comb_h = max(1, comb_y2 - comb_y1)

    cxL = int((left_box[0] + left_box[2]) / 2.0)
    cxR = int((right_box[0] + right_box[2]) / 2.0)
    if cxR - cxL < max(12, int(0.25 * comb_h)):
        return False

    comb_crop = gray[comb_y1:comb_y2, comb_x1:comb_x2]
    if comb_crop.size > 0 and float(np.median(comb_crop)) < 95:
        return False

    pad_y = max(70, int(0.65 * comb_h))
    ry1 = max(0, comb_y1 - pad_y)
    ry2 = min(H, comb_y2 + pad_y)
    rx1 = max(0, cxL)
    rx2 = min(W, cxR + 1)
    w_len = rx2 - rx1
    ph = ry2 - ry1
    if w_len < 24 or ph < 24:
        return False

    patch = gray[ry1:ry2, rx1:rx2]
    if float(np.min(patch)) >= 145:
        return False

    gap_mid_lx = int(((left_box[2] + right_box[0]) / 2.0) - rx1)
    neck_half = max(3, int(round(0.18 * w_len)))
    n1 = max(int(0.15 * w_len), gap_mid_lx - neck_half)
    n2 = min(int(0.85 * w_len), gap_mid_lx + neck_half + 1)
    if n1 < 4 or n2 - n1 < 3 or w_len - n2 < 4:
        n1 = max(4, int(0.25 * w_len))
        n2 = min(w_len - 4, int(0.75 * w_len))
        if n2 <= n1 + 2:
            return False

    border = (patch < 145).astype(np.uint8)
    if mask_u8 is not None and mask_u8.size > 0 and mask_u8.shape[0] >= ry2 and mask_u8.shape[1] >= rx2:
        mask_patch = (mask_u8[ry1:ry2, rx1:rx2] > 35).astype(np.uint8)
        if np.any(mask_patch):
            mask_dil = cv2.dilate(mask_patch, np.ones((3, 3), dtype=np.uint8), iterations=1)
            border[mask_dil > 0] = 0

    for b in (left_box, right_box):
        ibx1 = max(0, b[0] - rx1)
        iby1 = max(0, b[1] - ry1)
        ibx2 = min(w_len, b[2] - rx1)
        iby2 = min(ph, b[3] - ry1)
        if ibx2 > ibx1 and iby2 > iby1:
            if gap_x < 4:
                # Touching split-half probes: preserve neck band [n1:n2] so pinched walls are not erased
                if ibx1 < n1:
                    border[iby1:iby2, ibx1:min(ibx2, n1)] = 0
                if ibx2 > n2:
                    border[iby1:iby2, max(ibx1, n2):ibx2] = 0
            else:
                border[iby1:iby2, ibx1:ibx2] = 0

    # Filter border components: keep only components connected to the outer bubble perimeter
    # (pixels above comb_y1 - ry1 or below comb_y2 - ry1), erasing isolated interior text strokes.
    cy_local_top = max(0, comb_y1 - ry1)
    cy_local_bot = min(ph, comb_y2 - ry1)
    if cy_local_top > 0 or cy_local_bot < ph:
        n_cc, cc_lab = cv2.connectedComponents(border, connectivity=8)
        if n_cc > 1:
            outer_lbls = set()
            if cy_local_top > 0:
                outer_lbls.update(int(v) for v in np.unique(cc_lab[:cy_local_top, :]) if v > 0)
            if cy_local_bot < ph:
                outer_lbls.update(int(v) for v in np.unique(cc_lab[cy_local_bot:, :]) if v > 0)
            if outer_lbls:
                border = np.isin(cc_lab, list(outer_lbls)).astype(np.uint8)
            else:
                border[:] = 0

    yc = max(0, min(ph - 1, (max(left_box[1], right_box[1]) + min(left_box[3], right_box[3])) // 2 - ry1))
    tops = np.zeros(w_len, dtype=np.float32)
    bots = np.zeros(w_len, dtype=np.float32)
    top_hit = np.zeros(w_len, dtype=bool)
    bot_hit = np.zeros(w_len, dtype=bool)

    for lx in range(w_len):
        gx = rx1 + lx
        if gap_x < 4 and n1 <= lx < n2:
            y_up = yc
            y_dn = yc
        elif gx <= left_box[2] - 2:
            y_up = max(0, min(ph - 1, left_box[1] - ry1))
            y_dn = max(0, min(ph - 1, left_box[3] - ry1))
        elif gx >= right_box[0] + 2:
            y_up = max(0, min(ph - 1, right_box[1] - ry1))
            y_dn = max(0, min(ph - 1, right_box[3] - ry1))
        else:
            y_up = yc
            y_dn = yc

        ty = -1
        for yy in range(y_up, -1, -1):
            if border[yy, lx]:
                ty = yy
                break
        if ty >= 0:
            tops[lx] = float(ry1 + ty)
            top_hit[lx] = True
        else:
            tops[lx] = float(ry1)

        by = -1
        for yy in range(y_dn, ph):
            if border[yy, lx]:
                by = yy
                break
        if by >= 0:
            bots[lx] = float(ry1 + by)
            bot_hit[lx] = True
        else:
            bots[lx] = float(ry1 + ph - 1)

    if w_len >= 5:
        tops_med = cv2.medianBlur(tops.reshape(1, -1), 5).ravel()
        bots_med = cv2.medianBlur(bots.reshape(1, -1), 5).ravel()
    else:
        tops_med = tops
        bots_med = bots

    valid_top = (
        float(np.mean(top_hit[:n1])) >= 0.50
        and float(np.mean(top_hit[n1:n2])) >= 0.50
        and float(np.mean(top_hit[n2:])) >= 0.50
    )
    valid_bot = (
        float(np.mean(bot_hit[:n1])) >= 0.50
        and float(np.mean(bot_hit[n1:n2])) >= 0.50
        and float(np.mean(bot_hit[n2:])) >= 0.50
    )
    if not valid_top and not valid_bot:
        return False

    top_cusp = (
        float(np.max(tops_med[n1:n2])) - max(float(np.min(tops_med[:n1])), float(np.min(tops_med[n2:])))
        if valid_top else 0.0
    )
    bot_cusp = (
        min(float(np.max(bots_med[:n1])), float(np.max(bots_med[n2:]))) - float(np.min(bots_med[n1:n2]))
        if valid_bot else 0.0
    )

    spans_med = bots_med - tops_med
    neck_span_min = float(np.min(spans_med[n1:n2]))
    if neck_span_min < 4.0:
        return False

    neck_white_ratios = []
    for lx in range(n1, n2):
        t_y = max(0, min(ph - 1, int(round(tops_med[lx] - ry1)) + 1))
        b_y = max(0, min(ph, int(round(bots_med[lx] - ry1))))
        if b_y > t_y:
            neck_white_ratios.append(float(np.mean(patch[t_y:b_y, lx] >= 170)))
    if not neck_white_ratios or float(np.median(neck_white_ratios)) < 0.50:
        return False

    ref_span = min(float(np.max(spans_med[:n1])), float(np.max(spans_med[n2:])))
    pinch_ratio = neck_span_min / max(1.0, ref_span)

    if (
        top_cusp >= max(4.0, 0.08 * ref_span)
        and bot_cusp >= max(4.0, 0.08 * ref_span)
        and (top_cusp + bot_cusp) >= max(10.0, 0.18 * ref_span)
        and pinch_ratio <= 0.90
    ):
        return True
    if max(top_cusp, bot_cusp) >= max(8.0, 0.15 * ref_span) and pinch_ratio <= 0.85:
        return True
    return False


# Aliases for compatibility
_get_box_border_mask = get_box_border_mask
_line_crosses_border_mask = line_crosses_border_mask
_is_dark_barrier_between_x = is_dark_barrier_between_x
_is_dark_barrier_between_y = is_dark_barrier_between_y

__all__ = [
    "_white_cc_cache",
    "_pipeline_cache_lock",
    "_get_gray_cache_key",
    "get_white_cc_labels",
    "_get_box_white_mode_label",
    "are_different_white_contours",
    "get_box_border_mask",
    "_get_box_border_mask",
    "line_crosses_border_mask",
    "_line_crosses_border_mask",
    "_has_continuous_dark_segment",
    "is_dark_barrier_between_x",
    "_is_dark_barrier_between_x",
    "is_dark_barrier_between_y",
    "_is_dark_barrier_between_y",
    "has_pinched_bubble_waist",
]
