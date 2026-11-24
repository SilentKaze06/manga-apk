"""
Geometry and spatial barrier utilities for manga text detection and clustering.
Contains border mask computation, ray-crossing barriers, white connected components,
and dark barrier detection.
"""

import math
import hashlib
import threading
import logging
from typing import Optional, List, Tuple, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_white_cc_cache = {}
_pipeline_cache_lock = threading.Lock()


def _get_gray_cache_key(gray):
    H, W = gray.shape[:2]
    step_y = max(1, H // 32)
    step_x = max(1, W // 32)
    sub = np.ascontiguousarray(gray[::step_y, ::step_x])
    h = hashlib.sha256(sub.tobytes()).hexdigest()[:16]
    return (H, W, h)


def get_white_cc_labels(gray, return_stats=False):
    """
    Computes 8-way connected components and stats for white background pixels (gray > 200).
    Uses content-hashed locked cache to avoid address reuse collisions.
    """
    if gray is None or gray.size == 0:
        return (None, None) if return_stats else None
    cache_key = _get_gray_cache_key(gray)
    with _pipeline_cache_lock:
        if cache_key in _white_cc_cache:
            labels, stats = _white_cc_cache[cache_key]
            return (labels, stats) if return_stats else labels
    white_mask = (gray > 200).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    with _pipeline_cache_lock:
        if len(_white_cc_cache) > 16:
            _white_cc_cache.clear()
        _white_cc_cache[cache_key] = (labels, stats)
    return (labels, stats) if return_stats else labels


def _get_box_white_mode_label(labels, stats, gray, box, min_area=200, pad=3):
    """
    Returns the mode (most frequent non-zero with area > min_area) white component label
    from a 2-4 px expanded perimeter around the box.
    """
    if labels is None or gray is None:
        return 0
    H, W = gray.shape[:2]
    xA1, yA1, xA2, yA2 = box
    px1, py1 = max(0, int(xA1) - pad), max(0, int(yA1) - pad)
    px2, py2 = min(W, int(xA2) + pad), min(H, int(yA2) + pad)
    if px2 <= px1 or py2 <= py1:
        return 0

    sub_labels = labels[py1:py2, px1:px2]
    sub_gray = gray[py1:py2, px1:px2]

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
            if area > min_area:
                return lbl
        elif stats is None:
            return lbl
    return 0


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
    so bold/SFX letters on high-res pages are not flagged as box borders.
    """
    if gray is None or gray.size == 0:
        return None
    H, W = gray.shape[:2]
    dark = (gray < 160).astype(np.uint8)
    kw = max(24, min(int(round(W * 0.035)), 80))
    kh = max(24, min(int(round(H * 0.035)), 80))
    line_kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    line_kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
    horiz_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, line_kernel_h)
    vert_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, line_kernel_v)

    # Exclude solid large dark regions (e.g. dark narration boxes, dark clothing/artwork)
    # so they are not treated as straight box borders
    thick_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 16))
    solid_dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, thick_kernel)
    horiz_lines = cv2.bitwise_and(horiz_lines, cv2.bitwise_not(solid_dark))
    vert_lines = cv2.bitwise_and(vert_lines, cv2.bitwise_not(solid_dark))
    return cv2.bitwise_or(horiz_lines, vert_lines)


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
        x_start = max(top_box[0], bot_box[0])
        x_end = min(top_box[2], bot_box[2])
        corridor_w = x_end - x_start

        if corridor_w >= 8 and top_box[3] < bot_box[1]:
            y_gap_top = top_box[3]
            y_gap_bot = bot_box[1]
            search_y1 = max(0, y_gap_top - 2)
            search_y2 = min(H, y_gap_bot + 3)
            min_consec_w = max(10, int(0.35 * corridor_w))

            for r in range(search_y1, search_y2):
                row_slice = border_mask[r, x_start:x_end]
                if np.mean(row_slice > 0) >= 0.45:
                    return True
                max_run = 0
                cur_run = 0
                for val in row_slice:
                    if val > 0:
                        cur_run += 1
                        if cur_run > max_run:
                            max_run = cur_run
                    else:
                        cur_run = 0
                if max_run >= min_consec_w:
                    return True

        # 2. Horizontally adjacent / facing boxes (columns)
        left_box = boxA if xA1 < xB1 else boxB
        right_box = boxB if xA1 < xB1 else boxA
        y_start = max(left_box[1], right_box[1])
        y_end = min(left_box[3], right_box[3])
        corridor_h = y_end - y_start

        if corridor_h >= 8 and left_box[2] < right_box[0]:
            x_gap_left = left_box[2]
            x_gap_right = right_box[0]
            search_x1 = max(0, x_gap_left - 2)
            search_x2 = min(W, x_gap_right + 3)
            min_consec_h = max(10, int(0.35 * corridor_h))

            for c in range(search_x1, search_x2):
                col_slice = border_mask[y_start:y_end, c]
                if np.mean(col_slice > 0) >= 0.45:
                    return True
                max_run = 0
                cur_run = 0
                for val in col_slice:
                    if val > 0:
                        cur_run += 1
                        if cur_run > max_run:
                            max_run = cur_run
                    else:
                        cur_run = 0
                if max_run >= min_consec_h:
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
    is_dark = (arr_1d < dark_thresh)
    consec = 0
    for v in is_dark:
        if v:
            consec += 1
            if consec >= min_len:
                return True
        else:
            consec = 0
    return False


def is_dark_barrier_between_x(gray, boxA, boxB, dark_thresh=100, min_line_ratio=0.40, border_mask=None):
    if gray is None or gray.size == 0:
        return False
    H, W = gray.shape[:2]
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB

    # Check if both boxes sit inside a dark narration box
    comb_crop = gray[min(yA1, yB1):max(yA2, yB2), min(xA1, xB1):max(xA2, xB2)]
    if comb_crop.size > 0:
        med_val = float(np.median(comb_crop))
        if med_val < 95 or (med_val < 120 and float(np.mean(comb_crop < 110)) >= 0.50):
            # Inside a dark narration box: dark pixels are normal background, not a barrier
            y_top = max(yA1, yB1)
            y_bot = min(yA2, yB2)
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
    if border_mask is not None and line_crosses_border_mask(border_mask, (cxA, cyA), (cxB, cyB), boxA=boxA, boxB=boxB):
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
        if xB1_c - xA2_c <= 8:
            b_left = max(0, xA2_c - 3)
            b_right = min(W, xB1_c + 3)
            b_slice = gray[y_top:y_bot, b_left:b_right]
            if b_slice.size > 0 and b_slice.shape[0] >= 4:
                for c in range(b_slice.shape[1]):
                    if _has_continuous_dark_segment(b_slice[:, c], dark_thresh=110, min_len=min_consec):
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
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB

    # Check if both boxes sit inside a dark narration box
    comb_crop = gray[min(yA1, yB1):max(yA2, yB2), min(xA1, xB1):max(xA2, xB2)]
    if comb_crop.size > 0:
        med_val = float(np.median(comb_crop))
        if med_val < 95 or (med_val < 120 and float(np.mean(comb_crop < 110)) >= 0.50):
            # Inside a dark narration box: dark pixels are normal background, not a barrier
            x_left = max(xA1, xB1)
            x_right = min(xA2, xB2)
            if x_right > x_left and yA2 < yB1:
                gap_slice = gray[max(0, yA2):min(H, yB1), max(0, x_left):min(W, x_right)]
                if gap_slice.size > 0 and np.mean(gap_slice > 185) >= 0.40:
                    return True
            return False

    cxA = int((xA1 + xA2) / 2.0)
    cyA = int((yA1 + yA2) / 2.0)
    cxB = int((xB1 + xB2) / 2.0)
    cyB = int((yB1 + yB2) / 2.0)
    if border_mask is None:
        border_mask = get_box_border_mask(gray)
    if border_mask is not None and line_crosses_border_mask(border_mask, (cxA, cyA), (cxB, cyB), boxA=boxA, boxB=boxB):
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
        if yB1_c - yA2_c <= 8:
            b_top = max(0, yA2_c - 3)
            b_bot = min(H, yB1_c + 3)
            b_slice = gray[b_top:b_bot, x_left:x_right]
            if b_slice.size > 0 and b_slice.shape[1] >= 4:
                for r in range(b_slice.shape[0]):
                    if _has_continuous_dark_segment(b_slice[r, :], dark_thresh=110, min_len=min_consec):
                        return True
        return False
    else:
        # When yA2 >= yB1:
        # Check if there is a continuous horizontal dark frame line between the two boxes.
        cyA = int((yA1 + yA2) / 2.0)
        cyB = int((yB1 + yB2) / 2.0)
        cy_min = max(0, min(cyA, cyB))
        cy_max = min(H - 1, max(cyA, cyB))
        if cy_max - cy_min < 2:
            return False
        center_slice = gray[cy_min:cy_max + 1, x_left:x_right]
        if center_slice.size > 0 and center_slice.shape[1] >= 4:
            dark_counts = np.sum(center_slice < dark_thresh, axis=1)
            row_ratios = dark_counts / float(center_slice.shape[1])
            dark_rows = (row_ratios >= 0.70)
            if np.any(dark_rows):
                consecutive = 0
                max_consecutive = 0
                for is_dark in dark_rows:
                    if is_dark:
                        consecutive += 1
                        max_consecutive = max(max_consecutive, consecutive)
                    else:
                        consecutive = 0
                if 1 <= max_consecutive <= 4:
                    return True
            b_mid = int((min(yA2, yB2) + max(yA1, yB1)) / 2.0)
            b_y1 = max(cy_min, b_mid - 4)
            b_y2 = min(cy_max + 1, b_mid + 5)
            boundary_slice = gray[b_y1:b_y2, x_left:x_right]
            if boundary_slice.size > 0 and boundary_slice.shape[1] >= 4:
                for r in range(boundary_slice.shape[0]):
                    if _has_continuous_dark_segment(boundary_slice[r, :], dark_thresh=110, min_len=min_consec):
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
]
