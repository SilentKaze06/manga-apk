"""
Text/bubble detection, clustering, column splitting, geometric barriers, and contour expansion.
"""
import hashlib
import logging
import re
import threading
import cv2
import numpy as np

try:
    from ocr_engine import is_manga_punctuation, normalize_manga_punctuation, is_sfx_text
except (ImportError, ValueError):
    from .ocr_engine import is_manga_punctuation, normalize_manga_punctuation, is_sfx_text

logger = logging.getLogger(__name__)

_white_cc_cache = {}
_pipeline_cache_lock = threading.Lock()


def _crop_reading_key(r, reading_direction: str = "rtl", col_map: dict = None, line_idx_map: dict = None):
    """
    Sort key function for OCR items inside a crop, supporting RTL manga reading direction
    and multi-column/line grouping.
    """
    col = col_map.get(id(r), 0) if col_map else 0
    line_idx = line_idx_map.get(id(r), None) if line_idx_map else None

    if isinstance(r, dict) and "box" in r:
        box = r["box"]
        y_c = (box[1] + box[3]) / 2.0
        x1 = box[0]
        x_c = (box[0] + box[2]) / 2.0
    elif isinstance(r, (list, tuple)) and len(r) > 0 and isinstance(r[0], (list, tuple)):
        ys = [pt[1] / 2.0 if isinstance(pt, (list, tuple)) else pt for pt in r[0]]
        xs = [pt[0] / 2.0 if isinstance(pt, (list, tuple)) else pt for pt in r[0]]
        y_c = (min(ys) + max(ys)) / 2.0
        x1 = min(xs)
        x_c = (min(xs) + max(xs)) / 2.0
    elif isinstance(r, (list, tuple)) and len(r) >= 4 and not isinstance(r[0], (list, tuple)):
        y_c = (r[1] + r[3]) / 2.0
        x1 = r[0]
        x_c = (r[0] + r[2]) / 2.0
    else:
        y_c = 0.0
        x1 = 0.0
        x_c = 0.0

    if col_map is not None or line_idx_map is not None:
        y_val = line_idx if line_idx is not None else y_c
        return (col, y_val, x1)

    if reading_direction == "rtl":
        return (y_c, -x_c)
    else:
        return (y_c, x1)


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


def can_merge_text_boxes(boxA, boxB, gray=None, textA="", textB="", box_border_mask=None, border_mask=None, max_page_h=2000):
    if box_border_mask is None:
        box_border_mask = border_mask

    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB

    # Detect if both lines sit inside a dark narration box
    comb_crop = gray[min(yA1, yB1):max(yA2, yB2), min(xA1, xB1):max(xA2, xB2)] if gray is not None and gray.size > 0 else np.array([])
    is_in_dark_box = False
    if comb_crop.size > 0:
        med_comb = float(np.median(comb_crop))
        if med_comb < 95 or (med_comb < 120 and float(np.mean(comb_crop < 110)) >= 0.45):
            is_in_dark_box = True

    if not is_in_dark_box and are_different_white_contours(gray, boxA, boxB):
        return False

    if not is_in_dark_box:
        if box_border_mask is None and gray is not None:
            box_border_mask = get_box_border_mask(gray)

        if box_border_mask is not None and box_border_mask.size > 0:
            cxA = int((boxA[0] + boxA[2]) / 2.0)
            cyA = int((boxA[1] + boxA[3]) / 2.0)
            cxB = int((boxB[0] + boxB[2]) / 2.0)
            cyB = int((boxB[1] + boxB[3]) / 2.0)
            if line_crosses_border_mask(box_border_mask, (cxA, cyA), (cxB, cyB), boxA=boxA, boxB=boxB):
                return False

    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB
    wA, hA = max(1, xA2 - xA1), max(1, yA2 - yA1)
    wB, hB = max(1, xB2 - xB1), max(1, yB2 - yB1)

    is_vert_A = hA > 1.25 * wA and wA > 20
    is_vert_B = hB > 1.25 * wB and wB > 20
    is_both_vertical = is_vert_A and is_vert_B

    # Horizontal text (standard English/Russian/Uzbek manga dialogue)
    if not is_both_vertical:
        # Rule 1: Words on the exact same horizontal line
        y_overlap = min(yA2, yB2) - max(yA1, yB1)
        min_h = min(hA, hB)
        if min_h > 0 and (y_overlap / min_h) >= 0.65:
            gap_x = max(0, max(xA1, xB1) - min(xA2, xB2))
            if gap_x <= max(8, int(min_h * 0.35)):
                left_box = boxA if xA1 < xB1 else boxB
                right_box = boxB if xA1 < xB1 else boxA
                left_text = (textA if xA1 < xB1 else textB).strip()
                right_text = (textB if xA1 < xB1 else textA).strip()
                if (left_text.endswith(('!', '?')) or right_text.endswith(('!', '?'))) and gap_x > 0:
                    return False
                if not is_dark_barrier_between_x(gray, left_box, right_box):
                    return True

        # Rule 2: Consecutive lines stacked vertically in the same dialogue bubble
        top_box = boxA if yA1 < yB1 else boxB
        bot_box = boxB if yA1 < yB1 else boxA
        top_text = (textA if yA1 < yB1 else textB).strip()
        bot_text = (textB if yA1 < yB1 else textA).strip()

        # Check dark barrier in X for side-by-side / diagonal / staggered boxes
        left_box = boxA if xA1 < xB1 else boxB
        right_box = boxB if xA1 < xB1 else boxA
        if is_dark_barrier_between_x(gray, left_box, right_box):
            return False

        gap_y = max(0, bot_box[1] - top_box[3])
        avg_h = (hA + hB) / 2.0

        if not is_in_dark_box:
            # If there is a horizontal dark frame line between two boxes (like stacked #3 narration boxes),
            # DO NOT MERGE them even if gap_y is small.
            if is_dark_barrier_between_y(gray, top_box, bot_box):
                return False
            if gray is not None and gray.size > 0:
                H_img, W_img = gray.shape[:2]
                b_x1 = max(0, max(top_box[0], bot_box[0]))
                b_x2 = min(W_img, min(top_box[2], bot_box[2]))
                if b_x2 - b_x1 >= 8:
                    span_w = b_x2 - b_x1
                    min_c = max(12, int(0.25 * span_w))
                    b_y1 = max(0, min(top_box[3], bot_box[1]) - 4)
                    b_y2 = min(H_img, max(top_box[3], bot_box[1]) + 5)
                    for r in range(b_y1, b_y2):
                        if _has_continuous_dark_segment(gray[r, b_x1:b_x2], dark_thresh=110, min_len=min_c):
                            return False

        # Check if both lines sit inside the same speech bubble interior
        crop_between = gray[top_box[1]:bot_box[3], min(xA1, xB1):max(xA2, xB2)] if gray is not None and gray.size > 0 else np.array([])
        is_same_bubble_interior = (
            is_in_dark_box
            or (
                crop_between.size > 0
                and (np.mean(crop_between) > 150 or np.percentile(crop_between, 85) > 185 or np.mean(crop_between) < 85)
                and not is_dark_barrier_between_y(gray, top_box, bot_box)
                and not is_dark_barrier_between_x(gray, left_box, right_box)
            )
        )

        overlap = min(xA2, xB2) - max(xA1, xB1)
        if not is_same_bubble_interior:
            # Different regions or crossing barriers: do not merge across sentences
            if top_text.endswith(('!', '?')) and (
                gap_y > 0 and (overlap < min(wA, wB) * 0.70 or bot_text.endswith(('!', '?')) or gap_y > 4)
            ):
                return False
            if top_text.endswith(('.', '…', '!', '?')) and (
                gap_y > 6 or (gap_y > 0 and overlap < min(wA, wB) * 0.45)
            ):
                return False
        else:
            # Inside the same speech bubble interior: do NOT split multi-sentence bubbles at '.'
            if (top_text.endswith(('!', '?')) and bot_text.endswith(('!', '?'))) and gap_y > max(16, int(avg_h * 0.85)):
                return False

        # Maximum allowed gap between consecutive lines in one sentence/paragraph
        max_gap_y = max(16, int(avg_h * 0.85)) if is_same_bubble_interior else max(14, int(avg_h * 0.70))
        if gap_y <= max_gap_y:
            midA = (xA1 + xA2) / 2.0
            midB = (xB1 + xB2) / 2.0
            x_overlap = min(xA2, xB2) - max(xA1, xB1)
            min_w = min(wA, wB)

            # Must have horizontal overlap OR closely aligned centers
            has_align = (min_w > 0 and (x_overlap / min_w) >= 0.25) or abs(midA - midB) <= max(24, int(min_w * 0.35))
            if is_same_bubble_interior:
                has_align = has_align or (x_overlap > 0) or abs(midA - midB) <= max(35, int(min_w * 0.50))

            if has_align:
                crop_area = gray[top_box[1]:bot_box[3], min(xA1, xB1):max(xA2, xB2)] if gray is not None and gray.size > 0 else np.array([])
                if crop_area.size == 0:
                    return False
                is_bubble_bg = (
                    is_in_dark_box or
                    np.mean(crop_area) > 70 or
                    np.median(crop_area) > 70 or
                    np.percentile(crop_area, 85) > 140 or
                    np.mean(crop_area) < 80
                )
                if is_bubble_bg and (is_in_dark_box or (not is_dark_barrier_between_y(gray, top_box, bot_box) and not is_dark_barrier_between_x(gray, left_box, right_box))):
                    return True

    # Traditional vertical text (Japanese)
    if is_both_vertical:
        x_overlap = min(xA2, xB2) - max(xA1, xB1)
        min_w = min(wA, wB)
        if min_w > 0 and (x_overlap / min_w) >= 0.60:
            gap_y = max(0, max(yA1, yB1) - min(yA2, yB2))
            if gap_y <= max(8, int(min_w * 0.70)):
                top_box = boxA if yA1 < yB1 else boxB
                bot_box = boxB if yA1 < yB1 else boxA
                if not is_dark_barrier_between_y(gray, top_box, bot_box):
                    return True

        y_overlap = min(yA2, yB2) - max(yA1, yB1)
        min_h = min(hA, hB)
        if min_h > 0 and (y_overlap / min_h) >= 0.35:
            gap_x = max(0, max(xA1, xB1) - min(xA2, xB2))
            avg_w = (wA + wB) / 2.0
            if gap_x <= max(14, int(avg_w * 1.15)):
                left_box = boxA if xA1 < xB1 else boxB
                right_box = boxB if xA1 < xB1 else boxA
                if not is_dark_barrier_between_x(gray, left_box, right_box):
                    return True

    return False


def detect_stepped_box_split_x(gray, cluster_box):
    """
    Detects stepped (Z-shaped / L-shaped) narration / speech box split boundary.
    Extracts the main white connected component in gray > 200 within cluster_box.
    If fill_ratio < 0.80 and at least one corner (15%x15% zone) does not belong to the
    main white component, checks for a sharp step transition in y_top(x) or y_bot(x)
    where abs(diff) >= max(15, int(0.15 * bh)).
    Returns absolute image coordinate x_split = x1 + step_x, or None.
    """
    if gray is None or gray.size == 0 or cluster_box is None:
        return None

    if isinstance(cluster_box, dict):
        cluster_box = cluster_box.get("box", cluster_box.get("text_box", cluster_box))
    if not isinstance(cluster_box, (list, tuple)) or len(cluster_box) < 4:
        return None

    gh, gw = gray.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in cluster_box[:4]]
    x1 = max(0, min(gw, x1))
    x2 = max(0, min(gw, x2))
    y1 = max(0, min(gh, y1))
    y2 = max(0, min(gh, y2))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 10 or bh <= 10:
        return None

    patch = gray[y1:y2, x1:x2]
    white_mask = (patch > 200).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    if num_labels <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_label = 1 + int(np.argmax(areas))
    main_white_comp = (labels == max_label)
    white_area = int(stats[max_label, cv2.CC_STAT_AREA])
    fill_ratio = white_area / float(bw * bh)

    if fill_ratio >= 0.80:
        return None

    # Check corners (15% x 15% zones)
    cw = max(1, int(0.15 * bw))
    ch = max(1, int(0.15 * bh))

    tl = main_white_comp[0:ch, 0:cw]
    tr = main_white_comp[0:ch, bw - cw:bw]
    bl = main_white_comp[bh - ch:bh, 0:cw]
    br = main_white_comp[bh - ch:bh, bw - cw:bw]

    corner_means = [float(np.mean(tl)), float(np.mean(tr)), float(np.mean(bl)), float(np.mean(br))]
    if not any(m < 0.30 for m in corner_means):
        return None

    # Compute top and bottom white boundaries for each column
    y_top = np.zeros(bw, dtype=np.int32)
    y_bot = np.zeros(bw, dtype=np.int32)
    valid = np.zeros(bw, dtype=bool)

    for x in range(bw):
        rows = np.where(main_white_comp[:, x])[0]
        if len(rows) > 0:
            y_top[x] = rows[0]
            y_bot[x] = rows[-1]
            valid[x] = True

    if not np.any(valid):
        return None

    valid_indices = np.where(valid)[0]
    for x in range(bw):
        if not valid[x]:
            nearest = valid_indices[np.argmin(np.abs(valid_indices - x))]
            y_top[x] = y_top[nearest]
            y_bot[x] = y_bot[nearest]

    min_jump = max(15, int(0.15 * bh))
    x_start = max(1, int(0.10 * bw))
    x_end = min(bw - 1, int(0.90 * bw))

    step_top = None
    max_d_top = 0
    for x in range(x_start, x_end):
        for k in (1, 2, 3):
            if x + k < bw:
                d = abs(int(y_top[x + k]) - int(y_top[x]))
                if d >= min_jump and d > max_d_top:
                    max_d_top = d
                    step_top = x + (k + 1) // 2

    step_bot = None
    max_d_bot = 0
    for x in range(x_start, x_end):
        for k in (1, 2, 3):
            if x + k < bw:
                d = abs(int(y_bot[x + k]) - int(y_bot[x]))
                if d >= min_jump and d > max_d_bot:
                    max_d_bot = d
                    step_bot = x + (k + 1) // 2

    if step_top is not None and step_bot is not None:
        step_x = int(round((step_top + step_bot) / 2.0))
    elif step_top is not None:
        step_x = step_top
    elif step_bot is not None:
        step_x = step_bot
    else:
        return None

    return x1 + step_x


def split_cluster_into_columns(box_indices, raw_boxes, reading_direction: str = "rtl", gray=None):
    """
    Detects if a clustered text group contains multiple distinct vertical columns
    (e.g., adjacent dialogue, loud dialogue with small aside text, or multi-speaker bubbles),
    or a stepped (Z-shaped / L-shaped) narration box.
    If distinct non-overlapping columns exist with overlapping vertical spans,
    splits them into separate sub-clusters in manga reading order (right-to-left).
    """
    if len(box_indices) <= 1:
        return [box_indices]

    # Check stepped (Z-shaped / L-shaped) narration box splitting if gray image is provided
    if gray is not None and len(box_indices) >= 2:
        bx1 = min(raw_boxes[idx]["box"][0] for idx in box_indices)
        by1 = min(raw_boxes[idx]["box"][1] for idx in box_indices)
        bx2 = max(raw_boxes[idx]["box"][2] for idx in box_indices)
        by2 = max(raw_boxes[idx]["box"][3] for idx in box_indices)
        cluster_box = [bx1, by1, bx2, by2]

        x_split = None
        try:
            bubble_box = expand_to_bubble_contour(gray, cluster_box)
            x_split = detect_stepped_box_split_x(gray, bubble_box)
        except Exception as e_step:
            logger.debug("detect_stepped_box_split_x on bubble_box error: %s", e_step)
        if x_split is None:
            try:
                x_split = detect_stepped_box_split_x(gray, cluster_box)
            except Exception as e_step2:
                logger.debug("detect_stepped_box_split_x on cluster_box error: %s", e_step2)

        if x_split is not None:
            left_indices = [
                idx for idx in box_indices
                if (raw_boxes[idx]["box"][0] + raw_boxes[idx]["box"][2]) / 2.0 < x_split
            ]
            right_indices = [
                idx for idx in box_indices
                if (raw_boxes[idx]["box"][0] + raw_boxes[idx]["box"][2]) / 2.0 >= x_split
            ]
            if left_indices and right_indices:
                res_left = split_cluster_into_columns(left_indices, raw_boxes, reading_direction=reading_direction, gray=gray)
                res_right = split_cluster_into_columns(right_indices, raw_boxes, reading_direction=reading_direction, gray=gray)
                if reading_direction == "rtl":
                    return res_right + res_left
                else:
                    return res_left + res_right

    if len(box_indices) <= 2:
        return [box_indices]

    items = [raw_boxes[idx] for idx in box_indices]
    spans = [(b["box"][0], b["box"][2], idx) for idx, b in zip(box_indices, items)]

    min_x = min(s[0] for s in spans)
    max_x = max(s[1] for s in spans)
    w_span = max_x - min_x
    if w_span < 40:
        return [box_indices]

    # Sort boxes by horizontal center (left to right)
    sorted_by_x = sorted(zip(box_indices, items), key=lambda x: (x[1]["box"][0] + x[1]["box"][2]) / 2.0)

    best_split_idx = -1
    max_gap = -9999

    for i in range(len(sorted_by_x) - 1):
        left_group = sorted_by_x[:i+1]
        right_group = sorted_by_x[i+1:]

        left_max_x = max(item[1]["box"][2] for item in left_group)
        right_min_x = min(item[1]["box"][0] for item in right_group)
        gap = right_min_x - left_max_x

        left_y1 = min(item[1]["box"][1] for item in left_group)
        left_y2 = max(item[1]["box"][3] for item in left_group)
        right_y1 = min(item[1]["box"][1] for item in right_group)
        right_y2 = max(item[1]["box"][3] for item in right_group)

        y_overlap = min(left_y2, right_y2) - max(left_y1, right_y1)
        min_h = min(left_y2 - left_y1, right_y2 - right_y1)

        mid_left = sum((item[1]["box"][0] + item[1]["box"][2]) / 2.0 for item in left_group) / len(left_group)
        mid_right = sum((item[1]["box"][0] + item[1]["box"][2]) / 2.0 for item in right_group) / len(right_group)

        has_barrier = False
        if gray is not None and gap >= -8:
            barrier_x = int(round((left_max_x + right_min_x) / 2.0))
            if 0 <= barrier_x < gray.shape[1]:
                y_top = max(0, min(left_y1, right_y1))
                y_bot = min(gray.shape[0], max(left_y2, right_y2))
                strip = gray[y_top:y_bot, max(0, barrier_x - 1):min(gray.shape[1], barrier_x + 2)]
                if strip.size > 0 and np.mean(strip < 85) >= 0.40:
                    has_barrier = True

        valid_column_split = (gap > 0) or has_barrier
        if min_h > 0 and y_overlap > 0.30 * min_h and (mid_right - mid_left) > 25 and valid_column_split:
            if gap > max_gap:
                max_gap = gap
                best_split_idx = i

    if best_split_idx != -1:
        left_indices = [x[0] for x in sorted_by_x[:best_split_idx+1]]
        right_indices = [x[0] for x in sorted_by_x[best_split_idx+1:]]
        is_cjk = any(
            any('\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u30ff' or '\u3400' <= ch <= '\u4dbf' for ch in raw_boxes[idx].get("text", ""))
            for idx in box_indices
        )
        res_left = split_cluster_into_columns(left_indices, raw_boxes, reading_direction=reading_direction, gray=gray)
        res_right = split_cluster_into_columns(right_indices, raw_boxes, reading_direction=reading_direction, gray=gray)
        if reading_direction == "rtl" or is_cjk:
            # In manga reading order (RTL), right column is read first, left column second
            return res_right + res_left
        else:
            # In LTR reading order, left column is read first, right column second
            return res_left + res_right

    return [box_indices]


def smart_cluster_text_boxes(raw_boxes, gray, reading_direction: str = "rtl"):
    """
    Groups individual text lines/words into distinct dialogue bubbles
    via adjacency graph connected components, and splits multi-column bubbles.
    Filters out spurious OCR noise and guarantees correct natural reading order.
    """
    if not raw_boxes:
        return []

    # Pre-filter spurious noise lines (tiny low-confidence artifacts, panel boundary lines, stray symbols)
    valid_boxes = []
    for rb in raw_boxes:
        t = rb.get("text", "").strip()
        # Normalize common comic font scanlation OCR substitutions (prevent corrupting CLASS 1-1, 1-A, 1st):
        t = re.sub(r'(?<![\d\-])\b1\b(?!\s*(?:st|nd|rd|th)\b)(?![\d\-])', 'I', t, flags=re.I)
        t = re.sub(r'\b1(?!(?:st|nd|rd|th)\b)(?=[A-Za-z])', 'I', t, flags=re.I)
        t = re.sub(r'\bUT\b', 'IT', t)
        rb["text"] = t
        conf = float(rb.get("confidence", rb.get("conf", 1.0)))
        box = rb["box"]
        bw = box[2] - box[0]
        bh = box[3] - box[1]

        # Expressive manga punctuation (?, !, ?!, ...) is strictly preserved!
        if is_manga_punctuation(t):
            rb["text"] = normalize_manga_punctuation(t)
            valid_boxes.append(rb)
            continue

        # Check if inside speech bubble interior (light > 160 or dark < 90)
        is_bubble_interior = False
        if gray is not None and gray.size > 0:
            gh, gw = gray.shape[:2]
            cx1 = max(0, min(gw, box[0]))
            cy1 = max(0, min(gh, box[1]))
            cx2 = max(0, min(gw, box[2]))
            cy2 = max(0, min(gh, box[3]))
            if cx2 > cx1 and cy2 > cy1:
                patch = gray[cy1:cy2, cx1:cx2]
                if patch.size > 0:
                    p_mean = float(np.mean(patch))
                    if p_mean > 160 or p_mean < 90:
                        is_bubble_interior = True

        # Ignore very low confidence noise or single stray non-Cyrillic marks
        # (Preserve single "I", short interjections, and valid bubble-interior trailing lines)
        has_cyr = any('\u0400' <= c <= '\u04ff' for c in t)
        clean_t = re.sub(r'[^\w]', '', t)
        is_short_interjection = (2 <= len(clean_t) <= 3 and any(c in "?!" for c in t) and clean_t.isalpha())
        is_short_valid = (clean_t.lower() in {"i", "it", "a", "no", "to", "so", "he", "we", "me", "my", "ok", "hm", "mm", "eh", "ha", "uh", "um", "yo", "ya"}) or is_short_interjection or (len(clean_t) >= 2 and clean_t.isalpha())
        if conf < 0.10 and len(t) <= 2 and not has_cyr:
            if not (is_bubble_interior and is_short_valid):
                continue
        # Ignore tall thin vertical artifacts (line noise, panel borders)
        # (Preserve single capital "I" inside a bubble interior)
        if bw <= 8 and bh >= 25 and len(t) <= 1:
            if not (is_bubble_interior and t.upper() == "I"):
                continue
        valid_boxes.append(rb)

    raw_boxes = valid_boxes
    n = len(raw_boxes)
    if n == 0:
        return []

    box_border_mask = get_box_border_mask(gray) if gray is not None else None
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if can_merge_text_boxes(
                raw_boxes[i]["box"], raw_boxes[j]["box"], gray,
                raw_boxes[i].get("text", ""), raw_boxes[j].get("text", ""),
                box_border_mask=box_border_mask
            ):
                adj[i].append(j)
                adj[j].append(i)

    visited = set()
    clusters = []
    for i in range(n):
        if i in visited:
            continue
        comp = []
        queue = [i]
        visited.add(i)
        while queue:
            cur = queue.pop(0)
            comp.append(cur)
            for neighbor in adj[cur]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Check if the cluster contains multiple side-by-side vertical columns
        sub_comps = split_cluster_into_columns(comp, raw_boxes, reading_direction=reading_direction, gray=gray)
        stepped_split_x = None
        if gray is not None and len(comp) >= 2:
            cbx1 = min(raw_boxes[idx]["box"][0] for idx in comp)
            cby1 = min(raw_boxes[idx]["box"][1] for idx in comp)
            cbx2 = max(raw_boxes[idx]["box"][2] for idx in comp)
            cby2 = max(raw_boxes[idx]["box"][3] for idx in comp)
            cbox = [cbx1, cby1, cbx2, cby2]
            try:
                bbox = expand_to_bubble_contour(gray, cbox)
                stepped_split_x = detect_stepped_box_split_x(gray, bbox)
            except Exception as e_bb:
                logger.debug("expand_to_bubble_contour error in smart_cluster_text_boxes: %s", e_bb)
            if stepped_split_x is None:
                try:
                    stepped_split_x = detect_stepped_box_split_x(gray, cbox)
                except Exception as e_cb:
                    logger.debug("detect_stepped_box_split_x error in smart_cluster_text_boxes: %s", e_cb)

        for sub_comp in sub_comps:
            # Check if the text is genuinely vertical Japanese text (not just one tall letter like 'I' or '1')
            vert_lines = sum(
                1 for idx in sub_comp
                if (raw_boxes[idx]["box"][3] - raw_boxes[idx]["box"][1]) > 1.4 * (raw_boxes[idx]["box"][2] - raw_boxes[idx]["box"][0])
                and len(raw_boxes[idx].get("text", "").strip()) > 1
            )
            is_vertical = (vert_lines / len(sub_comp)) >= 0.60 if len(sub_comp) >= 3 else False

            if is_vertical:
                sub_comp.sort(key=lambda idx: (-raw_boxes[idx]["box"][0], raw_boxes[idx]["box"][1]))
                lines = [[idx] for idx in sub_comp]
            else:
                # Group lines by vertical overlap y_overlap >= 0.45 * min(h1, h2)
                lines = []
                sorted_by_yc = sorted(sub_comp, key=lambda idx: (raw_boxes[idx]["box"][1] + raw_boxes[idx]["box"][3]) / 2.0)
                for idx in sorted_by_yc:
                    b = raw_boxes[idx]["box"]
                    h = b[3] - b[1]
                    matched_line = None
                    for line in lines:
                        l_y1 = min(raw_boxes[i]["box"][1] for i in line)
                        l_y2 = max(raw_boxes[i]["box"][3] for i in line)
                        l_min_h = min(raw_boxes[i]["box"][3] - raw_boxes[i]["box"][1] for i in line)
                        y_ov = min(b[3], l_y2) - max(b[1], l_y1)
                        if y_ov >= 0.45 * min(h, l_min_h):
                            matched_line = line
                            break
                    if matched_line is not None:
                        matched_line.append(idx)
                    else:
                        lines.append([idx])

                # Sort boxes within each line by x1 (left to right)
                for line in lines:
                    line.sort(key=lambda idx: raw_boxes[idx]["box"][0])
                # Sort lines top-to-bottom
                lines.sort(key=lambda line: min(raw_boxes[idx]["box"][1] for idx in line))
                sub_comp = [idx for line in lines for idx in line]

            bx1 = min(raw_boxes[idx]["box"][0] for idx in sub_comp)
            by1 = min(raw_boxes[idx]["box"][1] for idx in sub_comp)
            bx2 = max(raw_boxes[idx]["box"][2] for idx in sub_comp)
            by2 = max(raw_boxes[idx]["box"][3] for idx in sub_comp)

            # Preserve multi-sentence line breaks (\n) and restore line-end hyphenated words
            # e.g., TSU- at end of line + MIKI'S at start of next line -> TSUMIKI'S
            full_parts = []
            for l_idx, line in enumerate(lines):
                line_words = []
                for idx in line:
                    lt = raw_boxes[idx].get("text", "").strip()
                    lt = re.sub(r'^[1lI]I\s*', 'I ', lt)
                    line_words.append(lt)
                line_text = " ".join(line_words).strip()
                if not line_text:
                    continue

                if not full_parts:
                    full_parts.append(line_text)
                else:
                    prev_text = full_parts[-1]
                    if re.search(r'[A-Za-z]+-$', prev_text) and re.match(r'^[A-Za-z\']+', line_text):
                        # Restore hyphenated word broken across line end
                        full_parts[-1] = re.sub(r'-$', '', full_parts[-1]) + line_text
                    else:
                        prev_line = lines[l_idx - 1]
                        prev_y2 = max(raw_boxes[i]["box"][3] for i in prev_line)
                        curr_y1 = min(raw_boxes[i]["box"][1] for i in line)
                        gap = curr_y1 - prev_y2
                        if (prev_text.endswith(('.', '…', '!', '?'))) or gap >= 9:
                            full_parts.append("\n" + line_text)
                        else:
                            full_parts.append(" " + line_text)

            full_text = "".join(full_parts).strip()
            full_text = re.sub(r'(\b[A-Za-z]{2,})-\s+([A-Za-z\']{2,})', r'\1\2', full_text)
            avg_c = float(np.mean([raw_boxes[idx].get("conf", 1.0) for idx in sub_comp])) if sub_comp else 1.0
            c_dict = {
                "text_box": [bx1, by1, bx2, by2],
                "text": full_text,
                "conf": avg_c,
                "avg_conf": avg_c,
                "indices": sub_comp,
                "block_id": f"b_{len(clusters) + 1}"
            }
            if stepped_split_x is not None:
                sub_cx = (bx1 + bx2) / 2.0
                if sub_cx < stepped_split_x:
                    c_dict["limit_right"] = stepped_split_x
                else:
                    c_dict["limit_left"] = stepped_split_x
                c_dict["split_x"] = stepped_split_x
            clusters.append(c_dict)

    return merge_adjacent_bubble_columns(clusters, gray, reading_direction=reading_direction, box_border_mask=box_border_mask)


def merge_adjacent_bubble_columns(clusters, gray, reading_direction: str = "rtl", box_border_mask=None):
    """
    Second pass: Merges multi-column text that belongs to the same speech bubble.
    Columns in the same bubble have gap_x <= 12, vertical overlap > 0, and no dark barrier.
    """
    if len(clusters) <= 1:
        return clusters

    if box_border_mask is None and gray is not None:
        box_border_mask = get_box_border_mask(gray)

    n = len(clusters)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            bA = clusters[i]["text_box"]
            bB = clusters[j]["text_box"]
            xA1, yA1, xA2, yA2 = bA
            xB1, yB1, xB2, yB2 = bB
            gap_x = max(0, max(xA1, xB1) - min(xA2, xB2))
            y_overlap = min(yA2, yB2) - max(yA1, yB1)
            min_h = min(yA2 - yA1, yB2 - yB1)

            if box_border_mask is not None and box_border_mask.size > 0:
                cxA = int((xA1 + xA2) / 2.0)
                cyA = int((yA1 + yA2) / 2.0)
                cxB = int((xB1 + xB2) / 2.0)
                cyB = int((yB1 + yB2) / 2.0)
                if line_crosses_border_mask(box_border_mask, (cxA, cyA), (cxB, cyB), boxA=bA, boxB=bB):
                    continue

            if min_h > 0 and y_overlap > 0 and gap_x <= max(18, int(min_h * 0.35)):
                if gray is not None and are_different_white_contours(gray, bA, bB):
                    continue
                tA = clusters[i].get("text", "").strip()
                tB = clusters[j].get("text", "").strip()
                if is_sfx_text(tA) or is_sfx_text(tB):
                    continue
                is_laughter = lambda s: bool(re.search(r'\b(ha|ah|he|wa)+\b', s, re.I))
                if is_laughter(tA) != is_laughter(tB):
                    continue
                if (tA.endswith(('!', '?')) and tB.endswith(('!', '?'))) and gap_x > 12:
                    continue
                left_box = bA if xA1 < xB1 else bB
                right_box = bB if xA1 < xB1 else bA
                if not is_dark_barrier_between_x(gray, left_box, right_box):
                    if gap_x > 0 and gray is not None:
                        gap_slice = gray[max(yA1, yB1):min(yA2, yB2), min(xA2, xB2):max(xA1, xB1)]
                        if gap_slice.size > 0 and (np.mean(gap_slice) < 180 or np.min(gap_slice) < 110):
                            continue
                    if gray is not None:
                        crop_area = gray[min(yA1, yB1):max(yA2, yB2), min(xA1, xB1):max(xA2, xB2)]
                        if crop_area.size > 0 and (np.mean(crop_area) > 160 or np.median(crop_area) > 160):
                            adj[i].append(j)
                            adj[j].append(i)
                    else:
                        adj[i].append(j)
                        adj[j].append(i)

    visited = set()
    merged = []
    for i in range(n):
        if i in visited:
            continue
        comp = [i]
        visited.add(i)
        queue = [i]
        while queue:
            cur = queue.pop(0)
            for neighbor in adj[cur]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    comp.append(neighbor)
                    queue.append(neighbor)

        if len(comp) == 1:
            merged.append(clusters[comp[0]])
        else:
            # Multi-column bubble reading order:
            comp_texts = [clusters[idx].get("text", "") for idx in comp]
            is_cjk = any(
                any('\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u30ff' or '\u3400' <= ch <= '\u4dbf' for ch in t)
                for t in comp_texts
            )
            is_strongly_vertical = all(
                (clusters[idx]["text_box"][3] - clusters[idx]["text_box"][1]) > 1.3 * max(1, clusters[idx]["text_box"][2] - clusters[idx]["text_box"][0])
                for idx in comp
            )
            is_vertical_columns = all(
                (clusters[idx]["text_box"][3] - clusters[idx]["text_box"][1]) > max(1, clusters[idx]["text_box"][2] - clusters[idx]["text_box"][0])
                for idx in comp
            )
            is_sentence = lambda s: bool(re.search(r'[.!?…~]+[\s"\'”’)]*$', s.strip()))
            each_is_sentence = len(comp) >= 2 and all(is_sentence(t) for t in comp_texts)

            gap_x_comp = 0
            is_vertically_shifted = False
            if len(comp) == 2:
                b0 = clusters[comp[0]]["text_box"]
                b1 = clusters[comp[1]]["text_box"]
                gap_x_comp = max(0, max(b0[0], b1[0]) - min(b0[2], b1[2]))
                w0 = b0[2] - b0[0]
                w1 = b1[2] - b1[0]
                cx0 = (b0[0] + b0[2]) / 2.0
                cx1 = (b1[0] + b1[2]) / 2.0
                is_vertically_shifted = (
                    (abs(b0[1] - b1[1]) >= 12 or abs(b0[3] - b1[3]) >= 12)
                    and (abs(cx0 - cx1) > 0.35 * (w0 + w1))
                )

            # Two independent vertical columns or 2 independent sentence columns in rectangular boxes
            # (e.g. stepped #6 narration box, where len(comp) == 2 with gap_x >= 6 and each_is_sentence or stepped)
            # must NOT be merged into a single horizontal block; keep each column in its own vertical bbox.
            is_two_independent_columns = (
                is_strongly_vertical
                or is_vertically_shifted
                or (
                    len(comp) == 2
                    and gap_x_comp >= 6
                    and (
                        each_is_sentence
                        or is_vertical_columns
                        or abs(clusters[comp[0]]["text_box"][1] - clusters[comp[1]]["text_box"][1]) >= 4
                    )
                )
            )
            if is_two_independent_columns:
                if len(comp) == 2 and is_vertically_shifted:
                    c0_idx, c1_idx = comp[0], comp[1]
                    if clusters[c0_idx]["text_box"][0] < clusters[c1_idx]["text_box"][0]:
                        l_idx, r_idx = c0_idx, c1_idx
                    else:
                        l_idx, r_idx = c1_idx, c0_idx
                    mid_x = (clusters[l_idx]["text_box"][2] + clusters[r_idx]["text_box"][0]) // 2
                    if "limit_right" not in clusters[l_idx]:
                        clusters[l_idx]["limit_right"] = mid_x
                    if "limit_left" not in clusters[r_idx]:
                        clusters[r_idx]["limit_left"] = mid_x

                if reading_direction == "rtl":
                    comp.sort(key=lambda idx: -clusters[idx]["text_box"][0])
                else:
                    comp.sort(key=lambda idx: clusters[idx]["text_box"][0])
                for idx in comp:
                    merged.append(clusters[idx])
                continue

            if is_cjk or is_strongly_vertical:
                # Right column is read first, then left column (CJK vertical text)
                comp.sort(key=lambda idx: -clusters[idx]["text_box"][0])
                full_text = " ... ".join(clusters[idx]["text"] for idx in comp)
            else:
                # Reading order for side-by-side vs stacked columns:
                has_side_by_side = False
                for ci in range(len(comp)):
                    for cj in range(ci + 1, len(comp)):
                        b1 = clusters[comp[ci]]["text_box"]
                        b2 = clusters[comp[cj]]["text_box"]
                        y_ov = min(b1[3], b2[3]) - max(b1[1], b2[1])
                        min_h = min(b1[3] - b1[1], b2[3] - b2[1])
                        if min_h > 0 and (y_ov / min_h) >= 0.30:
                            has_side_by_side = True
                            break
                    if has_side_by_side:
                        break

                if has_side_by_side:
                    if reading_direction == "rtl":
                        # Manga RTL reading order: right column first, left column second
                        comp.sort(key=lambda idx: -clusters[idx]["text_box"][0])
                    else:
                        comp.sort(key=lambda idx: clusters[idx]["text_box"][0])
                else:
                    # Vertically stacked: top-to-bottom
                    comp.sort(key=lambda idx: (clusters[idx]["text_box"][1], clusters[idx]["text_box"][0]))
                full_text = " ".join(clusters[idx]["text"] for idx in comp)

            bx1 = min(clusters[idx]["text_box"][0] for idx in comp)
            by1 = min(clusters[idx]["text_box"][1] for idx in comp)
            bx2 = max(clusters[idx]["text_box"][2] for idx in comp)
            by2 = max(clusters[idx]["text_box"][3] for idx in comp)
            all_indices = []
            for idx in comp:
                all_indices.extend(clusters[idx].get("indices", []))
            avg_c = float(np.mean([clusters[idx].get("conf", 1.0) for idx in comp])) if comp else 1.0
            merged.append({
                "text_box": [bx1, by1, bx2, by2],
                "text": full_text,
                "conf": avg_c,
                "avg_conf": avg_c,
                "indices": all_indices,
                "block_id": clusters[comp[0]].get("block_id", f"b_{len(merged) + 1}")
            })

    return merged


def find_bubble_visual_center(gray_np, box):
    """
    Computes the exact mathematical centroid of the speech bubble interior
    using image moments of the largest light/white connected component.
    Guarantees text is 100% dead-centered in the physical speech bubble!
    """
    x1, y1, x2, y2 = box
    h, w = gray_np.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 + 8 or y2 <= y1 + 8:
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    patch = gray_np[y1:y2, x1:x2]
    geom_cx = (x1 + x2) / 2.0
    geom_cy = (y1 + y2) / 2.0
    if patch.size == 0 or float(np.median(patch)) < 120:
        return (geom_cx, geom_cy)

    p85 = float(np.percentile(patch, 85))
    if p85 > 210:
        white_mask = (patch > min(205, int(p85 * 0.90))).astype(np.uint8) * 255
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 80:
                cx = x1 + (M["m10"] / M["m00"])
                cy = y1 + (M["m01"] / M["m00"])
                cx = max(geom_cx - 4.0, min(geom_cx + 4.0, cx))
                cy = max(geom_cy - 4.0, min(geom_cy + 4.0, cy))
                return (cx, cy)

    return (geom_cx, geom_cy)


def expand_to_bubble_contour(gray, text_box, all_other_tboxes=None, max_pad_x=100, max_pad_y=100, limit_left=None, limit_right=None):
    """
    Snaps a text box to the speech bubble's true outer boundary.
    Prevents invasion into neighboring text boxes by checking midpoint boundaries.
    Detects thin oval bubble contours using ink density and dark pixel ratio.
    Supports bounding limits (limit_left / limit_right) so stepped lobes do not cross x_split.
    """
    h, w = gray.shape
    orig_x1, orig_y1, orig_x2, orig_y2 = text_box

    # Only expand into speech bubbles if the region is genuinely a uniform light bubble (>210)
    # If the text is on artwork, screentone, floor, or character body (e.g. SFX),
    # DO NOT expand! Keep it tightly bounded to the text stroke!
    patch = gray[max(0, orig_y1):min(h, orig_y2), max(0, orig_x1):min(w, orig_x2)]
    ring_pixels = []
    top_strip = gray[max(0, orig_y1 - 6):orig_y1, max(0, orig_x1 - 6):min(w, orig_x2 + 6)]
    bot_strip = gray[orig_y2:min(h, orig_y2 + 6), max(0, orig_x1 - 6):min(w, orig_x2 + 6)]
    left_strip = gray[orig_y1:orig_y2, max(0, orig_x1 - 6):orig_x1]
    right_strip = gray[orig_y1:orig_y2, orig_x2:min(w, orig_x2 + 6)]
    for strip in (top_strip, bot_strip, left_strip, right_strip):
        if strip.size > 0:
            ring_pixels.append(strip.ravel())
    ring_arr = np.concatenate(ring_pixels) if ring_pixels else np.array([])
    is_in_light_bubble = (
        (patch.size > 0 and np.percentile(patch, 85) >= 195 and np.percentile(patch, 50) >= 160) or
        (ring_arr.size > 0 and np.percentile(ring_arr, 75) >= 200)
    )
    if not is_in_light_bubble:
        pad_x = min(5, max(2, int((orig_x2 - orig_x1) * 0.10)))
        pad_y = min(4, max(2, int((orig_y2 - orig_y1) * 0.10)))
        res_x1 = max(0, orig_x1 - pad_x)
        res_x2 = min(w, orig_x2 + pad_x)
        if limit_left is not None:
            res_x1 = max(res_x1, limit_left)
        if limit_right is not None:
            res_x2 = min(res_x2, limit_right)
        return [res_x1, max(0, orig_y1 - pad_y), res_x2, min(h, orig_y2 + pad_y)]

    # Determine safe stopping bounds with respect to all other text boxes and page edges BEFORE min_pad
    bound_left = min(orig_x1, 4)
    bound_right = max(orig_x2, w - 4)
    bound_top = min(orig_y1, 4)
    bound_bot = max(orig_y2, h - 4)

    if limit_left is not None:
        bound_left = max(bound_left, limit_left)
    if limit_right is not None:
        bound_right = min(bound_right, limit_right)

    if all_other_tboxes:
        for ob in all_other_tboxes:
            ox1, oy1, ox2, oy2 = ob
            # Check vertical overlap to constrain horizontal expansion
            if max(orig_y1, oy1) < min(orig_y2, oy2):
                if ox2 <= orig_x1:
                    bound = (ox2 + orig_x1) // 2 + 1
                    if limit_left is not None:
                        bound = max(bound, limit_left)
                    bound_left = max(bound_left, bound)
                elif ox1 >= orig_x2:
                    bound = (orig_x2 + ox1) // 2 - 1
                    if limit_right is not None:
                        bound = min(bound, limit_right)
                    bound_right = min(bound_right, bound)

            # Check horizontal overlap to constrain vertical expansion
            if max(orig_x1, ox1) < min(orig_x2, ox2):
                if oy2 <= orig_y1:
                    bound_top = max(bound_top, (oy2 + orig_y1) // 2 + 1)
                elif oy1 >= orig_y2:
                    bound_bot = min(bound_bot, (orig_y2 + oy1) // 2 - 1)

    # Scan from the text bounds so a close outline cannot fall inside skipped padding.
    x1, y1, x2, y2 = text_box

    safe_max_pad_x = min(max_pad_x, max(30, int((x2 - x1) * 0.60)))
    safe_max_pad_y = min(max_pad_y, max(35, int((y2 - y1) * 0.80)))
    bound_left = max(bound_left, x1 - safe_max_pad_x)
    bound_right = min(bound_right, x2 + safe_max_pad_x)
    bound_top = max(bound_top, y1 - safe_max_pad_y)
    bound_bot = min(bound_bot, y2 + safe_max_pad_y)

    if limit_left is not None:
        bound_left = max(bound_left, limit_left)
    if limit_right is not None:
        bound_right = min(bound_right, limit_right)

    def is_line_border(arr):
        if arr.size == 0:
            return True
        span = len(arr)
        d_ratio = np.mean(arr < 130)
        mean_val = np.mean(arr)
        # Require a true box border line (continuous dark line, d_ratio >= 0.25)
        # rather than isolated letter stroke pixels (must not stop early on min_v < 45)
        if d_ratio >= 0.25:
            return True
        min_consec = max(10, int(0.25 * span))
        if _has_continuous_dark_segment(arr, dark_thresh=110, min_len=min_consec):
            return True
        if mean_val < 140:
            return True
        return False

    # Top
    cy1 = y1
    for ty in range(y1 - 1, bound_top - 1, -1):
        row = gray[ty, x1:x2]
        if is_line_border(row):
            cy1 = ty
            break
        cy1 = ty

    # Bottom
    cy2 = y2
    for ty in range(y2, min(h, bound_bot)):
        row = gray[ty, x1:x2]
        if is_line_border(row):
            cy2 = ty
            break
        cy2 = ty + 1

    vh = cy2 - cy1
    mid_cy1 = cy1 + int(0.20 * vh)
    mid_cy2 = cy2 - int(0.20 * vh)
    if mid_cy2 <= mid_cy1:
        mid_cy1, mid_cy2 = cy1, cy2

    # Left (use central 60% vertical band mid_cy1:mid_cy2)
    cx1 = x1
    for tx in range(x1 - 1, bound_left - 1, -1):
        col_text = gray[y1:y2, tx]
        col_mid = gray[mid_cy1:mid_cy2, tx]
        if is_line_border(col_text) or is_line_border(col_mid):
            cx1 = tx
            break
        cx1 = tx
    if limit_left is not None:
        cx1 = max(cx1, limit_left)

    # Right (use central 60% vertical band mid_cy1:mid_cy2)
    cx2 = x2
    for tx in range(x2, min(w, bound_right)):
        col_text = gray[y1:y2, tx]
        col_mid = gray[mid_cy1:mid_cy2, tx]
        if is_line_border(col_text) or is_line_border(col_mid):
            cx2 = tx
            break
        cx2 = tx + 1
    if limit_right is not None:
        cx2 = min(cx2, limit_right)

    return [cx1, cy1, cx2, cy2]
