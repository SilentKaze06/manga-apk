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


def _crop_reading_key(r, reading_direction: str = "rtl", col_map: dict = None, line_idx_map: dict = None, coord_scale: float = 2.0):
    """
    Sort key function for OCR items inside a crop, supporting RTL manga reading direction
    and multi-column/line grouping.
    """
    col = col_map.get(id(r), 0) if col_map else 0
    line_idx = line_idx_map.get(id(r), None) if line_idx_map else None
    div = coord_scale if coord_scale and coord_scale > 0 else 1.0

    if isinstance(r, dict) and "box" in r:
        box = r["box"]
        y_c = (box[1] + box[3]) / 2.0
        x1 = box[0]
        x_c = (box[0] + box[2]) / 2.0
    elif isinstance(r, (list, tuple)) and len(r) > 0 and isinstance(r[0], (list, tuple)):
        ys = [pt[1] / div if isinstance(pt, (list, tuple)) else pt for pt in r[0]]
        xs = [pt[0] / div if isinstance(pt, (list, tuple)) else pt for pt in r[0]]
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


try:
    from geometry import (
        _white_cc_cache,
        _pipeline_cache_lock,
        _get_gray_cache_key,
        get_white_cc_labels,
        _get_box_white_mode_label,
        are_different_white_contours,
        get_box_border_mask,
        _get_box_border_mask,
        line_crosses_border_mask,
        _line_crosses_border_mask,
        _has_continuous_dark_segment,
        is_dark_barrier_between_x,
        _is_dark_barrier_between_x,
        is_dark_barrier_between_y,
        _is_dark_barrier_between_y,
        has_pinched_bubble_waist,
    )
except (ImportError, ValueError):
    from .geometry import (
        _white_cc_cache,
        _pipeline_cache_lock,
        _get_gray_cache_key,
        get_white_cc_labels,
        _get_box_white_mode_label,
        are_different_white_contours,
        get_box_border_mask,
        _get_box_border_mask,
        line_crosses_border_mask,
        _line_crosses_border_mask,
        _has_continuous_dark_segment,
        is_dark_barrier_between_x,
        _is_dark_barrier_between_x,
        is_dark_barrier_between_y,
        _is_dark_barrier_between_y,
        has_pinched_bubble_waist,
    )


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


def detect_stepped_box_split_y(gray, cluster_box):
    """
    Symmetric vertical stepped-box detector (upper box + lower offset box).
    Extracts the main white connected component in gray > 200 within cluster_box.
    If fill_ratio < 0.80 and at least one corner (15%x15% zone) is empty,
    checks for a sharp horizontal step in x_left(y) or x_right(y) where
    abs(diff) >= max(15, int(0.15 * bw)).
    Returns absolute image coordinate y_split = y1 + step_y, or None.
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

    cw = max(1, int(0.15 * bw))
    ch = max(1, int(0.15 * bh))
    tl = main_white_comp[0:ch, 0:cw]
    tr = main_white_comp[0:ch, bw - cw:bw]
    bl = main_white_comp[bh - ch:bh, 0:cw]
    br = main_white_comp[bh - ch:bh, bw - cw:bw]
    corner_means = [float(np.mean(tl)), float(np.mean(tr)), float(np.mean(bl)), float(np.mean(br))]
    if not any(m < 0.30 for m in corner_means):
        return None

    x_left = np.zeros(bh, dtype=np.int32)
    x_right = np.zeros(bh, dtype=np.int32)
    valid = np.zeros(bh, dtype=bool)
    for y in range(bh):
        cols = np.where(main_white_comp[y, :])[0]
        if len(cols) > 0:
            x_left[y] = cols[0]
            x_right[y] = cols[-1]
            valid[y] = True

    if not np.any(valid):
        return None

    valid_indices = np.where(valid)[0]
    for y in range(bh):
        if not valid[y]:
            nearest = valid_indices[np.argmin(np.abs(valid_indices - y))]
            x_left[y] = x_left[nearest]
            x_right[y] = x_right[nearest]

    min_jump = max(15, int(0.15 * bw))
    y_start = max(1, int(0.10 * bh))
    y_end = min(bh - 1, int(0.90 * bh))

    step_left = None
    max_d_left = 0
    for y in range(y_start, y_end):
        for k in (1, 2, 3):
            if y + k < bh:
                d = abs(int(x_left[y + k]) - int(x_left[y]))
                if d >= min_jump and d > max_d_left:
                    max_d_left = d
                    step_left = y + (k + 1) // 2

    step_right = None
    max_d_right = 0
    for y in range(y_start, y_end):
        for k in (1, 2, 3):
            if y + k < bh:
                d = abs(int(x_right[y + k]) - int(x_right[y]))
                if d >= min_jump and d > max_d_right:
                    max_d_right = d
                    step_right = y + (k + 1) // 2

    if step_left is not None and step_right is not None:
        step_y = int(round((step_left + step_right) / 2.0))
    elif step_left is not None:
        step_y = step_left
    elif step_right is not None:
        step_y = step_right
    else:
        return None

    return y1 + step_y


def split_joined_bubble_lobes(gray: np.ndarray, bubble_box, mask_u8=None):
    """
    Universal geometry-based bubble-lobe splitter using Distance Transform + Watershed.
    Handles horizontal, vertical, and diagonal figure-8 / joined double-bubbles while
    preserving single oval/elongated bubbles via a mandatory saddle-depth guard.
    Returns a list of (lobe_mask_bool_in_patch, [lx1, ly1, lx2, ly2]) or [] if < 2 lobes.
    """
    if gray is None or gray.size == 0 or bubble_box is None:
        return []
    if not isinstance(bubble_box, (list, tuple)) or len(bubble_box) < 4:
        return []

    gh, gw = gray.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bubble_box[:4]]
    x1 = max(0, min(gw, x1))
    x2 = max(0, min(gw, x2))
    y1 = max(0, min(gh, y1))
    y2 = max(0, min(gh, y2))
    bw = x2 - x1
    bh = y2 - y1
    if min(bw, bh) < 36:
        return []

    patch = gray[y1:y2, x1:x2]
    if float(np.median(patch)) < 120.0:
        return []

    # Build clean binary mask of the bubble white interior:
    # 1) Pick main interior component BEFORE morphological closing so thin (2-5px) bubble outlines
    #    are never bridged into the outer white page background.
    white = (patch > 185).astype(np.uint8)
    if mask_u8 is not None and mask_u8.size > 0 and mask_u8.shape[0] >= y2 and mask_u8.shape[1] >= x2:
        mask_patch = (mask_u8[y1:y2, x1:x2] > 25).astype(np.uint8)
        white = np.maximum(white, mask_patch)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    if num_labels <= 1:
        return []

    # Pick the main component covering the central interior (fallback to largest component)
    cy1, cy2 = int(0.20 * bh), max(int(0.20 * bh) + 1, int(0.80 * bh))
    cx1, cx2 = int(0.20 * bw), max(int(0.20 * bw) + 1, int(0.80 * bw))
    center_lbls = labels[cy1:cy2, cx1:cx2].ravel()
    center_pos = center_lbls[center_lbls > 0]
    if center_pos.size > 0:
        counts = np.bincount(center_pos)
        main_lbl = int(np.argmax(counts))
    else:
        main_lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    main_comp = (labels == main_lbl).astype(np.uint8)
    # 2) Close interior text gaps & fill holes ONLY inside main_comp
    k_sz = max(5, int(0.04 * min(bw, bh)))
    if k_sz % 2 == 0:
        k_sz += 1
    ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
    main_comp = cv2.morphologyEx(main_comp, cv2.MORPH_CLOSE, ellipse_kernel)
    contours, _ = cv2.findContours(main_comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(main_comp, contours, -1, 1, thickness=-1)

    if np.all(main_comp > 0):
        return []

    dist = cv2.distanceTransform(main_comp, cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, (5, 5), 0)
    max_dist = float(dist.max())
    if max_dist < 10.0:
        return []

    peak_thresh = max(10.0, 0.28 * max_dist)
    ys_p, xs_p = np.where(dist >= peak_thresh)
    if ys_p.size < 2:
        return []

    # Sort candidate pixels descending by distance value and apply NMS with radius r
    vals = dist[ys_p, xs_p]
    order = np.argsort(vals)[::-1]
    nms_r = max(14, int(0.25 * min(bw, bh)))
    nms_r2 = nms_r * nms_r

    cand_peaks = []
    for idx in order:
        py, px = int(ys_p[idx]), int(xs_p[idx])
        too_close = False
        for ky, kx in cand_peaks:
            if (py - ky) * (py - ky) + (px - kx) * (px - kx) < nms_r2:
                too_close = True
                break
        if not too_close:
            cand_peaks.append((py, px))
            if len(cand_peaks) >= 8:
                break

    if len(cand_peaks) < 2:
        return []

    # Mandatory Anti-Over-Splitting Saddle Guard: keep peak only if pinched <= 0.68 * min_peak
    # or separated by a dark border crossing (gray < 110 outside main_comp)
    surviving_peaks = [cand_peaks[0]]
    for py, px in cand_peaks[1:]:
        distinct = True
        for ky, kx in surviving_peaks:
            seg_len = int(max(16, np.hypot(py - ky, px - kx)))
            ys_seg = np.clip(np.round(np.linspace(ky, py, seg_len)).astype(np.int32), 0, bh - 1)
            xs_seg = np.clip(np.round(np.linspace(kx, px, seg_len)).astype(np.int32), 0, bw - 1)
            saddle = float(np.min(dist[ys_seg, xs_seg]))
            min_peak = float(min(dist[py, px], dist[ky, kx]))
            has_border_crossing = bool(np.any((patch[ys_seg, xs_seg] < 110) & (main_comp[ys_seg, xs_seg] == 0)))
            if saddle > 0.68 * min_peak and not has_border_crossing:
                distinct = False
                break
        if distinct:
            surviving_peaks.append((py, px))
            if len(surviving_peaks) >= 4:
                break

    if len(surviving_peaks) < 2:
        return []

    # Partition main_comp pixels among surviving_peaks via power distance (radical axis of lobes)
    ys_m, xs_m = np.where(main_comp > 0)
    if ys_m.size == 0:
        return []
    peaks_arr = np.asarray(surviving_peaks, dtype=np.float32)  # (K, 2)
    peak_radii2 = np.asarray([float(dist[py, px]) ** 2 for py, px in surviving_peaks], dtype=np.float32)
    pts_arr = np.stack([ys_m, xs_m], axis=1).astype(np.float32)  # (M, 2)
    d2 = np.sum((pts_arr[:, None, :] - peaks_arr[None, :, :]) ** 2, axis=2) - peak_radii2[None, :]
    nearest_k = np.argmin(d2, axis=1)

    out_lobes = []
    for p_idx in range(len(surviving_peaks)):
        sel = (nearest_k == p_idx)
        if not np.any(sel):
            continue
        lobe_mask = np.zeros((bh, bw), dtype=bool)
        lobe_mask[ys_m[sel], xs_m[sel]] = True
        ys_l, xs_l = ys_m[sel], xs_m[sel]
        lx1 = x1 + int(xs_l.min())
        ly1 = y1 + int(ys_l.min())
        lx2 = x1 + int(xs_l.max()) + 1
        ly2 = y1 + int(ys_l.max()) + 1
        if lx2 > lx1 + 4 and ly2 > ly1 + 4:
            out_lobes.append((lobe_mask, [lx1, ly1, lx2, ly2]))

    if len(out_lobes) < 2:
        return []
    return out_lobes


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
        (patch.size > 0 and np.percentile(patch, 85) >= 195 and np.percentile(patch, 50) >= 150) or
        (ring_arr.size > 0 and np.percentile(ring_arr, 75) >= 190)
    )
    is_in_dark_bubble = (
        not is_in_light_bubble
        and (
            (patch.size > 0 and np.percentile(patch, 50) < 95 and np.percentile(patch, 95) > 165)
            or (ring_arr.size > 0 and np.percentile(ring_arr, 50) < 95 and float(np.mean(ring_arr < 115)) >= 0.50)
        )
    )
    if not is_in_light_bubble and not is_in_dark_bubble:
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

    safe_max_pad_x = min(max_pad_x, max(45, int((x2 - x1) * 1.1)))
    safe_max_pad_y = min(max_pad_y, max(45, int((y2 - y1) * 1.1)))
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
        if is_in_dark_bubble:
            return (np.mean(arr > 165) >= 0.40) or (np.mean(arr) > 150)
        span = len(arr)
        d_ratio = np.mean(arr < 130)
        mean_val = np.mean(arr)
        ratio_thresh = 0.35 if span <= 30 else 0.25
        if d_ratio >= ratio_thresh:
            return True
        min_consec = max(10, int(0.35 * span)) if span <= 30 else max(10, int(0.25 * span))
        if _has_continuous_dark_segment(arr, dark_thresh=110, min_len=min_consec):
            return True
        if mean_val < 135:
            return True
        return False

    # Top (step 1px back inside detected border so outer line isn't pulled in)
    cy1 = y1
    for ty in range(y1 - 1, bound_top - 1, -1):
        row = gray[ty, x1:x2]
        if is_line_border(row):
            cy1 = min(y1, ty + 1)
            break
        cy1 = ty

    # Bottom
    cy2 = y2
    for ty in range(y2, min(h, bound_bot)):
        row = gray[ty, x1:x2]
        if is_line_border(row):
            cy2 = max(y2, ty)
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
            cx1 = min(x1, tx + 1)
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
            cx2 = max(x2, tx)
            break
        cx2 = tx + 1
    if limit_right is not None:
        cx2 = min(cx2, limit_right)

    return [cx1, cy1, cx2, cy2]
