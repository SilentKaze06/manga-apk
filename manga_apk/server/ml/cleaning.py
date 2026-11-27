"""
Text masking, boundary protection, SimpleLama inpainting, and bubble crop cleaning.
"""
import logging
import os
import threading
from typing import Optional
import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Hardware optimization for AMD Ryzen 5 (4 cores, 8 threads)
if not torch.cuda.is_available():
    try:
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        cv2.setNumThreads(num_threads)
    except Exception as exc:
        logger.debug("CPU thread configuration skipped: %s", exc)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _estimate_bg_mode_and_std(gray_region: np.ndarray) -> tuple:
    """Estimate dominant background luminance and background-band std via histogram mode."""
    if gray_region is None or gray_region.size == 0:
        return 255.0, 0.0
    flat = gray_region.ravel()
    hist, _ = np.histogram(flat, bins=16, range=(0, 256))
    peak_bin = int(np.argmax(hist))
    peak_center = (peak_bin + 0.5) * 16.0
    band = flat[np.abs(flat.astype(np.float32) - peak_center) <= 28.0]
    if band.size < max(8, int(flat.size * 0.15)):
        return float(np.median(flat)), float(np.std(flat))
    return float(np.median(band)), float(np.std(band))


def generate_ink_stroke_mask(gray_np, box, pad=2):
    """
    Generates a precise binary mask covering dialogue text ink strokes and their anti-aliased halos,
    preventing damage to character faces, hair, panel borders, and screentones.
    """
    h, w = gray_np.shape
    x1, y1, x2, y2 = box
    px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
    px2, py2 = min(w, x2 + pad), min(h, y2 + pad)
    if px2 <= px1 or py2 <= py1:
        return np.zeros((h, w), dtype=np.uint8)

    patch = gray_np[py1:py2, px1:px2]
    p15 = float(np.percentile(patch, 15))
    p85 = float(np.percentile(patch, 85))
    p95 = float(np.percentile(patch, 95))
    med = float(np.median(patch))
    bg_mode, _ = _estimate_bg_mode_and_std(patch)
    hp, wp = patch.shape

    is_dark_mode = (med < 115) or (bg_mode < 115 and p85 < 165)
    # Mid-tone box (115..184) with bright/white text
    is_midtone_white_text = (
        not is_dark_mode
        and bg_mode < 180
        and med < 185
        and p95 >= max(195.0, bg_mode + 45.0)
        and (p95 - bg_mode) > (bg_mode - p15) + 20.0
        and float(np.mean(patch > bg_mode + 35.0)) < 0.42
    )

    if is_dark_mode:
        thresh_val = max(145, int(med + 45))
        stroke_binary = (patch > thresh_val).astype(np.uint8) * 255
    elif is_midtone_white_text:
        thresh_val = min(235, int(bg_mode + max(35.0, (p95 - bg_mode) * 0.45)))
        stroke_binary = (patch > thresh_val).astype(np.uint8) * 255
    elif (p85 - p15) > 38 and min(hp, wp) >= 12:
        # Smooth gradient or textured background: compare against local background estimate
        k_sz = max(11, min(31, (min(hp, wp) // 2) | 1))
        k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
        local_bg = cv2.morphologyEx(patch, cv2.MORPH_CLOSE, k_bg)
        stroke_binary = ((local_bg.astype(np.int16) - patch.astype(np.int16)) > 26).astype(np.uint8) * 255
    elif p85 >= 160:
        # Light/colored dialogue bubbles with dark text (background-relative offset)
        thresh_val = min(232, max(85, int(bg_mode - 30)))
        stroke_binary = (patch < thresh_val).astype(np.uint8) * 255
    else:
        thresh_val = min(180, max(65, int(bg_mode - 25)))
        stroke_binary = (patch < thresh_val).astype(np.uint8) * 255

    # Filter out long linear border segments from stroke mask limited to border-touching components
    kw = max(35, int(wp * 0.75))
    kh = max(35, int(hp * 0.75))
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
    long_lines = cv2.bitwise_or(
        cv2.morphologyEx(stroke_binary, cv2.MORPH_OPEN, k_h),
        cv2.morphologyEx(stroke_binary, cv2.MORPH_OPEN, k_v)
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(stroke_binary)
    lines = np.zeros_like(stroke_binary)
    for l in range(1, num_labels):
        st = stats[l]
        lx, ly = int(st[cv2.CC_STAT_LEFT]), int(st[cv2.CC_STAT_TOP])
        lw, lh = int(st[cv2.CC_STAT_WIDTH]), int(st[cv2.CC_STAT_HEIGHT])
        la = int(st[cv2.CC_STAT_AREA])
        touches_edge = (lx <= 1 or ly <= 1 or lx + lw >= wp - 1 or ly + lh >= hp - 1)
        if not touches_edge:
            continue
        sub_lbl = (labels[ly:ly + lh, lx:lx + lw] == l)
        is_border_wall = (
            np.any(long_lines[ly:ly + lh, lx:lx + lw][sub_lbl] > 0)
            or la > int(0.30 * wp * hp)
            or (lw >= int(0.60 * wp) and (ly <= 2 or ly + lh >= hp - 2))
            or (lh >= int(0.60 * hp) and (lx <= 2 or lx + lw >= wp - 2))
        )
        if is_border_wall:
            lines[ly:ly + lh, lx:lx + lw][sub_lbl] = 255

    stroke_clean = cv2.bitwise_and(stroke_binary, cv2.bitwise_not(lines))

    # For dark or midtone-white-text boxes, apply 2px dilation to swallow outline halos; else 1 iteration
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    stroke_dilated = cv2.dilate(stroke_clean, k, iterations=2 if (is_dark_mode or is_midtone_white_text) else 1)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[py1:py2, px1:px2] = stroke_dilated
    return mask


def clean_bubble_crop(
    patch_orig,
    patch_gray=None,
    patch_mask=None,
    neural_patch=None,
    inpainter_model=None,
    inpaint_transform=None,
    device=DEVICE,
):
    """
    Master Border-Protected Bubble Cleaning Algorithm:
    - White / cream bubble with border: preserves outer contour border; cleans interior text to local background color.
    - Dark narration box: preserves border contour; fills interior text with dark background color (never pure white).
    - Screentone / artwork / borderless: uses neural cleaner or U-Net inpainter if available, else Telea.
    """
    if patch_orig is None or patch_orig.size == 0:
        return patch_orig

    if patch_orig.ndim == 2:
        patch_orig = cv2.cvtColor(patch_orig, cv2.COLOR_GRAY2RGB)
    elif patch_orig.ndim == 3 and patch_orig.shape[2] == 1:
        patch_orig = cv2.cvtColor(patch_orig[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif patch_orig.ndim == 3 and patch_orig.shape[2] == 4:
        patch_orig = cv2.cvtColor(patch_orig, cv2.COLOR_RGBA2RGB)

    hp, wp = patch_orig.shape[:2]
    if hp <= 6 or wp <= 6:
        return patch_orig.copy()

    if patch_gray is None:
        patch_gray = cv2.cvtColor(patch_orig, cv2.COLOR_RGB2GRAY)
    elif patch_gray.ndim == 3:
        patch_gray = cv2.cvtColor(patch_gray, cv2.COLOR_RGB2GRAY) if patch_gray.shape[2] >= 3 else patch_gray[:, :, 0]

    k_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_rect3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Padded interior region (ignoring outer margin pixels that may contain borders or opposite-tone gutters)
    my = max(2, int(hp * 0.12))
    mx = max(2, int(wp * 0.12))
    if hp - 2 * my >= 4 and wp - 2 * mx >= 4:
        interior_gray = patch_gray[my:hp - my, mx:wp - mx]
        interior_orig = patch_orig[my:hp - my, mx:wp - mx]
    else:
        interior_gray = patch_gray
        interior_orig = patch_orig

    interior_median = float(np.median(interior_gray))
    interior_p15 = float(np.percentile(interior_gray, 15))
    interior_p25 = float(np.percentile(interior_gray, 25))
    interior_p85 = float(np.percentile(interior_gray, 85))
    interior_p95 = float(np.percentile(interior_gray, 95))

    full_median = float(np.median(patch_gray))
    full_p15 = float(np.percentile(patch_gray, 15))
    full_p85 = float(np.percentile(patch_gray, 85))

    bg_mode_luma, bg_mode_std = _estimate_bg_mode_and_std(interior_gray)
    dark_interior_pixels = interior_gray[interior_gray < 130]
    has_dark_interior_bg = (dark_interior_pixels.size >= 10 and float(np.median(dark_interior_pixels)) <= 90)

    # Determine is_dark_bubble from the INTERIOR region of the crop (ignoring outer margins)
    is_dark_bubble = (
        (interior_median < 95)
        or (bg_mode_luma < 95 and interior_p85 < 185)
        or (interior_p15 < 55 and interior_median < 120 and interior_p85 < 175)
        or (interior_median < 110 and interior_p25 < 65 and has_dark_interior_bg)
        or (full_median < 65)
        or (full_p15 < 40 and full_p85 < 110)
    )

    # Detect white/bright text on mid-tone colored/gray backgrounds so p85 never triggers white fill
    has_bright_text_on_midtone = (
        not is_dark_bubble
        and bg_mode_luma < 185
        and interior_p95 >= max(195.0, bg_mode_luma + 42.0)
        and (interior_p95 - bg_mode_luma) > (bg_mode_luma - interior_p15) + 18.0
        and float(np.mean(interior_gray > bg_mode_luma + 35.0)) < 0.42
    )

    # Uniform light / cream / pastel / colored bubbles (dark text on uniform >=165 bg, NOT gradient/screentone)
    is_uniform_light_bubble = False
    if not is_dark_bubble and not has_bright_text_on_midtone and bg_mode_luma >= 165.0:
        non_text_est = interior_gray[interior_gray >= max(100.0, bg_mode_luma - 32.0)]
        non_text_std = float(np.std(non_text_est)) if non_text_est.size >= 12 else bg_mode_std
        if non_text_std < 15.0:
            is_uniform_light_bubble = True

    # ──────────────────────────────────────────────────────────────────────────
    # 1. UNIFORM WHITE, CREAM, OR PASTEL LIGHT BUBBLES
    # ──────────────────────────────────────────────────────────────────────────
    if is_uniform_light_bubble:
        ink_thresh = min(230, max(90, int(round(bg_mode_luma - 30.0))))
        non_ink_interior = (interior_gray >= ink_thresh)
        if np.any(non_ink_interior):
            local_bg_color = np.round(np.median(interior_orig[non_ink_interior], axis=0)).astype(np.uint8)
        else:
            non_ink_full = (patch_gray >= ink_thresh)
            if np.any(non_ink_full):
                local_bg_color = np.round(np.median(patch_orig[non_ink_full], axis=0)).astype(np.uint8)
            else:
                local_bg_color = np.round(np.median(interior_orig.reshape(-1, 3), axis=0)).astype(np.uint8)

        # Dark ink mask relative to actual bubble background
        dark_mask = (patch_gray < ink_thresh).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)

        # Build protected_mask for true perimeter/border structures:
        # a) Straight box frames along the perimeter (top, bottom, left, right)
        frame_pad_y = max(4, int(hp * 0.15))
        frame_pad_x = max(4, int(wp * 0.15))
        kw = max(24, int(wp * 0.60))
        kh = max(24, int(hp * 0.60))
        horiz_frame = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1)))
        vert_frame = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh)))

        straight_lines = np.zeros_like(dark_mask)
        if np.any(horiz_frame):
            straight_lines[0:frame_pad_y, :] |= horiz_frame[0:frame_pad_y, :]
            straight_lines[hp - frame_pad_y:hp, :] |= horiz_frame[hp - frame_pad_y:hp, :]
        if np.any(vert_frame):
            straight_lines[:, 0:frame_pad_x] |= vert_frame[:, 0:frame_pad_x]
            straight_lines[:, wp - frame_pad_x:wp] |= vert_frame[:, wp - frame_pad_x:wp]

        # b) Identify components that are true borders, curved bubble contours, or outer artwork
        keep_labels = set()

        border_margin_x = max(2, int(wp * 0.08))
        border_margin_y = max(2, int(hp * 0.08))

        has_straight_lines = np.any(straight_lines)

        for lbl in range(1, num_labels):
            lx = stats[lbl, cv2.CC_STAT_LEFT]
            ly = stats[lbl, cv2.CC_STAT_TOP]
            lw = stats[lbl, cv2.CC_STAT_WIDTH]
            lh = stats[lbl, cv2.CC_STAT_HEIGHT]
            la = stats[lbl, cv2.CC_STAT_AREA]

            touches_top = (ly <= 1)
            touches_bot = (ly + lh >= hp - 1)
            touches_left = (lx <= 1)
            touches_right = (lx + lw >= wp - 1)

            if (touches_top or touches_bot or touches_left or touches_right):
                if (lw >= 0.70 * wp and (touches_top or touches_bot) and la >= 0.15 * (wp * hp)):
                    keep_labels.add(lbl)
                    continue
                if (lh >= 0.70 * hp and (touches_left or touches_right) and la >= 0.15 * (wp * hp)):
                    keep_labels.add(lbl)
                    continue

            if has_straight_lines and np.sum((labels[ly:ly + lh, lx:lx + lw] == lbl) & (straight_lines[ly:ly + lh, lx:lx + lw] > 0)) >= 0.25 * la:
                continue

            spans_width = (lx <= border_margin_x and (lx + lw) >= wp - border_margin_x and lw >= 0.70 * wp)
            spans_height = (ly <= border_margin_y and (ly + lh) >= hp - border_margin_y and lh >= 0.50 * hp)
            solidity = la / float(lw * lh) if (lw * lh > 0) else 1.0

            if spans_width and spans_height and solidity <= 0.35:
                cy1, cy2 = int(hp * 0.25), int(hp * 0.75)
                cx1, cx2 = int(wp * 0.25), int(wp * 0.75)
                center_pixels = np.sum(labels[cy1:cy2, cx1:cx2] == lbl)
                center_ratio = center_pixels / float(la) if la > 0 else 0.0
                if center_ratio <= 0.08:
                    keep_labels.add(lbl)
                    continue

        if keep_labels:
            protected_mask = np.isin(labels, list(keep_labels)).astype(np.uint8) * 255
        else:
            protected_mask = np.zeros_like(dark_mask)

        if has_straight_lines:
            protected_mask = cv2.bitwise_or(protected_mask, (straight_lines > 0).astype(np.uint8) * 255)

        # c) Identify letter ink to erase
        letter_mask = ((dark_mask > 0) & (protected_mask == 0)).astype(np.uint8)

        if patch_mask is not None and np.any(patch_mask):
            ext_mask = ((patch_mask > 0) & (protected_mask == 0)).astype(np.uint8)
            letter_mask = cv2.bitwise_or(letter_mask, ext_mask)

        dilated_letters = cv2.dilate(letter_mask, k_rect3, iterations=2)
        dilated_erase = cv2.bitwise_and(dilated_letters, cv2.bitwise_not(protected_mask))

        # Softly clean anti-aliased halo pixels touching erased text without erasing colored bg
        halo_hi = min(245, max(ink_thresh + 5, int(round(bg_mode_luma - 8.0))))
        anti_aliased_mask = ((patch_gray >= ink_thresh) & (patch_gray <= halo_hi)).astype(np.uint8)
        halo_zone = cv2.dilate(dilated_erase, k_rect3, iterations=1)
        halo_candidates = (halo_zone > 0) & (anti_aliased_mask > 0) & (protected_mask == 0)

        erase_mask = cv2.bitwise_and(
            cv2.bitwise_or(dilated_erase, halo_candidates.astype(np.uint8)),
            cv2.bitwise_not(protected_mask)
        )

        patch_clean = patch_orig.copy()
        if np.any(erase_mask):
            remaining_mask = (erase_mask > 0)
            if neural_patch is not None and neural_patch.shape == patch_orig.shape:
                diff_n = np.abs(neural_patch.astype(np.float32) - patch_orig.astype(np.float32)).max(axis=-1)
                neural_ok = remaining_mask & (diff_n > 5.0)
                if np.any(neural_ok):
                    patch_clean[neural_ok] = neural_patch[neural_ok]
                    remaining_mask = remaining_mask & (~neural_ok)

            if np.any(remaining_mask):
                ring = cv2.dilate(erase_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)), iterations=1)
                ring_excl = cv2.dilate(erase_mask.astype(np.uint8), k_rect3, iterations=1)
                ring_bg = (ring > 0) & (ring_excl == 0) & (protected_mask == 0) & (patch_gray >= ink_thresh)
                if np.sum(ring_bg) <= 20:
                    ring_bg = (ring > 0) & (erase_mask == 0) & (protected_mask == 0) & (patch_gray >= ink_thresh)
                ring_std = float(np.std(patch_gray[ring_bg])) if np.sum(ring_bg) > 20 else 0.0
                if ring_std >= 15.0:
                    rem_u8 = remaining_mask.astype(np.uint8) * 255
                    cleaned_rem = False
                    try:
                        lama = get_shared_simple_lama(device=device) if callable(get_shared_simple_lama) else None
                        if lama is not None:
                            c_lama = _run_cropped_lama_inpaint(patch_clean, rem_u8, lama)
                            if c_lama is not None and c_lama.shape == patch_orig.shape:
                                patch_clean[remaining_mask] = c_lama[remaining_mask]
                                cleaned_rem = True
                    except Exception as lama_err:
                        logger.debug("clean_bubble_crop light gradient SimpleLama: %s", lama_err)
                    if not cleaned_rem:
                        telea_out = cv2.inpaint(patch_clean, rem_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                        patch_clean[remaining_mask] = telea_out[remaining_mask]
                else:
                    if np.sum(ring_bg) > 20:
                        fill_color = np.round(np.median(patch_orig[ring_bg], axis=0)).astype(np.uint8)
                    else:
                        fill_color = local_bg_color
                    if ring_std < 5.0 and np.all(fill_color >= 248):
                        fill_color = np.array([255, 255, 255], dtype=np.uint8)
                    patch_clean[remaining_mask] = fill_color

        return patch_clean

    # ──────────────────────────────────────────────────────────────────────────
    # 2. DARK NARRATION BOXES / DARK BUBBLES
    # ──────────────────────────────────────────────────────────────────────────
    if is_dark_bubble:
        # Local dark background color from interior non-text pixels (preserve vivid dark colors)
        interior_dark = (interior_gray < 115)
        if np.any(interior_dark):
            dark_bg_color = np.round(np.median(interior_orig[interior_dark], axis=0)).astype(np.uint8)
        else:
            dark_bg_color = np.round(np.median(interior_orig, axis=(0, 1))).astype(np.uint8)
            lum = 0.299 * float(dark_bg_color[0]) + 0.587 * float(dark_bg_color[1]) + 0.114 * float(dark_bg_color[2])
            if lum > 125.0:
                dark_bg_color = np.array([25, 25, 25], dtype=np.uint8)

        # Light text ink inside the dark bubble
        dark_bg_luma = float(np.median(interior_gray[interior_dark])) if np.any(interior_dark) else float(np.median(interior_gray))
        light_thresh = max(145, int(dark_bg_luma + 55))
        light_mask = (patch_gray > light_thresh).astype(np.uint8)

        # Connected components on light_mask to isolate true outer light margins from wide interior white text
        num_l, l_labels, l_stats, _ = cv2.connectedComponentsWithStats(light_mask, connectivity=8)
        outer_light_labels = set()
        for lbl in range(1, num_l):
            lx = l_stats[lbl, cv2.CC_STAT_LEFT]
            ly = l_stats[lbl, cv2.CC_STAT_TOP]
            lw = l_stats[lbl, cv2.CC_STAT_WIDTH]
            lh = l_stats[lbl, cv2.CC_STAT_HEIGHT]
            la = l_stats[lbl, cv2.CC_STAT_AREA]
            touches_top_bot = (ly <= 1 or ly + lh >= hp - 1)
            touches_left_right = (lx <= 1 or lx + lw >= wp - 1)
            if (touches_top_bot and lw >= 0.70 * wp) or (touches_left_right and lh >= 0.70 * hp):
                outer_light_labels.add(lbl)
            elif (touches_top_bot or touches_left_right) and la >= 0.25 * (wp * hp):
                outer_light_labels.add(lbl)

        outer_margin_mask = np.isin(l_labels, list(outer_light_labels)).astype(np.uint8) * 255 if outer_light_labels else np.zeros_like(patch_gray)

        # Text ink: light components that do not belong to the outer margin
        text_light_mask = ((light_mask > 0) & (outer_margin_mask == 0)).astype(np.uint8)
        if patch_mask is not None and np.any(patch_mask):
            text_light_mask = cv2.bitwise_or(text_light_mask, ((patch_mask > 0) & (outer_margin_mask == 0)).astype(np.uint8))

        # Dilate text ink by (5, 5) to swallow black outlines around white text (halo protection)
        k_halo = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated_light = cv2.dilate(text_light_mask, k_halo, iterations=1)
        text_erase_mask = cv2.bitwise_and(dilated_light, cv2.bitwise_not(outer_margin_mask))

        patch_clean = patch_orig.copy()
        if np.any(text_erase_mask):
            remaining_mask = (text_erase_mask > 0)
            if neural_patch is not None and neural_patch.shape == patch_orig.shape:
                diff_n = np.abs(neural_patch.astype(np.float32) - patch_orig.astype(np.float32)).max(axis=-1)
                neural_gray = cv2.cvtColor(neural_patch, cv2.COLOR_RGB2GRAY) if neural_patch.ndim == 3 else neural_patch
                neural_ok = remaining_mask & (diff_n > 5.0) & (neural_gray <= dark_bg_luma + 35.0)
                if np.any(neural_ok):
                    patch_clean[neural_ok] = neural_patch[neural_ok]
                    remaining_mask = remaining_mask & (~neural_ok)

            if np.any(remaining_mask):
                # Compute local ring background color around light text
                ring = cv2.dilate(text_erase_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2)
                ring_bg = (ring > 0) & (text_erase_mask == 0) & (outer_margin_mask == 0) & (patch_gray < 115)
                if np.sum(ring_bg) > 20:
                    dark_fill = np.round(np.median(patch_orig[ring_bg], axis=0)).astype(np.uint8)
                else:
                    dark_fill = dark_bg_color

                # Smooth row-wise gradient fill if vertical gradient is present
                ring_gray_vals = patch_gray[ring_bg] if np.sum(ring_bg) > 20 else patch_gray[patch_gray < 115]
                if ring_gray_vals.size > 0 and float(np.std(ring_gray_vals)) >= 5.0:
                    for r_y in range(hp):
                        row_stroke = remaining_mask[r_y]
                        if not np.any(row_stroke):
                            continue
                        y_min = max(0, r_y - 2)
                        y_max = min(hp, r_y + 3)
                        band_mask = (text_erase_mask[y_min:y_max] == 0) & (ring[y_min:y_max] > 0) & (outer_margin_mask[y_min:y_max] == 0) & (patch_gray[y_min:y_max] < 115)
                        band_pix = patch_orig[y_min:y_max][band_mask]
                        if band_pix.shape[0] >= 5:
                            patch_clean[r_y, row_stroke] = np.round(np.median(band_pix.reshape(-1, 3), axis=0)).astype(np.uint8)
                        else:
                            patch_clean[r_y, row_stroke] = dark_fill
                else:
                    patch_clean[remaining_mask] = dark_fill

        return patch_clean

    # ──────────────────────────────────────────────────────────────────────────
    # 3. MID-TONE / COLORED / GRADIENT / ARTWORK / SCREENTONE / BORDERLESS
    # ──────────────────────────────────────────────────────────────────────────
    if patch_mask is not None and np.any(patch_mask > 0):
        text_dil = cv2.dilate((patch_mask > 0).astype(np.uint8) * 255, np.ones((5, 5), np.uint8), iterations=1)
    else:
        text_cand = np.zeros((hp, wp), dtype=np.uint8)
        # Detect dark strokes relative to background mode
        dark_thresh = min(215, max(50, int(round(bg_mode_luma - 28.0))))
        dark = (patch_gray < dark_thresh).astype(np.uint8) * 255
        num_d, d_labels, d_stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        for d_i in range(1, num_d):
            dw = d_stats[d_i, cv2.CC_STAT_WIDTH]
            dh = d_stats[d_i, cv2.CC_STAT_HEIGHT]
            da = d_stats[d_i, cv2.CC_STAT_AREA]
            if 4 <= da <= 0.40 * (wp * hp) and dw < 0.85 * wp and dh < 0.85 * hp:
                text_cand[d_labels == d_i] = 255

        # Also detect bright/white strokes on mid-tone colored/artwork backgrounds
        if interior_p95 > bg_mode_luma + 38.0:
            bright_thresh = min(240, int(round(bg_mode_luma + max(32.0, (interior_p95 - bg_mode_luma) * 0.45))))
            bright = (patch_gray > bright_thresh).astype(np.uint8) * 255
            num_b, b_labels, b_stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
            bright_cand = np.zeros((hp, wp), dtype=np.uint8)
            for b_i in range(1, num_b):
                bw_c = b_stats[b_i, cv2.CC_STAT_WIDTH]
                bh_c = b_stats[b_i, cv2.CC_STAT_HEIGHT]
                ba_c = b_stats[b_i, cv2.CC_STAT_AREA]
                if 4 <= ba_c <= 0.40 * (wp * hp) and bw_c < 0.85 * wp and bh_c < 0.85 * hp:
                    bright_cand[b_labels == b_i] = 255
            if np.any(bright_cand > 0):
                # Dilate slightly more to swallow dark outline halos around white text
                bright_dil = cv2.dilate(bright_cand, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
                text_cand = cv2.bitwise_or(text_cand, bright_dil)

        text_dil = (cv2.dilate(text_cand, k_ellipse, iterations=2) > 0).astype(np.uint8) * 255

    patch_clean = patch_orig.copy()
    if not np.any(text_dil > 0):
        return patch_clean

    remaining_mask = (text_dil > 0)
    if neural_patch is not None and neural_patch.shape == patch_orig.shape:
        diff_n = np.abs(neural_patch.astype(np.float32) - patch_orig.astype(np.float32)).max(axis=-1)
        if float(np.mean(diff_n[remaining_mask] > 5.0)) >= 0.50:
            neural_ok = remaining_mask & (diff_n > 5.0)
            patch_clean[neural_ok] = neural_patch[neural_ok]
            remaining_mask = remaining_mask & (~neural_ok)

    if np.any(remaining_mask):
        rem_u8 = remaining_mask.astype(np.uint8) * 255
        # Sample ring around text (and crop edges if ring too small) excluding text strokes
        ring = cv2.dilate(text_dil, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)), iterations=1)
        ring_excl = cv2.dilate(text_dil, k_rect3, iterations=1)
        ring_bg = (ring > 0) & (ring_excl == 0)
        if np.sum(ring_bg) < 20:
            ring_bg = (ring > 0) & (text_dil == 0)
        if np.sum(ring_bg) < 12:
            ring_bg = (text_dil == 0)

        bg_gray_vals = patch_gray[ring_bg] if np.any(ring_bg) else patch_gray
        bg_std = float(np.std(bg_gray_vals)) if bg_gray_vals.size > 0 else 0.0

        cleaned_rem = False
        if bg_std < 15.0 and np.any(ring_bg):
            bg_median = np.round(np.median(patch_orig[ring_bg], axis=0)).astype(np.uint8)
            patch_clean[remaining_mask] = bg_median
            cleaned_rem = True
        else:
            try:
                lama = get_shared_simple_lama(device=device) if callable(get_shared_simple_lama) else None
                if lama is not None:
                    c_lama = _run_cropped_lama_inpaint(patch_clean, rem_u8, lama)
                    if c_lama is not None and c_lama.shape == patch_orig.shape:
                        patch_clean[remaining_mask] = c_lama[remaining_mask]
                        cleaned_rem = True
            except Exception as lama_err:
                logger.error("clean_bubble_crop SimpleLama fallback: %s", lama_err, exc_info=True)
        if not cleaned_rem:
            clean_telea = cv2.inpaint(patch_clean, rem_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            patch_clean[remaining_mask] = clean_telea[remaining_mask]

    return patch_clean


_fallback_lama_infer_lock = threading.Lock()
try:
    from manga_text_cleaner import (
        get_shared_simple_lama,
        inpaint_mask_with_simple_lama,
        _shared_simple_lama_infer_lock,
    )
except (ImportError, ValueError):
    try:
        from .manga_text_cleaner import (
            get_shared_simple_lama,
            inpaint_mask_with_simple_lama,
            _shared_simple_lama_infer_lock,
        )
    except (ImportError, ValueError):
        try:
            from ml.manga_text_cleaner import (
                get_shared_simple_lama,
                inpaint_mask_with_simple_lama,
                _shared_simple_lama_infer_lock,
            )
        except (ImportError, ValueError):
            get_shared_simple_lama = None
            inpaint_mask_with_simple_lama = None
            _shared_simple_lama_infer_lock = _fallback_lama_infer_lock


def _run_cropped_lama_inpaint(patch_clean: np.ndarray, rem_u8: np.ndarray, lama) -> Optional[np.ndarray]:
    if callable(inpaint_mask_with_simple_lama):
        out = inpaint_mask_with_simple_lama(patch_clean, rem_u8, lama)
        if out is not None:
            return out
    hp, wp = patch_clean.shape[:2]
    lock = _shared_simple_lama_infer_lock or _fallback_lama_infer_lock
    with lock:
        inp_pil = lama(Image.fromarray(patch_clean), Image.fromarray(rem_u8))
    c_lama = np.array(inp_pil.convert("RGB"))
    if c_lama.shape[:2] != (hp, wp):
        c_lama = cv2.resize(c_lama, (wp, hp), interpolation=cv2.INTER_LANCZOS4)
    return c_lama

