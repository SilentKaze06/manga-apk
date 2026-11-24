"""
Text masking, boundary protection, SimpleLama inpainting, and bubble crop cleaning.
"""
import logging
import os
import threading
import cv2
import numpy as np
import torch
from PIL import Image

try:
    from detection import _has_continuous_dark_segment
except (ImportError, ValueError):
    from .detection import _has_continuous_dark_segment

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
    p85 = float(np.percentile(patch, 85))
    med = float(np.median(patch))

    # 1. Dark background / black narration box: check med < 115 FIRST
    if med < 115:
        thresh_val = max(145, int(med + 45))
        stroke_binary = (patch > thresh_val).astype(np.uint8) * 255
    elif p85 >= 160:
        # In manga, dialogue bubbles with dark text on light background (only if med >= 115)
        thresh_val = min(232, int(p85 * 0.90))
        stroke_binary = (patch < thresh_val).astype(np.uint8) * 255
    else:
        # Screentone or midtone
        thresh_val = min(180, int(p85 * 0.85))
        stroke_binary = (patch < thresh_val).astype(np.uint8) * 255

    hp, wp = patch.shape
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
        touches_edge = (
            st[cv2.CC_STAT_LEFT] <= 1 or
            st[cv2.CC_STAT_TOP] <= 1 or
            st[cv2.CC_STAT_LEFT] + st[cv2.CC_STAT_WIDTH] >= wp - 1 or
            st[cv2.CC_STAT_TOP] + st[cv2.CC_STAT_HEIGHT] >= hp - 1
        )
        is_border_wall = touches_edge and (
            np.any(long_lines[labels == l] > 0) or
            st[cv2.CC_STAT_AREA] > int(0.30 * wp * hp) or
            (st[cv2.CC_STAT_WIDTH] >= int(0.60 * wp) and (st[cv2.CC_STAT_TOP] <= 2 or st[cv2.CC_STAT_TOP] + st[cv2.CC_STAT_HEIGHT] >= hp - 2)) or
            (st[cv2.CC_STAT_HEIGHT] >= int(0.60 * hp) and (st[cv2.CC_STAT_LEFT] <= 2 or st[cv2.CC_STAT_LEFT] + st[cv2.CC_STAT_WIDTH] >= wp - 2))
        )
        if is_border_wall:
            lines[labels == l] = 255

    stroke_clean = cv2.bitwise_and(stroke_binary, cv2.bitwise_not(lines))

    # For dark background (med < 115), apply 2px dilation; for light bubbles, 1 iteration
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    stroke_dilated = cv2.dilate(stroke_clean, k, iterations=2 if med < 115 else 1)

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

    hp, wp = patch_orig.shape[:2]
    if hp <= 6 or wp <= 6:
        return patch_orig.copy()

    if patch_gray is None:
        patch_gray = cv2.cvtColor(patch_orig, cv2.COLOR_RGB2GRAY)

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

    full_median = float(np.median(patch_gray))
    full_p15 = float(np.percentile(patch_gray, 15))
    full_p85 = float(np.percentile(patch_gray, 85))

    dark_interior_pixels = interior_gray[interior_gray < 130]
    has_dark_interior_bg = (dark_interior_pixels.size >= 10 and float(np.median(dark_interior_pixels)) <= 90)

    # Determine is_dark_bubble from the INTERIOR region of the crop (ignoring outer margins)
    is_dark_bubble = (
        (interior_median < 95)
        or (interior_p15 < 55 and interior_median < 120 and interior_p85 < 175)
        or (interior_median < 110 and interior_p25 < 65 and has_dark_interior_bg)
        or (full_median < 65)
        or (full_p15 < 40 and full_p85 < 110)
    )

    is_white_bubble = False
    if not is_dark_bubble:
        is_white_bubble = (interior_median >= 175) or (interior_p85 >= 205 and interior_median >= 145) or (full_median >= 190)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. WHITE OR CREAM BUBBLES
    # ──────────────────────────────────────────────────────────────────────────
    if is_white_bubble:
        # Determine local background color from non-ink pixels inside the padded interior
        non_ink_interior = (interior_gray >= 185)
        if np.any(non_ink_interior):
            local_bg_color = np.median(interior_orig[non_ink_interior], axis=0).astype(np.uint8)
        else:
            non_ink_full = (patch_gray >= 185)
            if np.any(non_ink_full):
                local_bg_color = np.median(patch_orig[non_ink_full], axis=0).astype(np.uint8)
            else:
                local_bg_color = np.array([255, 255, 255], dtype=np.uint8)

        # Dark ink mask
        dark_mask = (patch_gray < 185).astype(np.uint8)
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
        # Note: Do NOT add straight_lines connected components to keep_labels; straight_lines
        # itself protects the structural frame directly without protecting interior letters touching it.
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

            # 1. Outer artwork / clothing touching crop boundaries and spanning across
            touches_top = (ly <= 1)
            touches_bot = (ly + lh >= hp - 1)
            touches_left = (lx <= 1)
            touches_right = (lx + lw >= wp - 1)

            # Large solid background artwork touching edge
            if (touches_top or touches_bot or touches_left or touches_right):
                if (lw >= 0.70 * wp and (touches_top or touches_bot) and la >= 0.15 * (wp * hp)):
                    keep_labels.add(lbl)
                    continue
                if (lh >= 0.70 * hp and (touches_left or touches_right) and la >= 0.15 * (wp * hp)):
                    keep_labels.add(lbl)
                    continue

            # If this component is primarily composed of straight box frames, let straight_lines
            # protect the frame directly without protecting interior letters touching the frame.
            if has_straight_lines and np.sum((labels == lbl) & (straight_lines > 0)) >= 0.25 * la:
                continue

            # 2. Large curved oval bubble contour outline spanning across the bubble
            # Must be hollow around center (center_ratio <= 0.08)
            # so large words/letters (e.g. 'MUS... TOR', 'HAR') are never misclassified as bubble contours.
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

        # Build protected_mask:
        if keep_labels:
            protected_mask = np.isin(labels, list(keep_labels)).astype(np.uint8) * 255
        else:
            protected_mask = np.zeros_like(dark_mask)

        if has_straight_lines:
            protected_mask = cv2.bitwise_or(protected_mask, (straight_lines > 0).astype(np.uint8) * 255)

        # c) Identify letter ink to erase
        letter_mask = ((dark_mask > 0) & (protected_mask == 0)).astype(np.uint8)

        # Include external OCR mask if provided (excluding protected structures)
        if patch_mask is not None and np.any(patch_mask):
            ext_mask = ((patch_mask > 0) & (protected_mask == 0)).astype(np.uint8)
            letter_mask = cv2.bitwise_or(letter_mask, ext_mask)

        # Dilate letter ink by 2 px with (3, 3) kernel
        dilated_letters = cv2.dilate(letter_mask, k_rect3, iterations=2)

        # CRITICAL: Subtract protected_mask AFTER dilating erase_mask so borders are never nicked
        dilated_erase = cv2.bitwise_and(dilated_letters, cv2.bitwise_not(protected_mask))

        # Softly clean anti-aliased gray halo pixels (185..240) touching the erased text
        anti_aliased_mask = ((patch_gray >= 185) & (patch_gray <= 240)).astype(np.uint8)
        halo_zone = cv2.dilate(dilated_erase, k_rect3, iterations=1)
        halo_candidates = (halo_zone > 0) & (anti_aliased_mask > 0) & (protected_mask == 0)

        erase_mask = cv2.bitwise_and(
            cv2.bitwise_or(dilated_erase, halo_candidates.astype(np.uint8)),
            cv2.bitwise_not(protected_mask)
        )

        patch_clean = patch_orig.copy()
        if np.any(erase_mask):
            # If neural_patch is available and already cleaned text
            if neural_patch is not None and neural_patch.shape == patch_orig.shape:
                diff_n = np.abs(neural_patch.astype(np.float32) - patch_orig.astype(np.float32)).max(axis=-1)
                if np.any(diff_n[erase_mask > 0] > 5.0):
                    patch_clean[erase_mask > 0] = neural_patch[erase_mask > 0]
                    return patch_clean

            # Compute local ring background color around the erased text if possible
            ring = cv2.dilate(erase_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2)
            ring_bg = (ring > 0) & (erase_mask == 0) & (protected_mask == 0) & (patch_gray >= 170)
            if np.sum(ring_bg) > 20:
                fill_color = np.median(patch_orig[ring_bg], axis=0).astype(np.uint8)
            else:
                fill_color = local_bg_color

            patch_clean[erase_mask > 0] = fill_color

        return patch_clean

    # ──────────────────────────────────────────────────────────────────────────
    # 2. DARK NARRATION BOXES / DARK BUBBLES
    # ──────────────────────────────────────────────────────────────────────────
    if is_dark_bubble and not is_white_bubble:
        # Local dark background color from interior non-text pixels
        interior_dark = (interior_gray < 110)
        if np.any(interior_dark):
            dark_bg_color = np.median(interior_orig[interior_dark], axis=0).astype(np.uint8)
        else:
            dark_bg_color = np.median(interior_orig, axis=(0, 1)).astype(np.uint8)
        if np.mean(dark_bg_color) > 100:
            dark_bg_color = np.array([25, 25, 25], dtype=np.uint8)

        # Light text ink inside the dark bubble
        dark_bg_luma = float(np.median(interior_gray[interior_dark])) if np.any(interior_dark) else float(np.median(interior_gray))
        light_thresh = max(125, int(dark_bg_luma + 35))
        light_mask = (patch_gray > light_thresh).astype(np.uint8)

        # Connected components on light_mask to isolate text from outer light margins
        num_l, l_labels, l_stats, _ = cv2.connectedComponentsWithStats(light_mask, connectivity=8)
        outer_light_labels = set()
        for lbl in range(1, num_l):
            lx = l_stats[lbl, cv2.CC_STAT_LEFT]
            ly = l_stats[lbl, cv2.CC_STAT_TOP]
            lw = l_stats[lbl, cv2.CC_STAT_WIDTH]
            lh = l_stats[lbl, cv2.CC_STAT_HEIGHT]
            la = l_stats[lbl, cv2.CC_STAT_AREA]
            touches_edge = (lx <= 0 or ly <= 0 or lx + lw >= wp or ly + lh >= hp)
            if touches_edge and (lw >= 0.40 * wp or lh >= 0.40 * hp or la >= 0.10 * (wp * hp)):
                outer_light_labels.add(lbl)

        # Filter: keep outer boundary margins in outer_light_labels
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
            if neural_patch is not None and neural_patch.shape == patch_orig.shape:
                diff_n = np.abs(neural_patch.astype(np.float32) - patch_orig.astype(np.float32)).max(axis=-1)
                if np.any(diff_n[text_erase_mask > 0] > 5.0):
                    patch_clean[text_erase_mask > 0] = neural_patch[text_erase_mask > 0]
                    return patch_clean

            # Compute local ring background color around light text
            ring = cv2.dilate(text_erase_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2)
            ring_bg = (ring > 0) & (text_erase_mask == 0) & (outer_margin_mask == 0) & (patch_gray < 110)
            if np.sum(ring_bg) > 20:
                dark_fill = np.median(patch_orig[ring_bg], axis=0).astype(np.uint8)
            else:
                dark_fill = dark_bg_color

            # Smooth row-wise gradient fill if vertical gradient is present (for gradient dark boxes)
            ring_gray_vals = patch_gray[ring_bg] if np.sum(ring_bg) > 20 else patch_gray[patch_gray < 110]
            if ring_gray_vals.size > 0 and float(np.std(ring_gray_vals)) >= 5.0:
                for r_y in range(hp):
                    row_stroke = text_erase_mask[r_y] > 0
                    if not np.any(row_stroke):
                        continue
                    y_min = max(0, r_y - 2)
                    y_max = min(hp, r_y + 3)
                    band_mask = (text_erase_mask[y_min:y_max] == 0) & (ring[y_min:y_max] > 0) & (outer_margin_mask[y_min:y_max] == 0) & (patch_gray[y_min:y_max] < 110)
                    band_pix = patch_orig[y_min:y_max][band_mask]
                    if band_pix.shape[0] >= 5:
                        patch_clean[r_y, row_stroke] = np.median(band_pix.reshape(-1, 3), axis=0).astype(np.uint8)
                    else:
                        patch_clean[r_y, row_stroke] = dark_fill
            else:
                patch_clean[text_erase_mask > 0] = dark_fill

        return patch_clean

    # ──────────────────────────────────────────────────────────────────────────
    # 3. ARTWORK / SCREENTONE / BORDERLESS
    # ──────────────────────────────────────────────────────────────────────────
    dark_thresh = min(215, int(interior_p85 * 0.88))
    dark = (patch_gray < dark_thresh).astype(np.uint8) * 255
    if patch_mask is not None and np.any(patch_mask > 0):
        text_dil = (patch_mask > 0).astype(np.uint8) * 255
    else:
        num_d, d_labels, d_stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        text_cand = np.zeros_like(dark)
        for d_i in range(1, num_d):
            dw = d_stats[d_i, cv2.CC_STAT_WIDTH]
            dh = d_stats[d_i, cv2.CC_STAT_HEIGHT]
            da = d_stats[d_i, cv2.CC_STAT_AREA]
            if 4 <= da <= 0.40 * (wp * hp) and dw < 0.85 * wp and dh < 0.85 * hp:
                text_cand |= (d_labels == d_i)
        text_dil = cv2.dilate(text_cand, k_ellipse, iterations=2)

    patch_clean = patch_orig.copy()
    cleaned_with_model = False
    if neural_patch is not None and neural_patch.shape == patch_orig.shape:
        patch_clean[text_dil > 0] = neural_patch[text_dil > 0]
        cleaned_with_model = True
    if not cleaned_with_model:
        # Smart hybrid inpainting:
        # Sample boundary/edge pixels of the crop (excluding text mask)
        edge_mask = np.zeros((hp, wp), dtype=bool)
        edge_mask[:2, :] = True
        edge_mask[-2:, :] = True
        edge_mask[:, :2] = True
        edge_mask[:, -2:] = True
        if text_dil is not None:
            edge_mask &= (text_dil == 0)

        edge_gray = patch_gray[edge_mask] if np.any(edge_mask) else patch_gray
        edge_std = float(np.std(edge_gray)) if edge_gray.size > 0 else 0.0

        if edge_std < 12.0:
            # Flat background: fast fill with median background color directly, skipping heavy SimpleLama
            bg_median = np.median(patch_orig[edge_mask], axis=0).astype(np.uint8) if np.any(edge_mask) else np.median(patch_orig.reshape(-1, 3), axis=0).astype(np.uint8)
            patch_clean[text_dil > 0] = bg_median
            cleaned_with_model = True
        else:
            # Complex, gradient, or artwork background (edge_std >= 12.0): use SimpleLama
            try:
                lama = get_shared_simple_lama(device=device)
                if lama is not None:
                    inp_pil = lama(Image.fromarray(patch_orig), Image.fromarray(text_dil))
                    c_lama = np.array(inp_pil.convert("RGB"))
                    if c_lama.shape[:2] != (hp, wp):
                        c_lama = cv2.resize(c_lama, (wp, hp), interpolation=cv2.INTER_LANCZOS4)
                    if c_lama.shape == patch_orig.shape:
                        patch_clean[text_dil > 0] = c_lama[text_dil > 0]
                        cleaned_with_model = True
            except Exception as lama_err:
                logger.error("clean_bubble_crop SimpleLama fallback: %s", lama_err, exc_info=True)
    if not cleaned_with_model:
        clean_telea = cv2.inpaint(patch_orig, text_dil, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        patch_clean[text_dil > 0] = clean_telea[text_dil > 0]

    return patch_clean


try:
    from manga_text_cleaner import get_shared_simple_lama
except (ImportError, ValueError):
    try:
        from .manga_text_cleaner import get_shared_simple_lama
    except (ImportError, ValueError):
        try:
            from ml.manga_text_cleaner import get_shared_simple_lama
        except (ImportError, ValueError):
            get_shared_simple_lama = None

