"""
2-path background cleaning & Manga-LaMa neural inpainting:
1. Fast path (uniform solid `text_bubble`): median ring-color fill when ring variance is very low.
2. Neural inpainting path (`text_free` and colored/gradient/textured `text_bubble`):
   padded crop aligned to multiple of 8 -> Manga-finetuned LaMa -> feathered mask composite
   so unmasked artwork remains 100% untouched.
"""
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

if not torch.cuda.is_available():
    try:
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        cv2.setNumThreads(num_threads)
    except Exception as exc:
        logger.debug("CPU thread configuration skipped: %s", exc)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def generate_ink_stroke_mask(gray_np: np.ndarray, box: List[int], pad: int = 2) -> np.ndarray:
    """
    Lightweight fallback stroke mask generator when a neural CTD mask is unavailable.
    Only operates strictly inside `box` (+ `pad`).
    """
    h, w = gray_np.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
    px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
    px2, py2 = min(w, x2 + pad), min(h, y2 + pad)
    if px2 <= px1 or py2 <= py1:
        return np.zeros((h, w), dtype=np.uint8)

    patch = gray_np[py1:py2, px1:px2]
    med = float(np.median(patch))
    if med < 120.0:
        thresh = max(145, int(med + 40))
        stroke = (patch > thresh).astype(np.uint8) * 255
    else:
        thresh = min(230, max(75, int(med - 30)))
        stroke = (patch < thresh).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    stroke = cv2.dilate(stroke, k, iterations=1)
    out = np.zeros((h, w), dtype=np.uint8)
    out[py1:py2, px1:px2] = stroke
    return out


def _run_lama_aligned_crop(
    crop_rgb: np.ndarray,
    crop_mask_u8: np.ndarray,
    lama: Any,
) -> Optional[np.ndarray]:
    """
    Align crop to a multiple of 8 using reflection padding, run Manga-LaMa,
    and unpad back to the exact original crop dimensions.
    """
    if lama is None or crop_rgb is None or crop_mask_u8 is None:
        return None
    ch, cw = crop_rgb.shape[:2]
    if ch <= 0 or cw <= 0:
        return None

    pad_h = (8 - ch % 8) % 8
    pad_w = (8 - cw % 8) % 8
    if pad_h > 0 or pad_w > 0:
        in_rgb = cv2.copyMakeBorder(crop_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        in_mask = cv2.copyMakeBorder(crop_mask_u8, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    else:
        in_rgb = crop_rgb
        in_mask = crop_mask_u8

    in_mask_u8 = (in_mask > 0).astype(np.uint8) * 255
    lock = _shared_simple_lama_infer_lock or _fallback_lama_infer_lock
    with lock:
        inp_pil = lama(Image.fromarray(in_rgb), Image.fromarray(in_mask_u8))
    out_rgb = np.array(inp_pil.convert("RGB"))
    if out_rgb.shape[:2] != in_rgb.shape[:2]:
        out_rgb = cv2.resize(out_rgb, (in_rgb.shape[1], in_rgb.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    return out_rgb[:ch, :cw]


def _feather_composite(
    dst_rgb: np.ndarray,
    src_rgb: np.ndarray,
    mask_u8: np.ndarray,
    feather_px: int = 2,
) -> np.ndarray:
    """
    Composite `src_rgb` onto `dst_rgb` strictly within `mask_u8` with subtle 1-2 px
    edge feathering so unmasked artwork remains 100% untouched.
    """
    core = (mask_u8 > 0).astype(np.float32)
    if not np.any(core > 0):
        return dst_rgb

    if feather_px <= 0:
        out = dst_rgb.copy()
        sel = core > 0
        out[sel] = src_rgb[sel]
        return out

    k_size = 3 if feather_px == 1 else 5
    # Dilate by 1 px for soft outer transition, then blur, while keeping unmasked outside 0
    outer = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    blurred = cv2.GaussianBlur(outer, (k_size, k_size), 0)
    alpha = np.where(core > 0, np.maximum(core, blurred), blurred * outer)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]

    blended = dst_rgb.astype(np.float32) * (1.0 - alpha) + src_rgb.astype(np.float32) * alpha
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)


def _is_uniform_ring(
    crop_rgb: np.ndarray,
    dilated_mask_u8: np.ndarray,
    ring_bounds: Tuple[int, int, int, int],
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Sample the ring of pixels immediately outside `dilated_mask_u8` inside `ring_bounds`.
    Returns (True, median_rgb) only when color variance is very low (near-uniform solid bubble).
    """
    ch, cw = crop_rgb.shape[:2]
    rx1, ry1, rx2, ry2 = ring_bounds
    rx1, ry1 = max(0, rx1), max(0, ry1)
    rx2, ry2 = min(cw, rx2), min(ch, ry2)
    if rx2 <= rx1 or ry2 <= ry1:
        return False, None

    bound_mask = np.zeros((ch, cw), dtype=bool)
    bound_mask[ry1:ry2, rx1:rx2] = True

    k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ring_outer = cv2.dilate((dilated_mask_u8 > 0).astype(np.uint8), k_ring, iterations=1) > 0
    ring = ring_outer & (dilated_mask_u8 == 0) & bound_mask
    if int(np.sum(ring)) < 16:
        return False, None

    ring_rgb = crop_rgb[ring]
    max_ch_std = float(np.max(np.std(ring_rgb.astype(np.float32), axis=0)))
    ring_gray = cv2.cvtColor(ring_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY).ravel()
    luma_std = float(np.std(ring_gray.astype(np.float32)))

    if max_ch_std < 10.0 and luma_std < 8.0:
        median_rgb = np.round(np.median(ring_rgb, axis=0)).astype(np.uint8)
        return True, median_rgb
    return False, None


def clean_block_with_mask(
    page_rgb: np.ndarray,
    block: Dict[str, Any],
    page_mask: Optional[np.ndarray],
    lama: Any = None,
    device: Any = DEVICE,
    dilate_px: int = 4,
    pad_px: int = 48,
    feather_px: int = 2,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Clean a single detected text block using the 2-path cleaner:
    - Fast path: `text_bubble` with very low ring color variance -> median ring fill.
    - Neural path: `text_free` or non-uniform/colored/gradient `text_bubble` -> Manga-LaMa.
    Returns (updated_page_rgb, applied_mask_u8_on_page, path_used).
    """
    h, w = page_rgb.shape[:2]
    tb = block.get("text_box") or block.get("box") or block.get("bbox") or [0, 0, 0, 0]
    tx1 = max(0, min(w, int(tb[0])))
    ty1 = max(0, min(h, int(tb[1])))
    tx2 = max(0, min(w, int(tb[2])))
    ty2 = max(0, min(h, int(tb[3])))
    if tx2 <= tx1 + 2 or ty2 <= ty1 + 2:
        return page_rgb, np.zeros((h, w), dtype=np.uint8), "skipped"

    # Extract character mask strictly within text_box
    char_mask = np.zeros((h, w), dtype=np.uint8)
    if page_mask is not None and page_mask.shape[:2] == (h, w):
        char_mask[ty1:ty2, tx1:tx2] = (page_mask[ty1:ty2, tx1:tx2] > 0).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2GRAY)
        char_mask = generate_ink_stroke_mask(gray, [tx1, ty1, tx2, ty2], pad=2)

    if not np.any(char_mask > 0):
        return page_rgb, np.zeros((h, w), dtype=np.uint8), "empty_mask"

    k_sz = max(3, int(dilate_px) | 1)
    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (k_sz, k_sz))

    # Determine context crop around text_box + dilated mask
    cx1 = max(0, tx1 - pad_px)
    cy1 = max(0, ty1 - pad_px)
    cx2 = min(w, tx2 + pad_px)
    cy2 = min(h, ty2 + pad_px)

    crop_rgb = page_rgb[cy1:cy2, cx1:cx2]
    crop_char = char_mask[cy1:cy2, cx1:cx2]
    crop_dilated = cv2.dilate(crop_char, k_dil, iterations=1)

    label = block.get("label", "")
    category = block.get("category", "")
    is_text_bubble = (label == "text_bubble") or (
        label != "text_free" and category in ("speech_bubble", "narration", "narration_box", "thought_bubble")
    )

    # Ring sampling bounds inside bubble (if bubble_box available) or text_box
    bb = block.get("bubble_box") or block.get("box") or [tx1, ty1, tx2, ty2]
    ring_bounds = (
        max(0, int(bb[0]) - cx1),
        max(0, int(bb[1]) - cy1),
        min(cx2 - cx1, int(bb[2]) - cx1),
        min(cy2 - cy1, int(bb[3]) - cy1),
    )

    out_page = page_rgb.copy()
    applied_mask = np.zeros((h, w), dtype=np.uint8)

    if is_text_bubble:
        is_uniform, median_rgb = _is_uniform_ring(crop_rgb, crop_dilated, ring_bounds)
        if is_uniform and median_rgb is not None:
            fill_patch = np.empty_like(crop_rgb)
            fill_patch[:, :] = median_rgb
            composited = _feather_composite(crop_rgb, fill_patch, crop_dilated, feather_px=1)
            out_page[cy1:cy2, cx1:cx2] = composited
            applied_mask[cy1:cy2, cx1:cx2] = crop_dilated
            return out_page, applied_mask, "fast_solid_fill"

    # Neural inpainting path (text_free or colored/gradient/textured text_bubble)
    if lama is None and callable(get_shared_simple_lama):
        lama = get_shared_simple_lama(device=device)

    inpainted_crop = None
    if lama is not None:
        try:
            inpainted_crop = _run_lama_aligned_crop(crop_rgb, crop_dilated, lama)
        except Exception as exc:
            logger.warning("Manga-LaMa crop inpainting failed, falling back to Telea: %s", exc)
            inpainted_crop = None

    if inpainted_crop is None:
        inpainted_crop = cv2.inpaint(crop_rgb, crop_dilated, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        path_used = "telea_fallback"
    else:
        path_used = "manga_lama"

    composited = _feather_composite(crop_rgb, inpainted_crop, crop_dilated, feather_px=feather_px)
    out_page[cy1:cy2, cx1:cx2] = composited
    applied_mask[cy1:cy2, cx1:cx2] = crop_dilated
    return out_page, applied_mask, path_used


def clean_page_blocks(
    page_rgb: np.ndarray,
    blocks: List[Dict[str, Any]],
    page_mask: Optional[np.ndarray],
    lama: Any = None,
    device: Any = DEVICE,
    dilate_px: int = 4,
    pad_px: int = 48,
    feather_px: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean all specified blocks on a page using the 2-path cleaner.
    Unmasked artwork remains 100% untouched.
    """
    h, w = page_rgb.shape[:2]
    cleaned = page_rgb.copy()
    total_mask = np.zeros((h, w), dtype=np.uint8)
    if not blocks:
        return cleaned, total_mask

    if lama is None and callable(get_shared_simple_lama):
        lama = get_shared_simple_lama(device=device)

    for block in blocks:
        cleaned, block_applied, path_used = clean_block_with_mask(
            cleaned,
            block,
            page_mask=page_mask,
            lama=lama,
            device=device,
            dilate_px=dilate_px,
            pad_px=pad_px,
            feather_px=feather_px,
        )
        if np.any(block_applied > 0):
            total_mask = np.maximum(total_mask, block_applied)
            block["cleaned"] = True
            block["clean_path"] = path_used
    return cleaned, total_mask


def clean_bubble_crop(
    patch_orig: np.ndarray,
    patch_gray: Optional[np.ndarray] = None,
    patch_mask: Optional[np.ndarray] = None,
    neural_patch: Optional[np.ndarray] = None,
    inpainter_model: Any = None,
    inpaint_transform: Any = None,
    device: Any = DEVICE,
    label: str = "text_bubble",
) -> np.ndarray:
    """
    Backward-compatible single-crop wrapper around the 2-path cleaner.
    """
    if patch_orig is None or patch_orig.size == 0:
        return patch_orig

    if patch_orig.ndim == 2:
        rgb = cv2.cvtColor(patch_orig, cv2.COLOR_GRAY2RGB)
    elif patch_orig.ndim == 3 and patch_orig.shape[2] == 1:
        rgb = cv2.cvtColor(patch_orig[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif patch_orig.ndim == 3 and patch_orig.shape[2] == 4:
        rgb = cv2.cvtColor(patch_orig, cv2.COLOR_RGBA2RGB)
    else:
        rgb = patch_orig

    hp, wp = rgb.shape[:2]
    if hp <= 4 or wp <= 4:
        return rgb.copy()

    fake_block = {
        "text_box": [0, 0, wp, hp],
        "box": [0, 0, wp, hp],
        "label": label,
        "category": "speech_bubble" if label == "text_bubble" else "free_text",
    }
    cleaned, _, _ = clean_block_with_mask(
        rgb,
        fake_block,
        page_mask=patch_mask,
        device=device,
        pad_px=0,
    )
    return cleaned
