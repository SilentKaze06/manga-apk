import os
import sys
import logging
import threading
from typing import Optional, List, Any
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

_shared_simple_lama = None
_shared_simple_lama_failed = False
_shared_simple_lama_lock = threading.Lock()
_shared_simple_lama_infer_lock = threading.Lock()


def inpaint_mask_with_simple_lama(
    image_rgb_arr: np.ndarray,
    mask_u8: np.ndarray,
    lama: Any,
    pad: int = 64,
    max_tile: int = 1024,
) -> Optional[np.ndarray]:
    """Run SimpleLama tightly cropped per connected region (with tiling > max_tile) under inference lock."""
    if lama is None or image_rgb_arr is None or mask_u8 is None:
        return None
    mask_bin = (np.asarray(mask_u8) > 0).astype(np.uint8)
    if not np.any(mask_bin):
        return None

    h, w = image_rgb_arr.shape[:2]
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    rois = []
    for lbl in range(1, num_labels):
        bx, by, bw, bh, _ = stats[lbl]
        if bw <= 0 or bh <= 0:
            continue
        rois.append([
            max(0, int(bx) - pad),
            max(0, int(by) - pad),
            min(w, int(bx + bw) + pad),
            min(h, int(by + bh) + pad),
        ])
    if not rois:
        return None

    merged = []
    for box in rois:
        cur = box
        changed = True
        while changed:
            changed = False
            rest = []
            for other in merged:
                if not (cur[2] <= other[0] or other[2] <= cur[0] or cur[3] <= other[1] or other[3] <= cur[1]):
                    cur = [
                        min(cur[0], other[0]),
                        min(cur[1], other[1]),
                        max(cur[2], other[2]),
                        max(cur[3], other[3]),
                    ]
                    changed = True
                else:
                    rest.append(other)
            merged = rest
        merged.append(cur)

    result = image_rgb_arr.copy()

    def _run_lama_patch(img_patch: np.ndarray, m_patch: np.ndarray) -> np.ndarray:
        m_img = (m_patch > 0).astype(np.uint8) * 255
        ph, pw = img_patch.shape[:2]
        with _shared_simple_lama_infer_lock:
            inp_pil = lama(Image.fromarray(img_patch), Image.fromarray(m_img))
        out = np.array(inp_pil.convert("RGB"))
        if out.shape[:2] != (ph, pw):
            out = cv2.resize(out, (pw, ph), interpolation=cv2.INTER_LANCZOS4)
        return out

    tile_pad = min(pad, max_tile // 4)
    step = max(64, max_tile - 2 * tile_pad)

    for rx1, ry1, rx2, ry2 in merged:
        rw, rh = rx2 - rx1, ry2 - ry1
        if rw <= max_tile and rh <= max_tile:
            crop_mask = mask_bin[ry1:ry2, rx1:rx2] > 0
            out = _run_lama_patch(result[ry1:ry2, rx1:rx2], crop_mask)
            result[ry1:ry2, rx1:rx2][crop_mask] = out[crop_mask]
        else:
            for cy1 in range(ry1, ry2, step):
                cy2 = min(ry2, cy1 + step)
                ty1 = max(ry1, cy1 - tile_pad)
                ty2 = min(ry2, cy2 + tile_pad)
                for cx1 in range(rx1, rx2, step):
                    cx2 = min(rx2, cx1 + step)
                    core_mask = mask_bin[cy1:cy2, cx1:cx2] > 0
                    if not np.any(core_mask):
                        continue
                    tx1 = max(rx1, cx1 - tile_pad)
                    tx2 = min(rx2, cx2 + tile_pad)
                    out = _run_lama_patch(result[ty1:ty2, tx1:tx2], mask_bin[ty1:ty2, tx1:tx2])
                    iy1, iy2 = cy1 - ty1, cy2 - ty1
                    ix1, ix2 = cx1 - tx1, cx2 - tx1
                    result[cy1:cy2, cx1:cx2][core_mask] = out[iy1:iy2, ix1:ix2][core_mask]
    return result


def get_shared_simple_lama(device=None):
    global _shared_simple_lama, _shared_simple_lama_failed
    if _shared_simple_lama is not None:
        return _shared_simple_lama
    if _shared_simple_lama_failed:
        return None
    with _shared_simple_lama_lock:
        if _shared_simple_lama is not None:
            return _shared_simple_lama
        if _shared_simple_lama_failed:
            return None
        try:
            from simple_lama_inpainting import SimpleLama
            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            _shared_simple_lama = SimpleLama(device=dev)
        except Exception as e:
            logger.warning("Failed to load shared SimpleLama: %s (disabling further attempts)", e)
            _shared_simple_lama_failed = True
            return None
    return _shared_simple_lama

MODEL_REPO = "a-b-c-x-y-z/Manga-Text-Segmentation-2025"
MODEL_FILENAME = "model.pth"
ENCODER = "tu-efficientnetv2_rw_m"


def convert_batchnorm_to_groupnorm(module):
    """
    Recursively replaces all nn.BatchNorm2d layers with nn.GroupNorm layers
    matching the exact architecture from petersunde/Manga-Text-Cleaner.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            num_groups = 8
            if num_channels < num_groups or num_channels % num_groups != 0:
                for groups in range(min(num_channels, 8), 1, -1):
                    if num_channels % groups == 0:
                        num_groups = groups
                        break
                else:
                    num_groups = 1
            setattr(
                module,
                name,
                nn.GroupNorm(num_groups=num_groups, num_channels=num_channels),
            )
        else:
            convert_batchnorm_to_groupnorm(child)


class MangaTextCleaner:
    """
    State-of-the-Art Neural Manga Text Segmentation & Page Cleaning Engine.
    Directly ported from Hugging Face Space 'petersunde/Manga-Text-Cleaner'
    using 'a-b-c-x-y-z/Manga-Text-Segmentation-2025'.
    """

    def __init__(self, device=None, model_path=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self._model = None
        self._load_failed = False
        self._model_lock = threading.RLock()

    def load_model(self):
        with self._model_lock:
            if self._model is not None:
                return self._model
            if self._load_failed:
                return None

            try:
                import segmentation_models_pytorch as smp
                from huggingface_hub import hf_hub_download
            except ImportError as e:
                logger.warning("MangaTextCleaner dependencies missing: %s", e)
                self._load_failed = True
                return None

            # Resolve checkpoint file path
            checkpoint_path = self.model_path
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                candidates = [
                    os.environ.get("MANGA_TEXT_CLEANER_WEIGHTS"),
                    "/opt/models/manga_text_cleaner/model.pth",
                    os.environ.get("MANGA_TEXT_SEG_MODEL_PATH"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "manga_text_segmentation.pth"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", MODEL_FILENAME),
                ]
                for c in candidates:
                    if c and os.path.isfile(c):
                        checkpoint_path = c
                        break

            if not checkpoint_path or not os.path.isfile(checkpoint_path):
                try:
                    cache_dir = os.environ.get("HF_HOME")
                    logger.info("Downloading %s from %s...", MODEL_FILENAME, MODEL_REPO)
                    local_offline = (os.environ.get("HF_HUB_OFFLINE") == "1")
                    checkpoint_path = hf_hub_download(
                        repo_id=MODEL_REPO,
                        filename=MODEL_FILENAME,
                        cache_dir=cache_dir,
                        local_files_only=local_offline,
                    )
                except Exception as exc:
                    logger.warning("Could not download model from %s: %s", MODEL_REPO, exc)
                    self._load_failed = True
                    return None

            try:
                net = smp.UnetPlusPlus(
                    encoder_name=ENCODER,
                    encoder_weights=None,
                    in_channels=3,
                    classes=1,
                    activation=None,
                    decoder_attention_type="scse",
                )
                convert_batchnorm_to_groupnorm(net.decoder)

                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                except TypeError:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)

                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]
                if isinstance(checkpoint, dict):
                    checkpoint = {
                        key.removeprefix("module."): value for key, value in checkpoint.items()
                    }

                net.load_state_dict(checkpoint)
                net.to(self.device)
                net.eval()
                self._model = net
                logger.info("MangaTextCleaner model loaded successfully on %s.", self.device)
                return self._model
            except Exception as exc:
                logger.error("Failed to initialize MangaTextCleaner model: %s", exc)
                self._load_failed = True
                return None

    def _infer_tile(self, net, tile_rgb: np.ndarray, tta_hflip: bool = False, tta_vflip: bool = False) -> np.ndarray:
        th, tw = tile_rgb.shape[:2]
        pad_h = (32 - th % 32) % 32
        pad_w = (32 - tw % 32) % 32

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = ((tile_rgb.astype(np.float32) / 255.0) - mean) / std
        tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)

        steps = 1 + int(tta_hflip) + int(tta_vflip)
        is_cuda = (self.device == "cuda" or (isinstance(self.device, torch.device) and self.device.type == "cuda")) and torch.cuda.is_available()
        with torch.inference_mode():
            if is_cuda:
                with torch.amp.autocast("cuda"):
                    probs = net(tensor).sigmoid()
            else:
                probs = net(tensor).sigmoid()

            accumulated_probs = probs
            if tta_hflip:
                tensor_flip = torch.flip(tensor, [3])
                if is_cuda:
                    with torch.amp.autocast("cuda"):
                        probs_flip = net(tensor_flip).sigmoid()
                else:
                    probs_flip = net(tensor_flip).sigmoid()
                accumulated_probs = accumulated_probs + torch.flip(probs_flip, [3])

            if tta_vflip:
                tensor_flip = torch.flip(tensor, [2])
                if is_cuda:
                    with torch.amp.autocast("cuda"):
                        probs_flip = net(tensor_flip).sigmoid()
                else:
                    probs_flip = net(tensor_flip).sigmoid()
                accumulated_probs = accumulated_probs + torch.flip(probs_flip, [2])

        return (accumulated_probs / steps)[0, 0, :th, :tw].cpu().numpy()

    def predict_mask(self, image: np.ndarray, tta_hflip=False, tta_vflip=False, allowed_boxes: Optional[List[list]] = None) -> Optional[np.ndarray]:
        """
        Runs neural text segmentation on an RGB image.
        Returns float32 probability map of shape (H, W) with values [0.0, 1.0],
        or None if neural segmentation is unavailable without allowed_boxes.
        """
        if image is None or image.size == 0:
            return None

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        height, width = image.shape[:2]

        net = self.load_model()
        if net is None:
            # Restrict heuristic fallback strictly to allowed_boxes so full-page artwork is never damaged;
            # signal failure (None) when allowed_boxes is absent so callers invoke clean_bubble_crop.
            if not allowed_boxes:
                return None
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
                blurred = cv2.GaussianBlur(gray, (25, 25), 0)
                diff = cv2.absdiff(gray, blurred)
                stroke_mask = (diff > 18).astype(np.float32)
                dark_ink = ((gray < 160) & (blurred > 180)).astype(np.float32)
                bright_ink = ((gray > 120) & (blurred < 90)).astype(np.float32)
                heur = np.maximum(stroke_mask, np.maximum(dark_ink, bright_ink))
                box_mask = np.zeros((height, width), dtype=np.float32)
                for box_item in allowed_boxes:
                    b = box_item.get("box", box_item) if isinstance(box_item, dict) else box_item
                    if hasattr(b, "__len__") and len(b) >= 4:
                        bx1 = max(0, int(round(b[0])))
                        by1 = max(0, int(round(b[1])))
                        bx2 = min(width, int(round(b[2])))
                        by2 = min(height, int(round(b[3])))
                        if bx2 > bx1 and by2 > by1:
                            box_mask[by1:by2, bx1:bx2] = 1.0
                return heur * box_mask
            except Exception:
                return None

        tile_size = 1024
        overlap = 128
        if height <= tile_size and width <= tile_size:
            return self._infer_tile(net, image, tta_hflip=tta_hflip, tta_vflip=tta_vflip)

        stride = tile_size - overlap
        prob_accum = np.zeros((height, width), dtype=np.float32)
        weight_accum = np.zeros((height, width), dtype=np.float32)

        y_starts = list(range(0, max(1, height - tile_size + 1), stride))
        if y_starts[-1] + tile_size < height:
            y_starts.append(max(0, height - tile_size))
        x_starts = list(range(0, max(1, width - tile_size + 1), stride))
        if x_starts[-1] + tile_size < width:
            x_starts.append(max(0, width - tile_size))

        for y0 in y_starts:
            y1 = min(height, y0 + tile_size)
            for x0 in x_starts:
                x1 = min(width, x0 + tile_size)
                tile_prob = self._infer_tile(net, image[y0:y1, x0:x1], tta_hflip=tta_hflip, tta_vflip=tta_vflip)
                th, tw = tile_prob.shape
                wy = np.ones(th, dtype=np.float32)
                wx = np.ones(tw, dtype=np.float32)
                ramp_y = min(overlap // 2, th // 4)
                ramp_x = min(overlap // 2, tw // 4)
                if y0 > 0 and ramp_y > 0:
                    wy[:ramp_y] = np.linspace(0.25, 1.0, ramp_y, dtype=np.float32)
                if y1 < height and ramp_y > 0:
                    wy[-ramp_y:] = np.linspace(1.0, 0.25, ramp_y, dtype=np.float32)
                if x0 > 0 and ramp_x > 0:
                    wx[:ramp_x] = np.linspace(0.25, 1.0, ramp_x, dtype=np.float32)
                if x1 < width and ramp_x > 0:
                    wx[-ramp_x:] = np.linspace(1.0, 0.25, ramp_x, dtype=np.float32)
                w2d = wy[:, None] * wx[None, :]
                prob_accum[y0:y1, x0:x1] += tile_prob * w2d
                weight_accum[y0:y1, x0:x1] += w2d

        return np.divide(prob_accum, np.maximum(weight_accum, 1e-6), dtype=np.float32)

    def clean_page(
        self,
        image_input,
        threshold: float = 0.5,
        padding_iter: int = 2,
        fill_holes: bool = False,
        close_gaps_kernel: int = 3,
        fill_color: str = "White",
        bubble_adaptive: bool = True,
        extra_mask: np.ndarray = None,
        allowed_boxes: Optional[List[list]] = None,
    ):
        """
        Generates a pristine, text-free manga page matching petersunde/Manga-Text-Cleaner.

        Parameters:
            image_input: PIL Image or np.ndarray (RGB)
            threshold: Probability threshold for text segmentation (default 0.5)
            padding_iter: Dilation iterations to swallow anti-aliasing silhouettes (default 2)
            fill_holes: Whether to fill interior holes in segmented text (default False)
            close_gaps_kernel: Kernel size to bridge character gaps (default 0)
            fill_color: Default fill ("White" or "Black")
            bubble_adaptive: If True, uses clean white fill in speech bubbles and Telea inpainting on dark artwork.
            extra_mask: Optional extra text-stroke mask (e.g. from comic-text-detector) to combine.
            allowed_boxes: Optional list of bounding boxes to restrict cleaning strictly within.

        Returns:
            cleaned_image (PIL.Image): Pristine cleaned manga page.
            edit_mask (np.ndarray): Binary uint8 mask (255 for erased text pixels).
            prob_map (np.ndarray): Float32 probability map.
        """
        if isinstance(image_input, Image.Image):
            image_rgb = np.array(image_input.convert("RGB"))
        else:
            image_rgb = np.array(image_input)
            if image_rgb.ndim == 2:
                image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
            elif image_rgb.ndim == 3 and image_rgb.shape[2] == 4:
                image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
            elif image_rgb.ndim == 3 and image_rgb.shape[2] == 1:
                image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)

        h, w = image_rgb.shape[:2]
        extra_arr = None
        if extra_mask is not None:
            extra_arr = np.asarray(extra_mask)
            if extra_arr.ndim == 3:
                extra_arr = extra_arr[:, :, 0] if extra_arr.shape[2] == 1 else np.max(extra_arr, axis=2)
            if extra_arr.size > 0 and extra_arr.shape[:2] != (h, w):
                interp = cv2.INTER_NEAREST if extra_arr.dtype in (np.uint8, bool) else cv2.INTER_LINEAR
                extra_arr = cv2.resize(extra_arr.astype(np.float32), (w, h), interpolation=interp)

        prob_map = self.predict_mask(image_rgb, allowed_boxes=allowed_boxes)
        if prob_map is None:
            if extra_arr is not None and np.any(extra_arr):
                max_val = float(np.max(extra_arr))
                prob_map = extra_arr.astype(np.float32) / (255.0 if max_val > 1.0 else 1.0)
            else:
                return Image.fromarray(image_rgb), None, None

        binary_mask = (prob_map > threshold).astype(np.uint8) * 255
        if extra_arr is not None and np.any(extra_arr):
            if extra_arr.dtype == np.uint8:
                extra_u8 = (extra_arr > 0).astype(np.uint8) * 255
            elif extra_arr.dtype == bool:
                extra_u8 = extra_arr.astype(np.uint8) * 255
            else:
                extra_u8 = (extra_arr > 0.35).astype(np.uint8) * 255
            binary_mask = cv2.bitwise_or(binary_mask, extra_u8)

        if close_gaps_kernel > 0:
            kernel_size = int(close_gaps_kernel)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # Build allowed_region_mask early so structural_borders can exclude interior letter stems
        allowed_region_mask = None
        if allowed_boxes is not None:
            allowed_region_mask = np.zeros((h, w), dtype=bool)
            for box_item in allowed_boxes:
                b = box_item.get("box", box_item) if isinstance(box_item, dict) else box_item
                if hasattr(b, "__len__") and len(b) >= 4:
                    bx1 = max(0, int(round(b[0])))
                    by1 = max(0, int(round(b[1])))
                    bx2 = min(w, int(round(b[2])))
                    by2 = min(h, int(round(b[3])))
                    if bx2 > bx1 and by2 > by1:
                        allowed_region_mask[by1:by2, bx1:bx2] = True

        # Compute structural borders early for boundary bleed protection and restoration
        gray_m = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        dark_orig = (gray_m < 75).astype(np.uint8) * 255
        kw = max(24, min(int(round(w * 0.04)), 80))
        kh = max(24, min(int(round(h * 0.04)), 80))
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
        k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
        h_borders = cv2.morphologyEx(dark_orig, cv2.MORPH_OPEN, k_h)
        v_borders = cv2.morphologyEx(dark_orig, cv2.MORPH_OPEN, k_v)
        structural_borders = cv2.bitwise_or(h_borders, v_borders)
        k_thick = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 16))
        thick_black = cv2.morphologyEx(dark_orig, cv2.MORPH_OPEN, k_thick)
        structural_borders = cv2.bitwise_and(structural_borders, cv2.bitwise_not(thick_black))
        structural_border_mask = np.zeros((h, w), dtype=bool)
        if np.any(structural_borders > 0):
            sb_num, sb_labels, sb_stats, _ = cv2.connectedComponentsWithStats((structural_borders > 0).astype(np.uint8), connectivity=8)
            for sb_i in range(1, sb_num):
                sx, sy, sw_b, sh_b, _ = sb_stats[sb_i]
                comp_slice = (sb_labels[sy:sy + sh_b, sx:sx + sw_b] == sb_i)
                if allowed_region_mask is not None:
                    # Restrict structural_borders strictly to borders crossing outside allowed_boxes
                    # so >=24px interior letter stems (T, I, L, —, 一) are never restored uncleaned.
                    if np.any(comp_slice & (~allowed_region_mask[sy:sy + sh_b, sx:sx + sw_b])):
                        structural_border_mask[sy:sy + sh_b, sx:sx + sw_b] |= comp_slice
                else:
                    # Without allowed_boxes, only preserve borders extending outside detected text strokes
                    # or spanning long panel border lengths
                    if np.any(comp_slice & (binary_mask[sy:sy + sh_b, sx:sx + sw_b] == 0)) or max(sw_b, sh_b) >= max(48, int(round(max(w, h) * 0.15))):
                        structural_border_mask[sy:sy + sh_b, sx:sx + sw_b] |= comp_slice

        if fill_holes:
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                c_area = cv2.contourArea(cnt)
                if c_area < 0.20 * (w * h):
                    cv2.drawContours(binary_mask, [cnt], -1, 255, -1)
            # Guard hole filling: do not bleed across structural borders
            binary_mask[structural_border_mask] = 0

        # Dilate mask to absorb all anti-aliasing edges, borders, and halos
        if padding_iter > 0:
            kernel_dil = np.ones((3, 3), np.uint8)
            edit_mask = cv2.dilate(binary_mask, kernel_dil, iterations=int(padding_iter))
            # Guard dilation from bleeding across straight panel/box borders into adjacent artwork
            edit_mask[structural_border_mask] = 0
        else:
            edit_mask = binary_mask.copy()
            edit_mask[structural_border_mask] = 0

        # If allowed_boxes is specified, restrict edit_mask strictly to allowed boxes
        # to ensure 0% damage to character faces and artwork outside.
        if allowed_region_mask is not None:
            edit_mask = np.where(allowed_region_mask, edit_mask, 0).astype(np.uint8)
            binary_mask = np.where(allowed_region_mask, binary_mask, 0).astype(np.uint8)

        cleaned = image_rgb.copy()

        if bubble_adaptive and np.any(edit_mask > 0):
            # Process each text component: preserve actual local bubble color (no forced white),
            # fill dark narration boxes with local dark color, and route textured/gradient regions to SimpleLama/Telea.
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edit_mask, connectivity=8)

            screentone_mask = np.zeros((h, w), dtype=np.uint8)
            default_fill = 255 if fill_color == "White" else 0
            halo_excl_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            outer_ring_k = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))

            for lbl in range(1, num_labels):
                bx, by, bw, bh, area = stats[lbl]

                # Sample surrounding background context (padded to allow halo erosion)
                cx1, cy1 = max(0, bx - 12), max(0, by - 12)
                cx2, cy2 = min(w, bx + bw + 12), min(h, by + bh + 12)
                local_comp = (labels[cy1:cy2, cx1:cx2] == lbl)
                local_gray = gray[cy1:cy2, cx1:cx2]
                local_rgb = image_rgb[cy1:cy2, cx1:cx2]
                local_edit = edit_mask[cy1:cy2, cx1:cx2]
                local_sb = structural_border_mask[cy1:cy2, cx1:cx2]

                # True background pixels: erode text halos out of bg_mask before sampling RGB
                comp_u8 = local_comp.astype(np.uint8)
                comp_edit_u8 = (local_comp | (local_edit > 0)).astype(np.uint8)
                inner_excl = cv2.dilate(comp_edit_u8, halo_excl_k, iterations=1) > 0
                outer_ring = cv2.dilate(comp_u8, outer_ring_k, iterations=1) > 0
                bg_mask = outer_ring & (~inner_excl) & (~local_sb)
                if np.sum(bg_mask) < 8:
                    bg_mask = outer_ring & (~local_comp) & (local_edit == 0) & (~local_sb)
                if not np.any(bg_mask):
                    bg_mask = (~local_comp) & (local_edit == 0)
                if not np.any(bg_mask):
                    bg_mask = (~local_comp)

                bg_pixels = local_gray[bg_mask]
                if bg_pixels.size > 0:
                    med_bg = float(np.median(bg_pixels))
                    std_bg = float(np.std(bg_pixels))
                    p25_bg = float(np.percentile(bg_pixels, 25))
                    p85_bg = float(np.percentile(bg_pixels, 85))
                else:
                    med_bg = 255.0
                    std_bg = 0.0
                    p25_bg = 255.0
                    p85_bg = 255.0

                # Sample dark/non-text pixels excluding bright text (local_gray > 140)
                p60_bg = float(np.percentile(bg_pixels, 60)) if bg_pixels.size > 0 else 255.0
                dark_bg_pixels = bg_pixels[bg_pixels < 140] if bg_pixels.size > 0 else np.array([])
                is_white_bg = (med_bg >= 170) or (p85_bg >= 210 and med_bg >= 140)
                is_dark_box = (
                    not is_white_bg and (
                        (bg_pixels.size >= 8 and (
                            (med_bg <= 95)
                            or (p60_bg <= 105 and med_bg <= 115)
                            or (p25_bg < 65 and float(np.mean(bg_pixels < 110)) >= 0.40)
                            or (dark_bg_pixels.size >= 10 and float(np.median(dark_bg_pixels)) <= 80 and med_bg < 125)
                        ))
                    )
                )

                if is_dark_box:
                    dark_bg_luma = float(np.median(dark_bg_pixels)) if dark_bg_pixels.size > 0 else 30.0
                    bright_thresh = max(110, int(dark_bg_luma + 25))
                    comp_dil = cv2.dilate(local_comp.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))) > 0
                    bright_strokes = ((local_gray > bright_thresh) & comp_dil).astype(np.uint8) * 255

                    # If this component is bright text or has bright text strokes inside its patch:
                    # Dilate bright strokes to swallow any black outline halo around the white text
                    if np.any(bright_strokes > 0) and np.any((bright_strokes > 0) & local_comp):
                        halo_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                        dilated_halo = cv2.dilate(bright_strokes, halo_kernel, iterations=1)
                        erase_comp = (local_comp | ((dilated_halo > 0) & comp_dil))
                        erase_comp = erase_comp & (~local_sb)
                    else:
                        erase_comp = local_comp

                    if allowed_region_mask is not None:
                        erase_comp = erase_comp & allowed_region_mask[cy1:cy2, cx1:cx2]

                    # Dark narration box: fill using local dark RGB median (never SimpleLama/Telea which smears white)
                    dark_rgb_mask = bg_mask & (local_gray < 110)
                    if np.any(dark_rgb_mask):
                        dark_fill = np.median(local_rgb[dark_rgb_mask], axis=0).astype(np.uint8)
                    elif dark_bg_pixels.size > 0:
                        d_val = int(np.median(dark_bg_pixels))
                        dark_fill = np.array([d_val, d_val, d_val], dtype=np.uint8)
                    else:
                        dark_fill = np.array([25, 25, 25], dtype=np.uint8)

                    # Smooth row-wise gradient fill if vertical gradient is present (never use SimpleLama on dark narration)
                    dark_gray_vals = local_gray[dark_rgb_mask] if np.any(dark_rgb_mask) else dark_bg_pixels
                    if dark_gray_vals.size > 0 and float(np.std(dark_gray_vals)) >= 5.0:
                        lh_box, lw_box = erase_comp.shape[:2]
                        for r_y in range(lh_box):
                            row_c = erase_comp[r_y]
                            if not np.any(row_c):
                                continue
                            y_min = max(0, r_y - 2)
                            y_max = min(lh_box, r_y + 3)
                            b_mask = dark_rgb_mask[y_min:y_max]
                            b_pix = local_rgb[y_min:y_max][b_mask]
                            if b_pix.shape[0] >= 5:
                                cleaned[cy1 + r_y, cx1:cx2][row_c] = np.median(b_pix.reshape(-1, 3), axis=0).astype(np.uint8)
                            else:
                                cleaned[cy1 + r_y, cx1:cx2][row_c] = dark_fill
                    else:
                        cleaned[cy1:cy2, cx1:cx2][erase_comp] = dark_fill
                elif std_bg < 18.0:
                    # Uniform light, cream, pastel, or colored bubble:
                    # Fill with actual local median RGB; only snap to pure white when near-pure white.
                    if bg_pixels.size > 0:
                        fill_rgb = np.round(np.median(local_rgb[bg_mask], axis=0)).astype(np.uint8)
                        if fill_color == "White" and std_bg < 5.0 and np.all(fill_rgb >= 248):
                            fill_rgb = np.array([255, 255, 255], dtype=np.uint8)
                        cleaned[cy1:cy2, cx1:cx2][local_comp] = fill_rgb
                    else:
                        cleaned[cy1:cy2, cx1:cx2][local_comp] = [default_fill, default_fill, default_fill]
                else:
                    # Textured, gradient, halftone, or screentone background (std_bg >= 18): route to SimpleLama/Telea
                    screentone_mask[cy1:cy2, cx1:cx2][local_comp] = 255

            if allowed_region_mask is not None:
                screentone_mask = np.where(allowed_region_mask, screentone_mask, 0).astype(np.uint8)

            if np.any(screentone_mask > 0):
                cleaned_lama = None
                try:
                    lama = get_shared_simple_lama(device=self.device)
                    if lama is not None:
                        cleaned_lama = inpaint_mask_with_simple_lama(cleaned, screentone_mask, lama)
                except Exception as lama_err:
                    logger.warning("SimpleLama inpainting fallback to Telea: %s", lama_err)

                if cleaned_lama is not None and cleaned_lama.shape[:2] == (h, w):
                    cleaned[screentone_mask == 255] = cleaned_lama[screentone_mask == 255]
                else:
                    cleaned_cv = cv2.inpaint(cleaned, screentone_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                    cleaned[screentone_mask == 255] = cleaned_cv[screentone_mask == 255]
        else:
            fill_value = 0 if fill_color == "Black" else 255
            cleaned[edit_mask == 255] = [fill_value] * image_rgb.shape[2]

        # 100% Guaranteed Structural Line & Panel Border Protection:
        # Restore structural panel/box borders strictly excluding text strokes
        try:
            border_mask_to_restore = np.asarray(structural_border_mask, dtype=bool) & (binary_mask == 0)
            cleaned[border_mask_to_restore] = image_rgb[border_mask_to_restore]
        except Exception as exc:
            logger.warning("Structural border restoration skipped: %s", exc)

        # 100% Guaranteed 0% damage: strictly enforce unchanged pixels outside allowed_boxes
        if allowed_region_mask is not None:
            cleaned[~allowed_region_mask] = image_rgb[~allowed_region_mask]

        cleaned_pil = Image.fromarray(cleaned)
        return cleaned_pil, edit_mask, prob_map



try:
    from cleaning import clean_bubble_crop
except (ImportError, ValueError):
    try:
        from .cleaning import clean_bubble_crop
    except (ImportError, ValueError):
        from ml.cleaning import clean_bubble_crop

