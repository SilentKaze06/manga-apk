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

    def load_model(self):
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

    def predict_mask(self, image: np.ndarray, tta_hflip=False, tta_vflip=False) -> np.ndarray:
        """
        Runs neural text segmentation on an RGB image.
        Returns float32 probability map of shape (H, W) with values [0.0, 1.0].
        """
        net = self.load_model()
        if net is None:
            # Graceful fallback when neural segmentation model is not available
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
                blurred = cv2.GaussianBlur(gray, (25, 25), 0)
                diff = cv2.absdiff(gray, blurred)
                stroke_mask = (diff > 18).astype(np.float32)
                dark_ink = ((gray < 160) & (blurred > 180)).astype(np.float32)
                bright_ink = ((gray > 120) & (blurred < 90)).astype(np.float32)
                return np.maximum(stroke_mask, np.maximum(dark_ink, bright_ink))
            except Exception:
                return None

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        height, width = image.shape[:2]

        pad_h = (32 - height % 32) % 32
        pad_w = (32 - width % 32) % 32

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = ((image.astype(np.float32) / 255.0) - mean) / std
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
                accumulated_probs += torch.flip(probs_flip, [3])

            if tta_vflip:
                tensor_flip = torch.flip(tensor, [2])
                if is_cuda:
                    with torch.amp.autocast("cuda"):
                        probs_flip = net(tensor_flip).sigmoid()
                else:
                    probs_flip = net(tensor_flip).sigmoid()
                accumulated_probs += torch.flip(probs_flip, [2])

        prob_map = (accumulated_probs / steps)[0, 0, :height, :width].cpu().numpy()
        return prob_map

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
        prob_map = self.predict_mask(image_rgb)
        if prob_map is None:
            if extra_mask is not None and np.any(extra_mask):
                prob_map = extra_mask.astype(np.float32) / (255.0 if extra_mask.max() > 1 else 1.0)
            else:
                return Image.fromarray(image_rgb), None, None

        binary_mask = (prob_map > threshold).astype(np.uint8) * 255
        if extra_mask is not None and np.any(extra_mask):
            if extra_mask.shape[:2] != (h, w):
                extra_mask = cv2.resize(extra_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            extra_u8 = (extra_mask > 0.35).astype(np.uint8) * 255 if extra_mask.dtype != np.uint8 else (extra_mask > 0).astype(np.uint8) * 255
            binary_mask = cv2.bitwise_or(binary_mask, extra_u8)

        if close_gaps_kernel > 0:
            kernel_size = int(close_gaps_kernel)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

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
        structural_border_mask = (structural_borders > 0)

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
            edit_mask = np.where(allowed_region_mask, edit_mask, 0).astype(np.uint8)
            binary_mask = np.where(allowed_region_mask, binary_mask, 0).astype(np.uint8)

        cleaned = image_rgb.copy()

        if bubble_adaptive and np.any(edit_mask > 0):
            # Process each text component: if inside a light speech bubble, fill with pure white (255)
            # or bubble color. If inside a dark narration box, fill with local dark background color.
            # Only use SimpleLama (PyTorch) inpainting on complex screentone/halftone/artwork.
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edit_mask, connectivity=8)

            screentone_mask = np.zeros((h, w), dtype=np.uint8)
            dark_lama_mask = np.zeros((h, w), dtype=np.uint8)
            default_fill = 255 if fill_color == "White" else 0

            for lbl in range(1, num_labels):
                comp = (labels == lbl)
                bx, by, bw, bh, area = stats[lbl]
                if area < 4:
                    continue

                # Sample surrounding background context
                cx1, cy1 = max(0, bx - 8), max(0, by - 8)
                cx2, cy2 = min(w, bx + bw + 8), min(h, by + bh + 8)
                local_comp = comp[cy1:cy2, cx1:cx2]
                local_gray = gray[cy1:cy2, cx1:cx2]
                local_rgb = image_rgb[cy1:cy2, cx1:cx2]
                local_edit = edit_mask[cy1:cy2, cx1:cx2]

                # True background pixels: 3-5px outer ring around component
                comp_u8 = local_comp.astype(np.uint8)
                ring_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                ring_dil = cv2.dilate(comp_u8, ring_k, iterations=1)
                bg_mask = (ring_dil > 0) & (~local_comp) & (local_edit == 0)
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
                is_dark_box = (
                    bg_pixels.size >= 8 and (
                        (med_bg <= 95)
                        or (p60_bg <= 105 and med_bg <= 115)
                        or (p25_bg < 65 and float(np.mean(bg_pixels < 110)) >= 0.50)
                    )
                )

                if is_dark_box:
                    dark_bg_luma = float(np.median(dark_bg_pixels)) if dark_bg_pixels.size > 0 else 30.0
                    bright_thresh = max(125, int(dark_bg_luma + 35))
                    comp_dil = cv2.dilate(local_comp.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))) > 0
                    bright_strokes = ((local_gray > bright_thresh) & comp_dil).astype(np.uint8) * 255

                    # If this component is bright text or has bright text strokes inside its patch:
                    # Dilate bright strokes to swallow any black outline halo around the white text
                    if np.any(bright_strokes > 0) and np.any((bright_strokes > 0) & local_comp):
                        halo_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                        dilated_halo = cv2.dilate(bright_strokes, halo_kernel, iterations=1)
                        erase_comp = (local_comp | ((dilated_halo > 0) & comp_dil))
                        erase_comp = erase_comp & (~structural_border_mask[cy1:cy2, cx1:cx2])
                    else:
                        erase_comp = local_comp

                    if allowed_region_mask is not None:
                        erase_comp = erase_comp & allowed_region_mask[cy1:cy2, cx1:cx2]

                    # Dark narration box: fill using local dark RGB median (never Telea inpaint which smears white)
                    dark_rgb_mask = bg_mask & (local_gray < 110)
                    if np.any(dark_rgb_mask):
                        dark_fill = np.median(local_rgb[dark_rgb_mask], axis=0).astype(np.uint8)
                    elif dark_bg_pixels.size > 0:
                        d_val = int(np.median(dark_bg_pixels))
                        dark_fill = np.array([d_val, d_val, d_val], dtype=np.uint8)
                    else:
                        dark_fill = np.array([25, 25, 25], dtype=np.uint8)

                    # Smooth row-wise gradient fill if vertical gradient is present
                    dark_gray_vals = local_gray[dark_rgb_mask] if np.any(dark_rgb_mask) else dark_bg_pixels
                    if dark_gray_vals.size > 0 and float(np.std(dark_gray_vals)) >= 5.0:
                        dark_lama_mask[cy1:cy2, cx1:cx2][erase_comp] = 255
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
                elif std_bg < 22 or (med_bg >= 200 and p85_bg >= 220):
                    # Speech bubble / uniform light background: uniform fill
                    if med_bg >= 220 and (std_bg < 20 or p85_bg >= 240):
                        cleaned[comp] = [default_fill, default_fill, default_fill]
                    elif bg_pixels.size > 0:
                        fill_rgb = np.median(local_rgb[bg_mask], axis=0).astype(np.uint8)
                        cleaned[comp] = fill_rgb
                    else:
                        cleaned[comp] = [default_fill, default_fill, default_fill]
                else:
                    # Screentone, halftone, spiky bubble, or textured background: mark for SimpleLama inpainting
                    screentone_mask[comp] = 255

            total_lama_mask = cv2.bitwise_or(screentone_mask, dark_lama_mask)
            if allowed_region_mask is not None:
                total_lama_mask = np.where(allowed_region_mask, total_lama_mask, 0).astype(np.uint8)
                screentone_mask = np.where(allowed_region_mask, screentone_mask, 0).astype(np.uint8)
                dark_lama_mask = np.where(allowed_region_mask, dark_lama_mask, 0).astype(np.uint8)

            if np.any(total_lama_mask > 0):
                cleaned_lama = None
                try:
                    lama = get_shared_simple_lama(device=self.device)
                    if lama is not None:
                        inp_pil = lama(Image.fromarray(image_rgb), Image.fromarray(total_lama_mask))
                        cleaned_lama = np.array(inp_pil.convert("RGB"))
                        if cleaned_lama.shape[:2] != (h, w):
                            cleaned_lama = cv2.resize(cleaned_lama, (w, h), interpolation=cv2.INTER_LANCZOS4)
                except Exception as lama_err:
                    logger.warning("SimpleLama inpainting fallback to Telea: %s", lama_err)

                if cleaned_lama is not None and cleaned_lama.shape[:2] == (h, w):
                    if np.any(screentone_mask > 0):
                        cleaned[screentone_mask == 255] = cleaned_lama[screentone_mask == 255]
                    if np.any(dark_lama_mask > 0):
                        cleaned[dark_lama_mask == 255] = cleaned_lama[dark_lama_mask == 255]
                elif np.any(screentone_mask > 0):
                    cleaned_cv = cv2.inpaint(cleaned, screentone_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                    cleaned[screentone_mask == 255] = cleaned_cv[screentone_mask == 255]
        else:
            fill_value = 0 if fill_color == "Black" else 255
            cleaned[edit_mask == 255] = [fill_value] * image_rgb.shape[2]

        # 100% Guaranteed Structural Line & Panel Border Protection:
        # Restore structural panel/box borders strictly excluding text strokes
        try:
            border_mask_to_restore = structural_border_mask & (~binary_mask)
            cleaned[border_mask_to_restore] = image_rgb[border_mask_to_restore]
        except Exception as exc:
            logger.warning("Structural border restoration skipped: %s", exc)

        # 100% Guaranteed 0% damage: strictly enforce unchanged pixels outside allowed_boxes
        if allowed_region_mask is not None:
            cleaned[~allowed_region_mask] = image_rgb[~allowed_region_mask]

        cleaned_pil = Image.fromarray(cleaned)
        return cleaned_pil, edit_mask, prob_map


def clean_bubble_crop(
    patch_orig,
    patch_gray=None,
    patch_mask=None,
    neural_patch=None,
    inpainter_model=None,
    inpaint_transform=None,
    device=None,
):
    """
    Master Border-Protected Bubble Cleaning Algorithm:
    - White / cream bubble with border: preserves outer contour border; cleans interior text to local background color.
    - Dark narration box: preserves border contour; fills interior text with dark background color (never pure white).
      Prioritizes SimpleLama for dark textured/gradient backgrounds (std >= 5.0).
    - Screentone / artwork / borderless: uses neural cleaner or SimpleLama if available, else Telea.
    """
    if patch_orig is None or patch_orig.size == 0:
        return patch_orig

    hp, wp = patch_orig.shape[:2]
    if hp <= 6 or wp <= 6:
        return patch_orig.copy()

    if patch_gray is None:
        patch_gray = cv2.cvtColor(patch_orig, cv2.COLOR_RGB2GRAY) if patch_orig.ndim == 3 else patch_orig

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

    # 1. WHITE OR CREAM BUBBLES
    if is_white_bubble:
        non_ink_interior = (interior_gray >= 185)
        if np.any(non_ink_interior):
            local_bg_color = np.median(interior_orig[non_ink_interior], axis=0).astype(np.uint8)
        else:
            non_ink_full = (patch_gray >= 185)
            if np.any(non_ink_full):
                local_bg_color = np.median(patch_orig[non_ink_full], axis=0).astype(np.uint8)
            else:
                local_bg_color = np.array([255, 255, 255], dtype=np.uint8)

        dark_mask = (patch_gray < 185).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)

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

            if has_straight_lines and np.sum((labels == lbl) & (straight_lines > 0)) >= 0.25 * la:
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

        letter_mask = ((dark_mask > 0) & (protected_mask == 0)).astype(np.uint8)
        if patch_mask is not None and np.any(patch_mask):
            ext_mask = ((patch_mask > 0) & (protected_mask == 0)).astype(np.uint8)
            letter_mask = cv2.bitwise_or(letter_mask, ext_mask)

        dilated_letters = cv2.dilate(letter_mask, k_rect3, iterations=2)
        dilated_erase = cv2.bitwise_and(dilated_letters, cv2.bitwise_not(protected_mask))

        anti_aliased_mask = ((patch_gray >= 185) & (patch_gray <= 240)).astype(np.uint8)
        halo_zone = cv2.dilate(dilated_erase, k_rect3, iterations=1)
        halo_candidates = (halo_zone > 0) & (anti_aliased_mask > 0) & (protected_mask == 0)

        erase_mask = cv2.bitwise_and(
            cv2.bitwise_or(dilated_erase, halo_candidates.astype(np.uint8)),
            cv2.bitwise_not(protected_mask)
        )

        patch_clean = patch_orig.copy()
        if np.any(erase_mask):
            if neural_patch is not None and neural_patch.shape == patch_orig.shape:
                diff_n = np.abs(neural_patch.astype(np.float32) - patch_orig.astype(np.float32)).max(axis=-1)
                if np.any(diff_n[erase_mask > 0] > 5.0):
                    patch_clean[erase_mask > 0] = neural_patch[erase_mask > 0]
                    return patch_clean

            ring = cv2.dilate(erase_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2)
            ring_bg = (ring > 0) & (erase_mask == 0) & (protected_mask == 0) & (patch_gray >= 170)
            if np.sum(ring_bg) > 20:
                fill_color = np.median(patch_orig[ring_bg], axis=0).astype(np.uint8)
            else:
                fill_color = local_bg_color

            patch_clean[erase_mask > 0] = fill_color

        return patch_clean

    # 2. DARK NARRATION BOXES / DARK BUBBLES
    if is_dark_bubble and not is_white_bubble:
        interior_dark = (interior_gray < 110)
        if np.any(interior_dark):
            dark_bg_color = np.median(interior_orig[interior_dark], axis=0).astype(np.uint8)
        else:
            dark_bg_color = np.median(interior_orig, axis=(0, 1)).astype(np.uint8)
        if np.mean(dark_bg_color) > 100:
            dark_bg_color = np.array([25, 25, 25], dtype=np.uint8)

        dark_bg_luma = float(np.median(interior_gray[interior_dark])) if np.any(interior_dark) else float(np.median(interior_gray))
        light_thresh = max(125, int(dark_bg_luma + 35))
        light_mask = (patch_gray > light_thresh).astype(np.uint8)

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

        outer_margin_mask = np.isin(l_labels, list(outer_light_labels)).astype(np.uint8) * 255 if outer_light_labels else np.zeros_like(patch_gray)

        text_light_mask = ((light_mask > 0) & (outer_margin_mask == 0)).astype(np.uint8)
        if patch_mask is not None and np.any(patch_mask):
            text_light_mask = cv2.bitwise_or(text_light_mask, ((patch_mask > 0) & (outer_margin_mask == 0)).astype(np.uint8))

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

            ring = cv2.dilate(text_erase_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2)
            ring_bg = (ring > 0) & (text_erase_mask == 0) & (outer_margin_mask == 0) & (patch_gray < 110)
            if np.sum(ring_bg) > 20:
                dark_fill = np.median(patch_orig[ring_bg], axis=0).astype(np.uint8)
            else:
                dark_fill = dark_bg_color

            ring_gray_vals = patch_gray[ring_bg] if np.sum(ring_bg) > 20 else patch_gray[patch_gray < 110]
            if ring_gray_vals.size > 0 and float(np.std(ring_gray_vals)) >= 5.0:
                # Prioritize SimpleLama for dark gradient/textured background
                cleaned_lama = None
                try:
                    lama = get_shared_simple_lama(device=device)
                    if lama is not None:
                        inp_pil = lama(Image.fromarray(patch_orig), Image.fromarray(text_erase_mask))
                        c_lama = np.array(inp_pil.convert("RGB"))
                        if c_lama.shape[:2] != (hp, wp):
                            c_lama = cv2.resize(c_lama, (wp, hp), interpolation=cv2.INTER_LANCZOS4)
                        if c_lama.shape == patch_orig.shape:
                            cleaned_lama = c_lama
                except Exception as lama_err:
                    logger.debug("clean_bubble_crop dark gradient SimpleLama: %s", lama_err)

                if cleaned_lama is not None:
                    patch_clean[text_erase_mask > 0] = cleaned_lama[text_erase_mask > 0]
                else:
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

    # 3. ARTWORK / SCREENTONE / BORDERLESS
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
