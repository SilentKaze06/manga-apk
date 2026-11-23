import os
import sys
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

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
        padding_iter: int = 4,
        fill_holes: bool = False,
        close_gaps_kernel: int = 3,
        fill_color: str = "White",
        bubble_adaptive: bool = True,
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
            return Image.fromarray(image_rgb), None, None

        binary_mask = (prob_map > threshold).astype(np.uint8) * 255

        if close_gaps_kernel > 0:
            kernel_size = int(close_gaps_kernel)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # Compute structural borders early for boundary bleed protection and restoration
        gray_m = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        dark_orig = (gray_m < 75).astype(np.uint8) * 255
        kw = max(36, min(int(round(w * 0.04)), 80))
        kh = max(36, min(int(round(h * 0.04)), 80))
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
        k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
        h_borders = cv2.morphologyEx(dark_orig, cv2.MORPH_OPEN, k_h)
        v_borders = cv2.morphologyEx(dark_orig, cv2.MORPH_OPEN, k_v)
        structural_borders = cv2.bitwise_or(h_borders, v_borders)
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

        cleaned = image_rgb.copy()

        if bubble_adaptive and np.any(edit_mask > 0):
            # Process each text component: if inside a light speech bubble, fill with pure white (255)
            # or bubble color. If inside a dark narration box, fill with local dark background color.
            # Only use Telea inpainting on complex screentone/halftone/artwork.
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edit_mask, connectivity=8)

            dark_art_mask = np.zeros((h, w), dtype=np.uint8)
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

                # True background pixels: not in this component and not in other text edits
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
                dark_bg_pixels = bg_pixels[bg_pixels < 140] if bg_pixels.size > 0 else np.array([])
                is_dark_box = (
                    (dark_bg_pixels.size >= 6 and (float(np.median(dark_bg_pixels)) <= 95 or p25_bg < 65))
                    or (med_bg <= 95 and float(np.mean(bg_pixels < 110)) >= 0.40)
                    or (p25_bg < 65 and med_bg < 110)
                )

                if is_dark_box:
                    # Dark narration box: fill using local dark RGB median (never Telea inpaint which smears white)
                    dark_rgb_mask = bg_mask & (local_gray < 110)
                    if np.any(dark_rgb_mask):
                        dark_fill = np.median(local_rgb[dark_rgb_mask], axis=0).astype(np.uint8)
                    elif dark_bg_pixels.size > 0:
                        d_val = int(np.median(dark_bg_pixels))
                        dark_fill = np.array([d_val, d_val, d_val], dtype=np.uint8)
                    else:
                        dark_fill = np.array([25, 25, 25], dtype=np.uint8)
                    cleaned[comp] = dark_fill
                elif med_bg >= 220 and (std_bg < 20 or p85_bg >= 240):
                    # Speech bubble / pure white background: uniform pure white fill eliminates 100% of ghosting
                    cleaned[comp] = [default_fill, default_fill, default_fill]
                else:
                    # Screentone, halftone, spiky bubble, or textured background: mark for texture-preserving Telea inpainting
                    dark_art_mask[comp] = 255

            if np.any(dark_art_mask > 0):
                cleaned_cv = cv2.inpaint(cleaned, dark_art_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                cleaned[dark_art_mask == 255] = cleaned_cv[dark_art_mask == 255]
        else:
            fill_value = 0 if fill_color == "Black" else 255
            cleaned[edit_mask == 255] = [fill_value] * image_rgb.shape[2]

        # 100% Guaranteed Structural Line & Panel Border Protection:
        # Restore structural panel/box borders from image_rgb wherever detected
        try:
            cleaned[structural_border_mask] = image_rgb[structural_border_mask]
        except Exception as exc:
            logger.debug("Structural border restoration skipped: %s", exc)

        cleaned_pil = Image.fromarray(cleaned)
        return cleaned_pil, edit_mask, prob_map
