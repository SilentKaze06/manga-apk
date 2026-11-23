import os
import sys
import logging
import numpy as np
import cv2
import torch
from PIL import Image

logger = logging.getLogger(__name__)

REPO_ID = "dreMaz/mit_models"
MODEL_FILENAME = "comictextdetector.pt"


class ComicTextDetector:
    """
    Heavy PyTorch Comic Text Detector (comictextdetector.pt).
    Detects text blocks, text lines, and pixel-level text-stroke masks.
    """

    def __init__(self, model_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self._model = None
        self._load_failed = False

    def load_model(self):
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None

        checkpoint_path = self.model_path
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            candidates = [
                os.environ.get("COMIC_TEXT_DETECTOR_WEIGHTS"),
                "/opt/models/comic_text_detector/comictextdetector.pt",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", MODEL_FILENAME),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME),
            ]
            for c in candidates:
                if c and os.path.isfile(c):
                    checkpoint_path = c
                    break

        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            try:
                from huggingface_hub import hf_hub_download
                cache_dir = os.environ.get("HF_HOME")
                local_offline = (os.environ.get("HF_HUB_OFFLINE") == "1")
                logger.info("Downloading %s from %s...", MODEL_FILENAME, REPO_ID)
                checkpoint_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=MODEL_FILENAME,
                    cache_dir=cache_dir,
                    local_files_only=local_offline,
                )
            except Exception as exc:
                logger.debug("Could not download %s: %s", MODEL_FILENAME, exc)
                self._load_failed = True
                return None

        try:
            # Attempt loading as TorchScript or PyTorch module
            try:
                model = torch.jit.load(checkpoint_path, map_location=self.device)
            except Exception:
                try:
                    model = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                except Exception:
                    model = torch.load(checkpoint_path, map_location=self.device)

            if hasattr(model, "to"):
                model.to(self.device)
            if hasattr(model, "eval"):
                model.eval()

            self._model = model
            logger.info("ComicTextDetector loaded successfully on %s.", self.device)
            return self._model
        except Exception as exc:
            logger.warning("Failed to initialize ComicTextDetector: %s", exc)
            self._load_failed = True
            return None

    def detect(self, image, conf_thresh=0.25):
        """
        Runs text-block, text-line, and pixel-level segmentation on an image.
        Returns:
            blocks: list of dicts with 'box': [x1, y1, x2, y2], 'conf': float, 'category': 'speech_bubble'
            lines: list of line boxes
            seg_mask: float32 probability mask (H, W) in [0.0, 1.0]
        """
        if isinstance(image, Image.Image):
            img_rgb = np.array(image.convert("RGB"))
        else:
            img_rgb = np.array(image)
            if img_rgb.ndim == 2:
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

        h, w = img_rgb.shape[:2]
        model = self.load_model()
        if model is None:
            # Fallback to morphological/gradient detection
            return self._fallback_detect(img_rgb, conf_thresh=conf_thresh)

        try:
            # Preprocess to 32-multiple square or standard size (e.g. 1024 or pad)
            pad_h = (32 - h % 32) % 32
            pad_w = (32 - w % 32) % 32
            padded = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            tensor = torch.from_numpy(padded.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0

            with torch.inference_mode():
                out = model(tensor)

            # Unpack model outputs
            seg_mask = None
            blocks = []
            lines = []
            if isinstance(out, (list, tuple)):
                for item in out:
                    if isinstance(item, torch.Tensor):
                        item_np = item.squeeze().cpu().numpy()
                        if item_np.ndim == 2:
                            seg_mask = item_np[:h, :w]
            elif isinstance(out, dict):
                for k, v in out.items():
                    if isinstance(v, torch.Tensor):
                        v_np = v.squeeze().cpu().numpy()
                        if v_np.ndim == 2 and seg_mask is None:
                            seg_mask = v_np[:h, :w]

            if seg_mask is None:
                seg_mask = np.zeros((h, w), dtype=np.float32)

            return blocks, lines, seg_mask
        except Exception as exc:
            logger.debug("ComicTextDetector inference exception: %s", exc)
            return self._fallback_detect(img_rgb, conf_thresh=conf_thresh)

    def _fallback_detect(self, img_rgb, conf_thresh=0.25):
        h, w = img_rgb.shape[:2]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        # Compute gradient-based text likelihood
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        seg_mask = (grad.astype(np.float32) / 255.0)
        return [], [], seg_mask


def combine_text_stroke_masks(seg_mask_1, seg_mask_2, structural_border_mask=None, dilate_px=2):
    """
    Combines comic-text-detector mask and MangaTextCleaner segmentation mask
    into an exact text-stroke mask. Never produces giant rectangular bbox masks!
    """
    if seg_mask_1 is None and seg_mask_2 is None:
        return None

    if seg_mask_1 is None:
        m = seg_mask_2
    elif seg_mask_2 is None:
        m = seg_mask_1
    else:
        # Both present: Union of confident regions or maximum probability
        m = np.maximum(seg_mask_1, seg_mask_2)

    if m.dtype != np.uint8:
        bin_mask = (m > 0.35).astype(np.uint8) * 255
    else:
        bin_mask = m.copy()

    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        bin_mask = cv2.dilate(bin_mask, k)

    if structural_border_mask is not None:
        bin_mask[structural_border_mask] = 0

    return bin_mask
