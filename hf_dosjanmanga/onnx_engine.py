from __future__ import annotations
"""
Manga & Comic ONNX Engine.
Runs Speech Bubble Detection, Multi-Language Manga OCR, and Neural Inpainting
using pure ONNX Runtime and NumPy for 4x faster CPU speed and <150MB RAM footprint.
"""

import os
import json
import logging
from typing import List, Dict, Optional
import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger("manga-onnx")

class MangaOnnxEngine:
    def __init__(self, models_dir: str, vocab_path: str):
        self.models_dir = models_dir
        self.vocab_path = vocab_path
        
        self.det_sess: Optional[ort.InferenceSession] = None
        self.ocr_sess: Optional[ort.InferenceSession] = None
        self.inp_sess: Optional[ort.InferenceSession] = None
        
        self.idx_to_char: Dict[int, str] = {}
        self.char_to_idx: Dict[str, int] = {}
        
        self._load_vocab()
        self._load_models()

    def _load_vocab(self):
        if os.path.exists(self.vocab_path):
            try:
                with open(self.vocab_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.char_to_idx = data.get("char_to_idx", {})
                    self.idx_to_char = {int(k): v for k, v in data.get("idx_to_char", {}).items()}
                logger.info("Manga OCR lug'ati yuklandi: %d ta belgi", len(self.idx_to_char))
            except Exception as e:
                logger.warning("Lug'atni o'qishda xato: %s", e)

    def _load_models(self):
        if ort is None:
            logger.error("onnxruntime kutubxonasi o'rnatilmagan!")
            return

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(2, os.cpu_count() or 1)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 1. Detector
        det_path = os.path.join(self.models_dir, "manga_detector.onnx")
        if os.path.exists(det_path):
            try:
                self.det_sess = ort.InferenceSession(det_path, sess_options=opts, providers=["CPUExecutionProvider"])
                logger.info("manga_detector.onnx yuklandi.")
            except Exception as e:
                logger.error("Detector yuklashda xato: %s", e)

        # 2. OCR
        ocr_path = os.path.join(self.models_dir, "manga_ocr.onnx")
        if os.path.exists(ocr_path):
            try:
                self.ocr_sess = ort.InferenceSession(ocr_path, sess_options=opts, providers=["CPUExecutionProvider"])
                logger.info("manga_ocr.onnx yuklandi.")
            except Exception as e:
                logger.error("OCR yuklashda xato: %s", e)

        # 3. Inpainter
        inp_path = os.path.join(self.models_dir, "manga_inpaint.onnx")
        if os.path.exists(inp_path):
            try:
                self.inp_sess = ort.InferenceSession(inp_path, sess_options=opts, providers=["CPUExecutionProvider"])
                logger.info("manga_inpaint.onnx yuklandi.")
            except Exception as e:
                logger.error("Inpainter yuklashda xato: %s", e)

    @property
    def is_ready(self) -> bool:
        return self.det_sess is not None or self.ocr_sess is not None

    def detect_bubbles(self, img: Image.Image, conf_thresh: float = 0.40, iou_thresh: float = 0.35) -> List[Dict]:
        """
        Detects speech bubbles on a manga page.
        Returns list of dicts: [{"box": [x1, y1, x2, y2], "score": float, "class_id": int}]
        Box coordinates are in pixel space: [left, top, right, bottom]
        """
        if self.det_sess is None:
            return []

        if img.mode != "RGB":
            img = img.convert("RGB")

        orig_w, orig_h = img.size
        # Dynamic high-resolution grid (multiples of 16, max dim 768 for high accuracy and fast CPU inference)
        scale = min(1.0, 768.0 / max(orig_w, orig_h))
        tw = max(256, int(round(orig_w * scale / 16.0)) * 16)
        th = max(256, int(round(orig_h * scale / 16.0)) * 16)
        resized = img.resize((tw, th), Image.Resampling.BILINEAR)
        arr = np.array(resized).astype(np.float32) / 255.0
        # Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_arr = (arr - mean) / std
        input_tensor = np.transpose(norm_arr, (2, 0, 1))[np.newaxis, ...]

        try:
            preds_raw = self.det_sess.run(None, {"image_input": input_tensor})[0]
            Gh, Gw = preds_raw.shape[2], preds_raw.shape[3]
            preds = preds_raw[0, 0].reshape(-1, 9)

            grid_y, grid_x = np.meshgrid(np.arange(Gh), np.arange(Gw), indexing='ij')
            grid = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)

            obj_scores = 1.0 / (1.0 + np.exp(np.clip(-preds[:, 4], -50.0, 50.0)))
            clipped_cls = np.clip(preds[:, 5:] - np.max(preds[:, 5:], axis=-1, keepdims=True), -50.0, 50.0)
            cls_probs = np.exp(clipped_cls)
            cls_probs = cls_probs / np.sum(cls_probs, axis=-1, keepdims=True)
            cls_indices = np.argmax(cls_probs, axis=-1)
            cls_scores = np.max(cls_probs, axis=-1)

            scores = obj_scores * cls_scores
            mask = scores > conf_thresh
            if not np.any(mask):
                return []

            filtered_preds = preds[mask]
            filtered_scores = scores[mask]
            filtered_cls = cls_indices[mask]
            filtered_grid = grid[mask]

            pred_xy = (1.0 / (1.0 + np.exp(np.clip(-filtered_preds[:, :2], -50.0, 50.0))) + filtered_grid) / np.array([Gw, Gh], dtype=np.float32)
            pred_wh = 1.0 / (1.0 + np.exp(np.clip(-filtered_preds[:, 2:4], -50.0, 50.0)))

            x1 = np.clip(pred_xy[:, 0] - pred_wh[:, 0] / 2.0, 0.0, 1.0)
            y1 = np.clip(pred_xy[:, 1] - pred_wh[:, 1] / 2.0, 0.0, 1.0)
            x2 = np.clip(pred_xy[:, 0] + pred_wh[:, 0] / 2.0, 0.0, 1.0)
            y2 = np.clip(pred_xy[:, 1] + pred_wh[:, 1] / 2.0, 0.0, 1.0)

            boxes = np.stack([x1, y1, x2, y2], axis=-1)
            keep_indices = self._nms(boxes, filtered_scores, iou_thresh, classes=filtered_cls)

            results = []
            for idx in keep_indices:
                b = boxes[idx]
                px1 = int(round(b[0] * orig_w))
                py1 = int(round(b[1] * orig_h))
                px2 = int(round(b[2] * orig_w))
                py2 = int(round(b[3] * orig_h))
                results.append({
                    "box": [max(0, px1), max(0, py1), min(orig_w, px2), min(orig_h, py2)],
                    "score": float(filtered_scores[idx]),
                    "class_id": int(filtered_cls[idx])
                })
            return results
        except Exception as e:
            logger.error("Bubble detection xatosi: %s", e)
            return []

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, thresh: float, classes: Optional[np.ndarray] = None) -> List[int]:
        if boxes.size == 0:
            return []

        if classes is not None:
            max_coord = float(np.max(boxes)) if boxes.size > 0 else 1.0
            offsets = classes[:, None].astype(np.float32) * (max_coord + 1.0)
            boxes_eval = boxes + offsets
        else:
            boxes_eval = boxes

        x1 = boxes_eval[:, 0]
        y1 = boxes_eval[:, 1]
        x2 = boxes_eval[:, 2]
        y2 = boxes_eval[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    def ocr_bubble(self, crop: Image.Image) -> str:
        """
        Runs Manga OCR on a cropped speech bubble or text region.
        Returns recognized string.
        """
        if self.ocr_sess is None or not self.idx_to_char:
            return ""

        if crop.mode != "RGB":
            crop = crop.convert("RGB")

        orig_w, orig_h = crop.size
        if orig_w < 4 or orig_h < 4:
            return ""

        # Normalize height to 64, preserving aspect ratio (width is multiple of 4, between 64 and 512)
        aspect = orig_w / float(orig_h)
        target_w = max(64, min(512, int(round(64.0 * aspect / 4.0)) * 4))
        resized = crop.resize((target_w, 64), Image.Resampling.BILINEAR)

        arr = np.array(resized).astype(np.float32) / 255.0
        # Normalization: mean=0.5, std=0.5
        norm_arr = (arr - 0.5) / 0.5
        input_tensor = np.transpose(norm_arr, (2, 0, 1))[np.newaxis, ...]  # (1, 3, 64, target_w)

        try:
            log_probs = self.ocr_sess.run(None, {"line_crop_input": input_tensor})[0]
            # Output: (T, 1, vocab_size)
            raw_ids = np.argmax(log_probs[:, 0, :], axis=-1)

            # CTC Greedy decoding
            chars = []
            prev = 0
            for idx in raw_ids:
                if idx != 0 and idx != prev:
                    ch = self.idx_to_char.get(int(idx), "")
                    if ch:
                        chars.append(ch)
                prev = idx
            return "".join(chars).strip()
        except Exception as e:
            logger.error("Manga OCR xatosi: %s", e)
            return ""

    def inpaint_bubbles(self, img: Image.Image, boxes: List[List[int]]) -> Image.Image:
        """
        Cleans detected speech bubbles, removing old text and restoring screentone/background.
        boxes: List of [x1, y1, x2, y2]
        """
        if self.inp_sess is None or not boxes:
            return img

        if img.mode != "RGB":
            img = img.convert("RGB")

        orig_img = img.copy()
        cleaned_img = img.copy()
        w, h = orig_img.size

        for box in boxes:
            bx1, by1, bx2, by2 = box
            # Add safe padding
            pad = 8
            cx1 = max(0, bx1 - pad)
            cy1 = max(0, by1 - pad)
            cx2 = min(w, bx2 + pad)
            cy2 = min(h, by2 + pad)

            pw = cx2 - cx1
            ph = cy2 - cy1
            if pw < 8 or ph < 8:
                continue

            # Crop patch from immutable original image
            patch = orig_img.crop((cx1, cy1, cx2, cy2))
            # Text area mask within patch
            mask_arr = np.zeros((ph, pw), dtype=np.float32)
            # Text region relative to patch
            tx1 = max(0, bx1 - cx1)
            ty1 = max(0, by1 - cy1)
            tx2 = min(pw, bx2 - cx1)
            ty2 = min(ph, by2 - cy1)
            mask_arr[ty1:ty2, tx1:tx2] = 1.0

            # Check if this patch has a dark interior (avoid blending white masks onto dark boxes)
            patch_np = np.array(patch)
            if cv2 is not None:
                patch_gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
            else:
                patch_gray = np.mean(patch_np, axis=2).astype(np.uint8)

            my = max(1, int(ph * 0.12))
            mx = max(1, int(pw * 0.12))
            int_gray = patch_gray[my:ph - my, mx:pw - mx] if ph > 2 * my and pw > 2 * mx else patch_gray
            int_median = float(np.median(int_gray))
            int_p15 = float(np.percentile(int_gray, 15))
            int_p25 = float(np.percentile(int_gray, 25))
            int_dark = int_gray[int_gray < 95]
            has_dark_bg = (int_dark.size >= 8 and float(np.median(int_dark)) <= 90)

            is_dark_box = (
                (int_median < 95)
                or (int_p15 < 55 and int_median < 130)
                or (int_p25 < 65 and has_dark_bg)
            )

            dark_bg_luma = 25.0
            dark_bg_color = np.array([25, 25, 25], dtype=np.uint8)
            if is_dark_box:
                int_np = patch_np[my:ph - my, mx:pw - mx] if ph > 2 * my and pw > 2 * mx else patch_np
                dark_core = (int_gray < 95)
                if np.any(dark_core):
                    dark_bg_color = np.median(int_np[dark_core], axis=0).astype(np.uint8)
                else:
                    dark_bg_pixels = patch_np[patch_gray < 95]
                    if dark_bg_pixels.size > 0:
                        dark_bg_color = np.median(dark_bg_pixels, axis=0).astype(np.uint8)
                    else:
                        dark_bg_color = np.median(patch_np, axis=(0, 1)).astype(np.uint8)
                if np.mean(dark_bg_color) > 100:
                    dark_bg_color = np.array([25, 25, 25], dtype=np.uint8)
                dark_bg_luma = float(np.mean(dark_bg_color))

            # Real Gaussian blur / feathering to mask_arr so no harsh rectangular edges occur
            if cv2 is not None:
                mask_arr = cv2.GaussianBlur(mask_arr, (7, 7), 2.0)
            else:
                mask_pil = Image.fromarray((mask_arr * 255).astype(np.uint8)).filter(
                    ImageFilter.GaussianBlur(radius=2.0)
                )
                mask_arr = np.array(mask_pil).astype(np.float32) / 255.0

            # Round dimensions to multiples of 8 for U-Net
            inp_w = ((pw + 7) // 8) * 8
            inp_h = ((ph + 7) // 8) * 8
            patch_resized = patch.resize((inp_w, inp_h), Image.Resampling.BILINEAR)
            mask_resized = Image.fromarray((np.clip(mask_arr, 0.0, 1.0) * 255).astype(np.uint8)).resize((inp_w, inp_h), Image.Resampling.BILINEAR)

            img_t = np.transpose(np.array(patch_resized).astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]
            mask_t = (np.array(mask_resized).astype(np.float32) / 255.0)[np.newaxis, np.newaxis, ...]

            try:
                # Inpainter returns: infilled, composite (protected for single or multi-output)
                outputs = self.inp_sess.run(None, {"image": img_t, "mask": mask_t})
                comp = outputs[1] if len(outputs) > 1 else outputs[0]
                comp_arr = comp[0] if getattr(comp, "ndim", 0) == 4 else comp
                if getattr(comp_arr, "ndim", 0) == 3 and comp_arr.shape[0] in (1, 3, 4):
                    comp_np = (comp_arr.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    comp_np = (comp_arr * 255.0).clip(0, 255).astype(np.uint8)

                if comp_np.ndim == 2:
                    restored_patch = Image.fromarray(comp_np, mode="L").convert("RGB").resize((pw, ph), Image.Resampling.BILINEAR)
                elif comp_np.shape[2] == 1:
                    restored_patch = Image.fromarray(comp_np[:, :, 0], mode="L").convert("RGB").resize((pw, ph), Image.Resampling.BILINEAR)
                else:
                    restored_patch = Image.fromarray(comp_np).resize((pw, ph), Image.Resampling.BILINEAR)

                # Ensure dark-interior boxes do not receive bright/white fills from the inpainter
                if is_dark_box:
                    restored_np = np.array(restored_patch)
                    if cv2 is not None:
                        rest_gray = cv2.cvtColor(restored_np, cv2.COLOR_RGB2GRAY)
                    else:
                        rest_gray = np.mean(restored_np, axis=2).astype(np.uint8)
                    text_mask_bool = mask_arr > 0.01
                    bad_bright = text_mask_bool & (rest_gray > dark_bg_luma + 25)
                    if np.any(bad_bright):
                        restored_np[bad_bright] = dark_bg_color
                    restored_patch = Image.fromarray(restored_np)

                # Create soft mask for clean composite (feathered) and paste onto cleaned_img
                blend_mask = Image.fromarray((np.clip(mask_arr, 0.0, 1.0) * 255.0).astype(np.uint8))
                cleaned_img.paste(restored_patch, (cx1, cy1), mask=blend_mask)
            except Exception as e:
                logger.error("Inpainting patch xatosi: %s", e)

        return cleaned_img

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "onnx_models")
    vocab_path = os.path.join(base_dir, "ocr_vocab.json")
    if not os.path.exists(vocab_path):
        vocab_path = os.path.join(base_dir, "..", "..", "mashinali oqitish", "ocr_vocab.json")
    if not os.path.exists(models_dir):
        models_dir = os.path.join(base_dir, "..", "..", "mashinali oqitish", "onnx_models")
    engine = MangaOnnxEngine(models_dir=models_dir, vocab_path=vocab_path)
    print("MangaOnnxEngine is_ready:", engine.is_ready)
    print("Det session:", engine.det_sess is not None)
    print("OCR session:", engine.ocr_sess is not None)
    print("Inpaint session:", engine.inp_sess is not None)

