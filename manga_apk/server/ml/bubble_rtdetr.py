"""
RT-DETR-v2 Bubble and Text Instance Detector.
Loads 'ogkalu/comic-text-and-bubble-detector' for fine-grained speech bubble
and text instance detection, permanently separating touching or adjacent bubbles.
"""
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

_DEFAULT_ID2LABEL = {
    0: "bubble",
    1: "text_bubble",
    2: "text_free",
}


def _box_iou(box1: List[int], box2: List[int]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    a1 = float(max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1]))
    a2 = float(max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _class_aware_nms(detections: List[Dict[str, Any]], iou_thresh: float = 0.45) -> List[Dict[str, Any]]:
    """Apply non-maximum suppression per class so touching bubbles of the same or different classes are preserved."""
    if len(detections) <= 1:
        return detections

    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for det in detections:
        by_class.setdefault(det["label"], []).append(det)

    kept: List[Dict[str, Any]] = []
    for cls_name, items in by_class.items():
        items.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
        cls_kept: List[Dict[str, Any]] = []
        for cand in items:
            b_c = cand["box"]
            suppress = False
            for prev in cls_kept:
                if _box_iou(b_c, prev["box"]) >= iou_thresh:
                    suppress = True
                    break
            if not suppress:
                cls_kept.append(cand)
        kept.extend(cls_kept)

    # Sort all kept detections by score descending
    kept.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
    return kept


class BubbleRTDetr:
    """
    Singleton wrapper for ogkalu/comic-text-and-bubble-detector (RT-DETR-v2).
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        conf_bubble: float = 0.35,
        conf_text_bubble: float = 0.30,
        conf_text_free: float = 0.30,
        nms_iou: float = 0.45,
        half: Optional[bool] = None,
    ):
        self.model_dir = model_dir or os.environ.get("MANGA_RTDETR_DIR", "/opt/models/rtdetr_bubble_detector")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_bubble = conf_bubble
        self.conf_text_bubble = conf_text_bubble
        self.conf_text_free = conf_text_free
        self.nms_iou = nms_iou
        if half is None:
            self.half = (self.device == "cuda") and (os.environ.get("MANGA_USE_FP16", "1").lower() in ("1", "true", "yes"))
        else:
            self.half = bool(half and self.device == "cuda")

        self.processor = None
        self.model = None
        self.label_map = dict(_DEFAULT_ID2LABEL)
        self._load_failed = False
        self._lock = threading.RLock()

    def _resolve_model_dir(self) -> Optional[Path]:
        candidates = [
            Path(self.model_dir),
            Path("/opt/models/rtdetr_bubble_detector"),
            Path(__file__).resolve().parent / "checkpoints" / "rtdetr_bubble_detector",
        ]
        for cand in candidates:
            if cand.is_dir() and (cand / "config.json").is_file():
                return cand
        return None

    def load_model(self) -> Optional[Any]:
        with self._lock:
            if self.model is not None and self.processor is not None:
                return self.model
            if self._load_failed:
                return None

            try:
                from transformers import AutoImageProcessor, AutoModelForObjectDetection
            except Exception as e:
                logger.warning("BubbleRTDetr: transformers AutoModelForObjectDetection unavailable: %s", e)
                self._load_failed = True
                return None

            local_dir = self._resolve_model_dir()
            offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"

            if local_dir is None:
                if offline:
                    logger.info("BubbleRTDetr weights missing at %s under HF_HUB_OFFLINE=1", self.model_dir)
                    self._load_failed = True
                    return None
                try:
                    from huggingface_hub import snapshot_download

                    dl_path = snapshot_download(
                        repo_id="ogkalu/comic-text-and-bubble-detector",
                        local_dir=self.model_dir,
                    )
                    local_dir = Path(dl_path)
                except Exception as exc:
                    logger.warning("Could not download ogkalu/comic-text-and-bubble-detector: %s", exc)
                    self._load_failed = True
                    return None

            try:
                logger.info("Loading BubbleRTDetr from %s on %s (half=%s)...", local_dir, self.device, self.half)
                processor = AutoImageProcessor.from_pretrained(str(local_dir), local_files_only=offline)
                model = AutoModelForObjectDetection.from_pretrained(str(local_dir), local_files_only=offline)
                model.eval()
                model.to(self.device)
                if self.half:
                    model.half()

                # Build dynamic label mapping
                raw_id2label = getattr(model.config, "id2label", None)
                if raw_id2label:
                    label_map = {}
                    for k, v in raw_id2label.items():
                        norm_v = str(v).lower()
                        if "free" in norm_v:
                            label_map[int(k)] = "text_free"
                        elif "text" in norm_v:
                            label_map[int(k)] = "text_bubble"
                        elif "bubble" in norm_v:
                            label_map[int(k)] = "bubble"
                        else:
                            label_map[int(k)] = norm_v
                    self.label_map = label_map
                else:
                    self.label_map = dict(_DEFAULT_ID2LABEL)

                self.processor = processor
                self.model = model
                logger.info("BubbleRTDetr loaded successfully. Labels: %s", self.label_map)
                return self.model
            except Exception as exc:
                logger.error("Failed to initialize BubbleRTDetr: %s", exc, exc_info=True)
                self._load_failed = True
                return None

    @property
    def is_available(self) -> bool:
        return self.load_model() is not None

    def _detect_single(self, rgb_img: Image.Image) -> List[Dict[str, Any]]:
        """Run RT-DETR-v2 on a single RGB image/slice and return raw thresholded detections."""
        w, h = rgb_img.size
        if w <= 10 or h <= 10:
            return []

        inputs = self.processor(images=rgb_img, return_tensors="pt")
        inputs = {
            k: (v.to(self.device).half() if (self.half and v.is_floating_point()) else v.to(self.device))
            for k, v in inputs.items()
        }

        with self._lock:
            with torch.inference_mode():
                outputs = self.model(**inputs)
                target_sizes = torch.tensor([[h, w]], device=self.device)
                raw_res = self.processor.post_process_object_detection(
                    outputs, target_sizes=target_sizes, threshold=0.20
                )[0]

        boxes = raw_res["boxes"].detach().cpu().numpy()
        scores = raw_res["scores"].detach().cpu().numpy()
        labels = raw_res["labels"].detach().cpu().numpy()

        candidates: List[Dict[str, Any]] = []
        for b, s, l in zip(boxes, scores, labels):
            cls_id = int(l)
            cls_name = self.label_map.get(cls_id, "bubble")
            score_val = float(s)

            if cls_name == "bubble" and score_val < self.conf_bubble:
                continue
            if cls_name == "text_bubble" and score_val < self.conf_text_bubble:
                continue
            if cls_name == "text_free" and score_val < self.conf_text_free:
                continue

            bx1 = max(0, min(w, int(round(b[0]))))
            by1 = max(0, min(h, int(round(b[1]))))
            bx2 = max(0, min(w, int(round(b[2]))))
            by2 = max(0, min(h, int(round(b[3]))))
            if bx2 <= bx1 + 4 or by2 <= by1 + 4:
                continue

            candidates.append({
                "box": [bx1, by1, bx2, by2],
                "score": round(score_val, 4),
                "label": cls_name,
            })
        return candidates

    def detect(self, pil_image: Image.Image) -> List[Dict[str, Any]]:
        """
        Runs object detection on the image.
        For tall webtoon/manhua pages (height / width > 2.0), slices vertically with overlap,
        maps boxes back to full-page coordinates, and deduplicates across slice seams.
        Returns a list of detections: [{"box": [x1, y1, x2, y2], "score": float, "label": str}, ...]
        """
        if self.load_model() is None or pil_image is None:
            return []

        w, h = pil_image.size
        if w <= 10 or h <= 10:
            return []

        rgb_img = pil_image.convert("RGB")
        try:
            if h / float(max(1, w)) <= 2.0:
                candidates = self._detect_single(rgb_img)
                return _class_aware_nms(candidates, iou_thresh=self.nms_iou)

            # Tall webtoon/manhua page: slice vertically with overlap
            slice_h = min(h, max(int(w * 1.5), 1024))
            overlap = max(128, int(slice_h * 0.20))
            stride = max(1, slice_h - overlap)

            all_candidates: List[Dict[str, Any]] = []
            y_start = 0
            while y_start < h:
                y_end = min(h, y_start + slice_h)
                if y_end == h and y_start > 0:
                    y_start = max(0, h - slice_h)
                slice_img = rgb_img.crop((0, y_start, w, y_end))
                slice_dets = self._detect_single(slice_img)
                for det in slice_dets:
                    bx1, by1, bx2, by2 = det["box"]
                    all_candidates.append({
                        "box": [bx1, by1 + y_start, bx2, by2 + y_start],
                        "score": det["score"],
                        "label": det["label"],
                    })
                if y_end >= h:
                    break
                y_start += stride

            return _class_aware_nms(all_candidates, iou_thresh=self.nms_iou)
        except Exception as e:
            logger.warning("BubbleRTDetr inference failed: %s", e)
            return []


_shared_rtdetr: Optional[BubbleRTDetr] = None
_shared_rtdetr_lock = threading.Lock()


def get_shared_bubble_rtdetr(model_dir: Optional[str] = None) -> BubbleRTDetr:
    global _shared_rtdetr
    if _shared_rtdetr is not None:
        return _shared_rtdetr
    with _shared_rtdetr_lock:
        if _shared_rtdetr is None:
            _shared_rtdetr = BubbleRTDetr(model_dir=model_dir)
    return _shared_rtdetr


_DISTINCT_COLORS = [
    (230, 25, 75),    # Red
    (60, 180, 75),    # Green
    (0, 130, 200),    # Blue
    (245, 130, 48),   # Orange
    (145, 30, 180),   # Purple
    (70, 240, 240),   # Cyan
    (240, 50, 230),   # Magenta
    (210, 245, 60),   # Lime
    (250, 190, 212),  # Pink
    (0, 128, 128),    # Teal
    (220, 190, 255),  # Lavender
    (170, 110, 40),   # Brown
    (128, 0, 0),      # Maroon
    (0, 0, 128),      # Navy
]


def _get_font(size: int = 18):
    try:
        for cand in [
            "arialbd.ttf",
            "Arial Bold.ttf",
            "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "arial.ttf",
        ]:
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                pass
    except Exception:
        pass
    return ImageFont.load_default()


def _draw_dashed_rect(draw: ImageDraw.ImageDraw, box: List[int], color: tuple, width: int = 2, dash_len: int = 6):
    x1, y1, x2, y2 = box
    for x in range(x1, x2, dash_len * 2):
        draw.line([(x, y1), (min(x + dash_len, x2), y1)], fill=color, width=width)
    for x in range(x1, x2, dash_len * 2):
        draw.line([(x, y2), (min(x + dash_len, x2), y2)], fill=color, width=width)
    for y in range(y1, y2, dash_len * 2):
        draw.line([(x1, y), (x1, min(y + dash_len, y2))], fill=color, width=width)
    for y in range(y1, y2, dash_len * 2):
        draw.line([(x2, y), (x2, min(y + dash_len, y2))], fill=color, width=width)


def annotate_detection_boxes(pil_img: Image.Image, blocks: List[Dict[str, Any]]) -> Image.Image:
    """
    Renders outer bubble/frame box in solid distinct color,
    inner text_box as a dashed rectangle,
    and a bold label '#1 (speech_bubble)' at top-left.
    """
    annotated = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = _get_font(size=18)
    w, h = annotated.size

    for idx, blk in enumerate(blocks):
        color = _DISTINCT_COLORS[idx % len(_DISTINCT_COLORS)]
        box = blk.get("box", [0, 0, 0, 0])
        tbox = blk.get("text_box", box)
        bid = blk.get("block_id", f"b_{idx + 1}")
        cat = blk.get("category", "speech_bubble")

        bx1, by1, bx2, by2 = max(0, min(w, int(box[0]))), max(0, min(h, int(box[1]))), max(0, min(w, int(box[2]))), max(0, min(h, int(box[3])))
        tx1, ty1, tx2, ty2 = max(0, min(w, int(tbox[0]))), max(0, min(h, int(tbox[1]))), max(0, min(w, int(tbox[2]))), max(0, min(h, int(tbox[3])))

        # 1. Outer box in distinct solid color
        draw.rectangle([bx1, by1, bx2, by2], outline=color, width=3)

        # 2. Inner text_box as dashed / distinct line
        _draw_dashed_rect(draw, [tx1, ty1, tx2, ty2], color=color, width=2, dash_len=5)

        # 3. Label tag
        tag_text = f"#{idx + 1} {bid} ({cat})"
        bbox_text = draw.textbbox((bx1, by1), tag_text, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
        badge_y1 = max(0, by1 - th - 6)
        badge_y2 = by1
        badge_x1 = bx1
        badge_x2 = min(w, bx1 + tw + 10)
        draw.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], fill=color)
        draw.text((badge_x1 + 4, badge_y1 + 2), tag_text, fill=(255, 255, 255), font=font)

    return annotated

