import os
import sys
import json
import logging
import threading
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as T

from models import MangaBubbleDetector, MangaOCRModel, MangaInpaintingModel
from dataset_loader import IDX_TO_CATEGORY
from self_learning_collector import SelfLearningCollector
from self_learning_trainer import SelfLearningTrainer
from utils import get_best_font, decode_ctc

logger = logging.getLogger(__name__)

sys.stdout.reconfigure(encoding='utf-8')

# Hardware optimization for AMD Ryzen 5 (4 cores, 8 threads)
if not torch.cuda.is_available():
    try:
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        cv2.setNumThreads(num_threads)
    except Exception:
        pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def is_dark_barrier_between_x(gray, boxA, boxB, dark_thresh=110, min_line_ratio=0.65):
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB
    if xA2 >= xB1:
        return False
    if xB1 - xA2 < 5:
        return False
    y_top = max(yA1, yB1)
    y_bot = min(yA2, yB2)
    if y_bot - y_top < 6:
        y_top = min(yA1, yB1)
        y_bot = max(yA2, yB2)
    gap_slice = gray[y_top:y_bot, xA2:xB1]
    if gap_slice.size == 0 or gap_slice.shape[1] < 2 or gap_slice.shape[0] < 4:
        return False
    dark_counts = np.sum(gap_slice < dark_thresh, axis=0)
    col_ratios = dark_counts / float(gap_slice.shape[0])
    return bool(np.any(col_ratios >= min_line_ratio))


def is_dark_barrier_between_y(gray, boxA, boxB, dark_thresh=110, min_line_ratio=0.65):
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB
    if yA2 >= yB1:
        return False
    if yB1 - yA2 < 5:
        return False
    x_left = max(xA1, xB1)
    x_right = min(xA2, xB2)
    if x_right - x_left < 6:
        x_left = min(xA1, xB1)
        x_right = max(xA2, xB2)
    gap_slice = gray[yA2:yB1, x_left:x_right]
    if gap_slice.size == 0 or gap_slice.shape[0] < 2 or gap_slice.shape[1] < 4:
        return False
    dark_counts = np.sum(gap_slice < dark_thresh, axis=1)
    row_ratios = dark_counts / float(gap_slice.shape[1])
    return bool(np.any(row_ratios >= min_line_ratio))


def can_merge_text_boxes(boxA, boxB, gray):
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB
    wA, hA = max(1, xA2 - xA1), max(1, yA2 - yA1)
    wB, hB = max(1, xB2 - xB1), max(1, yB2 - yB1)

    is_vert_A = hA > 1.25 * wA
    is_vert_B = hB > 1.25 * wB
    is_both_vertical = is_vert_A and is_vert_B

    # Horizontal text (standard English manga dialogue)
    if not is_both_vertical:
        # Rule 1: Words on the exact same horizontal line
        y_overlap = min(yA2, yB2) - max(yA1, yB1)
        min_h = min(hA, hB)
        if min_h > 0 and (y_overlap / min_h) >= 0.60:
            gap_x = max(0, max(xA1, xB1) - min(xA2, xB2))
            if gap_x <= max(8, int(min_h * 0.70)):
                left_box = boxA if xA1 < xB1 else boxB
                right_box = boxB if xA1 < xB1 else boxA
                if not is_dark_barrier_between_x(gray, left_box, right_box):
                    return True

        # Rule 2: Consecutive lines stacked vertically in the same dialogue
        x_overlap = min(xA2, xB2) - max(xA1, xB1)
        min_w = min(wA, wB)
        # Must have at least 35% horizontal overlap
        if min_w > 0 and (x_overlap / min_w) >= 0.35:
            # Centers must not be far apart horizontally
            midA = (xA1 + xA2) / 2.0
            midB = (xB1 + xB2) / 2.0
            if abs(midA - midB) <= max(wA, wB) * 0.75:
                gap_y = max(0, max(yA1, yB1) - min(yA2, yB2))
                avg_h = (hA + hB) / 2.0
                # Line spacing must be reasonable (not a paragraph/speaker gap)
                if gap_y <= max(14, int(avg_h * 1.15)):
                    top_box = boxA if yA1 < yB1 else boxB
                    bot_box = boxB if yA1 < yB1 else boxA
                    if not is_dark_barrier_between_y(gray, top_box, bot_box):
                        return True

    # Traditional vertical text (Japanese)
    if is_both_vertical:
        x_overlap = min(xA2, xB2) - max(xA1, xB1)
        min_w = min(wA, wB)
        if min_w > 0 and (x_overlap / min_w) >= 0.60:
            gap_y = max(0, max(yA1, yB1) - min(yA2, yB2))
            if gap_y <= max(8, int(min_w * 0.70)):
                top_box = boxA if yA1 < yB1 else boxB
                bot_box = boxB if yA1 < yB1 else boxA
                if not is_dark_barrier_between_y(gray, top_box, bot_box):
                    return True

        y_overlap = min(yA2, yB2) - max(yA1, yB1)
        min_h = min(hA, hB)
        if min_h > 0 and (y_overlap / min_h) >= 0.35:
            gap_x = max(0, max(xA1, xB1) - min(xA2, xB2))
            avg_w = (wA + wB) / 2.0
            if gap_x <= max(14, int(avg_w * 1.15)):
                left_box = boxA if xA1 < xB1 else boxB
                right_box = boxB if xA1 < xB1 else boxA
                if not is_dark_barrier_between_x(gray, left_box, right_box):
                    return True

    return False


def smart_cluster_text_boxes(raw_boxes, gray):
    """
    Groups individual text lines/words into distinct dialogue bubbles
    via adjacency graph connected components.
    """
    n = len(raw_boxes)
    if n == 0:
        return []

    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if can_merge_text_boxes(raw_boxes[i]["box"], raw_boxes[j]["box"], gray):
                adj[i].append(j)
                adj[j].append(i)

    visited = set()
    clusters = []
    for i in range(n):
        if i in visited:
            continue
        comp = []
        queue = [i]
        visited.add(i)
        while queue:
            cur = queue.pop(0)
            comp.append(cur)
            for neighbor in adj[cur]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Detect vertical vs horizontal layout:
        is_vertical = any(
            (raw_boxes[idx]["box"][3] - raw_boxes[idx]["box"][1]) > 1.3 * (raw_boxes[idx]["box"][2] - raw_boxes[idx]["box"][0])
            for idx in comp
        )
        if is_vertical:
            comp.sort(key=lambda idx: (-raw_boxes[idx]["box"][0], raw_boxes[idx]["box"][1]))
        else:
            comp.sort(key=lambda idx: (raw_boxes[idx]["box"][1], raw_boxes[idx]["box"][0]))

        bx1 = min(raw_boxes[idx]["box"][0] for idx in comp)
        by1 = min(raw_boxes[idx]["box"][1] for idx in comp)
        bx2 = max(raw_boxes[idx]["box"][2] for idx in comp)
        by2 = max(raw_boxes[idx]["box"][3] for idx in comp)

        full_text = " ".join(raw_boxes[idx]["text"] for idx in comp)
        clusters.append({
            "text_box": [bx1, by1, bx2, by2],
            "text": full_text,
            "indices": comp
        })

    return clusters


def expand_to_bubble_contour(gray, text_box, all_other_tboxes=None, max_pad_x=65, max_pad_y=65):
    """
    Snaps a text box to the speech bubble's true outer boundary.
    Guarantees a safe min_pad=5px so character ascenders/descenders/accents are never clipped.
    Prevents invasion into neighboring text boxes by checking midpoint boundaries.
    """
    h, w = gray.shape
    x1, y1, x2, y2 = text_box

    # Guaranteed safety padding to preserve text ascenders/descenders/accents
    min_pad = 5
    x1 = max(0, x1 - min_pad)
    y1 = max(0, y1 - min_pad)
    x2 = min(w, x2 + min_pad)
    y2 = min(h, y2 + min_pad)

    # Determine safe stopping bounds with respect to all other text boxes
    limit_left = 0
    limit_right = w - 1
    limit_top = 0
    limit_bot = h - 1

    if all_other_tboxes:
        for ob in all_other_tboxes:
            ox1, oy1, ox2, oy2 = ob
            # Check vertical overlap to constrain horizontal expansion
            if max(y1, oy1) < min(y2, oy2):
                if ox2 <= x1:
                    limit_left = max(limit_left, (ox2 + x1) // 2 + 1)
                elif ox1 >= x2:
                    limit_right = min(limit_right, (x2 + ox1) // 2 - 1)

            # Check horizontal overlap to constrain vertical expansion
            if max(x1, ox1) < min(x2, ox2):
                if oy2 <= y1:
                    limit_top = max(limit_top, (oy2 + y1) // 2 + 1)
                elif oy1 >= y2:
                    limit_bot = min(limit_bot, (y2 + oy1) // 2 - 1)

    # Left
    cx1 = x1
    for d in range(1, max_pad_x):
        tx = x1 - d
        if tx < limit_left:
            break
        col = gray[y1:y2, tx]
        if np.mean(col < 135) >= 0.40 or np.mean(col) < 170:
            cx1 = tx
            break
        cx1 = tx

    # Right
    cx2 = x2
    for d in range(1, max_pad_x):
        tx = x2 + d
        if tx > limit_right:
            break
        col = gray[y1:y2, tx]
        if np.mean(col < 135) >= 0.40 or np.mean(col) < 170:
            cx2 = tx
            break
        cx2 = tx

    # Top
    cy1 = y1
    for d in range(1, max_pad_y):
        ty = y1 - d
        if ty < limit_top:
            break
        row = gray[ty, cx1:cx2]
        if np.mean(row < 135) >= 0.40 or np.mean(row) < 170:
            cy1 = ty
            break
        cy1 = ty

    # Bottom
    cy2 = y2
    for d in range(1, max_pad_y):
        ty = y2 + d
        if ty > limit_bot:
            break
        row = gray[ty, cx1:cx2]
        if np.mean(row < 135) >= 0.40 or np.mean(row) < 170:
            cy2 = ty
            break
        cy2 = ty

    return [cx1, cy1, cx2, cy2]



def generate_ink_stroke_mask(gray_np, box, pad=3):
    """
    Generates a precise binary mask covering ONLY dialogue text ink strokes,
    preventing damage to character faces, hair, panel borders, and screentones.
    """
    h, w = gray_np.shape
    x1, y1, x2, y2 = box
    px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
    px2, py2 = min(w, x2 + pad), min(h, y2 + pad)
    if px2 <= px1 or py2 <= py1:
        return np.zeros((h, w), dtype=np.uint8)

    patch = gray_np[py1:py2, px1:px2]
    mean_val = np.mean(patch)
    if mean_val > 150:
        thresh_val = min(180, int(mean_val * 0.85))
        stroke_binary = (patch < thresh_val).astype(np.uint8) * 255
    else:
        thresh_val = max(100, int(mean_val * 1.2))
        stroke_binary = (patch > thresh_val).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    stroke_dilated = cv2.dilate(stroke_binary, k, iterations=1)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[py1:py2, px1:px2] = stroke_dilated
    return mask


class MangaTranslationPipeline:

    """
    End-to-End AI Manga Translation & Processing System:
    Step 1: Text-Guided Speech Bubble Detection (EasyOCR + MangaBubbleDetector ensemble with smart clustering)
    Step 2: Multi-Language OCR (Japanese / Korean / Chinese / English)
    Step 3: High-Resolution Deep Patch Inpainting & Background Restoration (U-Net)
    Step 4: Continuous Self-Learning & Dynamic Human Feedback Loop
    """
    def __init__(self, base_dir=None, vocab_path=None):
        self.base_dir = base_dir or BASE_DIR
        self.checkpoint_dir = os.path.join(self.base_dir, "checkpoints")
        self.vocab_path = vocab_path or os.path.join(self.base_dir, "ocr_vocab.json")
        self.model_lock = threading.RLock()
        
        # Self-Learning Components
        self.collector = SelfLearningCollector(base_dir=self.base_dir)
        self.trainer = SelfLearningTrainer(base_dir=self.base_dir)
        
        # Readers are selected per request. Never download weights implicitly.
        self._ocr_readers = {}
        self.easyocr_reader = None  # retained for callers inspecting availability
            
        self.det_transform = T.Compose([
            T.Resize((384, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.inpaint_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])
        self.reload_models()

    def reload_models(self):
        """Load all trained weights before swapping the live model references."""
        with self.model_lock:
            with open(self.vocab_path, encoding="utf-8") as stream:
                vocab_data = json.load(stream)
            char_to_idx = vocab_data["char_to_idx"]
            idx_to_char = {int(k): v for k, v in vocab_data["idx_to_char"].items()}
            vocab_size = max(char_to_idx.values()) + 1
            detector = MangaBubbleDetector(num_classes=4).to(DEVICE)
            ocr = MangaOCRModel(vocab_size=vocab_size, hidden_dim=128).to(DEVICE)
            inpainter = MangaInpaintingModel().to(DEVICE)
            for name, model in (("detector", detector), ("ocr", ocr), ("inpaint", inpainter)):
                path = os.path.join(self.checkpoint_dir, f"best_manga_{name}.pt")
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Trained checkpoint missing: {path}")
                state = torch.load(path, map_location=DEVICE, weights_only=True)
                if name == "ocr":
                    required = {"seq_fc.weight", "seq_fc.bias", "fc.4.weight", "features.0.weight", "rnn.weight_ih_l0"}
                    if not required.issubset(state):
                        raise ValueError("OCR checkpoint lacks trained sequence/character weights")
                    model.load_checkpoint_weights(state)
                else:
                    model.load_state_dict(state)
                if any(not torch.isfinite(t).all() for t in model.state_dict().values() if t.is_floating_point()):
                    raise ValueError(f"Non-finite {name} checkpoint")
                model.eval()
            with torch.inference_mode():
                detector(torch.zeros(1, 3, 384, 256, device=DEVICE))
                ocr(torch.zeros(1, 3, 64, 256, device=DEVICE), return_sequence=True)
                inpainter(torch.zeros(1, 3, 256, 256, device=DEVICE), torch.zeros(1, 1, 256, 256, device=DEVICE))
            self.char_to_idx, self.idx_to_char = char_to_idx, idx_to_char
            self.detector, self.ocr, self.inpainter = detector, ocr, inpainter

    def _get_ocr_reader(self, source_lang):
        languages = {"ja": ["ja", "en"], "ko": ["ko", "en"],
                     "zh": ["ch_sim", "en"], "en": ["en"]}
        with self.model_lock:
            if source_lang not in self._ocr_readers:
                try:
                    import easyocr
                    options = {}
                    if os.environ.get("MANGA_EASYOCR_MODEL_DIR"):
                        options["model_storage_directory"] = os.environ["MANGA_EASYOCR_MODEL_DIR"]
                    self._ocr_readers[source_lang] = easyocr.Reader(
                        languages[source_lang], gpu=torch.cuda.is_available(),
                        verbose=False, download_enabled=False, **options)
                except Exception as exc:
                    logger.warning("Local EasyOCR %s unavailable: %s", source_lang, exc)
                    self._ocr_readers[source_lang] = None
            return self._ocr_readers[source_lang]

    def process_page(self, image_input, known_bubbles=None, auto_collect=True, is_verified=False, progress_callback=None, source_lang="auto"):
        """
        Processes a single real manga page image with high-precision bubble detection.
        """
        if progress_callback:
            progress_callback(0.08, "Sahifa tahlilga tayyorlanmoqda...")

        if isinstance(image_input, str):
            orig_img = Image.open(image_input).convert("RGB")
        else:
            orig_img = image_input.convert("RGB")
            
        source_lang = (source_lang or "auto").strip().lower()
        source_lang = {"ch": "zh", "ch_sim": "zh", "jp": "ja"}.get(source_lang, source_lang)
        if source_lang not in {"auto", "ja", "ko", "zh", "en"}:
            raise ValueError(f"Unsupported OCR source language: {source_lang}")
        # Auto has no trustworthy language decision here: use the server's
        # multilingual fallback instead of assuming Japanese and erasing text.
        reader = self._get_ocr_reader(source_lang) if source_lang != "auto" else None
        self.easyocr_reader = reader
        w, h = orig_img.size
        if source_lang == "auto" and not known_bubbles:
            return {"original_image": orig_img, "annotated_image": orig_img.copy(),
                    "mask_image": Image.new("L", orig_img.size, 0),
                    "cleaned_image": orig_img.copy(), "dialogues": [],
                    "cleaned_bubble_ids": [], "learning_info": None,
                    "detected_source_lang": "auto", "requires_ocr_fallback": True}
        
        # 1. Text-Guided Speech Bubble Detection & Clustering
        detections = []
        if known_bubbles:
            detections = known_bubbles
        else:
            raw_boxes = []
            
            # Step A: Detect text lines via OCR
            if reader is not None:
                if progress_callback:
                    progress_callback(0.25, "EasyOCR butun sahifa matnlarini o'qimoqda...")
                try:
                    raw_ocr = reader.readtext(np.array(orig_img), canvas_size=2000, mag_ratio=1.8)
                    for item in raw_ocr:
                        bbox, text, conf = item
                        text = text.strip()
                        if conf < 0.50 or not text:
                            continue
                        # Filter standalone punctuation noise
                        if len(text) == 1 and not ('\u4e00' <= text <= '\u9fff' or '\u3040' <= text <= '\u309f' or '\u30a0' <= text <= '\u30ff' or text.isalnum()):
                            continue
                        xs = [p[0] for p in bbox]
                        ys = [p[1] for p in bbox]
                        raw_boxes.append({
                            "box": [max(0, int(min(xs))), max(0, int(min(ys))), min(w, int(max(xs))), min(h, int(max(ys)))],
                            "text": text,
                            "conf": float(conf)
                        })
                except Exception as e:
                    logger.debug("EasyOCR readtext exception: %s", e)

            # Step B: Boundary-Aware Dialogue Grouping & Bubble Contour Snapping
            gray_full = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2GRAY)
            grouped_clusters = smart_cluster_text_boxes(raw_boxes, gray_full)

            clusters = []
            all_tboxes = [c["text_box"] for c in grouped_clusters]
            for idx, c in enumerate(grouped_clusters):
                tbox = c["text_box"]
                other_tboxes = [b for j, b in enumerate(all_tboxes) if j != idx]
                bubble_box = expand_to_bubble_contour(gray_full, tbox, other_tboxes)
                clusters.append({
                    "category": "speech_bubble",
                    "box": bubble_box,
                    "text_box": tbox,
                    "text": c["text"]
                })


            # Step C: Complement with MangaBubbleDetector (Only if proposals have high whiteness / bubble shape)
            try:
                det_tensor = self.det_transform(orig_img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    model_preds = self.detector.predict(det_tensor, conf_thresh=0.60, iou_thresh=0.35)
                
                gray_np = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2GRAY)
                for p in model_preds:
                    box_norm = p["box"]
                    x1, y1 = int(box_norm[0] * w), int(box_norm[1] * h)
                    x2, y2 = int(box_norm[2] * w), int(box_norm[3] * h)
                    if x2 <= x1 + 25 or y2 <= y1 + 25:
                        continue
                    patch = gray_np[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    # Only accept detector box if it's genuinely a white bubble interior (>205)
                    if patch.size > 0 and np.mean(patch) > 205:
                        overlap = False
                        for c in clusters:
                            cb = c["box"]
                            if not (cb[2] < x1 or x2 < cb[0] or cb[3] < y1 or y2 < cb[1]):
                                overlap = True
                                break
                        if not overlap:
                            clusters.append({
                                "category": IDX_TO_CATEGORY.get(p["class_id"], "speech_bubble"),
                                "box": [max(0, x1), max(0, y1), min(w, x2), min(h, y2)],
                                "text": ""
                            })
            except Exception as e:
                logger.debug("MangaBubbleDetector verification error: %s", e)

            # Step D: Final filtering - eliminate dark false positives (bushes, dresses, legs, sky)
            gray_np = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2GRAY)
            for c in clusters:
                bx = c["box"]
                text = c.get("text", "").strip()
                patch = gray_np[bx[1]:bx[3], bx[0]:bx[2]]
                mean_lum = np.mean(patch) if patch.size > 0 else 0
                # Must have confirmed dialogue text OR be a bright speech bubble
                if text or mean_lum > 215:
                    detections.append(c)
                
        # 2. OCR and Inpainting Mask Generation
        if progress_callback:
            progress_callback(0.65, "Nutq bulutlari va matnlar klasterlanmoqda...")

        annotated_img = orig_img.copy()
        draw_annotated = ImageDraw.Draw(annotated_img)

        mask_np = np.zeros((h, w), dtype=np.uint8)
        gray_orig = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2GRAY)

        dialogue_records = []
        font_badge = get_best_font(size=20, bold=True)
        
        with self.model_lock:
            for idx, det in enumerate(detections):
                box = det["box"] # [x1, y1, x2, y2]
                cat = det.get("category", "speech_bubble")
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
                
                # Use detected text or run OCR crop recognition
                transcription = det.get("text", "").strip()
                ocr_trusted = bool(transcription) and float(det.get("conf", 1.0)) >= 0.50
                if not transcription and x2 > x1 and y2 > y1:
                    crop = orig_img.crop((x1, y1, x2, y2))
                    crop_np = np.array(crop)
                    # 1. Try EasyOCR with CLAHE contrast enhancement for low-contrast/screentoned bubbles
                    if reader is not None and crop_np.size > 0 and np.std(crop_np) > 8:
                        try:
                            crop_gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                            enhanced_gray = clahe.apply(crop_gray)
                            enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
                            res = reader.readtext(enhanced_rgb, mag_ratio=1.5)
                            if not res:
                                res = reader.readtext(crop_np)
                            if res:
                                transcription = " ".join(r[1].strip() for r in res if r[2] >= 0.50 and r[1].strip())
                                ocr_trusted = bool(transcription)
                        except Exception as e:
                            logger.debug("EasyOCR crop recognition error: %s", e)
                            
                    # 2. If still empty, use custom MangaOCRModel sequence prediction (CRNN + CTC)
                    if not transcription and hasattr(self, "ocr") and self.ocr is not None:
                        try:
                            seq_transform = T.Compose([
                                T.Resize((64, 256)),
                                T.ToTensor(),
                                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                            ])
                            seq_tensor = seq_transform(crop).unsqueeze(0).to(DEVICE)
                            with torch.no_grad():
                                log_probs = self.ocr(seq_tensor)  # (T, 1, vocab_size)
                                pred_text = decode_ctc(log_probs[:, 0, :], self.idx_to_char).strip()
                                if pred_text:
                                    transcription = pred_text
                        except Exception as e:
                            logger.debug("Sequence OCR fallback error: %s", e)

                    # 3. If still empty, fallback to character-level slicing
                    if not transcription:
                        try:
                            gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
                            _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                            kernel = np.ones((2, 2), np.uint8)
                            bin_clean = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kernel)
                            contours, _ = cv2.findContours(bin_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            c_boxes = []
                            ch_h, ch_w = gray.shape
                            min_area = max(12, int((ch_h * ch_w) * 0.0008))
                            for cnt in contours:
                                bx_c, by_c, bw_c, bh_c = cv2.boundingRect(cnt)
                                if bw_c >= 3 and bh_c >= 5 and (bw_c * bh_c) >= min_area:
                                    if bw_c <= ch_w * 0.95 or bh_c <= ch_h * 0.95:
                                        c_boxes.append((bx_c, by_c, bw_c, bh_c))
                            
                            if c_boxes:
                                line_thresh = max(14, int(ch_h * 0.35))
                                c_boxes = sorted(c_boxes, key=lambda b: (b[1] // line_thresh, b[0]))
                                
                                char_transform = T.Compose([
                                    T.Resize((64, 64)),
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                ])
                                
                                predicted_chars = []
                                with torch.no_grad():
                                    for bx_c, by_c, bw_c, bh_c in c_boxes:
                                        pad = 3
                                        px1, py1 = max(0, bx_c - pad), max(0, py1 - pad)
                                        px2, py2 = min(ch_w, bx_c + bw_c + pad), min(ch_h, by_c + bh_c + pad)
                                        char_crop = crop.crop((px1, py1, px2, py2))
                                        
                                        square_crop = Image.new("RGB", (64, 64), (255, 255, 255))
                                        cw_c, ch_c = char_crop.size
                                        scale_c = min(48.0 / max(cw_c, 1), 48.0 / max(ch_c, 1))
                                        nw_c, nh_c = max(4, int(cw_c * scale_c)), max(4, int(ch_c * scale_c))
                                        char_resized = char_crop.resize((nw_c, nh_c), Image.Resampling.BILINEAR)
                                        square_crop.paste(char_resized, ((64 - nw_c) // 2, (64 - nh_c) // 2))
                                        
                                        t_in = char_transform(square_crop).unsqueeze(0).to(DEVICE)
                                        logits = self.ocr(t_in)
                                        pred_id = int(torch.argmax(logits, dim=-1).item())
                                        char_str = self.idx_to_char.get(pred_id, "")
                                        if char_str:
                                            predicted_chars.append(char_str)
                                transcription = "".join(predicted_chars)
                        except Exception as e:
                            logger.debug("Character-level OCR fallback error: %s", e)
                        
                stroke_box = det.get("text_box", [x1, y1, x2, y2])
                dialogue_records.append({
                    "bubble_id": idx + 1,
                    "category": cat,
                    "bbox": [x1, y1, x2, y2],
                    "text_box": stroke_box,
                    "transcription": transcription,
                    "ocr_trusted": ocr_trusted,
                })
                
                # Draw on annotated image (Red box and #ID badge)
                color = (255, 0, 0)
                draw_annotated.rectangle([x1, y1, x2, y2], outline=color, width=4)
                bid_text = f"#{idx+1}"
                badge_w = max(42, int(len(bid_text) * 14 + 10))
                draw_annotated.rectangle([x1, max(0, y1 - 24), min(w, x1 + badge_w), max(0, y1)], fill=(255, 255, 255))
                draw_annotated.text((x1 + 4, max(0, y1 - 23)), bid_text, fill=color, font=font_badge)
                
                # Generate stroke-level inpainting mask (only masking text characters to preserve faces and line art)
                if transcription and ocr_trusted:
                    s_mask = generate_ink_stroke_mask(gray_orig, stroke_box, pad=3)
                    mask_np = np.maximum(mask_np, s_mask)

        mask_img = Image.fromarray(mask_np)
            
        # 3. Patch-Based Deep Neural Inpainting (MangaInpaintingModel U-Net)
        if progress_callback:
            progress_callback(0.85, "Fon tozalanmoqda (Inpainting)...")
        cleaned_page = orig_img.copy()
        cleaned_bubble_ids = []
        with self.model_lock:
            for det in dialogue_records:
                if not det["transcription"] or not det["ocr_trusted"]:
                    continue
                bx1, by1, bx2, by2 = det["bbox"]
                bw_patch, bh_patch = bx2 - bx1, by2 - by1
                if bw_patch < 8 or bh_patch < 8:
                    continue
                # Add context padding around bubble
                cx1 = max(0, bx1 - 12)
                cy1 = max(0, by1 - 12)
                cx2 = min(w, bx2 + 12)
                cy2 = min(h, by2 + 12)

                patch_img = orig_img.crop((cx1, cy1, cx2, cy2))
                patch_mask = mask_img.crop((cx1, cy1, cx2, cy2))

                patch_w, patch_h = patch_img.size
                t_img = self.inpaint_transform(patch_img).unsqueeze(0).to(DEVICE)
                t_mask = (self.inpaint_transform(patch_mask) > 0.5).float().unsqueeze(0).to(DEVICE)

                try:
                    with torch.no_grad():
                        _, comp = self.inpainter(t_img, t_mask)
                        comp_np = (comp.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

                    restored_patch = Image.fromarray(comp_np).resize((patch_w, patch_h), Image.Resampling.BILINEAR)
                    cleaned_page.paste(restored_patch, (cx1, cy1), mask=patch_mask)
                except Exception as e:
                    logger.debug("Neural inpainting patch failed, falling back to cv2: %s", e)
                    orig_cv = cv2.cvtColor(np.array(patch_img), cv2.COLOR_RGB2BGR)
                    mask_cv = np.array(patch_mask, dtype=np.uint8)
                    
                    # For sparse text stroke masks, cv2 TELEA inpainting is fast and crisp without downscaling
                    h_p, w_p = orig_cv.shape[:2]
                    scale = 1.0
                    max_dim = 800 # Safe max size for sparse stroke masks
                    if h_p > max_dim or w_p > max_dim:
                        scale = max_dim / max(h_p, w_p)
                        new_w = max(1, int(w_p * scale))
                        new_h = max(1, int(h_p * scale))
                        orig_cv = cv2.resize(orig_cv, (new_w, new_h))
                        mask_cv = cv2.resize(mask_cv, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    mask_cv = (mask_cv > 127).astype(np.uint8) * 255
                        
                    cleaned_cv = cv2.inpaint(orig_cv, mask_cv, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

                    
                    if scale < 1.0:
                        cleaned_cv = cv2.resize(cleaned_cv, (w_p, h_p))
                        
                    cleaned_page.paste(Image.fromarray(cv2.cvtColor(cleaned_cv, cv2.COLOR_BGR2RGB)), (cx1, cy1), mask=patch_mask)
                cleaned_bubble_ids.append(det["bubble_id"])
        
        # 4. Self-Learning Continuous Data Collection
        learning_info = None
        if auto_collect and self.collector is not None:
            try:
                learning_info = self.collector.record_processed_page(
                    orig_img=orig_img,
                    cleaned_img=cleaned_page,
                    mask_img=mask_img,
                    dialogues=dialogue_records,
                    is_verified=is_verified
                )
            except Exception as e:
                logger.error("[SelfLearning AutoCollect Error] %s", e)

        if progress_callback:
            progress_callback(1.0, "Jarayon muvaffaqiyatli yakunlandi!")

        return {
            "original_image": orig_img,
            "annotated_image": annotated_img,
            "mask_image": mask_img,
            "cleaned_image": cleaned_page,
            "dialogues": dialogue_records,
            "learning_info": learning_info,
            "cleaned_bubble_ids": cleaned_bubble_ids,
            "detected_source_lang": source_lang,
            "requires_ocr_fallback": not any(d["ocr_trusted"] for d in dialogue_records),
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Testing End-to-End Manga Pipeline...")
    pipeline = MangaTranslationPipeline()
    
    # Create test synthetic manga page
    w, h = 800, 1200
    test_page = Image.new("RGB", (w, h), (240, 240, 240))
    draw = ImageDraw.Draw(test_page)
    # Draw panels
    draw.rectangle([40, 40, 760, 560], outline=(20, 20, 20), width=4)
    draw.rectangle([40, 600, 760, 1140], outline=(20, 20, 20), width=4)
    
    # Draw sample bubble and Japanese text
    draw.ellipse([450, 80, 720, 380], fill=(255, 255, 255), outline=(0, 0, 0), width=3)
    font_test = get_best_font(size=20)
    draw.text((490, 180), "何をしているんだ？", fill=(0, 0, 0), font=font_test)
    
    results = pipeline.process_page(
        test_page,
        known_bubbles=[
            {"category": "speech_bubble", "box": [450, 80, 720, 380], "text": "何をしているんだ？"}
        ]
    )
    
    os.makedirs(os.path.join(BASE_DIR, "demo_output"), exist_ok=True)
    results["annotated_image"].save(os.path.join(BASE_DIR, "demo_output", "detected_page.png"))
    results["mask_image"].save(os.path.join(BASE_DIR, "demo_output", "inpaint_mask.png"))
    results["cleaned_image"].save(os.path.join(BASE_DIR, "demo_output", "cleaned_page.png"))
    
    with open(os.path.join(BASE_DIR, "demo_output", "transcriptions.json"), "w", encoding="utf-8") as f:
        json.dump(results["dialogues"], f, ensure_ascii=False, indent=2)
        
    logger.info("Pipeline executed successfully! Output saved to demo_output/")
