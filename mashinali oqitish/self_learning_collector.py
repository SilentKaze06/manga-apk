import os
import sys
import json
import time
import tempfile
import logging
import hashlib
import threading
import unicodedata
import cv2
import numpy as np
from PIL import Image

from utils import atomik_json_yoz

logger = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(BASE_DIR, "ocr_vocab.json")

class SelfLearningCollector:
    """
    Continuous Self-Learning Data Ingestion & Active Feedback Module:
    1. Collects and stores user-processed manga pages, cropped text bubbles, and inpainting triplets.
    2. Performs rigorous automated quality filtering (rejects empty text, blank crops, uninformative masks).
    3. Prevents duplicate ingestion via perceptual/MD5 image content hashing.
    4. Supports Human-in-the-Loop text corrections (Gold Standard ground truth updates).
    5. Dynamically updates vocabulary (ocr_vocab.json) when new valid characters are discovered.
    6. Maintains live statistics and triggers incremental model fine-tuning when threshold is reached.
    """
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or BASE_DIR
        self.vocab_path = os.path.join(self.base_dir, "ocr_vocab.json")
        self.data_dir = os.path.join(self.base_dir, "self_learning_data")
        
        self.ocr_crops_dir = os.path.join(self.data_dir, "ocr", "crops")
        self.det_pages_dir = os.path.join(self.data_dir, "detection", "pages")
        self.inpaint_raw_dir = os.path.join(self.data_dir, "inpainting", "raw")
        self.inpaint_mask_dir = os.path.join(self.data_dir, "inpainting", "mask")
        self.inpaint_clean_dir = os.path.join(self.data_dir, "inpainting", "clean")
        
        for d in [
            self.ocr_crops_dir, self.det_pages_dir,
            self.inpaint_raw_dir, self.inpaint_mask_dir, self.inpaint_clean_dir
        ]:
            os.makedirs(d, exist_ok=True)
            
        self.ocr_jsonl = os.path.join(self.data_dir, "ocr", "feedback_ocr.jsonl")
        self.det_jsonl = os.path.join(self.data_dir, "detection", "feedback_detection.jsonl")
        self.inpaint_jsonl = os.path.join(self.data_dir, "inpainting", "feedback_inpaint.jsonl")
        self.stats_file = os.path.join(self.data_dir, "stats.json")
        self._lock = threading.RLock()
        
        self._init_stats()

    def _init_stats(self):
        if not os.path.exists(self.stats_file):
            initial_stats = {
                "total_pages_collected": 0,
                "total_ocr_crops_collected": 0,
                "total_inpaint_triplets_collected": 0,
                "total_user_corrections": 0,
                "total_training_cycles": 0,
                "last_trained_timestamp": None,
                "last_train_loss": None,
                "auto_train_threshold": 5, # Fine-tune every 5 pages by default
                "unprocessed_pages_since_last_train": 0
            }
            self.save_stats(initial_stats)

    def load_stats(self):
        with self._lock:
            try:
                with open(self.stats_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Error loading stats from %s, returning defaults: %s", self.stats_file, e)
                return {
                    "total_pages_collected": 0,
                    "total_ocr_crops_collected": 0,
                    "total_inpaint_triplets_collected": 0,
                    "total_user_corrections": 0,
                    "total_training_cycles": 0,
                    "last_trained_timestamp": None,
                    "last_train_loss": None,
                    "auto_train_threshold": 5,
                    "unprocessed_pages_since_last_train": 0
                }

    def save_stats(self, stats):
        with self._lock:
            try:
                atomik_json_yoz(self.stats_file, stats)
            except Exception as e:
                logger.error("Failed to save stats to %s: %s", self.stats_file, e)

    def _jsonl_tozala(self, path, max_lines=5000):
        """
        Keeps JSONL file size bounded by pruning older lines if line count exceeds max_lines.
        Uses atomic file replacement to prevent data loss or corruption.
        """
        if not os.path.exists(path):
            return
        with self._lock:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                if len(lines) > max_lines:
                    pruned = lines[-max_lines:]
                    dir_name = os.path.dirname(os.path.abspath(path))
                    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
                        temp_name = tf.name
                        tf.writelines(pruned)
                        tf.flush()
                        os.fsync(tf.fileno())
                    os.replace(temp_name, path)
                    logger.info("Pruned JSONL %s from %d to %d records", path, len(lines), len(pruned))
            except Exception as e:
                logger.warning("Error pruning JSONL %s: %s", path, e)

    def _compute_img_hash(self, pil_img):
        """Computes MD5 hash from raw bytes of image."""
        return hashlib.md5(pil_img.tobytes()).hexdigest()

    def update_vocabulary_if_needed(self, new_text):
        """
        Dynamically adds valid new characters to ocr_vocab.json so the AI continuously grows its vocabulary.
        Excludes unprintable control characters and normalizes Unicode (NFKC).
        """
        if not new_text or not os.path.exists(self.vocab_path):
            return 0
            
        with self._lock:
            try:
                normalized_text = unicodedata.normalize("NFKC", str(new_text))
                with open(self.vocab_path, "r", encoding="utf-8") as f:
                    vocab_data = json.load(f)
                    
                char_to_idx = vocab_data.get("char_to_idx", {})
                idx_to_char = vocab_data.get("idx_to_char", {})
                
                added_count = 0
                next_idx = (max(int(v) for v in char_to_idx.values()) + 1) if char_to_idx else 1
                
                for ch in normalized_text:
                    # Valid printable character check
                    if len(ch) == 1 and (ord(ch) >= 32 or ch == "\n") and ord(ch) != 127:
                        if ch not in char_to_idx:
                            char_to_idx[ch] = next_idx
                            idx_to_char[str(next_idx)] = ch
                            next_idx += 1
                            added_count += 1
                            
                if added_count > 0:
                    vocab_data["char_to_idx"] = char_to_idx
                    vocab_data["idx_to_char"] = idx_to_char
                    atomik_json_yoz(self.vocab_path, vocab_data)
                    logger.info("[Vocabulary] Added %d new characters to ocr_vocab.json! Total tokens: %d", added_count, len(char_to_idx))
                    
                return added_count
            except Exception as e:
                logger.error("[Vocabulary Error] Could not update vocab: %s", e)
                return 0

    def segment_and_save_characters(self, pil_crop, text, is_verified=False, source="user_stream"):
        """
        Segments and stores each Latin letter (A-Z, a-z) and digit (0-9) as an individual image in ocr/crops/.
        Returns the count of new character crops saved.
        """
        with self._lock:
            try:
                with open(self.vocab_path, "r", encoding="utf-8") as f:
                    vocab_data = json.load(f)
                char_to_idx = vocab_data.get("char_to_idx", {})
            except Exception:
                char_to_idx = {}
                
            # Never drop unknown characters or map to 0: return 0 if any char is missing from vocabulary
            for c in text:
                if not c.isspace() and c not in char_to_idx:
                    logger.debug("Character %r missing from vocabulary; aborting segment_and_save_characters", c)
                    return 0

            clean_chars = [c for c in text if not c.isspace()]
            if not clean_chars:
                return 0
                
            try:
                gray = np.array(pil_crop.convert("L"))
            except Exception:
                return 0
                
            h, w = gray.shape
            if h < 8 or w < 8:
                return 0

            # Auto-invert dark crops (border median < 128)
            border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
            if border_pixels.size > 0 and np.median(border_pixels) < 128:
                gray = cv2.bitwise_not(gray)
                
            _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((2, 2), np.uint8)
            bin_clean = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(bin_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            boxes = []
            min_area = max(12, int((h * w) * 0.0008))
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw >= 2 and bh >= 6 and (bw * bh) >= min_area:
                    if bw > w * 0.95 and bh > h * 0.95:
                        continue
                    # Skip horizontal/vertical border slivers
                    if (bh <= 4 and bw >= 20) or (bw <= 3 and bh >= 20):
                        continue
                    boxes.append((x, y, bw, bh))
                    
            if not boxes:
                return 0
                
            # Special case: single character
            if len(clean_chars) == 1:
                all_x1 = max(0, min(b[0] for b in boxes) - 3)
                all_y1 = max(0, min(b[1] for b in boxes) - 3)
                all_x2 = min(w, max(b[0] + b[2] for b in boxes) + 3)
                all_y2 = min(h, max(b[1] + b[3] for b in boxes) + 3)
                matched_pairs = [((all_x1, all_y1, all_x2 - all_x1, all_y2 - all_y1), clean_chars[0])]
            else:
                # Merge vertically overlapping sub-contours (e.g. dot of 'i', 'j', '!', '?', ':')
                merged_boxes = []
                for box in sorted(boxes, key=lambda b: (b[0], b[1])):
                    bx, by, bw, bh = box
                    was_m = False
                    for k in range(len(merged_boxes)):
                        mx, my, mw, mh = merged_boxes[k]
                        x_overlap = max(0, min(bx + bw, mx + mw) - max(bx, mx))
                        x_union = max(bx + bw, mx + mw) - min(bx, mx)
                        is_horiz = (x_overlap > 0 or abs(bx - mx) < max(bw, mw) * 0.6)
                        is_vert = (by <= my + mh + int(h * 0.2)) and (my <= by + bh + int(h * 0.2))
                        if is_horiz and is_vert and x_union <= max(bw, mw, 25) * 1.4:
                            nx1 = min(bx, mx)
                            ny1 = min(by, my)
                            nx2 = max(bx + bw, mx + mw)
                            ny2 = max(by + bh, my + mh)
                            merged_boxes[k] = (nx1, ny1, nx2 - nx1, ny2 - ny1)
                            was_m = True
                            break
                    if not was_m:
                        merged_boxes.append((bx, by, bw, bh))

                # Cluster lines by vertical y-overlap before sorting left-to-right
                lines_clusters = []
                for box in sorted(merged_boxes, key=lambda b: b[1]):
                    bx, by, bw, bh = box
                    assigned = False
                    for line in lines_clusters:
                        l_y1 = min(b[1] for b in line)
                        l_y2 = max(b[1] + b[3] for b in line)
                        overlap = min(by + bh, l_y2) - max(by, l_y1)
                        if overlap >= 0.35 * min(bh, l_y2 - l_y1):
                            line.append(box)
                            assigned = True
                            break
                    if not assigned:
                        lines_clusters.append([box])

                sorted_boxes = []
                lines_clusters.sort(key=lambda line: min(b[1] for b in line))
                for line in lines_clusters:
                    line.sort(key=lambda b: b[0])
                    sorted_boxes.extend(line)
                merged_boxes = sorted_boxes

                # Only match if merged character boxes match clean character count exactly.
                # This protects the training dataset from being poisoned by misaligned strokes.
                if len(merged_boxes) == len(clean_chars):
                    matched_pairs = list(zip(merged_boxes, clean_chars))
                else:
                    logger.debug(
                        "Skipping character slicing: merged contours (%d) != chars (%d) to prevent mislabeling",
                        len(merged_boxes), len(clean_chars)
                    )
                    return 0
            
            saved_count = 0
            timestamp = int(time.time() * 1000)
            
            for i, ((x, y, bw, bh), ch) in enumerate(matched_pairs):
                pad = 3
                px1, py1 = max(0, x - pad), max(0, y - pad)
                px2, py2 = min(w, x + bw + pad), min(h, y + bh + pad)
                char_crop = pil_crop.crop((px1, py1, px2, py2))
                
                # Center into 64x64 canvas
                square_crop = Image.new("RGB", (64, 64), (255, 255, 255))
                cw, ch_dim = char_crop.size
                scale = min(48.0 / max(cw, 1), 48.0 / max(ch_dim, 1))
                nw, nh = max(4, int(cw * scale)), max(4, int(ch_dim * scale))
                char_resized = char_crop.resize((nw, nh), Image.Resampling.BILINEAR)
                ox = (64 - nw) // 2
                oy = (64 - nh) // 2
                square_crop.paste(char_resized, (ox, oy))
                
                # Minimum stroke intensity check (skip blank/dust speck)
                arr = np.array(square_crop.convert("L"))
                if np.sum(arr < 185) < 45:
                    continue
                
                prefix = "u" if ch.isupper() else ("l" if ch.islower() else "d")
                safe_ch = ch if ch.isalnum() else f"u{ord(ch):04x}"
                crop_id = f"char_{prefix}_{safe_ch}_{timestamp}_{i+1}"
                crop_path = os.path.join(self.ocr_crops_dir, f"{crop_id}.png")
                square_crop.save(crop_path)
                
                rec = {
                    "crop_id": crop_id,
                    "image_path": crop_path,
                    "char": ch,
                    "text": ch,
                    "class_id": char_to_idx[ch],
                    "category": "latin_char",
                    "verified": is_verified,
                    "source": source,
                    "timestamp": timestamp
                }
                with open(self.ocr_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                saved_count += 1
                
            return saved_count

    def record_ocr_correction(self, crop_img, clean_text, bubble_id=1, category="speech_bubble"):
        """
        Records human-verified OCR correction by segmenting individual Latin letters/digits
        into separate images in ocr/crops/.
        """
        if crop_img is None:
            return None
        if isinstance(crop_img, str):
            crop_img = Image.open(crop_img).convert("RGB")
        else:
            crop_img = crop_img.convert("RGB")
            
        clean_text = unicodedata.normalize("NFKC", str(clean_text)).strip()
        if not clean_text:
            return None
            
        with self._lock:
            self.update_vocabulary_if_needed(clean_text)
            count = self.segment_and_save_characters(crop_img, clean_text, is_verified=True, source="user_correction_gold")
            
            stats = self.load_stats()
            stats["total_user_corrections"] = stats.get("total_user_corrections", 0) + 1
            stats["total_ocr_crops_collected"] = stats.get("total_ocr_crops_collected", 0) + count
            self.save_stats(stats)
            return count
            
    def record_processed_page(self, orig_img, cleaned_img, mask_img, dialogues, is_verified=False, source_note="user_stream"):
        """
        Ingests a processed page and all its extracted components into the self-learning buffer.
        Automatically deduplicates and saves each character as an individual image.
        """
        with self._lock:
            if isinstance(orig_img, str):
                orig_img = Image.open(orig_img).convert("RGB")
            else:
                orig_img = orig_img.convert("RGB")
                
            if isinstance(cleaned_img, str):
                cleaned_img = Image.open(cleaned_img).convert("RGB")
            else:
                cleaned_img = cleaned_img.convert("RGB")
                
            if isinstance(mask_img, str):
                mask_img = Image.open(mask_img).convert("L")
            else:
                mask_img = mask_img.convert("L")
                
            w, h = orig_img.size
            timestamp = int(time.time() * 1000)
            page_hash = self._compute_img_hash(orig_img)
            page_id = f"self_page_{timestamp}"
            
            # 1. Check if detection page already exists by content hash
            page_img_path = os.path.join(self.det_pages_dir, f"{page_id}.png")
            page_is_duplicate = False
            
            if os.path.exists(self.det_jsonl):
                with open(self.det_jsonl, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            d = json.loads(line)
                            dp = d.get("image_path", "")
                            rec_hash = d.get("page_hash")
                            if rec_hash == page_hash:
                                page_is_duplicate = True
                                page_img_path = dp
                                break
                            if not rec_hash and os.path.exists(dp) and d.get("width") == w and d.get("height") == h:
                                try:
                                    with Image.open(dp) as im_disk:
                                        if self._compute_img_hash(im_disk.convert("RGB")) == page_hash:
                                            page_is_duplicate = True
                                            page_img_path = dp
                                            break
                                except Exception:
                                    pass
                        except Exception:
                            pass
                            
            if not page_is_duplicate:
                orig_img.save(page_img_path)
                
            det_annotations = []
            new_ocr_crops_count = 0
            new_inpaint_count = 0
            
            # 2. Extract OCR crops & Inpaint Patches
            for idx, item in enumerate(dialogues):
                bbox = item.get("bbox", [])
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if (x2 - x1) < 8 or (y2 - y1) < 8:
                    continue
                    
                cat = item.get("category", "speech_bubble")
                raw_text = item.get("transcription", "")
                clean_text = unicodedata.normalize("NFKC", str(raw_text)).strip()
                
                # Standardized bbox format [x1, y1, x2, y2]
                det_annotations.append({
                    "id": idx + 1,
                    "category": cat,
                    "bbox": [x1, y1, x2, y2],
                    "bbox_format": "xyxy"
                })
                
                # 2a. OCR: Save each character as an individual image
                if clean_text:
                    pad = 4
                    crop_x1, crop_y1 = max(0, x1 - pad), max(0, y1 - pad)
                    crop_x2, crop_y2 = min(w, x2 + pad), min(h, y2 + pad)
                    crop_img = orig_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    new_ocr_crops_count += self.segment_and_save_characters(crop_img, clean_text, is_verified=bool(is_verified), source=source_note)
                
                # 2b. Inpainting Triplet Quality Checks
                patch_w = x2 - x1
                patch_h = y2 - y1
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                half_side = max(64, max(patch_w, patch_h) // 2 + 16)
                
                px1, py1 = max(0, center_x - half_side), max(0, center_y - half_side)
                px2, py2 = min(w, center_x + half_side), min(h, center_y + half_side)
                
                mask_patch = mask_img.crop((px1, py1, px2, py2))
                m_arr = np.array(mask_patch)
                if m_arr.size == 0:
                    continue
                mask_ratio = np.count_nonzero(m_arr > 30) / float(m_arr.size)
                
                # Only record if mask has non-trivial coverage (0.5% - 90%)
                if 0.005 <= mask_ratio <= 0.90:
                    raw_patch = orig_img.crop((px1, py1, px2, py2))
                    clean_patch = cleaned_img.crop((px1, py1, px2, py2))
                    
                    # Check that clean patch is actually modified from raw patch
                    if not np.array_equal(np.array(raw_patch), np.array(clean_patch)):
                        triplet_id = f"{page_id}_patch_{idx+1}"
                        raw_path = os.path.join(self.inpaint_raw_dir, f"{triplet_id}.png")
                        clean_path = os.path.join(self.inpaint_clean_dir, f"{triplet_id}.png")
                        mask_path = os.path.join(self.inpaint_mask_dir, f"{triplet_id}.png")
                        
                        raw_patch.save(raw_path)
                        clean_patch.save(clean_path)
                        mask_patch.save(mask_path)
                        
                        inpaint_record = {
                            "triplet_id": triplet_id,
                            "raw_image": raw_path,
                            "mask": mask_path,
                            "ground_truth": clean_path,
                            "verified": False,
                            "ground_truth_source": "model_generated",
                            "timestamp": timestamp
                        }
                        with open(self.inpaint_jsonl, "a", encoding="utf-8") as f:
                            f.write(json.dumps(inpaint_record, ensure_ascii=False) + "\n")
                        new_inpaint_count += 1
                
            # 3. Append to Detection JSONL if page was new or verified
            if page_is_duplicate and is_verified:
                try:
                    if os.path.exists(self.det_jsonl):
                        with open(self.det_jsonl, "r", encoding="utf-8") as f:
                            records = [json.loads(line) for line in f if line.strip()]
                        updated = False
                        for rec in records:
                            if rec.get("page_id") == page_id or rec.get("page_hash") == page_hash:
                                rec["annotations"] = det_annotations
                                rec["verified"] = True
                                rec["annotations_verified"] = True
                                rec["timestamp"] = timestamp
                                updated = True
                                break
                        if updated:
                            dir_name = os.path.dirname(os.path.abspath(self.det_jsonl))
                            with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
                                temp_name = tf.name
                                for r in records:
                                    tf.write(json.dumps(r, ensure_ascii=False) + "\n")
                                tf.flush()
                                os.fsync(tf.fileno())
                            os.replace(temp_name, self.det_jsonl)
                except Exception as e:
                    logger.warning("Error updating detection JSONL for duplicate page: %s", e)
            elif not page_is_duplicate:
                det_record = {
                    "page_id": page_id,
                    "page_hash": page_hash,
                    "image_path": page_img_path,
                    "width": w,
                    "height": h,
                    "annotations": det_annotations,
                    "verified": False,
                    "annotations_verified": False,
                    "timestamp": timestamp
                }
                with open(self.det_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(det_record, ensure_ascii=False) + "\n")
                    
            # Prune buffers if needed
            self._jsonl_tozala(self.ocr_jsonl, max_lines=5000)
            self._jsonl_tozala(self.inpaint_jsonl, max_lines=5000)
            self._jsonl_tozala(self.det_jsonl, max_lines=5000)
            
            # Update Stats
            stats = self.load_stats()
            if not page_is_duplicate:
                stats["total_pages_collected"] += 1
            stats["total_ocr_crops_collected"] += new_ocr_crops_count
            stats["total_inpaint_triplets_collected"] += new_inpaint_count
            if is_verified:
                stats["total_user_corrections"] += 1
            stats["unprocessed_pages_since_last_train"] += 1
            self.save_stats(stats)
            
            threshold = stats.get("auto_train_threshold", 5)
            should_auto_train = (stats["unprocessed_pages_since_last_train"] >= threshold)
            
            return {
                "page_id": page_id,
                "ocr_crops_added": new_ocr_crops_count,
                "inpaint_triplets_added": new_inpaint_count,
                "total_pages": stats["total_pages_collected"],
                "unprocessed_count": stats["unprocessed_pages_since_last_train"],
                "should_auto_train": should_auto_train
            }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    collector = SelfLearningCollector()
    logger.info("SelfLearningCollector successfully initialized!")
    stats = collector.load_stats()
    logger.info("Current Self-Learning Stats: %s", json.dumps(stats, indent=2))
