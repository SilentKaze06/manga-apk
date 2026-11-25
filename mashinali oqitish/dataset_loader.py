import os
import json
import logging
import hashlib
import stat
import string
import zipfile
import zlib
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torchvision.transforms as T
import numpy as np

import ml_paths
ml_paths.ensure_ml_paths()
from utils import get_best_font

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Foreground-only classes; absence of objects is handled via objectness, not a bg class.
CATEGORY_TO_IDX = {
    "speech_bubble": 0,
    "thought_bubble": 1,
    "narration_box": 2,
    "sfx": 3,
}
CATEGORY_ALIASES = {
    "speech_bubble": "speech_bubble",
    "thought_bubble": "thought_bubble",
    "narration_box": "narration_box",
    "narration": "narration_box",
    "sfx": "sfx",
    "text_bubble": "speech_bubble",
    "text_line": "speech_bubble",
    "bubble": "speech_bubble",
}
IDX_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_IDX.items()}
NUM_DET_CLASSES = len(CATEGORY_TO_IDX)

CHARACTER_ARCHIVE_RELATIVE_PATH = os.path.join("manga_ml_datasets", "characters", "characters.zip")
ALPHANUMERIC_CHARACTERS = string.digits + string.ascii_uppercase + string.ascii_lowercase


def normalize_detection_category(category):
    """Map dataset category aliases (e.g. text_line, text_bubble) to canonical 4-class labels."""
    if not category:
        return None
    return CATEGORY_ALIASES.get(str(category).strip().lower())


def parse_annotation_bbox(ann, orig_w=800.0, orig_h=1200.0):
    """Parse an annotation's bbox into ordered (x1, y1, x2, y2, is_norm).

    Handles explicit bbox_format ('xyxy', 'xywh', 'yxyx'), infers xyxy vs yxyx
    from polygon coordinates when bbox_format is omitted (as in Comic-Text-Detector),
    and converts string numeric coordinates safely.
    """
    raw_bbox = ann.get("bbox", [])
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        b0, b1, b2, b3 = (float(c) for c in raw_bbox)
    except (ValueError, TypeError):
        return None

    fmt = ann.get("bbox_format")
    if fmt == "xyxy":
        x1, y1, x2, y2 = b0, b1, b2, b3
    elif fmt == "xywh":
        x1, y1, x2, y2 = b0, b1, b0 + b2, b1 + b3
    elif fmt == "yxyx":
        y1, x1, y2, x2 = b0, b1, b2, b3
    else:
        poly = ann.get("polygon")
        inferred = None
        if isinstance(poly, (list, tuple)) and len(poly) >= 3:
            try:
                xs = [float(p[0]) for p in poly]
                ys = [float(p[1]) for p in poly]
                px1, py1, px2, py2 = min(xs), min(ys), max(xs), max(ys)
                d_xyxy = abs(b0 - px1) + abs(b1 - py1) + abs(b2 - px2) + abs(b3 - py2)
                d_yxyx = abs(b1 - px1) + abs(b0 - py1) + abs(b3 - px2) + abs(b2 - py2)
                if d_xyxy < d_yxyx:
                    inferred = "xyxy"
                elif d_yxyx < d_xyxy:
                    inferred = "yxyx"
            except (ValueError, TypeError, IndexError):
                inferred = None
        if inferred is None and orig_w > 1 and orig_h > 1:
            if (max(b0, b2) > orig_h and max(b0, b2) <= orig_w) or (max(b1, b3) > orig_w and max(b1, b3) <= orig_h):
                inferred = "xyxy"
        if inferred == "xyxy":
            x1, y1, x2, y2 = b0, b1, b2, b3
        else:
            y1, x1, y2, x2 = b0, b1, b2, b3

    ox1, ox2 = min(x1, x2), max(x1, x2)
    oy1, oy2 = min(y1, y2), max(y1, y2)
    has_float_coord = any(isinstance(c, float) for c in raw_bbox)
    is_norm = bool(ann.get("normalized")) or (
        min(ox1, oy1) >= 0.0
        and max(ox2, oy2) <= 1.0
        and has_float_coord
        and (orig_w > 1 or orig_h > 1)
    )
    return ox1, oy1, ox2, oy2, is_norm


def resolve_sample_image(item, key="image_path", img_dir=None, sample_id=None):
    candidates = []
    if item.get(key):
        raw = item[key]
        candidates.extend([raw, os.path.join(BASE_DIR, raw), ml_paths.resolve_data_path(raw, BASE_DIR)])
    if img_dir and sample_id:
        candidates.append(os.path.join(img_dir, sample_id + ".png"))
    return next((p for p in candidates if p and os.path.isfile(p)), None)


class AlphanumericArchiveDataset(Dataset):
    """Permanent 62-class character data, read directly from its ZIP archive.

    Folder names are ``0`` through ``9``, ``A_U`` through ``Z_U``, and
    ``a_L`` through ``z_L``. Labels use the application's existing vocabulary.
    A stable SHA-256 bucket assigns each image to train (80%), validation
    (10%), or test (10%); validation/test entries never enter the train split.
    """

    def __init__(self, archive_path, char_to_idx, split="train", augment=True, target_size=(64, 64)):
        if split not in {"train", "val", "test", "all"}:
            raise ValueError("split must be train, val, test, or all")
        missing_labels = set(ALPHANUMERIC_CHARACTERS) - set(char_to_idx)
        if missing_labels:
            raise ValueError(f"Vocabulary is missing character labels: {sorted(missing_labels)}")
        if any(not isinstance(char_to_idx[ch], int) or char_to_idx[ch] < 0 for ch in ALPHANUMERIC_CHARACTERS):
            raise ValueError("Character vocabulary labels must be nonnegative integers")
        if len(set(char_to_idx[ch] for ch in ALPHANUMERIC_CHARACTERS)) != 62:
            raise ValueError("Each character must have a distinct vocabulary label")
        if len(target_size) != 2 or any(size <= 0 for size in target_size):
            raise ValueError("target_size must contain two positive dimensions")

        self.archive_path = os.path.abspath(os.fspath(archive_path))
        self.split = split
        self.samples = []
        self.class_counts = dict.fromkeys(ALPHANUMERIC_CHARACTERS, 0)
        self.selected_class_counts = dict.fromkeys(ALPHANUMERIC_CHARACTERS, 0)
        self.split_counts = dict.fromkeys(("train", "val", "test"), 0)
        self._archive = None
        self._archive_pid = None

        seen = set()
        with zipfile.ZipFile(self.archive_path) as archive:
            for entry in sorted(archive.infolist(), key=lambda info: info.filename):
                name = entry.filename
                raw_name = entry.orig_filename
                # Validate raw components before any normalization can hide traversal.
                parts = name.rstrip("/").split("/")
                if raw_name != name or "\\" in raw_name or "\x00" in raw_name or any(part in {"", ".", ".."} for part in parts):
                    raise ValueError(f"Invalid character archive path: {raw_name!r}")
                if stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1:
                    raise ValueError(f"Unsupported character archive entry: {name!r}")
                if entry.is_dir() and parts == ["dataset"]:
                    continue
                if len(parts) < 2 or parts[0] != "dataset":
                    raise ValueError(f"Unexpected character archive entry: {name!r}")
                char = self._character_from_folder(parts[1])
                if entry.is_dir() and len(parts) == 2:
                    continue
                if entry.is_dir() or len(parts) != 3 or not parts[2].lower().endswith(".png"):
                    raise ValueError(f"Unexpected character image path: {name!r}")
                if name in seen:
                    raise ValueError(f"Duplicate character image path: {name!r}")
                seen.add(name)
                member_split = self.split_for_member(name)
                self.class_counts[char] += 1
                self.split_counts[member_split] += 1
                if split == "all" or member_split == split:
                    self.samples.append((name, char_to_idx[char]))
                    self.selected_class_counts[char] += 1

        missing_classes = [char for char, count in self.class_counts.items() if not count]
        if missing_classes:
            raise ValueError(f"Character archive is missing classes: {missing_classes}")
        self.total_count = sum(self.class_counts.values())
        transforms = [T.Resize(tuple(target_size))]
        if augment and split == "train":
            transforms.append(T.ColorJitter(brightness=0.15, contrast=0.15))
        transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.transform = T.Compose(transforms)

    @staticmethod
    def _character_from_folder(folder):
        if len(folder) == 1 and folder in string.digits:
            return folder
        if len(folder) == 3 and folder.endswith("_U") and folder[0] in string.ascii_uppercase:
            return folder[0]
        if len(folder) == 3 and folder.endswith("_L") and folder[0] in string.ascii_lowercase:
            return folder[0]
        raise ValueError(f"Unknown character class folder: {folder!r}")

    @staticmethod
    def split_for_member(name):
        bucket = int(hashlib.sha256(name.encode("utf-8")).hexdigest(), 16) % 10
        return "train" if bucket < 8 else ("val" if bucket == 8 else "test")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        if self._archive is None or self._archive_pid != os.getpid():
            self.close()
            self._archive = zipfile.ZipFile(self.archive_path)
            self._archive_pid = os.getpid()
        try:
            with self._archive.open(name) as image_file:
                with Image.open(image_file) as image:
                    image = image.convert("RGB")
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise RuntimeError(f"Cannot read character image {name!r} from {self.archive_path}") from exc
        return self.transform(image), torch.tensor(label, dtype=torch.long)

    def close(self):
        if getattr(self, "_archive", None) is not None:
            self._archive.close()
            self._archive = None
        self._archive_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_archive"] = None
        state["_archive_pid"] = None
        return state

    def __del__(self):
        self.close()

# ==============================================================================
# 1. DETECTION DATASET
# ==============================================================================

class MangaDetectionDataset(Dataset):
    """
    PyTorch Dataset for Manga Speech Bubble & Text Detection.
    Loads page metadata, resizes image to fixed size, scales bounding box annotations.
    """
    def __init__(self, jsonl_file, img_dir=None, target_size=(384, 256), is_train=False, allow_synthetic=False):
        self.target_h, self.target_w = target_size
        self.img_dir = img_dir
        self.is_train = is_train
        self.allow_synthetic = allow_synthetic
        self.samples = []
        
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    self.samples.append(json.loads(line))
                except Exception as e:
                    logger.debug("Error reading line in detection jsonl: %s", e)

        self.missing_image_count = sum(resolve_sample_image(
            item, img_dir=img_dir, sample_id=item.get("page_id", f"page_{i}")) is None
            for i, item in enumerate(self.samples))
        if self.missing_image_count and not allow_synthetic:
            raise FileNotFoundError(
                f"Detection dataset {jsonl_file}: {self.missing_image_count}/{len(self.samples)} "
                "page images missing. Supply real images; synthetic rendering requires explicit allow_synthetic=True (demo only).")
        if self.missing_image_count:
            logger.warning("DEMO ONLY: %d/%d detection pages are synthetic; metrics are not real-page accuracy",
                           self.missing_image_count, len(self.samples))
                
        transforms_list = [
            T.Resize((self.target_h, self.target_w)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        self.transform = T.Compose(transforms_list)
        self.tensor_norm = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        item = self.samples[idx]
        orig_w = max(1.0, float(item.get("width", 800) or 800))
        orig_h = max(1.0, float(item.get("height", 1200) or 1200))
        
        # Load image if exists, otherwise generate synthetic representation
        img_path = None
        if self.img_dir:
            cand1 = os.path.join(self.img_dir, item.get("page_id", f"page_{idx}") + ".png")
            if os.path.exists(cand1):
                img_path = cand1
        if not img_path and item.get("image_path"):
            cand_p = item.get("image_path")
            if os.path.exists(cand_p):
                img_path = cand_p
            elif os.path.exists(os.path.join(BASE_DIR, cand_p)):
                img_path = os.path.join(BASE_DIR, cand_p)
        if img_path and os.path.exists(img_path):
            try:
                with Image.open(img_path) as im:
                    orig_w, orig_h = float(im.size[0]), float(im.size[1])
                    img = im.convert("RGB")
            except Exception as e:
                raise RuntimeError(f"Cannot read detection image: {img_path}") from e
            img_tensor = self.transform(img)
        else:
            if not self.allow_synthetic:
                raise FileNotFoundError(f"Detection image disappeared: {item.get('page_id', idx)}")
            # High-speed synthetic canvas rendered directly at target resolution
            rng = np.random.default_rng(zlib.crc32(str(item.get("page_id", idx)).encode("utf-8")))
            noise = rng.integers(235, 255, size=(self.target_h, self.target_w, 3), dtype=np.uint8)
            img = Image.fromarray(noise)
            draw = ImageDraw.Draw(img)
            draw.rectangle([5, 5, self.target_w - 5, self.target_h - 5], outline=(40, 40, 40), width=2)
            
            scale_x = self.target_w / orig_w
            scale_y = self.target_h / orig_h
            
            # Draw synthetic speech bubbles matching ground truth directly at target size
            for ann in item.get("annotations", []):
                parsed = parse_annotation_bbox(ann, orig_w, orig_h)
                if parsed is None:
                    continue
                x1, y1, x2, y2, is_norm = parsed
                if is_norm:
                    bx_min, bx_max = x1 * self.target_w, x2 * self.target_w
                    by_min, by_max = y1 * self.target_h, y2 * self.target_h
                else:
                    bx_min, bx_max = x1 * scale_x, x2 * scale_x
                    by_min, by_max = y1 * scale_y, y2 * scale_y
                if bx_max > bx_min and by_max > by_min:
                    draw.ellipse([bx_min, by_min, bx_max, by_max], fill=(255, 255, 255), outline=(30, 30, 30), width=2)
            img_tensor = self.tensor_norm(img)

        # Format normalized bboxes [cx, cy, bw, bh] and class labels
        boxes = []
        labels = []
        for ann in item.get("annotations", []):
            cat = normalize_detection_category(ann.get("category", "speech_bubble"))
            if cat is None or cat not in CATEGORY_TO_IDX:
                continue
            cat_id = CATEGORY_TO_IDX[cat]
            parsed = parse_annotation_bbox(ann, orig_w, orig_h)
            if parsed is None:
                continue
            x1, y1, x2, y2, is_norm = parsed
            if is_norm:
                xmin = max(0.0, x1)
                xmax = min(1.0, x2)
                ymin = max(0.0, y1)
                ymax = min(1.0, y2)
            else:
                xmin = max(0.0, x1 / orig_w)
                xmax = min(1.0, x2 / orig_w)
                ymin = max(0.0, y1 / orig_h)
                ymax = min(1.0, y2 / orig_h)
            if xmax <= xmin or ymax <= ymin:
                continue
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            bw = max(0.001, xmax - xmin)
            bh = max(0.001, ymax - ymin)
            boxes.append([cx, cy, bw, bh])
            labels.append(cat_id)

        return {
            "image": img_tensor,
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
            "page_id": item.get("page_id", f"page_{idx}")
        }

# ==============================================================================
# 2. OCR DATASET
# ==============================================================================

def prepare_ocr_image(img, target_h=64, target_w=256, direction=None):
    """Aspect-preserving OCR preprocessing with multiple-of-8 width and vertical rotation."""
    target_w = max(64, ((int(target_w) + 7) // 8) * 8)
    target_h = max(32, ((int(target_h) + 7) // 8) * 8)
    if img.mode != "RGB":
        img = img.convert("RGB")
    cw, ch = img.size
    if cw < 1 or ch < 1:
        return Image.new("RGB", (target_w, target_h), (255, 255, 255))

    if (direction == "vertical" and ch > cw) or (direction is None and ch > cw * 1.25):
        img = img.transpose(Image.Transpose.ROTATE_90)
        cw, ch = img.size

    if (cw, ch) == (target_w, target_h):
        return img

    scale = min(target_w / float(cw), target_h / float(ch))
    nw = max(1, min(target_w, int(round(cw * scale))))
    nh = max(1, min(target_h, int(round(ch * scale))))
    resized = img.resize((nw, nh), Image.Resampling.BILINEAR)

    arr = np.array(resized)
    edges = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]], axis=0)
    bg_color = tuple(int(v) for v in np.median(edges, axis=0))
    canvas = Image.new("RGB", (target_w, target_h), bg_color)
    canvas.paste(resized, (0, (target_h - nh) // 2))
    return canvas


class MangaOCRDataset(Dataset):
    """
    PyTorch Dataset for Multi-Language Manga OCR (Japanese, Korean, Chinese, English, Russian).
    Accepts text crops, applies aspect-preserving resize to multiple-of-8 width, tokenizes into character indices.
    """
    def __init__(self, jsonl_file, char_to_idx, img_dir=None, target_size=(64, 256), max_text_len=64, is_train=False, allow_synthetic=False):
        self.target_h = max(32, ((int(target_size[0]) + 7) // 8) * 8)
        self.target_w = max(64, ((int(target_size[1]) + 7) // 8) * 8)
        self.char_to_idx = char_to_idx
        self.max_text_len = max_text_len
        self.img_dir = img_dir
        self.is_train = is_train
        self.allow_synthetic = allow_synthetic
        self.samples = []
        self.filtered_count = 0
        
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    text = item.get("text", "")
                    if text.strip():
                        # CTC needs a time step per symbol plus repeated-symbol separators.
                        required_steps = len(text) + sum(a == b for a, b in zip(text, text[1:]))
                        if len(text) > max_text_len or required_steps > self.target_w // 8:
                            self.filtered_count += 1
                            continue
                        self.samples.append(item)
                except Exception as e:
                    logger.debug("Error reading line in ocr jsonl: %s", e)
        if self.filtered_count:
            logger.warning("OCR dataset %s: excluded %d labels exceeding full-text CTC capacity; labels were not truncated",
                           jsonl_file, self.filtered_count)
        self.missing_image_count = sum(resolve_sample_image(
            item, img_dir=img_dir, sample_id=item.get("crop_id", f"crop_{i}")) is None
            for i, item in enumerate(self.samples))
        if self.missing_image_count and not allow_synthetic:
            raise FileNotFoundError(f"OCR dataset {jsonl_file}: {self.missing_image_count} images missing; synthetic rendering is demo-only")
                    
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item.get("text", "")
        crop_id = item.get("crop_id", f"crop_{idx}")
        
        # Load or generate synthetic crop image with text rendered
        crop_path = resolve_sample_image(item, img_dir=self.img_dir, sample_id=crop_id)
        render_text = text
        
        if crop_path and os.path.exists(crop_path):
            try:
                with Image.open(crop_path) as im:
                    img = im.convert("RGB")
            except Exception as e:
                raise RuntimeError(f"Cannot read OCR image: {crop_path}") from e
            img = prepare_ocr_image(img, target_h=self.target_h, target_w=self.target_w, direction=item.get("direction"))
        else:
            if not self.allow_synthetic:
                raise FileNotFoundError(f"OCR image disappeared: {crop_id}")
            # Synthetic crop with glyph-verified font rendering & width fitting
            font_size = 22
            font = get_best_font(size=font_size, sample_text=render_text)
            bounds = font.getbbox(render_text)
            while font_size > 10 and (bounds[2] - bounds[0]) > self.target_w - 16:
                font_size -= 2
                font = get_best_font(size=font_size, sample_text=render_text)
                bounds = font.getbbox(render_text)
            img = Image.new("RGB", (self.target_w, self.target_h), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            th = max(1, bounds[3] - bounds[1])
            pos_y = max(4, (self.target_h - th) // 2 - bounds[1])
            draw.text((8, pos_y), render_text, font=font, fill=(0, 0, 0))
            
        img_tensor = self.transform(img)
        
        # Tokenize the actual rendered text
        tokens = [self.char_to_idx.get(c, self.char_to_idx.get("<unk>", 1)) for c in render_text]
        if not tokens:
            tokens = [self.char_to_idx.get("<unk>", 1)]
        target_len = len(tokens)
        
        return {
            "image": img_tensor,
            "target": torch.tensor(tokens, dtype=torch.long),
            "target_len": torch.tensor(target_len, dtype=torch.long),
            "text": render_text,
            "crop_id": crop_id
        }

# ==============================================================================
# 3. INPAINTING DATASET
# ==============================================================================

class MangaInpaintingDataset(Dataset):
    """
    PyTorch Dataset for Manga Speech Bubble / Text Removal Inpainting.
    Yields (Raw Image, Binary Mask, Clean Ground Truth).
    """
    def __init__(self, jsonl_file, img_dir=None, target_size=(256, 256), is_train=False, allow_synthetic=False):
        self.target_h, self.target_w = target_size
        self.img_dir = img_dir
        self.is_train = is_train
        self.allow_synthetic = allow_synthetic
        self.samples = []
        
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    self.samples.append(json.loads(line))
                except Exception as e:
                    logger.debug("Error reading line in inpaint jsonl: %s", e)

        self.missing_triplet_count = sum(any(path is None for path in self._paths(item, i))
                                         for i, item in enumerate(self.samples))
        if self.missing_triplet_count and not allow_synthetic:
            raise FileNotFoundError(f"Inpainting dataset {jsonl_file}: {self.missing_triplet_count} triplets incomplete; synthetic data is demo-only")
                    
        self.img_transform = T.Compose([
            T.Resize((self.target_h, self.target_w)),
            T.ToTensor()
        ])
        self.mask_transform = T.Compose([
            T.Resize((self.target_h, self.target_w), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])
        
    def __len__(self):
        return len(self.samples)

    def _paths(self, item, idx):
        triplet_id = item.get("triplet_id", f"inpaint_{idx}")
        return tuple(resolve_sample_image(item, key=key,
                     img_dir=os.path.join(self.img_dir, folder) if self.img_dir else None,
                     sample_id=triplet_id)
                     for key, folder in (("raw_image", "raw"), ("mask", "mask"), ("ground_truth", "clean")))

    def __getitem__(self, idx):
        item = self.samples[idx]
        triplet_id = item.get("triplet_id", f"inpaint_{idx}")

        raw_path, mask_path, clean_path = self._paths(item, idx)

        if raw_path and mask_path and clean_path and os.path.exists(raw_path) and os.path.exists(mask_path) and os.path.exists(clean_path):
            try:
                with Image.open(raw_path) as im: raw_img = im.convert("RGB")
                with Image.open(mask_path) as im: mask_img = im.convert("L")
                with Image.open(clean_path) as im: clean_img = im.convert("RGB")
            except Exception as e:
                raise RuntimeError(f"Cannot read inpainting triplet: {triplet_id}") from e
        else:
            if not self.allow_synthetic:
                raise FileNotFoundError(f"Inpainting triplet disappeared: {triplet_id}")
            # Realistic synthetic manga texture generator with sample-specific diversity
            rng = np.random.default_rng(zlib.crc32(str(triplet_id).encode("utf-8")))
            clean_np = np.ones((self.target_h, self.target_w, 3), dtype=np.uint8) * 255
            step = int(rng.integers(4, 9))
            clean_np[::step, ::step] = int(rng.integers(150, 210))
            clean_img = Image.fromarray(clean_np)
            draw_clean = ImageDraw.Draw(clean_img)

            max_x = max(61, self.target_w - 60)
            max_y = max(61, self.target_h - 60)
            min_x = min(60, max_x - 1)
            min_y = min(60, max_y - 1)
            bcx = int(rng.integers(min_x, max_x))
            bcy = int(rng.integers(min_y, max_y))
            brx = int(rng.integers(35, 75))
            bry = int(rng.integers(30, 65))
            is_dark = (idx % 3 == 1)
            fill_col = (24, 24, 28) if is_dark else ((248, 242, 218) if idx % 3 == 2 else (255, 255, 255))
            text_col = (245, 245, 245) if is_dark else (10, 10, 10)
            draw_clean.ellipse([bcx - brx, bcy - bry, bcx + brx, bcy + bry], fill=fill_col, outline=(30, 30, 30), width=3)

            mask_img = Image.new("L", (self.target_w, self.target_h), 0)
            draw_mask = ImageDraw.Draw(mask_img)
            mw = int(rng.integers(20, max(25, brx - 10)))
            mh = int(rng.integers(12, max(18, bry - 10)))
            draw_mask.rectangle([bcx - mw, bcy - mh, bcx + mw, bcy + mh], fill=255)

            raw_img = clean_img.copy()
            draw_raw = ImageDraw.Draw(raw_img)
            font_syn = get_best_font(size=16, sample_text="TEXT")
            draw_raw.text((bcx - mw + 4, bcy - 8), "TEXT", fill=text_col, font=font_syn)

        raw_tensor = self.img_transform(raw_img)
        mask_tensor = (self.mask_transform(mask_img)[:1] > 0.5).float()  # (1, H, W) strict binary via NEAREST
        clean_tensor = self.img_transform(clean_img)
        
        return {
            "raw": raw_tensor,
            "mask": mask_tensor, # (1, H, W)
            "clean": clean_tensor, # (3, H, W)
            "triplet_id": triplet_id
        }

# Collate functions for DataLoader
def ocr_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    targets = [item["target"] for item in batch]
    target_lens = torch.stack([item["target_len"] for item in batch])
    texts = [item["text"] for item in batch]
    crop_ids = [item["crop_id"] for item in batch]
    
    # Flatten targets for CTC loss
    flat_targets = torch.cat(targets)
    return {
        "images": images,
        "targets": flat_targets,
        "target_lens": target_lens,
        "texts": texts,
        "crop_ids": crop_ids
    }

def detection_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    boxes = [item["boxes"] for item in batch]
    labels = [item["labels"] for item in batch]
    page_ids = [item["page_id"] for item in batch]
    return {
        "images": images,
        "boxes": boxes,
        "labels": labels,
        "page_ids": page_ids
    }

