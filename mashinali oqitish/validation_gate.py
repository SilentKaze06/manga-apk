"""Fail-closed promotion of locally trained candidates using independent annotations.

Validation manifests live in validation/{ocr,detector,inpainting}.jsonl.
They are never generated from model output. See ML_GUIDE.md for their schema.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T

from dataset_loader import AlphanumericArchiveDataset, CHARACTER_ARCHIVE_RELATIVE_PATH, CATEGORY_TO_IDX
from utils import decode_ctc


class ValidationUnavailable(ValueError):
    pass


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_digest(path):
    # Decoded pixels also catch renamed or recompressed copies of a training image.
    with Image.open(path) as image:
        image = image.convert("RGB")
        return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def training_image_paths(dataset):
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base = dataset.dataset
        for idx in dataset.indices:
            if hasattr(base, "samples") and 0 <= idx < len(base.samples):
                sample = base.samples[idx]
                if isinstance(sample, dict):
                    for k in ("image_path", "raw_image", "ground_truth", "mask"):
                        p = sample.get(k)
                        if p:
                            yield p
                elif isinstance(sample, (str, Path)):
                    yield str(sample)
                elif isinstance(sample, (list, tuple)):
                    for s in sample:
                        if isinstance(s, (str, Path)):
                            yield str(s)
            elif hasattr(base, "__getitem__"):
                try:
                    item = base[idx]
                    if isinstance(item, dict):
                        for k in ("image_path", "raw_image", "ground_truth", "mask"):
                            p = item.get(k)
                            if p:
                                yield p
                    elif isinstance(item, (str, Path)):
                        yield str(item)
                    elif isinstance(item, (list, tuple)):
                        for s in item:
                            if isinstance(s, (str, Path)):
                                yield str(s)
                except Exception:
                    pass
    elif hasattr(dataset, "datasets"):
        for child in dataset.datasets:
            yield from training_image_paths(child)
    else:
        for sample in getattr(dataset, "samples", []):
            if isinstance(sample, dict):
                for k in ("image_path", "raw_image", "ground_truth", "mask"):
                    p = sample.get(k)
                    if p:
                        yield p
            elif isinstance(sample, (str, Path)):
                yield str(sample)
            elif isinstance(sample, (list, tuple)):
                for s in sample:
                    if isinstance(s, (str, Path)):
                        yield str(s)


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = row
    return previous[-1]


def should_promote(baseline, candidate):
    """Every reported metric must be finite and non-regressing; one must improve."""
    if not baseline or baseline.keys() != candidate.keys():
        return False
    directions = {"char_accuracy": 1, "line_cer": -1, "precision": 1,
                  "recall": 1, "f1": 1, "hole_mae": -1}
    improvements = []
    for name, old in baseline.items():
        new = candidate[name]
        if name not in directions or not math.isfinite(old) or not math.isfinite(new):
            return False
        improvements.append(directions[name] * (new - old))
    return all(delta >= -1e-8 for delta in improvements) and any(delta > 1e-8 for delta in improvements)


def atomic_promote(model, checkpoint_path, expected_digest):
    """Serialize completely before replacing; preserve incumbent on any failure."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file() or file_digest(checkpoint_path) != expected_digest:
        raise ValidationUnavailable("Incumbent changed during training; candidate was not promoted")
    # A held-out metric may not exercise every parameter (for example an
    # unused output class). Never publish weights that live reload will reject.
    state = model.state_dict()
    if any(not torch.isfinite(value).all() for value in state.values()
           if value.is_floating_point() or value.is_complex()):
        raise ValidationUnavailable("Candidate contains non-finite weights; incumbent retained")
    temporary = []
    try:
        with tempfile.NamedTemporaryFile(dir=checkpoint_path.parent, suffix=".pt", delete=False) as out:
            temporary.append(out.name)
            torch.save(state, out)
            out.flush()
            os.fsync(out.fileno())
        with tempfile.NamedTemporaryFile(dir=checkpoint_path.parent, suffix=".pt", delete=False) as backup:
            temporary.append(backup.name)
        shutil.copyfile(checkpoint_path, temporary[1])
        if file_digest(checkpoint_path) != expected_digest:
            raise ValidationUnavailable("Incumbent changed before promotion")
        os.replace(temporary[1], checkpoint_path.with_name(checkpoint_path.stem + "_prev.pt"))
        os.replace(temporary[0], checkpoint_path)
    finally:
        for name in temporary:
            if os.path.exists(name):
                os.unlink(name)


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / max(union, 1e-9)


class ValidationGate:
    def __init__(self, base_dir, task, char_to_idx, training_dataset, min_samples=10):
        self.base_dir = Path(base_dir)
        self.task = task
        self.idx_to_char = {int(v): k for k, v in char_to_idx.items()}
        self.char_dataset = None
        manifest = self.base_dir / "validation" / (task + ".jsonl")
        if not manifest.is_file():
            raise ValidationUnavailable(f"Independent verified validation required: {manifest}")
        self.manifest = manifest
        self.manifest_digest = file_digest(manifest)
        self.samples = []
        self.image_fingerprints = {}
        training_hashes = {image_digest(p) for p in training_image_paths(training_dataset)}
        seen = set()
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if (item.get("verified") is not True or item.get("split") != "val"
                    or item.get("source_type") != "real_manga_page"):
                raise ValidationUnavailable("Validation requires verified=true, split=val, source_type=real_manga_page")
            keys = ("raw_image", "mask", "ground_truth") if task == "inpainting" else ("image_path",)
            for key in keys:
                path = Path(item.get(key, ""))
                if not path.is_absolute():
                    path = self.base_dir / path
                if not path.is_file():
                    raise ValidationUnavailable(f"Missing validation {key}: {path}")
                item[key] = str(path)
                self.image_fingerprints[str(path)] = file_digest(path)
            input_path = item[keys[0]]
            fingerprint = image_digest(input_path)
            if fingerprint in seen or fingerprint in training_hashes:
                raise ValidationUnavailable("Validation images overlap training or duplicate one another")
            seen.add(fingerprint)
            if task == "ocr":
                text = item.get("text", "")
                if (not text.strip() or any(c not in char_to_idx for c in text)
                        or len(text) + sum(a == b for a, b in zip(text, text[1:])) > 32):
                    raise ValidationUnavailable("Validation OCR needs complete known labels fitting 32 CTC steps")
            elif task == "detector":
                for ann in item.get("annotations", []):
                    box = ann.get("bbox", [])
                    if (ann.get("bbox_format") != "xyxy" or ann.get("category") not in CATEGORY_TO_IDX
                            or len(box) != 4 or not all(math.isfinite(float(x)) for x in box)
                            or box[2] <= box[0] or box[3] <= box[1]):
                        raise ValidationUnavailable("Validation boxes must be valid xyxy boxes with known categories")
            elif task == "inpainting":
                if item.get("ground_truth_source") not in {"human_cleaned", "paired_original"}:
                    raise ValidationUnavailable("Inpainting validation cannot use model-generated ground truth")
                with Image.open(item["raw_image"]) as raw, Image.open(item["mask"]) as mask, Image.open(item["ground_truth"]) as clean:
                    if raw.size != mask.size or raw.size != clean.size or mask.convert("L").getextrema()[1] == 0:
                        raise ValidationUnavailable("Validation triplets need matching dimensions and a nonempty mask")
            self.samples.append(item)
        if len(self.samples) < min_samples:
            raise ValidationUnavailable(f"Need at least {min_samples} independent verified validation samples; found {len(self.samples)}")
        if task == "detector" and not any(s.get("annotations") for s in self.samples):
            raise ValidationUnavailable("Detector validation requires positive examples")
        if task == "ocr":
            archive = self.base_dir / CHARACTER_ARCHIVE_RELATIVE_PATH
            if not archive.is_file():
                raise ValidationUnavailable("Independent character validation archive is missing")
            self.char_dataset = AlphanumericArchiveDataset(archive, char_to_idx, split="val", augment=False)
            if not len(self.char_dataset):
                raise ValidationUnavailable("No held-out character validation images")
            self.char_indices = random.Random(42).sample(range(len(self.char_dataset)), min(256, len(self.char_dataset)))

    def verify_unchanged(self):
        if file_digest(self.manifest) != self.manifest_digest:
            raise ValidationUnavailable("Validation manifest changed during training")
        for path, digest in self.image_fingerprints.items():
            if file_digest(path) != digest:
                raise ValidationUnavailable("Validation images changed during training")

    @property
    def counts(self):
        result = {"verified_pages": len(self.samples)}
        if self.char_dataset is not None:
            result["held_out_characters"] = len(self.char_indices)
        return result

    def evaluate(self, model, device):
        self.verify_unchanged()
        model.eval()
        with torch.no_grad():
            if self.task == "ocr":
                correct = total = 0
                for images, labels in DataLoader(Subset(self.char_dataset, self.char_indices), batch_size=32):
                    correct += (model(images.to(device)).argmax(-1).cpu() == labels).sum().item()
                    total += len(labels)
                transform = T.Compose([T.Resize((64, 256)), T.ToTensor(), T.Normalize([.5]*3, [.5]*3)])
                edits = characters = 0
                for item in self.samples:
                    p = Path(item["image_path"])
                    if not p.is_absolute():
                        p = self.base_dir / p
                    with Image.open(p) as image:
                        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
                    pred = decode_ctc(model(tensor, return_sequence=True)[:, 0, :], self.idx_to_char)
                    edits += edit_distance(pred, item["text"])
                    characters += len(item["text"])
                char_acc = (correct / total) if total > 0 else 0.0
                line_cer = (edits / characters) if characters > 0 else 0.0
                return {"char_accuracy": char_acc, "line_cer": line_cer}
            if self.task == "detector":
                transform = T.Compose([T.Resize((384, 256)), T.ToTensor(), T.Normalize([.485,.456,.406], [.229,.224,.225])])
                tp = fp = fn = 0
                for item in self.samples:
                    p = Path(item["image_path"])
                    if not p.is_absolute():
                        p = self.base_dir / p
                    with Image.open(p) as image:
                        width, height = image.size
                        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
                    predictions = model.predict(tensor, conf_thresh=.60, iou_thresh=.35)
                    truth = [(CATEGORY_TO_IDX[a["category"]], [a["bbox"][0]/width, a["bbox"][1]/height,
                              a["bbox"][2]/width, a["bbox"][3]/height]) for a in item.get("annotations", [])
                             if a.get("category") in CATEGORY_TO_IDX]
                    matched = set()
                    for pred in sorted(predictions, key=lambda p: p["score"], reverse=True):
                        choices = [(box_iou(pred["box"], b), i) for i, (c, b) in enumerate(truth)
                                   if i not in matched and c == pred["class_id"]]
                        overlap, index = max(choices, default=(0, -1))
                        if overlap >= .5:
                            tp += 1
                            matched.add(index)
                        else:
                            fp += 1
                    fn += len(truth) - len(matched)
                precision = tp / max(1, tp + fp)
                recall = tp / max(1, tp + fn)
                return {"precision": precision, "recall": recall, "f1": 2*tp/max(1, 2*tp+fp+fn)}
            transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
            total_error = pixels = 0.
            for item in self.samples:
                raw_p = Path(item["raw_image"])
                if not raw_p.is_absolute():
                    raw_p = self.base_dir / raw_p
                with Image.open(raw_p) as image:
                    raw = transform(image.convert("RGB")).unsqueeze(0).to(device)
                clean_p = Path(item["ground_truth"])
                if not clean_p.is_absolute():
                    clean_p = self.base_dir / clean_p
                with Image.open(clean_p) as image:
                    clean = transform(image.convert("RGB")).unsqueeze(0).to(device)
                mask_p = Path(item["mask"])
                if not mask_p.is_absolute():
                    mask_p = self.base_dir / mask_p
                with Image.open(mask_p) as image:
                    mask_img = image.convert("L").resize((256, 256), Image.Resampling.NEAREST)
                    mask = (T.ToTensor()(mask_img) > .5).float().unsqueeze(0).to(device)
                if not mask.any():
                    raise ValidationUnavailable("Mask disappeared after resize")
                pred, _ = model(raw, mask)
                total_error += ((pred - clean).abs() * mask).sum().item()
                pixels += mask.sum().item() * 3
            return {"hole_mae": (total_error / pixels) if pixels > 0 else 0.0}

    def close(self):
        if self.char_dataset is not None:
            self.char_dataset.close()
