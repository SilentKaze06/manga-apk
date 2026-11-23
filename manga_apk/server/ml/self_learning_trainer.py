import os
import sys
import json
import time
import threading
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler, RandomSampler
import torchvision.transforms as T
from PIL import Image
import numpy as np

from models import MangaBubbleDetector, MangaOCRModel, MangaInpaintingModel
from dataset_loader import (
    CATEGORY_TO_IDX, NUM_DET_CLASSES, detection_collate_fn,
    AlphanumericArchiveDataset, CHARACTER_ARCHIVE_RELATIVE_PATH,
)
from utils import atomik_json_yoz
from validation_gate import ValidationGate, ValidationUnavailable, atomic_promote, file_digest, should_promote

logger = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(BASE_DIR, "training_metrics", "self_learning_history.json")


# ==============================================================================
# HELPER FUNCTIONS FOR DETECTION TRAINING
# ==============================================================================

def build_det_targets(boxes_list, labels_list, num_anchors, grid_h, grid_w, device):
    """
    Assign each GT box to the nearest grid cell with local cell-offset targets.
    Returns objectness (B, A, H, W), box coordinates (B, A, H, W, 4), and class labels (B, A, H, W).
    """
    B = len(boxes_list)
    obj = torch.zeros(B, num_anchors, grid_h, grid_w, device=device)
    box_t = torch.zeros(B, num_anchors, grid_h, grid_w, 4, device=device)
    cls_t = torch.zeros(B, num_anchors, grid_h, grid_w, dtype=torch.long, device=device)

    for b, (boxes, labels) in enumerate(zip(boxes_list, labels_list)):
        if boxes.numel() == 0:
            continue
        boxes = boxes.to(device)
        labels = labels.to(device)
        for box, lab in zip(boxes, labels):
            lab_i = int(lab.item())
            if lab_i < 0 or lab_i >= NUM_DET_CLASSES:
                continue
            cx, cy, bw, bh = box.tolist()
            gi = min(grid_w - 1, max(0, int(cx * grid_w)))
            gj = min(grid_h - 1, max(0, int(cy * grid_h)))
            best_a = 0
            for a in range(num_anchors):
                if obj[b, a, gj, gi] == 0:
                    best_a = a
                    break
            obj[b, best_a, gj, gi] = 1.0
            
            tx = cx * grid_w - gi
            ty = cy * grid_h - gj
            box_t[b, best_a, gj, gi] = torch.tensor([tx, ty, bw, bh], device=device)
            cls_t[b, best_a, gj, gi] = lab_i
    return obj, box_t, cls_t


def detection_loss(preds, boxes_list, labels_list, num_classes):
    """
    Detection multi-task loss (Objectness + Box Regression + Classification).
    """
    B, A, H, W, _ = preds.shape
    device = preds.device
    obj_t, box_t, cls_t = build_det_targets(boxes_list, labels_list, A, H, W, device)

    pos_weight = torch.tensor([15.0], device=device)
    obj_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(preds[..., 4], obj_t)
    pos = obj_t > 0.5
    
    if pos.any():
        pred_raw = preds[..., :4][pos]
        pred_tx_ty = torch.sigmoid(pred_raw[:, :2])
        pred_bw_bh = torch.sigmoid(pred_raw[:, 2:4])
        pred_box_encoded = torch.cat([pred_tx_ty, pred_bw_bh], dim=-1)
        
        box_loss = nn.SmoothL1Loss()(pred_box_encoded, box_t[pos])
        cls_logits = preds[..., 5:][pos].reshape(-1, num_classes)
        cls_targets = cls_t[pos].reshape(-1)
        cls_loss = nn.CrossEntropyLoss()(cls_logits, cls_targets)
    else:
        box_loss = preds.new_tensor(0.0)
        cls_loss = preds.new_tensor(0.0)

    total_loss = 1.0 * obj_loss + 20.0 * box_loss + 1.0 * cls_loss
    return total_loss, obj_loss.item(), box_loss.item(), cls_loss.item()


# ==============================================================================
# DATASETS WITH EXPERIENCE REPLAY BUFFER & AUGMENTATION
# ==============================================================================

class DynamicOCRDataset(Dataset):
    """
    Collected character images labeled with the current multilingual vocabulary.
    """
    def __init__(self, feedback_jsonl, base_jsonl=None, char_to_idx=None, target_size=(64, 64)):
        self.target_h, self.target_w = target_size
        self.char_to_idx = char_to_idx or {}
        self.samples = []
        
        sources = [feedback_jsonl]
        if base_jsonl and base_jsonl != feedback_jsonl:
            sources.append(base_jsonl)

        for src in sources:
            if not os.path.exists(src):
                continue
            crops_dir = os.path.join(os.path.dirname(os.path.abspath(src)), "crops")
            with open(src, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if item.get("verified") is not True or item.get("split", "train") != "train":
                            continue
                        ch = item.get("char") or item.get("text", "")
                        img_path = item.get("image_path", "")
                        if not os.path.exists(img_path):
                            cand = os.path.join(crops_dir, os.path.basename(img_path))
                            if os.path.exists(cand):
                                img_path = cand
                        if ch in self.char_to_idx and os.path.exists(img_path):
                            self.samples.append({
                                "image_path": img_path,
                                "label": self.char_to_idx[ch],
                                "char": ch
                            })
                    except Exception as e:
                        logger.debug("Failed parsing character line: %s", e)
                        
        self.transform = T.Compose([
            T.Resize((self.target_h, self.target_w)),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        try:
            with Image.open(item["image_path"]) as im:
                img = im.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Cannot read verified OCR image: {item['image_path']}") from exc
            
        img_tensor = self.transform(img)
        label = item["label"]
        return img_tensor, torch.tensor(label, dtype=torch.long)


def build_ocr_training_loader(base_dir, char_to_idx, batch_size=32, max_batches=None):
    """Mix permanent training images and live corrections without pruning either source."""
    if batch_size < 1 or (max_batches is not None and max_batches < 1):
        raise ValueError("batch_size and max_batches must be positive")

    feedback = DynamicOCRDataset(
        os.path.join(base_dir, "self_learning_data", "ocr", "feedback_ocr.jsonl"),
        char_to_idx=char_to_idx,
    )
    archive_path = os.path.join(base_dir, CHARACTER_ARCHIVE_RELATIVE_PATH)
    permanent = (
        AlphanumericArchiveDataset(archive_path, char_to_idx, split="train", augment=True)
        if os.path.isfile(archive_path) else None
    )
    counts = {"permanent": len(permanent) if permanent is not None else 0,
              "feedback": len(feedback)}
    sources = [source for source in (permanent, feedback)
               if source is not None and len(source) > 0]
    if not sources:
        return None, counts

    if len(sources) == 1:
        src = sources[0]
        if max_batches is not None:
            sample_count = min(len(src), max_batches * batch_size)
            sampler = RandomSampler(src, replacement=(sample_count > len(src)), num_samples=sample_count)
            return DataLoader(src, batch_size=batch_size, sampler=sampler), counts
        return DataLoader(src, batch_size=batch_size, shuffle=True), counts

    dataset = ConcatDataset(sources)
    # Give each source equal probability; a large archive must not drown corrections.
    weights = torch.cat([
        torch.full((len(source),), 1.0 / len(source), dtype=torch.double)
        for source in sources
    ])
    sample_count = len(dataset)
    if max_batches is not None:
        sample_count = min(sample_count, max_batches * batch_size)
    sampler = WeightedRandomSampler(weights, num_samples=sample_count, replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler), counts


class DynamicInpaintDataset(Dataset):
    """
    Combined Inpainting dataset of real user text-removal patches and screentone backgrounds.
    Includes data augmentation (random horizontal flip) and validation.
    """
    def __init__(self, feedback_jsonl, base_jsonl=None, target_size=(256, 256)):
        self.target_h, self.target_w = target_size
        self.samples = []
        
        sources = [feedback_jsonl]
        if base_jsonl and base_jsonl != feedback_jsonl:
            sources.append(base_jsonl)

        for src in sources:
            if not os.path.exists(src):
                continue
            inp_dir = os.path.dirname(os.path.abspath(src))
            with open(src, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if (item.get("verified") is not True or item.get("split", "train") != "train"
                                or item.get("ground_truth_source") not in {"human_cleaned", "paired_original"}):
                            continue
                        rp = item.get("raw_image", "")
                        mp = item.get("mask", "")
                        cp = item.get("ground_truth", "")
                        if not os.path.exists(rp):
                            cand = os.path.join(inp_dir, "raw", os.path.basename(rp))
                            if os.path.exists(cand): rp = item["raw_image"] = cand
                        if not os.path.exists(mp):
                            cand = os.path.join(inp_dir, "mask", os.path.basename(mp))
                            if os.path.exists(cand): mp = item["mask"] = cand
                        if not os.path.exists(cp):
                            cand = os.path.join(inp_dir, "clean", os.path.basename(cp))
                            if os.path.exists(cand): cp = item["ground_truth"] = cand
                        if os.path.exists(rp) and os.path.exists(mp) and os.path.exists(cp):
                            self.samples.append(item)
                    except Exception as e:
                        logger.debug("Failed parsing inpaint line: %s", e)
                        
        self.resize = T.Resize((self.target_h, self.target_w))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        try:
            raw_img = Image.open(item["raw_image"]).convert("RGB")
            mask_img = Image.open(item["mask"]).convert("L")
            clean_img = Image.open(item["ground_truth"]).convert("RGB")
            
            # Data augmentation: Random horizontal flip
            if np.random.rand() > 0.5:
                raw_img = raw_img.transpose(Image.FLIP_LEFT_RIGHT)
                mask_img = mask_img.transpose(Image.FLIP_LEFT_RIGHT)
                clean_img = clean_img.transpose(Image.FLIP_LEFT_RIGHT)
                
            raw_img = self.resize(raw_img)
            mask_img = self.resize(mask_img)
            clean_img = self.resize(clean_img)
        except Exception as e:
            raise RuntimeError("Cannot read verified inpainting triplet") from e

        return {
            "raw": self.to_tensor(raw_img),
            "mask": (self.to_tensor(mask_img) > 0.5).float(),
            "clean": self.to_tensor(clean_img),
            "triplet_id": f"dyn_inp_{idx}"
        }


class DynamicDetectionDataset(Dataset):
    """
    Combined Detection dataset of collected manga pages and bubble bounding boxes.
    """
    def __init__(self, feedback_jsonl, target_size=(384, 256)):
        self.target_h, self.target_w = target_size
        self.samples = []
        
        if os.path.exists(feedback_jsonl):
            det_dir = os.path.dirname(os.path.abspath(feedback_jsonl))
            with open(feedback_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if (item.get("verified") is not True or item.get("split", "train") != "train"
                                or item.get("annotations_verified") is not True):
                            continue
                        p = item.get("image_path", "")
                        if not os.path.exists(p):
                            cand = os.path.join(det_dir, "pages", os.path.basename(p))
                            if os.path.exists(cand): p = item["image_path"] = cand
                        if os.path.exists(p) and item.get("annotations"):
                            self.samples.append(item)
                    except Exception as e:
                        logger.debug("Failed parsing detection line: %s", e)
                        
        self.transform = T.Compose([
            T.Resize((self.target_h, self.target_w)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = item.get("image_path", "")
        
        try:
            with Image.open(img_path) as im:
                orig_w, orig_h = float(im.size[0]), float(im.size[1])
                img = im.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Cannot read verified detection image: {img_path}") from exc
            
        img_tensor = self.transform(img)
        
        boxes = []
        labels = []
        scale_x = 1.0 / max(1.0, orig_w)
        scale_y = 1.0 / max(1.0, orig_h)
        
        for ann in item.get("annotations", []):
            cat = ann.get("category", "speech_bubble")
            cat_id = CATEGORY_TO_IDX.get(cat, 0)
            b = ann.get("bbox", [])
            if len(b) == 4:
                bbox_format = ann.get("bbox_format", "xyxy")
                if bbox_format == "xywh":
                    bx, by, bw_raw, bh_raw = [float(v) for v in b]
                    x1, y1 = bx, by
                    x2, y2 = bx + bw_raw, by + bh_raw
                else:
                    x1, y1, x2, y2 = [float(v) for v in b]

                x1 = max(0.0, min(orig_w, x1))
                y1 = max(0.0, min(orig_h, y1))
                x2 = max(0.0, min(orig_w, x2))
                y2 = max(0.0, min(orig_h, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                cx = ((x1 + x2) / 2.0) * scale_x
                cy = ((y1 + y2) / 2.0) * scale_y
                bw = (x2 - x1) * scale_x
                bh = (y2 - y1) * scale_y
                
                cx = min(1.0, max(0.0, cx))
                cy = min(1.0, max(0.0, cy))
                bw = min(1.0, max(0.001, bw))
                bh = min(1.0, max(0.001, bh))
                
                boxes.append([cx, cy, bw, bh])
                labels.append(cat_id)
                
        if not boxes:
            boxes = [[0.5, 0.5, 0.01, 0.01]]
            labels = [0]
            
        return {
            "image": img_tensor,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "page_id": item.get("page_id", f"dyn_page_{idx}")
        }


# ==============================================================================
# SELF-LEARNING TRAINER ENGINE
# ==============================================================================

class SelfLearningTrainer:
    """
    Autonomous Model Continual Learning Engine:
    - Incremental learning for OCR, Inpainting, and Detection
    - Dynamic linear layer expansion if vocabulary increases
    - Multi-threaded background execution with live UI callback
    - Automatic checkpoint backup and versioning
    - CosineAnnealingLR scheduling and gradient clipping for stability
    """
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or BASE_DIR
        self.checkpoint_dir = os.path.join(self.base_dir, "checkpoints")
        self.data_dir = os.path.join(self.base_dir, "self_learning_data")
        self.vocab_path = os.path.join(self.base_dir, "ocr_vocab.json")
        self.stats_file = os.path.join(self.data_dir, "stats.json")
        
        self.is_training = False
        self.lock = threading.RLock()
        self._train_lock = threading.Lock()
        self.current_log = []
        self.promotion_results = {}

    def _prepare_validation(self, task, model, dataset, checkpoint_path, char_to_idx, log_callback):
        gate = None
        try:
            if not os.path.isfile(checkpoint_path):
                raise ValidationUnavailable("No incumbent checkpoint; initialize a reviewed baseline before self-learning")
            expected_digest = file_digest(checkpoint_path)
            gate = ValidationGate(self.base_dir, task, char_to_idx, dataset)
            baseline = gate.evaluate(model, DEVICE)
            return gate, baseline, expected_digest
        except Exception as exc:
            if gate is not None:
                gate.close()
            reason = f"{task}: training skipped; incumbent unchanged. {exc}"
            self.promotion_results[task] = {"promoted": False, "reason": reason}
            logger.warning(reason)
            if log_callback:
                log_callback(reason)
            return None

    def _promote_candidate(self, task, model, checkpoint_path, validation, log_callback):
        gate, baseline, expected_digest = validation
        result = {"promoted": False, "baseline": baseline, "validation_counts": gate.counts}
        try:
            candidate = gate.evaluate(model, DEVICE)
            result["candidate"] = candidate
            if should_promote(baseline, candidate):
                atomic_promote(model, checkpoint_path, expected_digest)
                result.update(promoted=True, reason="Passed held-out regression gate")
            else:
                result["reason"] = "Candidate did not improve every required held-out check without regression; incumbent retained"
        except Exception as exc:
            result["reason"] = f"Validation/promotion failed; incumbent retained: {exc}"
        finally:
            gate.close()
        self.promotion_results[task] = result
        message = f"{task}: {result['reason']}"
        logger.info(message)
        if log_callback:
            log_callback(message)
        return result["promoted"]

    def _sync_ocr_head(self, model, new_vocab_size):
        """
        Dynamically adjusts OCR output layer if new characters were added to vocabulary.
        Preserves existing weights for known tokens across both character and sequence heads!
        """
        old_head = model.fc[-1]
        old_vocab_size = old_head.out_features
        if old_vocab_size == new_vocab_size:
            return model
            
        logger.info("[Self-Training] Expanding OCR Head: %d -> %d classes", old_vocab_size, new_vocab_size)
        hidden_dim = old_head.in_features
        new_head = nn.Linear(hidden_dim, new_vocab_size).to(DEVICE)
        
        # Copy existing weights for character head
        with torch.no_grad():
            min_c = min(old_vocab_size, new_vocab_size)
            new_head.weight[:min_c] = old_head.weight[:min_c]
            new_head.bias[:min_c] = old_head.bias[:min_c]
            
        model.fc[-1] = new_head
        
        # Also sync sequence head if present
        if hasattr(model, "seq_fc"):
            seq_old_vocab = model.seq_fc.out_features
            if seq_old_vocab != new_vocab_size:
                new_seq_fc = nn.Linear(model.seq_fc.in_features, new_vocab_size).to(DEVICE)
                with torch.no_grad():
                    min_s = min(seq_old_vocab, new_vocab_size)
                    new_seq_fc.weight[:min_s] = model.seq_fc.weight[:min_s]
                    new_seq_fc.bias[:min_s] = model.seq_fc.bias[:min_s]
                model.seq_fc = new_seq_fc
                
        model.vocab_size = new_vocab_size
        return model

    def fine_tune_ocr(self, epochs=2, lr=5e-4, batch_size=32, max_batches=60, log_callback=None):
        """
        Fine-tunes the character classifier using permanent images and live feedback.
        """
        msg = "🧠 [1/3] OCR Modelini Öqitiş (Individual Character CNN)..."
        logger.info(msg)
        print(msg, flush=True)
        if log_callback: log_callback(msg)
        
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        char_to_idx = vocab_data["char_to_idx"]
        vocab_size = max(int(v) for v in char_to_idx.values()) + 1
        
        loader, source_counts = build_ocr_training_loader(
            self.base_dir, char_to_idx, batch_size=batch_size, max_batches=max_batches
        )
        if loader is None:
            msg = "OCR uçun yangi namunalar topilmadi, ötkazib yuborildi."
            if log_callback: log_callback(msg)
            return None, None
            
        dataset = loader.dataset
        source_msg = (
            f"OCR bazasi: {source_counts['permanent']} ta doimiy öquv rasmi + "
            f"{source_counts['feedback']} ta yiğilgan namuna."
        )
        logger.info(source_msg)
        if log_callback:
            log_callback(source_msg)
        
        ckpt_path = os.path.join(self.checkpoint_dir, "best_manga_ocr.pt")
        state = None
        ckpt_vocab_size = vocab_size
        if os.path.exists(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
                for k in ("fc.4.weight", "seq_fc.weight", "fc.weight"):
                    if k in state:
                        ckpt_vocab_size = state[k].shape[0]
                        break
            except Exception as e:
                raise ValidationUnavailable(f"OCR incumbent could not load; refusing replacement: {e}") from e

        model = MangaOCRModel(vocab_size=ckpt_vocab_size, hidden_dim=128).to(DEVICE)
        if state is not None:
            model.load_checkpoint_weights(state)
        model = self._sync_ocr_head(model, vocab_size)
        validation = self._prepare_validation("ocr", model, dataset, ckpt_path, char_to_idx, log_callback)
        if validation is None:
            return None, None
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
        model.train()
        total_loss = 0.0
        
        for ep in range(1, epochs + 1):
            ep_loss = 0.0
            correct = 0
            total = 0
            steps = 0
            for images, labels in loader:
                if max_batches is not None and steps >= max_batches:
                    break
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()
                    ep_loss += loss.item()
                    
                    preds = torch.argmax(logits, dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    steps += 1
                    
            scheduler.step()
            avg_ep_loss = ep_loss / max(1, steps)
            acc = (correct / max(1, total)) * 100.0
            total_loss += avg_ep_loss
            log_line = f"  -> OCR Epoxa [{ep}/{epochs}] | Loss: {avg_ep_loss:.4f} | Aniqlik: {acc:.2f}% | Namunalar: {len(dataset)}"
            logger.info(log_line)
            print(log_line, flush=True)
            if log_callback: log_callback(log_line)
            
        final_loss = total_loss / epochs
        promoted = self._promote_candidate("ocr", model, ckpt_path, validation, log_callback)
        return (model if promoted else None), final_loss

    def fine_tune_inpainting(self, epochs=2, lr=1e-4, batch_size=4, max_batches=25, log_callback=None):
        """
        Fine-tunes MangaInpaintingModel on collected cleaned user inpainting triplets.
        """
        msg = "🎨 [2/3] Inpainting Modelini Öqitiş (Fine-Tuning U-Net)..."
        logger.info(msg)
        if log_callback: log_callback(msg)
        
        feedback_jsonl = os.path.join(self.data_dir, "inpainting", "feedback_inpaint.jsonl")
        dataset = DynamicInpaintDataset(feedback_jsonl)
        if len(dataset) == 0:
            msg = "Inpainting uçun yangi namunalar topilmadi, ötkazib yuborildi."
            if log_callback: log_callback(msg)
            return None, None

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = MangaInpaintingModel().to(DEVICE)
        ckpt_path = os.path.join(self.checkpoint_dir, "best_manga_inpaint.pt")
        if os.path.exists(ckpt_path):
            try:
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
            except Exception as e:
                raise ValidationUnavailable(f"Inpainting incumbent could not load; refusing replacement: {e}") from e

        validation = self._prepare_validation("inpainting", model, dataset, ckpt_path, {}, log_callback)
        if validation is None:
            return None, None
                
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        model.train()
        total_loss = 0.0
        
        for ep in range(1, epochs + 1):
            ep_loss = 0.0
            steps = 0
            for batch in loader:
                if max_batches is not None and steps >= max_batches:
                    break
                raw = batch["raw"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)
                clean = batch["clean"].to(DEVICE)
                
                optimizer.zero_grad()
                pred, composite = model(raw, mask)
                
                eps = 1e-6
                mask_3c = mask.expand(-1, 3, -1, -1) if mask.shape[1] == 1 else mask
                l1_valid = (torch.abs(pred - clean) * (1.0 - mask_3c)).sum() / ((1.0 - mask_3c).sum() + eps)
                l1_hole = (torch.abs(pred - clean) * mask_3c).sum() / (mask_3c.sum() + eps)
                loss = l1_valid + 6.0 * l1_hole

                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()
                    ep_loss += loss.item()
                    steps += 1

            scheduler.step()
            avg_ep_loss = ep_loss / max(1, steps)
            total_loss += avg_ep_loss
            log_line = f"  -> Inpainting Epoxa [{ep}/{epochs}] | Loss: {avg_ep_loss:.4f} | Namunalar: {len(dataset)}"
            logger.info(log_line)
            if log_callback: log_callback(log_line)
            
        final_loss = total_loss / epochs
        promoted = self._promote_candidate("inpainting", model, ckpt_path, validation, log_callback)
        return (model if promoted else None), final_loss

    def fine_tune_detector(self, epochs=2, lr=1e-4, batch_size=4, max_batches=20, log_callback=None):
        """
        Fine-tunes MangaBubbleDetector on collected and cleaned manga pages.
        """
        msg = "📐 [3/3] Detection Modelini Öqitiş (Bubble Detector)..."
        logger.info(msg)
        if log_callback: log_callback(msg)
        
        feedback_jsonl = os.path.join(self.data_dir, "detection", "feedback_detection.jsonl")
        dataset = DynamicDetectionDataset(feedback_jsonl)
        if len(dataset) == 0:
            msg = "Detection uçun yangi namunalar topilmadi, ötkazib yuborildi."
            if log_callback: log_callback(msg)
            return None, None

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=detection_collate_fn)

        model = MangaBubbleDetector(num_classes=4).to(DEVICE)
        ckpt_path = os.path.join(self.checkpoint_dir, "best_manga_detector.pt")
        if os.path.exists(ckpt_path):
            try:
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
            except Exception as e:
                raise ValidationUnavailable(f"Detector incumbent could not load; refusing replacement: {e}") from e

        validation = self._prepare_validation("detector", model, dataset, ckpt_path, {}, log_callback)
        if validation is None:
            return None, None
                
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        model.train()
        total_loss = 0.0
        
        for ep in range(1, epochs + 1):
            ep_loss = 0.0
            steps = 0
            for batch in loader:
                if max_batches is not None and steps >= max_batches:
                    break
                images = batch["images"].to(DEVICE)
                boxes = batch["boxes"]
                labels = batch["labels"]
                
                optimizer.zero_grad()
                preds = model(images)
                loss, _, _, _ = detection_loss(preds, boxes, labels, num_classes=4)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()
                    ep_loss += loss.item()
                    steps += 1

            scheduler.step()
            avg_ep_loss = ep_loss / max(1, steps)
            total_loss += avg_ep_loss
            log_line = f"  -> Detection Epoxa [{ep}/{epochs}] | Loss: {avg_ep_loss:.4f} | Sahifalar: {len(dataset)}"
            logger.info(log_line)
            if log_callback: log_callback(log_line)
            
        final_loss = total_loss / epochs
        promoted = self._promote_candidate("detector", model, ckpt_path, validation, log_callback)
        return (model if promoted else None), final_loss

    def execute_self_training_cycle(self, epochs=2, max_batches=30, log_callback=None, pipeline_ref=None):
        """Train only eligible candidates; publish only independently validated ones."""
        if epochs < 1 or max_batches < 1:
            raise ValueError("epochs and max_batches must be positive")
        with self._train_lock:
            with self.lock:
                self.is_training = True
                self.current_log = []
                self.promotion_results = {}
                start_time = time.time()

                def report(message):
                    self.current_log.append(message)
                    if log_callback:
                        log_callback(message)

                try:
                    losses = {}
                    for task, train in (("ocr", self.fine_tune_ocr),
                                        ("inpainting", self.fine_tune_inpainting),
                                        ("detector", self.fine_tune_detector)):
                        try:
                            _, loss = train(epochs=epochs, max_batches=max_batches, log_callback=report)
                            losses[task] = loss
                            self.promotion_results.setdefault(task, {
                                "promoted": False, "reason": "No eligible verified training samples"})
                        except Exception as exc:
                            losses[task] = None
                            self.promotion_results[task] = {"promoted": False, "reason": str(exc)}
                            report(f"{task}: stopped; incumbent retained. {exc}")
                    promoted = [task for task, result in self.promotion_results.items() if result["promoted"]]
                    reload_ok = True
                    if promoted and pipeline_ref is not None:
                        try:
                            pipeline_ref.reload_models()
                        except Exception as exc:
                            reload_ok = False
                            report(f"Checkpoint saved, but live reload failed: {exc}")

                    collector_lock = getattr(getattr(self, "collector", None), "_lock", None)
                    def _update_stats():
                        try:
                            with open(self.stats_file, encoding="utf-8") as stream:
                                stats = json.load(stream)
                        except (OSError, ValueError):
                            stats = {}
                        stats["total_training_attempts"] = stats.get("total_training_attempts", 0) + 1
                        if promoted:
                            stats["total_training_cycles"] = stats.get("total_training_cycles", 0) + 1
                            stats["unprocessed_pages_since_last_train"] = 0
                            stats["last_trained_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                            stats["last_train_loss"] = losses
                        stats["last_promotion_results"] = self.promotion_results
                        atomik_json_yoz(self.stats_file, stats)
                        return stats

                    if collector_lock:
                        with collector_lock:
                            stats = _update_stats()
                    else:
                        stats = _update_stats()
                    history_path = os.path.join(self.base_dir, "training_metrics", "self_learning_history.json")
                    try:
                        with open(history_path, encoding="utf-8") as stream:
                            history = json.load(stream)
                    except (OSError, ValueError):
                        history = []
                    elapsed = time.time() - start_time
                    history.append({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "cycle": stats.get("total_training_cycles", 0),
                                    "training_losses": losses, "promotion_results": self.promotion_results,
                                    "promoted_tasks": promoted, "elapsed_seconds": round(elapsed, 2)})
                    atomik_json_yoz(history_path, history)
                    report("Updated validated models: " + (", ".join(promoted) if promoted else "none; all incumbents retained"))
                    return {"success": bool(promoted) and reload_ok,
                            "cycle": stats.get("total_training_cycles", 0), "elapsed": elapsed,
                            "promoted_tasks": promoted, "promotion_results": self.promotion_results,
                            "logs": "\n".join(self.current_log)}
                finally:
                    self.is_training = False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    trainer = SelfLearningTrainer()
    logger.info("Testing SelfLearningTrainer on cleaned datasets...")
    res = trainer.execute_self_training_cycle(epochs=1)
    logger.info("Execution Result: %s", res["success"])
