import os
import sys
import copy
import json
import math
import time
import random
import argparse
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import ml_paths
ml_paths.ensure_ml_paths()

from models import MangaBubbleDetector, MangaOCRModel, MangaInpaintingModel
from dataset_loader import (
    MangaDetectionDataset, MangaOCRDataset, MangaInpaintingDataset,
    AlphanumericArchiveDataset, CHARACTER_ARCHIVE_RELATIVE_PATH,
    ocr_collate_fn, detection_collate_fn, CATEGORY_TO_IDX, NUM_DET_CLASSES
)
from validation_gate import (
    ValidationGate, ValidationUnavailable, atomic_promote, file_digest, should_promote
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Optimize CPU multithreading when CUDA is not present.
if not torch.cuda.is_available():
    num_threads = min(4, os.cpu_count() or 4)
    torch.set_num_threads(num_threads)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "manga_ml_datasets")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
METRICS_DIR = os.path.join(BASE_DIR, "training_metrics")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def set_seed(seed=42):
    """Ensure complete reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_levenshtein(s1, s2):
    """Compute Character-level Levenshtein edit distance."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def decode_ctc(log_probs_tb, idx_to_char):
    """
    Standard Greedy CTC decode for a single sample: log_probs_tb is (T, C).
    Takes raw argmax across classes, collapses consecutive duplicates,
    and removes blank token (0) and <unk> token (1).
    """
    preds_idx = log_probs_tb.argmax(dim=-1)
    toks = []
    prev = None
    for t in preds_idx.cpu().numpy():
        t = int(t)
        if t != prev and t > 1:  # collapse repeats and discard CTC blank (0) and <unk> (1)
            toks.append(t)
        prev = t
    return "".join(idx_to_char.get(t, "") for t in toks)


def masked_l1(pred, target, mask, eps=1e-6):
    """
    Mean absolute error strictly over region designated by mask.
    Handles 1-channel mask broadcasting across 3-channel RGB images correctly.
    """
    if mask.dim() == 4 and mask.shape[1] == 1 and pred.shape[1] == 3:
        mask_3c = mask.expand(-1, 3, -1, -1)
    else:
        mask_3c = mask
    abs_err = torch.abs(pred - target) * mask_3c
    return abs_err.sum() / (mask_3c.sum() + eps)


def compute_box_iou(box1, box2):
    """
    Compute pairwise IoU between two tensors of normalized [cx, cy, w, h] boxes.
    Output range: [0.0, 1.0].
    """
    x1_1 = box1[..., 0] - box1[..., 2] / 2.0
    y1_1 = box1[..., 1] - box1[..., 3] / 2.0
    x2_1 = box1[..., 0] + box1[..., 2] / 2.0
    y2_1 = box1[..., 1] + box1[..., 3] / 2.0

    x1_2 = box2[..., 0] - box2[..., 2] / 2.0
    y1_2 = box2[..., 1] - box2[..., 3] / 2.0
    x2_2 = box2[..., 0] + box2[..., 2] / 2.0
    y2_2 = box2[..., 1] + box2[..., 3] / 2.0

    xi1 = torch.max(x1_1, x1_2)
    yi1 = torch.max(y1_1, y1_2)
    xi2 = torch.min(x2_1, x2_2)
    yi2 = torch.min(y2_1, y2_2)

    inter_w = torch.clamp(xi2 - xi1, min=0.0)
    inter_h = torch.clamp(yi2 - yi1, min=0.0)
    inter_area = inter_w * inter_h

    area1 = torch.clamp(x2_1 - x1_1, min=0.0) * torch.clamp(y2_1 - y1_1, min=0.0)
    area2 = torch.clamp(x2_2 - x1_2, min=0.0) * torch.clamp(y2_2 - y1_2, min=0.0)
    union_area = area1 + area2 - inter_area

    return inter_area / torch.clamp(union_area, min=1e-6)


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
            
            # Local offset relative to grid cell
            tx = cx * grid_w - gi
            ty = cy * grid_h - gj
            box_t[b, best_a, gj, gi] = torch.tensor([tx, ty, bw, bh], device=device)
            cls_t[b, best_a, gj, gi] = lab_i
    return obj, box_t, cls_t


def detection_loss(preds, boxes_list, labels_list, num_classes):
    """
    Multi-task object detection loss:
    1. Weighted BCEWithLogits for objectness
    2. SmoothL1 on bounding box offsets [tx, ty, bw, bh]
    3. CrossEntropy for multi-class classification
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
        with torch.no_grad():
            pos_indices = pos.nonzero(as_tuple=False)  # (N, 4) -> [b, a, gj, gi]
            gj = pos_indices[:, 2].float()
            gi = pos_indices[:, 3].float()
            pred_cx = (gi + pred_tx_ty[:, 0]) / float(W)
            pred_cy = (gj + pred_tx_ty[:, 1]) / float(H)
            pred_global = torch.stack([pred_cx, pred_cy, pred_bw_bh[:, 0], pred_bw_bh[:, 1]], dim=-1)
            tgt = box_t[pos]
            tgt_cx = (gi + tgt[:, 0]) / float(W)
            tgt_cy = (gj + tgt[:, 1]) / float(H)
            tgt_global = torch.stack([tgt_cx, tgt_cy, tgt[:, 2], tgt[:, 3]], dim=-1)
            mean_iou = compute_box_iou(pred_global, tgt_global).mean().item()
    else:
        box_loss = preds.new_tensor(0.0)
        cls_loss = preds.new_tensor(0.0)
        mean_iou = 0.0

    total_loss = 1.0 * obj_loss + 20.0 * box_loss + 1.0 * cls_loss
    return total_loss, obj_loss.item(), box_loss.item(), cls_loss.item(), mean_iou


# ==============================================================================
# VALIDATION GATE & ATOMIC CHECKPOINT PROMOTION
# ==============================================================================

def _save_atomic_state(state, dest_path):
    dest_dir = os.path.dirname(os.path.abspath(dest_path))
    os.makedirs(dest_dir, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=dest_dir, suffix=".pt", delete=False) as tf:
            tmp_name = tf.name
            torch.save(state, tf)
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(tmp_name, dest_path)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _promote_with_validation_gate(
    model,
    incumbent_factory,
    best_state,
    task,
    ckpt_path,
    base_dir,
    char_to_idx,
    train_dataset,
    min_val_samples=10,
    allow_bootstrap=False,
):
    """Apply ValidationGate before overwriting any checkpoint."""
    if best_state is not None:
        model.load_state_dict(best_state)
    state = model.state_dict()
    if any(not torch.isfinite(v).all() for v in state.values() if v.is_floating_point() or v.is_complex()):
        raise ValidationUnavailable(f"{task}: candidate contains non-finite weights")

    candidate_path = os.path.join(os.path.dirname(ckpt_path), f"candidate_manga_{task}.pt")
    incumbent_exists = os.path.isfile(ckpt_path)
    gate = None
    try:
        gate = ValidationGate(base_dir, task, char_to_idx or {}, train_dataset, min_samples=min_val_samples)
        candidate_metrics = gate.evaluate(model, DEVICE)
        if incumbent_exists:
            incumbent_digest = file_digest(ckpt_path)
            incumbent = incumbent_factory().to(DEVICE)
            incumbent_state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            if hasattr(incumbent, "load_checkpoint_weights") and task == "ocr":
                incumbent.load_checkpoint_weights(incumbent_state)
            else:
                incumbent.load_state_dict(incumbent_state)
            baseline_metrics = gate.evaluate(incumbent, DEVICE)
            if should_promote(baseline_metrics, candidate_metrics):
                atomic_promote(model, ckpt_path, incumbent_digest)
                print(f"[{task.upper()}] Promoted via ValidationGate: {baseline_metrics} -> {candidate_metrics}")
                return {"promoted": True, "baseline": baseline_metrics, "candidate": candidate_metrics, "path": ckpt_path}
            _save_atomic_state(state, candidate_path)
            print(f"[{task.upper()}] ValidationGate rejected promotion (incumbent retained): baseline={baseline_metrics}, candidate={candidate_metrics}")
            return {"promoted": False, "baseline": baseline_metrics, "candidate": candidate_metrics, "path": candidate_path}
        _save_atomic_state(state, ckpt_path)
        print(f"[{task.upper()}] Initial checkpoint published via ValidationGate: {candidate_metrics}")
        return {"promoted": True, "baseline": None, "candidate": candidate_metrics, "path": ckpt_path}
    except ValidationUnavailable as exc:
        if not incumbent_exists and allow_bootstrap:
            _save_atomic_state(state, ckpt_path)
            print(f"[{task.upper()}] Bootstrapped new checkpoint (no prior incumbent; {exc})")
            return {"promoted": True, "reason": str(exc), "path": ckpt_path}
        _save_atomic_state(state, candidate_path)
        print(f"[{task.upper()}] ValidationGate blocked overwrite ({exc}); incumbent preserved, candidate saved to {candidate_path}")
        return {"promoted": False, "reason": str(exc), "path": candidate_path}
    finally:
        if gate is not None:
            gate.close()


# ==============================================================================
# 1. OCR TRAINING & EVALUATION (CRNN + CTC Loss + Character Head)
# ==============================================================================

def train_ocr_model(
    epochs=3,
    batch_size=16,
    lr=5e-4,
    max_batches=None,
    val_batches=None,
    from_scratch=False,
    base_dir=BASE_DIR,
    data_dir=DATA_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
    vocab_file=None,
    min_val_samples=10,
    allow_bootstrap=False,
    char_loss_weight=0.5,
):
    set_seed(42)
    print("\n" + "=" * 60)
    print("STARTING TRAINING: Manga OCR Model (CRNN + CTC + Char Head)")
    print("=" * 60)

    vocab_file = vocab_file or ml_paths.resolve_vocab_path(base_dir)
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"Missing required vocabulary file: {vocab_file}")

    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    char_to_idx = vocab_data["char_to_idx"]
    idx_to_char = {int(k): v for k, v in vocab_data["idx_to_char"].items()}
    vocab_size = max(int(v) for v in char_to_idx.values()) + 1

    ocr_img_dir = os.path.join(data_dir, "ocr", "crops")
    if not os.path.isdir(ocr_img_dir):
        ocr_img_dir = os.path.join(data_dir, "ocr", "images")
    if not os.path.isdir(ocr_img_dir):
        ocr_img_dir = None

    train_dataset = MangaOCRDataset(
        os.path.join(data_dir, "ocr", "train.jsonl"), char_to_idx, img_dir=ocr_img_dir, is_train=True
    )
    val_dataset = MangaOCRDataset(
        os.path.join(data_dir, "ocr", "val.jsonl"), char_to_idx, img_dir=ocr_img_dir, is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=ocr_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=ocr_collate_fn)

    # Jointly load AlphanumericArchiveDataset so the character head (fc) is actively trained
    char_train_dataset = None
    char_val_dataset = None
    char_loader = None
    char_archive = os.path.join(base_dir, CHARACTER_ARCHIVE_RELATIVE_PATH)
    if os.path.isfile(char_archive):
        try:
            char_train_dataset = AlphanumericArchiveDataset(char_archive, char_to_idx, split="train", augment=True)
            char_val_dataset = AlphanumericArchiveDataset(char_archive, char_to_idx, split="val", augment=False)
            char_loader = DataLoader(char_train_dataset, batch_size=32, shuffle=True)
            print(f"Character head training enabled: {len(char_train_dataset)} train / {len(char_val_dataset)} val crops")
        except Exception as exc:
            print(f"Character archive unavailable for joint fc training ({exc})")

    def _make_model():
        return MangaOCRModel(vocab_size=vocab_size, hidden_dim=128)

    model = _make_model().to(DEVICE)
    ckpt_path = os.path.join(checkpoint_dir, "best_manga_ocr.pt")

    if not from_scratch and os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            model.load_checkpoint_weights(state)
            print("Loaded pre-trained OCR checkpoint for warm-start fine-tuning.")
        except Exception as e:
            print(f"Starting OCR with freshly initialized weights ({e})")
    else:
        with torch.no_grad():
            if hasattr(model, "seq_fc") and hasattr(model.seq_fc, "bias") and model.seq_fc.bias is not None:
                model.seq_fc.bias[0] = -3.0
            if hasattr(model, "fc") and len(model.fc) >= 5:
                model.fc[4].bias[0] = -3.0

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    char_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=5e-5)

    history = {"train_loss": [], "val_loss": [], "val_cer": [], "val_char_acc": []}
    best_cer = float("inf")
    best_state = None
    fixed_val_batches = len(val_loader) if (val_batches is None and max_batches is None) else min(val_batches or max_batches, len(val_loader))

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            total_train_loss = 0.0
            train_steps = 0
            start_time = time.time()
            char_iter = iter(char_loader) if char_loader is not None else None

            for b_idx, batch in enumerate(train_loader):
                if max_batches is not None and b_idx >= max_batches:
                    break
                images = batch["images"].to(DEVICE)
                targets = batch["targets"].to(DEVICE)
                target_lens = batch["target_lens"].to(DEVICE)

                optimizer.zero_grad()
                log_probs = model(images, return_sequence=True)  # (T, B, C)
                T_steps, B, _ = log_probs.shape
                input_lens = torch.full(size=(B,), fill_value=T_steps, dtype=torch.long, device=DEVICE)

                ctc_loss = criterion(log_probs, targets, input_lens, target_lens)

                # Jointly train character classification head `fc`
                char_loss = ctc_loss.new_tensor(0.0)
                if char_iter is not None:
                    try:
                        c_imgs, c_lbls = next(char_iter)
                    except StopIteration:
                        char_iter = iter(char_loader)
                        c_imgs, c_lbls = next(char_iter)
                    c_logits = model(c_imgs.to(DEVICE), return_sequence=False)
                    char_loss = char_criterion(c_logits, c_lbls.to(DEVICE))
                else:
                    single_indices = [i for i, txt in enumerate(batch["texts"]) if len(txt.strip()) == 1]
                    if single_indices:
                        c_imgs = F.interpolate(images[single_indices], size=(64, 64), mode="bilinear", align_corners=False)
                        c_lbls = torch.tensor(
                            [char_to_idx.get(batch["texts"][i].strip(), char_to_idx.get("<unk>", 1)) for i in single_indices],
                            dtype=torch.long,
                            device=DEVICE,
                        )
                        c_logits = model(c_imgs, return_sequence=False)
                        char_loss = char_criterion(c_logits, c_lbls)

                loss = ctc_loss + char_loss_weight * char_loss
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    total_train_loss += loss.item()
                    train_steps += 1
                    if (b_idx + 1) % 15 == 0 or (b_idx + 1) == max_batches:
                        print(f"  [OCR Ep {epoch}] Step [{b_idx + 1}/{max_batches or len(train_loader)}] | Batch Loss: {loss.item():.4f}", flush=True)

            scheduler.step()
            avg_train_loss = total_train_loss / max(1, train_steps)

            model.eval()
            total_val_loss = 0.0
            val_steps = 0
            total_edit_distance = 0
            total_chars = 0
            char_correct = 0
            char_total = 0

            with torch.no_grad():
                for b_idx, batch in enumerate(val_loader):
                    if b_idx >= fixed_val_batches:
                        break
                    images = batch["images"].to(DEVICE)
                    targets = batch["targets"].to(DEVICE)
                    target_lens = batch["target_lens"].to(DEVICE)

                    log_probs = model(images, return_sequence=True)
                    T_steps, B, _ = log_probs.shape
                    input_lens = torch.full(size=(B,), fill_value=T_steps, dtype=torch.long, device=DEVICE)

                    v_loss = criterion(log_probs, targets, input_lens, target_lens)
                    if torch.isfinite(v_loss):
                        total_val_loss += v_loss.item()
                        val_steps += 1

                    for i in range(B):
                        ref_str = batch["texts"][i]
                        pred_str = decode_ctc(log_probs[:, i, :], idx_to_char)
                        total_edit_distance += compute_levenshtein(pred_str, ref_str)
                        total_chars += max(1, len(ref_str))

                if char_val_dataset is not None and len(char_val_dataset) > 0:
                    val_char_loader = DataLoader(char_val_dataset, batch_size=64, shuffle=False)
                    for cb_idx, (c_imgs, c_lbls) in enumerate(val_char_loader):
                        if cb_idx >= 8:
                            break
                        preds = model(c_imgs.to(DEVICE), return_sequence=False).argmax(dim=-1).cpu()
                        char_correct += int((preds == c_lbls).sum().item())
                        char_total += len(c_lbls)

            avg_val_loss = total_val_loss / max(1, val_steps)
            cer = (total_edit_distance / max(1, total_chars)) * 100.0
            char_acc = (char_correct / char_total) * 100.0 if char_total > 0 else 0.0
            elapsed = time.time() - start_time

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_cer"].append(cer)
            history["val_char_acc"].append(char_acc)

            print(
                f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | CER: {cer:.1f}% | Char Acc: {char_acc:.1f}% | Time: {elapsed:.1f}s",
                flush=True,
            )

            if val_steps > 0 and cer <= best_cer:
                best_cer = cer
                best_state = copy.deepcopy(model.state_dict())

        promo = _promote_with_validation_gate(
            model,
            _make_model,
            best_state,
            "ocr",
            ckpt_path,
            base_dir,
            char_to_idx,
            train_dataset,
            min_val_samples=min_val_samples,
            allow_bootstrap=allow_bootstrap,
        )
        history["promotion"] = promo
        print(f"OCR Training finished (Best Val CER: {best_cer:.1f}% | Promoted: {promo.get('promoted')})")
        return model, history
    finally:
        if char_train_dataset is not None:
            char_train_dataset.close()
        if char_val_dataset is not None:
            char_val_dataset.close()


# ==============================================================================
# 2. INPAINTING TRAINING & EVALUATION (Deep U-Net)
# ==============================================================================

def train_inpainting_model(
    epochs=3,
    batch_size=2,
    lr=2e-4,
    max_batches=None,
    val_batches=None,
    from_scratch=False,
    base_dir=BASE_DIR,
    data_dir=DATA_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
    min_val_samples=10,
    allow_bootstrap=False,
):
    set_seed(42)
    print("\n" + "=" * 60)
    print("STARTING TRAINING: Manga Inpainting Model (Deep U-Net)")
    print("=" * 60)

    inpaint_img_dir = os.path.join(data_dir, "inpainting")
    if not os.path.isdir(os.path.join(inpaint_img_dir, "raw")):
        inpaint_img_dir = None

    train_dataset = MangaInpaintingDataset(
        os.path.join(data_dir, "inpainting", "train.jsonl"), img_dir=inpaint_img_dir, is_train=True
    )
    val_dataset = MangaInpaintingDataset(
        os.path.join(data_dir, "inpainting", "val.jsonl"), img_dir=inpaint_img_dir, is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MangaInpaintingModel().to(DEVICE)
    ckpt_path = os.path.join(checkpoint_dir, "best_manga_inpaint.pt")

    if not from_scratch and os.path.exists(ckpt_path):
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
            print("Loaded pre-trained Inpainting checkpoint for warm-start fine-tuning.")
        except Exception as e:
            print(f"Starting Inpainting with freshly initialized weights ({e})")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-5)

    history = {"train_loss": [], "val_psnr": [], "val_hole_psnr": [], "val_l1": []}
    best_psnr = -float("inf")
    best_state = None
    fixed_val_batches = len(val_loader) if (val_batches is None and max_batches is None) else min(val_batches or max_batches, len(val_loader))

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_steps = 0
        start_time = time.time()

        for b_idx, batch in enumerate(train_loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            raw = batch["raw"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            clean = batch["clean"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            pred, composite = model(raw, mask)

            l1_hole = masked_l1(pred, clean, mask)
            l1_valid = masked_l1(pred, clean, 1.0 - mask)
            composite_l1 = torch.mean(torch.abs(composite - clean))
            total_loss = 4.0 * l1_hole + 0.5 * l1_valid + 1.0 * composite_l1
            if torch.isfinite(total_loss):
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_train_loss += total_loss.item()
                train_steps += 1
            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == max_batches:
                print(f"  [Inpaint Ep {epoch}] Step [{b_idx + 1}/{max_batches or len(train_loader)}] | Batch Loss: {total_loss.item():.4f}", flush=True)

            del raw, mask, clean, pred, composite, l1_hole, l1_valid, composite_l1, total_loss

        scheduler.step()
        avg_train_loss = total_train_loss / max(1, train_steps)

        model.eval()
        total_mse = 0.0
        total_hole_mse = 0.0
        total_l1 = 0.0
        val_steps = 0

        with torch.no_grad():
            for b_idx, batch in enumerate(val_loader):
                if b_idx >= fixed_val_batches:
                    break
                raw = batch["raw"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)
                clean = batch["clean"].to(DEVICE)

                pred, composite = model(raw, mask)
                mse = torch.mean((composite - clean) ** 2).item()
                hole_mse = (torch.sum(((pred - clean) * mask) ** 2) / (mask.expand_as(pred).sum() + 1e-6)).item()
                l1 = torch.mean(torch.abs(composite - clean)).item()

                if math.isfinite(mse) and math.isfinite(hole_mse):
                    total_mse += mse
                    total_hole_mse += hole_mse
                    total_l1 += l1
                    val_steps += 1

                del raw, mask, clean, pred, composite

        avg_mse = total_mse / max(1, val_steps)
        avg_hole_mse = total_hole_mse / max(1, val_steps)
        avg_l1 = total_l1 / max(1, val_steps)

        full_psnr = 10 * math.log10(1.0 / max(1e-10, avg_mse))
        hole_psnr = 10 * math.log10(1.0 / max(1e-10, avg_hole_mse))
        elapsed = time.time() - start_time

        history["train_loss"].append(avg_train_loss)
        history["val_psnr"].append(full_psnr)
        history["val_hole_psnr"].append(hole_psnr)
        history["val_l1"].append(avg_l1)

        print(
            f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_train_loss:.4f} | "
            f"Hole PSNR: {hole_psnr:.2f} dB | Full PSNR: {full_psnr:.2f} dB | Time: {elapsed:.1f}s",
            flush=True,
        )

        if val_steps > 0 and hole_psnr > best_psnr:
            best_psnr = hole_psnr
            best_state = copy.deepcopy(model.state_dict())

    promo = _promote_with_validation_gate(
        model,
        MangaInpaintingModel,
        best_state,
        "inpainting",
        ckpt_path,
        base_dir,
        {},
        train_dataset,
        min_val_samples=min_val_samples,
        allow_bootstrap=allow_bootstrap,
    )
    history["promotion"] = promo
    print(f"Inpainting Training finished (Best Hole PSNR: {best_psnr:.2f} dB | Promoted: {promo.get('promoted')})")
    return model, history


# ==============================================================================
# 3. DETECTION TRAINING & EVALUATION (MangaBubbleDetector)
# ==============================================================================

def train_detection_model(
    epochs=3,
    batch_size=8,
    lr=1e-3,
    max_batches=None,
    val_batches=None,
    from_scratch=False,
    synthetic_demo=False,
    base_dir=BASE_DIR,
    data_dir=DATA_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
    min_val_samples=10,
    allow_bootstrap=False,
):
    set_seed(42)
    print("\n" + "=" * 60)
    print("STARTING TRAINING: Manga Bubble & Text Detector")
    print("=" * 60)

    det_img_dir = os.path.join(data_dir, "detection", "images")
    if not os.path.isdir(det_img_dir):
        det_img_dir = None

    train_dataset = MangaDetectionDataset(
        os.path.join(data_dir, "detection", "train.jsonl"), img_dir=det_img_dir, is_train=True, allow_synthetic=synthetic_demo
    )
    val_dataset = MangaDetectionDataset(
        os.path.join(data_dir, "detection", "val.jsonl"), img_dir=det_img_dir, is_train=False, allow_synthetic=synthetic_demo
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=detection_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=detection_collate_fn
    )

    def _make_detector():
        return MangaBubbleDetector(num_classes=NUM_DET_CLASSES, num_anchors=1)

    model = _make_detector().to(DEVICE)
    ckpt_path = os.path.join(checkpoint_dir, "demo_manga_detector.pt" if synthetic_demo else "best_manga_detector.pt")
    if synthetic_demo:
        print("DEMO ONLY: synthetic metrics; active detector checkpoint will not be changed.")

    if not from_scratch and os.path.exists(ckpt_path):
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
            print("Loaded pre-trained Detector checkpoint for warm-start fine-tuning.")
        except Exception as e:
            print(f"Starting Detector with freshly initialized weights ({e})")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-4)

    history = {"train_loss": [], "val_loss": [], "val_grid_iou": []}
    best_loss = float("inf")
    best_state = None
    fixed_val_batches = len(val_loader) if (val_batches is None and max_batches is None) else min(val_batches or max_batches, len(val_loader))

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_steps = 0
        start_time = time.time()

        for b_idx, batch in enumerate(train_loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            images = batch["images"].to(DEVICE)
            optimizer.zero_grad()

            preds = model(images)
            loss, obj_l, box_l, cls_l, _ = detection_loss(
                preds, batch["boxes"], batch["labels"], model.num_classes
            )

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_train_loss += loss.item()
                train_steps += 1
            if (b_idx + 1) % 15 == 0 or (b_idx + 1) == max_batches:
                print(f"  [Detector Ep {epoch}] Step [{b_idx + 1}/{max_batches or len(train_loader)}] | Batch Loss: {loss.item():.4f}", flush=True)

        scheduler.step()
        avg_train_loss = total_train_loss / max(1, train_steps)

        model.eval()
        total_val_loss = 0.0
        total_iou = 0.0
        val_steps = 0

        with torch.no_grad():
            for b_idx, batch in enumerate(val_loader):
                if b_idx >= fixed_val_batches:
                    break
                images = batch["images"].to(DEVICE)
                preds = model(images)
                v_loss, _, _, _, m_iou = detection_loss(
                    preds, batch["boxes"], batch["labels"], model.num_classes
                )
                if torch.isfinite(v_loss):
                    total_val_loss += v_loss.item()
                    total_iou += m_iou
                    val_steps += 1

        avg_val_loss = total_val_loss / max(1, val_steps)
        avg_iou = total_iou / max(1, val_steps)
        elapsed = time.time() - start_time

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_grid_iou"].append(avg_iou)

        print(
            f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | GT-cell regression IoU: {avg_iou:.3f} | Time: {elapsed:.1f}s",
            flush=True,
        )

        if val_steps > 0 and avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())

    if synthetic_demo:
        if best_state is not None:
            model.load_state_dict(best_state)
        _save_atomic_state(model.state_dict(), ckpt_path)
        history["promotion"] = {"promoted": False, "reason": "synthetic_demo", "path": ckpt_path}
        print(f"Synthetic demo detector saved to {ckpt_path}")
    else:
        promo = _promote_with_validation_gate(
            model,
            _make_detector,
            best_state,
            "detector",
            ckpt_path,
            base_dir,
            {},
            train_dataset,
            min_val_samples=min_val_samples,
            allow_bootstrap=allow_bootstrap,
        )
        history["promotion"] = promo
        print(f"Detector Training finished (Best Val Loss: {best_loss:.4f} | Promoted: {promo.get('promoted')})")
    return model, history


# ==============================================================================
# MAIN TRAINING RUNNER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Task Manga AI Training & Evaluation")
    parser.add_argument("--task", type=str, default="all", choices=["all", "ocr", "inpaint", "detector"],
                        help="Specific task to train: ocr, inpaint, detector, or all")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override default task batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override default task learning rate")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test on 5 batches")
    parser.add_argument("--full", action="store_true", help="Train on all dataset batches without batch caps (default)")
    parser.add_argument("--max-batches", type=int, default=None, help="Explicit cap on training batches per epoch")
    parser.add_argument("--val-batches", type=int, default=None, help="Explicit cap on validation batches per epoch")
    parser.add_argument("--min-val-samples", type=int, default=10, help="Minimum verified validation samples required by ValidationGate")
    parser.add_argument("--allow-bootstrap", action="store_true", help="Allow initial checkpoint creation when no incumbent exists and validation manifest is absent")
    parser.add_argument("--from-scratch", action="store_true", help="Initialize models from scratch")
    parser.add_argument("--synthetic-demo", action="store_true", help="Explicit detector-only synthetic demo; writes demo checkpoint, never active detector")
    args = parser.parse_args()
    if args.synthetic_demo and args.task != "detector":
        parser.error("--synthetic-demo requires --task detector")

    set_seed(42)

    if args.smoke_test:
        max_b = 5
        val_b = args.val_batches or 5
        mode_str = "SMOKE TEST (5 batches per epoch)"
    elif args.max_batches is not None:
        max_b = args.max_batches
        val_b = args.val_batches or args.max_batches
        mode_str = f"EXPLICIT CAP ({args.max_batches} batches per epoch)"
    else:
        max_b = None
        val_b = args.val_batches
        mode_str = "FULL DATASET (Uncapped)"

    print("=" * 60)
    print("MANGA MULTI-TASK MACHINE LEARNING PIPELINE")
    print(f"Device: {DEVICE} (Threads: {torch.get_num_threads()})")
    print(f"Task: {args.task} | Epochs: {args.epochs} | Mode: {mode_str}")
    print(f"Warm-start: {'Disabled (from scratch)' if args.from_scratch else 'Enabled (loads existing checkpoints)'}")
    print(f"Detection classes ({NUM_DET_CLASSES}): {list(CATEGORY_TO_IDX.keys())}")
    print("=" * 60)

    ocr_hist = inpaint_hist = det_hist = None

    if args.task in ["all", "ocr"]:
        _, ocr_hist = train_ocr_model(
            epochs=args.epochs,
            batch_size=args.batch_size or 16,
            lr=args.lr or 5e-4,
            max_batches=max_b,
            val_batches=val_b,
            from_scratch=args.from_scratch,
            min_val_samples=args.min_val_samples,
            allow_bootstrap=args.allow_bootstrap,
        )

    if args.task in ["all", "inpaint"]:
        _, inpaint_hist = train_inpainting_model(
            epochs=args.epochs,
            batch_size=args.batch_size or 2,
            lr=args.lr or 2e-4,
            max_batches=max_b,
            val_batches=val_b,
            from_scratch=args.from_scratch,
            min_val_samples=args.min_val_samples,
            allow_bootstrap=args.allow_bootstrap,
        )

    if args.task in ["all", "detector"]:
        _, det_hist = train_detection_model(
            epochs=args.epochs,
            batch_size=args.batch_size or 8,
            lr=args.lr or 1e-3,
            max_batches=max_b,
            val_batches=val_b,
            from_scratch=args.from_scratch,
            synthetic_demo=args.synthetic_demo,
            min_val_samples=args.min_val_samples,
            allow_bootstrap=args.allow_bootstrap,
        )

    # Plot metrics for tasks that were executed
    active_plots = [h for h in [ocr_hist, inpaint_hist, det_hist] if h is not None]
    if active_plots and plt is not None:
        n_plots = len(active_plots)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        if ocr_hist is not None:
            axes[plot_idx].plot(ocr_hist.get("train_loss", []), label="Train CTC Loss", color="blue", marker="o")
            axes[plot_idx].plot(ocr_hist.get("val_loss", []), label="Val Loss", color="cyan", marker="s")
            final_cer = f"{ocr_hist['val_cer'][-1]:.1f}%" if ocr_hist.get("val_cer") else "N/A"
            axes[plot_idx].set_title(f"OCR CTC Loss (Final CER: {final_cer})")
            axes[plot_idx].set_xlabel("Epoch")
            axes[plot_idx].set_ylabel("Loss")
            axes[plot_idx].legend()
            plot_idx += 1

        if inpaint_hist is not None:
            axes[plot_idx].plot(inpaint_hist.get("val_hole_psnr", []), label="Hole PSNR (dB)", color="green", marker="o")
            axes[plot_idx].plot(inpaint_hist.get("val_psnr", []), label="Full PSNR (dB)", color="lightgreen", linestyle="--", marker="s")
            final_psnr = f"{inpaint_hist['val_hole_psnr'][-1]:.2f} dB" if inpaint_hist.get("val_hole_psnr") else "N/A"
            axes[plot_idx].set_title(f"Inpainting PSNR (Final: {final_psnr})")
            axes[plot_idx].set_xlabel("Epoch")
            axes[plot_idx].set_ylabel("PSNR (dB)")
            axes[plot_idx].legend()
            plot_idx += 1

        if det_hist is not None:
            axes[plot_idx].plot(det_hist.get("train_loss", []), label="Train Loss", color="red", marker="o")
            axes[plot_idx].plot(det_hist.get("val_loss", []), label="Val Loss", color="orange", marker="s")
            final_iou = f"{det_hist['val_grid_iou'][-1]:.3f}" if det_hist.get("val_grid_iou") else "N/A"
            axes[plot_idx].set_title(f"Detector Loss (Final GT-cell regression IoU: {final_iou})")
            axes[plot_idx].set_xlabel("Epoch")
            axes[plot_idx].set_ylabel("Loss")
            axes[plot_idx].legend()
            plot_idx += 1

        plt.tight_layout()
        plot_path = os.path.join(METRICS_DIR, "training_curves.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\nTraining curves successfully saved to: {plot_path}")

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Checkpoints directory: {CHECKPOINT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
