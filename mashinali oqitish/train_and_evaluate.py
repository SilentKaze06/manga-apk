import os
import sys
import json
import math
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from models import MangaBubbleDetector, MangaOCRModel, MangaInpaintingModel
from dataset_loader import (
    MangaDetectionDataset, MangaOCRDataset, MangaInpaintingDataset,
    ocr_collate_fn, detection_collate_fn, CATEGORY_TO_IDX, NUM_DET_CLASSES
)

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
# 1. OCR TRAINING & EVALUATION (CRNN + CTC Loss)
# ==============================================================================

def train_ocr_model(epochs=3, batch_size=16, lr=5e-4, max_batches=None, from_scratch=False):
    set_seed(42)
    print("\n" + "=" * 60)
    print("STARTING TRAINING: Manga OCR Model (CRNN + CTC)")
    print("=" * 60)

    vocab_file = os.path.join(BASE_DIR, "ocr_vocab.json")
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"Missing required vocabulary file: {vocab_file}")

    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    char_to_idx = vocab_data["char_to_idx"]
    idx_to_char = {int(k): v for k, v in vocab_data["idx_to_char"].items()}
    vocab_size = len(char_to_idx)

    ocr_img_dir = os.path.join(DATA_DIR, "ocr", "crops")
    if not os.path.isdir(ocr_img_dir):
        ocr_img_dir = os.path.join(DATA_DIR, "ocr", "images")
    if not os.path.isdir(ocr_img_dir):
        ocr_img_dir = None

    train_dataset = MangaOCRDataset(
        os.path.join(DATA_DIR, "ocr", "train.jsonl"), char_to_idx, img_dir=ocr_img_dir, is_train=True
    )
    val_dataset = MangaOCRDataset(
        os.path.join(DATA_DIR, "ocr", "val.jsonl"), char_to_idx, img_dir=ocr_img_dir, is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=ocr_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=ocr_collate_fn)

    model = MangaOCRModel(vocab_size=vocab_size, hidden_dim=128).to(DEVICE)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_manga_ocr.pt")

    # Warm-start from pre-trained weights if available
    if not from_scratch and os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            model.load_checkpoint_weights(state)
            print("Loaded pre-trained OCR checkpoint for warm-start fine-tuning.")
        except Exception as e:
            print(f"Starting OCR with freshly initialized weights ({e})")
    else:
        with torch.no_grad():
            if hasattr(model, "fc") and len(model.fc) >= 5:
                model.fc[4].bias[0] = -3.0

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)

    history = {"train_loss": [], "val_loss": [], "val_cer": []}
    best_cer = float("inf")
    fixed_val_batches = min(25 if max_batches is None else max_batches, len(val_loader))

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_steps = 0
        start_time = time.time()

        for b_idx, batch in enumerate(train_loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            images = batch["images"].to(DEVICE)
            targets = batch["targets"].to(DEVICE)
            target_lens = batch["target_lens"].to(DEVICE)

            optimizer.zero_grad()
            log_probs = model(images)  # (T, B, C)
            T, B, _ = log_probs.shape
            input_lens = torch.full(size=(B,), fill_value=T, dtype=torch.long, device=DEVICE)

            loss = criterion(log_probs, targets, input_lens, target_lens)
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_train_loss += loss.item()
                train_steps += 1
                if (b_idx + 1) % 15 == 0 or (b_idx + 1) == max_batches:
                    print(f"  [OCR Ep {epoch}] Step [{b_idx + 1}/{max_batches or len(train_loader)}] | Batch Loss: {loss.item():.4f}", flush=True)

        scheduler.step()
        avg_train_loss = total_train_loss / max(1, train_steps)

        # Validation: evaluated on identical fixed batch count across all epochs
        model.eval()
        total_val_loss = 0.0
        val_steps = 0
        total_edit_distance = 0
        total_chars = 0

        with torch.no_grad():
            for b_idx, batch in enumerate(val_loader):
                if b_idx >= fixed_val_batches:
                    break
                images = batch["images"].to(DEVICE)
                targets = batch["targets"].to(DEVICE)
                target_lens = batch["target_lens"].to(DEVICE)

                log_probs = model(images)
                T, B, _ = log_probs.shape
                input_lens = torch.full(size=(B,), fill_value=T, dtype=torch.long, device=DEVICE)

                v_loss = criterion(log_probs, targets, input_lens, target_lens)
                if torch.isfinite(v_loss):
                    total_val_loss += v_loss.item()
                    val_steps += 1

                for i in range(B):
                    ref_str = batch["texts"][i]
                    pred_str = decode_ctc(log_probs[:, i, :], idx_to_char)
                    dist = compute_levenshtein(pred_str, ref_str)
                    total_edit_distance += dist
                    total_chars += max(1, len(ref_str))

        avg_val_loss = total_val_loss / max(1, val_steps)
        cer = (total_edit_distance / max(1, total_chars)) * 100.0
        elapsed = time.time() - start_time

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_cer"].append(cer)

        print(
            f"Epoch [{epoch}/{epochs}] | Train CTC Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | CER: {cer:.1f}% | Time: {elapsed:.1f}s",
            flush=True,
        )

        if cer <= best_cer:
            best_cer = cer
            torch.save(model.state_dict(), ckpt_path)

    print(f"OCR Model saved to {ckpt_path} (Best CER: {best_cer:.1f}%)")
    return model, history


# ==============================================================================
# 2. INPAINTING TRAINING & EVALUATION (Deep U-Net)
# ==============================================================================

def train_inpainting_model(epochs=3, batch_size=2, lr=2e-4, max_batches=None, from_scratch=False):
    set_seed(42)
    print("\n" + "=" * 60)
    print("STARTING TRAINING: Manga Inpainting Model (Deep U-Net)")
    print("=" * 60)

    inpaint_img_dir = os.path.join(DATA_DIR, "inpainting")
    if not os.path.isdir(os.path.join(inpaint_img_dir, "raw")):
        inpaint_img_dir = None

    train_dataset = MangaInpaintingDataset(
        os.path.join(DATA_DIR, "inpainting", "train.jsonl"), img_dir=inpaint_img_dir, is_train=True
    )
    val_dataset = MangaInpaintingDataset(
        os.path.join(DATA_DIR, "inpainting", "val.jsonl"), img_dir=inpaint_img_dir, is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MangaInpaintingModel().to(DEVICE)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_manga_inpaint.pt")

    if not from_scratch and os.path.exists(ckpt_path):
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
            print("Loaded pre-trained Inpainting checkpoint for warm-start fine-tuning.")
        except Exception as e:
            print(f"Starting Inpainting with freshly initialized weights ({e})")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {"train_loss": [], "val_psnr": [], "val_hole_psnr": [], "val_l1": []}
    best_psnr = -float("inf")
    fixed_val_batches = min(20 if max_batches is None else max_batches, len(val_loader))

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

        if hole_psnr > best_psnr:
            best_psnr = hole_psnr
            torch.save(model.state_dict(), ckpt_path)

    print(f"Inpainting Model saved to {ckpt_path} (Best Hole PSNR: {best_psnr:.2f} dB)")
    return model, history


# ==============================================================================
# 3. DETECTION TRAINING & EVALUATION (MangaBubbleDetector)
# ==============================================================================

def train_detection_model(epochs=3, batch_size=8, lr=1e-3, max_batches=None, from_scratch=False, synthetic_demo=False):
    set_seed(42)
    print("\n" + "=" * 60)
    print("STARTING TRAINING: Manga Bubble & Text Detector")
    print("=" * 60)

    det_img_dir = os.path.join(DATA_DIR, "detection", "images")
    if not os.path.isdir(det_img_dir):
        det_img_dir = None

    train_dataset = MangaDetectionDataset(
        os.path.join(DATA_DIR, "detection", "train.jsonl"), img_dir=det_img_dir, is_train=True, allow_synthetic=synthetic_demo
    )
    val_dataset = MangaDetectionDataset(
        os.path.join(DATA_DIR, "detection", "val.jsonl"), img_dir=det_img_dir, is_train=False, allow_synthetic=synthetic_demo
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=detection_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=detection_collate_fn
    )

    model = MangaBubbleDetector(num_classes=NUM_DET_CLASSES, num_anchors=1).to(DEVICE)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "demo_manga_detector.pt" if synthetic_demo else "best_manga_detector.pt")
    if synthetic_demo:
        print("DEMO ONLY: synthetic metrics; active detector checkpoint will not be changed.")

    if not from_scratch and os.path.exists(ckpt_path):
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
            print("Loaded pre-trained Detector checkpoint for warm-start fine-tuning.")
        except Exception as e:
            print(f"Starting Detector with freshly initialized weights ({e})")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    history = {"train_loss": [], "val_loss": [], "val_grid_iou": []}
    best_loss = float("inf")
    fixed_val_batches = min(20 if max_batches is None else max_batches, len(val_loader))

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

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), ckpt_path)

    print(f"Detector Model saved to {ckpt_path} (Best Val Loss: {best_loss:.4f})")
    return model, history


# ==============================================================================
# MAIN TRAINING RUNNER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Task Manga AI Training & Evaluation")
    parser.add_argument("--task", type=str, default="all", choices=["all", "ocr", "inpaint", "detector"],
                        help="Specific task to train: ocr, inpaint, detector, or all")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test on 5 batches")
    parser.add_argument("--full", action="store_true", help="Train on all dataset batches without batch caps")
    parser.add_argument("--max-batches", type=int, default=None, help="Explicit cap on batches per epoch")
    parser.add_argument("--from-scratch", action="store_true", help="Initialize models from scratch")
    parser.add_argument("--synthetic-demo", action="store_true", help="Explicit detector-only synthetic demo; writes demo checkpoint, never active detector")
    args = parser.parse_args()
    if args.synthetic_demo and args.task != "detector":
        parser.error("--synthetic-demo requires --task detector")

    set_seed(42)

    if args.smoke_test:
        ocr_max_b = 5
        inpaint_max_b = 5
        det_max_b = 5
        mode_str = "SMOKE TEST (5 batches per epoch)"
    elif args.full:
        ocr_max_b = None
        inpaint_max_b = None
        det_max_b = None
        mode_str = "FULL DATASET (All batches, uncapped)"
    elif args.max_batches is not None:
        ocr_max_b = args.max_batches
        inpaint_max_b = args.max_batches
        det_max_b = args.max_batches
        mode_str = f"EXPLICIT CAP ({args.max_batches} batches per epoch)"
    else:
        ocr_max_b = 60
        inpaint_max_b = 35
        det_max_b = 60
        mode_str = "OPTIMIZED ~10-MINUTE PROFILE"

    print("=" * 60)
    print("MANGA MULTI-TASK MACHINE LEARNING PIPELINE")
    print(f"Device: {DEVICE} (Threads: {torch.get_num_threads()})")
    print(f"Task: {args.task} | Epochs: {args.epochs} | Mode: {mode_str}")
    print(f"Warm-start: {'Disabled (from scratch)' if args.from_scratch else 'Enabled (loads existing checkpoints)'}")
    print(f"Detection classes ({NUM_DET_CLASSES}): {list(CATEGORY_TO_IDX.keys())}")
    print("=" * 60)

    ocr_hist = inpaint_hist = det_hist = None

    if args.task in ["all", "ocr"]:
        _, ocr_hist = train_ocr_model(epochs=args.epochs, max_batches=ocr_max_b, from_scratch=args.from_scratch)

    if args.task in ["all", "inpaint"]:
        _, inpaint_hist = train_inpainting_model(epochs=args.epochs, max_batches=inpaint_max_b, from_scratch=args.from_scratch)

    if args.task in ["all", "detector"]:
        _, det_hist = train_detection_model(epochs=args.epochs, max_batches=det_max_b, from_scratch=args.from_scratch, synthetic_demo=args.synthetic_demo)

    # Plot metrics for tasks that were executed
    active_plots = [h for h in [ocr_hist, inpaint_hist, det_hist] if h is not None]
    if active_plots:
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
