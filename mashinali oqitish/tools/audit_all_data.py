import os
import sys
import json
import hashlib
from PIL import Image
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import ml_paths
ml_paths.ensure_ml_paths()

DATA_DIR = os.path.join(BASE_DIR, "self_learning_data")
VOCAB_PATH = ml_paths.resolve_vocab_path(BASE_DIR)

print("=" * 70)
print("COMPREHENSIVE AUDIT OF COLLECTED DATA")
print("=" * 70)

# ==============================================================================
# 1. OCR AUDIT
# ==============================================================================
ocr_jsonl = os.path.join(DATA_DIR, "ocr", "feedback_ocr.jsonl")
ocr_records = []
if os.path.exists(ocr_jsonl):
    with open(ocr_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                rec = json.loads(line)
                rec["orig_line"] = idx
                ocr_records.append(rec)
            except Exception as e:
                print(f"OCR JSON parse error at line {idx}: {e}")

print(f"Total OCR records: {len(ocr_records)}")

# Image hash mapping
ocr_by_imghash = {}
for rec in ocr_records:
    p = ml_paths.resolve_data_path(rec.get("image_path", ""), BASE_DIR)
    rec["_resolved_image_path"] = p
    if p and os.path.exists(p):
        with open(p, "rb") as fp:
            h = hashlib.md5(fp.read()).hexdigest()
        rec["img_hash"] = h
        if h not in ocr_by_imghash:
            ocr_by_imghash[h] = []
        ocr_by_imghash[h].append(rec)
    else:
        rec["img_hash"] = None

print(f"Unique image hashes in OCR: {len(ocr_by_imghash)}")

# Classify OCR records
ocr_empty_text = []
ocr_blank_image = []
ocr_missing_file = []
ocr_conflicts_resolved_by_verified = 0

for h, recs in ocr_by_imghash.items():
    # Check if there's conflict
    texts = set(r.get("text", "").strip() for r in recs)
    has_verified = any(r.get("verified", False) for r in recs)
    if len(texts) > 1 and has_verified:
        ocr_conflicts_resolved_by_verified += 1

for r in ocr_records:
    t = r.get("text", "").strip()
    p = r.get("_resolved_image_path") or ml_paths.resolve_data_path(r.get("image_path", ""), BASE_DIR)
    if not t:
        ocr_empty_text.append(r)
    if not p or not os.path.exists(p):
        ocr_missing_file.append(r)
    else:
        try:
            with Image.open(p) as im:
                arr = np.array(im.convert("L"))
                if np.std(arr) < 1.5:
                    ocr_blank_image.append((r, np.std(arr)))
        except Exception:
            pass

print(f"OCR empty text count: {len(ocr_empty_text)}")
print(f"OCR blank images count: {len(ocr_blank_image)}")
print(f"OCR missing image paths: {len(ocr_missing_file)}")
print(f"OCR image clusters with conflicting unverified vs verified: {ocr_conflicts_resolved_by_verified}")

# ==============================================================================
# 2. INPAINTING AUDIT
# ==============================================================================
inp_jsonl = os.path.join(DATA_DIR, "inpainting", "feedback_inpaint.jsonl")
inp_records = []
if os.path.exists(inp_jsonl):
    with open(inp_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                rec = json.loads(line)
                rec["orig_line"] = idx
                inp_records.append(rec)
            except Exception as e:
                print(f"Inpaint JSON error at line {idx}: {e}")

print(f"\nTotal Inpainting records: {len(inp_records)}")
inp_by_triplet_hash = {}
inp_identical_raw_clean = []
inp_empty_mask = []
inp_huge_mask = []
inp_dim_mismatch = []

for rec in inp_records:
    r_p = ml_paths.resolve_data_path(rec.get("raw_image", ""), BASE_DIR)
    m_p = ml_paths.resolve_data_path(rec.get("mask", ""), BASE_DIR)
    c_p = ml_paths.resolve_data_path(rec.get("ground_truth", ""), BASE_DIR)

    if r_p and m_p and c_p and os.path.exists(r_p) and os.path.exists(m_p) and os.path.exists(c_p):
        with open(r_p, "rb") as fp: hr = hashlib.md5(fp.read()).hexdigest()
        with open(m_p, "rb") as fp: hm = hashlib.md5(fp.read()).hexdigest()
        with open(c_p, "rb") as fp: hc = hashlib.md5(fp.read()).hexdigest()
        thash = (hr, hm, hc)
        rec["triplet_hash"] = thash
        if thash not in inp_by_triplet_hash:
            inp_by_triplet_hash[thash] = []
        inp_by_triplet_hash[thash].append(rec)
        
        with Image.open(r_p) as r_im, Image.open(m_p) as m_im, Image.open(c_p) as c_im:
            if r_im.size != m_im.size or r_im.size != c_im.size:
                inp_dim_mismatch.append(rec)
            else:
                m_arr = np.array(m_im.convert("L"))
                coverage = np.count_nonzero(m_arr > 30) / float(m_arr.size)
                if coverage < 0.005:
                    inp_empty_mask.append((rec, coverage))
                elif coverage > 0.95:
                    inp_huge_mask.append((rec, coverage))
                
                # Check pixel equality of raw and clean
                r_arr = np.array(r_im)
                c_arr = np.array(c_im)
                if np.array_equal(r_arr, c_arr):
                    inp_identical_raw_clean.append(rec)

print(f"Unique inpainting triplets: {len(inp_by_triplet_hash)}")
print(f"Inpainting identical raw and clean: {len(inp_identical_raw_clean)}")
print(f"Inpainting empty mask (<0.5% masked): {len(inp_empty_mask)}")
print(f"Inpainting oversized mask (>95% masked): {len(inp_huge_mask)}")
print(f"Inpainting dimension mismatches: {len(inp_dim_mismatch)}")

# ==============================================================================
# 3. DETECTION AUDIT
# ==============================================================================
det_jsonl = os.path.join(DATA_DIR, "detection", "feedback_detection.jsonl")
det_records = []
if os.path.exists(det_jsonl):
    with open(det_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                rec = json.loads(line)
                rec["orig_line"] = idx
                det_records.append(rec)
            except Exception as e:
                print(f"Detection JSON error at line {idx}: {e}")

print(f"\nTotal Detection page records: {len(det_records)}")
det_by_imghash = {}
det_empty_ann = []
det_invalid_boxes = []

def _check_det_boxes(records):
    invalid = []
    for rec in records:
        for a in rec.get("annotations", []):
            box = a.get("bbox", [])
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                invalid.append((rec, box))
                continue
            try:
                b = [float(c) for c in box]
            except (ValueError, TypeError):
                invalid.append((rec, box))
                continue
            if b[2] <= b[0] or b[3] <= b[1] or (b[2] - b[0]) < 8 or (b[3] - b[1]) < 8:
                invalid.append((rec, box))
    return invalid


for rec in det_records:
    p = ml_paths.resolve_data_path(rec.get("image_path", ""), BASE_DIR)
    if p and os.path.exists(p):
        with open(p, "rb") as fp:
            h = hashlib.md5(fp.read()).hexdigest()
        rec["img_hash"] = h
        if h not in det_by_imghash:
            det_by_imghash[h] = []
        det_by_imghash[h].append(rec)

        anns = rec.get("annotations", [])
        if not anns:
            det_empty_ann.append(rec)
        det_invalid_boxes.extend(_check_det_boxes([rec]))

# Also audit manga_ml_datasets/detection/full.jsonl bbox validity (without crashing on string coords)
ml_det_jsonl = os.path.join(BASE_DIR, "manga_ml_datasets", "detection", "full.jsonl")
ml_det_invalid_boxes = []
ml_det_total = 0
if os.path.exists(ml_det_jsonl):
    with open(ml_det_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                ml_det_total += 1
                ml_det_invalid_boxes.extend(_check_det_boxes([rec]))
            except Exception as e:
                print(f"manga_ml detection JSON error at line {idx}: {e}")

print(f"Unique Detection pages: {len(det_by_imghash)}")
print(f"Empty annotations: {len(det_empty_ann)}")
print(f"Invalid / tiny boxes: {len(det_invalid_boxes)}")
if ml_det_total:
    print(f"manga_ml_datasets detection pages: {ml_det_total} | Invalid/tiny boxes: {len(ml_det_invalid_boxes)}")

# ==============================================================================
# 4. VOCABULARY AUDIT
# ==============================================================================
if os.path.exists(VOCAB_PATH):
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vdata = json.load(f)
    c2i = vdata.get("char_to_idx", {})
    print(f"\nOCR Vocabulary size: {len(c2i)} tokens")

print("=" * 70)
print("AUDIT COMPLETE!")
print("=" * 70)
