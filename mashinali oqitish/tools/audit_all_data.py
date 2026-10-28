import os
import sys
import json
import hashlib
from PIL import Image
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "self_learning_data")
VOCAB_PATH = os.path.join(BASE_DIR, "ocr_vocab.json")

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
    p = os.path.normpath(rec.get("image_path", ""))
    if os.path.exists(p):
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
    p = r.get("image_path", "")
    if not t:
        ocr_empty_text.append(r)
    if not os.path.exists(p):
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
    r_p = os.path.normpath(rec.get("raw_image", ""))
    m_p = os.path.normpath(rec.get("mask", ""))
    c_p = os.path.normpath(rec.get("ground_truth", ""))
    
    if os.path.exists(r_p) and os.path.exists(m_p) and os.path.exists(c_p):
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

for rec in det_records:
    p = os.path.normpath(rec.get("image_path", ""))
    if os.path.exists(p):
        with open(p, "rb") as fp:
            h = hashlib.md5(fp.read()).hexdigest()
        rec["img_hash"] = h
        if h not in det_by_imghash:
            det_by_imghash[h] = []
        det_by_imghash[h].append(rec)
        
        anns = rec.get("annotations", [])
        if not anns:
            det_empty_ann.append(rec)
        w, h_dim = rec.get("width", 0), rec.get("height", 0)
        for a in anns:
            box = a.get("bbox", [])
            if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1] or (box[2]-box[0]) < 8 or (box[3]-box[1]) < 8:
                det_invalid_boxes.append((rec, box))

print(f"Unique Detection pages: {len(det_by_imghash)}")
print(f"Empty annotations: {len(det_empty_ann)}")
print(f"Invalid / tiny boxes: {len(det_invalid_boxes)}")

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
