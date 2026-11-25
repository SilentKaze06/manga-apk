import os
import sys
import json
import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import ml_paths
ml_paths.ensure_ml_paths()

SELF_LEARNING = os.path.join(BASE_DIR, "self_learning_data")
MANGA_ML = os.path.join(BASE_DIR, "manga_ml_datasets")

report = {
    "ocr_feedback": {},
    "inpaint_feedback": {},
    "detection_feedback": {},
    "manga_ml_ocr": {},
    "manga_ml_inpainting": {},
    "manga_ml_detection": {},
    "vocab": {}
}

# 1. OCR FEEDBACK
ocr_jsonl = os.path.join(SELF_LEARNING, "ocr", "feedback_ocr.jsonl")
if os.path.exists(ocr_jsonl):
    lines = open(ocr_jsonl, "r", encoding="utf-8").readlines()
    total = len(lines)
    empty_text = 0
    whitespace_only = 0
    short_noise = 0
    missing_img = 0
    corrupt_img = 0
    blank_img = 0
    tiny_img = 0
    duplicates = {}
    samples_bad = []
    samples_good = []
    
    for i, line in enumerate(lines):
        try:
            d = json.loads(line)
        except Exception as e:
            samples_bad.append({"idx": i, "reason": f"json_error: {e}"})
            continue
        
        text = d.get("text", "")
        img_p = d.get("image_path", "")
        verified = d.get("verified", False)
        
        # text checks
        if not text:
            empty_text += 1
            samples_bad.append({"idx": i, "reason": "empty_text", "text": text, "path": img_p})
            continue
        if not text.strip():
            whitespace_only += 1
            samples_bad.append({"idx": i, "reason": "whitespace_only", "text": repr(text), "path": img_p})
            continue
        
        # duplicates
        clean_t = text.strip()
        duplicates[clean_t] = duplicates.get(clean_t, 0) + 1
        
        # check image
        img_p = ml_paths.resolve_data_path(img_p, BASE_DIR)
        if not img_p or not os.path.exists(img_p):
            missing_img += 1
            samples_bad.append({"idx": i, "reason": "missing_image", "text": text, "path": img_p})
            continue
        
        try:
            with Image.open(img_p) as im:
                w, h = im.size
                if w < 5 or h < 5:
                    tiny_img += 1
                    samples_bad.append({"idx": i, "reason": f"tiny_img_{w}x{h}", "text": text, "path": img_p})
                arr = np.array(im.convert("L"))
                std = np.std(arr)
                if std < 1.0:
                    blank_img += 1
                    samples_bad.append({"idx": i, "reason": f"blank_img_std_{std:.2f}", "text": text, "path": img_p})
        except Exception as e:
            corrupt_img += 1
            samples_bad.append({"idx": i, "reason": f"corrupt_img: {e}", "text": text, "path": img_p})
            continue
            
        if len(samples_good) < 20:
            samples_good.append({"text": text, "verified": verified, "w": w, "h": h})
            
    dup_counts = sum(count - 1 for count in duplicates.values() if count > 1)
    report["ocr_feedback"] = {
        "total": total,
        "empty_text": empty_text,
        "whitespace_only": whitespace_only,
        "missing_img": missing_img,
        "corrupt_img": corrupt_img,
        "blank_img": blank_img,
        "tiny_img": tiny_img,
        "exact_text_duplicates": dup_counts,
        "samples_bad": samples_bad[:25],
        "samples_good_sample": samples_good[:5]
    }

# 2. INPAINTING FEEDBACK
inpaint_jsonl = os.path.join(SELF_LEARNING, "inpainting", "feedback_inpaint.jsonl")
if os.path.exists(inpaint_jsonl):
    lines = open(inpaint_jsonl, "r", encoding="utf-8").readlines()
    total = len(lines)
    missing_raw = 0
    missing_mask = 0
    missing_clean = 0
    mask_empty = 0
    mask_all_white = 0
    raw_clean_identical = 0
    dim_mismatch = 0
    corrupt = 0
    samples_bad = []
    
    for i, line in enumerate(lines):
        try:
            d = json.loads(line)
        except Exception as e:
            samples_bad.append({"idx": i, "reason": f"json_error: {e}"})
            continue
        
        raw_p = ml_paths.resolve_data_path(d.get("raw_image", ""), BASE_DIR)
        mask_p = ml_paths.resolve_data_path(d.get("mask", ""), BASE_DIR)
        clean_p = ml_paths.resolve_data_path(d.get("ground_truth", ""), BASE_DIR)

        if not raw_p or not os.path.exists(raw_p):
            missing_raw += 1
        if not mask_p or not os.path.exists(mask_p):
            missing_mask += 1
        if not clean_p or not os.path.exists(clean_p):
            missing_clean += 1

        if not (raw_p and mask_p and clean_p and os.path.exists(raw_p) and os.path.exists(mask_p) and os.path.exists(clean_p)):
            samples_bad.append({"idx": i, "reason": "missing_files", "raw": raw_p, "mask": mask_p, "clean": clean_p})
            continue
            
        try:
            with Image.open(raw_p) as r_im, Image.open(mask_p) as m_im, Image.open(clean_p) as c_im:
                if r_im.size != m_im.size or r_im.size != c_im.size:
                    dim_mismatch += 1
                    samples_bad.append({"idx": i, "reason": f"dim_mismatch_{r_im.size}_{m_im.size}_{c_im.size}"})
                    continue
                
                m_arr = np.array(m_im.convert("L"))
                non_zero = np.count_nonzero(m_arr > 30)
                total_pixels = m_arr.size
                mask_ratio = non_zero / float(total_pixels)
                
                if mask_ratio == 0:
                    mask_empty += 1
                    samples_bad.append({"idx": i, "reason": "mask_completely_empty", "path": mask_p})
                elif mask_ratio > 0.98:
                    mask_all_white += 1
                    samples_bad.append({"idx": i, "reason": "mask_all_white", "path": mask_p})
                    
                # check if raw and clean are 100% identical
                r_arr = np.array(r_im)
                c_arr = np.array(c_im)
                if np.array_equal(r_arr, c_arr):
                    raw_clean_identical += 1
                    samples_bad.append({"idx": i, "reason": "raw_clean_identical", "path": raw_p})
        except Exception as e:
            corrupt += 1
            samples_bad.append({"idx": i, "reason": f"corrupt: {e}"})
            
    report["inpaint_feedback"] = {
        "total": total,
        "missing_raw": missing_raw,
        "missing_mask": missing_mask,
        "missing_clean": missing_clean,
        "mask_empty": mask_empty,
        "mask_all_white": mask_all_white,
        "raw_clean_identical": raw_clean_identical,
        "dim_mismatch": dim_mismatch,
        "corrupt": corrupt,
        "samples_bad": samples_bad[:25]
    }

# 3. DETECTION FEEDBACK
det_jsonl = os.path.join(SELF_LEARNING, "detection", "feedback_detection.jsonl")
if os.path.exists(det_jsonl):
    lines = open(det_jsonl, "r", encoding="utf-8").readlines()
    total = len(lines)
    missing_img = 0
    empty_annotations = 0
    invalid_bboxes = 0
    samples_bad = []
    
    for i, line in enumerate(lines):
        try:
            d = json.loads(line)
        except Exception as e:
            samples_bad.append({"idx": i, "reason": f"json_error: {e}"})
            continue
            
        img_p = ml_paths.resolve_data_path(d.get("image_path", ""), BASE_DIR)
        if not img_p or not os.path.exists(img_p):
            missing_img += 1
            samples_bad.append({"idx": i, "reason": "missing_image", "path": img_p})
            continue
            
        anns = d.get("annotations", [])
        if not anns:
            empty_annotations += 1
            
        w = d.get("width", 0)
        h = d.get("height", 0)
        for ann in anns:
            b = ann.get("bbox", [])
            try:
                bf = [float(x) for x in b]
                bad = len(bf) != 4 or bf[2] <= bf[0] or bf[3] <= bf[1]
            except (ValueError, TypeError):
                bad = True
            if bad:
                invalid_bboxes += 1
                samples_bad.append({"idx": i, "reason": f"invalid_bbox_{b}"})
                
    report["detection_feedback"] = {
        "total": total,
        "missing_img": missing_img,
        "empty_annotations": empty_annotations,
        "invalid_bboxes": invalid_bboxes,
        "samples_bad": samples_bad[:20]
    }

# 4. VOCABULARY CHECK
vocab_file = ml_paths.resolve_vocab_path(BASE_DIR)
if os.path.exists(vocab_file):
    with open(vocab_file, "r", encoding="utf-8") as f:
        vdata = json.load(f)
    char_to_idx = vdata.get("char_to_idx", {})
    control_chars = []
    for c in char_to_idx:
        if len(c) == 1 and ord(c) < 32 and c not in ("\n", "\t"):
            control_chars.append(repr(c))
    report["vocab"] = {
        "total_tokens": len(char_to_idx),
        "control_chars": control_chars,
        "has_blank": "<blank>" in char_to_idx,
        "has_unk": "<unk>" in char_to_idx
    }

# 5. MANGA_ML DATASETS CHECK
for task in ["ocr", "inpainting", "detection"]:
    fpath = os.path.join(MANGA_ML, task, "train.jsonl")
    if os.path.exists(fpath):
        lines = open(fpath, "r", encoding="utf-8").readlines()
        report[f"manga_ml_{task}"] = {
            "train_samples": len(lines),
            "sample_keys": list(json.loads(lines[0]).keys()) if lines else []
        }

REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)
with open(os.path.join(REPORT_DIR, "data_inspection_report.json"), "w", encoding="utf-8") as out:
    json.dump(report, out, indent=2, ensure_ascii=False)

print(f"DATA INSPECTION COMPLETE! Written to {os.path.join(REPORT_DIR, 'data_inspection_report.json')}")
