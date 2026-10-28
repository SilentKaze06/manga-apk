import os
import sys
import json
import re
import random
import zipfile
import xml.etree.ElementTree as ET
import io
import logging

from utils import atomik_json_yoz

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_datasets")

sys.stdout.reconfigure(encoding='utf-8')

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACTED_DIR = os.path.join(BASE_DIR, "extracted_data")
DATA_OUT_DIR = os.path.join(BASE_DIR, "manga_ml_datasets")
VOCAB_PATH = os.path.join(BASE_DIR, "ocr_vocab.json")

os.makedirs(EXTRACTED_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_OUT_DIR, "detection"), exist_ok=True)
os.makedirs(os.path.join(DATA_OUT_DIR, "ocr"), exist_ok=True)
os.makedirs(os.path.join(DATA_OUT_DIR, "inpainting"), exist_ok=True)

def extract_docx_to_text(docx_bytes):
    """
    Extracts plain text paragraphs from a .docx binary byte stream.
    """
    with zipfile.ZipFile(io.BytesIO(docx_bytes), "r") as zf_docx:
        xml_content = zf_docx.read("word/document.xml")
    tree = ET.fromstring(xml_content)
    paras = []
    for p in tree.iter():
        if p.tag.endswith("}p"):
            p_text = "".join([t.text for t in p.iter() if t.tag.endswith("}t") and t.text])
            if p_text.strip():
                paras.append(p_text.strip())
    return "\n".join(paras)

def unpack_zip_archive(zip_path):
    """
    Extracts all .docx files from zip archive and saves them as .txt files in extracted_data/.
    """
    if not os.path.exists(zip_path):
        logger.warning("Zip path not found: %s", zip_path)
        return
        
    logger.info("Extracting .docx files from archive: %s", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".docx"):
                continue
                
            # Create standardized text filename
            safe_name = name.replace(".docx", "").replace(" ", "_").replace("(", "_").replace(")", "_")
            safe_name = re.sub(r"_+", "_", safe_name)
            txt_filename = f"{safe_name}.txt"
            target_path = os.path.join(EXTRACTED_DIR, txt_filename)
            
            logger.info("Extracting: %s -> %s", name, txt_filename)
            docx_bytes = zf.read(name)
            content = extract_docx_to_text(docx_bytes)
            
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("Saved %s (%d chars)", txt_filename, len(content))

def extract_json_objects_from_file(filepath):
    valid_objs = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    lines = content.split('\n')
    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue
        matches = re.findall(r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})', line_s)
        for m in matches:
            if '"' in m and ':' in m:
                try:
                    obj = json.loads(m)
                    if isinstance(obj, dict):
                        valid_objs.append(obj)
                except Exception:
                    pass
    return valid_objs

def build_ocr_vocab_from_dataset(ocr_full_path, vocab_path):
    """
    Rebuilds complete OCR character vocabulary from all unique characters in the OCR dataset.
    """
    chars = set()
    with open(ocr_full_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                text = item.get("text", "")
                chars.update(list(text))
            except Exception:
                pass
                
    sorted_chars = sorted(list(chars))
    vocab = ["<blank>", "<unk>"] + sorted_chars
    char_to_idx = {c: i for i, c in enumerate(vocab)}
    idx_to_char = {i: c for i, c in enumerate(vocab)}
    
    atomik_json_yoz(vocab_path, {"char_to_idx": char_to_idx, "idx_to_char": idx_to_char})
    logger.info("OCR Vocabulary built: %d unique tokens. Saved to %s", len(vocab), vocab_path)
    return char_to_idx, idx_to_char

def process_all_datasets(download_zip=None):
    if download_zip and os.path.exists(download_zip):
        unpack_zip_archive(download_zip)

    logger.info("Scanning and categorizing all files in %s...", EXTRACTED_DIR)
    all_files = sorted(os.listdir(EXTRACTED_DIR))
    
    detection_data = []
    ocr_data = []
    inpaint_data = []
    
    for fname in all_files:
        if not fname.endswith(".txt"):
            continue
            
        fpath = os.path.join(EXTRACTED_DIR, fname)
        objs = extract_json_objects_from_file(fpath)
        logger.info("File %s -> %d JSON objects parsed", fname, len(objs))
        
        # Categorize
        det_objs = [o for o in objs if "annotations" in o or ("bbox" in o and "page_id" in o)]
        ocr_objs = [o for o in objs if "text" in o and ("crop_id" in o or "language" in o or "direction" in o)]
        inp_objs = [o for o in objs if "triplet_id" in o or ("raw_image" in o and "mask" in o)]
        
        if det_objs:
            detection_data.extend(det_objs)
            logger.info("  + %d Detection objects added from %s", len(det_objs), fname)
        if ocr_objs:
            ocr_data.extend(ocr_objs)
            logger.info("  + %d OCR objects added from %s", len(ocr_objs), fname)
        if inp_objs:
            inpaint_data.extend(inp_objs)
            logger.info("  + %d Inpainting objects added from %s", len(inp_objs), fname)

    def split_and_save(data_list, task_name, id_key):
        unique_data = []
        seen = set()
        for item in data_list:
            key = item.get(id_key) or json.dumps(item, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
                
        random.shuffle(unique_data)
        n = len(unique_data)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        
        train_set = unique_data[:n_train]
        val_set = unique_data[n_train:n_train + n_val]
        test_set = unique_data[n_train + n_val:]
        
        task_dir = os.path.join(DATA_OUT_DIR, task_name)
        for s_name, s_data in [("train", train_set), ("val", val_set), ("test", test_set), ("full", unique_data)]:
            out_path = os.path.join(task_dir, f"{s_name}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for item in s_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    
        logger.info("[%s] Total Unique: %d (Train: %d, Val: %d, Test: %d)",
                    task_name.upper(), n, len(train_set), len(val_set), len(test_set))
        return n

    n_det = split_and_save(detection_data, "detection", "page_id")
    n_ocr = split_and_save(ocr_data, "ocr", "crop_id")
    n_inp = split_and_save(inpaint_data, "inpainting", "triplet_id")
    
    # Rebuild OCR vocab
    ocr_full_path = os.path.join(DATA_OUT_DIR, "ocr", "full.jsonl")
    build_ocr_vocab_from_dataset(ocr_full_path, VOCAB_PATH)
    
    logger.info("🎉 All datasets expanded and updated successfully!")
    logger.info("Detection: %d samples | OCR: %d samples | Inpainting: %d samples", n_det, n_ocr, n_inp)

if __name__ == "__main__":
    zip_arg = sys.argv[1] if len(sys.argv) > 1 else None
    process_all_datasets(zip_arg)
