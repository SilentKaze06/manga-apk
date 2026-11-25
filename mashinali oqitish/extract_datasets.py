import os
import sys
import json
import re
import random
import zipfile
import xml.etree.ElementTree as ET
import io
import logging

import ml_paths
ml_paths.ensure_ml_paths()
from utils import atomik_json_yoz

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_datasets")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACTED_DIR = os.path.join(BASE_DIR, "extracted_data")
DATA_OUT_DIR = os.path.join(BASE_DIR, "manga_ml_datasets")
VOCAB_PATH = ml_paths.resolve_vocab_path(BASE_DIR)
VOCAB_OUT_PATH = os.path.join(BASE_DIR, "ocr_vocab.json")

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
        raise FileNotFoundError(f"Zip archive not found: {zip_path}")

    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    logger.info("Extracting .docx files from archive: %s", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".docx"):
                continue

            # Create standardized text filename
            safe_name = os.path.basename(name).replace(".docx", "").replace(" ", "_").replace("(", "_").replace(")", "_")
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
    """Parse all valid JSON dict objects from text (supports nested & multi-line JSON)."""
    valid_objs = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    decoder = json.JSONDecoder()
    idx = 0
    n = len(content)
    while idx < n:
        brace_idx = content.find("{", idx)
        if brace_idx == -1:
            break
        try:
            obj, end_idx = decoder.raw_decode(content, brace_idx)
            if isinstance(obj, dict):
                valid_objs.append(obj)
            idx = max(brace_idx + 1, end_idx)
        except json.JSONDecodeError:
            idx = brace_idx + 1
    return valid_objs


def build_ocr_vocab_from_dataset(ocr_full_path, vocab_path=VOCAB_OUT_PATH, base_vocab_path=VOCAB_PATH):
    """
    Synchronizes OCR character vocabulary from unique characters in the OCR dataset
    without renumbering existing mappings.
    """
    char_to_idx = {}
    idx_to_char = {}
    read_path = vocab_path if os.path.exists(vocab_path) else base_vocab_path
    if read_path and os.path.exists(read_path):
        try:
            with open(read_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                char_to_idx = dict(data.get("char_to_idx", {}))
                idx_to_char = {int(k): v for k, v in data.get("idx_to_char", {}).items()}
        except Exception:
            pass

    if "<blank>" not in char_to_idx:
        char_to_idx["<blank>"] = 0
        idx_to_char[0] = "<blank>"
    if "<unk>" not in char_to_idx:
        char_to_idx["<unk>"] = 1
        idx_to_char[1] = "<unk>"

    next_idx = (max(char_to_idx.values()) + 1) if char_to_idx else 0

    chars = set()
    if os.path.isfile(ocr_full_path):
        with open(ocr_full_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    text = item.get("text", "")
                    chars.update(list(text))
                except Exception:
                    pass

    if not chars and os.path.exists(vocab_path):
        logger.warning("No OCR characters found in %s; preserving existing vocabulary at %s", ocr_full_path, vocab_path)
        return char_to_idx, idx_to_char

    sorted_chars = sorted(list(chars))
    for c in sorted_chars:
        if c not in char_to_idx and (ord(c) >= 32 or c == '\n'):
            char_to_idx[c] = next_idx
            idx_to_char[next_idx] = c
            next_idx += 1

    idx_to_char_str = {str(k): v for k, v in idx_to_char.items()}
    atomik_json_yoz(vocab_path, {"char_to_idx": char_to_idx, "idx_to_char": idx_to_char_str})
    logger.info("OCR Vocabulary synchronized: %d unique tokens. Saved to %s", len(char_to_idx), vocab_path)
    return char_to_idx, idx_to_char


def process_all_datasets(download_zip=None, extracted_dir=EXTRACTED_DIR, data_out_dir=DATA_OUT_DIR, vocab_out_path=VOCAB_OUT_PATH):
    if download_zip:
        unpack_zip_archive(download_zip)

    if not os.path.isdir(extracted_dir):
        raise FileNotFoundError(
            f"Source directory '{extracted_dir}' does not exist. "
            "Pass a .zip archive or populate extracted_data/ before extracting; aborting to preserve existing datasets."
        )

    logger.info("Scanning and categorizing all files in %s...", extracted_dir)
    all_files = [f for f in sorted(os.listdir(extracted_dir)) if f.endswith(".txt")]
    if not all_files:
        raise RuntimeError(
            f"No .txt annotation files found in '{extracted_dir}'; aborting without overwriting existing datasets."
        )

    detection_data = []
    ocr_data = []
    inpaint_data = []

    for fname in all_files:
        fpath = os.path.join(extracted_dir, fname)
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

    if not (detection_data or ocr_data or inpaint_data):
        raise RuntimeError(
            f"No valid detection, OCR, or inpainting records found in '{extracted_dir}'; aborting without overwriting existing datasets."
        )

    def split_and_save(data_list, task_name, id_key):
        if not data_list:
            logger.info("[%s] No records found in extracted_data; leaving existing files untouched.", task_name.upper())
            return 0
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

        task_dir = os.path.join(data_out_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
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

    # Rebuild OCR vocab only when OCR data was actually extracted
    if n_ocr > 0:
        ocr_full_path = os.path.join(data_out_dir, "ocr", "full.jsonl")
        build_ocr_vocab_from_dataset(ocr_full_path, vocab_out_path, base_vocab_path=VOCAB_PATH)

    logger.info("All datasets expanded and updated successfully!")
    logger.info("Detection: %d samples | OCR: %d samples | Inpainting: %d samples", n_det, n_ocr, n_inp)
    return {"detection": n_det, "ocr": n_ocr, "inpainting": n_inp}

if __name__ == "__main__":
    zip_arg = sys.argv[1] if len(sys.argv) > 1 else None
    process_all_datasets(zip_arg)
