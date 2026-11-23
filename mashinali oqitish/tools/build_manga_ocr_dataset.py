import os
import sys
import json
import random
import unicodedata
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

sys.stdout.reconfigure(encoding='utf-8')
random.seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OCR_DIR = os.path.join(BASE_DIR, "manga_ml_datasets", "ocr")
CROPS_DIR = os.path.join(OCR_DIR, "crops")
VOCAB_PATH = os.path.join(BASE_DIR, "ocr_vocab.json")

os.makedirs(CROPS_DIR, exist_ok=True)

# 1. Available Fonts
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\YuGothB.ttc",
    r"C:\Windows\Fonts\YuGothM.ttc",
    r"C:\Windows\Fonts\msgothic.ttc",
    r"C:\Windows\Fonts\GOTHICB.TTF",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\arial.ttf"
]
VALID_FONTS = [f for f in FONT_CANDIDATES if os.path.exists(f)]
if not VALID_FONTS:
    VALID_FONTS = [None]

def get_random_font(size=22):
    fpath = random.choice(VALID_FONTS)
    if fpath:
        try:
            return ImageFont.truetype(fpath, size)
        except Exception:
            pass
    return ImageFont.load_default()

def generate_screentone_bg(w, h):
    """Generates authentic manga texture: dots, hatching, or speech bubble white."""
    bg_type = random.choice(["pure_white", "off_white", "halftone", "screentone_dots", "dark_sfx"])
    
    if bg_type == "pure_white":
        return Image.new("RGB", (w, h), (255, 255, 255)), (0, 0, 0)
    elif bg_type == "off_white":
        val = random.randint(245, 253)
        return Image.new("RGB", (w, h), (val, val, val)), (random.randint(0, 25), random.randint(0, 25), random.randint(0, 25))
    elif bg_type == "halftone":
        arr = np.ones((h, w, 3), dtype=np.uint8) * 255
        step = random.choice([4, 6, 8])
        arr[::step, ::step] = random.randint(160, 210)
        return Image.fromarray(arr), (10, 10, 10)
    elif bg_type == "screentone_dots":
        arr = np.ones((h, w, 3), dtype=np.uint8) * random.randint(248, 255)
        dot_step = random.choice([5, 7])
        arr[::dot_step, 1::dot_step] = random.randint(180, 220)
        return Image.fromarray(arr), (0, 0, 0)
    else:  # dark_sfx: inverted text (white text on dark screentone)
        arr = np.ones((h, w, 3), dtype=np.uint8) * random.randint(25, 45)
        step = 4
        arr[::step, ::step] = random.randint(60, 90)
        return Image.fromarray(arr), (250, 250, 250)

def render_synthetic_crop(text, target_size=(256, 64)):
    """Renders text with authentic manga styling, font, screentone, and padding."""
    target_w, target_h = target_size
    img, text_color = generate_screentone_bg(target_w, target_h)
    draw = ImageDraw.Draw(img)
    
    font_size = random.randint(18, 24)
    font = get_random_font(font_size)
    
    # Calculate text positioning
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    
    # Center text with slight random jitter
    pos_x = max(6, (target_w - tw) // 2 + random.randint(-4, 4))
    pos_y = max(4, (target_h - th) // 2 + random.randint(-2, 2))
    
    # Optional stroke/outline for comic feel
    if random.random() < 0.35 and text_color == (0, 0, 0):
        draw.text((pos_x, pos_y), text, font=font, fill=(0, 0, 0), stroke_width=1, stroke_fill=(30, 30, 30))
    else:
        draw.text((pos_x, pos_y), text, font=font, fill=text_color)
        
    # Optional subtle noise or blur simulating scan quality
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
        
    return img

def main():
    print("=" * 65)
    print("BUILDING NEW HIGH-QUALITY REAL MANGA OCR DATASET")
    print("=" * 65)

    all_records = []
    crop_counter = 1

    # -------------------------------------------------------------
    # 1. EXTRACT REAL CROPS FROM ACTUAL MANGA PAGES USING EASYOCR
    # -------------------------------------------------------------
    print("\n[Step 1] Extracting real text crops from available manga pages...")
    manga_pages = []
    
    # test_chapter_57 pages
    ch57_dir = os.path.join(BASE_DIR, "test_chapter_57")
    if os.path.exists(ch57_dir):
        for f in os.listdir(ch57_dir):
            if f.lower().endswith(('.jpg', '.png')):
                manga_pages.append(os.path.join(ch57_dir, f))
                
    # sample_manga.jpg
    sample_img = os.path.join(BASE_DIR, "sample_manga.jpg")
    if os.path.exists(sample_img):
        manga_pages.append(sample_img)
        
    # self_learning_data/detection/pages
    self_det_dir = os.path.join(BASE_DIR, "self_learning_data", "detection", "pages")
    if os.path.exists(self_det_dir):
        for f in os.listdir(self_det_dir):
            if f.lower().endswith(('.jpg', '.png')) and os.path.getsize(os.path.join(self_det_dir, f)) > 50000:
                manga_pages.append(os.path.join(self_det_dir, f))

    print(f"Found {len(manga_pages)} real manga pages to inspect.")
    
    import easyocr
    reader = easyocr.Reader(['ja', 'en'], gpu=False, verbose=False)
    
    real_crops_extracted = 0
    for page_idx, page_path in enumerate(manga_pages[:35], start=1):
        try:
            pil_page = Image.open(page_path).convert("RGB")
            pw, ph = pil_page.size
            if pw < 100 or ph < 100:
                continue
            np_page = np.array(pil_page)
            # High-accuracy OCR extraction on manga page
            res = reader.readtext(np_page, canvas_size=1600, mag_ratio=1.2, low_text=0.25)
            for box, text, conf in res:
                text = unicodedata.normalize("NFKC", text.strip())
                if conf < 0.35 or len(text) < 2:
                    continue
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                bx1, by1 = max(0, int(min(xs))), max(0, int(min(ys)))
                bx2, by2 = min(pw, int(max(xs))), min(ph, int(max(ys)))
                bw, bh = bx2 - bx1, by2 - by1
                if bw < 12 or bh < 12:
                    continue
                    
                pad = 4
                cx1, cy1 = max(0, bx1 - pad), max(0, by1 - pad)
                cx2, cy2 = min(pw, bx2 + pad), min(ph, by2 + pad)
                crop_img = pil_page.crop((cx1, cy1, cx2, cy2))
                
                # Standardize to 256x64 format
                crop_resized = crop_img.resize((256, 64), Image.Resampling.BILINEAR)
                
                crop_filename = f"real_crop_{crop_counter:05d}.png"
                crop_filepath = os.path.join(CROPS_DIR, crop_filename)
                crop_resized.save(crop_filepath, "PNG")
                
                is_ja = any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' for c in text)
                lang = "ja" if is_ja else "en"
                direction = "vertical" if bh > bw * 1.2 else "horizontal"
                
                rec = {
                    "crop_id": f"real_manga_{crop_counter:05d}",
                    "image_path": os.path.join("manga_ml_datasets", "ocr", "crops", crop_filename),
                    "text": text,
                    "language": lang,
                    "direction": direction,
                    "source_type": "real_manga_page",
                    "confidence": round(float(conf), 3),
                    "char_count": len(text)
                }
                all_records.append(rec)
                crop_counter += 1
                real_crops_extracted += 1
        except Exception as e:
            print(f"  Warning on page {page_path}: {e}")
            
    print(f"Extracted {real_crops_extracted} genuine real manga text crops with images saved to disk.")

    # -------------------------------------------------------------
    # 2. GENERATE HIGH-QUALITY SYNTHETIC MANGA TYPOGRAPHY CROPS
    # -------------------------------------------------------------
    print("\n[Step 2] Generating authentic manga screentone typography crops...")
    
    # Read text sentences from backup old dataset or authentic manga phrases
    backup_file = os.path.join(BASE_DIR, "backup_old_datasets", "ocr", "full.jsonl")
    candidate_texts = []
    if os.path.exists(backup_file):
        with open(backup_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    t = d.get("text", "").strip()
                    if t and len(t) <= 24: # Fit safely in T=32 CTC alignment
                        candidate_texts.append((t, d.get("language", "ja"), d.get("direction", "horizontal")))
                except Exception:
                    pass

    # Ensure rich Japanese, English, and SFX coverage
    essential_phrases = [
        ("どうしたの？", "ja", "horizontal"),
        ("ほんとうに？", "ja", "horizontal"),
        ("ありがとう！", "ja", "horizontal"),
        ("お前は誰だ？！", "ja", "horizontal"),
        ("行こうぜ！", "ja", "horizontal"),
        ("なんだこれは…", "ja", "horizontal"),
        ("待て！", "ja", "horizontal"),
        ("信じられない！", "ja", "horizontal"),
        ("大丈夫ですか？", "ja", "horizontal"),
        ("やったー！！", "ja", "horizontal"),
        ("ゴゴゴゴゴ…", "ja", "horizontal"),
        ("ドカーン！！", "ja", "horizontal"),
        ("バキッ！！", "ja", "horizontal"),
        ("ザッ…", "ja", "horizontal"),
        ("WHOAAAAA!", "en", "horizontal"),
        ("LOOK OUT!", "en", "horizontal"),
        ("WHAT IS THAT?!", "en", "horizontal"),
        ("GET DOWN NOW!", "en", "horizontal"),
        ("HE IS FAST!", "en", "horizontal"),
        ("DON'T MOVE!", "en", "horizontal"),
        ("TA-PAAA!", "en", "horizontal"),
        ("KRAAA-BOOM!", "en", "horizontal"),
        ("YOU RELAX, OKAY?", "en", "horizontal"),
        ("YES, SIR!", "en", "horizontal"),
        ("NO WAY...", "en", "horizontal"),
        ("I PROMISE.", "en", "horizontal")
    ]
    candidate_texts.extend(essential_phrases * 10)
    random.shuffle(candidate_texts)
    
    # Cap total synthetic samples to 3,500 diverse samples
    target_synth_count = 3500
    synth_samples = candidate_texts[:target_synth_count]
    
    synth_created = 0
    for text, lang, direction in synth_samples:
        clean_t = unicodedata.normalize("NFKC", text.strip())
        if not clean_t:
            continue
            
        crop_img = render_synthetic_crop(clean_t, target_size=(256, 64))
        crop_filename = f"synth_crop_{crop_counter:05d}.png"
        crop_filepath = os.path.join(CROPS_DIR, crop_filename)
        crop_img.save(crop_filepath, "PNG")
        
        rec = {
            "crop_id": f"synth_manga_{crop_counter:05d}",
            "image_path": os.path.join("manga_ml_datasets", "ocr", "crops", crop_filename),
            "text": clean_t,
            "language": lang,
            "direction": direction,
            "source_type": "screentone_typography",
            "char_count": len(clean_t)
        }
        all_records.append(rec)
        crop_counter += 1
        synth_created += 1

    print(f"Generated {synth_created} authentic screentone typography crops with images saved to disk.")
    print(f"Total new OCR dataset size: {len(all_records)} samples (100% with real PNG files on disk).")

    # -------------------------------------------------------------
    # 3. VERIFY 100% OF GENERATED IMAGES EXIST
    # -------------------------------------------------------------
    print("\n[Step 3] Verifying 100% file existence on disk...")
    verified = 0
    for rec in all_records:
        full_p = os.path.join(BASE_DIR, rec["image_path"])
        if os.path.exists(full_p):
            verified += 1
    print(f"Verification result: {verified}/{len(all_records)} images confirmed on disk!")
    assert verified == len(all_records), "Every single image must exist!"

    # -------------------------------------------------------------
    # 4. PARTITION INTO TRAIN (80%), VAL (10%), TEST (10%)
    # -------------------------------------------------------------
    print("\n[Step 4] Partitioning into train (80%), val (10%), test (10%)...")
    random.shuffle(all_records)
    n = len(all_records)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)
    
    train_split = all_records[:n_train]
    val_split = all_records[n_train:n_train + n_val]
    test_split = all_records[n_train + n_val:]
    
    def write_jsonl(filepath, records):
        with open(filepath, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
    write_jsonl(os.path.join(OCR_DIR, "full.jsonl"), all_records)
    write_jsonl(os.path.join(OCR_DIR, "train.jsonl"), train_split)
    write_jsonl(os.path.join(OCR_DIR, "val.jsonl"), val_split)
    write_jsonl(os.path.join(OCR_DIR, "test.jsonl"), test_split)
    
    print(f"Saved full.jsonl ({len(all_records)}), train.jsonl ({len(train_split)}), val.jsonl ({len(val_split)}), test.jsonl ({len(test_split)}).")

    # -------------------------------------------------------------
    # 5. SYNCHRONIZE VOCABULARY (ocr_vocab.json)
    # -------------------------------------------------------------
    print("\n[Step 5] Synchronizing vocabulary (ocr_vocab.json)...")
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
        
    char_to_idx = vocab.get("char_to_idx", {})
    idx_to_char = vocab.get("idx_to_char", {})
    
    # Ensure blank and unk tokens
    if "<blank>" not in char_to_idx:
        char_to_idx["<blank>"] = 0
        idx_to_char["0"] = "<blank>"
    if "<unk>" not in char_to_idx:
        char_to_idx["<unk>"] = 1
        idx_to_char["1"] = "<unk>"
        
    next_idx = (max(char_to_idx.values()) + 1) if char_to_idx else 0
    new_chars_added = 0
    for r in all_records:
        for ch in r["text"]:
            if ch not in char_to_idx and (ord(ch) >= 32 or ch == '\n'):
                char_to_idx[ch] = next_idx
                idx_to_char[str(next_idx)] = ch
                next_idx += 1
                new_chars_added += 1
                
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump({"char_to_idx": char_to_idx, "idx_to_char": idx_to_char}, f, ensure_ascii=False, indent=2)
        
    print(f"Vocabulary synchronized! Total tokens: {len(char_to_idx)} (Added: {new_chars_added}).")
    print("\n" + "=" * 65)
    print("NEW MANGA OCR DATASET BUILD COMPLETE!")
    print("=" * 65)

if __name__ == "__main__":
    main()
