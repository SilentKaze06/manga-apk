import os
import sys
import json
import random
import unicodedata
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import ml_paths
ml_paths.ensure_ml_paths()

from utils import _has_glyph_coverage, atomik_json_yoz
from dataset_loader import prepare_ocr_image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
random.seed(42)
np.random.seed(42)

OCR_DIR = os.path.join(BASE_DIR, "manga_ml_datasets", "ocr")
CROPS_DIR = os.path.join(OCR_DIR, "crops")
VOCAB_PATH = ml_paths.resolve_vocab_path(BASE_DIR)
VOCAB_OUT_PATH = os.path.join(BASE_DIR, "ocr_vocab.json")

os.makedirs(CROPS_DIR, exist_ok=True)

WIN_FONTS_DIR = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
SERVER_FONTS_DIR = os.path.join(ml_paths.SERVER_DIR, "fonts")

FONT_CANDIDATES = [
    os.path.join(WIN_FONTS_DIR, "malgun.ttf"),
    os.path.join(WIN_FONTS_DIR, "malgunbd.ttf"),
    os.path.join(WIN_FONTS_DIR, "msyh.ttc"),
    os.path.join(WIN_FONTS_DIR, "msyhbd.ttc"),
    os.path.join(WIN_FONTS_DIR, "simsun.ttc"),
    os.path.join(WIN_FONTS_DIR, "msjh.ttc"),
    os.path.join(WIN_FONTS_DIR, "YuGothB.ttc"),
    os.path.join(WIN_FONTS_DIR, "YuGothM.ttc"),
    os.path.join(WIN_FONTS_DIR, "msgothic.ttc"),
    os.path.join(WIN_FONTS_DIR, "arialbd.ttf"),
    os.path.join(WIN_FONTS_DIR, "arial.ttf"),
    os.path.join(WIN_FONTS_DIR, "segoeuib.ttf"),
    os.path.join(WIN_FONTS_DIR, "segoeui.ttf"),
    os.path.join(SERVER_FONTS_DIR, "arialbd.ttf"),
    os.path.join(SERVER_FONTS_DIR, "arial.ttf"),
    os.path.join(SERVER_FONTS_DIR, "Pangolin-Regular.ttf"),
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
VALID_FONTS = [f for f in FONT_CANDIDATES if os.path.exists(f)]

VALID_LANGUAGES = {"ja", "ko", "zh", "en", "ru", "uz"}


def detect_text_language(text, fallback="en"):
    """Determine clean ISO language code for a crop; never returns 'horizontal'."""
    for ch in text:
        if "\uac00" <= ch <= "\ud7af" or "\u1100" <= ch <= "\u11ff":
            return "ko"
    has_kana = any("\u3040" <= ch <= "\u309f" or "\u30a0" <= ch <= "\u30ff" or ch == "\u30fc" for ch in text)
    if has_kana:
        return "ja"
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return "zh"
    if any("\u0400" <= ch <= "\u04ff" for ch in text):
        return "ru"
    if any(ch.isalpha() for ch in text):
        return fallback if fallback in VALID_LANGUAGES else "en"
    return fallback if fallback in VALID_LANGUAGES else "en"


def get_font_for_text(text, size=22):
    """Return a PIL font with 100% glyph coverage for `text`, or None if unavailable."""
    if not text or not VALID_FONTS:
        return None
    candidates = list(VALID_FONTS)
    random.shuffle(candidates)
    for fpath in candidates:
        try:
            font = ImageFont.truetype(fpath, size)
            if _has_glyph_coverage(font, text):
                return font
        except Exception:
            continue
    return None


def generate_screentone_bg(w, h):
    """Generates authentic manga texture: white, off-white, halftone, screentone dots, or dark SFX."""
    bg_type = random.choice(["pure_white", "off_white", "cream", "halftone", "screentone_dots", "dark_sfx"])

    if bg_type == "pure_white":
        return Image.new("RGB", (w, h), (255, 255, 255)), (0, 0, 0)
    if bg_type == "off_white":
        val = random.randint(235, 252)
        return Image.new("RGB", (w, h), (val, val, val)), (random.randint(0, 25), random.randint(0, 25), random.randint(0, 25))
    if bg_type == "cream":
        return Image.new("RGB", (w, h), (250, 243, 222)), (15, 15, 20)
    if bg_type == "halftone":
        arr = np.ones((h, w, 3), dtype=np.uint8) * 255
        step = random.choice([4, 6, 8])
        arr[::step, ::step] = random.randint(160, 210)
        return Image.fromarray(arr), (10, 10, 10)
    if bg_type == "screentone_dots":
        arr = np.ones((h, w, 3), dtype=np.uint8) * random.randint(245, 255)
        dot_step = random.choice([5, 7])
        arr[::dot_step, 1::dot_step] = random.randint(180, 220)
        return Image.fromarray(arr), (0, 0, 0)

    arr = np.ones((h, w, 3), dtype=np.uint8) * random.randint(25, 45)
    step = 4
    arr[::step, ::step] = random.randint(60, 90)
    return Image.fromarray(arr), (250, 250, 250)


def _wrap_two_lines(text):
    """Split text into two balanced lines near the midpoint (preferring whitespace)."""
    if " " in text.strip():
        words = text.strip().split()
        if len(words) >= 2:
            mid = len(text) // 2
            best_idx = 1
            best_diff = 10**9
            cur_len = 0
            for i in range(1, len(words)):
                cur_len += len(words[i - 1]) + (1 if i > 1 else 0)
                diff = abs(cur_len - mid)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            return [" ".join(words[:best_idx]), " ".join(words[best_idx:])]
    mid = max(1, len(text) // 2)
    return [text[:mid], text[mid:]]


def render_synthetic_crop(text, target_size=(256, 64), direction="horizontal"):
    """Render text with glyph-verified font, zero clipping, and proper vertical rotation.

    Returns None if no installed font supports all glyphs in `text` (never draws tofu boxes).
    """
    target_w, target_h = target_size
    target_w = max(64, ((int(target_w) + 7) // 8) * 8)

    if direction == "vertical":
        # Draw stacked characters top-to-bottom on a vertical column, then rotate 90 deg CCW
        # so top-to-bottom reads left-to-right for CTC alignment without squishing.
        font_size = random.randint(18, 22)
        probe_font = get_font_for_text(text, font_size)
        if probe_font is None:
            return None
        chars = [c for c in text if not c.isspace()]
        if not chars:
            return None
        col_w = 64
        line_step = font_size + 4
        strip_h = max(target_w, len(chars) * line_step + 16)
        img, text_color = generate_screentone_bg(col_w, strip_h)
        draw = ImageDraw.Draw(img)
        cur_y = max(6, (strip_h - len(chars) * line_step) // 2)
        for ch in chars:
            cb = draw.textbbox((0, 0), ch, font=probe_font)
            cw = max(1, cb[2] - cb[0])
            cx = max(4, (col_w - cw) // 2 - cb[0])
            draw.text((cx, cur_y - cb[1]), ch, font=probe_font, fill=text_color)
            cur_y += line_step
        rotated = img.transpose(Image.Transpose.ROTATE_90)
        out_img = prepare_ocr_image(rotated, target_h=target_h, target_w=target_w, direction="horizontal")
        if random.random() < 0.25:
            out_img = out_img.filter(ImageFilter.GaussianBlur(radius=0.4))
        return out_img

    img, text_color = generate_screentone_bg(target_w, target_h)
    draw = ImageDraw.Draw(img)

    font_size = random.randint(18, 24)
    font = get_font_for_text(text, font_size)
    if font is None:
        return None

    lines = [text]
    max_w = target_w - 12
    max_h = target_h - 8

    # Shrink font size first to avoid clipping; wrap into 2 lines if still wider than canvas.
    while font_size > 11:
        bbox = draw.textbbox((0, 0), text, font=font)
        if (bbox[2] - bbox[0]) <= max_w and (bbox[3] - bbox[1]) <= max_h:
            break
        font_size -= 2
        next_font = get_font_for_text(text, font_size)
        if next_font is not None:
            font = next_font

    bbox = draw.textbbox((0, 0), text, font=font)
    if (bbox[2] - bbox[0]) > max_w and len(text) >= 4:
        lines = _wrap_two_lines(text)
        while font_size > 9:
            line_boxes = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
            max_lw = max(b[2] - b[0] for b in line_boxes)
            tot_lh = sum(max(1, b[3] - b[1]) for b in line_boxes) + 4
            if max_lw <= max_w and tot_lh <= max_h:
                break
            font_size -= 1
            next_font = get_font_for_text(text, font_size)
            if next_font is not None:
                font = next_font

    line_boxes = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
    line_heights = [max(1, b[3] - b[1]) for b in line_boxes]
    gap = 4 if len(lines) > 1 else 0
    total_h = sum(line_heights) + gap * (len(lines) - 1)
    cur_y = max(3, (target_h - total_h) // 2)

    for ln, lb, lh in zip(lines, line_boxes, line_heights):
        lw = max(1, lb[2] - lb[0])
        pos_x = max(4, min(target_w - lw - 4, (target_w - lw) // 2)) - lb[0]
        pos_y = cur_y - lb[1]
        if random.random() < 0.35 and text_color == (0, 0, 0):
            draw.text((pos_x, pos_y), ln, font=font, fill=(0, 0, 0), stroke_width=1, stroke_fill=(30, 30, 30))
        else:
            draw.text((pos_x, pos_y), ln, font=font, fill=text_color)
        cur_y += lh + gap

    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.4))

    return img


def partition_records_grouped(records, train_ratio=0.80, val_ratio=0.10, seed=42):
    """Partition OCR records into disjoint train/val/test splits grouped by BOTH source_page and text.

    Guarantees zero source-page leakage and zero phrase leakage across splits.
    """
    n = len(records)
    if n == 0:
        return [], [], []

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_text = {}
    by_page = {}
    for idx, rec in enumerate(records):
        norm_text = unicodedata.normalize("NFKC", rec.get("text", "").strip()).lower()
        if norm_text:
            if norm_text in by_text:
                union(idx, by_text[norm_text])
            else:
                by_text[norm_text] = idx
        src_page = rec.get("source_page")
        if src_page:
            if src_page in by_page:
                union(idx, by_page[src_page])
            else:
                by_page[src_page] = idx

    groups = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(records[idx])

    group_list = list(groups.values())
    rng = random.Random(seed)
    rng.shuffle(group_list)

    target_train = int(round(n * train_ratio))
    target_val = int(round(n * val_ratio))

    train_split, val_split, test_split = [], [], []
    for grp in group_list:
        if len(train_split) < target_train or not train_split:
            train_split.extend(grp)
        elif len(val_split) < target_val or (len(group_list) >= 3 and not val_split):
            val_split.extend(grp)
        else:
            test_split.extend(grp)

    return train_split, val_split, test_split


def main():
    print("=" * 65)
    print("BUILDING HIGH-QUALITY GLYPH-VERIFIED MANGA OCR DATASET")
    print("=" * 65)

    all_records = []
    seen_page_text = set()
    crop_counter = 1

    # -------------------------------------------------------------
    # 1. EXTRACT HIGH-CONFIDENCE REAL CROPS FROM AVAILABLE MANGA PAGES
    # -------------------------------------------------------------
    print("\n[Step 1] Extracting real text crops from available manga pages...")
    manga_pages = []
    for search_dir in (
        os.path.join(BASE_DIR, "test_chapter_57"),
        os.path.join(BASE_DIR, "self_learning_data", "detection", "pages"),
    ):
        if os.path.isdir(search_dir):
            for f in sorted(os.listdir(search_dir)):
                fp = os.path.join(search_dir, f)
                if f.lower().endswith(('.jpg', '.png')) and os.path.getsize(fp) > 50000:
                    manga_pages.append(fp)

    sample_img = os.path.join(BASE_DIR, "sample_manga.jpg")
    if os.path.exists(sample_img):
        manga_pages.append(sample_img)

    print(f"Found {len(manga_pages)} real manga pages to inspect.")

    real_crops_extracted = 0
    if manga_pages:
        import easyocr
        from ocr_engine import is_ocr_garbage_noise
        reader = easyocr.Reader(['ja', 'en'], gpu=False, verbose=False)

        for page_path in manga_pages[:35]:
            page_name = os.path.basename(page_path)
            try:
                pil_page = Image.open(page_path).convert("RGB")
                pw, ph = pil_page.size
                if pw < 100 or ph < 100:
                    continue
                np_page = np.array(pil_page)
                res = reader.readtext(np_page, canvas_size=1600, mag_ratio=1.2, low_text=0.25)
                for box, text, conf in res:
                    text = unicodedata.normalize("NFKC", text.strip())
                    if conf < 0.70 or len(text) < 2:
                        continue
                    lang = detect_text_language(text, fallback="en")
                    if is_ocr_garbage_noise(text, source_lang=lang, is_inside_bubble=True):
                        continue
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    bx1, by1 = max(0, int(min(xs))), max(0, int(min(ys)))
                    bx2, by2 = min(pw, int(max(xs))), min(ph, int(max(ys)))
                    bw, bh = bx2 - bx1, by2 - by1
                    if bw < 12 or bh < 12:
                        continue
                    if (page_name, text) in seen_page_text:
                        continue
                    seen_page_text.add((page_name, text))

                    pad = 4
                    cx1, cy1 = max(0, bx1 - pad), max(0, by1 - pad)
                    cx2, cy2 = min(pw, bx2 + pad), min(ph, by2 + pad)
                    crop_img = pil_page.crop((cx1, cy1, cx2, cy2))

                    direction = "vertical" if bh > bw * 1.2 else "horizontal"
                    crop_resized = prepare_ocr_image(crop_img, target_h=64, target_w=256, direction=direction)

                    crop_filename = f"real_crop_{crop_counter:05d}.png"
                    crop_filepath = os.path.join(CROPS_DIR, crop_filename)
                    crop_resized.save(crop_filepath, "PNG")

                    rec = {
                        "crop_id": f"real_manga_{crop_counter:05d}",
                        "image_path": os.path.join("manga_ml_datasets", "ocr", "crops", crop_filename),
                        "text": text,
                        "language": lang,
                        "direction": direction,
                        "source_page": page_name,
                        "source_type": "real_manga_page",
                        "confidence": round(float(conf), 3),
                        "char_count": len(text)
                    }
                    all_records.append(rec)
                    crop_counter += 1
                    real_crops_extracted += 1
            except Exception as e:
                print(f"  Warning on page {page_path}: {e}")

    print(f"Extracted {real_crops_extracted} high-confidence real manga text crops.")

    # -------------------------------------------------------------
    # 2. GENERATE GLYPH-VERIFIED SYNTHETIC MANGA TYPOGRAPHY CROPS
    # -------------------------------------------------------------
    print("\n[Step 2] Generating glyph-verified manga screentone typography crops...")

    backup_file = os.path.join(BASE_DIR, "backup_old_datasets", "ocr", "full.jsonl")
    candidate_map = {}
    if os.path.exists(backup_file):
        with open(backup_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    t = unicodedata.normalize("NFKC", d.get("text", "").strip())
                    if t and len(t) <= 24:
                        lang = d.get("language", "")
                        if lang not in VALID_LANGUAGES:
                            lang = detect_text_language(t, fallback="ja")
                        direction = d.get("direction", "horizontal")
                        if direction not in {"horizontal", "vertical"}:
                            direction = "horizontal"
                        candidate_map.setdefault(t, (t, lang, direction))
                except Exception:
                    pass

    essential_phrases = [
        ("どうしたの？", "ja", "horizontal"),
        ("ほんとうに？", "ja", "horizontal"),
        ("ありがとう！", "ja", "horizontal"),
        ("お前は誰だ？！", "ja", "horizontal"),
        ("行こうぜ！", "ja", "horizontal"),
        ("なんだこれは…", "ja", "vertical"),
        ("待て！", "ja", "vertical"),
        ("信じられない！", "ja", "vertical"),
        ("大丈夫ですか？", "ja", "horizontal"),
        ("やったー！！", "ja", "horizontal"),
        ("ゴゴゴゴゴ…", "ja", "vertical"),
        ("ドカーン！！", "ja", "horizontal"),
        ("バキッ！！", "ja", "horizontal"),
        ("ザッ…", "ja", "vertical"),
        ("WHOAAAAA!", "en", "horizontal"),
        ("LOOK OUT!", "en", "horizontal"),
        ("WHAT IS THAT?!", "en", "horizontal"),
        ("GET DOWN NOW!", "en", "horizontal"),
        ("HE IS FAST!", "en", "horizontal"),
        ("DON'T MOVE!", "en", "horizontal"),
        ("YES, SIR!", "en", "horizontal"),
        ("NO WAY...", "en", "horizontal"),
        ("I PROMISE.", "en", "horizontal"),
        ("진짜 뭐야?!", "ko", "horizontal"),
        ("잠깐만 기다려!", "ko", "horizontal"),
        ("믿을 수 없어!", "ko", "vertical"),
        ("도망쳐! 빨리!", "ko", "horizontal"),
        ("안녕하세요!", "ko", "horizontal"),
        ("怎么可能？！", "zh", "horizontal"),
        ("快逃吧！", "zh", "vertical"),
        ("你是谁？", "zh", "horizontal"),
        ("太厉害了！", "zh", "horizontal"),
        ("Привет, мир!", "ru", "horizontal"),
        ("Что происходит?!", "ru", "horizontal"),
        ("Осторожно!", "ru", "horizontal"),
        ("Не может быть!", "ru", "horizontal"),
        ("Nima gap?!", "uz", "horizontal"),
        ("Tezroq yugur!", "uz", "horizontal"),
    ]
    for t, lang, direction in essential_phrases:
        norm_t = unicodedata.normalize("NFKC", t.strip())
        candidate_map.setdefault(norm_t, (norm_t, lang, direction))

    candidate_texts = list(candidate_map.values())
    random.shuffle(candidate_texts)

    synth_created = 0
    skipped_missing_glyph = 0
    for text, lang, direction in candidate_texts[:3500]:
        required_steps = len(text) + sum(a == b for a, b in zip(text, text[1:]))
        if required_steps > 32:
            continue
        crop_img = render_synthetic_crop(text, target_size=(256, 64), direction=direction)
        if crop_img is None:
            skipped_missing_glyph += 1
            continue
        crop_filename = f"synth_crop_{crop_counter:05d}.png"
        crop_filepath = os.path.join(CROPS_DIR, crop_filename)
        crop_img.save(crop_filepath, "PNG")

        rec = {
            "crop_id": f"synth_manga_{crop_counter:05d}",
            "image_path": os.path.join("manga_ml_datasets", "ocr", "crops", crop_filename),
            "text": text,
            "language": lang if lang in VALID_LANGUAGES else detect_text_language(text),
            "direction": direction,
            "source_type": "screentone_typography",
            "char_count": len(text)
        }
        all_records.append(rec)
        crop_counter += 1
        synth_created += 1

    print(f"Generated {synth_created} glyph-verified crops (skipped {skipped_missing_glyph} unsupported glyph phrases).")

    # -------------------------------------------------------------
    # 3. VERIFY 100% OF GENERATED IMAGES EXIST
    # -------------------------------------------------------------
    verified = sum(1 for rec in all_records if os.path.exists(os.path.join(BASE_DIR, rec["image_path"])))
    assert verified == len(all_records), "Every single image must exist!"

    # -------------------------------------------------------------
    # 4. GROUPED PARTITION INTO TRAIN (80%), VAL (10%), TEST (10%)
    # -------------------------------------------------------------
    train_split, val_split, test_split = partition_records_grouped(all_records, 0.80, 0.10, seed=42)

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
    char_to_idx = {}
    idx_to_char = {}
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        char_to_idx = dict(vocab.get("char_to_idx", {}))
        idx_to_char = dict(vocab.get("idx_to_char", {}))

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

    atomik_json_yoz(VOCAB_OUT_PATH, {"char_to_idx": char_to_idx, "idx_to_char": idx_to_char})
    print(f"Vocabulary synchronized! Total tokens: {len(char_to_idx)} (Added: {new_chars_added}). Saved to {VOCAB_OUT_PATH}")


if __name__ == "__main__":
    main()
