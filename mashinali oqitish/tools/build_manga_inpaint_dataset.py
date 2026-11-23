"""
High-Precision Authentic Manga Inpainting Dataset Generator.
Extracts real manga page textures and creates high-fidelity speech bubble,
screentone, and SFX inpainting triplets (raw, mask, clean) saved directly to disk.
"""

import os
import sys
import glob
import math
import json
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "manga_ml_datasets", "inpainting")
RAW_DIR = os.path.join(DATA_DIR, "raw")
MASK_DIR = os.path.join(DATA_DIR, "mask")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")

for d in [RAW_DIR, MASK_DIR, CLEAN_DIR]:
    os.makedirs(d, exist_ok=True)

# Font discovery for authentic Japanese & English manga typography
WIN_DIR = os.environ.get("WINDIR", r"C:\Windows")
FONTS_DIR = os.path.join(WIN_DIR, "Fonts")

FONT_CANDIDATES = [
    os.path.join(FONTS_DIR, "YuGothB.ttc"),
    os.path.join(FONTS_DIR, "msgothic.ttc"),
    os.path.join(FONTS_DIR, "msyh.ttc"),
    os.path.join(FONTS_DIR, "arialbd.ttf"),
    os.path.join(FONTS_DIR, "GOTHICB.TTF"),
    os.path.join(FONTS_DIR, "segoeuib.ttf"),
]

AVAILABLE_FONTS = [f for f in FONT_CANDIDATES if os.path.exists(f)]
if not AVAILABLE_FONTS:
    AVAILABLE_FONTS = [None]


def get_font(size=20):
    f_path = random.choice(AVAILABLE_FONTS)
    if f_path:
        try:
            return ImageFont.truetype(f_path, size)
        except Exception:
            pass
    return ImageFont.load_default()


# Authentic dialogue phrases for speech bubbles
JAPANESE_DIALOGUES = [
    "何だと！？", "そんなバカな…", "逃げろ！早く！", "お前は…誰だ？",
    "信じられない！", "待ってくれ！", "行くぞ、全力で！", "これで終わりだ！",
    "どうして…？", "絶対に許さない！", "大丈夫か！？", "よし、やるぞ！",
    "嘘でしょ…", "冗談じゃない！", "勝負だ！", "見つけたぞ！",
    "静かにしろ…", "何が起きたんだ？", "急ごう！時間がない", "頼んだぞ…",
    "バカな…こんな所で…", "やはりそうか", "私の勝ちだ！", "気をつけろ！",
    "まさか…そんなはずは…", "ここから出せ！", "約束したはずだ", "助けてくれ！",
    "お前の力はそんなものか？", "消えろ！", "今すぐ逃げるんだ！", "なぜだ…"
]

ENGLISH_DIALOGUES = [
    "WHAT?!", "NO WAY...", "GET OUT OF HERE!", "WHO ARE YOU?!",
    "I CAN'T BELIEVE IT!", "WAIT FOR ME!", "LET'S GO ALL OUT!",
    "THIS IS THE END!", "WHY...?", "I WON'T FORGIVE YOU!",
    "ARE YOU OKAY?!", "ALRIGHT, LET'S DO THIS!", "YOU'VE GOT TO BE KIDDING!",
    "DON'T MESS WITH ME!", "IT'S OVER NOW!", "FOUND YOU!",
    "SHUT UP!", "WHAT HAPPENED HERE?", "HURRY, NO TIME!",
    "I'M COUNTING ON YOU!", "HOW COULD THIS BE...", "JUST AS I THOUGHT.",
    "VICTORY IS MINE!", "WATCH OUT!", "IMPOSSIBLE...",
    "LET ME OUT!", "YOU PROMISED ME!", "HELP ME!",
    "IS THAT ALL YOU'VE GOT?", "DISAPPEAR!", "RUN AWAY NOW!"
]

SFX_TEXTS = [
    "ドドドド", "ゴゴゴゴ", "バーン！", "ドカン！", "バキッ", "ザッ…",
    "ズズズズ", "ガガガガ", "ハッ！", "ギャアア", "クッ…", "ダンッ！",
    "チッ", "フッ…", "ピキッ", "ドンッ！", "シャッ", "スッ…",
    "BOOM!", "CRASH!", "BAM!", "WHOOSH!", "THUD!", "SLAM!",
    "SNAP!", "CLANG!", "SLASH!", "RUMBLE...", "SHHH...", "PUNCH!"
]


def create_screentone_patch(size=256, tone_type="dots"):
    """Generates authentic manga screentone textures."""
    arr = np.ones((size, size, 3), dtype=np.uint8) * 255
    if tone_type == "dots":
        freq = random.choice([4, 6, 8, 10])
        gray_val = random.randint(110, 200)
        arr[::freq, ::freq] = gray_val
        if freq >= 6:
            arr[1::freq, 1::freq] = gray_val + 20
    elif tone_type == "hatch":
        step = random.randint(5, 9)
        gray_val = random.randint(120, 190)
        for i in range(0, size * 2, step):
            for x in range(size):
                y = i - x
                if 0 <= y < size:
                    arr[y, x] = gray_val
    elif tone_type == "speedlines":
        center_x = random.randint(size // 4, 3 * size // 4)
        center_y = random.randint(size // 4, 3 * size // 4)
        for _ in range(random.randint(25, 45)):
            angle = random.uniform(0, 2 * math.pi)
            length = random.uniform(40, size)
            x2 = int(center_x + length * math.cos(angle))
            y2 = int(center_y + length * math.sin(angle))
            cv2.line(arr, (center_x, center_y), (x2, y2), (random.randint(40, 160),) * 3, thickness=random.choice([1, 2]))
    elif tone_type == "gradient":
        grad = np.tile(np.linspace(255, random.randint(140, 210), size, dtype=np.uint8)[:, None], (1, size))
        arr = np.stack([grad, grad, grad], axis=-1)
    return Image.fromarray(arr)


def render_speech_bubble(draw, b_box, bubble_type="oval"):
    """Renders speech bubble boundary on ground-truth clean canvas."""
    x1, y1, x2, y2 = b_box
    border_w = random.choice([2, 3, 4])
    border_col = (random.randint(10, 30), random.randint(10, 30), random.randint(10, 30))
    fill_col = (random.randint(250, 255), random.randint(250, 255), random.randint(250, 255))
    
    if bubble_type == "oval":
        draw.ellipse([x1, y1, x2, y2], fill=fill_col, outline=border_col, width=border_w)
    elif bubble_type == "rect":
        draw.rounded_rectangle([x1, y1, x2, y2], radius=8, fill=fill_col, outline=border_col, width=border_w)
    elif bubble_type == "cloud":
        # Draw cloudy thought bubble via overlapping circles
        draw.ellipse([x1, y1, x2, y2], fill=fill_col, outline=border_col, width=border_w)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rw, rh = (x2 - x1) // 2, (y2 - y1) // 2
        for angle in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            px = int(cx + rw * 0.9 * math.cos(angle))
            py = int(cy + rh * 0.9 * math.sin(angle))
            draw.ellipse([px - 14, py - 14, px + 14, py + 14], fill=fill_col, outline=border_col, width=2)
    elif bubble_type == "shout":
        # Spiky shout bubble polygon
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rw, rh = (x2 - x1) // 2, (y2 - y1) // 2
        pts = []
        n_spikes = random.randint(12, 18)
        for i, angle in enumerate(np.linspace(0, 2 * math.pi, n_spikes, endpoint=False)):
            r_mult = 1.15 if i % 2 == 0 else 0.85
            px = int(cx + rw * r_mult * math.cos(angle))
            py = int(cy + rh * r_mult * math.sin(angle))
            pts.append((px, py))
        draw.polygon(pts, fill=fill_col, outline=border_col)


def make_bubble_triplet(triplet_id, bg_img=None):
    """
    Creates a triplet where clean is a manga background + speech bubble,
    raw is the speech bubble with text, and mask is the text region to inpaint.
    """
    size = 256
    if bg_img is not None:
        clean = bg_img.copy().resize((size, size), Image.Resampling.LANCZOS)
    else:
        tt = random.choice(["dots", "hatch", "speedlines", "gradient"])
        clean = create_screentone_patch(size, tt)
        
    draw_clean = ImageDraw.Draw(clean)
    
    # Random bubble dimensions
    bw = random.randint(120, 200)
    bh = random.randint(90, 180)
    bx1 = random.randint(15, size - bw - 15)
    by1 = random.randint(15, size - bh - 15)
    bx2 = bx1 + bw
    by2 = by1 + bh
    
    btype = random.choice(["oval", "oval", "rect", "cloud", "shout"])
    render_speech_bubble(draw_clean, [bx1, by1, bx2, by2], btype)
    
    # Raw image = clean + text
    raw = clean.copy()
    draw_raw = ImageDraw.Draw(raw)
    
    # Mask image = text pixels dilated
    mask = Image.new("L", (size, size), 0)
    draw_mask = ImageDraw.Draw(mask)
    
    # Select text
    is_jp = random.random() < 0.65
    dialogue_pool = JAPANESE_DIALOGUES if is_jp else ENGLISH_DIALOGUES
    text = random.choice(dialogue_pool)
    if random.random() < 0.4:
        text += "\n" + random.choice(dialogue_pool)
        
    font_size = random.randint(16, 26)
    font = get_font(font_size)
    
    # Center text inside bubble
    lines = text.split("\n")
    tot_h = len(lines) * (font_size + 4)
    cur_y = (by1 + by2) // 2 - tot_h // 2
    
    text_color = (random.randint(0, 30), random.randint(0, 30), random.randint(0, 30))
    for line in lines:
        try:
            bbox = draw_raw.textbbox((0, 0), line, font=font)
            lw = bbox[2] - bbox[0]
        except Exception:
            lw = len(line) * (font_size // 2)
            
        cur_x = (bx1 + bx2) // 2 - lw // 2
        draw_raw.text((cur_x, cur_y), line, fill=text_color, font=font)
        draw_mask.text((cur_x, cur_y), line, fill=255, font=font)
        cur_y += font_size + 4
        
    # Dilate mask slightly to match real manga text detector bboxes
    mask_arr = np.array(mask)
    k_size = random.choice([3, 5, 7])
    mask_dilated = cv2.dilate(mask_arr, np.ones((k_size, k_size), np.uint8), iterations=1)
    mask = Image.fromarray(mask_dilated)
    
    cov = float(np.count_nonzero(mask_dilated)) / float(size * size) * 100.0
    return raw, mask, clean, cov, "speech_bubble"


def make_sfx_triplet(triplet_id, bg_img=None):
    """
    Creates a triplet where clean is an action/screentone panel,
    raw has large stylized manga SFX across it, and mask covers the SFX strokes.
    """
    size = 256
    if bg_img is not None:
        clean = bg_img.copy().resize((size, size), Image.Resampling.LANCZOS)
    else:
        tt = random.choice(["dots", "hatch", "speedlines", "gradient"])
        clean = create_screentone_patch(size, tt)
        
    raw = clean.copy()
    draw_raw = ImageDraw.Draw(raw)
    
    mask = Image.new("L", (size, size), 0)
    draw_mask = ImageDraw.Draw(mask)
    
    sfx = random.choice(SFX_TEXTS)
    font_size = random.randint(28, 48)
    font = get_font(font_size)
    
    try:
        bbox = draw_raw.textbbox((0, 0), sfx, font=font)
        sw, sh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        sw, sh = len(sfx) * 22, font_size
        
    sx = random.randint(10, max(12, size - sw - 10))
    sy = random.randint(10, max(12, size - sh - 10))
    
    # SFX with thick outline (common in action manga)
    sfx_outline_col = (255, 255, 255)
    sfx_fill_col = (random.randint(0, 30), random.randint(0, 30), random.randint(0, 30))
    
    outline_r = random.choice([2, 3, 4])
    for dx in range(-outline_r, outline_r + 1):
        for dy in range(-outline_r, outline_r + 1):
            if dx * dx + dy * dy <= outline_r * outline_r:
                draw_raw.text((sx + dx, sy + dy), sfx, fill=sfx_outline_col, font=font)
                draw_mask.text((sx + dx, sy + dy), sfx, fill=255, font=font)
                
    draw_raw.text((sx, sy), sfx, fill=sfx_fill_col, font=font)
    draw_mask.text((sx, sy), sfx, fill=255, font=font)
    
    mask_arr = np.array(mask)
    mask_dilated = cv2.dilate(mask_arr, np.ones((5, 5), np.uint8), iterations=1)
    mask = Image.fromarray(mask_dilated)
    
    cov = float(np.count_nonzero(mask_dilated)) / float(size * size) * 100.0
    return raw, mask, clean, cov, "sfx"


def main():
    random.seed(42)
    np.random.seed(42)
    print("=" * 60)
    print("STARTING MANGA INPAINTING DATASET BUILDER")
    print("=" * 60)
    
    # 1. Discover all real manga page backgrounds
    real_page_paths = glob.glob(os.path.join(BASE_DIR, "test_chapter_57", "*.png"))
    real_page_paths += glob.glob(os.path.join(BASE_DIR, "test_chapter_57", "*.jpg"))
    sample_manga = os.path.join(BASE_DIR, "sample_manga.jpg")
    if os.path.exists(sample_manga):
        real_page_paths.append(sample_manga)
    det_pages = glob.glob(os.path.join(BASE_DIR, "self_learning_data", "detection", "pages", "*.png"))
    real_page_paths += det_pages
    
    print(f"Discovered {len(real_page_paths)} authentic manga pages.")
    
    # Pre-extract random 256x256 patches from real manga pages
    real_patches = []
    for p_path in real_page_paths[:25]:
        try:
            with Image.open(p_path) as im:
                p_img = im.convert("RGB")
                pw, ph = p_img.size
                if pw >= 256 and ph >= 256:
                    for _ in range(8):
                        rx = random.randint(0, pw - 256)
                        ry = random.randint(0, ph - 256)
                        patch = p_img.crop((rx, ry, rx + 256, ry + 256))
                        # Skip pure black or pure white patches
                        arr = np.array(patch.convert("L"))
                        if np.std(arr) > 10:
                            real_patches.append(patch)
        except Exception:
            continue
            
    print(f"Extracted {len(real_patches)} diverse background patches from real manga pages.")
    
    triplet_records = []
    idx = 1
    
    # Category 1: Speech Bubbles on real manga patches (~650 samples)
    n_bubble_real = 650
    print(f"Generating {n_bubble_real} real-background speech bubble triplets...")
    for _ in range(n_bubble_real):
        tid = f"inpaint_{idx:05d}"
        bg = random.choice(real_patches) if real_patches else None
        raw, mask, clean, cov, cat = make_bubble_triplet(tid, bg)
        
        raw_p = os.path.join(RAW_DIR, f"{tid}.png")
        mask_p = os.path.join(MASK_DIR, f"{tid}.png")
        clean_p = os.path.join(CLEAN_DIR, f"{tid}.png")
        
        raw.save(raw_p, "PNG")
        mask.save(mask_p, "PNG")
        clean.save(clean_p, "PNG")
        
        triplet_records.append({
            "triplet_id": tid,
            "raw_image": f"manga_ml_datasets/inpainting/raw/{tid}.png",
            "mask": f"manga_ml_datasets/inpainting/mask/{tid}.png",
            "ground_truth": f"manga_ml_datasets/inpainting/clean/{tid}.png",
            "category": cat,
            "resolution": [256, 256],
            "mask_coverage_pct": round(cov, 2)
        })
        idx += 1
        
    # Category 2: Speech Bubbles on screentones (~400 samples)
    n_bubble_syn = 400
    print(f"Generating {n_bubble_syn} screentone speech bubble triplets...")
    for _ in range(n_bubble_syn):
        tid = f"inpaint_{idx:05d}"
        raw, mask, clean, cov, cat = make_bubble_triplet(tid, bg_img=None)
        
        raw_p = os.path.join(RAW_DIR, f"{tid}.png")
        mask_p = os.path.join(MASK_DIR, f"{tid}.png")
        clean_p = os.path.join(CLEAN_DIR, f"{tid}.png")
        
        raw.save(raw_p, "PNG")
        mask.save(mask_p, "PNG")
        clean.save(clean_p, "PNG")
        
        triplet_records.append({
            "triplet_id": tid,
            "raw_image": f"manga_ml_datasets/inpainting/raw/{tid}.png",
            "mask": f"manga_ml_datasets/inpainting/mask/{tid}.png",
            "ground_truth": f"manga_ml_datasets/inpainting/clean/{tid}.png",
            "category": cat,
            "resolution": [256, 256],
            "mask_coverage_pct": round(cov, 2)
        })
        idx += 1
        
    # Category 3: Manga SFX / Sound Effect inpainting (~450 samples)
    n_sfx = 450
    print(f"Generating {n_sfx} action SFX inpainting triplets...")
    for _ in range(n_sfx):
        tid = f"inpaint_{idx:05d}"
        bg = random.choice(real_patches) if (real_patches and random.random() < 0.5) else None
        raw, mask, clean, cov, cat = make_sfx_triplet(tid, bg)
        
        raw_p = os.path.join(RAW_DIR, f"{tid}.png")
        mask_p = os.path.join(MASK_DIR, f"{tid}.png")
        clean_p = os.path.join(CLEAN_DIR, f"{tid}.png")
        
        raw.save(raw_p, "PNG")
        mask.save(mask_p, "PNG")
        clean.save(clean_p, "PNG")
        
        triplet_records.append({
            "triplet_id": tid,
            "raw_image": f"manga_ml_datasets/inpainting/raw/{tid}.png",
            "mask": f"manga_ml_datasets/inpainting/mask/{tid}.png",
            "ground_truth": f"manga_ml_datasets/inpainting/clean/{tid}.png",
            "category": cat,
            "resolution": [256, 256],
            "mask_coverage_pct": round(cov, 2)
        })
        idx += 1
        
    # Category 4: Verified real feedback triplets from self_learning_data/inpainting/
    fb_json = os.path.join(BASE_DIR, "self_learning_data", "inpainting", "feedback_inpaint.jsonl")
    fb_count = 0
    if os.path.exists(fb_json):
        print("Incorporating verified real self-learning inpaint patches...")
        with open(fb_json, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item.get("verified") is not True:
                        continue
                    if item.get("ground_truth_source") == "model_generated":
                        continue
                    rp = item.get("raw_image")
                    mp = item.get("mask")
                    cp = item.get("ground_truth")
                    if rp and mp and cp and os.path.exists(rp) and os.path.exists(mp) and os.path.exists(cp):
                        r_im = Image.open(rp).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
                        m_im = Image.open(mp).convert("L").resize((256, 256), Image.Resampling.NEAREST)
                        c_im = Image.open(cp).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
                        
                        m_arr = np.array(m_im)
                        cov = float(np.count_nonzero(m_arr > 30)) / (256 * 256) * 100.0
                        if 0.5 <= cov <= 75.0:
                            tid = f"inpaint_{idx:05d}"
                            raw_p = os.path.join(RAW_DIR, f"{tid}.png")
                            mask_p = os.path.join(MASK_DIR, f"{tid}.png")
                            clean_p = os.path.join(CLEAN_DIR, f"{tid}.png")
                            
                            r_im.save(raw_p, "PNG")
                            m_im.save(mask_p, "PNG")
                            c_im.save(clean_p, "PNG")
                            
                            triplet_records.append({
                                "triplet_id": tid,
                                "raw_image": f"manga_ml_datasets/inpainting/raw/{tid}.png",
                                "mask": f"manga_ml_datasets/inpainting/mask/{tid}.png",
                                "ground_truth": f"manga_ml_datasets/inpainting/clean/{tid}.png",
                                "category": "real_stream_patch",
                                "resolution": [256, 256],
                                "mask_coverage_pct": round(cov, 2)
                            })
                            idx += 1
                            fb_count += 1
                            if fb_count >= 200:
                                break
                except Exception:
                    continue
                    
    print(f"Incorporated {fb_count} real stream feedback patches.")
    print(f"Total triplets generated: {len(triplet_records)}")
    
    # Verify 100% of generated files on disk
    print("Verifying all triplets on disk...")
    for rec in triplet_records:
        rp = os.path.join(BASE_DIR, rec["raw_image"])
        mp = os.path.join(BASE_DIR, rec["mask"])
        cp = os.path.join(BASE_DIR, rec["ground_truth"])
        assert os.path.exists(rp), f"Missing raw file: {rp}"
        assert os.path.exists(mp), f"Missing mask file: {mp}"
        assert os.path.exists(cp), f"Missing clean file: {cp}"
    print("Verification PASSED: 100% of triplets exist as real PNG files on disk!")
    
    # Shuffle and split 80% train, 10% val, 10% test
    random.shuffle(triplet_records)
    n_total = len(triplet_records)
    n_train = int(n_total * 0.80)
    n_val = int(n_total * 0.10)
    
    train_recs = triplet_records[:n_train]
    val_recs = triplet_records[n_train:n_train + n_val]
    test_recs = triplet_records[n_train + n_val:]
    
    full_path = os.path.join(DATA_DIR, "full.jsonl")
    train_path = os.path.join(DATA_DIR, "train.jsonl")
    val_path = os.path.join(DATA_DIR, "val.jsonl")
    test_path = os.path.join(DATA_DIR, "test.jsonl")
    
    with open(full_path, "w", encoding="utf-8") as f:
        for r in triplet_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(test_path, "w", encoding="utf-8") as f:
        for r in test_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print("Dataset successfully saved:")
    print(f"  Full:  {len(triplet_records)} records -> {full_path}")
    print(f"  Train: {len(train_recs)} records -> {train_path}")
    print(f"  Val:   {len(val_recs)} records -> {val_path}")
    print(f"  Test:  {len(test_recs)} records -> {test_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
