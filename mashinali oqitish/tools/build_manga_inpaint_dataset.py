"""
High-Precision Authentic Manga Inpainting Dataset Generator.
Extracts real manga page textures and creates diverse white, cream, dark,
colored, gradient, screentone bubble, and SFX inpainting triplets (raw, mask, clean)
with source-page-grouped train/val/test splits to eliminate leakage.
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import ml_paths
ml_paths.ensure_ml_paths()
from utils import _has_glyph_coverage

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.join(BASE_DIR, "manga_ml_datasets", "inpainting")
RAW_DIR = os.path.join(DATA_DIR, "raw")
MASK_DIR = os.path.join(DATA_DIR, "mask")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")

for d in [RAW_DIR, MASK_DIR, CLEAN_DIR]:
    os.makedirs(d, exist_ok=True)

WIN_DIR = os.environ.get("WINDIR", r"C:\Windows")
FONTS_DIR = os.path.join(WIN_DIR, "Fonts")
SERVER_FONTS_DIR = os.path.join(ml_paths.SERVER_DIR, "fonts")

FONT_CANDIDATES = [
    os.path.join(FONTS_DIR, "malgunbd.ttf"),
    os.path.join(FONTS_DIR, "malgun.ttf"),
    os.path.join(FONTS_DIR, "msyhbd.ttc"),
    os.path.join(FONTS_DIR, "msyh.ttc"),
    os.path.join(FONTS_DIR, "YuGothB.ttc"),
    os.path.join(FONTS_DIR, "msgothic.ttc"),
    os.path.join(FONTS_DIR, "arialbd.ttf"),
    os.path.join(FONTS_DIR, "segoeuib.ttf"),
    os.path.join(SERVER_FONTS_DIR, "arialbd.ttf"),
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

AVAILABLE_FONTS = [f for f in FONT_CANDIDATES if os.path.exists(f)]


def get_font_for_text(text, size=20):
    candidates = list(AVAILABLE_FONTS)
    random.shuffle(candidates)
    for f_path in candidates:
        try:
            font = ImageFont.truetype(f_path, size)
            if _has_glyph_coverage(font, text):
                return font
        except Exception:
            continue
    return ImageFont.load_default()


JAPANESE_DIALOGUES = [
    "何だと！？", "そんなバカな…", "逃げろ！早く！", "お前は…誰だ？",
    "信じられない！", "待ってくれ！", "行くぞ、全力で！", "これで終わりだ！",
    "どうして…？", "絶対に許さない！", "大丈夫か！？", "よし、やるぞ！",
    "嘘でしょ…", "冗談じゃない！", "勝負だ！", "見つけたぞ！",
]

KOREAN_DIALOGUES = [
    "진짜 뭐야?!", "말도 안 돼...", "빨리 도망쳐!", "넌 누구야?",
    "믿을 수 없어!", "잠깐만!", "여기서 끝이다!", "괜찮아?!",
]

CHINESE_DIALOGUES = [
    "怎么可能？！", "不可能…", "快跑！", "你是谁？",
    "太强了！", "等一下！", "到此为止！", "你没事吧？",
]

ENGLISH_DIALOGUES = [
    "WHAT?!", "NO WAY...", "GET OUT OF HERE!", "WHO ARE YOU?!",
    "I CAN'T BELIEVE IT!", "WAIT FOR ME!", "LET'S GO ALL OUT!",
    "THIS IS THE END!", "WHY...?", "I WON'T FORGIVE YOU!",
    "ARE YOU OKAY?!", "ALRIGHT, LET'S DO THIS!", "FOUND YOU!",
    "SHUT UP!", "WHAT HAPPENED HERE?", "HURRY, NO TIME!",
]

RUSSIAN_DIALOGUES = [
    "ЧТО?!", "НЕ МОЖЕТ БЫТЬ...", "БЕГИ БЫСТРЕЕ!", "КТО ТЫ ТАКОЙ?!",
    "ПОДОЖДИ МЕНЯ!", "ЭТО КОНЕЦ!", "ТЫ В ПОРЯДКЕ?!", "НЕ ДВИГАЙСЯ!",
]

SFX_TEXTS = [
    "ドドドド", "ゴゴゴゴ", "バーン！", "ドカン！", "バキッ", "ザッ…",
    "BOOM!", "CRASH!", "BAM!", "WHOOSH!", "THUD!", "SLAM!",
    "SNAP!", "CLANG!", "SLASH!", "RUMBLE...", "SHHH...", "PUNCH!"
]

BUBBLE_STYLES = [
    "white", "cream", "pastel", "dark", "midtone_colored", "gradient", "screentone"
]
BUBBLE_STYLE_WEIGHTS = [0.20, 0.15, 0.15, 0.22, 0.12, 0.10, 0.06]


def create_screentone_patch(size=256, tone_type="dots"):
    """Generates authentic manga screentone textures."""
    arr = np.ones((size, size, 3), dtype=np.uint8) * 255
    if tone_type == "dots":
        freq = random.choice([4, 6, 8, 10])
        gray_val = random.randint(110, 200)
        arr[::freq, ::freq] = gray_val
        if freq >= 6:
            arr[1::freq, 1::freq] = min(255, gray_val + 20)
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


def _sample_bubble_palette(style):
    """Return (fill_col, border_col, text_col, outline_col) for a given bubble style."""
    if style == "white":
        fill = (random.randint(248, 255), random.randint(248, 255), random.randint(248, 255))
        return fill, (random.randint(10, 30),) * 3, (random.randint(5, 28),) * 3, None
    if style == "cream":
        fill = (random.randint(242, 253), random.randint(230, 244), random.randint(195, 224))
        return fill, (random.randint(20, 45),) * 3, (random.randint(10, 35),) * 3, None
    if style == "pastel":
        palettes = [
            (252, 222, 228), (218, 236, 252), (222, 248, 228),
            (248, 242, 195), (236, 224, 250),
        ]
        fill = random.choice(palettes)
        return fill, (25, 25, 35), (15, 15, 25), None
    if style == "dark":
        palettes = [
            (18, 18, 22), (28, 30, 38), (18, 28, 56),
            (62, 16, 24), (20, 46, 36), (48, 22, 54),
        ]
        fill = random.choice(palettes)
        border = (235, 235, 240) if random.random() < 0.35 else (10, 10, 12)
        text = (random.randint(235, 255), random.randint(235, 255), random.randint(235, 255))
        return fill, border, text, (15, 15, 15) if random.random() < 0.3 else None
    if style == "midtone_colored":
        palettes = [
            (135, 140, 150), (175, 125, 115), (115, 148, 175),
            (130, 165, 130), (165, 145, 110), (145, 125, 168),
        ]
        fill = random.choice(palettes)
        use_white_text = sum(fill) / 3.0 < 150 or random.random() < 0.5
        text = (248, 248, 250) if use_white_text else (15, 15, 18)
        outline = (20, 20, 20) if use_white_text else (245, 245, 245)
        return fill, (20, 20, 24), text, outline
    if style == "gradient":
        dark_grad = random.random() < 0.5
        if dark_grad:
            c1 = np.array([random.randint(18, 45), random.randint(20, 45), random.randint(28, 60)], dtype=np.float32)
            c2 = np.array([random.randint(75, 125), random.randint(80, 125), random.randint(90, 140)], dtype=np.float32)
            return (c1, c2), (15, 15, 20), (248, 248, 252), (15, 15, 15)
        c1 = np.array([random.randint(220, 250), random.randint(210, 245), random.randint(190, 235)], dtype=np.float32)
        c2 = np.array([random.randint(155, 195), random.randint(150, 190), random.randint(145, 190)], dtype=np.float32)
        return (c1, c2), (20, 20, 25), (15, 15, 20), None

    # screentone bubble
    return (230, 230, 232), (20, 20, 20), (10, 10, 10), (245, 245, 245)


def _draw_bubble_shape_mask(size, b_box, bubble_type):
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    x1, y1, x2, y2 = b_box
    if bubble_type == "oval":
        draw.ellipse([x1, y1, x2, y2], fill=255)
    elif bubble_type == "rect":
        draw.rounded_rectangle([x1, y1, x2, y2], radius=8, fill=255)
    elif bubble_type == "cloud":
        draw.ellipse([x1, y1, x2, y2], fill=255)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rw, rh = (x2 - x1) // 2, (y2 - y1) // 2
        for angle in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            px = int(cx + rw * 0.9 * math.cos(angle))
            py = int(cy + rh * 0.9 * math.sin(angle))
            draw.ellipse([px - 14, py - 14, px + 14, py + 14], fill=255)
    elif bubble_type == "shout":
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rw, rh = (x2 - x1) // 2, (y2 - y1) // 2
        pts = []
        n_spikes = random.randint(12, 18)
        for i, angle in enumerate(np.linspace(0, 2 * math.pi, n_spikes, endpoint=False)):
            r_mult = 1.15 if i % 2 == 0 else 0.85
            pts.append((int(cx + rw * r_mult * math.cos(angle)), int(cy + rh * r_mult * math.sin(angle))))
        draw.polygon(pts, fill=255)
    return mask


def render_speech_bubble(clean_img, b_box, bubble_type="oval", style=None):
    """Render a diverse speech/narration bubble onto `clean_img` and return (text_col, outline_col, style)."""
    if style is None:
        style = random.choices(BUBBLE_STYLES, weights=BUBBLE_STYLE_WEIGHTS, k=1)[0]

    fill_info, border_col, text_col, outline_col = _sample_bubble_palette(style)
    x1, y1, x2, y2 = b_box
    border_w = random.choice([2, 3, 4])
    size = clean_img.size[0]

    if style in {"gradient", "screentone"}:
        shape_mask = _draw_bubble_shape_mask(size, b_box, bubble_type)
        m_arr = np.array(shape_mask) > 127
        clean_arr = np.array(clean_img)

        if style == "gradient":
            c1, c2 = fill_info
            t = np.linspace(0.0, 1.0, size, dtype=np.float32)
            if random.random() < 0.5:
                t = t[:, None]
                grad = c1[None, None, :] * (1.0 - t[:, :, None]) + c2[None, None, :] * t[:, :, None]
                grad = np.broadcast_to(grad, (size, size, 3))
            else:
                t = t[None, :]
                grad = c1[None, None, :] * (1.0 - t[:, :, None]) + c2[None, None, :] * t[:, :, None]
                grad = np.broadcast_to(grad, (size, size, 3))
            fill_arr = np.clip(grad, 0, 255).astype(np.uint8)
        else:
            tone = random.choice(["dots", "hatch"])
            fill_arr = np.array(create_screentone_patch(size=size, tone_type=tone))

        clean_arr[m_arr] = fill_arr[m_arr]
        # Draw border from mask contour
        edges = cv2.Canny(np.array(shape_mask), 50, 150)
        if border_w > 1:
            edges = cv2.dilate(edges, np.ones((border_w, border_w), np.uint8), iterations=1)
        clean_arr[edges > 0] = border_col
        updated = Image.fromarray(clean_arr)
        clean_img.paste(updated)
        return text_col, outline_col, style

    draw = ImageDraw.Draw(clean_img)
    fill_col = fill_info
    if bubble_type == "oval":
        draw.ellipse([x1, y1, x2, y2], fill=fill_col, outline=border_col, width=border_w)
    elif bubble_type == "rect":
        draw.rounded_rectangle([x1, y1, x2, y2], radius=8, fill=fill_col, outline=border_col, width=border_w)
    elif bubble_type == "cloud":
        draw.ellipse([x1, y1, x2, y2], fill=fill_col, outline=border_col, width=border_w)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rw, rh = (x2 - x1) // 2, (y2 - y1) // 2
        for angle in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            px = int(cx + rw * 0.9 * math.cos(angle))
            py = int(cy + rh * 0.9 * math.sin(angle))
            draw.ellipse([px - 14, py - 14, px + 14, py + 14], fill=fill_col, outline=border_col, width=2)
    elif bubble_type == "shout":
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rw, rh = (x2 - x1) // 2, (y2 - y1) // 2
        pts = []
        n_spikes = random.randint(12, 18)
        for i, angle in enumerate(np.linspace(0, 2 * math.pi, n_spikes, endpoint=False)):
            r_mult = 1.15 if i % 2 == 0 else 0.85
            pts.append((int(cx + rw * r_mult * math.cos(angle)), int(cy + rh * r_mult * math.sin(angle))))
        draw.polygon(pts, fill=fill_col, outline=border_col)

    return text_col, outline_col, style


def make_bubble_triplet(triplet_id, bg_img=None, style=None):
    """
    Creates a triplet where clean is a manga background + diverse bubble/box,
    raw is the bubble with text, and mask is the text region to inpaint.
    """
    size = 256
    if bg_img is not None:
        clean = bg_img.copy().resize((size, size), Image.Resampling.LANCZOS)
    else:
        tt = random.choice(["dots", "hatch", "speedlines", "gradient"])
        clean = create_screentone_patch(size, tt)

    bw = random.randint(120, 200)
    bh = random.randint(90, 180)
    bx1 = random.randint(15, size - bw - 15)
    by1 = random.randint(15, size - bh - 15)
    bx2 = bx1 + bw
    by2 = by1 + bh

    chosen_style = style or random.choices(BUBBLE_STYLES, weights=BUBBLE_STYLE_WEIGHTS, k=1)[0]
    btype = "rect" if chosen_style in {"dark", "midtone_colored", "gradient"} and random.random() < 0.7 else random.choice(["oval", "oval", "rect", "cloud", "shout"])
    text_color, outline_color, actual_style = render_speech_bubble(clean, [bx1, by1, bx2, by2], btype, style=chosen_style)

    raw = clean.copy()
    draw_raw = ImageDraw.Draw(raw)
    mask = Image.new("L", (size, size), 0)
    draw_mask = ImageDraw.Draw(mask)

    pool = random.choice([JAPANESE_DIALOGUES, JAPANESE_DIALOGUES, ENGLISH_DIALOGUES, KOREAN_DIALOGUES, CHINESE_DIALOGUES, RUSSIAN_DIALOGUES])
    text = random.choice(pool)
    if random.random() < 0.35:
        text += "\n" + random.choice(pool)

    font_size = random.randint(16, 24)
    font = get_font_for_text(text.replace("\n", ""), font_size)
    lines = text.split("\n")
    max_text_w = max(40, bw - 20)
    while font_size > 10:
        max_lw = max((draw_raw.textbbox((0, 0), ln, font=font)[2] - draw_raw.textbbox((0, 0), ln, font=font)[0]) for ln in lines)
        if max_lw <= max_text_w and len(lines) * (font_size + 4) <= bh - 16:
            break
        font_size -= 2
        font = get_font_for_text(text.replace("\n", ""), font_size)

    tot_h = len(lines) * (font_size + 4)
    cur_y = (by1 + by2) // 2 - tot_h // 2

    for line in lines:
        bbox = draw_raw.textbbox((0, 0), line, font=font)
        lw = max(1, bbox[2] - bbox[0])
        cur_x = (bx1 + bx2) // 2 - lw // 2 - bbox[0]
        if outline_color is not None:
            draw_raw.text((cur_x, cur_y), line, fill=text_color, font=font, stroke_width=1, stroke_fill=outline_color)
            draw_mask.text((cur_x, cur_y), line, fill=255, font=font, stroke_width=1, stroke_fill=255)
        else:
            draw_raw.text((cur_x, cur_y), line, fill=text_color, font=font)
            draw_mask.text((cur_x, cur_y), line, fill=255, font=font)
        cur_y += font_size + 4

    mask_arr = np.array(mask)
    k_size = random.choice([3, 5, 7])
    mask_dilated = cv2.dilate(mask_arr, np.ones((k_size, k_size), np.uint8), iterations=1)
    mask = Image.fromarray(mask_dilated)

    cov = float(np.count_nonzero(mask_dilated)) / float(size * size) * 100.0
    category = "narration_box" if actual_style in {"dark", "midtone_colored", "gradient"} else "speech_bubble"
    return raw, mask, clean, cov, category, actual_style


def make_sfx_triplet(triplet_id, bg_img=None):
    """Creates a triplet where clean is an action/screentone panel and raw has stylized SFX."""
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
    font_size = random.randint(28, 44)
    font = get_font_for_text(sfx, font_size)

    bbox = draw_raw.textbbox((0, 0), sfx, font=font)
    sw, sh = max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])
    while font_size > 16 and (sw > size - 20 or sh > size - 20):
        font_size -= 4
        font = get_font_for_text(sfx, font_size)
        bbox = draw_raw.textbbox((0, 0), sfx, font=font)
        sw, sh = max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])

    sx = random.randint(10, max(12, size - sw - 10)) - bbox[0]
    sy = random.randint(10, max(12, size - sh - 10)) - bbox[1]

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


def partition_triplets_by_source_page(triplet_records, train_ratio=0.80, val_ratio=0.10, seed=42):
    """Group triplets by `source_group` (e.g. source manga page) so no source page crosses splits."""
    groups = {}
    for rec in triplet_records:
        grp = rec.get("source_group") or rec.get("triplet_id")
        groups.setdefault(grp, []).append(rec)

    group_list = list(groups.values())
    rng = random.Random(seed)
    rng.shuffle(group_list)

    n_total = len(triplet_records)
    target_train = int(round(n_total * train_ratio))
    target_val = int(round(n_total * val_ratio))

    train_recs, val_recs, test_recs = [], [], []
    for grp in group_list:
        if len(train_recs) < target_train or not train_recs:
            train_recs.extend(grp)
        elif len(val_recs) < target_val or (len(group_list) >= 3 and not val_recs):
            val_recs.extend(grp)
        else:
            test_recs.extend(grp)
    return train_recs, val_recs, test_recs


def main():
    random.seed(42)
    np.random.seed(42)
    print("=" * 60)
    print("STARTING DIVERSE MANGA INPAINTING DATASET BUILDER")
    print("=" * 60)

    real_page_paths = glob.glob(os.path.join(BASE_DIR, "test_chapter_57", "*.png"))
    real_page_paths += glob.glob(os.path.join(BASE_DIR, "test_chapter_57", "*.jpg"))
    sample_manga = os.path.join(BASE_DIR, "sample_manga.jpg")
    if os.path.exists(sample_manga):
        real_page_paths.append(sample_manga)
    det_pages = glob.glob(os.path.join(BASE_DIR, "self_learning_data", "detection", "pages", "*.png"))
    real_page_paths += det_pages

    print(f"Discovered {len(real_page_paths)} authentic manga pages.")

    real_patches = []
    for p_path in sorted(real_page_paths)[:25]:
        page_name = os.path.basename(p_path)
        try:
            with Image.open(p_path) as im:
                p_img = im.convert("RGB")
                pw, ph = p_img.size
                if pw >= 256 and ph >= 256:
                    for _ in range(8):
                        rx = random.randint(0, pw - 256)
                        ry = random.randint(0, ph - 256)
                        patch = p_img.crop((rx, ry, rx + 256, ry + 256))
                        arr = np.array(patch.convert("L"))
                        if np.std(arr) > 10:
                            real_patches.append((patch, f"page:{page_name}"))
        except Exception:
            continue

    print(f"Extracted {len(real_patches)} diverse background patches from real manga pages.")

    triplet_records = []
    idx = 1

    # Category 1: Diverse Bubbles on real manga patches (650 samples)
    for _ in range(650):
        tid = f"inpaint_{idx:05d}"
        if real_patches:
            bg, src_group = random.choice(real_patches)
        else:
            bg, src_group = None, f"synth:{tid}"
        raw, mask, clean, cov, cat, style = make_bubble_triplet(tid, bg)

        raw.save(os.path.join(RAW_DIR, f"{tid}.png"), "PNG")
        mask.save(os.path.join(MASK_DIR, f"{tid}.png"), "PNG")
        clean.save(os.path.join(CLEAN_DIR, f"{tid}.png"), "PNG")

        triplet_records.append({
            "triplet_id": tid,
            "raw_image": f"manga_ml_datasets/inpainting/raw/{tid}.png",
            "mask": f"manga_ml_datasets/inpainting/mask/{tid}.png",
            "ground_truth": f"manga_ml_datasets/inpainting/clean/{tid}.png",
            "category": cat,
            "bubble_style": style,
            "source_group": src_group,
            "resolution": [256, 256],
            "mask_coverage_pct": round(cov, 2)
        })
        idx += 1

    # Category 2: Diverse Bubbles on screentones (400 samples)
    for _ in range(400):
        tid = f"inpaint_{idx:05d}"
        raw, mask, clean, cov, cat, style = make_bubble_triplet(tid, bg_img=None)

        raw.save(os.path.join(RAW_DIR, f"{tid}.png"), "PNG")
        mask.save(os.path.join(MASK_DIR, f"{tid}.png"), "PNG")
        clean.save(os.path.join(CLEAN_DIR, f"{tid}.png"), "PNG")

        triplet_records.append({
            "triplet_id": tid,
            "raw_image": f"manga_ml_datasets/inpainting/raw/{tid}.png",
            "mask": f"manga_ml_datasets/inpainting/mask/{tid}.png",
            "ground_truth": f"manga_ml_datasets/inpainting/clean/{tid}.png",
            "category": cat,
            "bubble_style": style,
            "source_group": f"synth:{tid}",
            "resolution": [256, 256],
            "mask_coverage_pct": round(cov, 2)
        })
        idx += 1

    # Category 3: Manga SFX inpainting (450 samples)
    for _ in range(450):
        tid = f"inpaint_{idx:05d}"
        if real_patches and random.random() < 0.5:
            bg, src_group = random.choice(real_patches)
        else:
            bg, src_group = None, f"synth:{tid}"
        raw, mask, clean, cov, cat = make_sfx_triplet(tid, bg)

        raw.save(os.path.join(RAW_DIR, f"{tid}.png"), "PNG")
        mask.save(os.path.join(MASK_DIR, f"{tid}.png"), "PNG")
        clean.save(os.path.join(CLEAN_DIR, f"{tid}.png"), "PNG")

        triplet_records.append({
            "triplet_id": tid,
            "raw_image": f"manga_ml_datasets/inpainting/raw/{tid}.png",
            "mask": f"manga_ml_datasets/inpainting/mask/{tid}.png",
            "ground_truth": f"manga_ml_datasets/inpainting/clean/{tid}.png",
            "category": cat,
            "bubble_style": "sfx",
            "source_group": src_group,
            "resolution": [256, 256],
            "mask_coverage_pct": round(cov, 2)
        })
        idx += 1

    # Category 4: Verified real feedback triplets
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
                    rp = ml_paths.resolve_data_path(item.get("raw_image"), BASE_DIR)
                    mp = ml_paths.resolve_data_path(item.get("mask"), BASE_DIR)
                    cp = ml_paths.resolve_data_path(item.get("ground_truth"), BASE_DIR)
                    if rp and mp and cp and os.path.exists(rp) and os.path.exists(mp) and os.path.exists(cp):
                        with Image.open(rp) as rim:
                            r_im = rim.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
                        with Image.open(mp) as mim:
                            m_im = mim.convert("L").resize((256, 256), Image.Resampling.NEAREST)
                        with Image.open(cp) as cim:
                            c_im = cim.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)

                        m_arr = np.array(m_im)
                        cov = float(np.count_nonzero(m_arr > 30)) / (256 * 256) * 100.0
                        if 0.5 <= cov <= 75.0:
                            tid = f"inpaint_{idx:05d}"
                            r_im.save(os.path.join(RAW_DIR, f"{tid}.png"), "PNG")
                            m_im.save(os.path.join(MASK_DIR, f"{tid}.png"), "PNG")
                            c_im.save(os.path.join(CLEAN_DIR, f"{tid}.png"), "PNG")

                            raw_tid = item.get("triplet_id", tid)
                            src_page = raw_tid.split("_patch_")[0] if "_patch_" in raw_tid else raw_tid
                            triplet_records.append({
                                "triplet_id": tid,
                                "raw_image": f"manga_ml_datasets/inpainting/raw/{tid}.png",
                                "mask": f"manga_ml_datasets/inpainting/mask/{tid}.png",
                                "ground_truth": f"manga_ml_datasets/inpainting/clean/{tid}.png",
                                "category": "real_stream_patch",
                                "bubble_style": "real",
                                "source_group": f"fb:{src_page}",
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

    for rec in triplet_records:
        assert os.path.exists(os.path.join(BASE_DIR, rec["raw_image"]))
        assert os.path.exists(os.path.join(BASE_DIR, rec["mask"]))
        assert os.path.exists(os.path.join(BASE_DIR, rec["ground_truth"]))

    train_recs, val_recs, test_recs = partition_triplets_by_source_page(triplet_records, 0.80, 0.10, seed=42)

    for fname, recs in [("full.jsonl", triplet_records), ("train.jsonl", train_recs), ("val.jsonl", val_recs), ("test.jsonl", test_recs)]:
        with open(os.path.join(DATA_DIR, fname), "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved full={len(triplet_records)}, train={len(train_recs)}, val={len(val_recs)}, test={len(test_recs)}")


if __name__ == "__main__":
    main()
