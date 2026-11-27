"""
Stage-isolated cache runner for Manga Translation Pipeline.
Saves and loads independent snapshots per stage so detection, inpainting,
translation, and typesetting can be iterated on individually without re-running heavy models.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

SERVER_DIR = Path(__file__).resolve().parent.parent
ML_DIR = SERVER_DIR / "ml"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))


def run_stage_detect(image_path: Path, cache_dir: Path) -> List[Dict]:
    """Stage 1: Detect speech bubbles, narration boxes, and text instances."""
    from comic_detector import get_shared_comic_detector

    print(f"[Stage 1 - Detect] Running ComicTextDetector on {image_path.name}...")
    t0 = time.perf_counter()
    pil_img = Image.open(image_path).convert("RGB")
    det = get_shared_comic_detector()
    blocks, mask_bin = det.detect(pil_img)
    dt = time.perf_counter() - t0

    detect_file = cache_dir / "stage1_detect.json"
    mask_file = cache_dir / "stage1_mask.png"
    with open(detect_file, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2, ensure_ascii=False)
    if mask_bin is not None and mask_bin.size > 0:
        Image.fromarray(mask_bin).save(mask_file)

    print(f"[Stage 1 - Detect] Found {len(blocks)} blocks in {dt:.2f}s -> {detect_file}")
    return blocks


def run_stage_clean(image_path: Path, cache_dir: Path) -> Path:
    """Stage 2: Neural MangaTextCleaner + anime-manga-big-lama inpainting."""
    from manga_text_cleaner import MangaTextCleaner

    detect_file = cache_dir / "stage1_detect.json"
    if not detect_file.is_file():
        raise FileNotFoundError(f"Missing {detect_file}. Run --stages detect first.")

    with open(detect_file, "r", encoding="utf-8") as f:
        blocks = json.load(f)

    mask_file = cache_dir / "stage1_mask.png"
    extra_mask = None
    if mask_file.is_file():
        extra_mask = np.array(Image.open(mask_file).convert("L"))

    print(f"[Stage 2 - Clean] Running MangaTextCleaner + LaMa on {len(blocks)} allowed boxes...")
    t0 = time.perf_counter()
    pil_img = Image.open(image_path).convert("RGB")
    allowed_boxes = [b["box"] for b in blocks if b.get("category") != "sfx"]

    cleaner = MangaTextCleaner()
    cleaned_pil, _, _ = cleaner.clean_page(
        pil_img,
        extra_mask=extra_mask,
        allowed_boxes=allowed_boxes if allowed_boxes else None,
    )
    dt = time.perf_counter() - t0

    clean_out = cache_dir / "stage2_clean.png"
    cleaned_pil.save(clean_out)
    print(f"[Stage 2 - Clean] Cleaned page saved in {dt:.2f}s -> {clean_out}")
    return clean_out


def run_stage_translate(image_path: Path, cache_dir: Path, target_lang: str = "uz") -> List[Dict]:
    """Stage 3: OCR + translation using Qwen3.5-9B (or mock fallback if unavailable)."""
    detect_file = cache_dir / "stage1_detect.json"
    if not detect_file.is_file():
        raise FileNotFoundError(f"Missing {detect_file}. Run --stages detect first.")

    with open(detect_file, "r", encoding="utf-8") as f:
        blocks = json.load(f)

    pil_img = Image.open(image_path).convert("RGB")
    w, h = pil_img.size

    crops = []
    crop_indices = []
    for idx, b in enumerate(blocks):
        tb = b.get("text_box", b.get("box", [0, 0, 0, 0]))
        tx1, ty1, tx2, ty2 = [max(0, int(tb[i])) for i in range(4)]
        if tx2 > tx1 + 4 and ty2 > ty1 + 4:
            crops.append(pil_img.crop((max(0, tx1 - 4), max(0, ty1 - 4), min(w, tx2 + 4), min(h, ty2 + 4))))
            crop_indices.append(idx)

    translations: List[Dict] = []
    t0 = time.perf_counter()

    qwen = None
    try:
        from qwen_vl_engine import get_shared_qwen_engine
        eng = get_shared_qwen_engine()
        if eng.is_available:
            qwen = eng
    except Exception:
        qwen = None

    if qwen is not None and crops:
        print(f"[Stage 3 - Translate] Translating {len(crops)} text crops via Qwen3.5-9B...")
        pairs = qwen.ocr_and_translate_crops(crops, target_lang=target_lang, source_lang="auto", batch_size=4)
        for c_idx, (src, tgt) in zip(crop_indices, pairs):
            blocks[c_idx]["text"] = src
            blocks[c_idx]["transcription"] = src
            blocks[c_idx]["translation"] = tgt
            blocks[c_idx]["translated_text"] = tgt
    else:
        print("[Stage 3 - Translate] Qwen unavailable; preserving existing or mock texts...")
        for b in blocks:
            if not b.get("translation"):
                b["translation"] = b.get("text") or "..."

    dt = time.perf_counter() - t0
    trans_file = cache_dir / "stage3_translate.json"
    with open(trans_file, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2, ensure_ascii=False)

    print(f"[Stage 3 - Translate] Translations saved in {dt:.2f}s -> {trans_file}")
    return blocks


def run_stage_render(cache_dir: Path) -> Path:
    """Stage 4: Typeset translated dialogues into the cleaned image in <0.5s."""
    from typesetting import render_translated_dialogues

    clean_file = cache_dir / "stage2_clean.png"
    if not clean_file.is_file():
        raise FileNotFoundError(f"Missing {clean_file}. Run --stages clean first.")

    trans_file = cache_dir / "stage3_translate.json"
    if not trans_file.is_file():
        raise FileNotFoundError(f"Missing {trans_file}. Run --stages translate first.")

    with open(trans_file, "r", encoding="utf-8") as f:
        dialogues = json.load(f)

    print(f"[Stage 4 - Render] Typesetting {len(dialogues)} dialogues onto {clean_file.name}...")
    t0 = time.perf_counter()
    clean_img = Image.open(clean_file).convert("RGB")

    formatted_dialogues = []
    for d in dialogues:
        txt = d.get("translation") or d.get("translated_text") or d.get("text") or ""
        box = d.get("text_box") or d.get("box") or [0, 0, 0, 0]
        formatted_dialogues.append({
            "text": txt,
            "translation": txt,
            "translated_text": txt,
            "box": box,
            "bbox": box,
            "text_box": box,
            "category": d.get("category", "speech_bubble"),
            "is_dark": bool(d.get("is_dark", False)),
        })

    rendered = render_translated_dialogues(clean_img, formatted_dialogues)
    dt = time.perf_counter() - t0

    render_out = cache_dir / "stage4_render.png"
    rendered.save(render_out)
    print(f"[Stage 4 - Render] Rendered typeset page in {dt:.3f}s -> {render_out}")
    return render_out


def main():
    parser = argparse.ArgumentParser(description="Stage-isolated cache runner for manga translation pipeline")
    parser.add_argument("--image", required=True, help="Path to input manga page image")
    parser.add_argument("--cache-dir", default=".stage_cache", help="Directory to store stage artifacts")
    parser.add_argument(
        "--stages",
        default="all",
        help="Stages to run: detect, clean, translate, render, or all (comma-separated)",
    )
    parser.add_argument("--target-lang", default="uz", help="Target translation language")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        print(f"Error: image not found: {img_path}")
        sys.exit(1)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    stages = [s.strip().lower() for s in args.stages.split(",")]
    run_all = ("all" in stages)

    print(f"\nStage Cache Runner: image={img_path.name}, cache_dir={cache_dir}, stages={stages}")

    if run_all or "detect" in stages:
        run_stage_detect(img_path, cache_dir)

    if run_all or "clean" in stages:
        run_stage_clean(img_path, cache_dir)

    if run_all or "translate" in stages:
        run_stage_translate(img_path, cache_dir, target_lang=args.target_lang)

    if run_all or "render" in stages:
        run_stage_render(cache_dir)

    print("\nStage cache runner finished successfully.\n")


if __name__ == "__main__":
    main()
