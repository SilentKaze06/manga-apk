"""
Fast, stage-isolated detection debugger for ComicTextDetector + RT-DETR-v2.
Runs in 2-5 seconds without loading Qwen3.5-9B or SimpleLama.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add server and ml paths to sys.path
SERVER_DIR = Path(__file__).resolve().parent.parent
ML_DIR = SERVER_DIR / "ml"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

# Vibrant color palette for differentiating separate block_ids
from bubble_rtdetr import annotate_detection_boxes, _DISTINCT_COLORS as PALETTE


def run_debug_detect(
    images: List[Path],
    out_dir: Path,
    expected_map: Optional[Dict] = None,
    device: Optional[str] = None,
):
    from comic_detector import ComicTextDetector

    out_dir.mkdir(parents=True, exist_ok=True)
    det = ComicTextDetector(device=device)

    print(f"\n{'='*75}")
    print(f" ComicTextDetector + RT-DETR-v2 Debug Runner (device={det.device})")
    print(f"{'='*75}")
    header = f"{'Page':<25} | {'Total':<6} | {'Speech':<7} | {'Narr':<6} | {'Free':<6} | {'Status':<8} | {'Time':<6}"
    print(header)
    print("-" * len(header))

    for img_path in images:
        if not img_path.is_file():
            continue
        t0 = time.perf_counter()
        pil_img = Image.open(img_path).convert("RGB")
        blocks, _ = det.detect(pil_img)
        dt = time.perf_counter() - t0

        cat_counts: Dict[str, int] = {}
        for b in blocks:
            c = b.get("category", "unknown")
            cat_counts[c] = cat_counts.get(c, 0) + 1

        total = len(blocks)
        n_speech = cat_counts.get("speech_bubble", 0)
        n_narr = cat_counts.get("narration", 0)
        n_free = cat_counts.get("free_text", 0)

        status = "OK"
        if expected_map:
            key = img_path.name
            stem_key = img_path.stem
            exp = expected_map.get(key, expected_map.get(stem_key))
            if exp is not None:
                exp_total = exp if isinstance(exp, int) else exp.get("total", exp.get("count"))
                if exp_total is not None:
                    status = "PASS" if total == exp_total else f"WARN({exp_total})"

        print(f"{img_path.name:<25} | {total:<6} | {n_speech:<7} | {n_narr:<6} | {n_free:<6} | {status:<8} | {dt:.2f}s")

        # Save annotated image
        overlay = annotate_detection_boxes(pil_img, blocks)
        overlay_path = out_dir / f"{img_path.stem}_boxes.png"
        overlay.save(overlay_path)

        # Save detection JSON
        json_path = out_dir / f"{img_path.stem}_detect.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(blocks, f, indent=2, ensure_ascii=False)

    print("-" * len(header))
    print(f"Done! Overlays and JSON saved to {out_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Debug manga text detection without Qwen or LaMa")
    parser.add_argument("--images", required=True, help="Path to image file or directory of images")
    parser.add_argument("--out", default="debug_out", help="Output directory for debug overlays and json")
    parser.add_argument("--expected", default=None, help="Optional JSON file mapping image filename to expected block counts")
    parser.add_argument("--device", default=None, help="Device to run on (cpu or cuda)")
    args = parser.parse_args()

    img_input = Path(args.images)
    if img_input.is_file():
        img_list = [img_input]
    elif img_input.is_dir():
        img_list = sorted([p for p in img_input.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
    else:
        print(f"Error: path not found: {args.images}")
        sys.exit(1)

    expected_data = None
    if args.expected and Path(args.expected).is_file():
        with open(args.expected, "r", encoding="utf-8") as f:
            expected_data = json.load(f)

    run_debug_detect(img_list, Path(args.out), expected_map=expected_data, device=args.device)


if __name__ == "__main__":
    main()
