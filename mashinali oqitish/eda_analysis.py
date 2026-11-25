
import os
import sys
import json
import collections
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import ml_paths
ml_paths.ensure_ml_paths()
from dataset_loader import parse_annotation_bbox

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "manga_ml_datasets")
CHARTS_DIR = os.path.join(BASE_DIR, "eda_charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']

print("=" * 60)
print("1. EXPLORATORY DATA ANALYSIS (EDA) - DETECTION DATASET")
print("=" * 60)

det_categories = collections.Counter()
det_aspect_ratios = []
det_bbox_areas = []
det_counts_per_page = []
det_langs = collections.Counter()

det_full_path = os.path.join(DATA_DIR, "detection", "full.jsonl")
if os.path.exists(det_full_path):
    with open(det_full_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            det_langs[item.get("lang", "unknown")] += 1
            anns = item.get("annotations", [])
            det_counts_per_page.append(len(anns))
            p_w = float(item.get("width", 800) or 800)
            p_h = float(item.get("height", 1200) or 1200)
            p_area = max(1.0, p_w * p_h)

            for ann in anns:
                cat = ann.get("category", "unknown")
                det_categories[cat] += 1
                parsed = parse_annotation_bbox(ann, p_w, p_h)
                if parsed is not None:
                    xmin, ymin, xmax, ymax, is_norm = parsed
                    w = (xmax - xmin) * (p_w if is_norm else 1.0)
                    h = (ymax - ymin) * (p_h if is_norm else 1.0)
                    if w > 0 and h > 0:
                        det_aspect_ratios.append(w / h)
                        det_bbox_areas.append((w * h) / p_area)
else:
    print(f"Warning: {det_full_path} not found.")

print(f"Total Detection Pages: {len(det_counts_per_page)}")
print("Categories distribution:", dict(det_categories))
print("Language distribution:", dict(det_langs))
print(f"Avg annotations per page: {sum(det_counts_per_page)/max(1, len(det_counts_per_page)):.2f}")

# Plot Detection EDA
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Category distribution
cat_names = list(det_categories.keys())
cat_counts = [det_categories[k] for k in cat_names]
sns.barplot(x=cat_names, y=cat_counts, ax=axes[0], palette="viridis")
axes[0].set_title("Detection: Object Category Counts")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis='x', rotation=30)

# Aspect ratio histogram
sns.histplot(det_aspect_ratios, bins=30, ax=axes[1], color="teal", kde=True)
axes[1].set_title("Bounding Box Aspect Ratios (W/H)")
axes[1].set_xlabel("Aspect Ratio")

# Annotations per page
sns.histplot(det_counts_per_page, bins=15, ax=axes[2], color="coral", discrete=True)
axes[2].set_title("Distribution of Annotations Per Page")
axes[2].set_xlabel("Count of Bubbles/Text per Page")

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "detection_eda.png"), dpi=150)
plt.close()
print("Saved detection_eda.png")

print("=" * 60)
print("2. EXPLORATORY DATA ANALYSIS (EDA) - OCR DATASET")
print("=" * 60)

ocr_langs = collections.Counter()
ocr_directions = collections.Counter()
ocr_bubble_types = collections.Counter()
ocr_char_lengths = []

ocr_full_path = os.path.join(DATA_DIR, "ocr", "full.jsonl")
if os.path.exists(ocr_full_path):
    with open(ocr_full_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            ocr_langs[item.get("language", "unknown")] += 1
            ocr_directions[item.get("direction", "unknown")] += 1
            ocr_bubble_types[item.get("bubble_type", "unknown")] += 1
            text = item.get("text", "")
            ocr_char_lengths.append(len(text))
else:
    print(f"Warning: {ocr_full_path} not found.")

print(f"Total OCR Samples: {len(ocr_char_lengths)}")
print("OCR Languages:", dict(ocr_langs))
print("OCR Directions:", dict(ocr_directions))
print("OCR Bubble types:", dict(ocr_bubble_types))
print(f"Avg text length: {sum(ocr_char_lengths)/max(1, len(ocr_char_lengths)):.2f} chars")

# Plot OCR EDA
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Languages
l_names = list(ocr_langs.keys())
l_counts = [ocr_langs[k] for k in l_names]
sns.barplot(x=l_names, y=l_counts, ax=axes[0], palette="mako")
axes[0].set_title("OCR: Language Distribution")
axes[0].set_ylabel("Count")

# Text direction
d_names = list(ocr_directions.keys())
d_counts = [ocr_directions[k] for k in d_names]
sns.barplot(x=d_names, y=d_counts, ax=axes[1], palette="flare")
axes[1].set_title("OCR: Text Direction (Vertical vs Horizontal)")
axes[1].set_ylabel("Count")

# Char length
sns.histplot(ocr_char_lengths, bins=25, ax=axes[2], color="purple", kde=True)
axes[2].set_title("Text Character Length Distribution")
axes[2].set_xlabel("Number of Characters")

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "ocr_eda.png"), dpi=150)
plt.close()
print("Saved ocr_eda.png")

print("=" * 60)
print("3. EXPLORATORY DATA ANALYSIS (EDA) - INPAINTING DATASET")
print("=" * 60)

inpaint_types = collections.Counter()
inpaint_full_path = os.path.join(DATA_DIR, "inpainting", "full.jsonl")
if os.path.exists(inpaint_full_path):
    with open(inpaint_full_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            cat = item.get("category") or item.get("type") or item.get("source_db") or "general"
            inpaint_types[cat] += 1
else:
    print(f"Warning: {inpaint_full_path} not found.")

print(f"Total Inpainting Samples: {sum(inpaint_types.values())}")
print("Inpainting Types:", dict(inpaint_types))

# Plot Inpainting EDA
plt.figure(figsize=(8, 5))
inp_names = list(inpaint_types.keys())
inp_counts = [inpaint_types[k] for k in inp_names]
sns.barplot(x=inp_names, y=inp_counts, palette="crest")
plt.title("Inpainting & Text Removal: Category Distribution")
plt.ylabel("Count")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "inpainting_eda.png"), dpi=150)
plt.close()
print("Saved inpainting_eda.png")

print("\nAll EDA charts generated successfully in eda_charts/!")
