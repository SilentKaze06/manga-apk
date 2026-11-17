import os
import sys
import re
import json
import time
import copy
import collections
import hashlib
import html
import inspect
import logging
import io
import base64
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T

from utils import get_best_font, atomik_json_yoz

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("app")

# Hardware optimization for multi-core CPUs
if not torch.cuda.is_available():
    try:
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        cv2.setNumThreads(num_threads)
    except Exception:
        pass

# 1. Python 3.14 + Typer / Click subscriptability compatibility fix
import click
if not hasattr(click.Choice, "__class_getitem__"):
    click.Choice.__class_getitem__ = classmethod(lambda cls, item: cls)

# 2. Gradio Queue robustness fix
try:
    import gradio.queueing as gr_queueing
    if hasattr(gr_queueing, "Queue") and hasattr(gr_queueing.Queue, "push"):
        _orig_queue_push = gr_queueing.Queue.push

        if inspect.iscoroutinefunction(_orig_queue_push):
            async def _safe_queue_push(self, *args, **kwargs):
                try:
                    return await _orig_queue_push(self, *args, **kwargs)
                except KeyError as e:
                    logger.warning("Queue push KeyError intercepted: %s", e)
                    return False, "Session expired, please refresh", "error"
        else:
            def _safe_queue_push(self, *args, **kwargs):
                try:
                    return _orig_queue_push(self, *args, **kwargs)
                except KeyError as e:
                    logger.warning("Queue push KeyError intercepted: %s", e)
                    return False, "Session expired, please refresh", "error"

        gr_queueing.Queue.push = _safe_queue_push
        logger.info("Gradio queue robustness patch applied successfully.")
except Exception as e:
    logger.warning("Gradio queue patch could not be applied: %s", e)

import gradio as gr
from manga_pipeline import MangaTranslationPipeline, generate_ink_stroke_mask, clean_bubble_crop

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = MangaTranslationPipeline(base_dir=BASE_DIR)
collector = pipeline.collector
trainer = pipeline.trainer


# ==============================================================================
# GRADIO VERSION COMPATIBILITY HELPERS
# ==============================================================================
def make_clean_image_component(**kwargs):
    """Creates a gr.Image without toolbar buttons across Gradio 4.x and 5.x+."""
    sig = inspect.signature(gr.Image.__init__).parameters
    if "buttons" in sig:
        kwargs["buttons"] = []
    else:
        for opt in ("show_download_button", "show_share_button", "show_fullscreen_button"):
            if opt in sig:
                kwargs[opt] = False
    return gr.Image(**kwargs)


def format_page_info_html(page_info_str):
    return f"<div class='page-info-pill'>{html.escape(str(page_info_str))}</div>"


def format_red_badge_header(count):
    return (
        f'<div class="panel-subhead">'
        f'<span class="panel-title">1 · Nutq bulutlari</span>'
        f'<span class="badge-tag-red">{int(count)} ta belgilandi</span>'
        f'</div>'
    )


# ==============================================================================
# DYNAMIC RECENT SAMPLES (FIFO RECENT PAGE BUFFER)
# ==============================================================================
RECENT_SAMPLES_DIR = os.path.join(BASE_DIR, "recent_samples")
os.makedirs(RECENT_SAMPLES_DIR, exist_ok=True)
RECENT_SAMPLES_META = os.path.join(RECENT_SAMPLES_DIR, "samples_meta.json")


def img_to_base64_thumb(img, max_dim=140):
    try:
        thumb = img.copy()
        thumb.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=85)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"
    except Exception as e:
        logger.warning("Error generating thumbnail: %s", e)
        return ""


def get_recent_samples():
    """Returns a list of 3 recent sample dicts: [{'path': ..., 'name': ..., 'thumb_b64': ...}, ...]"""
    samples = []
    if os.path.exists(RECENT_SAMPLES_META):
        try:
            with open(RECENT_SAMPLES_META, "r", encoding="utf-8") as f:
                samples = json.load(f)
        except Exception as e:
            logger.warning("Error reading recent_samples_meta: %s", e)

    valid_samples = [s for s in samples if os.path.exists(s.get("path", ""))]

    default_seeds = [
        ("sahifa_04.jpg", os.path.join(BASE_DIR, "test_chapter_57", "ogami_tsumiki_to_kinichijou_57_4.jpg")),
        ("sahifa_05.jpg", os.path.join(BASE_DIR, "test_chapter_57", "ogami_tsumiki_to_kinichijou_57_5.jpg")),
        ("sahifa_01.jpg", os.path.join(BASE_DIR, "sample_manga.jpg"))
    ]
    if len(valid_samples) < 3:
        for name, seed_path in default_seeds:
            if os.path.exists(seed_path) and not any(s.get("path") == seed_path for s in valid_samples):
                try:
                    with Image.open(seed_path) as raw_s:
                        s_img = raw_s.convert("RGB")
                    b64 = img_to_base64_thumb(s_img)
                    valid_samples.append({
                        "name": name,
                        "path": seed_path,
                        "thumb_b64": b64
                    })
                except Exception:
                    pass
            if len(valid_samples) >= 3:
                break

    return valid_samples[:3]


def save_recent_samples(samples):
    try:
        atomik_json_yoz(RECENT_SAMPLES_META, samples[:3])
    except Exception as e:
        logger.warning("Error saving samples_meta: %s", e)


def register_uploaded_sample(img, filename):
    """
    Called when a new manga page is uploaded.
    Saves persistent copy in recent_samples/, generates thumbnail,
    pushes to slot 0, shifting older ones down (strict FIFO 3-slot queue).
    """
    try:
        samples = get_recent_samples()
        clean_name = os.path.basename(filename)
        h = hashlib.sha256(f"{clean_name}_{time.time()}".encode()).hexdigest()[:10]
        save_fname = f"sample_{h}_{clean_name}"
        save_path = os.path.join(RECENT_SAMPLES_DIR, save_fname)
        img.save(save_path, "JPEG", quality=95)

        b64 = img_to_base64_thumb(img)
        new_entry = {
            "name": clean_name,
            "path": save_path,
            "thumb_b64": b64
        }

        updated = [new_entry] + [s for s in samples if s.get("path") != save_path][:2]
        save_recent_samples(updated)
        return updated
    except Exception as e:
        logger.error("Error registering uploaded sample: %s", e)
        return get_recent_samples()


def generate_samples_css(samples=None):
    if samples is None:
        samples = get_recent_samples()

    rules = []
    for i in range(3):
        cls_idx = i + 1
        if i < len(samples) and samples[i].get("thumb_b64"):
            b64_url = samples[i]["thumb_b64"]
            rules.append(
                f".sample-thumb-{cls_idx} {{ background-image: url('{b64_url}') !important; "
                f"background-size: cover !important; background-position: center !important; "
                f"background-color: #1a1f32 !important; }}"
            )
        else:
            rules.append(f".sample-thumb-{cls_idx} {{ background-color: #1a1f32 !important; }}")

    return "<style id='dynamic-sample-thumbs'>" + "\n".join(rules) + "</style>"


# ==============================================================================
# SMART REAL-TIME ESTIMATOR FOR OCR RUNTIME
# ==============================================================================
class OCRRuntimeEstimator:
    def __init__(self):
        self.history = collections.deque(maxlen=25)

    def estimate(self, width, height):
        mp = max(0.4, (width * height) / 1_000_000.0)
        if self.history:
            avg_per_mp = sum(t / m for t, m in list(self.history)[-5:]) / len(list(self.history)[-5:])
            est = 8.0 + (mp * avg_per_mp * 0.7)
        else:
            est = 14.0 + (mp * 5.8)

        est = max(18.0, min(50.0, est))
        est_low = max(15, int(round(est - 3)))
        est_high = int(round(est + 3))
        return round(est, 1), est_low, est_high

    def record(self, width, height, elapsed):
        mp = max(0.4, (width * height) / 1_000_000.0)
        self.history.append((elapsed, mp))


ocr_estimator = OCRRuntimeEstimator()


def get_default_session():
    """Creates a fresh, isolated state dictionary for each user / browser session."""
    return {
        "orig_img": None,
        "cleaned_img": None,
        "mask_img": None,
        "annotated_img": None,
        "dialogues": [],
        "crops": {},
        "click_pt1": None,
        "click_pt2": None,
        "source_lang": "ja",
        "page_info": "sahifa_04.jpg · 800 × 1168"
    }


def redraw_annotated_page(session, active_bid=None):
    """
    Redraws bold coral bounding boxes and #ID badges over a clean copy of the original image.
    """
    if session is None or session.get("orig_img") is None:
        return None

    annotated = session["orig_img"].copy()
    draw = ImageDraw.Draw(annotated)
    font_badge = get_best_font(size=18, bold=True)

    w, h = annotated.size
    for d in session.get("dialogues", []):
        bid = d["bubble_id"]
        box = d["bbox"]
        x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])

        is_active = (active_bid is not None and bid == active_bid)
        box_color = (255, 120, 30) if is_active else (234, 88, 12)
        box_w = 4 if is_active else 3

        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_w)
        badge_w = 34 if bid < 10 else 44
        draw.rectangle([x1, max(0, y1 - 20), min(w, x1 + badge_w), max(0, y1)], fill=box_color)
        draw.text((x1 + 4, max(0, y1 - 19)), f"#{bid}", fill=(255, 255, 255), font=font_badge)

    session["annotated_img"] = annotated
    return annotated


def recompute_inpaint_and_annotated(session, target_bid=None, old_bbox=None, active_bid=None):
    """
    Recomputes annotated page and inpainted cleaned page using the unannotated original image.
    Restores old_bbox pixels before single-bubble updates so moved/shrunk boxes leave no artifacts.
    """
    if session is None or session.get("orig_img") is None:
        return None, None, None

    raw_orig = session["orig_img"]
    w, h = raw_orig.size

    if active_bid is None:
        active_bid = target_bid

    # 1. Build annotated image on a separate copy
    annotated = redraw_annotated_page(session, active_bid=active_bid)

    # 2. Build full mask image with precise ink strokes (protecting faces, hair, artwork)
    raw_gray = cv2.cvtColor(np.array(raw_orig), cv2.COLOR_RGB2GRAY)
    mask_np = np.zeros((h, w), dtype=np.uint8)
    for d in session.get("dialogues", []):
        s_box = d.get("text_box", d["bbox"])
        s_m = generate_ink_stroke_mask(raw_gray, s_box, pad=3)
        mask_np = np.maximum(mask_np, s_m)

    mask_img = Image.fromarray(mask_np)
    session["mask_img"] = mask_img

    # 3. Prepare base cleaned_page and determine which bubbles need inpainting
    if target_bid is not None and session.get("cleaned_img") is not None:
        cleaned_page = session["cleaned_img"].copy()
        if old_bbox is not None:
            ox1, oy1 = max(0, old_bbox[0] - 12), max(0, old_bbox[1] - 12)
            ox2, oy2 = min(w, old_bbox[2] + 12), min(h, old_bbox[3] + 12)
            if ox2 > ox1 and oy2 > oy1:
                cleaned_page.paste(raw_orig.crop((ox1, oy1, ox2, oy2)), (ox1, oy1))

            target_dialogue = next((d for d in session.get("dialogues", []) if d.get("bubble_id") == target_bid), None)
            new_bbox = target_dialogue.get("bbox") if target_dialogue else None
            nx1 = max(0, new_bbox[0] - 12) if new_bbox else ox1
            ny1 = max(0, new_bbox[1] - 12) if new_bbox else oy1
            nx2 = min(w, new_bbox[2] + 12) if new_bbox else ox2
            ny2 = min(h, new_bbox[3] + 12) if new_bbox else oy2

            # Re-inpaint target_bid and any bubble that overlapped with old_bbox or new_bbox
            dialogues_to_inpaint = []
            for d in session.get("dialogues", []):
                bx1, by1, bx2, by2 = d["bbox"]
                overlaps_old = not (bx2 < ox1 or bx1 > ox2 or by2 < oy1 or by1 > oy2)
                overlaps_new = not (bx2 < nx1 or bx1 > nx2 or by2 < ny1 or by1 > ny2)
                if d["bubble_id"] == target_bid or overlaps_old or overlaps_new:
                    dialogues_to_inpaint.append(d)
        else:
            dialogues_to_inpaint = [d for d in session.get("dialogues", []) if d["bubble_id"] == target_bid]
    else:
        cleaned_page = raw_orig.copy()
        dialogues_to_inpaint = session.get("dialogues", [])

    # 4. Master border-protected bubble cleaning strictly from raw_orig (never from annotated)
    raw_np = np.array(raw_orig)
    cleaned_np = np.array(cleaned_page)
    with pipeline.model_lock:
        for d in dialogues_to_inpaint:
            box = d["bbox"]
            bx1, by1 = max(0, int(box[0])), max(0, int(box[1]))
            bx2, by2 = min(w, int(box[2])), min(h, int(box[3]))
            if bx2 <= bx1 + 6 or by2 <= by1 + 6:
                continue
            patch_orig = raw_np[by1:by2, bx1:bx2].copy()
            patch_gray = raw_gray[by1:by2, bx1:bx2].copy()
            patch_clean = clean_bubble_crop(
                patch_orig,
                patch_gray=patch_gray,
                inpainter_model=getattr(pipeline, "inpainter", None),
                inpaint_transform=getattr(pipeline, "inpaint_transform", None),
                device=DEVICE,
            )
            cleaned_np[by1:by2, bx1:bx2] = patch_clean

    cleaned_page = Image.fromarray(cleaned_np)
    session["cleaned_img"] = cleaned_page
    return annotated, cleaned_page, mask_img


def get_stats_text():
    stats = collector.load_stats()
    pages_val = stats.get("total_pages_collected", 0)
    crops_val = stats.get("total_ocr_crops_collected", 0)
    corr_val = stats.get("total_user_corrections", 0)
    cycles_val = stats.get("total_training_cycles", 0)

    return (
        f'<div class="stats-row">'
        f'<div class="stat-pill"><span class="pill-lbl">Sahifalar</span><span class="pill-val">{pages_val}</span></div>'
        f'<div class="stat-pill"><span class="pill-lbl">OCR namunalar</span><span class="pill-val">{crops_val}</span></div>'
        f'<div class="stat-pill"><span class="pill-lbl">To\'g\'rilashlar</span><span class="pill-val">{corr_val}</span></div>'
        f'<div class="stat-pill"><span class="pill-lbl">O\'qitish sikllari</span><span class="pill-val">{cycles_val}</span></div>'
        f'<div class="model-pill"><span class="dot-green"></span> best_manga_ocr.pt <span class="badge-active">Faol</span></div>'
        f'</div>'
    )


def format_all_dialogues_summary(session, active_bid=None):
    if not session or not session.get("dialogues"):
        return "<div class='dialogues-empty-msg'>(Hozircha nutq bulutlari aniqlanmadi)</div>"
    items = []
    if active_bid is None and session.get("dialogues"):
        active_bid = session["dialogues"][0]["bubble_id"]
    for d in session["dialogues"]:
        bid = d["bubble_id"]
        raw_text = d.get("transcription", "") or "(böş)"
        is_active = (bid == active_bid)
        active_cls = " active-dialogue-card" if is_active else ""
        items.append(
            f'<div class="dialogue-card{active_cls}" data-bid="{bid}">'
            f'<span class="dialogue-tag">#{bid}</span>'
            f'<span class="dialogue-text">{html.escape(raw_text)}</span>'
            f'</div>'
        )
    return f'<div class="dialogues-container">{"".join(items)}</div>'


def extract_bid_from_choice(choice_val):
    if choice_val is None:
        return 1
    if isinstance(choice_val, (list, tuple)):
        choice_val = choice_val[0] if choice_val else 1
    if isinstance(choice_val, int):
        return choice_val
    if hasattr(choice_val, "value"):
        choice_val = choice_val.value
    try:
        m = re.search(r"#?(\d+)", str(choice_val))
        return int(m.group(1)) if m else 1
    except Exception:
        return 1


def get_bubble_pill_choices(dialogues):
    if not dialogues:
        return ["#1"]
    return [f"#{d['bubble_id']}" for d in dialogues]


def get_bubble_details(session, target_bid):
    dialogues = session.get("dialogues", [])
    found = None
    for d in dialogues:
        if d["bubble_id"] == target_bid:
            found = d
            break
    if not found and dialogues:
        found = dialogues[0]
        target_bid = found["bubble_id"]

    if not found:
        crop = None
        text = ""
        lbl = "<div class='bubble-title-box'><h3>Bulut #(tanlanmadi)</h3><span class='dim-badge'>—</span></div>"
        x1, y1, x2, y2 = 0, 0, 0, 0
    else:
        crop = session.get("crops", {}).get(target_bid)
        text = found.get("transcription", "")
        box = found.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        lbl = f"<div class='bubble-title-box'><h3>Bulut #{target_bid}</h3><span class='dim-badge'>{x2-x1} × {y2-y1} px</span></div>"

    summary_html = format_all_dialogues_summary(session, active_bid=target_bid if found else None)
    return crop, text, lbl, x1, y1, x2, y2, summary_html


def nav_bubble(choice_val, direction, session_state):
    session = session_state if isinstance(session_state, dict) else get_default_session()
    dialogues = session.get("dialogues", [])
    if not dialogues:
        return (
            session.get("annotated_img"),
            gr.update(choices=["#1"], value="#1"),
            None, "",
            "<div class='bubble-title-box'><h3>Bulut #(tanlanmadi)</h3><span class='dim-badge'>—</span></div>",
            0, 0, 0, 0,
            format_all_dialogues_summary(None)
        )

    cur_id = extract_bid_from_choice(choice_val)
    n = len(dialogues)
    cur_idx = 0
    for i, d in enumerate(dialogues):
        if d["bubble_id"] == cur_id:
            cur_idx = i
            break

    new_idx = (cur_idx + direction) % n
    target_d = dialogues[new_idx]
    target_bid = target_d["bubble_id"]
    target_pill = f"#{target_bid}"

    crop, text, lbl, x1, y1, x2, y2, summary_html = get_bubble_details(session, target_bid)
    choices = get_bubble_pill_choices(dialogues)
    annotated = redraw_annotated_page(session, active_bid=target_bid)
    return annotated, gr.update(choices=choices, value=target_pill), crop, text, lbl, x1, y1, x2, y2, summary_html


def on_select_bubble(choice_val, session_state):
    session = session_state if isinstance(session_state, dict) else get_default_session()
    bid = extract_bid_from_choice(choice_val)
    crop, text, lbl, x1, y1, x2, y2, summary_html = get_bubble_details(session, bid)
    annotated = redraw_annotated_page(session, active_bid=bid if session.get("dialogues") else None)
    return annotated, crop, text, lbl, x1, y1, x2, y2, summary_html


def format_click_status(pt1, pt2=None):
    if pt1 is None:
        p1 = "<span class='coord-pill'>1-burçak: kutilmoqda</span>"
        p2 = "<span class='coord-pill'>2-burçak: kutilmoqda</span>"
    elif pt2 is None:
        p1 = f"<span class='coord-pill active'>1-burçak ({pt1[0]}, {pt1[1]})</span>"
        p2 = "<span class='coord-pill'>2-burçak: kutilmoqda</span>"
    else:
        p1 = f"<span class='coord-pill active'>1-burçak ({pt1[0]}, {pt1[1]})</span>"
        p2 = f"<span class='coord-pill active'>2-burçak ({pt2[0]}, {pt2[1]})</span>"
    return f"<div class='coord-status-group'>{p1}{p2}</div>"


def on_annotated_image_click(evt: gr.SelectData, session_state):
    session = session_state if isinstance(session_state, dict) else get_default_session()
    if session.get("orig_img") is None or evt is None or evt.index is None:
        return format_click_status(None), session

    click_x, click_y = int(evt.index[0]), int(evt.index[1])
    pt1 = session.get("click_pt1")

    if pt1 is None or session.get("click_pt2") is not None:
        session["click_pt1"] = (click_x, click_y)
        session["click_pt2"] = None
        return format_click_status((click_x, click_y), None), session
    else:
        session["click_pt2"] = (click_x, click_y)
        return format_click_status(pt1, (click_x, click_y)), session


def reset_click_points(session_state):
    session = session_state if isinstance(session_state, dict) else get_default_session()
    session["click_pt1"] = None
    session["click_pt2"] = None
    return format_click_status(None), session


def process_and_inspect_manga(image_input, auto_collect, session_state, filename_override=None):
    session = session_state if isinstance(session_state, dict) else get_default_session()
    if image_input is None:
        image_input = session.get("orig_img")

    if image_input is None:
        p_info = session.get("page_info", "sahifa · 0 × 0")
        return (
            None, None,
            gr.update(choices=["#1"], value="#1"),
            None, "", "<div class='bubble-title-box'><h3>Bulut #(tanlanmadi)</h3><span class='dim-badge'>—</span></div>",
            0, 0, 0, 0,
            format_all_dialogues_summary(None),
            "⚠️ Iltimos, avval manga sahifasini yuklang!",
            get_stats_text(),
            format_page_info_html(p_info),
            format_red_badge_header(0),
            session
        )

    if isinstance(image_input, str):
        with Image.open(image_input) as raw_img:
            orig = raw_img.convert("RGB")
        page_name = os.path.basename(image_input)
    else:
        orig = image_input.convert("RGB")
        page_name = "sahifa_04.jpg"

    w, h = orig.size
    page_info = filename_override if filename_override else f"{page_name} · {w} × {h}"
    session["page_info"] = page_info
    session["click_pt1"] = None
    session["click_pt2"] = None

    t0 = time.time()
    est_sec, est_low, est_high = ocr_estimator.estimate(w, h)

    results = pipeline.process_page(orig, auto_collect=auto_collect, source_lang=session.get("source_lang", "ja"))
    cleaned = results["cleaned_image"]
    mask = results["mask_image"]
    dialogues = results["dialogues"]

    elapsed = time.time() - t0
    ocr_estimator.record(w, h, elapsed)

    session["orig_img"] = orig
    session["cleaned_img"] = cleaned
    session["mask_img"] = mask
    session["dialogues"] = dialogues
    session["crops"] = {}

    for d in dialogues:
        bid = d["bubble_id"]
        box = d["bbox"]
        x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
        pad = 6
        c_x1, c_y1 = max(0, x1 - pad), max(0, y1 - pad)
        c_x2, c_y2 = min(w, x2 + pad), min(h, y2 + pad)
        session["crops"][bid] = orig.crop((c_x1, c_y1, c_x2, c_y2))

    choices = get_bubble_pill_choices(dialogues)
    first_bid = dialogues[0]["bubble_id"] if dialogues else 1
    first_choice = f"#{first_bid}" if dialogues else "#1"

    annotated = redraw_annotated_page(session, active_bid=first_bid if dialogues else None)

    crop, text, id_lbl, x1, y1, x2, y2, summary_text = get_bubble_details(session, first_bid)
    status_msg = (
        f"🟢 **{len(dialogues)} ta nutq buluti** aniqlandi | "
        f"⏱️ Sarflangan vaqt: **{elapsed:.1f} s** (Taxmin qilingan: ~{est_low}-{est_high} s). "
        f"Tahrirlaş uçun öngdagi röyxatdan tanlang."
    )
    if results.get("requires_ocr_fallback"):
        status_msg = "⚠️ İşonçli matn topilmadi. Asl sahifa saqlandi. Tanlangan til va mahalliy OCR modeli mavjudligini tekşiring."

    return (
        annotated, cleaned,
        gr.update(choices=choices, value=first_choice),
        crop, text, id_lbl,
        x1, y1, x2, y2,
        summary_text, status_msg, get_stats_text(),
        format_page_info_html(page_info),
        format_red_badge_header(len(dialogues)),
        session
    )


def handle_upload_action(file_obj, auto_collect, session_state):
    session = session_state if isinstance(session_state, dict) else get_default_session()
    if file_obj is None:
        return (
            session.get("annotated_img"), session.get("cleaned_img"),
            gr.update(), None, "", "", 0, 0, 0, 0,
            format_all_dialogues_summary(session),
            "⚠️ Fayl tanlanmadi",
            get_stats_text(),
            format_page_info_html(session.get("page_info", "")),
            format_red_badge_header(len(session.get("dialogues", []))),
            generate_samples_css(),
            session
        )
    fpath = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    fname = os.path.basename(fpath)
    try:
        with Image.open(fpath) as raw_img:
            img = raw_img.convert("RGB")
        w, h = img.size
        page_lbl = f"{fname} · {w} × {h}"

        register_uploaded_sample(img, fname)
        samples_css = generate_samples_css()

        res = process_and_inspect_manga(img, auto_collect, session, filename_override=page_lbl)
        return (
            res[0], res[1], res[2], res[3], res[4], res[5],
            res[6], res[7], res[8], res[9], res[10], res[11],
            res[12], res[13], res[14],
            samples_css,
            res[15]
        )
    except Exception as e:
        logger.error("Upload process error: %s", e)
        return (
            session.get("annotated_img"), session.get("cleaned_img"),
            gr.update(), None, "", "", 0, 0, 0, 0,
            format_all_dialogues_summary(session),
            f"❌ Faylni oçib bölmadi: {e}",
            get_stats_text(),
            format_page_info_html(session.get("page_info", "")),
            format_red_badge_header(len(session.get("dialogues", []))),
            generate_samples_css(),
            session
        )


def load_sample_page(sample_idx, auto_collect, session_state):
    session = session_state if isinstance(session_state, dict) else get_default_session()
    samples = get_recent_samples()
    if 0 <= sample_idx < len(samples) and os.path.exists(samples[sample_idx].get("path", "")):
        target_file = samples[sample_idx]["path"]
        fname = samples[sample_idx]["name"]
    else:
        target_file = os.path.join(BASE_DIR, "sample_manga.jpg")
        fname = f"sahifa_0{sample_idx+1}.jpg"

    if not os.path.exists(target_file):
        return process_and_inspect_manga(None, auto_collect, session)

    with Image.open(target_file) as raw_img:
        img = raw_img.convert("RGB")
    w, h = img.size
    page_lbl = f"{fname} · {w} × {h}"
    return process_and_inspect_manga(img, auto_collect, session, filename_override=page_lbl)


def update_bubble_boundary(choice_str, x1, y1, x2, y2, session_state):
    session = session_state if isinstance(session_state, dict) else get_default_session()
    if not choice_str or not session.get("dialogues") or session.get("orig_img") is None:
        return (
            session.get("annotated_img"), session.get("cleaned_img"),
            None, "<div class='bubble-title-box'><h3>Bulut #(tanlanmadi)</h3><span class='dim-badge'>—</span></div>",
            0, 0, 0, 0, format_all_dialogues_summary(session), "⚠️ Avval bulutni tanlang!", session
        )

    bid = extract_bid_from_choice(choice_str)
    found_idx = -1
    for i, d in enumerate(session["dialogues"]):
        if d["bubble_id"] == bid:
            found_idx = i
            break

    if found_idx == -1:
        return (
            session.get("annotated_img"), session.get("cleaned_img"),
            None, "<div class='bubble-title-box'><h3>Bulut #(tanlanmadi)</h3><span class='dim-badge'>—</span></div>",
            0, 0, 0, 0, format_all_dialogues_summary(session), "❌ Bulut topilmadi.", session
        )

    orig = session["orig_img"]
    w, h = orig.size
    try:
        x1_val, y1_val, x2_val, y2_val = int(round(float(x1))), int(round(float(y1))), int(round(float(x2))), int(round(float(y2)))
    except Exception:
        return (
            session.get("annotated_img"), session.get("cleaned_img"),
            session.get("crops", {}).get(bid), f"<div class='bubble-title-box'><h3>Bulut #{bid}</h3></div>",
            x1, y1, x2, y2, format_all_dialogues_summary(session, active_bid=bid), "❌ Xato koordinatalar", session
        )

    old_bbox = list(session["dialogues"][found_idx]["bbox"])

    x1_val = max(0, min(w - 5, x1_val))
    y1_val = max(0, min(h - 5, y1_val))
    x2_val = max(x1_val + 5, min(w, x2_val))
    y2_val = max(y1_val + 5, min(h, y2_val))

    session["dialogues"][found_idx]["bbox"] = [x1_val, y1_val, x2_val, y2_val]
    session["dialogues"][found_idx]["text_box"] = [x1_val, y1_val, x2_val, y2_val]
    pad = 6
    c_x1, c_y1 = max(0, x1_val - pad), max(0, y1_val - pad)
    c_x2, c_y2 = min(w, x2_val + pad), min(h, y2_val + pad)
    new_crop = orig.crop((c_x1, c_y1, c_x2, c_y2))
    session["crops"][bid] = new_crop

    annotated, cleaned, _ = recompute_inpaint_and_annotated(
        session, target_bid=bid, old_bbox=old_bbox, active_bid=bid
    )
    crop, text, lbl, x1_out, y1_out, x2_out, y2_out, summary_html = get_bubble_details(session, bid)
    status_msg = f"✅ Bulut #{bid} çegarasi yangilandi ({x2_val-x1_val} × {y2_val-y1_val} px)."
    return (
        annotated, cleaned,
        crop, lbl,
        x1_out, y1_out, x2_out, y2_out,
        summary_html, status_msg, session
    )


def adjust_bubble_delta(choice_str, action, step, session_state):
    session = session_state if isinstance(session_state, dict) else get_default_session()
    if not choice_str or not session.get("dialogues") or session.get("orig_img") is None:
        return (
            session.get("annotated_img"), session.get("cleaned_img"),
            None, "", 0, 0, 0, 0, format_all_dialogues_summary(session), "⚠️ Bulut tanlanmagan", session
        )
    bid = extract_bid_from_choice(choice_str)
    found = None
    for d in session["dialogues"]:
        if d["bubble_id"] == bid:
            found = d
            break
    if not found:
        return (
            session.get("annotated_img"), session.get("cleaned_img"),
            None, "", 0, 0, 0, 0, format_all_dialogues_summary(session), "❌ Bulut topilmadi", session
        )
    box = found["bbox"]
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    try:
        s = int(step)
    except Exception:
        s = 10

    if action == "left":
        x1, x2 = x1 - s, x2 - s
    elif action == "right":
        x1, x2 = x1 + s, x2 + s
    elif action == "up":
        y1, y2 = y1 - s, y2 - s
    elif action == "down":
        y1, y2 = y1 + s, y2 + s
    elif action == "expand":
        x1, y1, x2, y2 = x1 - s, y1 - s, x2 + s, y2 + s
    elif action == "shrink":
        if (x2 - x1) > 2 * s and (y2 - y1) > 2 * s:
            x1, y1, x2, y2 = x1 + s, y1 + s, x2 - s, y2 - s

    return update_bubble_boundary(choice_str, x1, y1, x2, y2, session)


def add_from_clicked_coords(x1, y1, x2, y2, session_state):
    session = session_state if isinstance(session_state, dict) else get_default_session()
    if session.get("orig_img") is None:
        return (
            None, None, gr.update(), None, "", "", 0, 0, 0, 0,
            format_all_dialogues_summary(None),
            format_click_status(None),
            "⚠️ Avval manga sahifasini yuklang!",
            get_stats_text(),
            format_red_badge_header(0),
            session
        )

    orig = session["orig_img"]
    w, h = orig.size

    pt1 = session.get("click_pt1")
    pt2 = session.get("click_pt2")
    if pt1 is None or pt2 is None:
        cur_dialogues = session.get("dialogues", [])
        active_bid = cur_dialogues[0]["bubble_id"] if cur_dialogues else 1
        crop, text, lbl, rx1, ry1, rx2, ry2, summary_html = get_bubble_details(session, active_bid)
        choices = get_bubble_pill_choices(cur_dialogues)
        return (
            session.get("annotated_img"), session.get("cleaned_img"),
            gr.update(choices=choices, value=choices[0] if choices else "#1"),
            crop, text, lbl, rx1, ry1, rx2, ry2,
            summary_html,
            format_click_status(pt1, pt2),
            "⚠️ Yangi bulut yaratış uçun çap rasmda qarama-qarşi 2 ta burçakni bosing!",
            get_stats_text(),
            format_red_badge_header(len(cur_dialogues)),
            session
        )

    x1_val, x2_val = min(pt1[0], pt2[0]), max(pt1[0], pt2[0])
    y1_val, y2_val = min(pt1[1], pt2[1]), max(pt1[1], pt2[1])

    if x2_val <= x1_val + 5 or y2_val <= y1_val + 5:
        cur_dialogues = session.get("dialogues", [])
        active_bid = cur_dialogues[0]["bubble_id"] if cur_dialogues else 1
        crop, text, lbl, rx1, ry1, rx2, ry2, summary_html = get_bubble_details(session, active_bid)
        choices = get_bubble_pill_choices(cur_dialogues)
        return (
            session.get("annotated_img"), session.get("cleaned_img"),
            gr.update(choices=choices, value=choices[0] if choices else "#1"),
            crop, text, lbl, rx1, ry1, rx2, ry2,
            summary_html,
            format_click_status(pt1, pt2),
            "⚠️ Yangi bulut yaratış uçun çap rasmda qarama-qarşi 2 ta burçakni bosing!",
            get_stats_text(),
            format_red_badge_header(len(cur_dialogues)),
            session
        )

    x1_val = max(0, min(w - 5, x1_val))
    y1_val = max(0, min(h - 5, y1_val))
    x2_val = max(x1_val + 5, min(w, x2_val))
    y2_val = max(y1_val + 5, min(h, y2_val))

    pad = 6
    c_x1, c_y1 = max(0, x1_val - pad), max(0, y1_val - pad)
    c_x2, c_y2 = min(w, x2_val + pad), min(h, y2_val + pad)
    crop = orig.crop((c_x1, c_y1, c_x2, c_y2))

    recognized_text = ""
    with pipeline.model_lock:
        res = pipeline._get_ocr_reader(session.get("source_lang", "ja"))
        reader = res[0] if isinstance(res, tuple) else res
        if reader is not None:
            try:
                ocr_res = reader.readtext(np.array(crop))
                if ocr_res:
                    recognized_text = " ".join([r[1] for r in ocr_res if len(r[1].strip()) > 0]).strip()
            except Exception as e:
                logger.debug("EasyOCR read error on manual crop: %s", e)

    existing_bids = [d.get("bubble_id", 0) for d in session.get("dialogues", [])]
    next_id = (max(existing_bids) + 1) if existing_bids else 1
    new_entry = {
        "bubble_id": next_id,
        "category": "speech_bubble",
        "bbox": [x1_val, y1_val, x2_val, y2_val],
        "text_box": [x1_val, y1_val, x2_val, y2_val],
        "transcription": recognized_text
    }
    session["dialogues"].append(new_entry)
    session["crops"][next_id] = crop
    session["click_pt1"] = None
    session["click_pt2"] = None

    annotated, cleaned, _ = recompute_inpaint_and_annotated(session, target_bid=next_id, active_bid=next_id)
    choices = get_bubble_pill_choices(session["dialogues"])
    current_choice = f"#{next_id}"
    crop, text, lbl, rx1, ry1, rx2, ry2, summary_html = get_bubble_details(session, next_id)
    status_msg = f"➕ **Yangi Bulut #{next_id}** yaratildi ({x2_val-x1_val} × {y2_val-y1_val} px)."

    return (
        annotated, cleaned,
        gr.update(choices=choices, value=current_choice),
        crop, text, lbl,
        rx1, ry1, rx2, ry2,
        summary_html,
        format_click_status(None),
        status_msg,
        get_stats_text(),
        format_red_badge_header(len(session["dialogues"])),
        session
    )


def delete_selected_bubble(choice_str, session_state):
    session = session_state if isinstance(session_state, dict) else get_default_session()
    if not choice_str or not session.get("dialogues") or session.get("orig_img") is None:
        return (
            session.get("annotated_img"), session.get("cleaned_img"),
            None, "",
            "<div class='bubble-title-box'><h3>Bulut #(tanlanmadi)</h3><span class='dim-badge'>—</span></div>",
            0, 0, 0, 0, gr.update(choices=["#1"], value="#1"),
            format_all_dialogues_summary(session),
            "⚠️ Oçiriş uçun bulut tanlanmagan!",
            format_red_badge_header(len(session.get("dialogues", []))),
            session
        )

    bid = extract_bid_from_choice(choice_str)
    found_idx = -1
    for i, d in enumerate(session["dialogues"]):
        if d["bubble_id"] == bid:
            found_idx = i
            break

    if found_idx == -1:
        return (
            session.get("annotated_img"), session.get("cleaned_img"),
            None, "", "", 0, 0, 0, 0, gr.update(),
            format_all_dialogues_summary(session),
            "❌ Bulut topilmadi.",
            format_red_badge_header(len(session.get("dialogues", []))),
            session
        )

    deleted_entry = session["dialogues"][found_idx]
    deleted_box = deleted_entry.get("bbox")
    del session["dialogues"][found_idx]
    orig = session["orig_img"]
    w, h = orig.size

    # Restore deleted box from original image instead of re-inpainting entire page
    if deleted_box and session.get("cleaned_img") is not None:
        dx1, dy1 = max(0, int(deleted_box[0])), max(0, int(deleted_box[1]))
        dx2, dy2 = min(w, int(deleted_box[2])), min(h, int(deleted_box[3]))
        if dx2 > dx1 and dy2 > dy1:
            session["cleaned_img"].paste(orig.crop((dx1, dy1, dx2, dy2)), (dx1, dy1))
            if session.get("mask_img") is not None:
                d_mask = ImageDraw.Draw(session["mask_img"])
                d_mask.rectangle([dx1, dy1, dx2, dy2], fill=0)

    new_crops = {}
    for new_id, d in enumerate(session["dialogues"], start=1):
        d["bubble_id"] = new_id
        box = d["bbox"]
        pad = 6
        c_x1, c_y1 = max(0, box[0] - pad), max(0, box[1] - pad)
        c_x2, c_y2 = min(w, box[2] + pad), min(h, box[3] + pad)
        new_crops[new_id] = orig.crop((c_x1, c_y1, c_x2, c_y2))
    session["crops"] = new_crops

    first_bid = 1 if session["dialogues"] else None
    session["annotated_img"] = redraw_annotated_page(session, active_bid=first_bid)
    annotated = session["annotated_img"]
    cleaned = session.get("cleaned_img")
    choices = get_bubble_pill_choices(session["dialogues"])
    first_choice = choices[0] if choices else "#1"
    crop, text, lbl, rx1, ry1, rx2, ry2, summary_html = get_bubble_details(session, first_bid)
    status_msg = f"🗑️ **Bulut #{bid}** olib taşlandi! Qolgan {len(session['dialogues'])} ta bulut qayta raqamlandi."

    return (
        annotated, cleaned,
        crop, text, lbl,
        rx1, ry1, rx2, ry2,
        gr.update(choices=choices, value=first_choice),
        summary_html, status_msg,
        format_red_badge_header(len(session["dialogues"])),
        session
    )


def save_single_bubble_correction(selected_choice, edited_text, session_state):
    session = session_state if isinstance(session_state, dict) else get_default_session()
    if not selected_choice or not session.get("dialogues"):
        return (
            session.get("annotated_img"), gr.update(),
            format_all_dialogues_summary(session), edited_text,
            "⚠️ Avval bulutni tanlang!", get_stats_text(), session
        )

    bid = extract_bid_from_choice(selected_choice)
    clean_text = (edited_text or "").strip()
    if not clean_text:
        return (
            session.get("annotated_img"), gr.update(),
            format_all_dialogues_summary(session, active_bid=bid), edited_text,
            "⚠️ Iltimos, matnni kiriting!", get_stats_text(), session
        )

    found_idx = -1
    for i, d in enumerate(session["dialogues"]):
        if d["bubble_id"] == bid:
            found_idx = i
            break

    if found_idx == -1:
        return (
            session.get("annotated_img"), gr.update(),
            format_all_dialogues_summary(session), edited_text,
            "❌ Bulut topilmadi.", get_stats_text(), session
        )

    session["dialogues"][found_idx]["transcription"] = clean_text
    crop = session.get("crops", {}).get(bid)
    if crop is not None:
        try:
            collector.record_ocr_correction(crop, clean_text, bubble_id=bid, category="speech_bubble")
        except Exception as e:
            logger.warning("Error recording OCR correction: %s", e)

    if session.get("cleaned_img") is not None and any(d.get("translation") for d in session.get("dialogues", [])):
        try:
            from manga_pipeline import render_translated_dialogues
            session["annotated_img"] = render_translated_dialogues(session["cleaned_img"], session["dialogues"])
        except Exception as e:
            logger.debug("Render translated dialogues error: %s", e)

    choices = get_bubble_pill_choices(session["dialogues"])
    current_choice = f"#{bid}"
    summary_html = format_all_dialogues_summary(session, active_bid=bid)
    status_msg = f"💾 **Bulut #{bid}** saqlandi va öqitiş bazasiga qöşildi!"

    return (
        session.get("annotated_img"),
        gr.update(choices=choices, value=current_choice),
        summary_html,
        clean_text,
        status_msg,
        get_stats_text(),
        session
    )


def rerun_ocr_for_bubble(choice_str, session_state):
    session = session_state if isinstance(session_state, dict) else get_default_session()
    if not choice_str or not session.get("dialogues"):
        return None, "", gr.update(), format_all_dialogues_summary(session), "⚠️ Bulut tanlanmagan!", session
    bid = extract_bid_from_choice(choice_str)
    found_idx = -1
    for i, d in enumerate(session["dialogues"]):
        if d["bubble_id"] == bid:
            found_idx = i
            break
    if found_idx == -1:
        return None, "", gr.update(), format_all_dialogues_summary(session), "❌ Bulut topilmadi.", session

    crop = session.get("crops", {}).get(bid)
    if crop is None and session.get("orig_img") is not None:
        box = session["dialogues"][found_idx]["bbox"]
        crop = session["orig_img"].crop((box[0], box[1], box[2], box[3]))
        session["crops"][bid] = crop

    if crop is None:
        return None, "", gr.update(), format_all_dialogues_summary(session, active_bid=bid), "❌ Kadr topilmadi.", session

    recognized_text = ""
    with pipeline.model_lock:
        res = pipeline._get_ocr_reader(session.get("source_lang", "ja"))
        reader = res[0] if isinstance(res, tuple) else res
        if reader is not None:
            try:
                ocr_res = reader.readtext(np.array(crop))
                if ocr_res:
                    recognized_text = " ".join([r[1] for r in ocr_res if len(r[1].strip()) > 0]).strip()
            except Exception as e:
                logger.debug("EasyOCR re-run exception: %s", e)

    session["dialogues"][found_idx]["transcription"] = recognized_text
    choices = get_bubble_pill_choices(session["dialogues"])
    current_choice = f"#{bid}"
    summary_text = format_all_dialogues_summary(session, active_bid=bid)
    status_msg = f"🔍 **Bulut #{bid}** matni qayta öqildi: '{recognized_text}'"
    return crop, recognized_text, gr.update(choices=choices, value=current_choice), summary_text, status_msg, session


def get_training_data_description():
    """Describe the permanent base without scanning its images during UI startup."""
    archive_dir = os.path.join(BASE_DIR, "manga_ml_datasets", "characters")
    if os.path.isfile(os.path.join(archive_dir, "characters.zip")):
        try:
            with open(os.path.join(archive_dir, "dataset_info.json"), encoding="utf-8") as f:
                info = json.load(f)
            image_count = f"{int(info['image_count']):,}".replace(",", " ")
            class_count = int(info["class_count"])
            return (
                f"**Doimiy belgilar bazasi: {image_count} ta rasm, {class_count} ta sinf "
                "(0–9, A–Z, a–z).**\n\n"
                "OCR öqitişda uşbu bazaning öquv qismi va yiğilgan tuzatişlar "
                "birga işlatiladi. Doimiy baza 5 000 ta namuna çekloviga kirmaydi."
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Character base description unavailable: %s", exc)
    return "Töğrilangan namunalar asosida neyron tarmoqni qayta öqitiş (fine-tuning)."


def trigger_manual_fine_tune(epochs):
    log_lines = []

    def log_cb(text):
        log_lines.append(f"[{time.strftime('%H:%M:%S')}] {text}")

    log_cb("Öz-özini öqitiş jarayoni başlanmoqda...")
    yield "\n".join(log_lines), "⏳ Öqitiş jarayoni ketmoqda...", get_stats_text()

    res = trainer.execute_self_training_cycle(epochs=int(epochs), log_callback=log_cb, pipeline_ref=pipeline)
    status_banner = (
        f"✅ Tekşirilgan modellar yangilandi: {', '.join(res['promoted_tasks'])}."
        if res["success"] else "ℹ️ Modellar yangilanmadi. Tekşiruv sabablarini jurnalda köring."
    )
    yield "\n".join(log_lines), status_banner, get_stats_text()


def update_threshold(val):
    stats = collector.load_stats()
    stats["auto_train_threshold"] = int(val)
    collector.save_stats(stats)
    return f"✅ Avto-öqitiş çegarasi saqlandi: {val}", get_stats_text()


def hot_reload_models_action():
    pipeline.reload_models()
    return "✅ Modellar xotirada yangilandi!", get_stats_text()


def batch_learn_chapter(num_pages, chapter_dir_input, source_lang="ja"):
    if not os.path.exists(chapter_dir_input):
        return f"❌ Papka topilmadi: {chapter_dir_input}", get_stats_text()
    valid_exts = (".jpg", ".jpeg", ".png", ".webp")
    image_files = sorted([
        f for f in os.listdir(chapter_dir_input)
        if f.lower().endswith(valid_exts) and not f.startswith("test_cleaned")
    ])
    if not image_files:
        return f"❌ Papkada rasm fayllari topilmadi: {chapter_dir_input}", get_stats_text()

    num = min(int(num_pages), len(image_files))
    target_files = image_files[:num]
    logs = [f"🚀 Manga bobini sinaş başlandi ({num} ta sahifa)..."]
    for i, fname in enumerate(target_files, 1):
        fpath = os.path.join(chapter_dir_input, fname)
        t0 = time.time()
        try:
            res = pipeline.process_page(fpath, auto_collect=True, source_lang=source_lang)
            elapsed = time.time() - t0
            logs.append(f"  [{i}/{num}] {fname} -> {len(res['dialogues'])} ta bulut aniqlandi ({elapsed:.2f}s)")
        except Exception as e:
            logs.append(f"  [{i}/{num}] {fname} -> Xatolik: {e}")
    logs.append("✅ Sinov yakunlandi!")
    return "\n".join(logs), get_stats_text()


# ==============================================================================
# INITIAL SHOWCASE SESSION BUILDER
# ==============================================================================
def _build_initial_session():
    sample_path = os.path.join(BASE_DIR, "test_chapter_57", "ogami_tsumiki_to_kinichijou_57_4.jpg")
    if not os.path.exists(sample_path):
        sample_path = os.path.join(BASE_DIR, "sample_manga.jpg")

    s = get_default_session()
    if os.path.exists(sample_path):
        with Image.open(sample_path) as raw_img:
            orig = raw_img.convert("RGB")
        s["orig_img"] = orig
        w, h = orig.size
        s["page_info"] = f"sahifa_04.jpg · {w} × {h}"

        sx = w / 800.0
        sy = h / 1168.0
        s["dialogues"] = [
            {
                "bubble_id": 1,
                "category": "speech_bubble",
                "bbox": [int(115 * sx), int(72 * sy), int(278 * sx), int(218 * sy)],
                "text_box": [int(115 * sx), int(72 * sy), int(278 * sx), int(218 * sy)],
                "transcription": "どうしたの？"
            },
            {
                "bubble_id": 2,
                "category": "speech_bubble",
                "bbox": [int(408 * sx), int(51 * sy), int(520 * sx), int(152 * sy)],
                "text_box": [int(408 * sx), int(51 * sy), int(520 * sx), int(152 * sy)],
                "transcription": "なんでもないよ、ただ少し考えていただけ。"
            },
            {
                "bubble_id": 3,
                "category": "speech_bubble",
                "bbox": [int(288 * sx), int(395 * sy), int(468 * sx), int(480 * sy)],
                "text_box": [int(288 * sx), int(395 * sy), int(468 * sx), int(480 * sy)],
                "transcription": "ほんとうに？"
            },
            {
                "bubble_id": 4,
                "category": "speech_bubble",
                "bbox": [int(325 * sx), int(545 * sy), int(475 * sx), int(630 * sy)],
                "text_box": [int(325 * sx), int(545 * sy), int(475 * sx), int(630 * sy)],
                "transcription": "ありがとう。"
            }
        ]
        s["crops"] = {}
        for d in s["dialogues"]:
            bid = d["bubble_id"]
            box = d["bbox"]
            pad = 6
            c_x1, c_y1 = max(0, box[0] - pad), max(0, box[1] - pad)
            c_x2, c_y2 = min(w, box[2] + pad), min(h, box[3] + pad)
            s["crops"][bid] = orig.crop((c_x1, c_y1, c_x2, c_y2))

        recompute_inpaint_and_annotated(s, active_bid=2)
    return s


INITIAL_SESSION = _build_initial_session()


def get_initial_session():
    return copy.deepcopy(INITIAL_SESSION)
_init_b2_box = INITIAL_SESSION["dialogues"][1]["bbox"] if len(INITIAL_SESSION.get("dialogues", [])) >= 2 else [408, 51, 520, 152]
_init_b2_w = _init_b2_box[2] - _init_b2_box[0]
_init_b2_h = _init_b2_box[3] - _init_b2_box[1]

# ==============================================================================
# CSS & JS: STRICT 100VH VIEWPORT LOCK (NO WINDOW SCROLL)
# ==============================================================================
CUSTOM_CSS = """
:root {
    --bg-dark: #0a0c13;
    --sidebar-bg: #10131e;
    --card-bg: #131724;
    --border-color: #1e2334;
    --accent-purple: #6366f1;
    --accent-coral: #ea580c;
    --accent-coral-glow: rgba(234, 88, 12, 0.25);
    --text-muted: #828a9e;
    --text-dim: #555e72;
    --text-light: #f1f5f9;
}

/* 1. VIEWPORT CONSTRAINT: ABSOLUTELY NO MAIN WINDOW SCROLL */
html, body {
    margin: 0 !important;
    padding: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    background-color: var(--bg-dark) !important;
    color: var(--text-light) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    box-sizing: border-box !important;
}

.gradio-container {
    width: 100vw !important;
    max-width: 100vw !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    padding: 6px 12px 6px 12px !important;
    box-sizing: border-box !important;
    background-color: var(--bg-dark) !important;
}

.gradio-container > .main,
.gradio-container > .wrap,
.gradio-container > div {
    height: 100% !important;
    max-height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 6px !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
}

footer {
    display: none !important;
}

/* 2. TOP HEADER BAR */
#top_header_row {
    flex: 0 0 46px !important;
    height: 46px !important;
    min-height: 46px !important;
    max-height: 46px !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    background: var(--sidebar-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    padding: 4px 14px !important;
    margin: 0 !important;
    box-sizing: border-box !important;
}

.brand-box {
    display: flex;
    align-items: center;
    gap: 10px;
}
.brand-icon-box {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    color: white;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.4);
}
.brand-text-box h2 {
    margin: 0 !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    letter-spacing: -0.2px;
    line-height: 1.2 !important;
}
.brand-text-box p {
    margin: 0 !important;
    font-size: 11px !important;
    color: var(--text-muted) !important;
    line-height: 1.2 !important;
}

.stats-row {
    display: flex;
    align-items: center;
    gap: 6px;
}
.stat-pill {
    background: #141826;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 3px 10px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-width: 64px;
}
.pill-lbl {
    font-size: 9px;
    color: var(--text-muted);
    font-weight: 500;
    line-height: 1.1;
}
.pill-val {
    font-size: 13px;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.2;
}
.model-pill {
    background: #141826;
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 6px;
    padding: 4px 10px;
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    font-weight: 600;
    color: #ffffff;
    height: 34px;
    box-sizing: border-box;
}
.dot-green {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 6px #10b981;
}
.badge-active {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
    font-size: 9px;
    padding: 1px 5px;
    border-radius: 4px;
    font-weight: 600;
}

/* 3. MAIN 3-COLUMN STUDIO LAYOUT */
#app_main_layout {
    flex: 1 1 auto !important;
    height: calc(100vh - 66px) !important;
    min-height: calc(100vh - 66px) !important;
    max-height: calc(100vh - 66px) !important;
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 8px !important;
    overflow: hidden !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    width: 100% !important;
}

/* 4. LEFT SIDEBAR */
#sidebar_col {
    flex: 0 0 160px !important;
    width: 160px !important;
    max-width: 160px !important;
    min-width: 160px !important;
    height: 100% !important;
    min-height: 100% !important;
    max-height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    background: var(--sidebar-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    padding: 10px 8px !important;
    box-sizing: border-box !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    gap: 6px !important;
    visibility: visible !important;
}

.sidebar-heading {
    font-size: 10px !important;
    font-weight: 700 !important;
    color: var(--text-dim) !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    margin: 4px 4px 2px 4px !important;
}

.sidebar-nav-btn {
    width: 100% !important;
    text-align: left !important;
    padding: 8px 10px !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    color: var(--text-muted) !important;
    transition: all 0.15s ease !important;
    height: 32px !important;
    min-height: 32px !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}
.sidebar-nav-btn:hover {
    background: #171b2a !important;
    color: #ffffff !important;
}
.sidebar-nav-btn.active-tab {
    background: #1a1f32 !important;
    color: #ffffff !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
}

.samples-preview-row {
    display: flex !important;
    flex-direction: row !important;
    gap: 6px !important;
    margin-top: 4px !important;
}
.sample-thumb-btn {
    flex: 1 !important;
    min-width: 0 !important;
    height: 56px !important;
    max-height: 56px !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 6px !important;
    padding: 0 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: contain !important;
    background-color: #eae5d9 !important;
}
.sample-thumb-btn:hover {
    border-color: var(--accent-coral) !important;
    box-shadow: 0 0 8px var(--accent-coral-glow) !important;
    transform: translateY(-1px) !important;
}

.sample-thumb-1, .sample-thumb-2, .sample-thumb-3 {
    background-color: #141826 !important;
    background-size: cover !important;
    background-position: center !important;
}

#chk_auto_collect {
    margin-top: auto !important;
    padding: 6px 8px !important;
    background: #141826 !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 6px !important;
    font-size: 11px !important;
}
#chk_auto_collect label {
    color: #cbd5e1 !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    line-height: 1.3 !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    cursor: pointer !important;
}
#chk_auto_collect input[type="checkbox"] {
    accent-color: #6366f1 !important;
    width: 14px !important;
    height: 14px !important;
    cursor: pointer !important;
}

/* 5. CENTER WORKSPACE */
#main_workspace_col {
    flex: 1 1 0% !important;
    min-width: 0 !important;
    width: 0 !important;
    height: 100% !important;
    min-height: 100% !important;
    max-height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    background: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    padding: 8px 12px !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
    gap: 6px !important;
}

#panel_studio {
    height: 100% !important;
    max-height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 6px !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
}

#center_action_bar {
    flex: 0 0 32px !important;
    height: 32px !important;
    min-height: 32px !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin: 0 !important;
    gap: 8px !important;
    flex-wrap: nowrap !important;
}

.view-toggle-group {
    display: inline-flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    background: #0d0f17 !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 6px !important;
    padding: 2px !important;
    gap: 2px !important;
    width: auto !important;
    min-width: 0 !important;
}
.view-pill-btn {
    height: 24px !important;
    min-height: 24px !important;
    padding: 0 10px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    border-radius: 4px !important;
    background: transparent !important;
    border: none !important;
    color: var(--text-muted) !important;
    cursor: pointer !important;
    width: auto !important;
    min-width: 0 !important;
    white-space: nowrap !important;
    flex: 0 0 auto !important;
}
.view-pill-btn.active-view {
    background: #1d2238 !important;
    color: #ffffff !important;
}

.page-info-pill {
    font-size: 12px !important;
    color: var(--text-muted) !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    white-space: nowrap !important;
}

.upload-toggle-btn {
    height: 26px !important;
    min-height: 26px !important;
    padding: 0 12px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    background: #171b2a !important;
    border: 1px solid var(--border-color) !important;
    color: #cbd5e1 !important;
    cursor: pointer !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
    white-space: nowrap !important;
}
.upload-toggle-btn:hover {
    border-color: var(--accent-purple) !important;
    color: #ffffff !important;
}

/* AI Action row */
#action_bar_row {
    flex: 0 0 36px !important;
    height: 36px !important;
    min-height: 36px !important;
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    margin: 0 !important;
    flex-wrap: nowrap !important;
}

#btn_ai_process {
    height: 34px !important;
    min-height: 34px !important;
    padding: 0 16px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    border: none !important;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.35) !important;
    cursor: pointer !important;
    white-space: nowrap !important;
    flex: 0 0 auto !important;
}
#btn_ai_process:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
}

#status_indicator,
#status_indicator p,
#status_indicator .prose,
#status_indicator .prose p {
    font-size: 12px !important;
    color: #cbd5e1 !important;
    margin: 0 !important;
    padding: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    white-space: nowrap !important;
}

/* Subheadings row */
.panel-headers-row {
    flex: 0 0 22px !important;
    height: 22px !important;
    min-height: 22px !important;
    display: flex !important;
    align-items: center !important;
    margin: 0 !important;
}
.panel-subhead {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 4px;
}
.panel-title {
    font-size: 12px;
    font-weight: 700;
    color: #e2e8f0;
}
.badge-tag-red {
    background: rgba(234, 88, 12, 0.15);
    border: 1px solid rgba(234, 88, 12, 0.35);
    color: #fb923c;
    font-size: 10px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 4px;
}
.badge-tag-green {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #34d399;
    font-size: 10px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 4px;
}

/* Images Row */
#visual_outputs_row {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    height: calc(100vh - 258px) !important;
    max-height: calc(100vh - 258px) !important;
    display: flex !important;
    flex-direction: row !important;
    gap: 8px !important;
    overflow: hidden !important;
    margin: 0 !important;
}

#col_annotated, #col_clean {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    height: 100% !important;
    max-height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    border-radius: 8px !important;
    background: #0d0f17 !important;
    border: 1px solid var(--border-color) !important;
}

#annotated_img_box, #clean_img_box {
    flex: 1 1 auto !important;
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0 !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
    background: #0d0f17 !important;
    padding: 0 !important;
    position: relative !important;
}

#annotated_img_box > div, #clean_img_box > div,
#annotated_img_box .image-frame, #clean_img_box .image-frame,
#annotated_img_box .image-container, #clean_img_box .image-container,
#annotated_img_box .wrap, #clean_img_box .wrap {
    height: 100% !important;
    max-height: 100% !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

#annotated_img_box button, #clean_img_box button {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    height: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    background: transparent !important;
    cursor: default !important;
}
#annotated_img_box button {
    cursor: crosshair !important;
}

#annotated_img_box img, #clean_img_box img {
    max-height: 100% !important;
    max-width: 100% !important;
    width: auto !important;
    height: auto !important;
    object-fit: contain !important;
    display: block !important;
    margin: auto !important;
    border-radius: 4px !important;
}

#annotated_img_box .image-controls, #clean_img_box .image-controls,
#annotated_img_box .icon-buttons, #clean_img_box .icon-buttons,
#annotated_img_box [aria-label="Download"], #clean_img_box [aria-label="Download"],
#annotated_img_box [aria-label="Clear"], #clean_img_box [aria-label="Clear"],
#annotated_img_box [aria-label="Share"], #clean_img_box [aria-label="Share"],
#annotated_img_box [aria-label="Fullscreen"], #clean_img_box [aria-label="Fullscreen"] {
    display: none !important;
}

/* Bottom Card: Yangi bulut qo'shish */
#new_bubble_card {
    flex: 0 0 42px !important;
    height: 42px !important;
    min-height: 42px !important;
    background: #10131f !important;
    border: 1px dashed #242a3e !important;
    border-radius: 8px !important;
    padding: 2px 10px !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 8px !important;
    box-sizing: border-box !important;
}

.new-bubble-title {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    font-weight: 700;
    color: #ffffff;
}
.new-bubble-subtitle {
    font-size: 11px;
    color: var(--text-muted);
}
.coord-status-group {
    display: flex;
    gap: 6px;
}
.coord-pill {
    background: #181c2c;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 10px;
    color: var(--text-muted);
    font-family: monospace;
}
.coord-pill.active {
    background: #23283e;
    border-color: rgba(99, 102, 241, 0.5);
    color: #ffffff;
    font-weight: 600;
}

/* 6. RIGHT INSPECTOR */
#inspector_col {
    flex: 0 0 340px !important;
    width: 340px !important;
    max-width: 340px !important;
    min-width: 340px !important;
    height: calc(100vh - 66px) !important;
    min-height: calc(100vh - 66px) !important;
    max-height: calc(100vh - 66px) !important;
    display: flex !important;
    flex-direction: column !important;
    background: var(--sidebar-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    padding: 6px 10px !important;
    box-sizing: border-box !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    gap: 3px !important;
}

.inspector-top-row {
    flex: 0 0 auto !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin: 0 !important;
}
.bubble-title-box {
    display: flex;
    align-items: baseline;
    gap: 6px;
}
.bubble-title-box h3 {
    margin: 0 !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}
.bubble-title-box .dim-badge {
    font-size: 10px !important;
    color: var(--text-muted) !important;
    font-family: monospace !important;
}
.inspector-icon-group {
    display: flex !important;
    gap: 4px !important;
}
.icon-nav-btn {
    width: 24px !important;
    height: 24px !important;
    min-width: 24px !important;
    padding: 0 !important;
    border-radius: 5px !important;
    background: #171b2a !important;
    border: 1px solid var(--border-color) !important;
    color: #cbd5e1 !important;
    font-size: 10px !important;
    cursor: pointer !important;
}
.icon-del-btn {
    width: 24px !important;
    height: 24px !important;
    min-width: 24px !important;
    padding: 0 !important;
    border-radius: 5px !important;
    background: rgba(239, 68, 68, 0.12) !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
    color: #f87171 !important;
    font-size: 11px !important;
}

/* Horizontal Bubble Pills */
.bubble-pills-radio {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}
.bubble-pills-radio .wrap {
    display: flex !important;
    flex-direction: row !important;
    gap: 5px !important;
    flex-wrap: wrap !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
.bubble-pills-radio label {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: #171b2a !important;
    border: 1px solid #23283e !important;
    border-radius: 9999px !important;
    padding: 1px 8px !important;
    min-width: 28px !important;
    height: 22px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    color: #94a3b8 !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    margin: 0 !important;
}
.bubble-pills-radio label input[type="radio"] {
    position: absolute !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
    pointer-events: none !important;
    margin: 0 !important;
}
.bubble-pills-radio label:hover {
    background: #20263c !important;
    color: #ffffff !important;
    border-color: #6366f1 !important;
}
.bubble-pills-radio label.selected,
.bubble-pills-radio label:has(input:checked) {
    background: rgba(234, 88, 12, 0.15) !important;
    border: 1.5px solid #ea580c !important;
    color: #ea580c !important;
    box-shadow: 0 0 10px rgba(234, 88, 12, 0.25) !important;
}

/* Crop image preview container */
#bubble_crop_box {
    flex: 0 0 66px !important;
    height: 66px !important;
    max-height: 66px !important;
    min-height: 66px !important;
    border-radius: 8px !important;
    background: #0d0f17 !important;
    border: 1px solid var(--border-color) !important;
    overflow: hidden !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 2px !important;
    box-sizing: border-box !important;
}
#bubble_crop_box > div,
#bubble_crop_box .image-frame,
#bubble_crop_box .image-container,
#bubble_crop_box .wrap {
    height: 100% !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
#bubble_crop_box button {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    height: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    background: transparent !important;
}
#bubble_crop_box img {
    max-height: 100% !important;
    max-width: 100% !important;
    width: auto !important;
    height: auto !important;
    object-fit: contain !important;
    display: block !important;
    margin: auto !important;
}
#bubble_crop_box .icon-buttons,
#bubble_crop_box [aria-label="Download"],
#bubble_crop_box [aria-label="Clear"] {
    display: none !important;
}

/* Nutq matni textarea */
.nutq-matni-lbl {
    font-size: 10px !important;
    color: var(--text-muted) !important;
    margin: 1px 0 !important;
}
#corrected_text_input textarea {
    height: 38px !important;
    min-height: 38px !important;
    max-height: 38px !important;
    background: #0d0f17 !important;
    border: 1px solid var(--border-color) !important;
    color: #f8fafc !important;
    font-size: 11px !important;
    border-radius: 6px !important;
    padding: 4px 6px !important;
    resize: none !important;
}

.text-actions-row {
    display: flex !important;
    gap: 6px !important;
    margin: 1px 0 !important;
}
#btn_save_bubble {
    flex: 2 !important;
    height: 25px !important;
    min-height: 25px !important;
    background: #4f46e5 !important;
    color: #ffffff !important;
    border-radius: 6px !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    border: none !important;
}
#btn_re_ocr {
    flex: 1 !important;
    height: 25px !important;
    min-height: 25px !important;
    background: #171b2a !important;
    border: 1px solid var(--border-color) !important;
    color: #cbd5e1 !important;
    border-radius: 6px !important;
    font-size: 10px !important;
    font-weight: 600 !important;
}

/* Coordinates inputs */
.coords-grid {
    display: grid !important;
    grid-template-columns: repeat(4, 1fr) !important;
    gap: 4px !important;
    margin: 1px 0 !important;
}
.coords-grid .form {
    padding: 0 !important;
    margin: 0 !important;
}
.coords-grid input {
    height: 22px !important;
    min-height: 22px !important;
    background: #0d0f17 !important;
    border: 1px solid var(--border-color) !important;
    color: #f8fafc !important;
    font-size: 10px !important;
    text-align: center !important;
    padding: 1px !important;
    border-radius: 4px !important;
}

/* D-Pad & Controls */
.dpad-container {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    gap: 6px !important;
    margin: 1px 0 !important;
}
.dpad-grid {
    display: grid !important;
    grid-template-columns: repeat(3, 22px) !important;
    grid-gap: 2px !important;
}
.dpad-btn {
    width: 22px !important;
    height: 22px !important;
    min-width: 22px !important;
    padding: 0 !important;
    background: #171b2a !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
    font-size: 9px !important;
    color: #cbd5e1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
}
.dpad-btn-center {
    color: #ea580c !important;
}
.dpad-right-actions {
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
    flex: 1 !important;
}
.resize-btn {
    height: 19px !important;
    min-height: 19px !important;
    padding: 0 4px !important;
    font-size: 9px !important;
    font-weight: 600 !important;
    background: #171b2a !important;
    border: 1px solid var(--border-color) !important;
    color: #cbd5e1 !important;
    border-radius: 4px !important;
}
.step-radio .wrap {
    display: flex !important;
    gap: 3px !important;
}
.step-radio label {
    padding: 1px 5px !important;
    font-size: 9px !important;
    border-radius: 4px !important;
    background: #171b2a !important;
    border: 1px solid var(--border-color) !important;
}

#btn_apply_bbox_full {
    height: 24px !important;
    min-height: 24px !important;
    background: #181d2f !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    color: #e2e8f0 !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    border-radius: 5px !important;
    width: 100% !important;
    margin-top: 1px !important;
}

/* Dialogues list */
.dialogues-container {
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
    max-height: 85px !important;
    overflow-y: auto !important;
    padding-right: 2px !important;
    margin-top: 1px !important;
}
.dialogue-card {
    background: #0d0f17 !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 5px !important;
    padding: 2px 6px !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
    font-size: 10px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}
.dialogue-card.active-dialogue-card {
    background: #18141e !important;
    border-color: #ea580c !important;
    box-shadow: 0 0 8px rgba(234, 88, 12, 0.15) !important;
}
.dialogue-tag {
    background: #1c2132 !important;
    color: #94a3b8 !important;
    font-weight: 700 !important;
    font-size: 9px !important;
    padding: 1px 4px !important;
    border-radius: 3px !important;
}
.active-dialogue-card .dialogue-tag {
    background: #ea580c !important;
    color: #ffffff !important;
}
.dialogue-text {
    flex: 1 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    color: #cbd5e1 !important;
}

#inspector_col::-webkit-scrollbar,
.dialogues-container::-webkit-scrollbar,
#sidebar_col::-webkit-scrollbar {
    width: 4px;
}
#inspector_col::-webkit-scrollbar-thumb,
.dialogues-container::-webkit-scrollbar-thumb,
#sidebar_col::-webkit-scrollbar-thumb {
    background: #252b3e;
    border-radius: 4px;
}

/* RESPONSIVE LAYOUT */
@media (max-width: 1100px) {
    .gradio-container {
        padding: 4px 8px 4px 8px !important;
    }
    #sidebar_col {
        flex: 0 0 120px !important;
        width: 120px !important;
        max-width: 120px !important;
        min-width: 120px !important;
        padding: 8px 6px !important;
    }
    #inspector_col {
        flex: 0 0 280px !important;
        width: 280px !important;
        max-width: 280px !important;
        min-width: 280px !important;
    }
    .stat-pill {
        display: none !important;
    }
}

@media (max-width: 900px) {
    .gradio-container {
        padding: 4px 6px !important;
    }
    #sidebar_col {
        flex: 0 0 36px !important;
        width: 36px !important;
        max-width: 36px !important;
        min-width: 36px !important;
        padding: 6px 4px !important;
        overflow: hidden !important;
    }
    #sidebar_col .sidebar-heading,
    #sidebar_col .sidebar-nav-btn span,
    #sidebar_col .samples-preview-row,
    #sidebar_col #chk_auto_collect {
        display: none !important;
    }
    #sidebar_col .sidebar-nav-btn {
        padding: 6px 4px !important;
        justify-content: center !important;
        font-size: 14px !important;
    }
    #inspector_col {
        flex: 0 0 240px !important;
        width: 240px !important;
        max-width: 240px !important;
        min-width: 240px !important;
        padding: 4px 6px !important;
        gap: 2px !important;
    }
    #top_header_row {
        padding: 4px 10px !important;
    }
    .brand-text-box p,
    .model-pill {
        display: none !important;
    }
    #app_main_layout {
        gap: 5px !important;
    }
}

@media (max-width: 800px) {
    .gradio-container {
        padding: 3px 4px !important;
    }
    #sidebar_col {
        display: none !important;
    }
    #inspector_col {
        flex: 0 0 210px !important;
        width: 210px !important;
        max-width: 210px !important;
        min-width: 210px !important;
        height: calc(100vh - 58px) !important;
        min-height: calc(100vh - 58px) !important;
        max-height: calc(100vh - 58px) !important;
        padding: 4px 5px !important;
        gap: 2px !important;
        font-size: 11px !important;
    }
    #app_main_layout {
        height: calc(100vh - 58px) !important;
        min-height: calc(100vh - 58px) !important;
        max-height: calc(100vh - 58px) !important;
        gap: 4px !important;
    }
    #top_header_row {
        flex: 0 0 40px !important;
        height: 40px !important;
        min-height: 40px !important;
        max-height: 40px !important;
        padding: 4px 8px !important;
    }
    .brand-icon-box {
        width: 26px !important;
        height: 26px !important;
        font-size: 13px !important;
    }
    .brand-text-box h2 {
        font-size: 13px !important;
    }
    .brand-text-box p,
    .stats-row {
        display: none !important;
    }
    #btn_ai_process {
        font-size: 11px !important;
        padding: 0 10px !important;
    }
    #action_bar_row {
        flex: 0 0 30px !important;
        height: 30px !important;
        min-height: 30px !important;
        gap: 6px !important;
    }
    .bubble-title-box h3 {
        font-size: 12px !important;
    }
    .bubble-pills-radio .wrap {
        gap: 3px !important;
    }
    .bubble-pills-radio label {
        font-size: 9px !important;
        padding: 1px 5px !important;
        height: 18px !important;
        min-width: 22px !important;
    }
    #bubble_crop_box {
        flex: 0 0 50px !important;
        height: 50px !important;
        min-height: 50px !important;
        max-height: 50px !important;
    }
    #corrected_text_input textarea {
        height: 32px !important;
        min-height: 32px !important;
        max-height: 32px !important;
    }
    .coords-grid {
        grid-template-columns: repeat(4, 1fr) !important;
        gap: 2px !important;
    }
    .coords-grid input {
        height: 18px !important;
        font-size: 9px !important;
    }
    .dpad-btn {
        width: 18px !important;
        height: 18px !important;
        min-width: 18px !important;
        font-size: 8px !important;
    }
    .dpad-grid {
        grid-template-columns: repeat(3, 18px) !important;
    }
    .resize-btn {
        height: 16px !important;
        font-size: 8px !important;
    }
    #btn_save_bubble,
    #btn_re_ocr {
        height: 22px !important;
        min-height: 22px !important;
        font-size: 9px !important;
    }
    .dialogues-container {
        max-height: 70px !important;
    }
    .dialogue-card {
        font-size: 9px !important;
        padding: 2px 4px !important;
    }
    #visual_outputs_row {
        height: calc(100vh - 220px) !important;
        max-height: calc(100vh - 220px) !important;
    }
}

/* TARGETED SPINNER FILTER */
[data-testid="status-tracker"],
[data-testid="block-loading"],
.status-tracker,
.loading-overlay,
.loading-spinner,
.loading-text,
.load-wrap,
.loader,
.loading,
.generating,
.eta-bar,
.progress-bar,
.progress-bar-wrap,
.progress-level,
.progress-level-inner,
.progress-text,
.meta-text,
.meta-text-center,
.clear-status {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    height: 0 !important;
    width: 0 !important;
    min-height: 0 !important;
    max-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
}

.generating,
*:not(#annotated_img_box):not(#clean_img_box):not(#bubble_crop_box):not(#col_annotated):not(#col_clean).generating {
    border: none !important;
    box-shadow: none !important;
    animation: none !important;
}

.textarea-container.loading,
.dialogue-container.loading {
    opacity: 1 !important;
}

#annotated_img_box [data-testid="status-tracker"].hide,
#clean_img_box [data-testid="status-tracker"].hide,
#bubble_crop_box [data-testid="status-tracker"].hide,
#col_annotated [data-testid="status-tracker"].hide,
#col_clean [data-testid="status-tracker"].hide,
.status-tracker.hide {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    background: transparent !important;
    backdrop-filter: none !important;
    z-index: -1 !important;
}

#annotated_img_box [data-testid="status-tracker"]:not(.hide),
#clean_img_box [data-testid="status-tracker"]:not(.hide),
#bubble_crop_box [data-testid="status-tracker"]:not(.hide),
#col_annotated [data-testid="status-tracker"]:not(.hide),
#col_clean [data-testid="status-tracker"]:not(.hide),
#annotated_img_box .status-tracker:not(.hide),
#clean_img_box .status-tracker:not(.hide),
#bubble_crop_box .status-tracker:not(.hide),
#col_annotated .status-tracker:not(.hide),
#col_clean .status-tracker:not(.hide) {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 100% !important;
    max-height: 100% !important;
    background: rgba(10, 12, 19, 0.75) !important;
    backdrop-filter: blur(2px) !important;
    border-radius: 8px !important;
    z-index: 50 !important;
    align-items: center !important;
    justify-content: center !important;
    flex-direction: column !important;
    gap: 8px !important;
    margin: 0 !important;
    padding: 10px !important;
    box-sizing: border-box !important;
}

#annotated_img_box [data-testid="status-tracker"]:not(.hide) *,
#clean_img_box [data-testid="status-tracker"]:not(.hide) *,
#col_annotated [data-testid="status-tracker"]:not(.hide) *,
#col_clean [data-testid="status-tracker"]:not(.hide) * {
    visibility: visible !important;
}

#annotated_img_box [data-testid="status-tracker"]:not(.hide) .loading,
#clean_img_box [data-testid="status-tracker"]:not(.hide) .loading,
#col_annotated [data-testid="status-tracker"]:not(.hide) .loading,
#col_clean [data-testid="status-tracker"]:not(.hide) .loading,
#annotated_img_box [data-testid="status-tracker"]:not(.hide) .loader,
#clean_img_box [data-testid="status-tracker"]:not(.hide) .loader,
#annotated_img_box [data-testid="status-tracker"]:not(.hide) .loading-spinner,
#clean_img_box [data-testid="status-tracker"]:not(.hide) .loading-spinner {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    height: auto !important;
    width: auto !important;
}

#annotated_img_box [data-testid="status-tracker"]:not(.hide) svg,
#clean_img_box [data-testid="status-tracker"]:not(.hide) svg,
#col_annotated [data-testid="status-tracker"]:not(.hide) svg,
#col_clean [data-testid="status-tracker"]:not(.hide) svg {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    width: 32px !important;
    height: 32px !important;
}

#annotated_img_box [data-testid="status-tracker"]:not(.hide) .progress-bar-wrap,
#clean_img_box [data-testid="status-tracker"]:not(.hide) .progress-bar-wrap {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    width: 60% !important;
    height: 6px !important;
    background: #1e2436 !important;
    border-radius: 4px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    overflow: hidden !important;
}

#annotated_img_box [data-testid="status-tracker"]:not(.hide) .progress-bar,
#clean_img_box [data-testid="status-tracker"]:not(.hide) .progress-bar {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    height: 100% !important;
    background: linear-gradient(90deg, #6366f1, #ea580c) !important;
}

#annotated_img_box [data-testid="status-tracker"]:not(.hide) .meta-text,
#clean_img_box [data-testid="status-tracker"]:not(.hide) .meta-text,
#col_annotated [data-testid="status-tracker"]:not(.hide) .meta-text,
#col_clean [data-testid="status-tracker"]:not(.hide) .meta-text {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    height: auto !important;
    width: auto !important;
    font-size: 11px !important;
    color: #e2e8f0 !important;
    position: relative !important;
    font-family: monospace !important;
    background: rgba(0, 0, 0, 0.4) !important;
    padding: 2px 8px !important;
    border-radius: 4px !important;
}

#bubble_crop_box [data-testid="status-tracker"]:not(.hide) *,
#bubble_crop_box .loading-overlay:not(.hide) * {
    visibility: visible !important;
}
#bubble_crop_box [data-testid="status-tracker"]:not(.hide) .loading,
#bubble_crop_box [data-testid="status-tracker"]:not(.hide) .loader,
#bubble_crop_box .loading-spinner {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}
#bubble_crop_box [data-testid="status-tracker"]:not(.hide) svg {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    width: 20px !important;
    height: 20px !important;
}
#bubble_crop_box [data-testid="status-tracker"] .meta-text,
#bubble_crop_box [data-testid="status-tracker"] .progress-bar-wrap {
    display: none !important;
}

#top_header_row [data-testid="status-tracker"],
#sidebar_col [data-testid="status-tracker"],
#center_action_bar [data-testid="status-tracker"],
#action_bar_row [data-testid="status-tracker"],
#new_bubble_card [data-testid="status-tracker"],
.coords-grid [data-testid="status-tracker"],
.coords-grid * [data-testid="status-tracker"],
#corrected_text_input [data-testid="status-tracker"],
#corrected_text_input * [data-testid="status-tracker"],
.bubble-pills-radio [data-testid="status-tracker"],
.bubble-pills-radio * [data-testid="status-tracker"],
.dpad-container [data-testid="status-tracker"],
.dialogues-container [data-testid="status-tracker"],
#panel_trainer [data-testid="status-tracker"],
#panel_batch [data-testid="status-tracker"],
#status_indicator [data-testid="status-tracker"],
.coords-grid .loading-overlay,
#corrected_text_input .loading-overlay,
.bubble-pills-radio .loading-overlay {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    height: 0 !important;
    width: 0 !important;
}
"""

CUSTOM_HEAD = """
<meta name="google" content="notranslate">
<meta name="robots" content="notranslate">
<script>
document.addEventListener("DOMContentLoaded", function() {
    document.documentElement.classList.add("notranslate");
    document.documentElement.setAttribute("translate", "no");

    // 1. Dialogue card click selects corresponding bubble pill
    document.addEventListener("click", function(e) {
        var card = e.target.closest(".dialogue-card");
        if (card) {
            var bid = card.getAttribute("data-bid");
            if (bid) {
                var targetPill = "#" + bid;
                var radioLabels = document.querySelectorAll(".bubble-pills-radio label");
                radioLabels.forEach(function(lbl) {
                    if (lbl.textContent.trim() === targetPill) {
                        var inp = lbl.querySelector("input[type='radio']");
                        if (inp) {
                            inp.click();
                        } else {
                            lbl.click();
                        }
                    }
                });
            }
        }
    });

    // 2. Show clean single status next to AI button during processing
    document.addEventListener("click", function(e) {
        var btn = e.target.closest("#btn_ai_process, .sample-thumb-btn");
        if (btn) {
            var statusEl = document.getElementById("status_indicator");
            if (statusEl) {
                statusEl.innerHTML = "<span style='color: #fb923c; font-size: 12px; font-weight: 600;'>⏳ Tahlil qilinmoqda: EasyOCR matnlarni o\\'qimoqda va fon tozalanmoqda (~15-20 s kutilmoqda)...</span>";
            }
        }
    });

    // 3. Dynamic MutationObserver: Suppress any rogue status-trackers outside the 3 images
    function purgeRogueTrackers() {
        var trackers = document.querySelectorAll('[data-testid="status-tracker"], .status-tracker, .loading-overlay');
        trackers.forEach(function(el) {
            var allowed = el.closest('#annotated_img_box, #clean_img_box, #bubble_crop_box, #col_annotated, #col_clean, #panel_trainer, #panel_batch');
            if (!allowed) {
                el.style.setProperty('display', 'none', 'important');
                el.style.setProperty('visibility', 'hidden', 'important');
                el.style.setProperty('opacity', '0', 'important');
                el.style.setProperty('height', '0px', 'important');
                el.style.setProperty('pointer-events', 'none', 'important');
            }
        });
    }

    var observer = new MutationObserver(function() {
        purgeRogueTrackers();
    });
    observer.observe(document.body, { childList: true, subtree: true });
});
</script>
"""

# ==============================================================================
# GRADIO BLOCKS APPLICATION
# ==============================================================================
_blocks_kwargs = {"title": "AI Manga Studio"}
_blocks_sig = inspect.signature(gr.Blocks.__init__).parameters
if "css" in _blocks_sig:
    _blocks_kwargs["css"] = CUSTOM_CSS
if "head" in _blocks_sig:
    _blocks_kwargs["head"] = CUSTOM_HEAD

with gr.Blocks(**_blocks_kwargs) as demo:
    session_state = gr.State(value=get_initial_session)

    # 1. TOP HEADER BAR
    with gr.Row(elem_id="top_header_row"):
        gr.HTML("""
        <div class="brand-box">
            <div class="brand-icon-box">💬</div>
            <div class="brand-text-box">
                <h2>AI Manga Studio</h2>
                <p>Nutq bulutlarini toping, matnni öqing, fonni tozalang</p>
            </div>
        </div>
        """)
        stats_banner = gr.HTML(value=get_stats_text())

    # 2. MAIN 3-COLUMN STUDIO LAYOUT
    with gr.Row(elem_id="app_main_layout"):

        # ----------------------------------------------------
        # COLUMN 1: LEFT SIDEBAR
        # ----------------------------------------------------
        with gr.Column(elem_id="sidebar_col"):
            gr.Markdown("<p class='sidebar-heading'>İŞ MAYDONİ</p>")
            tab_btn_studio = gr.Button("🎯 Asosiy studio", elem_classes=["sidebar-nav-btn", "active-tab"])
            tab_btn_trainer = gr.Button("📊 Öz-özini öqitiş", elem_classes=["sidebar-nav-btn"])
            tab_btn_batch = gr.Button("📚 Bobni sinaş", elem_classes=["sidebar-nav-btn"])

            gr.Markdown("<p class='sidebar-heading' style='margin-top: 8px;'>NAMUNALAR</p>")
            samples_style_html = gr.HTML(value=generate_samples_css(), visible=True)
            with gr.Row(elem_classes=["samples-preview-row"]):
                btn_sample_1 = gr.Button("", elem_classes=["sample-thumb-btn", "sample-thumb-1"])
                btn_sample_2 = gr.Button("", elem_classes=["sample-thumb-btn", "sample-thumb-2"])
                btn_sample_3 = gr.Button("", elem_classes=["sample-thumb-btn", "sample-thumb-3"])

            auto_collect_chk = gr.Checkbox(
                value=False,
                label="Sahifani avtomatik örganiş bazasiga yiğiş",
                elem_id="chk_auto_collect"
            )
            source_language = gr.Dropdown(
                choices=[("Yaponcha", "ja"), ("Koreyscha", "ko"), ("Xitoycha", "zh"), ("Inglizcha", "en")],
                value="ja", label="Sahifadagi matn tili")

        # ----------------------------------------------------
        # COLUMN 2: CENTER WORKSPACE
        # ----------------------------------------------------
        with gr.Column(elem_id="main_workspace_col"):
            # PRIMARY STUDIO PANEL
            with gr.Column(visible=True, elem_id="panel_studio") as panel_studio:
                with gr.Row(elem_id="center_action_bar"):
                    with gr.Row(elem_classes=["view-toggle-group"]):
                        btn_view_both = gr.Button("Yonma-yon", size="sm", elem_classes=["view-pill-btn", "active-view"])
                        btn_view_annotated = gr.Button("Bulutlar", size="sm", elem_classes=["view-pill-btn"])
                        btn_view_clean = gr.Button("Tozalangan", size="sm", elem_classes=["view-pill-btn"])

                    page_info_label = gr.HTML(
                        value=format_page_info_html(INITIAL_SESSION.get("page_info", "sahifa_04.jpg · 1200 × 1700"))
                    )

                    btn_upload = gr.UploadButton(
                        "📁 Başqa sahifa",
                        file_types=["image"],
                        elem_classes=["upload-toggle-btn"]
                    )

                with gr.Row(elem_id="action_bar_row"):
                    process_btn = gr.Button("✨ AI bilan belgilaş va öqiş", elem_id="btn_ai_process")
                    status_output = gr.Markdown(
                        "🟢 **4 ta nutq buluti** aniqlandi. Tahrirlaş uçun öngdagi röyxatdan tanlang.",
                        elem_id="status_indicator"
                    )

                with gr.Row(elem_classes=["panel-headers-row"]):
                    with gr.Column() as subhead_annotated_col:
                        badge_red_banner = gr.HTML(value=format_red_badge_header(len(INITIAL_SESSION.get("dialogues", []))))
                    with gr.Column() as subhead_clean_col:
                        gr.HTML("""
                        <div class="panel-subhead">
                            <span class="panel-title">2 · Tozalangan fon</span>
                            <span class="badge-tag-green">Inpainting tayyor</span>
                        </div>
                        """)

                with gr.Row(elem_id="visual_outputs_row"):
                    with gr.Column(elem_id="col_annotated") as col_annotated:
                        annotated_output = make_clean_image_component(
                            value=INITIAL_SESSION.get("annotated_img"),
                            show_label=False,
                            elem_id="annotated_img_box",
                            interactive=False
                        )
                    with gr.Column(elem_id="col_clean") as col_clean:
                        clean_output = make_clean_image_component(
                            value=INITIAL_SESSION.get("cleaned_img"),
                            show_label=False,
                            elem_id="clean_img_box",
                            interactive=False
                        )

                with gr.Group(elem_id="new_bubble_card"):
                    with gr.Row(elem_classes=["new-bubble-header"]):
                        gr.HTML("""
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 15px;">🎯</span>
                            <div>
                                <span class="new-bubble-title">Yangi bulut qöşiş</span>
                                <span class="new-bubble-subtitle"> — Çap rasmda qarama-qarşi ikki burçakni bosing.</span>
                            </div>
                        </div>
                        """)
                        click_status_msg = gr.HTML(value=format_click_status(None))
                        with gr.Row():
                            btn_add_clicked_bubble = gr.Button("➕ Bulut yaratış", size="sm")
                            btn_reset_clicks = gr.Button("Tozalaş", size="sm")

            # SECONDARY PANEL: MODEL TRAINER
            with gr.Column(visible=False, elem_id="panel_trainer") as panel_trainer:
                gr.Markdown("### ⚙️ Doimiy Örganiş va Modelni Yangilaş")
                gr.Markdown(get_training_data_description())
                with gr.Row():
                    with gr.Column(scale=1):
                        thresh_slider = gr.Slider(
                            minimum=1, maximum=20, value=collector.load_stats().get("auto_train_threshold", 5), step=1,
                            label="Avto-Öqitiş çegarasi"
                        )
                        thresh_btn = gr.Button("Çegarani Saqlaş")
                        thresh_msg = gr.Markdown("")
                        epochs_slider = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Öqitiş sikli (Epoxalar)")
                        fine_tune_btn = gr.Button("🚀 AI Modellarni Qayta Öqitiş", variant="primary", size="lg")
                        reload_btn = gr.Button("🔄 Modellarni Xotirada Yangilaş (Reload)")
                        reload_msg = gr.Markdown("")
                    with gr.Column(scale=2):
                        train_log_output = gr.Textbox(label="Öqitiş Konsol Loglari", lines=12, placeholder="Loglar bu yerda körsatiladi...")
                        train_status_banner = gr.Markdown("")

            # SECONDARY PANEL: BATCH CHAPTER TESTING
            with gr.Column(visible=False, elem_id="panel_batch") as panel_batch:
                gr.Markdown("### 📂 Manga Bob Sahifalarini Sinaş")
                gr.Markdown("Bir neçta sahifani birdaniga tahlil qiliş.")
                with gr.Row():
                    with gr.Column(scale=1):
                        chapter_path_in = gr.Textbox(value=os.path.join(BASE_DIR, "test_chapter_57"), label="Bob Papkasi Manzili")
                        batch_pages_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Sahifalar soni")
                        batch_learn_btn = gr.Button("⚡ Bob Sahifalarini Tahlil Qiliş", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        batch_log_output = gr.Textbox(label="Jarayon Natijalari", lines=10, placeholder="Natijalar bu yerda körsatiladi...")

        # ----------------------------------------------------
        # COLUMN 3: RIGHT INSPECTOR
        # ----------------------------------------------------
        with gr.Column(elem_id="inspector_col") as panel_inspector:
            with gr.Row(elem_classes=["inspector-top-row"]):
                bubble_id_label = gr.HTML(
                    value=f"<div class='bubble-title-box'><h3>Bulut #2</h3><span class='dim-badge'>{_init_b2_w} × {_init_b2_h} px</span></div>"
                )
                with gr.Row(elem_classes=["inspector-icon-group"]):
                    btn_prev_bubble = gr.Button("◀", elem_classes=["icon-nav-btn"])
                    btn_next_bubble = gr.Button("▶", elem_classes=["icon-nav-btn"])
                    delete_bubble_btn = gr.Button("🗑️", elem_classes=["icon-del-btn"])

            _init_choices = get_bubble_pill_choices(INITIAL_SESSION.get("dialogues", []))
            _init_val = "#2" if "#2" in _init_choices else (_init_choices[0] if _init_choices else "#1")
            bubble_pills = gr.Radio(
                choices=_init_choices,
                value=_init_val,
                elem_classes=["bubble-pills-radio"],
                show_label=False
            )

            bubble_crop_preview = make_clean_image_component(
                value=INITIAL_SESSION.get("crops", {}).get(2),
                show_label=False,
                elem_id="bubble_crop_box",
                interactive=False
            )

            gr.HTML("<p class='nutq-matni-lbl'>Nutq matni (OCR natijasi, tahrirlaş mumkin)</p>")
            corrected_text_input = gr.Textbox(
                value="なんでもないよ、ただ少し考えていただけ。",
                show_label=False,
                elem_id="corrected_text_input",
                lines=2,
                container=False
            )
            with gr.Row(elem_classes=["text-actions-row"]):
                save_bubble_btn = gr.Button("💾 Saqlaş va örgatiş", elem_id="btn_save_bubble")
                btn_re_ocr = gr.Button("🔄 Qayta OCR", elem_id="btn_re_ocr")

            gr.HTML("<p class='nutq-matni-lbl' style='margin-top: 4px;'>Çegara koordinatalari</p>")
            with gr.Row(elem_classes=["coords-grid"]):
                bbox_x1 = gr.Number(value=_init_b2_box[0], label="X1 - çap", precision=0, container=False)
                bbox_y1 = gr.Number(value=_init_b2_box[1], label="Y1 - tepa", precision=0, container=False)
                bbox_x2 = gr.Number(value=_init_b2_box[2], label="X2 - öng", precision=0, container=False)
                bbox_y2 = gr.Number(value=_init_b2_box[3], label="Y2 - past", precision=0, container=False)

            with gr.Row(elem_classes=["dpad-container"]):
                with gr.Column(elem_classes=["dpad-grid"]):
                    gr.HTML("<div style='width:22px; height:22px;'></div>")
                    btn_nudge_up = gr.Button("▲", elem_classes=["dpad-btn"])
                    gr.HTML("<div style='width:22px; height:22px;'></div>")
                    btn_nudge_left = gr.Button("◀", elem_classes=["dpad-btn"])
                    btn_center = gr.Button("●", elem_classes=["dpad-btn", "dpad-btn-center"])
                    btn_nudge_right = gr.Button("▶", elem_classes=["dpad-btn"])
                    gr.HTML("<div style='width:22px; height:22px;'></div>")
                    btn_nudge_down = gr.Button("▼", elem_classes=["dpad-btn"])
                    gr.HTML("<div style='width:22px; height:22px;'></div>")

                with gr.Column(elem_classes=["dpad-right-actions"]):
                    btn_expand = gr.Button("➕ Kengaytiriş", elem_classes=["resize-btn"])
                    btn_shrink = gr.Button("➖ Qisqartiriş", elem_classes=["resize-btn"])
                    step_choice = gr.Radio(choices=[5, 10, 20], value=10, label="Qadam", elem_classes=["step-radio"], container=False)

            btn_apply_bbox = gr.Button("Yangi çegarani qöllaş", elem_id="btn_apply_bbox_full")

            gr.HTML("<p class='nutq-matni-lbl' style='margin-top: 6px;'>Sahifadagi barça matnlar</p>")
            all_dialogues_view = gr.HTML(value=format_all_dialogues_summary(INITIAL_SESSION, active_bid=2))

    # ----------------------------------------------------
    # EVENT BINDINGS
    # ----------------------------------------------------
    source_language.change(
        fn=lambda lang, state: {**(state or {}), "source_lang": lang},
        inputs=[source_language, session_state], outputs=[session_state], show_progress="hidden")
    process_btn.click(
        fn=lambda chk, s: process_and_inspect_manga(None, chk, s),
        inputs=[auto_collect_chk, session_state],
        outputs=[
            annotated_output, clean_output,
            bubble_pills,
            bubble_crop_preview, corrected_text_input, bubble_id_label,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            all_dialogues_view, status_output, stats_banner,
            page_info_label, badge_red_banner,
            session_state
        ],
        show_progress="full",
        concurrency_limit=2
    )

    btn_upload.upload(
        fn=handle_upload_action,
        inputs=[btn_upload, auto_collect_chk, session_state],
        outputs=[
            annotated_output, clean_output,
            bubble_pills,
            bubble_crop_preview, corrected_text_input, bubble_id_label,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            all_dialogues_view, status_output, stats_banner,
            page_info_label, badge_red_banner,
            samples_style_html,
            session_state
        ],
        show_progress="full",
        concurrency_limit=2
    )

    btn_sample_1.click(
        fn=lambda chk, s: load_sample_page(0, chk, s),
        inputs=[auto_collect_chk, session_state],
        outputs=[
            annotated_output, clean_output,
            bubble_pills,
            bubble_crop_preview, corrected_text_input, bubble_id_label,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            all_dialogues_view, status_output, stats_banner,
            page_info_label, badge_red_banner,
            session_state
        ],
        show_progress="full",
        concurrency_limit=5
    )
    btn_sample_2.click(
        fn=lambda chk, s: load_sample_page(1, chk, s),
        inputs=[auto_collect_chk, session_state],
        outputs=[
            annotated_output, clean_output,
            bubble_pills,
            bubble_crop_preview, corrected_text_input, bubble_id_label,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            all_dialogues_view, status_output, stats_banner,
            page_info_label, badge_red_banner,
            session_state
        ],
        show_progress="full",
        concurrency_limit=5
    )
    btn_sample_3.click(
        fn=lambda chk, s: load_sample_page(2, chk, s),
        inputs=[auto_collect_chk, session_state],
        outputs=[
            annotated_output, clean_output,
            bubble_pills,
            bubble_crop_preview, corrected_text_input, bubble_id_label,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            all_dialogues_view, status_output, stats_banner,
            page_info_label, badge_red_banner,
            session_state
        ],
        show_progress="full",
        concurrency_limit=5
    )

    # 2-Click visual bounding box creation on image (does not corrupt active bubble coordinates)
    annotated_output.select(
        fn=on_annotated_image_click,
        inputs=[session_state],
        outputs=[click_status_msg, session_state],
        show_progress="hidden",
        concurrency_limit=4
    )
    btn_reset_clicks.click(
        fn=reset_click_points,
        inputs=[session_state],
        outputs=[click_status_msg, session_state],
        show_progress="hidden",
        concurrency_limit=4
    )
    btn_add_clicked_bubble.click(
        fn=add_from_clicked_coords,
        inputs=[bbox_x1, bbox_y1, bbox_x2, bbox_y2, session_state],
        outputs=[
            annotated_output, clean_output,
            bubble_pills,
            bubble_crop_preview, corrected_text_input, bubble_id_label,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            all_dialogues_view, click_status_msg, status_output, stats_banner,
            badge_red_banner, session_state
        ],
        show_progress="hidden",
        concurrency_limit=2
    )

    # Bubble Pills Selection & Navigation
    bubble_pills.change(
        fn=on_select_bubble,
        inputs=[bubble_pills, session_state],
        outputs=[
            annotated_output,
            bubble_crop_preview, corrected_text_input, bubble_id_label,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            all_dialogues_view
        ],
        show_progress="hidden",
        concurrency_limit=10
    )
    btn_prev_bubble.click(
        fn=lambda cur, s: nav_bubble(cur, -1, s),
        inputs=[bubble_pills, session_state],
        outputs=[
            annotated_output,
            bubble_pills,
            bubble_crop_preview, corrected_text_input, bubble_id_label,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            all_dialogues_view
        ],
        show_progress="hidden",
        concurrency_limit=10
    )
    btn_next_bubble.click(
        fn=lambda cur, s: nav_bubble(cur, 1, s),
        inputs=[bubble_pills, session_state],
        outputs=[
            annotated_output,
            bubble_pills,
            bubble_crop_preview, corrected_text_input, bubble_id_label,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            all_dialogues_view
        ],
        show_progress="hidden",
        concurrency_limit=10
    )

    # Text Save & Fine-tune trigger
    save_bubble_btn.click(
        fn=save_single_bubble_correction,
        inputs=[bubble_pills, corrected_text_input, session_state],
        outputs=[
            annotated_output, bubble_pills, all_dialogues_view,
            corrected_text_input, status_output, stats_banner,
            session_state
        ],
        show_progress="hidden",
        concurrency_limit=5
    )
    btn_re_ocr.click(
        fn=rerun_ocr_for_bubble,
        inputs=[bubble_pills, session_state],
        outputs=[
            bubble_crop_preview,
            corrected_text_input,
            bubble_pills, all_dialogues_view,
            status_output, session_state
        ],
        show_progress="full",
        concurrency_limit=5
    )

    # Delete Bubble
    delete_bubble_btn.click(
        fn=delete_selected_bubble,
        inputs=[bubble_pills, session_state],
        outputs=[
            annotated_output, clean_output,
            bubble_crop_preview, corrected_text_input, bubble_id_label,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            bubble_pills, all_dialogues_view, status_output,
            badge_red_banner, session_state
        ],
        show_progress="hidden",
        concurrency_limit=5
    )

    # Coordinate edits & D-Pad (including center button)
    for _bbox_trigger in (btn_apply_bbox, btn_center):
        _bbox_trigger.click(
            fn=update_bubble_boundary,
            inputs=[bubble_pills, bbox_x1, bbox_y1, bbox_x2, bbox_y2, session_state],
            outputs=[
                annotated_output, clean_output,
                bubble_crop_preview, bubble_id_label,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                all_dialogues_view, status_output, session_state
            ],
            show_progress="hidden",
            concurrency_limit=5
        )

    btn_nudge_left.click(
        fn=lambda choice, step, s: adjust_bubble_delta(choice, "left", step, s),
        inputs=[bubble_pills, step_choice, session_state],
        outputs=[
            annotated_output, clean_output,
            bubble_crop_preview, bubble_id_label,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            all_dialogues_view, status_output, session_state
        ],
        show_progress="hidden",
        concurrency_limit=5
    )
    btn_nudge_right.click(
        fn=lambda choice, step, s: adjust_bubble_delta(choice, "right", step, s),
        inputs=[bubble_pills, step_choice, session_state],
        outputs=[
            annotated_output, clean_output,
            bubble_crop_preview, bubble_id_label,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            all_dialogues_view, status_output, session_state
        ],
        show_progress="hidden",
        concurrency_limit=5
    )
    btn_nudge_up.click(
        fn=lambda choice, step, s: adjust_bubble_delta(choice, "up", step, s),
        inputs=[bubble_pills, step_choice, session_state],
        outputs=[
            annotated_output, clean_output,
            bubble_crop_preview, bubble_id_label,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            all_dialogues_view, status_output, session_state
        ],
        show_progress="hidden",
        concurrency_limit=5
    )
    btn_nudge_down.click(
        fn=lambda choice, step, s: adjust_bubble_delta(choice, "down", step, s),
        inputs=[bubble_pills, step_choice, session_state],
        outputs=[
            annotated_output, clean_output,
            bubble_crop_preview, bubble_id_label,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            all_dialogues_view, status_output, session_state
        ],
        show_progress="hidden",
        concurrency_limit=5
    )
    btn_expand.click(
        fn=lambda choice, step, s: adjust_bubble_delta(choice, "expand", step, s),
        inputs=[bubble_pills, step_choice, session_state],
        outputs=[
            annotated_output, clean_output,
            bubble_crop_preview, bubble_id_label,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            all_dialogues_view, status_output, session_state
        ],
        show_progress="hidden",
        concurrency_limit=5
    )
    btn_shrink.click(
        fn=lambda choice, step, s: adjust_bubble_delta(choice, "shrink", step, s),
        inputs=[bubble_pills, step_choice, session_state],
        outputs=[
            annotated_output, clean_output,
            bubble_crop_preview, bubble_id_label,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            all_dialogues_view, status_output, session_state
        ],
        show_progress="hidden",
        concurrency_limit=5
    )

    # View Mode Toggles ("Yonma-yon", "Bulutlar", "Tozalangan")
    btn_view_both.click(
        fn=lambda: (
            gr.update(visible=True), gr.update(visible=True),
            gr.update(visible=True), gr.update(visible=True),
            gr.update(elem_classes=["view-pill-btn", "active-view"]),
            gr.update(elem_classes=["view-pill-btn"]),
            gr.update(elem_classes=["view-pill-btn"])
        ),
        outputs=[
            col_annotated, col_clean,
            subhead_annotated_col, subhead_clean_col,
            btn_view_both, btn_view_annotated, btn_view_clean
        ],
        show_progress="hidden"
    )
    btn_view_annotated.click(
        fn=lambda: (
            gr.update(visible=True), gr.update(visible=False),
            gr.update(visible=True), gr.update(visible=False),
            gr.update(elem_classes=["view-pill-btn"]),
            gr.update(elem_classes=["view-pill-btn", "active-view"]),
            gr.update(elem_classes=["view-pill-btn"])
        ),
        outputs=[
            col_annotated, col_clean,
            subhead_annotated_col, subhead_clean_col,
            btn_view_both, btn_view_annotated, btn_view_clean
        ],
        show_progress="hidden"
    )
    btn_view_clean.click(
        fn=lambda: (
            gr.update(visible=False), gr.update(visible=True),
            gr.update(visible=False), gr.update(visible=True),
            gr.update(elem_classes=["view-pill-btn"]),
            gr.update(elem_classes=["view-pill-btn"]),
            gr.update(elem_classes=["view-pill-btn", "active-view"])
        ),
        outputs=[
            col_annotated, col_clean,
            subhead_annotated_col, subhead_clean_col,
            btn_view_both, btn_view_annotated, btn_view_clean
        ],
        show_progress="hidden"
    )

    # Trainer & Batch actions
    thresh_btn.click(
        fn=update_threshold,
        inputs=[thresh_slider],
        outputs=[thresh_msg, stats_banner],
        show_progress="hidden",
        concurrency_limit=5
    )
    fine_tune_btn.click(
        fn=trigger_manual_fine_tune,
        inputs=[epochs_slider],
        outputs=[train_log_output, train_status_banner, stats_banner],
        show_progress="full",
        concurrency_limit=1
    )
    reload_btn.click(
        fn=hot_reload_models_action,
        outputs=[reload_msg, stats_banner],
        show_progress="minimal",
        concurrency_limit=1
    )
    batch_learn_btn.click(
        fn=batch_learn_chapter,
        inputs=[batch_pages_slider, chapter_path_in, source_language],
        outputs=[batch_log_output, stats_banner],
        show_progress="minimal",
        concurrency_limit=1
    )

    # Sidebar Navigation between Studio, Trainer, and Batch
    def nav_to_studio():
        return (
            gr.update(elem_classes=["sidebar-nav-btn", "active-tab"]),
            gr.update(elem_classes=["sidebar-nav-btn"]),
            gr.update(elem_classes=["sidebar-nav-btn"]),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False)
        )

    def nav_to_trainer():
        return (
            gr.update(elem_classes=["sidebar-nav-btn"]),
            gr.update(elem_classes=["sidebar-nav-btn", "active-tab"]),
            gr.update(elem_classes=["sidebar-nav-btn"]),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False)
        )

    def nav_to_batch():
        return (
            gr.update(elem_classes=["sidebar-nav-btn"]),
            gr.update(elem_classes=["sidebar-nav-btn"]),
            gr.update(elem_classes=["sidebar-nav-btn", "active-tab"]),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True)
        )

    tab_btn_studio.click(
        fn=nav_to_studio,
        inputs=[],
        outputs=[tab_btn_studio, tab_btn_trainer, tab_btn_batch, panel_studio, panel_inspector, panel_trainer, panel_batch],
        show_progress="hidden",
        concurrency_limit=10
    )
    tab_btn_trainer.click(
        fn=nav_to_trainer,
        inputs=[],
        outputs=[tab_btn_studio, tab_btn_trainer, tab_btn_batch, panel_studio, panel_inspector, panel_trainer, panel_batch],
        show_progress="hidden",
        concurrency_limit=10
    )
    tab_btn_batch.click(
        fn=nav_to_batch,
        inputs=[],
        outputs=[tab_btn_studio, tab_btn_trainer, tab_btn_batch, panel_studio, panel_inspector, panel_trainer, panel_batch],
        show_progress="hidden",
        concurrency_limit=10
    )

demo.queue(default_concurrency_limit=2, max_size=40)

if __name__ == "__main__":
    port = 7860
    _launch_kwargs = {
        "server_name": "127.0.0.1",
        "share": False,
        "inbrowser": True,
        "show_error": True,
        "max_threads": 40
    }
    _launch_sig = inspect.signature(demo.launch).parameters
    if "css" in _launch_sig:
        _launch_kwargs["css"] = CUSTOM_CSS
    if "head" in _launch_sig:
        _launch_kwargs["head"] = CUSTOM_HEAD

    try:
        logger.info("AI Manga Studio işga tuşirilmoqda: http://127.0.0.1:%d", port)
        demo.launch(server_port=port, **_launch_kwargs)
    except OSError:
        logger.warning("Port %d band, avtomatik ravişda keyingi oçiq port tanlanmoqda...", port)
        demo.launch(**_launch_kwargs)
