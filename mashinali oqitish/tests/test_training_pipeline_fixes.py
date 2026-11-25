import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

import ml_paths
from dataset_loader import (
    MangaDetectionDataset,
    MangaInpaintingDataset,
    MangaOCRDataset,
    normalize_detection_category,
    parse_annotation_bbox,
    prepare_ocr_image,
)
import extract_datasets
import validation_gate
from validation_gate import ValidationGate, ValidationUnavailable
import export_to_onnx
import train_and_evaluate
from tools import build_manga_ocr_dataset, build_manga_inpaint_dataset


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def test_ml_paths_resolves_server_ml_and_vocab():
    assert Path(ml_paths.SERVER_ML_DIR).is_dir()
    vocab = Path(ml_paths.resolve_vocab_path())
    assert vocab.is_file()
    data = json.loads(vocab.read_text(encoding="utf-8"))
    assert "<blank>" in data["char_to_idx"]


def test_extract_datasets_missing_dir_aborts_without_wiping_existing(tmp_path):
    out_dir = tmp_path / "manga_ml_datasets"
    ocr_dir = out_dir / "ocr"
    ocr_dir.mkdir(parents=True)
    existing = ocr_dir / "full.jsonl"
    existing.write_text('{"crop_id": "keep", "text": "HELLO"}\n', encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        extract_datasets.process_all_datasets(
            extracted_dir=str(tmp_path / "missing_extracted_data"),
            data_out_dir=str(out_dir),
            vocab_out_path=str(tmp_path / "ocr_vocab.json"),
        )
    assert existing.read_text(encoding="utf-8") == '{"crop_id": "keep", "text": "HELLO"}\n'


def test_extract_json_objects_handles_nested_and_multiline(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text(
        'header noise\n'
        '{\n  "page_id": "p1",\n  "annotations": [{"category": "speech_bubble", "bbox": [1, 2, 3, 4]}]\n}\n'
        '{"crop_id": "c1", "text": "HI", "meta": {"nested": {"ok": true}}}\n',
        encoding="utf-8",
    )
    objs = extract_datasets.extract_json_objects_from_file(str(sample))
    assert len(objs) == 2
    assert objs[0]["page_id"] == "p1"
    assert objs[1]["meta"]["nested"]["ok"] is True


def test_validation_gate_resolves_relative_train_paths_and_inpaint_triplets(tmp_path, monkeypatch):
    # Run from a different CWD to catch relative-path resolution bugs on training_image_paths
    other_cwd = tmp_path / "other_cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    base = tmp_path / "project"
    (base / "train_imgs").mkdir(parents=True)
    (base / "val_imgs").mkdir(parents=True)

    raw_train = base / "train_imgs" / "raw.png"
    mask_train = base / "train_imgs" / "mask.png"
    gt_train = base / "train_imgs" / "gt.png"
    Image.new("RGB", (64, 64), (20, 20, 20)).save(raw_train)
    mask_img = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(mask_img).rectangle([16, 16, 48, 48], fill=255)
    mask_img.save(mask_train)
    Image.new("RGB", (64, 64), (25, 25, 25)).save(gt_train)

    # Validation item reuses training ground_truth with a fresh raw image -> must still be rejected!
    raw_val = base / "val_imgs" / "raw_val.png"
    Image.new("RGB", (64, 64), (30, 30, 30)).save(raw_val)
    _write_jsonl(base / "validation" / "inpainting.jsonl", [{
        "verified": True,
        "split": "val",
        "source_type": "real_manga_page",
        "raw_image": "val_imgs/raw_val.png",
        "mask": "train_imgs/mask.png",
        "ground_truth": "train_imgs/gt.png",
        "ground_truth_source": "human_cleaned",
    }])

    train_ds = SimpleNamespace(samples=[{
        "raw_image": "train_imgs/raw.png",
        "mask": "train_imgs/mask.png",
        "ground_truth": "train_imgs/gt.png",
    }])

    with pytest.raises(ValidationUnavailable, match="overlap"):
        ValidationGate(base, "inpainting", {}, train_ds, min_samples=1)


def test_dataset_loader_bbox_xyxy_vs_yxyx_aliases_and_nearest_mask(tmp_path):
    # 1. Polygon-based xyxy vs yxyx disambiguation & string coordinate parsing
    ann_xyxy = {
        "category": "text_line",
        "bbox": ["120", 150, 400, 300],
        "polygon": [[120, 150], [400, 150], [400, 300], [120, 300]],
    }
    x1, y1, x2, y2, is_norm = parse_annotation_bbox(ann_xyxy, 800, 1200)
    assert (x1, y1, x2, y2, is_norm) == (120.0, 150.0, 400.0, 300.0, False)

    ann_yxyx = {
        "category": "text_bubble",
        "bbox": [200, 280, 440, 640],
        "polygon": [[280, 200], [640, 200], [640, 440], [280, 440]],
    }
    x1, y1, x2, y2, is_norm = parse_annotation_bbox(ann_yxyx, 1080, 1920)
    assert (x1, y1, x2, y2, is_norm) == (280.0, 200.0, 640.0, 440.0, False)

    # Integer [0, 0, 1, 1] is 1px box, NOT full-page normalized [0..1]
    _, _, _, _, norm_int = parse_annotation_bbox({"bbox": [0, 0, 1, 1], "bbox_format": "xyxy"}, 800, 1200)
    assert norm_int is False
    _, _, _, _, norm_flt = parse_annotation_bbox({"bbox": [0.1, 0.2, 0.5, 0.6], "bbox_format": "xyxy"}, 800, 1200)
    assert norm_flt is True

    assert normalize_detection_category("text_line") == "speech_bubble"
    assert normalize_detection_category("text_bubble") == "speech_bubble"

    # 2. Aspect-preserving OCR preparation & multiple-of-8 width
    tall = Image.new("RGB", (32, 128), (255, 255, 255))
    prep = prepare_ocr_image(tall, target_h=64, target_w=250)
    assert prep.size == (256, 64)

    # 3. NEAREST mask resizing preserves sharp 1px strokes better than bilinear
    raw_p = tmp_path / "raw.png"
    mask_p = tmp_path / "mask.png"
    clean_p = tmp_path / "clean.png"
    Image.new("RGB", (512, 512), (240, 240, 240)).save(raw_p)
    Image.new("RGB", (512, 512), (240, 240, 240)).save(clean_p)
    m = Image.new("L", (512, 512), 0)
    ImageDraw.Draw(m).line([(100, 100), (400, 100)], fill=255, width=2)
    m.save(mask_p)
    inp_jsonl = _write_jsonl(tmp_path / "inp.jsonl", [{
        "triplet_id": "t1",
        "raw_image": str(raw_p),
        "mask": str(mask_p),
        "ground_truth": str(clean_p),
    }])
    ds = MangaInpaintingDataset(str(inp_jsonl))
    sample = ds[0]
    assert sample["mask"].shape == (1, 256, 256)
    assert sample["mask"].sum().item() > 0


def test_build_ocr_dataset_glyph_coverage_vertical_and_grouped_split():
    # Missing-glyph text returns None instead of rendering tofu boxes
    assert build_manga_ocr_dataset.get_font_for_text("\U0010ffff") is None
    assert build_manga_ocr_dataset.render_synthetic_crop("HELLO\U0010ffff") is None

    # Horizontal & vertical crops render cleanly within (256, 64)
    horiz = build_manga_ocr_dataset.render_synthetic_crop("THIS IS A VERY LONG DIALOGUE SENTENCE!", direction="horizontal")
    assert horiz is not None and horiz.size == (256, 64)
    vert = build_manga_ocr_dataset.render_synthetic_crop("どうしたの？", direction="vertical")
    assert vert is not None and vert.size == (256, 64)

    # Grouped split prevents source-page and phrase leakage
    records = [
        {"crop_id": "1", "text": "HELLO", "source_page": "p1.png"},
        {"crop_id": "2", "text": "HELLO", "source_page": "p2.png"},
        {"crop_id": "3", "text": "WORLD", "source_page": "p2.png"},
        {"crop_id": "4", "text": "BYE", "source_page": "p3.png"},
        {"crop_id": "5", "text": "NO", "source_page": "p4.png"},
        {"crop_id": "6", "text": "YES", "source_page": "p5.png"},
    ]
    tr, va, te = build_manga_ocr_dataset.partition_records_grouped(records, seed=42)
    tr_texts = {r["text"] for r in tr}
    va_texts = {r["text"] for r in va}
    te_texts = {r["text"] for r in te}
    assert tr_texts.isdisjoint(va_texts) and tr_texts.isdisjoint(te_texts) and va_texts.isdisjoint(te_texts)
    tr_pages = {r["source_page"] for r in tr}
    va_pages = {r["source_page"] for r in va}
    te_pages = {r["source_page"] for r in te}
    assert tr_pages.isdisjoint(va_pages) and tr_pages.isdisjoint(te_pages) and va_pages.isdisjoint(te_pages)


def test_build_inpaint_dataset_diverse_styles_and_grouped_split():
    dark_raw, _, dark_clean, _, dark_cat, style = build_manga_inpaint_dataset.make_bubble_triplet("t_dark", style="dark")
    assert style == "dark" and dark_cat == "narration_box"
    assert np.median(np.array(dark_clean)) < 120

    cream_raw, _, cream_clean, _, _, c_style = build_manga_inpaint_dataset.make_bubble_triplet("t_cream", style="cream")
    assert c_style == "cream"
    assert cream_raw.size == (256, 256)

    recs = [
        {"triplet_id": f"t{i}", "source_group": f"page_{i // 3}.png"}
        for i in range(15)
    ]
    tr, va, te = build_manga_inpaint_dataset.partition_triplets_by_source_page(recs, seed=42)
    g_tr = {r["source_group"] for r in tr}
    g_va = {r["source_group"] for r in va}
    g_te = {r["source_group"] for r in te}
    assert g_tr.isdisjoint(g_va) and g_tr.isdisjoint(g_te) and g_va.isdisjoint(g_te)


def test_train_ocr_trains_fc_and_validation_gate_blocks_overwrite(tmp_path):
    vocab_path = tmp_path / "ocr_vocab.json"
    char_to_idx = {"<blank>": 0, "<unk>": 1, "A": 2, "B": 3}
    idx_to_char = {"0": "<blank>", "1": "<unk>", "2": "A", "3": "B"}
    vocab_path.write_text(json.dumps({"char_to_idx": char_to_idx, "idx_to_char": idx_to_char}), encoding="utf-8")

    crops_dir = tmp_path / "data" / "ocr" / "crops"
    crops_dir.mkdir(parents=True)
    rows = []
    for i, txt in enumerate(["A", "B", "AB"]):
        p = crops_dir / f"c_{i}.png"
        Image.new("RGB", (64, 64), (255, 255, 255)).save(p)
        rows.append({"crop_id": f"c_{i}", "image_path": str(p), "text": txt})
    _write_jsonl(tmp_path / "data" / "ocr" / "train.jsonl", rows)
    _write_jsonl(tmp_path / "data" / "ocr" / "val.jsonl", rows)

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    incumbent_ckpt = ckpt_dir / "best_manga_ocr.pt"
    sentinel_model = train_and_evaluate.MangaOCRModel(vocab_size=4, hidden_dim=128)
    torch.save(sentinel_model.state_dict(), incumbent_ckpt)
    before_bytes = incumbent_ckpt.read_bytes()

    model, hist = train_and_evaluate.train_ocr_model(
        epochs=1,
        batch_size=3,
        max_batches=1,
        from_scratch=True,
        base_dir=str(tmp_path),
        data_dir=str(tmp_path / "data"),
        checkpoint_dir=str(ckpt_dir),
        vocab_file=str(vocab_path),
    )
    # Without validation/ocr.jsonl, incumbent checkpoint MUST NOT be overwritten
    assert incumbent_ckpt.read_bytes() == before_bytes
    assert hist["promotion"]["promoted"] is False
    # Verify fc weights changed relative to fresh init when single-char samples existed
    fresh = train_and_evaluate.MangaOCRModel(vocab_size=4, hidden_dim=128)
    assert not torch.equal(model.fc[4].weight.detach().cpu(), fresh.fc[4].weight.detach().cpu())


def test_export_inpainter_onnx_reflect_and_multichannel_mask(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    model = export_to_onnx.MangaInpaintingModel()
    torch.save(model.state_dict(), ckpt_dir / "best_manga_inpaint.pt")

    out_dir = tmp_path / "onnx_models"
    onnx_path = export_to_onnx.export_inpainter(output_dir=str(out_dir), checkpoint_dir=str(ckpt_dir))
    assert onnx_path and Path(onnx_path).is_file()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    # Feed non-multiple-of-8 image and 3-channel mask to verify mask[:, :1] wasn't traced away
    img = np.random.rand(1, 3, 125, 137).astype(np.float32)
    mask3 = np.ones((1, 3, 125, 137), dtype=np.float32)
    infilled, comp = sess.run(None, {"image": img, "mask": mask3})
    assert infilled.shape == (1, 3, 125, 137)
    assert comp.shape == (1, 3, 125, 137)


def test_app_guards_deferred_session_and_cleanup(monkeypatch, tmp_path):
    if "gradio" not in sys.modules:
        monkeypatch.setitem(sys.modules, "gradio", MagicMock())
    import app

    # Importing app must NOT eagerly build INITIAL_SESSION synchronously
    assert app.INITIAL_SESSION is None
    app_src = Path(app.__file__).read_text(encoding="utf-8")
    assert '("Ruscha", "ru")' in app_src
    assert '("Avto (Barcha tillar)", "auto")' in app_src

    # Collector / trainer None guards must not raise
    stats_html = app.get_stats_text()
    assert "Sahifalar" in stats_html
    msg, _ = app.update_threshold(7)
    assert "kollektori" in msg.lower() or "faol emas" in msg.lower()
    outputs = list(app.trigger_manual_fine_tune(1))
    assert outputs and "train_and_evaluate.py" in outputs[0][1]

    # Orphaned sample cleanup removes evicted sample_* files in recent_samples/
    rec_dir = tmp_path / "recent_samples"
    rec_dir.mkdir()
    monkeypatch.setattr(app, "RECENT_SAMPLES_DIR", str(rec_dir))
    monkeypatch.setattr(app, "RECENT_SAMPLES_META", str(rec_dir / "samples_meta.json"))
    orphan = rec_dir / "sample_old_orphan.jpg"
    orphan.write_bytes(b"old")
    app._cleanup_orphaned_recent_uploads([str(rec_dir / "sample_keep.jpg")])
    assert not orphan.exists()
