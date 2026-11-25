import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image, ImageDraw

from dataset_loader import MangaDetectionDataset, MangaOCRDataset, MangaInpaintingDataset
from validation_gate import ValidationGate, ValidationUnavailable, atomic_promote, file_digest, should_promote

try:
    from self_learning_collector import SelfLearningCollector
except ImportError:
    SelfLearningCollector = None

try:
    from self_learning_trainer import DynamicOCRDataset, DynamicDetectionDataset, DynamicInpaintDataset, SelfLearningTrainer
except ImportError:
    DynamicOCRDataset = DynamicDetectionDataset = DynamicInpaintDataset = SelfLearningTrainer = None


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_detection_requires_real_images_unless_demo_is_explicit(tmp_path):
    path = jsonl(tmp_path / "train.jsonl", [{"page_id": "missing", "annotations": []}])
    with pytest.raises(FileNotFoundError, match="1/1"):
        MangaDetectionDataset(path)
    demo = MangaDetectionDataset(path, allow_synthetic=True)
    assert demo.missing_image_count == 1
    assert demo[0]["image"].shape == (3, 384, 256)


def test_ocr_preserves_complete_label_and_reports_impossible_ctc_labels(tmp_path, caplog):
    image = tmp_path / "crop.png"
    Image.new("RGB", (256, 64), "white").save(image)
    text = "abc" * 9  # 27 characters: previously silently truncated to 24.
    path = jsonl(tmp_path / "ocr.jsonl", [
        {"text": text, "image_path": str(image)},
        {"text": "a" * 17, "image_path": str(image)},  # repeats require 33 CTC steps
    ])
    data = MangaOCRDataset(path, {"<blank>": 0, "<unk>": 1, "a": 2, "b": 3, "c": 4})
    assert len(data) == 1 and data.filtered_count == 1
    assert data[0]["text"] == text and len(data[0]["target"]) == 27
    assert "labels were not truncated" in caplog.text


def test_missing_ocr_or_triplet_never_silently_becomes_synthetic(tmp_path):
    ocr = jsonl(tmp_path / "ocr.jsonl", [{"text": "abc"}])
    with pytest.raises(FileNotFoundError):
        MangaOCRDataset(ocr, {"a": 2, "b": 3, "c": 4})
    inpaint = jsonl(tmp_path / "inpaint.jsonl", [{"triplet_id": "missing"}])
    with pytest.raises(FileNotFoundError):
        MangaInpaintingDataset(inpaint)


@pytest.mark.skipif(DynamicOCRDataset is None, reason="self_learning_trainer purged in Phase 3")
def test_unverified_and_held_out_feedback_never_enters_training(tmp_path):
    image = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), "white").save(image)
    rows = [{"image_path": str(image), "char": "A", "verified": flag, "split": split}
            for flag, split in [(False, "train"), (True, "val"), (True, "test"), (True, "train")]]
    path = jsonl(tmp_path / "ocr.jsonl", rows)
    assert len(DynamicOCRDataset(path, char_to_idx={"A": 2})) == 1
    det = {"image_path": str(image), "verified": True, "annotations": [{"bbox": [1, 1, 20, 20]}]}
    path = jsonl(tmp_path / "det.jsonl", [det, {**det, "annotations_verified": True}])
    assert len(DynamicDetectionDataset(path)) == 1
    item = {"raw_image": str(image), "mask": str(image), "ground_truth": str(image), "verified": True}
    path = jsonl(tmp_path / "inpaint.jsonl", [{**item, "ground_truth_source": "model_generated"},
                                            {**item, "ground_truth_source": "human_cleaned"}])
    assert len(DynamicInpaintDataset(path)) == 1


@pytest.mark.skipif(SelfLearningCollector is None, reason="self_learning_collector purged in Phase 3")
def test_automatic_page_cannot_verify_its_own_generated_targets(tmp_path):
    (tmp_path / "ocr_vocab.json").write_text(json.dumps({"char_to_idx": {"A": 2}, "idx_to_char": {"2": "A"}}))
    collector = SelfLearningCollector(tmp_path)
    raw = Image.new("RGB", (128, 128), "white")
    ImageDraw.Draw(raw).rectangle((40, 40, 70, 70), fill="black")
    mask = Image.new("L", raw.size, 0)
    ImageDraw.Draw(mask).rectangle((40, 40, 70, 70), fill=255)
    collector.record_processed_page(raw, Image.new("RGB", raw.size, "white"), mask,
                                    [{"bbox": [20, 20, 100, 100], "transcription": "A"}], is_verified=True)
    inpaint = json.loads(open(collector.inpaint_jsonl, encoding="utf-8").readline())
    assert inpaint["verified"] is False and inpaint["ground_truth_source"] == "model_generated"
    detector = json.loads(open(collector.det_jsonl, encoding="utf-8").readline())
    assert detector["verified"] is False and detector["annotations_verified"] is False


def test_promotion_requires_character_and_line_ocr_non_regression():
    baseline = {"char_accuracy": .8, "line_cer": .2}
    assert not should_promote(baseline, {"char_accuracy": .9, "line_cer": .3})
    assert not should_promote(baseline, baseline)
    assert not should_promote(baseline, {"char_accuracy": float("nan"), "line_cer": .1})
    assert should_promote(baseline, {"char_accuracy": .9, "line_cer": .2})
    assert not should_promote({"precision": .8, "recall": .8, "f1": .8}, {"precision": .9, "recall": .7, "f1": .79})
    assert should_promote({"hole_mae": .2}, {"hole_mae": .1})


@pytest.mark.skipif(SelfLearningTrainer is None, reason="self_learning_trainer purged in Phase 3")
def test_missing_validation_skips_training_and_keeps_checkpoint(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    checkpoint = checkpoints / "best_manga_ocr.pt"
    torch.save({"weight": torch.ones(1)}, checkpoint)
    before = checkpoint.read_bytes()
    trainer = SelfLearningTrainer(tmp_path)
    result = trainer._prepare_validation("ocr", object(), SimpleNamespace(samples=[]), checkpoint, {}, None)
    assert result is None
    assert checkpoint.read_bytes() == before
    assert not trainer.promotion_results["ocr"]["promoted"]
    assert "validation" in trainer.promotion_results["ocr"]["reason"]


def test_atomic_promotion_preserves_backup_and_rejects_changed_incumbent(tmp_path):
    checkpoint = tmp_path / "best.pt"
    torch.save({"old": torch.ones(1)}, checkpoint)
    original = checkpoint.read_bytes()
    digest = file_digest(checkpoint)
    model = torch.nn.Linear(1, 1)
    atomic_promote(model, checkpoint, digest)
    assert (tmp_path / "best_prev.pt").read_bytes() == original
    assert torch.equal(torch.load(checkpoint, weights_only=True)["weight"], model.weight)
    changed = checkpoint.read_bytes()
    with pytest.raises(ValidationUnavailable, match="changed"):
        atomic_promote(model, checkpoint, digest)
    assert checkpoint.read_bytes() == changed


def test_failed_serialization_keeps_incumbent_and_cleans_temp(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    torch.save({"old": torch.ones(1)}, checkpoint)
    original = checkpoint.read_bytes()
    digest = file_digest(checkpoint)
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        atomic_promote(torch.nn.Linear(1, 1), checkpoint, digest)
    assert checkpoint.read_bytes() == original
    assert list(tmp_path.iterdir()) == [checkpoint]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_candidate_never_replaces_incumbent(tmp_path, invalid):
    checkpoint = tmp_path / "best.pt"
    torch.save({"old": torch.ones(1)}, checkpoint)
    original = checkpoint.read_bytes()
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(invalid)
    with pytest.raises(ValidationUnavailable, match="non-finite"):
        atomic_promote(model, checkpoint, file_digest(checkpoint))
    assert checkpoint.read_bytes() == original
    assert list(tmp_path.iterdir()) == [checkpoint]


def test_validation_rejects_pseudolabels_and_duplicate_training_images(tmp_path):
    manifest = tmp_path / "validation" / "detector.jsonl"
    jsonl(manifest, [{"verified": False}])
    with pytest.raises(ValidationUnavailable, match="verified=true"):
        ValidationGate(tmp_path, "detector", {}, SimpleNamespace(samples=[]), min_samples=1)
    image = tmp_path / "page.png"
    Image.new("RGB", (40, 60), "white").save(image)
    jsonl(manifest, [{"verified": True, "split": "val", "source_type": "real_manga_page", "image_path": str(image)}])
    with pytest.raises(ValidationUnavailable, match="overlap"):
        ValidationGate(tmp_path, "detector", {}, SimpleNamespace(samples=[{"image_path": str(image)}]), min_samples=1)


@pytest.mark.skipif(SelfLearningTrainer is None, reason="self_learning_trainer purged in Phase 3")
def test_unpromoted_cycle_never_hot_reloads_or_reports_success(tmp_path, monkeypatch):
    trainer = SelfLearningTrainer(tmp_path)
    for method in ("fine_tune_ocr", "fine_tune_inpainting", "fine_tune_detector"):
        monkeypatch.setattr(trainer, method, lambda **kwargs: (None, None))
    class Pipeline:
        def reload_models(self):
            raise AssertionError("Unpromoted candidate must not reload")
    result = trainer.execute_self_training_cycle(epochs=1, max_batches=1, pipeline_ref=Pipeline())
    assert not result["success"] and not result["promoted_tasks"] and result["cycle"] == 0
