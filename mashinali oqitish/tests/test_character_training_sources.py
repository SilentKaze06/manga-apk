import io
import json
import string
import zipfile

import torch
from PIL import Image

from dataset_loader import AlphanumericArchiveDataset, CHARACTER_ARCHIVE_RELATIVE_PATH
from self_learning_trainer import build_ocr_training_loader


CHARACTERS = string.digits + string.ascii_uppercase + string.ascii_lowercase
VOCABULARY = {char: 100 + index * 3 for index, char in enumerate(CHARACTERS)}
VOCABULARY["漢"] = 999


def write_archive(base_dir):
    path = base_dir / CHARACTER_ARCHIVE_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    buffer = io.BytesIO()
    Image.new("L", (12, 20), 255).save(buffer, format="PNG")
    with zipfile.ZipFile(path, "w") as archive:
        for char in CHARACTERS:
            folder = char if char.isdigit() else f"{char}_{'U' if char.isupper() else 'L'}"
            for index in range(30):
                archive.writestr(f"dataset/{folder}/{index}.png", buffer.getvalue())
    return path


def write_feedback(base_dir):
    path = base_dir / "self_learning_data" / "ocr" / "feedback_ocr.jsonl"
    crops = path.parent / "crops"
    crops.mkdir(parents=True)
    rows = []
    for index, char in enumerate(("漢", "a")):
        image_path = crops / f"feedback_{index}.png"
        Image.new("RGB", (18, 25), "white").save(image_path)
        rows.append(json.dumps({"char": char, "image_path": str(image_path), "verified": True}, ensure_ascii=False))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_archive_is_available_without_feedback_and_excludes_held_out_images(tmp_path):
    write_archive(tmp_path)
    loader, counts = build_ocr_training_loader(tmp_path, VOCABULARY, batch_size=8)
    dataset = loader.dataset
    try:
        assert counts == {"permanent": len(dataset), "feedback": 0}
        assert dataset.total_count == 62 * 30
        assert len(dataset) == dataset.split_counts["train"]
        assert dataset.split_counts["val"] > 0 and dataset.split_counts["test"] > 0
        assert all(AlphanumericArchiveDataset.split_for_member(name) == "train"
                   for name, _ in dataset.samples)
        assert {label for _, label in dataset.samples} == {VOCABULARY[char] for char in CHARACTERS}
        for name, label in dataset.samples:
            assert label == VOCABULARY[name.split("/")[1][0]]
        images, labels = next(iter(loader))
        assert images.shape == (8, 3, 64, 64)
        assert labels.dtype == torch.long
        assert torch.isfinite(images).all()
    finally:
        dataset.close()


def test_feedback_remains_usable_without_archive_and_keeps_multilingual_labels(tmp_path):
    write_feedback(tmp_path)
    loader, counts = build_ocr_training_loader(tmp_path, VOCABULARY, batch_size=8)
    images, labels = next(iter(loader))
    assert counts == {"permanent": 0, "feedback": 2}
    assert images.shape == (2, 3, 64, 64)
    assert set(labels.tolist()) == {VOCABULARY["漢"], VOCABULARY["a"]}


def test_empty_project_returns_no_loader(tmp_path):
    loader, counts = build_ocr_training_loader(tmp_path, VOCABULARY)
    assert loader is None
    assert counts == {"permanent": 0, "feedback": 0}


def test_mixed_sources_balance_small_feedback_set_and_respect_batch_budget(tmp_path):
    archive_path = write_archive(tmp_path)
    feedback_path = write_feedback(tmp_path)
    archive_before = archive_path.read_bytes()
    feedback_before = feedback_path.read_bytes()
    loader, counts = build_ocr_training_loader(
        tmp_path, VOCABULARY, batch_size=16, max_batches=64
    )
    permanent = loader.dataset.datasets[0]
    try:
        assert counts["permanent"] > 1000 and counts["feedback"] == 2
        assert len(loader) == 64
        loader.sampler.generator = torch.Generator().manual_seed(1234)
        sampled_indices = list(loader.sampler)
        assert len(sampled_indices) == 16 * 64
        feedback_fraction = sum(index >= counts["permanent"] for index in sampled_indices) / len(sampled_indices)
        assert 0.4 < feedback_fraction < 0.6
        assert all(AlphanumericArchiveDataset.split_for_member(name) == "train"
                   for name, _ in permanent.samples)

        images, labels = next(iter(loader))
        assert images.shape == (16, 3, 64, 64)
        assert VOCABULARY["漢"] in labels.tolist()
        assert any(label not in {VOCABULARY["漢"], VOCABULARY["a"]} for label in labels.tolist())
        assert set(labels.tolist()) <= set(VOCABULARY.values())
        assert archive_path.read_bytes() == archive_before
        assert feedback_path.read_bytes() == feedback_before
    finally:
        permanent.close()
