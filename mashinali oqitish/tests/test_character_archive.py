import io
import pickle
import string
import zipfile

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset_loader import AlphanumericArchiveDataset


CHARACTERS = string.digits + string.ascii_uppercase + string.ascii_lowercase
# Deliberately unrelated to 0..61, to catch incorrect use of folder enumeration.
VOCABULARY = {char: 100 + idx * 3 for idx, char in enumerate(CHARACTERS)}


def folder_for(char):
    return char if char.isdigit() else f"{char}_{'U' if char.isupper() else 'L'}"


def make_archive(path, samples_per_class=3, excluded=None, extra=None, corrupt=None, reverse=False):
    buffer = io.BytesIO()
    Image.new("L", (12, 20), 255).save(buffer, format="PNG")
    entries = [
        (f"dataset/{folder_for(char)}/{idx}.png", buffer.getvalue())
        for char in CHARACTERS if char != excluded
        for idx in range(samples_per_class)
    ]
    if extra:
        entries.append((extra, buffer.getvalue()))
    if reverse:
        entries.reverse()
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in entries:
            # Preserve malformed raw names; ZipInfo otherwise normalizes Windows
            # backslashes and truncates NULs before the loader can inspect them.
            info = zipfile.ZipInfo(name)
            info.filename = info.orig_filename = name
            archive.writestr(info, b"not an image" if name == corrupt else payload)
    return path


def test_all_classes_use_current_vocabulary_and_read_without_extraction(tmp_path):
    path = make_archive(tmp_path / "characters.zip")
    dataset = AlphanumericArchiveDataset(path, VOCABULARY, split="all")
    assert dataset.total_count == len(dataset) == 62 * 3
    assert set(dataset.class_counts) == set(CHARACTERS)
    assert dataset._archive is None
    for index, (name, label) in enumerate(dataset.samples):
        folder = name.split("/")[1]
        char = folder if len(folder) == 1 else folder[0]
        image, actual_label = dataset[index]
        assert image.shape == (3, 64, 64)
        assert torch.equal(image, torch.ones_like(image))
        assert actual_label.item() == label == VOCABULARY[char]
        assert actual_label.dtype == torch.long
    dataset.close()
    assert list(tmp_path.iterdir()) == [path]


def test_splits_are_disjoint_complete_and_independent_of_zip_entry_order(tmp_path):
    first = make_archive(tmp_path / "first.zip", samples_per_class=20)
    second = make_archive(tmp_path / "second.zip", samples_per_class=20, reverse=True)
    split_names = {}
    for split in ("train", "val", "test"):
        left = AlphanumericArchiveDataset(first, VOCABULARY, split=split)
        right = AlphanumericArchiveDataset(second, VOCABULARY, split=split)
        assert left.samples == right.samples
        assert len(left) == left.split_counts[split]
        split_names[split] = {name for name, _ in left.samples}
        assert all(AlphanumericArchiveDataset.split_for_member(name) == split for name in split_names[split])
    assert not split_names["train"] & split_names["val"]
    assert not split_names["train"] & split_names["test"]
    assert not split_names["val"] & split_names["test"]
    assert len(set.union(*split_names.values())) == 62 * 20


@pytest.mark.parametrize("invalid_name", [
    "dataset/../escape.png",
    "dataset/0/../escape.png",
    "/dataset/0/1.png",
    "dataset\\0\\1.png",
    "dataset/unknown/1.png",
    "dataset/0/unexpected.txt",
    "dataset/0/extra.png\x00ignored",
])
def test_malformed_or_unknown_archive_entries_are_rejected(tmp_path, invalid_name):
    path = make_archive(tmp_path / "invalid.zip", extra=invalid_name)
    with pytest.raises(ValueError):
        AlphanumericArchiveDataset(path, VOCABULARY)


def test_missing_class_or_vocabulary_label_is_rejected(tmp_path):
    incomplete = make_archive(tmp_path / "incomplete.zip", excluded="z")
    with pytest.raises(ValueError, match="missing classes"):
        AlphanumericArchiveDataset(incomplete, VOCABULARY)
    complete = make_archive(tmp_path / "complete.zip")
    missing_label = {char: index for char, index in VOCABULARY.items() if char != "A"}
    with pytest.raises(ValueError, match="missing character labels"):
        AlphanumericArchiveDataset(complete, missing_label)


def test_corrupt_sample_raises_instead_of_training_on_a_blank_image(tmp_path):
    corrupt_name = "dataset/A_U/0.png"
    path = make_archive(tmp_path / "corrupt.zip", corrupt=corrupt_name)
    dataset = AlphanumericArchiveDataset(path, VOCABULARY, split="all")
    index = next(i for i, (name, _) in enumerate(dataset.samples) if name == corrupt_name)
    with pytest.raises(RuntimeError, match="Cannot read character image"):
        dataset[index]
    dataset.close()


def test_open_archive_survives_pickle_and_windows_style_worker_spawn(tmp_path):
    path = make_archive(tmp_path / "characters.zip", samples_per_class=1)
    dataset = AlphanumericArchiveDataset(path, VOCABULARY, split="all")
    first_image, first_label = dataset[0]
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._archive is None
    image, label = restored[0]
    assert torch.equal(image, first_image)
    assert label == first_label
    restored.close()
    loader = DataLoader(dataset, batch_size=16, num_workers=1, multiprocessing_context="spawn")
    batches = list(loader)
    assert sum(images.shape[0] for images, _ in batches) == len(dataset)
    assert torch.equal(batches[0][0][0], first_image)
    dataset.close()


def test_validation_and_test_transforms_do_not_augment(tmp_path):
    path = make_archive(tmp_path / "characters.zip", samples_per_class=20)
    for split in ("val", "test", "all"):
        dataset = AlphanumericArchiveDataset(path, VOCABULARY, split=split, augment=True)
        assert not any(type(transform).__name__ == "ColorJitter" for transform in dataset.transform.transforms)
