import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T

from manga_pipeline import MangaTranslationPipeline


def bare_pipeline(reader=None):
    pipe = MangaTranslationPipeline.__new__(MangaTranslationPipeline)
    pipe.model_lock = threading.RLock()
    pipe.collector = None
    pipe.ocr = None
    pipe._get_ocr_reader = lambda lang: reader
    pipe.det_transform = T.Compose([T.Resize((384, 256)), T.ToTensor()])
    pipe.inpaint_transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    pipe.detector = SimpleNamespace(predict=lambda *args, **kwargs: [])
    class Inpainter:
        calls = 0
        def __call__(self, raw, mask):
            self.calls += 1
            return torch.ones_like(raw), raw * (1-mask) + mask
    pipe.inpainter = Inpainter()
    return pipe


def page():
    image = Image.new("RGB", (120, 120), "white")
    ImageDraw.Draw(image).rectangle((35, 45, 65, 65), fill="black")
    return image


def test_auto_keeps_original_and_requests_multilingual_fallback():
    pipe = bare_pipeline()
    pipe._get_ocr_reader = lambda lang: pytest.fail("Auto must not guess Japanese")
    original = page()
    result = pipe.process_page(original, auto_collect=False, source_lang="auto")
    assert result["detected_source_lang"] == "auto"
    assert result["requires_ocr_fallback"] and not result["cleaned_bubble_ids"]
    assert result["cleaned_image"].tobytes() == original.tobytes()


@pytest.mark.parametrize("confidence", [0., .49])
def test_uncertain_or_empty_ocr_never_erases_source(confidence):
    reader = SimpleNamespace(readtext=lambda *args, **kwargs: [([[30, 30], [70, 30], [70, 80], [30, 80]], "uncertain", confidence)])
    pipe = bare_pipeline(reader)
    original = page()
    result = pipe.process_page(original, known_bubbles=[{"box": [25, 25, 85, 85], "text": ""}], auto_collect=False, source_lang="ko")
    assert result["cleaned_image"].tobytes() == original.tobytes()
    assert result["mask_image"].getextrema() == (0, 0)
    assert result["requires_ocr_fallback"] and not result["cleaned_bubble_ids"]
    assert pipe.inpainter.calls == 0


def test_recognized_bubble_marks_only_cleaned_regions():
    pipe = bare_pipeline()
    original = page()
    result = pipe.process_page(original, known_bubbles=[{"box": [25, 25, 85, 85], "text": "HELLO"}], auto_collect=False, source_lang="en")
    assert result["cleaned_bubble_ids"] == [1]
    assert result["dialogues"][0]["ocr_trusted"]
    assert pipe.inpainter.calls == 1
    mask = np.array(result["mask_image"])
    assert np.array_equal(np.array(original)[mask == 0], np.array(result["cleaned_image"])[mask == 0])


def test_readers_are_selected_per_language_without_download(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(Reader=lambda langs, **opts: calls.append((langs, opts)) or object()))
    monkeypatch.setenv("MANGA_EASYOCR_MODEL_DIR", "/local/models")
    pipe = MangaTranslationPipeline.__new__(MangaTranslationPipeline)
    pipe.model_lock = threading.RLock()
    pipe._ocr_readers = {}
    for language in ("ja", "ko", "zh", "en", "ko"):
        pipe._get_ocr_reader(language)
    assert [c[0] for c in calls] == [["ja", "en"], ["ko", "en"], ["ch_sim", "en"], ["en"]]
    assert all(c[1]["download_enabled"] is False for c in calls)
    assert all(c[1]["model_storage_directory"] == "/local/models" for c in calls)


def test_failed_reload_keeps_live_models(tmp_path):
    (tmp_path / "ocr_vocab.json").write_text('{"char_to_idx": {"a": 0}, "idx_to_char": {"0": "a"}}')
    pipe = bare_pipeline()
    pipe.vocab_path = str(tmp_path / "ocr_vocab.json")
    pipe.checkpoint_dir = str(tmp_path / "missing")
    old = pipe.detector, pipe.ocr, pipe.inpainter
    with pytest.raises(FileNotFoundError):
        pipe.reload_models()
    assert old == (pipe.detector, pipe.ocr, pipe.inpainter)
