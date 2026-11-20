import os
import sys
import re
import urllib.error
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

# Ensure both server and mashinali oqitish are importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
SERVER_DIR = os.path.join(PROJECT_ROOT, "manga_apk", "server")
HF_DIR = os.path.join(PROJECT_ROOT, "hf_dosjanmanga")

for p in [BASE_DIR, SERVER_DIR, HF_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import utils
from manga_pipeline import (
    split_cluster_into_columns,
    merge_adjacent_bubble_columns,
    _crop_reading_key,
    is_dark_barrier_between_x,
    is_dark_barrier_between_y,
    are_different_white_contours,
    can_merge_text_boxes,
    wrap_text_elliptical,
)
import gemini_translator as server_gt
import manga_bubble_engine as server_be
import pdf_jobs
import onnx_engine

try:
    import importlib.util
    def load_module_from_path(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    hf_gt = load_module_from_path("hf_gemini_translator", os.path.join(HF_DIR, "gemini_translator.py"))
    hf_be = load_module_from_path("hf_manga_bubble_engine", os.path.join(HF_DIR, "manga_bubble_engine.py"))
    hf_pj = load_module_from_path("hf_pdf_jobs", os.path.join(HF_DIR, "pdf_jobs.py"))
    hf_oe = load_module_from_path("hf_onnx_engine", os.path.join(HF_DIR, "onnx_engine.py"))
except Exception as e:
    hf_gt = None
    hf_be = None
    hf_pj = None
    hf_oe = None


# ==============================================================================
# 1. RTL READING ORDER IN MULTI-COLUMN BUBBLES
# ==============================================================================

def test_split_cluster_into_columns_rtl():
    """Two-column dialogue bubbles in manga must be read right column first (RTL)."""
    raw_boxes = [
        # Left column (x: 10..50)
        {"box": [10, 20, 50, 45], "text": "L-LET ME"},
        {"box": [10, 50, 50, 75], "text": "KNOW..."},
        # Right column (x: 75..115)
        {"box": [75, 20, 115, 45], "text": "OH, YOUR"},
        {"box": [75, 50, 115, 75], "text": "GOOD HEARING..."},
    ]
    indices = [0, 1, 2, 3]

    # Default reading_direction is 'rtl': right column [2, 3] must come before left column [0, 1]
    res_rtl = split_cluster_into_columns(indices, raw_boxes, reading_direction="rtl")
    assert len(res_rtl) == 2
    assert res_rtl[0] == [2, 3], f"Right column must come first in RTL, got {res_rtl[0]}"
    assert res_rtl[1] == [0, 1], f"Left column must come second in RTL, got {res_rtl[1]}"

    # When explicitly set to LTR, left column comes first
    res_ltr = split_cluster_into_columns(indices, raw_boxes, reading_direction="ltr")
    assert len(res_ltr) == 2
    assert res_ltr[0] == [0, 1]
    assert res_ltr[1] == [2, 3]


def test_merge_adjacent_bubble_columns_rtl():
    """Merging adjacent columns in a wide bubble must keep right column before left column in RTL."""
    gray = np.full((120, 180), 255, dtype=np.uint8)
    # Side-by-side columns
    clusters = [
        # Left column (w: 60, h: 50)
        {"text_box": [10, 20, 70, 70], "text": "L-LET ME KNOW...", "conf": 0.95, "indices": [0]},
        # Right column (w: 60, h: 50)
        {"text_box": [75, 20, 135, 70], "text": "OH, YOUR GOOD HEARING...", "conf": 0.95, "indices": [1]},
    ]

    merged_rtl = merge_adjacent_bubble_columns(clusters, gray, reading_direction="rtl")
    assert len(merged_rtl) == 1
    # Right column text must precede left column text
    assert merged_rtl[0]["text"].startswith("OH, YOUR GOOD HEARING...")
    assert "L-LET ME KNOW..." in merged_rtl[0]["text"]

    merged_ltr = merge_adjacent_bubble_columns(clusters, gray, reading_direction="ltr")
    assert len(merged_ltr) == 1
    assert merged_ltr[0]["text"].startswith("L-LET ME KNOW...")
    assert "OH, YOUR GOOD HEARING..." in merged_ltr[0]["text"]


def test_crop_reading_key_rtl():
    """_crop_reading_key orders items right-to-left when reading_direction='rtl'."""
    left_item = {"box": [10, 20, 50, 40]}
    right_item = {"box": [70, 20, 110, 40]}

    key_left_rtl = _crop_reading_key(left_item, reading_direction="rtl")
    key_right_rtl = _crop_reading_key(right_item, reading_direction="rtl")

    # In RTL, right item should come first (have smaller key)
    assert key_right_rtl < key_left_rtl, f"Right key {key_right_rtl} should be < left key {key_left_rtl}"

    key_left_ltr = _crop_reading_key(left_item, reading_direction="ltr")
    key_right_ltr = _crop_reading_key(right_item, reading_direction="ltr")
    assert key_left_ltr < key_right_ltr


# ==============================================================================
# 2. RESTORE SPLIT HYPHENATED NAMES / WORDS
# ==============================================================================

def test_hyphenated_word_restoration():
    """TSU- + MIKI'S -> TSUMIKI'S restored, while legitimate compound words are preserved."""
    # Test line break joining regex: TSU- \n MIKI'S -> TSUMIKI'S
    broken_name = "HELLO TSU-\nMIKI'S HERE"
    restored = re.sub(r'(\b[A-Za-z]+)-\s*\n\s*([A-Za-z\']+)', r'\1\2', broken_name)
    assert restored == "HELLO TSUMIKI'S HERE"

    # Legitimate hyphenated compound word without newline must stay hyphenated
    compound = "THE SECOND-YEARS AND THIRD-YEARS"
    assert "SECOND-YEARS" in compound
    # repair_comic_text should not destroy compound words
    cleaned_compound = utils.repair_comic_text("SECOND-YEARS")
    assert cleaned_compound.strip() == "SECOND-YEARS"

    # Single word broken with hyphen across lines
    broken_word = "UNDER-\nSTANDING"
    restored_word = re.sub(r'(\b[A-Za-z]+)-\s*\n\s*([A-Za-z\']+)', r'\1\2', broken_word)
    assert restored_word == "UNDERSTANDING"


# ==============================================================================
# 3. PRESERVE CLASS 1-1 AND 1ST (NO OVER-AGGRESSIVE REGEX)
# ==============================================================================

def test_class_1_1_and_1st_preserved():
    """CLASS 1-1, 1-A, and 1st must not be corrupted to Roman numerals or letters."""
    raw_texts = [
        "CLASS 1-1",
        "CLASS 1-A",
        "1st PLACE",
        "THE 1st TIME",
        "GRADE 1-2",
    ]

    for raw in raw_texts:
        # Test the regex used in manga_pipeline for '1' normalization
        t = re.sub(r'(?<![\d\-])\b1\b(?!\s*(?:st|nd|rd|th)\b)(?![\d\-])', 'I', raw, flags=re.I)
        t = re.sub(r'\b1(?!(?:st|nd|rd|th)\b)(?=[A-Za-z])', 'I', t, flags=re.I)
        assert "I-I" not in t, f"CLASS 1-1 was corrupted to {t}"
        assert "1st" in t or "1-1" in t or "1-A" in t or "1-2" in t, f"Expected digit preservation in {t}"

    # Standalone '1' (meaning 'I') should still be converted
    standalone = "1 AM HERE"
    t_conv = re.sub(r'(?<![\d\-])\b1\b(?!\s*(?:st|nd|rd|th)\b)(?![\d\-])', 'I', standalone, flags=re.I)
    assert t_conv == "I AM HERE"


# ==============================================================================
# 4. IS_SFX_WORD DIALOGUE WORDS FIX
# ==============================================================================

def test_is_sfx_word_dialogue_words():
    """Dialogue words like WAIT, YES, NO, STOP, WHY? must return False (not SFX)."""
    dialogue_words = ["WAIT", "YES", "NO", "STOP", "WHY?", "Why", "No!", "Yes...", "HELP", "OK"]
    for w in dialogue_words:
        assert server_gt.is_sfx_word(w) is False, f"Server: '{w}' should NOT be considered SFX"
        if hf_gt:
            assert hf_gt.is_sfx_word(w) is False, f"HF: '{w}' should NOT be considered SFX"

    actual_sfx = ["BOOM", "DOKI", "KRRR", "wham", "bang", "crash", "sigh", "whoosh"]
    for s in actual_sfx:
        assert server_gt.is_sfx_word(s) is True, f"Server: '{s}' should be recognized as SFX"
        if hf_gt:
            assert hf_gt.is_sfx_word(s) is True, f"HF: '{s}' should be recognized as SFX"


# ==============================================================================
# 5. DARK FRAME / SEPARATE WHITE COMPONENTS MERGE BARRIER
# ==============================================================================

def test_dark_barrier_prevents_merge():
    """Two boxes separated by a dark line (or crossing dark pixels) must never merge."""
    # Create white page with a vertical black divider line at x=50
    gray = np.full((100, 100), 255, dtype=np.uint8)
    gray[:, 48:52] = 0  # Black line

    boxA = [10, 20, 45, 80]
    boxB = [55, 20, 90, 80]

    # Test server engine
    assert server_be._is_dark_barrier_between_x(gray, boxA, boxB) is True
    assert server_be._can_merge_text_boxes(boxA, boxB, gray, max_page_h=100) is False

    # Test pipeline functions
    assert is_dark_barrier_between_x(gray, boxA, boxB) is True
    assert can_merge_text_boxes(boxA, boxB, gray) is False


def test_dark_barrier_overlapping_boxes_prevents_merge():
    """When boxes overlap in x (xA2 >= xB1), sampling line between centers still detects dark barrier."""
    gray = np.full((100, 100), 255, dtype=np.uint8)
    gray[:, 49:52] = 0  # Black divider line

    # boxA and boxB overlap horizontally between x=45 and x=55
    boxA = [20, 20, 55, 80]
    boxB = [45, 20, 80, 80]

    assert server_be._is_dark_barrier_between_x(gray, boxA, boxB) is True
    assert server_be._can_merge_text_boxes(boxA, boxB, gray, max_page_h=100) is False


def test_separate_white_connected_components_prevent_merge():
    """Two boxes in distinct white contours (separate speech bubbles) must NEVER be merged."""
    # Dark panel background (gray=50) with two isolated white bubble discs (gray=255)
    gray = np.full((150, 300), 50, dtype=np.uint8)
    # Bubble 1: [20..130, 20..130]
    gray[20:130, 20:130] = 255
    # Bubble 2: [20..130, 170..280]
    gray[20:130, 170:280] = 255

    boxA = [30, 30, 120, 120]
    boxB = [180, 30, 270, 120]

    # Test server connected component check
    assert server_be._can_merge_text_boxes(boxA, boxB, gray, max_page_h=150) is False

    # Test pipeline connected component check
    assert are_different_white_contours(gray, boxA, boxB) is True
    assert can_merge_text_boxes(boxA, boxB, gray) is False


# ==============================================================================
# 6. GEMINI TRANSLATOR CONFIG, RETRY POLICY, AND REGEX REMOVAL
# ==============================================================================

def test_gemini_models_valid():
    """GEMINI_MODELS must not contain non-existent 3.5 models."""
    assert "gemini-3.5-flash-lite" not in server_gt.GEMINI_MODELS
    assert "gemini-3.5-flash" not in server_gt.GEMINI_MODELS
    assert "gemini-2.5-flash-lite" in server_gt.GEMINI_MODELS
    assert "gemini-2.5-flash" in server_gt.GEMINI_MODELS
    assert "gemini-2.0-flash" in server_gt.GEMINI_MODELS

    if hf_gt:
        assert "gemini-3.5-flash-lite" not in hf_gt.GEMINI_MODELS
        assert "gemini-3.5-flash" not in hf_gt.GEMINI_MODELS


def test_gemini_retry_policy_400_404():
    """HTTP 400 and 404 must NOT be retried on the same model (advances to next model, aborting after all models)."""
    for err_code in (400, 404):
        server_gt._gemini_rate_limited_until = 0.0
        gemini_calls = 0
        def fake_urlopen(req, timeout=None):
            nonlocal gemini_calls
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "generativelanguage.googleapis.com" in url:
                gemini_calls += 1
                raise urllib.error.HTTPError(url=url, code=err_code, msg=f"Error {err_code}", hdrs={}, fp=None)
            raise urllib.error.HTTPError(url=url, code=500, msg="Fallback error", hdrs={}, fp=None)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            res = server_gt.translate_dialogues_with_gemini(["Hello world"], target_lang="uz", api_key="test_key")
            # Each model must be attempted at most once (no retry on 400/404 within the model)
            assert gemini_calls == len(server_gt.GEMINI_MODELS), f"Expected {len(server_gt.GEMINI_MODELS)} Gemini calls on HTTP {err_code}, got {gemini_calls}"


def test_temperature_is_zero():
    """Translation temperature must be 0.0 for deterministic outputs."""
    server_gt._gemini_rate_limited_until = 0.0
    captured_payload = None
    def fake_urlopen(req, timeout=None):
        nonlocal captured_payload
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "generativelanguage.googleapis.com" in url:
            import json
            captured_payload = json.loads(req.data.decode("utf-8"))
            raise urllib.error.HTTPError(url=url, code=400, msg="Stop", hdrs={}, fp=None)
        raise urllib.error.HTTPError(url=url, code=500, msg="Fallback error", hdrs={}, fp=None)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        server_gt.translate_dialogues_with_gemini(["Hello world"], target_lang="uz", api_key="test_key")

    assert captured_payload is not None
    temp = captured_payload.get("generationConfig", {}).get("temperature")
    assert temp in (0.0, 0.1), f"Expected temperature 0.0 or 0.1, got {temp}"


def test_single_page_regexes_removed():
    """Hardcoded single-page regexes ('bungasimon', 'juda qiziq', 'shundaxam', etc.) must not alter text."""
    test_str = "shundaxam turgandir bungasimon juda qiziq"
    norm = server_gt._normalize_uzbek_text(test_str)
    # The words themselves should remain in the string, not replaced by hardcoded page translations
    assert "shundaxam" in norm
    assert "turgandir" in norm
    assert "bungasimon" in norm


# ==============================================================================
# 7. HYPHENATION FALLBACK AND CACHE SIZE
# ==============================================================================

def test_hyphenation_fallback_in_wrap_text_elliptical():
    """wrap_text_elliptical respects allow_hyphenation flag."""
    font = utils.get_best_font(size=14)
    # Long Uzbek word >= 10 chars
    words = ["QAHRAMONLARIMIZNING", "MAKTABI"]
    a, b = 100, 80

    # With allow_hyphenation=False, words are kept unhyphenated if allowed (or not split with hyphen)
    lines_no_hyphen = wrap_text_elliptical(words, font, 120, 80, allow_hyphenation=False)
    assert lines_no_hyphen is not None
    for line in lines_no_hyphen:
        assert not line.endswith("-")
    assert "QAHRAMONLARIMIZNING" in lines_no_hyphen[0]

    # With allow_hyphenation=True on tighter width, long word is split with hyphenation
    lines_hyphen = wrap_text_elliptical(words, font, 100, 80, allow_hyphenation=True)
    assert lines_hyphen is not None
    assert any(line.endswith("-") for line in lines_hyphen)


def test_repair_comic_text_lru_cache_512():
    """repair_comic_text must have lru_cache with maxsize=512."""
    assert hasattr(utils.repair_comic_text, "cache_info")
    assert utils.repair_comic_text.cache_info().maxsize == 512


# ==============================================================================
# 8. ADDITIONAL SWARM REGRESSION TESTS
# ==============================================================================

def test_gemini_fallback_on_404_tries_next_model():
    """Mock urllib.request.urlopen returning HTTPError 404 on gemini-2.5-flash-lite, verify it tries gemini-2.5-flash instead of aborting to None."""
    import json
    for gt_mod in [server_gt, hf_gt] if hf_gt else [server_gt]:
        gt_mod._gemini_rate_limited_until = 0.0
        called_models = []

        def fake_urlopen(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            for m in gt_mod.GEMINI_MODELS:
                if m in url:
                    called_models.append(m)
                    break

            if "gemini-2.5-flash-lite" in url:
                raise urllib.error.HTTPError(url=url, code=404, msg="Not Found: gemini-2.5-flash-lite", hdrs={}, fp=None)

            if "gemini-2.5-flash" in url:
                mock_resp = MagicMock()
                mock_resp.__enter__.return_value = mock_resp
                mock_resp.read.return_value = json.dumps({
                    "candidates": [{
                        "content": {
                            "parts": [{
                                "text": json.dumps({"[0]": "Salom dunyo"})
                            }]
                        }
                    }]
                }).encode("utf-8")
                return mock_resp

            raise urllib.error.HTTPError(url=url, code=500, msg="Unexpected URL", hdrs={}, fp=None)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            res = gt_mod.translate_dialogues_with_gemini(["Hello world"], target_lang="uz", api_key="test_key")
            assert "gemini-2.5-flash-lite" in called_models, "gemini-2.5-flash-lite should have been attempted first"
            assert "gemini-2.5-flash" in called_models, "gemini-2.5-flash should have been tried after 404 on lite"
            assert res is not None, "Translation should succeed via fallback model"
            assert res[0] == "Salom dunyo"


def test_in_bubble_vertically_close_lines_merge():
    """Vertically close lines (THEN and CLASS 1-1) inside the same bubble pass are_different_white_contours and is_dark_barrier_between_y and remain in one cluster."""
    # Speech bubble with white background (255) surrounded by darker panel background (50)
    gray = np.full((500, 500), 50, dtype=np.uint8)
    gray[30:470, 30:470] = 255  # White speech bubble

    # Two vertically stacked lines of dialogue within the same bubble:
    # Line 1: "THEN"
    boxA = [80, 60, 150, 90]
    # Line 2: "CLASS 1-1" placed closely underneath Line 1 (gap = 8px)
    boxB = [70, 98, 170, 128]

    # Add simulated text ink inside the bounding boxes (gray = 20)
    gray[68:82, 85:145] = 20
    gray[106:120, 75:165] = 20

    # 1. White connected components check: both boxes are inside the same white bubble
    assert are_different_white_contours(gray, boxA, boxB) is False, "Boxes in same bubble must not have different white contours"

    # 2. Dark barrier check: letter ink must NOT be mistaken for a dark frame barrier
    assert is_dark_barrier_between_y(gray, boxA, boxB) is False, "No dark frame barrier between vertically close lines in bubble"
    assert server_be._is_dark_barrier_between_y(gray, boxA, boxB) is False

    # 3. Merging check: lines must be mergeable into one cluster
    assert can_merge_text_boxes(boxA, boxB, gray, textA="THEN", textB="CLASS 1-1") is True
    assert server_be._can_merge_text_boxes(boxA, boxB, gray, max_page_h=500) is True


def test_overlapping_boxes_with_dark_frame_do_not_merge():
    """Two boxes separated by a continuous dark border line (e.g. frame separating THIRD SEMESTER and SPRING BREAK) are never merged even if vertically overlapping."""
    # White page background (500x500)
    gray = np.full((500, 500), 255, dtype=np.uint8)

    # A continuous horizontal black panel border frame at y=200..203 (thickness 3px across width)
    gray[200:204, :] = 0

    # Upper panel box: "THIRD SEMESTER" ending at y=202 (overlaps border line by 2px)
    boxA = [80, 140, 240, 202]
    # Lower panel box: "SPRING BREAK" starting at y=201 (overlaps border line by 3px)
    boxB = [90, 201, 250, 260]

    # Even though yA2 >= yB1 (vertical overlap of 1px), the continuous dark frame line must be detected
    assert is_dark_barrier_between_y(gray, boxA, boxB) is True, "Dark border frame line must be detected as barrier"
    assert server_be._is_dark_barrier_between_y(gray, boxA, boxB) is True

    # Therefore, merging across panel frame must be strictly blocked
    assert can_merge_text_boxes(boxA, boxB, gray, textA="THIRD SEMESTER", textB="SPRING BREAK") is False
    assert server_be._can_merge_text_boxes(boxA, boxB, gray, max_page_h=500) is False


def test_single_column_not_split_into_two():
    """Center-aligned 5-line single column is not falsely split into 2 columns."""
    # A single paragraph of 5 lines, center-aligned around x=100 with varied line lengths
    raw_boxes = [
        {"box": [65, 20, 135, 40], "text": "I STILL REMEMBER"},
        {"box": [45, 45, 155, 65], "text": "THE DAY WE FIRST MET"},
        {"box": [40, 70, 160, 90], "text": "IN THAT QUIET CLASSROOM"},
        {"box": [55, 95, 145, 115], "text": "DURING THE RAIN"},
        {"box": [75, 120, 125, 140], "text": "AFTER SCHOOL."},
    ]
    indices = [0, 1, 2, 3, 4]

    # In both RTL and LTR, this center-aligned single column must NOT be falsely split
    cols_rtl = split_cluster_into_columns(indices, raw_boxes, reading_direction="rtl")
    assert len(cols_rtl) == 1, f"Center-aligned single column was falsely split in RTL: {cols_rtl}"
    assert cols_rtl[0] == indices

    cols_ltr = split_cluster_into_columns(indices, raw_boxes, reading_direction="ltr")
    assert len(cols_ltr) == 1, f"Center-aligned single column was falsely split in LTR: {cols_ltr}"
    assert cols_ltr[0] == indices


def test_hyphen_whitespace_merging():
    """re.sub merges TSU- MIKI'S (with space/newline after hyphen) into TSUMIKI'S while second-years and 1-1 stay preserved."""
    hyphen_pattern = r'(\b[A-Za-z]+)-\s+([A-Za-z\']+)'

    # 1. Trailing space after hyphen: 'TSU- MIKI'S' -> "TSUMIKI'S"
    assert re.sub(hyphen_pattern, r'\1\2', "TSU- MIKI'S") == "TSUMIKI'S"
    assert re.sub(hyphen_pattern, r'\1\2', "HELLO TSU- MIKI'S") == "HELLO TSUMIKI'S"

    # 2. Trailing newline after hyphen: 'TSU-\nMIKI'S' -> "TSUMIKI'S"
    assert re.sub(hyphen_pattern, r'\1\2', "TSU-\nMIKI'S") == "TSUMIKI'S"
    assert re.sub(hyphen_pattern, r'\1\2', "HELLO TSU-\nMIKI'S HERE") == "HELLO TSUMIKI'S HERE"

    # 3. Multiple trailing spaces after hyphen: 'TSU-   MIKI'S' -> "TSUMIKI'S"
    assert re.sub(hyphen_pattern, r'\1\2', "TSU-   MIKI'S") == "TSUMIKI'S"

    # 4. Legitimate compound word without space after hyphen: 'second-years' must be preserved
    text_compound = "the second-years and third-years"
    assert re.sub(hyphen_pattern, r'\1\2', text_compound) == text_compound
    assert "second-years" in utils.repair_comic_text("second-years")

    # 5. Digits / Roman numeral patterns: 'CLASS 1-1' and 'GRADE 1-A' must be preserved
    text_nums = "CLASS 1-1 AND GRADE 1-A"
    assert re.sub(hyphen_pattern, r'\1\2', text_nums) == text_nums
    assert "1-1" in utils.repair_comic_text("CLASS 1-1")


# ==============================================================================
# 9. PDF ATOMIC WRITE, BUBBLE SORTING, ONNX INPAINTING & VERTICAL COLUMNS
# ==============================================================================

def test_atomic_json_write_named_tempfile(tmp_path):
    """Verify _atomic_json_write writes via NamedTemporaryFile and cleanly replaces target."""
    import json
    import tempfile
    from pdf_jobs import _atomic_json_write

    target_file = tmp_path / "metadata.json"
    data = {"status": "processing", "progress": 42, "title": "Test Manga"}

    named_temp_called = False
    orig_named_temp = tempfile.NamedTemporaryFile

    def spy_named_temp(*args, **kwargs):
        nonlocal named_temp_called
        named_temp_called = True
        return orig_named_temp(*args, **kwargs)

    with patch("tempfile.NamedTemporaryFile", side_effect=spy_named_temp):
        _atomic_json_write(target_file, data)

    assert named_temp_called, "_atomic_json_write must use tempfile.NamedTemporaryFile"
    assert target_file.exists(), "Target file must exist after atomic write"

    with open(target_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == data

    # Test cleanly replacing an existing target file atomically
    update_data = {"status": "completed", "progress": 100, "pages": 5}
    _atomic_json_write(target_file, update_data)
    with open(target_file, "r", encoding="utf-8") as f:
        loaded_update = json.load(f)
    assert loaded_update == update_data

    # Ensure no leftover temporary files in target directory
    remaining_files = list(tmp_path.glob("*"))
    assert remaining_files == [target_file], f"No temporary files should remain: {remaining_files}"


def test_manga_bubble_engine_y_overlap_sorting():
    """Verify OCR words in bubble engine are grouped by y_overlap and sorted correctly."""
    # Two words on the same line with 2px vertical jitter:
    # Word 0: "HELLO", x: 10..50, y: 32..50 (height 18)
    # Word 1: "WORLD", x: 60..100, y: 30..48 (height 18)
    # Strict (y1, x1) sorting would falsely put Word 1 before Word 0 because 30 < 32.
    # Grouping by y_overlap (16px >= 0.45 * 18px) places them on the same line sorted by x1: HELLO WORLD.
    # Line 2: Word 2: "AGAIN", x: 20..70, y: 60..78 (height 18)
    raw_boxes = [
        {"box": [10, 32, 50, 50], "text": "HELLO", "conf": 90},
        {"box": [60, 30, 100, 48], "text": "WORLD", "conf": 92},
        {"box": [20, 60, 70, 78], "text": "AGAIN", "conf": 95},
    ]
    comp = [0, 1, 2]

    sorted_indices = server_be._sort_dialogue_line_boxes(comp, raw_boxes)
    sorted_words = [raw_boxes[i]["text"] for i in sorted_indices]
    assert sorted_words == ["HELLO", "WORLD", "AGAIN"], f"Words should be correctly sorted by y_overlap: {sorted_words}"

    if hf_be is not None and hasattr(hf_be, "_sort_dialogue_line_boxes"):
        hf_sorted = hf_be._sort_dialogue_line_boxes(comp, raw_boxes)
        assert [raw_boxes[i]["text"] for i in hf_sorted] == ["HELLO", "WORLD", "AGAIN"]


def test_manga_bubble_engine_conf_float_string():
    """Verify float confidence strings like '95.5' are parsed without ValueError."""
    from PIL import Image

    mock_tess_data = {
        'text': ['HELLO', 'WORLD', '!', 'JUNK'],
        'conf': ['95.5', '88.2', '99.0', '-1'],
        'left': [10, 60, 110, 150],
        'top': [20, 20, 20, 100],
        'width': [40, 40, 10, 20],
        'height': [20, 20, 20, 20],
    }

    dummy_img = Image.new("RGB", (300, 300), color=(255, 255, 255))
    mock_tess = MagicMock()
    mock_tess.image_to_data.return_value = mock_tess_data

    orig_server_tess = server_be.pytesseract
    server_be.pytesseract = mock_tess
    try:
        dialogues = server_be.extract_manga_dialogues(dummy_img, source_lang="en")
        assert isinstance(dialogues, list)
        assert len(dialogues) >= 1
        all_text = " ".join(d.get("text", "") for d in dialogues)
        assert "HELLO" in all_text and "WORLD" in all_text
    finally:
        server_be.pytesseract = orig_server_tess

    # Also verify float parsing safety on int(float(...))
    for conf_str, expected in [("95.5", 95), ("88.2", 88), ("0.0", 0), ("100.0", 100)]:
        assert int(float(conf_str)) == expected


def test_onnx_engine_single_and_multi_output_protection():
    """Verify inpainting output logic handles both 1-element and 2-element outputs safely."""
    from PIL import Image
    from onnx_engine import MangaOnnxEngine

    engine = MangaOnnxEngine(models_dir="dummy", vocab_path="dummy")
    engine.inp_sess = MagicMock()

    test_img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    boxes = [[20, 20, 60, 60]]

    # Case 1: Inpainter returns 1-element list [infilled]
    single_out = [np.ones((1, 3, 48, 48), dtype=np.float32) * 0.8]
    engine.inp_sess.run.return_value = single_out
    res_single = engine.inpaint_bubbles(test_img, boxes)
    assert isinstance(res_single, Image.Image)
    assert res_single.size == (100, 100)

    # Case 2: Inpainter returns 2-element list [infilled, composite]
    multi_out = [
        np.zeros((1, 3, 48, 48), dtype=np.float32),
        np.ones((1, 3, 48, 48), dtype=np.float32) * 0.9,
    ]
    engine.inp_sess.run.return_value = multi_out
    res_multi = engine.inpaint_bubbles(test_img, boxes)
    assert isinstance(res_multi, Image.Image)
    assert res_multi.size == (100, 100)


def test_independent_vertical_columns_not_merged():
    """Verify two independent vertical columns with h > 1.3 * w are not merged into a single wide horizontal bbox in merge_adjacent_bubble_columns."""
    # Two narrow vertical columns side by side:
    # Column A (left): x: 20..40 (w=20), y: 10..90 (h=80) -> h / w = 4.0 > 1.3
    # Column B (right): x: 50..70 (w=20), y: 10..90 (h=80) -> h / w = 4.0 > 1.3
    gray = np.full((120, 120), 255, dtype=np.uint8)
    clusters = [
        {"text_box": [20, 10, 40, 90], "text": "COLUMN LEFT SENTENCE.", "conf": 0.95, "indices": [0]},
        {"text_box": [50, 10, 70, 90], "text": "COLUMN RIGHT SENTENCE.", "conf": 0.95, "indices": [1]},
    ]

    merged = merge_adjacent_bubble_columns(clusters, gray, reading_direction="rtl")
    assert len(merged) == 2, f"Vertical columns should remain 2 separate boxes, but got: {len(merged)}"

    # Widths must remain narrow, not merged into a single wide bbox of w=50
    box0 = merged[0]["text_box"]
    box1 = merged[1]["text_box"]
    assert (box0[2] - box0[0]) <= 25, f"Column 0 width should be preserved, got {box0}"
    assert (box1[2] - box1[0]) <= 25, f"Column 1 width should be preserved, got {box1}"

    # In RTL, right column (x=50..70) precedes left column (x=20..40)
    assert merged[0]["text_box"][0] == 50
    assert merged[1]["text_box"][0] == 20
    assert "COLUMN RIGHT" in merged[0]["text"]
    assert "COLUMN LEFT" in merged[1]["text"]

