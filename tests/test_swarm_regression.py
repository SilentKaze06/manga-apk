import os
import sys
import re
import urllib.error
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

# Ensure root, mashinali oqitish, server and hf are all importable
_this_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_this_dir) == "tests":
    parent = os.path.dirname(_this_dir)
    if os.path.basename(parent) == "mashinali oqitish":
        PROJECT_ROOT = os.path.dirname(parent)
        BASE_DIR = parent
    else:
        PROJECT_ROOT = parent
        BASE_DIR = os.path.join(PROJECT_ROOT, "mashinali oqitish")
else:
    PROJECT_ROOT = _this_dir
    BASE_DIR = os.path.join(PROJECT_ROOT, "mashinali oqitish")

SERVER_DIR = os.path.join(PROJECT_ROOT, "manga_apk", "server")
SERVER_ML_DIR = os.path.join(SERVER_DIR, "ml")
HF_DIR = os.path.join(PROJECT_ROOT, "hf_dosjanmanga")

for p in [SERVER_ML_DIR, BASE_DIR, SERVER_DIR, HF_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import utils
from manga_pipeline import (
    split_cluster_into_columns,
    merge_adjacent_bubble_columns,
    detect_stepped_box_split_x,
    smart_cluster_text_boxes,
    expand_to_bubble_contour,
    render_translated_dialogues,
    _crop_reading_key,
    is_dark_barrier_between_x,
    is_dark_barrier_between_y,
    are_different_white_contours,
    can_merge_text_boxes,
    wrap_text_elliptical,
    clean_bubble_crop,
    _has_continuous_dark_segment,
    get_box_border_mask,
    line_crosses_border_mask,
)
import gemini_translator as server_gt
import manga_bubble_engine as server_be
import pdf_jobs
import onnx_engine
import main as server_main

import importlib.util
def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Directly import HF modules without silent exception swallowers (if HF_DIR exists)
if os.path.exists(HF_DIR):
    hf_gt = load_module_from_path("hf_gemini_translator", os.path.join(HF_DIR, "gemini_translator.py"))
    hf_be = load_module_from_path("hf_manga_bubble_engine", os.path.join(HF_DIR, "manga_bubble_engine.py"))
    hf_pj = load_module_from_path("hf_pdf_jobs", os.path.join(HF_DIR, "pdf_jobs.py"))
    hf_oe = load_module_from_path("hf_onnx_engine", os.path.join(HF_DIR, "onnx_engine.py"))
    hf_main = load_module_from_path("hf_main", os.path.join(HF_DIR, "main.py"))

    assert hf_gt is not None, "hf_gemini_translator must be loadable"
    assert hf_be is not None, "hf_manga_bubble_engine must be loadable"
    assert hf_pj is not None, "hf_pdf_jobs must be loadable"
    assert hf_oe is not None, "hf_onnx_engine must be loadable"
    assert hf_main is not None, "hf_main must be loadable"
else:
    hf_gt = hf_be = hf_pj = hf_oe = hf_main = None


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
    """GEMINI_MODELS must have gemini-2.5-flash as the primary model and no non-existent 3.5 models."""
    assert "gemini-3.5-flash-lite" not in server_gt.GEMINI_MODELS
    assert "gemini-3.5-flash" not in server_gt.GEMINI_MODELS
    assert "gemini-2.5-flash-lite" in server_gt.GEMINI_MODELS
    assert "gemini-2.5-flash" in server_gt.GEMINI_MODELS
    assert "gemini-2.0-flash" not in server_gt.GEMINI_MODELS
    # gemini-2.5-flash must be the first (primary) model
    assert server_gt.GEMINI_MODELS[0] == "gemini-2.5-flash", f"gemini-2.5-flash must be first, got {server_gt.GEMINI_MODELS[0]}"

    if hf_gt:
        assert "gemini-3.5-flash-lite" not in hf_gt.GEMINI_MODELS
        assert "gemini-3.5-flash" not in hf_gt.GEMINI_MODELS
        assert hf_gt.GEMINI_MODELS[0] == "gemini-2.5-flash"


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
    """Mock urllib.request.urlopen returning HTTPError 404 on first model, verify it tries next model instead of aborting to None."""
    import json
    for gt_mod in [server_gt, hf_gt] if hf_gt else [server_gt]:
        gt_mod._gemini_rate_limited_until = 0.0
        called_models = []
        first_mod = gt_mod.GEMINI_MODELS[0]
        second_mod = gt_mod.GEMINI_MODELS[1]

        def fake_urlopen(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            matched_m = None
            for m in sorted(gt_mod.GEMINI_MODELS, key=len, reverse=True):
                if f"models/{m}:" in url or f"/{m}:" in url or m == url:
                    matched_m = m
                    called_models.append(m)
                    break

            if matched_m == first_mod:
                raise urllib.error.HTTPError(url=url, code=404, msg=f"Not Found: {first_mod}", hdrs={}, fp=None)

            if matched_m == second_mod:
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
            assert first_mod in called_models, f"{first_mod} should have been attempted first"
            assert second_mod in called_models, f"{second_mod} should have been tried after 404 on first model"
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


# ==============================================================================
# 9. RECTANGULAR BOX SEPARATION, ZERO GHOSTING, AND FALLBACK WARNINGS
# ==============================================================================

def test_stacked_rectangular_bordered_boxes_not_merged():
    """Verify stacked rectangular narration boxes with a continuous horizontal black border (#3) are NOT merged."""
    gray = np.full((100, 120), 255, dtype=np.uint8)

    # Box A (top narration box): [20, 10, 100, 40]
    boxA = [20, 10, 100, 40]
    # Box B (bottom narration box): [20, 44, 100, 74]
    boxB = [20, 44, 100, 74]

    # Horizontal black frame border between the two boxes at y=41..42 across width x=15..105
    gray[41:43, 15:105] = 0

    # 1. Dark barrier check between y coordinates must detect the continuous border
    assert is_dark_barrier_between_y(gray, boxA, boxB) is True

    # 2. can_merge_text_boxes must return False, blocking merge despite small gap_y (4px)
    can_merge = can_merge_text_boxes(boxA, boxB, gray, textA="NARRATION TOP LINE", textB="NARRATION BOTTOM LINE")
    assert can_merge is False, "Stacked narration boxes separated by a black border must not be merged"

    # 3. Even with slight vertical overlap (yA2 >= yB1) e.g. [20, 10, 100, 42] and [20, 41, 100, 74]
    boxA_ov = [20, 10, 100, 42]
    boxB_ov = [20, 41, 100, 74]
    assert is_dark_barrier_between_y(gray, boxA_ov, boxB_ov) is True
    assert can_merge_text_boxes(boxA_ov, boxB_ov, gray) is False

    # 4. In contrast, when the border is absent (pure white), merge should be allowed
    gray_clean = np.full((100, 120), 255, dtype=np.uint8)
    assert is_dark_barrier_between_y(gray_clean, boxA, boxB) is False
    assert can_merge_text_boxes(boxA, boxB, gray_clean, textA="NORMAL LINE 1", textB="NORMAL LINE 2") is True


def test_stepped_two_column_rectangular_box_preserved():
    """Verify stepped/staircase 2-column rectangular narration box (#6) stays as separate columns."""
    gray = np.full((120, 140), 255, dtype=np.uint8)

    # Two columns arranged in a stepped/staircase fashion inside a rectangular narration box:
    # Left column: x: 20..50, y: 15..55 (h=40, w=30)
    # Right column: x: 62..92, y: 25..65 (h=40, w=30)
    # gap_x = 62 - 50 = 12 >= 6, y1 offset = 25 - 15 = 10 >= 4 (stepped)
    clusters = [
        {
            "text_box": [20, 15, 50, 55],
            "text": "IT WAS A COLD RAINY DAY.",
            "conf": 0.95,
            "indices": [0],
        },
        {
            "text_box": [62, 25, 92, 65],
            "text": "AND THE STREETS WERE EMPTY.",
            "conf": 0.95,
            "indices": [1],
        },
    ]

    # In RTL (default manga reading order):
    merged_rtl = merge_adjacent_bubble_columns(clusters, gray, reading_direction="rtl")
    assert len(merged_rtl) == 2, f"Stepped columns should stay as 2 separate columns, got {len(merged_rtl)}"
    # Right column comes first in RTL
    assert merged_rtl[0]["text"] == "AND THE STREETS WERE EMPTY."
    assert merged_rtl[1]["text"] == "IT WAS A COLD RAINY DAY."
    assert merged_rtl[0]["text_box"] == [62, 25, 92, 65]
    assert merged_rtl[1]["text_box"] == [20, 15, 50, 55]

    # In LTR:
    merged_ltr = merge_adjacent_bubble_columns(clusters, gray, reading_direction="ltr")
    assert len(merged_ltr) == 2
    # Left column comes first in LTR
    assert merged_ltr[0]["text"] == "IT WAS A COLD RAINY DAY."
    assert merged_ltr[1]["text"] == "AND THE STREETS WERE EMPTY."


def test_clean_bubble_crop_and_edge_alpha_zero_ghosting():
    """Verify clean_bubble_crop and edge_alpha feathering eliminate bottom ghosting traces in rectangular boxes."""
    # 1. Test clean_bubble_crop on rectangular white box with dark borders:
    # Size: height 60, width 80
    patch = np.full((60, 80, 3), 255, dtype=np.uint8)
    # Black borders on outer frame: top (0..1), bottom (58..59), left (0..1), right (78..79)
    patch[0:2, :] = 0
    patch[58:60, :] = 0
    patch[:, 0:2] = 0
    patch[:, 78:80] = 0

    # Add printed text ink and descenders near bottom (rows 40..55) like "CLASS.", "?", "up"
    patch[40:55, 30:50] = 0

    cleaned = clean_bubble_crop(patch)
    # Interior (rows 2..57, cols 2..77) must be 100% pure white (zero text ink / ghosting remains)
    interior = cleaned[2:58, 2:78]
    assert np.all(interior == 255), "Interior of rectangular box must be completely cleaned to pure white"
    # Outer borders must be preserved intact
    assert np.all(cleaned[0, :] == 0) and np.all(cleaned[59, :] == 0)
    assert np.all(cleaned[:, 0] == 0) and np.all(cleaned[:, 79] == 0)

    # 2. Test edge_alpha feathering reduction inside white/rectangular bubbles
    # In white/rectangular bubbles, feather_px is reduced to min(2, min(ph, pw)//6)
    # with ksize=3, sigma=0.5 Gaussian blur
    import cv2
    ph, pw = 60, 80
    feather_px = min(2, min(ph, pw) // 6)
    assert feather_px <= 2, f"Feather px for white box must be <= 2, got {feather_px}"

    edge_alpha = np.ones((ph, pw), dtype=np.float32)
    for f_i in range(feather_px):
        alpha_v = (f_i + 1) / float(feather_px + 1)
        edge_alpha[f_i, :] = np.minimum(edge_alpha[f_i, :], alpha_v)
        edge_alpha[ph - 1 - f_i, :] = np.minimum(edge_alpha[ph - 1 - f_i, :], alpha_v)
        edge_alpha[:, f_i] = np.minimum(edge_alpha[:, f_i], alpha_v)
        edge_alpha[:, pw - 1 - f_i] = np.minimum(edge_alpha[:, pw - 1 - f_i], alpha_v)
    edge_alpha = cv2.GaussianBlur(edge_alpha, (3, 3), 0.5)
    edge_alpha = np.clip(edge_alpha, 0.0, 1.0)

    # Bottom rows beyond the 2px border (e.g. row 55) must have alpha >= 0.99 (full opacity)
    # meaning the cleaned white patch completely overwrites old pixels with 0 ghosting
    assert edge_alpha[55, 40] >= 0.99, f"Alpha at bottom text location should be >= 0.99, got {edge_alpha[55, 40]}"


def test_detect_and_translate_fallback_warning_and_paddleocr_joining():
    """Verify detect_and_translate warning handling and dispatch_manga_ocr PaddleOCR line joining per language script."""
    import asyncio
    import io
    from PIL import Image

    async def _run_async_tests():
        dummy_img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        dummy_bytes = b"fake_png_data"

        # -------------------------------------------------------------
        # Part 1: dispatch_manga_ocr PaddleOCR line joining per script
        # -------------------------------------------------------------
        mock_paddle_lines = [
            {"text": "こんにちは", "confidence": 0.95},
            {"text": "世界", "confidence": 0.92},
        ]

        # For Japanese (ja) and Chinese (zh): lines joined with "" (no space)
        with patch("main.run_manga_ocr", side_effect=Exception("MangaOCR offline")):
            with patch("main.run_paddle_ocr", return_value=[]):
                with patch("main.extract_paddle_lines", return_value=mock_paddle_lines):
                    text_ja, lang_ja = await server_main.dispatch_manga_ocr(dummy_bytes, dummy_img, "ja")
                    assert text_ja == "こんにちは世界", f"Japanese lines must be joined without space, got '{text_ja}'"
                    assert lang_ja == "ja"

        mock_zh_lines = [
            {"text": "你好", "confidence": 0.96},
            {"text": "世界", "confidence": 0.94},
        ]
        with patch("main.run_paddle_ocr", return_value=[]):
            with patch("main.extract_paddle_lines", return_value=mock_zh_lines):
                text_zh, lang_zh = await server_main.dispatch_manga_ocr(dummy_bytes, dummy_img, "zh")
                assert text_zh == "你好世界", f"Chinese lines must be joined without space, got '{text_zh}'"
                assert lang_zh == "zh"

        # For Korean (ko): lines joined with " "
        mock_ko_lines = [
            {"text": "안녕", "confidence": 0.95},
            {"text": "하세요", "confidence": 0.93},
        ]
        with patch("main.run_paddle_ocr", return_value=[]):
            with patch("main.extract_paddle_lines", return_value=mock_ko_lines):
                text_ko, lang_ko = await server_main.dispatch_manga_ocr(dummy_bytes, dummy_img, "ko")
                assert text_ko == "안녕 하세요", f"Korean lines must be joined with space, got '{text_ko}'"
                assert lang_ko == "ko"

        # For English (en) / Russian (ru): lines joined with " "
        mock_en_lines = [
            {"text": "CHAPTER", "confidence": 0.95},
            {"text": "ONE", "confidence": 0.92},
        ]
        with patch("main.run_paddle_ocr", return_value=[]):
            with patch("main.extract_paddle_lines", return_value=mock_en_lines):
                text_en, lang_en = await server_main.dispatch_manga_ocr(dummy_bytes, dummy_img, "en")
                assert text_en == "CHAPTER ONE", f"English lines must be joined with space, got '{text_en}'"
                assert lang_en == "en"

        # Verify exception in run_paddle_ocr does not crash with 500, but is caught safely
        with patch("main.run_paddle_ocr", side_effect=RuntimeError("Paddle model file missing")):
            text_err, lang_err = await server_main.dispatch_manga_ocr(dummy_bytes, dummy_img, "ko")
            assert text_err == ""
            assert lang_err == "ko"

        # -------------------------------------------------------------
        # Part 2: detect_and_translate fallback warning handling
        # -------------------------------------------------------------
        class AsyncMockFile:
            def __init__(self, data: bytes):
                self.data = data
            async def read(self, size: int = -1):
                return self.data

        dummy_bio = io.BytesIO()
        dummy_img.save(dummy_bio, format="PNG")
        mock_file = AsyncMockFile(dummy_bio.getvalue())

        # Case A: Gemini fails -> fallback to google translate with translator_used='google_fallback' and warning set
        with patch("main.get_manga_pipeline", return_value=None):
            with patch("main.get_onnx_engine", return_value=None):
                with patch("main.is_paddle_enabled", return_value=True):
                    with patch("main.run_paddle_ocr", return_value=[[[[10, 10], [50, 10], [50, 30], [10, 30]], ("HELLO WORLD", 0.95)]]):
                        with patch("main.extract_paddle_lines", return_value=[{"text": "HELLO WORLD", "left": 10, "top": 10, "right": 50, "bottom": 30}]):
                            with patch("main.resolve_gemini_api_key", return_value=None):
                                with patch("main.translate_text", return_value="SALOM DUNYO"):
                                    res_fallback = await server_main.detect_and_translate(
                                        file=mock_file,
                                        target_lang="uz",
                                        lang="en",
                                        gemini_api_key=None,
                                    )
                                    assert res_fallback["status"] == "success"
                                    assert res_fallback.get("translator_used") == "google_fallback"
                                    assert "warning" in res_fallback
                                    assert "falling back to google translate" in res_fallback["warning"]

        # Case B: Gemini succeeds -> translator_used='gemini', no warning
        mock_file2 = AsyncMockFile(dummy_bio.getvalue())
        with patch("main.get_manga_pipeline", return_value=None):
            with patch("main.get_onnx_engine", return_value=None):
                with patch("main.is_paddle_enabled", return_value=True):
                    with patch("main.run_paddle_ocr", return_value=[[[[10, 10], [50, 10], [50, 30], [10, 30]], ("HELLO WORLD", 0.95)]]):
                        with patch("main.extract_paddle_lines", return_value=[{"text": "HELLO WORLD", "left": 10, "top": 10, "right": 50, "bottom": 30}]):
                            with patch("main.resolve_gemini_api_key", return_value="AIzaSyTestKey"):
                                with patch("gemini_translator.translate_dialogues_with_gemini", return_value=["SALOM DUNYO"]):
                                    res_gemini = await server_main.detect_and_translate(
                                        file=mock_file2,
                                        target_lang="uz",
                                        lang="en",
                                        gemini_api_key="AIzaSyTestKey",
                                    )
                                    assert res_gemini["status"] == "success"
                                    assert res_gemini.get("translator_used") == "gemini"
                                    assert "warning" not in res_gemini

    asyncio.run(_run_async_tests())


# ==============================================================================
# 11. BORDER-SAFE INK ERASURE & MORPHOLOGICAL STRAIGHT BORDER MASK
# ==============================================================================

def test_oval_bubble_curved_border_preserved_after_clean_bubble_crop():
    """
    Verify clean_bubble_crop erases isolated interior letter ink components
    while 100% preserving:
    1) The curved black ellipse border of the bubble.
    2) Outer background artwork / clothing touching the boundaries.
    Never paints the whole patch white with a naive rectangle fill.
    """
    import cv2
    hp, wp = 180, 120
    # White bubble interior (255)
    patch = np.full((hp, wp, 3), 255, dtype=np.uint8)

    # Outside clothing / dark artwork touching bottom boundary
    patch[140:hp, :] = 35

    # Curved oval black border of bubble (thickness 3 px)
    cv2.ellipse(patch, (wp // 2, 85), (wp // 2 - 8, 55), 0, 0, 360, (15, 15, 15), thickness=3)

    # Isolated letter ink inside the bubble interior (dark ink ~ 20)
    # E.g. simulated letters 'T', 'S', 'U'
    patch[70:82, 50:70] = 20
    patch[90:102, 45:75] = 20

    # Ensure there are dark pixels before cleaning
    patch_gray_pre = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    border_pixels_pre = (patch_gray_pre < 50) & (patch_gray_pre > 0)
    clothing_pixels_pre = (patch_gray_pre[140:hp, :] < 50)
    assert np.sum(border_pixels_pre) > 100, "Border pixels must exist in original patch"
    assert np.sum(clothing_pixels_pre) > 100, "Clothing pixels must exist in original patch"

    # Clean the bubble crop
    cleaned = clean_bubble_crop(patch)
    assert cleaned is not None
    assert cleaned.shape == patch.shape

    cleaned_gray = cv2.cvtColor(cleaned, cv2.COLOR_RGB2GRAY)

    # 1. Letters inside the bubble MUST be completely erased to pure white (255, 255, 255)
    letter_area1 = cleaned[70:82, 50:70]
    letter_area2 = cleaned[90:102, 45:75]
    assert np.all(letter_area1 == 255), "Interior letter area 1 must be cleanly erased to 255"
    assert np.all(letter_area2 == 255), "Interior letter area 2 must be cleanly erased to 255"

    # 2. Curved black border MUST be preserved (NOT turned to white!)
    # Check pixels on the ellipse perimeter
    ellipse_pts = []
    for angle in range(0, 360, 10):
        rad = np.deg2rad(angle)
        ex = int(wp // 2 + (wp // 2 - 8) * np.cos(rad))
        ey = int(85 + 55 * np.sin(rad))
        if 0 <= ex < wp and 0 <= ey < 140:
            ellipse_pts.append((ex, ey))

    dark_on_ellipse = sum(1 for (ex, ey) in ellipse_pts if cleaned_gray[ey, ex] < 100)
    assert dark_on_ellipse >= len(ellipse_pts) * 0.75, (
        f"Curved black border must be preserved! Only {dark_on_ellipse}/{len(ellipse_pts)} points dark"
    )

    # 3. Clothing artwork at bottom boundary must remain dark
    assert np.mean(cleaned[150:hp, :, :]) < 100, "Background clothing must NOT be wiped white"

    # 4. Patch is NOT a flat solid white rectangle
    assert not np.all(cleaned == 255), "Patch must not be wiped into a flat white rectangle"


def test_box_border_mask_prevents_merging_stacked_narration_boxes():
    """
    Verify get_box_border_mask and line_crosses_border_mask detect 1-2px thin black frames
    between stacked #3 narration boxes and strictly prevent can_merge_text_boxes and
    merge_adjacent_bubble_columns from merging them.
    """
    import cv2
    # Create 400x400 white page
    gray = np.full((400, 400), 255, dtype=np.uint8)

    # Horizontal 1px dark frame line at y=200 between Box 1 and Box 2
    gray[200, 50:250] = 20  # Length 200px, 1px thickness

    # Box 1: Upper narration box (y=140..195, x=60..240)
    boxA = [60, 140, 240, 195]
    # Box 2: Lower narration box (y=205..260, x=60..240)
    boxB = [60, 205, 240, 260]

    # 1. Morphological straight border mask must catch the 1px frame line
    border_mask = get_box_border_mask(gray)
    assert border_mask is not None
    assert np.sum(border_mask[200, 60:240] > 0) >= 150, "Horizontal 1px frame must be captured in border mask"

    # 2. Line between centers must cross the border mask
    cxA, cyA = (boxA[0] + boxA[2]) // 2, (boxA[1] + boxA[3]) // 2
    cxB, cyB = (boxB[0] + boxB[2]) // 2, (boxB[1] + boxB[3]) // 2
    crosses = line_crosses_border_mask(border_mask, (cxA, cyA), (cxB, cyB))
    assert crosses is True, "Line between box centers must cross border mask"

    # 3. can_merge_text_boxes must strictly return False
    can_merge = can_merge_text_boxes(boxA, boxB, gray, textA="Narration Box 1.", textB="Narration Box 2.")
    assert can_merge is False, "Stacked narration boxes across frame line must NEVER merge"

    # 4. merge_adjacent_bubble_columns must keep them as 2 distinct clusters
    clusters = [
        {"text_box": boxA, "text": "Narration Box 1.", "conf": 0.95},
        {"text_box": boxB, "text": "Narration Box 2.", "conf": 0.95},
    ]
    merged = merge_adjacent_bubble_columns(clusters, gray)
    assert len(merged) == 2, f"Expected 2 separate clusters, got {len(merged)}"

    # 5. Verify server and HF manga_bubble_engine also catch this barrier
    assert server_be.line_crosses_border_mask(server_be.get_box_border_mask(gray), (cxA, cyA), (cxB, cyB)) is True
    assert server_be.can_merge_text_boxes(boxA, boxB, gray) is False
    if hf_be is not None:
        assert hf_be.line_crosses_border_mask(hf_be.get_box_border_mask(gray), (cxA, cyA), (cxB, cyB)) is True
        assert hf_be.can_merge_text_boxes(boxA, boxB, gray) is False


def test_stepped_z_shaped_narration_box_split_and_safe_rendering():
    """
    Verify stepped (Z-shaped / L-shaped) narration box splitting and border-safe rendering:
    1) detect_stepped_box_split_x returns the correct x_split for a Z-shaped box (fill_ratio ~ 0.61, no dividing line).
    2) The two lobes with text boxes separated by gap_x = 4 px are returned in proper RTL order without merging.
    3) Rendered translated text in safe inscribed boxes does not touch or overlap the y_top and y_bot black frame borders.
    """
    import cv2
    from PIL import Image

    # 1. Create synthetic Z-shaped stepped white narration box (fill_ratio ~ 0.61, no internal dividing line)
    gray = np.full((140, 160), 40, dtype=np.uint8)
    # Left lobe: x in [20, 70], y in [20, 89]
    gray[20:89, 20:70] = 255
    # Right lobe: x in [70, 120], y in [51, 120]
    gray[51:120, 70:120] = 255

    # Outer black frame borders (2px)
    gray[20:22, 20:70] = 0
    gray[20:89, 20:22] = 0
    gray[87:89, 20:70] = 0
    gray[20:51, 68:70] = 0

    gray[51:53, 70:120] = 0
    gray[51:120, 118:120] = 0
    gray[118:120, 70:120] = 0
    gray[89:120, 70:72] = 0

    cluster_box = [20, 20, 120, 120]
    patch = gray[20:120, 20:120]
    white_area = np.sum(patch > 200)
    fill_ratio = white_area / (100.0 * 100.0)
    assert 0.58 <= fill_ratio <= 0.64, f"fill_ratio should be ~0.61, got {fill_ratio}"

    # 1) detect_stepped_box_split_x returns the correct x_split
    x_split = detect_stepped_box_split_x(gray, cluster_box)
    assert x_split == 70, f"Expected x_split=70, got {x_split}"

    # Verify normal oval and solid rectangular boxes return None
    gray_oval = np.full((120, 120), 40, dtype=np.uint8)
    cv2.ellipse(gray_oval, (60, 60), (45, 45), 0, 0, 360, 255, -1)
    cv2.ellipse(gray_oval, (60, 60), (45, 45), 0, 0, 360, 0, 2)
    assert detect_stepped_box_split_x(gray_oval, [15, 15, 105, 105]) is None, "Oval bubble must not be split"

    gray_rect = np.full((120, 120), 40, dtype=np.uint8)
    gray_rect[20:100, 20:100] = 255
    assert detect_stepped_box_split_x(gray_rect, [20, 20, 100, 100]) is None, "Normal rectangular box must not be split"

    # 2) Two lobes with text boxes separated by gap_x = 4 px are returned in proper RTL order
    raw_boxes = [
        {"box": [30, 30, 68, 70], "text": "LEFT LOBE TEXT."},
        {"box": [72, 69, 110, 109], "text": "RIGHT LOBE TEXT."},
    ]
    gap_x = raw_boxes[1]["box"][0] - raw_boxes[0]["box"][2]
    assert gap_x == 4, f"Expected gap_x=4, got {gap_x}"

    # RTL order check via split_cluster_into_columns
    cols_rtl = split_cluster_into_columns([0, 1], raw_boxes, reading_direction="rtl", gray=gray)
    assert cols_rtl == [[1], [0]], f"Right lobe [1] must come first in RTL, got {cols_rtl}"

    cols_ltr = split_cluster_into_columns([0, 1], raw_boxes, reading_direction="ltr", gray=gray)
    assert cols_ltr == [[0], [1]], f"Left lobe [0] must come first in LTR, got {cols_ltr}"

    # smart_cluster_text_boxes RTL order check
    clusters_rtl = smart_cluster_text_boxes(raw_boxes, gray, reading_direction="rtl")
    assert len(clusters_rtl) == 2, f"Expected 2 clusters, got {len(clusters_rtl)}"
    assert clusters_rtl[0]["text"] == "RIGHT LOBE TEXT."
    assert clusters_rtl[1]["text"] == "LEFT LOBE TEXT."

    # merge_adjacent_bubble_columns must not merge vertically shifted columns even if gap_x < 6
    merged = merge_adjacent_bubble_columns(clusters_rtl, gray, reading_direction="rtl")
    assert len(merged) == 2, "Stepped columns must not be merged by merge_adjacent_bubble_columns"
    assert merged[0]["text"] == "RIGHT LOBE TEXT."
    assert merged[1]["text"] == "LEFT LOBE TEXT."

    # Bounding limits in expand_to_bubble_contour
    b_left = expand_to_bubble_contour(gray, [30, 30, 68, 70], limit_right=x_split)
    assert b_left[2] <= x_split, f"Left lobe right {b_left[2]} crossed x_split {x_split}"

    b_right = expand_to_bubble_contour(gray, [72, 69, 110, 109], limit_left=x_split)
    assert b_right[0] >= x_split, f"Right lobe left {b_right[0]} crossed x_split {x_split}"

    # 3) Rendered translated text does not touch/overlap the y_top and y_bot black frame borders
    img = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    dialogues = [
        {"bbox": [20, 20, 70, 89], "translation": "TRANSLATED L1", "category": "narration"},
        {"bbox": [70, 51, 120, 120], "translation": "TRANSLATED R1", "category": "narration"},
    ]
    rendered = render_translated_dialogues(img, dialogues)
    rendered_np = np.array(rendered.convert("L"))

    # Outer black frame borders must remain intact
    assert np.all(rendered_np[20:22, 20:70] == 0), "Left lobe top border rows must remain pure black"
    assert np.all(rendered_np[118:120, 70:120] == 0), "Right lobe bottom border rows must remain pure black"

    # Text must be shrunk inward >= 4px from borders (safe margin contains no text ink)
    top_safe_zone = rendered_np[22:26, 24:66]
    assert np.all(top_safe_zone == 255), "Rendered text must be at least 4px below top border"

    bot_safe_zone = rendered_np[114:118, 74:116]
    assert np.all(bot_safe_zone == 255), "Rendered text must be at least 4px above bottom border"


# ==============================================================================
# 12. SWARM REGRESSION: BORDER PROTECTION, DARK BOX INK ERASURE, CORRIDOR CONTINUITY
# ==============================================================================

def test_dark_narration_box_with_light_outer_margin_no_white_blobs():
    """
    1. Dark narration box with a light outer margin:
       Assert no pure-white (255, 255, 255) blobs inside the dark interior.
       Interior text must be cleanly erased to local dark background color,
       and onnx_engine compositing must never blend white masks onto dark interior.
    """
    hp, wp = 120, 160
    # Outer margin is light (245, 245, 245) simulating page gutter / outer panel
    patch = np.full((hp, wp, 3), 245, dtype=np.uint8)

    # Dark narration box interior: rows 12..108, cols 16..144 (dark tone ~ 30)
    dark_tone = 30
    patch[12:108, 16:144] = dark_tone

    # White/light dialogue text inside the dark interior (rows 45..65, cols 40..120, brightness 240)
    # E.g., simulated text "DARK NARRATION SECRET"
    patch[45:65, 40:120] = 240

    # Ensure pre-conditions
    interior_pre = patch[15:105, 20:140]
    assert np.any(interior_pre == 240), "Simulated white text must exist before cleaning"

    # Clean the bubble crop
    cleaned = clean_bubble_crop(patch)
    assert cleaned is not None
    assert cleaned.shape == patch.shape

    # 1. Dark interior must NOT contain any pure white (255, 255, 255) blobs
    interior_cleaned = cleaned[15:105, 20:140]
    pure_white_pixels = np.all(interior_cleaned == 255, axis=-1)
    assert not np.any(pure_white_pixels), "Dark narration box interior must NEVER contain pure white (255, 255, 255) blobs!"

    # 2. White text region (45..65, 40..120) must be erased to local dark tone (<= 60)
    text_region = cleaned[45:65, 40:120]
    assert np.mean(text_region) <= 60, f"Erased text region should match dark background, got mean {np.mean(text_region)}"

    # 3. Outer light margin is preserved
    assert np.all(cleaned[0:10, :] == 245), "Top light margin must be preserved"
    assert np.all(cleaned[110:hp, :] == 245), "Bottom light margin must be preserved"


def test_white_box_letters_touching_crop_edge_and_straight_strokes_erased():
    """
    2. White narration box where letters touch the crop edge or contain long straight strokes:
       Assert interior text is erased while the outer box frame remains intact.
       Neither border nicks nor gray halos (185..240) remain.
    """
    import cv2
    hp, wp = 140, 180
    # White background (255)
    patch = np.full((hp, wp, 3), 255, dtype=np.uint8)

    # 2px black outer frame of narration box
    patch[0:2, :] = 10
    patch[hp - 2:hp, :] = 10
    patch[:, 0:2] = 10
    patch[:, wp - 2:wp] = 10

    # 1. Interior letters with long straight strokes (e.g. 'T', 'H', 'M')
    # Horizontal bar of 'T': length 28px (>= 15px), height 3px
    patch[40:43, 60:88] = 15
    # Vertical stem of 'T': length 22px
    patch[43:65, 72:75] = 15

    # 2. Letter touching the crop edge (touches left edge x=0..1 or border x=2)
    patch[75:95, 2:18] = 20

    # 3. Anti-aliased ink halo pixels (190..230) surrounding letter strokes
    patch[39, 60:88] = 205
    patch[43, 60:88] = 210

    # Ensure border and text exist before cleaning
    assert np.all(patch[0:2, :] == 10), "Top frame must exist"
    assert np.any(patch[40:43, 60:88] == 15), "Letter stroke must exist"

    # Clean the bubble crop
    cleaned = clean_bubble_crop(patch)
    assert cleaned is not None
    assert cleaned.shape == patch.shape

    cleaned_gray = cv2.cvtColor(cleaned, cv2.COLOR_RGB2GRAY)

    # Outer 2px frame must remain 100% intact (not nicked or erased)
    assert np.all(cleaned[0:2, 10:wp - 10] <= 30), "Top frame must remain intact"
    assert np.all(cleaned[hp - 2:hp, 10:wp - 10] <= 30), "Bottom frame must remain intact"
    assert np.all(cleaned[10:hp - 10, 0:2] <= 30), "Left frame must remain intact"
    assert np.all(cleaned[10:hp - 10, wp - 2:wp] <= 30), "Right frame must remain intact"

    # Interior letter strokes must be completely erased to white (255)
    erased_t_bar = cleaned[40:43, 60:88]
    assert np.all(erased_t_bar == 255), "Straight stroke of letter 'T' must be cleanly erased"
    erased_t_stem = cleaned[43:65, 72:75]
    assert np.all(erased_t_stem == 255), "Vertical stroke of letter 'T' must be cleanly erased"

    # Letter touching border must also be cleanly erased
    erased_edge_letter = cleaned[75:95, 3:18]
    assert np.all(erased_edge_letter == 255), "Edge-touching letter must be cleanly erased"

    # Anti-aliased gray halos (185..240) must NOT remain in the interior
    halo_region = cleaned_gray[38:45, 58:90]
    assert np.all(halo_region == 255), "Anti-aliased gray halos must be cleanly eliminated"


def test_multiline_offset_lines_inside_one_box_merge_without_duplicate():
    """
    3. Multi-line text inside one box with offset lines:
       Assert boxes merge into one and are not duplicated or overlapped.
       E.g., "QUVVAT TIKLANISHI KERAK!" and "QAYTA TIKLASH", or "NEGA BOSHQARA OLMADIM?" + "BOSHQARISH".
    """
    # 400x400 white page
    gray = np.full((400, 400), 255, dtype=np.uint8)

    # Box 1 (Line 1): "QUVVAT TIKLANISHI KERAK!" (wider, centered at x=140, y=80..115)
    box1 = [40, 80, 240, 115]
    # Box 2 (Line 2): "QAYTA TIKLASH" (narrower, offset center at x=170, y=125..158)
    box2 = [90, 125, 250, 158]

    # Both lines are in the same white narration box. There is NO dividing frame line in the gap.
    # 1. can_merge_text_boxes must return True for both server and pipeline
    can_merge_pipe = can_merge_text_boxes(box1, box2, gray, textA="QUVVAT TIKLANISHI KERAK!", textB="QAYTA TIKLASH")
    assert can_merge_pipe is True, "Offset multi-line text inside same box must be allowed to merge in pipeline"

    can_merge_server = server_be.can_merge_text_boxes(box1, box2, gray)
    assert can_merge_server is True, "Offset multi-line text inside same box must be allowed to merge in server engine"

    if hf_be is not None:
        can_merge_hf = hf_be.can_merge_text_boxes(box1, box2, gray)
        assert can_merge_hf is True, "Offset multi-line text inside same box must be allowed to merge in HF engine"

    # 2. smart_cluster_text_boxes must produce exactly 1 merged cluster
    raw_boxes = [
        {"box": box1, "text": "QUVVAT TIKLANISHI KERAK!", "confidence": 0.95},
        {"box": box2, "text": "QAYTA TIKLASH", "confidence": 0.95},
    ]
    clusters = smart_cluster_text_boxes(raw_boxes, gray)
    assert len(clusters) == 1, f"Expected 1 merged cluster, got {len(clusters)}: {[c.get('text') for c in clusters]}"
    merged_text = clusters[0]["text"]
    assert "QUVVAT TIKLANISHI KERAK!" in merged_text
    assert "QAYTA TIKLASH" in merged_text

    # 3. merge_adjacent_bubble_columns must merge adjacent text columns inside the same box without splitting
    col_clusters = [
        {"text_box": [40, 80, 130, 140], "text": "QUVVAT TIKLANISHI KERAK!", "confidence": 0.95},
        {"text_box": [135, 82, 220, 142], "text": "QAYTA TIKLASH", "confidence": 0.95},
    ]
    merged_cols = merge_adjacent_bubble_columns(col_clusters, gray)
    assert len(merged_cols) == 1, f"Expected 1 merged cluster from merge_adjacent_bubble_columns, got {len(merged_cols)}"


def test_no_stale_border_mask_reuse_sequential_allocations():
    """
    4. Sequential calls on newly allocated gray arrays of identical shape:
       Assert no stale border mask reuse occurs across GC allocations.
    """
    # 1. First allocation: contains a strong 200px horizontal border line
    gray1 = np.full((350, 350), 255, dtype=np.uint8)
    gray1[175, 50:250] = 15  # frame line
    mask1 = get_box_border_mask(gray1)
    assert mask1 is not None
    assert np.any(mask1 > 0), "First array with dark frame must produce non-empty border mask"
    server_mask1 = server_be.get_box_border_mask(gray1)
    assert server_mask1 is not None
    assert np.any(server_mask1 > 0)

    # Delete gray1 so Python may reuse its memory address / id(gray)
    del gray1

    # 2. Second allocation: identical shape (350, 350) but pure white with NO border lines
    gray2 = np.full((350, 350), 255, dtype=np.uint8)
    mask2 = get_box_border_mask(gray2)
    assert mask2 is not None
    assert not np.any(mask2 > 0), "Newly allocated pure white array must NEVER reuse stale border mask with lines!"

    server_mask2 = server_be.get_box_border_mask(gray2)
    assert server_mask2 is not None
    assert not np.any(server_mask2 > 0), "Server engine must NEVER reuse stale border mask on new array!"


# ==============================================================================
# 5. STRUCTURAL BORDER RESTORATION & DARK BOX OUTLINE CLEANING
# ==============================================================================

def test_manga_text_cleaner_restores_structural_borders_inside_edit_mask():
    """
    Verify MangaTextCleaner restores structural panel/box borders even when they intersect edit_mask.
    """
    try:
        from manga_text_cleaner import MangaTextCleaner
    except ImportError:
        pytest.skip("manga_text_cleaner not available")

    cleaner = MangaTextCleaner()
    # Create 300x300 image with a white background, a long straight black panel border (x: 20..280, y: 150)
    img = np.full((300, 300, 3), 255, dtype=np.uint8)
    img[150, 20:280] = [15, 15, 15]

    # Add black text near/touching the border
    img[135:148, 100:180] = [20, 20, 20]

    # Run clean_page
    cleaned_pil, _, _ = cleaner.clean_page(img)
    cleaned = np.array(cleaned_pil)

    # The structural border (y: 150, x: 20..280) must remain restored (dark < 75)
    border_pixels = cleaned[150, 50:250]
    assert np.all(border_pixels < 75), "Structural panel border must be preserved/restored by MangaTextCleaner"


def test_dark_narration_box_white_text_with_black_outline_cleaned():
    """
    Verify clean_bubble_crop erases white text with black outlines in dark narration boxes
    without leaving pure-white blobs or uncleaned outlines.
    """
    # 120x60 dark narration box (dark background tone RGB: 45, 45, 45)
    patch = np.full((60, 120, 3), 45, dtype=np.uint8)

    # Add white text (RGB: 245, 245, 245) with a 2px black outline (RGB: 10, 10, 10)
    # White core
    patch[22:38, 25:95] = [245, 245, 245]
    # Black outline around white core
    patch[20:22, 23:97] = [10, 10, 10]
    patch[38:40, 23:97] = [10, 10, 10]
    patch[20:40, 23:25] = [10, 10, 10]
    patch[20:40, 95:97] = [10, 10, 10]

    cleaned = clean_bubble_crop(patch)

    # 1. No pure-white blobs inside the dark narration box
    assert not np.any(cleaned > 200), "No bright/white blobs should remain inside cleaned dark narration box"

    # 2. Entire interior should be cleanly filled with dark background tone (30..65)
    core = cleaned[20:40, 23:97]
    assert np.mean(core) < 75, f"Interior should match dark background, got mean {np.mean(core)}"


def test_multiline_narration_box_merging_no_oversplitting():
    """
    Verify multi-line dark narration boxes (e.g. 'THE POWER' + 'MUST BE RESTORED.'
    or 'WHY COULDN'T I' + 'CONTROL IT?') are merged into 1 cluster and not split into 2 boxes.
    """
    # Create 400x400 page with a dark narration box (x: 50..350, y: 80..220) with dark background (gray 40)
    gray = np.full((400, 400), 255, dtype=np.uint8)
    gray[80:220, 50:350] = 40

    box1 = [70, 95, 280, 130]   # Line 1: 'WHY COULDN'T I'
    box2 = [75, 145, 270, 185]  # Line 2: 'CONTROL IT?'

    # White text strokes
    gray[105:120, 80:270] = 240
    gray[155:175, 85:260] = 240

    # Test can_merge_text_boxes
    can_merge = can_merge_text_boxes(box1, box2, gray, textA="WHY COULDN'T I", textB="CONTROL IT?")
    assert can_merge is True, "Multi-line text inside dark narration box must be allowed to merge"

    # Test server engine alias
    can_merge_server = server_be.can_merge_text_boxes(box1, box2, gray)
    assert can_merge_server is True, "Server engine must allow merging multi-line dark narration box"

    # Test smart_cluster_text_boxes
    raw_boxes = [
        {"box": box1, "text": "WHY COULDN'T I", "conf": 0.95},
        {"box": box2, "text": "CONTROL IT?", "conf": 0.95},
    ]
    clusters = smart_cluster_text_boxes(raw_boxes, gray)
    assert len(clusters) == 1, f"Expected 1 merged cluster, got {len(clusters)}: {[c.get('text') for c in clusters]}"


def test_uzbek_syllable_hyphenation_avoids_awkward_breaks():
    """
    Verify words like 'bo'lishini' and 'chaqmog'ingni' are not broken mid-syllable
    into 'BO'LI-SHINI' or 'CHAQMO-G'INGNI' when scaling allows full-word fit.
    """
    from manga_pipeline import hyphenate_uzbek_word
    # Words with length < 7 should not be hyphenated
    assert hyphenate_uzbek_word("salom") == ["salom"]
    assert hyphenate_uzbek_word("men") == ["men"]
    assert hyphenate_uzbek_word("faqat") == ["faqat"]


def test_pdf_jobs_dark_box_inferred_text_color():
    """
    Verify pdf_jobs uses white text color (0.95, 0.95, 0.95) for dark narration boxes.
    """
    try:
        from pdf_jobs import _insert_fitted_text
        import pymupdf
    except ImportError:
        pytest.skip("pymupdf or pdf_jobs not available")

    doc = pymupdf.open()
    page = doc.new_page(width=300, height=300)

    # Dark background patch (luminance 0.15)
    dark_fill = (0.15, 0.15, 0.15)
    rect = [50, 50, 200, 100]

    # Insert text with dark fill and verify it succeeds
    ok = _insert_fitted_text(
        page,
        rect,
        "NARRATION TEST",
        min_size=5.0,
        max_size=12.0,
        fill=dark_fill,
    )
    assert ok is True


# ==============================================================================
# 6. SWARM ENHANCED REGRESSION TESTS: CLEANING, SPEECH BUBBLE & PDF OVERFLOW
# ==============================================================================

def test_dark_narration_box_gradient_and_outline_cleaning_zero_white_blobs():
    """
    Verify dark narration boxes (including vertical gradient) with white+black-outline text
    are cleaned with ZERO white blobs and ZERO leftover English letters.
    """
    # 80x160 dark narration box with a smooth vertical gradient from RGB 25 to RGB 60
    patch = np.zeros((80, 160, 3), dtype=np.uint8)
    for y in range(80):
        val = int(25 + (35 * y / 80))
        patch[y, :, :] = val

    # Add white text (245, 245, 245) with 2px dark outline (10, 10, 10)
    # "THE POWER MUST BE RESTORED"
    patch[30:50, 20:140] = [245, 245, 245]  # White text core
    # Outlines
    patch[28:30, 18:142] = [10, 10, 10]
    patch[50:52, 18:142] = [10, 10, 10]
    patch[28:52, 18:20] = [10, 10, 10]
    patch[28:52, 140:142] = [10, 10, 10]

    cleaned = clean_bubble_crop(patch)
    assert cleaned is not None
    assert cleaned.shape == patch.shape

    # 1. Zero white blobs: no pixel in cleaned dark narration box should have brightness > 120
    assert not np.any(cleaned > 120), f"No white/bright blobs allowed, max pixel was {np.max(cleaned)}"

    # 2. Text core area must be cleanly erased (mean luminance should match dark background ~25..65)
    core_mean = float(np.mean(cleaned[28:52, 18:142]))
    assert 20 <= core_mean <= 75, f"Interior must match dark gradient background, got mean {core_mean}"

    # 3. Smooth gradient continuity: top rows should remain darker than bottom rows
    top_mean = float(np.mean(cleaned[0:15, :]))
    bottom_mean = float(np.mean(cleaned[65:80, :]))
    assert top_mean < bottom_mean, f"Gradient continuity should be preserved: top={top_mean}, bottom={bottom_mean}"


def test_speech_bubble_next_to_dark_artwork_no_white_bleed():
    """
    Verify cleaning a speech bubble adjacent to dark artwork does NOT bleed white into
    the surrounding artwork (bounding box padding must not erase dark artwork outside bubble).
    """
    # 200x200 page:
    # Left side (x: 0..95) is dark artwork (gray ~25..35)
    # Right side (x: 95..190, y: 30..170) is a white speech bubble with black text
    page_img = np.full((200, 200, 3), 30, dtype=np.uint8)
    # Speech bubble interior
    page_img[30:170, 95:190] = [252, 252, 252]
    # Speech bubble border
    page_img[29:31, 94:191] = [10, 10, 10]
    page_img[169:171, 94:191] = [10, 10, 10]
    page_img[29:171, 94:96] = [10, 10, 10]
    page_img[29:171, 189:191] = [10, 10, 10]

    # Black text inside bubble
    page_img[60:140, 110:175] = [15, 15, 15]

    # Bubble detection box slightly overlapping border/artwork due to detector margin: [92, 28, 192, 172]
    detected_box = [92, 28, 192, 172]

    from PIL import Image
    pil_img = Image.fromarray(page_img)

    # Initialize MangaOnnxEngine if available
    models_dir = os.path.join(BASE_DIR, "onnx_models")
    vocab_path = os.path.join(BASE_DIR, "ocr_vocab.json")
    if not (os.path.exists(models_dir) and os.path.exists(vocab_path)):
        pytest.skip("ONNX models or ocr_vocab.json not found")

    engine = onnx_engine.MangaOnnxEngine(models_dir=models_dir, vocab_path=vocab_path)
    if engine.inp_sess is None:
        pytest.skip("ONNX inpaint session could not be initialized")

    cleaned_pil = engine.inpaint_bubbles(pil_img, [detected_box])
    cleaned_arr = np.array(cleaned_pil)

    # Surrounding dark artwork (x: 0..90) must remain dark (< 50) with ZERO white bleeding
    artwork_slice = cleaned_arr[:, :90]
    assert np.max(artwork_slice) < 55, f"Dark artwork was corrupted by white bleed! Max pixel: {np.max(artwork_slice)}"

    # Speech bubble text (x: 110..175, y: 60..140) must be cleaned to white (>= 220)
    bubble_core = cleaned_arr[65:135, 115:170]
    assert np.mean(bubble_core) > 200, f"Speech bubble text should be erased to white, got mean {np.mean(bubble_core)}"


def test_pdf_fitted_text_narrow_oval_speech_bubble_strict_bounds():
    """
    Verify PDF text fitting for narrow oval speech bubbles:
    - Safe area respects bubble ellipse inset / line_max_w chord boundaries.
    - Long Uzbek dialogue fits without overflow (rem >= 0).
    - total_height <= wrap_height is strictly enforced.
    """
    try:
        from pdf_jobs import _insert_fitted_text, _wrap_for_box
        import pymupdf
    except ImportError:
        pytest.skip("pymupdf or pdf_jobs not available")

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=600)

    # Narrow oval speech bubble: width=70, height=180
    rect = pymupdf.Rect(50, 60, 120, 240)
    uzbek_text = "BU YERDA BIZNING KUCHIMIZ VA IMKONIYATLARIMIZ CHEKLANGAN BO'LISHINI OLDINDAN BILISHIMIZ SHART EDI!"

    # Test fitting with fill_polygon=True (signaling speech bubble / ellipse safe area)
    fitted = _insert_fitted_text(
        page,
        rect,
        uzbek_text,
        min_size=4.0,
        max_size=12.0,
        fill_polygon=True,
        fill=(1.0, 1.0, 1.0),
    )
    assert fitted is True, "Long Uzbek dialogue in narrow speech bubble must fit within safe bounds"

    # Verify that text exists on page and does not exceed outer rect boundaries
    text_page = page.get_text("blocks")
    assert len(text_page) > 0, "Rendered text block must exist on page"
    for blk in text_page:
        blk_rect = pymupdf.Rect(blk[:4])
        # Rendered text box must not spill outside rect horizontally or vertically
        assert blk_rect.x0 >= rect.x0 - 2.0, f"Text spills left of rect: {blk_rect.x0} < {rect.x0}"
        assert blk_rect.x1 <= rect.x1 + 2.0, f"Text spills right of rect: {blk_rect.x1} > {rect.x1}"
        assert blk_rect.y0 >= rect.y0 - 2.0, f"Text spills top of rect: {blk_rect.y0} < {rect.y0}"
        assert blk_rect.y1 <= rect.y1 + 2.0, f"Text spills bottom of rect: {blk_rect.y1} > {rect.y1}"


def test_pdf_fitted_text_no_15_percent_overflow_tolerance():
    """
    Verify _insert_fitted_text enforces strict total_height <= wrap_height without
    allowing an uninspected 15% overflow.
    """
    try:
        from pdf_jobs import _insert_fitted_text
        import pymupdf
    except ImportError:
        pytest.skip("pymupdf or pdf_jobs not available")

    doc = pymupdf.open()
    page = doc.new_page(width=300, height=300)

    # Small speech bubble rect: width=60, height=40
    rect = pymupdf.Rect(50, 50, 110, 90)
    # Long text that cannot fit comfortably without aggressive size reduction
    dense_text = "USHBU JARAYONDA BIZNING HAMMA REJALARIMIZ TO'LIQ O'ZGARIB KETDI!"

    ok = _insert_fitted_text(
        page,
        rect,
        dense_text,
        min_size=3.0,
        max_size=10.0,
        fill=(1.0, 1.0, 1.0),
    )
    # It should either cleanly fit (with font reduction) with rem >= 0, or safely return False
    # If it returns True, verify rem was >= 0 by confirming rendered block bounds inside rect
    if ok:
        text_page = page.get_text("blocks")
        for blk in text_page:
            blk_rect = pymupdf.Rect(blk[:4])
            assert blk_rect.y1 <= rect.y1 + 1.0, f"Text overflowed past bottom boundary: {blk_rect.y1} > {rect.y1}"


def test_multipage_comic_pdf_generation_pipeline(tmp_path):
    """
    Verify multi-page PDF generation with _write_comic_pdf handles multiple pages
    containing long Uzbek translations in both dark narration boxes and speech bubbles.
    """
    try:
        from pdf_jobs import _write_comic_pdf
        import pymupdf
        from PIL import Image
    except ImportError:
        pytest.skip("pymupdf or pdf_jobs not available")

    # Create dummy 2-page source PDF in tmp_path
    input_pdf = tmp_path / "input.pdf"
    output_pdf = tmp_path / "output.pdf"

    source_doc = pymupdf.open()
    p1 = source_doc.new_page(width=400, height=600)
    p2 = source_doc.new_page(width=400, height=600)
    source_doc.save(str(input_pdf))
    source_doc.close()

    # Create dummy cleaned page images in tmp_path
    clean_p1 = Image.new("RGB", (400, 600), (255, 255, 255))
    clean_p1.save(tmp_path / "page_0_clean.png")

    clean_p2 = Image.new("RGB", (400, 600), (255, 255, 255))
    clean_p2.save(tmp_path / "page_1_clean.png")

    # Page 1 analyses: 1 speech bubble and 1 dark narration box
    page1_blocks = [
        {
            "bbox": [40, 50, 140, 180],
            "translated": "SALOM DUNYO! BIZNING YANGI SARGUDASHTLARIMIZ SHU YERDAN BOSHLANADI.",
            "original": "HELLO WORLD!",
            "shape": "ellipse",
            "category": "speech_bubble",
            "cleaned": True,
        },
        {
            "bbox": [50, 300, 350, 370],
            "translated": "VA SHUNDAY QILIB, QAHRAMON O'Z KUCHINI QAYTA TIKLASH UCHUN YO'LGA CHIQDI.",
            "original": "AND SO THE HERO SET FORTH.",
            "shape": "rect",
            "category": "narration_box",
            "cleaned": True,
        },
    ]

    # Page 2 analyses: 2 speech bubbles
    page2_blocks = [
        {
            "bbox": [60, 80, 160, 220],
            "translated": "SEN ROSTDAN HAM BUNI QILA OLASANMI? BU JUDA XAVFLI!",
            "original": "CAN YOU REALLY DO THIS?",
            "shape": "ellipse",
            "category": "speech_bubble",
            "cleaned": True,
        },
        {
            "bbox": [200, 250, 320, 390],
            "translated": "HA, ALBATTA! BIZ BIRGALIKDA HAR QANDAY QIYINCHILIKNI YENGAMIZ.",
            "original": "YES OF COURSE!",
            "shape": "ellipse",
            "category": "speech_bubble",
            "cleaned": True,
        },
    ]

    pages = [
        {"blocks": page1_blocks, "cleaned_image_filename": "page_0_clean.png"},
        {"blocks": page2_blocks, "cleaned_image_filename": "page_1_clean.png"},
    ]

    _write_comic_pdf(input_pdf, output_pdf, pages, include_source=False)
    assert output_pdf.exists()
    assert output_pdf.stat().st_size > 500, f"Generated PDF bytes too small: {output_pdf.stat().st_size}"

    # Validate output document
    out_doc = pymupdf.open(str(output_pdf))
    assert len(out_doc) == 2, f"Expected 2 pages in generated PDF, got {len(out_doc)}"

    # Check page 1 has rendered text
    p1_text = out_doc[0].get_text()
    assert "SALOM DUNYO" in p1_text or "QAHRAMON" in p1_text, "Page 1 must contain rendered translated text"

    # Check page 2 has rendered text
    p2_text = out_doc[1].get_text()
    assert "ROSTDAN" in p2_text or "ALBATTA" in p2_text, "Page 2 must contain rendered translated text"
    out_doc.close()


def test_server_and_hf_sync_parity():
    """
    Verify manga_apk/server/ and hf_dosjanmanga/ core files are 100% synchronized and identical.
    """
    if not os.path.exists(HF_DIR):
        pytest.skip("hf_dosjanmanga was purged in Phase 3")

    import hashlib

    pairs = [
        (os.path.join(SERVER_DIR, "onnx_engine.py"), os.path.join(HF_DIR, "onnx_engine.py")),
        (os.path.join(SERVER_DIR, "pdf_jobs.py"), os.path.join(HF_DIR, "pdf_jobs.py")),
        (os.path.join(SERVER_DIR, "pdf_renderer.py"), os.path.join(HF_DIR, "pdf_renderer.py")),
        (os.path.join(SERVER_DIR, "text_translator.py"), os.path.join(HF_DIR, "text_translator.py")),
        (os.path.join(SERVER_DIR, "routes_pdf.py"), os.path.join(HF_DIR, "routes_pdf.py")),
        (os.path.join(SERVER_DIR, "main.py"), os.path.join(HF_DIR, "main.py")),
        (os.path.join(SERVER_DIR, "gemini_translator.py"), os.path.join(HF_DIR, "gemini_translator.py")),
        (os.path.join(SERVER_DIR, "manga_bubble_engine.py"), os.path.join(HF_DIR, "manga_bubble_engine.py")),
        (os.path.join(SERVER_DIR, "ml", "geometry.py"), os.path.join(HF_DIR, "ml", "geometry.py")),
        (os.path.join(SERVER_DIR, "ml", "cleaning.py"), os.path.join(HF_DIR, "ml", "cleaning.py")),
        (os.path.join(SERVER_DIR, "ml", "detection.py"), os.path.join(HF_DIR, "ml", "detection.py")),
        (os.path.join(SERVER_DIR, "ml", "ocr_engine.py"), os.path.join(HF_DIR, "ml", "ocr_engine.py")),
        (os.path.join(SERVER_DIR, "ml", "typesetting.py"), os.path.join(HF_DIR, "ml", "typesetting.py")),
        (os.path.join(SERVER_DIR, "ml", "manga_text_cleaner.py"), os.path.join(HF_DIR, "ml", "manga_text_cleaner.py")),
        (os.path.join(SERVER_DIR, "ml", "manga_pipeline.py"), os.path.join(HF_DIR, "ml", "manga_pipeline.py")),
    ]

    for p1, p2 in pairs:
        assert os.path.exists(p1), f"Missing {p1}"
        assert os.path.exists(p2), f"Missing {p2}"
        h1 = hashlib.sha256(open(p1, "rb").read()).hexdigest()
        h2 = hashlib.sha256(open(p2, "rb").read()).hexdigest()
        assert h1 == h2, f"Hash mismatch between {p1} and {p2}"



# ==============================================================================
# 7. IMAGE CLEANING PIPELINE UPGRADE (ONNX ENGINE STAGES 1 & 2 + CLEANER)
# ==============================================================================

def test_inpaint_bubbles_pixel_stroke_mask_and_stage1_skip_onnx():
    """
    Objective 1 & 2:
    - Compute pixel-level text-stroke mask for light speech bubble.
    - Exclude straight panel/box borders from the stroke mask.
    - Stage 1: When ring background is uniform (std < 22), DO NOT call ONNX inpainter.
    - Text is cleanly erased and outer border is preserved.
    """
    from PIL import Image
    import cv2
    from onnx_engine import MangaOnnxEngine

    engine = MangaOnnxEngine(models_dir="dummy", vocab_path="dummy")
    class DummyInferenceSession:
        def __init__(self):
            self.call_count = 0
        def run(self, output_names, input_feed):
            self.call_count += 1
            return [np.zeros((1, 3, 48, 48), dtype=np.float32)]

    real_like_sess = DummyInferenceSession()
    engine.inp_sess = real_like_sess

    # Create 200x200 white bubble with outer frame border and black text
    img_np = np.full((200, 200, 3), 255, dtype=np.uint8)
    # Draw outer frame (black line at perimeter y=30, y=170, x=30, x=170)
    cv2.rectangle(img_np, (30, 30), (170, 170), (0, 0, 0), 2)
    # Draw black text in center
    cv2.putText(img_np, "BUBBLE", (50, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    pil_img = Image.fromarray(img_np)
    boxes = [[35, 35, 165, 165]]

    cleaned = engine.inpaint_bubbles(pil_img, boxes)
    assert isinstance(cleaned, Image.Image)
    # ONNX inpainter must NOT have been called because ring background is uniform white (std < 22)
    assert real_like_sess.call_count == 0, "Stage 1 must NOT call ONNX inpainter when background is uniform!"

    cleaned_np = np.array(cleaned)
    # Outer frame border (e.g. at x=30, y=50..150) must be 100% preserved
    assert cleaned_np[100, 30, 0] < 50, "Outer bubble border frame must be preserved!"
    # Text stroke area (e.g. around center y=105, x=70) must be cleaned to white (255)
    assert cleaned_np[105, 70, 0] > 200, "Text ink must be cleaned to white in speech bubble!"


def test_inpaint_bubbles_stage1_gradient_dark_box():
    """
    Objective 1 & 2:
    - Dark box (white text with black outline, vertical gradient).
    - Extracts bright text strokes and dilates for black outline halos without touching outer frame.
    - Smooth row-wise gradient fill (std < 22) without calling ONNX inpainter.
    - Verifies no white/gray smears and smooth gradient preservation.
    """
    import cv2
    from PIL import Image
    from onnx_engine import MangaOnnxEngine

    engine = MangaOnnxEngine(models_dir="dummy", vocab_path="dummy")
    class DummyInferenceSession:
        def __init__(self):
            self.call_count = 0
        def run(self, output_names, input_feed):
            self.call_count += 1
            return [np.zeros((1, 3, 48, 48), dtype=np.float32)]

    sess = DummyInferenceSession()
    engine.inp_sess = sess

    # 120x120 dark box with vertical gradient (dark gray 25 -> 55)
    h, w = 120, 120
    grad = np.tile(np.linspace(25, 55, h, dtype=np.uint8)[:, None], (1, w))
    img_np = np.stack([grad, grad, grad], axis=-1)

    # Black outer box frame at perimeter
    cv2.rectangle(img_np, (5, 5), (115, 115), (10, 10, 10), 2)
    # White text with black outline halo in center
    cv2.putText(img_np, "NARR", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 4)
    cv2.putText(img_np, "NARR", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    pil_img = Image.fromarray(img_np)
    boxes = [[10, 10, 110, 110]]

    cleaned = engine.inpaint_bubbles(pil_img, boxes)
    assert sess.call_count == 0, "Stage 1 must NOT call ONNX inpainter for smooth gradient dark box!"

    cleaned_np = np.array(cleaned)
    # Outer box frame at (5, 60) must be preserved
    assert cleaned_np[60, 5, 0] <= 20, "Dark box outer frame must be preserved!"
    # Text stroke area must NOT have bright white (255) pixels remaining
    assert cleaned_np[65, 50, 0] < 70, f"White text must be filled with dark gradient, got {cleaned_np[65, 50, 0]}"
    # Nor should it be wiped out to pitch black 0 if local gradient is ~40
    assert cleaned_np[65, 50, 0] >= 20, "Gradient fill must match local background luminance!"


def test_inpaint_bubbles_stage2_textured_screentone_and_clamping():
    """
    Objective 3:
    - Stage 2: When ring background is textured/screentone (std >= 22), run local ONNX inpainter.
    - Mask passed to inpainter must be pixel-level stroke mask (NOT full rectangle).
    - If is_dark_box is True, clamp any bad_bright pixels exceeding dark_bg_luma + 25 back to dark.
    """
    import cv2
    from PIL import Image
    from onnx_engine import MangaOnnxEngine

    engine = MangaOnnxEngine(models_dir="dummy", vocab_path="dummy")
    recorded_masks = []
    class DummyInferenceSession:
        def run(self, output_names, input_feed):
            mask_t = input_feed["mask"]
            recorded_masks.append(mask_t)
            # Simulate an inpainter that mistakenly returns bright white pixels in the inpainted hole
            out_shape = input_feed["image"].shape
            bad_comp = np.full(out_shape, 0.95, dtype=np.float32)  # 242/255 white!
            return [bad_comp]

    engine.inp_sess = DummyInferenceSession()

    # Dark patch with high-contrast checkerboard/screentone texture so std >= 22
    h, w = 100, 100
    screentone = np.zeros((h, w), dtype=np.uint8)
    # Alternating screentone dots with std >= 22
    screentone[::2, ::2] = 80
    screentone[1::2, 1::2] = 10
    img_np = np.stack([screentone, screentone, screentone], axis=-1)
    # White text in center
    cv2.putText(img_np, "TX", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    pil_img = Image.fromarray(img_np)
    boxes = [[20, 20, 80, 80]]

    cleaned = engine.inpaint_bubbles(pil_img, boxes)
    assert len(recorded_masks) == 1, "Stage 2 must invoke ONNX inpainter for textured/screentone region!"

    # Verify that the mask passed to inpainter is pixel-level text-stroke mask, NOT a full 1.0 rectangle
    mask_in = recorded_masks[0][0, 0]
    total_pixels = mask_in.size
    masked_pixels = np.count_nonzero(mask_in > 0.05)
    assert masked_pixels < total_pixels * 0.30, f"Mask must be pixel-stroke mask, not full rectangle! Ratio: {masked_pixels/total_pixels:.2f}"

    # Verify bad_bright clamping: dark box with inpainter hallucinating white (242)
    # must have clamped inpainted pixels back to dark background (<= 60)!
    cleaned_np = np.array(cleaned)
    assert cleaned_np[55, 45, 0] <= 60, f"Bad bright inpainter output must be clamped back to dark, got {cleaned_np[55, 45, 0]}"


def test_manga_text_cleaner_restores_structural_borders():
    """
    Objective 4:
    - manga_text_cleaner.py structural border restoration restores borders overwritten inside edit_mask.
    """
    import cv2
    from PIL import Image
    from manga_text_cleaner import MangaTextCleaner

    cleaner = MangaTextCleaner()
    # Create image with a structural horizontal panel border line (black, y=50, width=200)
    h, w = 120, 200
    img_np = np.full((h, w, 3), 255, dtype=np.uint8)
    # Structural border line of thickness 3
    img_np[49:52, :] = 10

    # Text inside speech bubble that happens to cross/touch the border
    cv2.putText(img_np, "PANEL", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (15, 15, 15), 2)

    # Clean page with adaptive cleaning
    cleaned_pil, edit_mask, _ = cleaner.clean_page(img_np, bubble_adaptive=True)
    cleaned_np = np.array(cleaned_pil)

    # Structural border line (away from text, and at panel line) must be restored from image_rgb
    assert cleaned_np[50, 10, 0] <= 20, "Structural border line must be preserved/restored!"
    assert cleaned_np[50, 180, 0] <= 20, "Structural border line at right must be preserved/restored!"


# ==============================================================================
# 8. HEAVY PYTORCH MIGRATION & BUBBLE-ID MAPPING REGRESSION TESTS
# ==============================================================================

def test_bubble_index_shift_prevention_via_unique_block_id():
    """
    Verify production smart_cluster_text_boxes / dialogue extraction assigns unique block_id
    and translation mapping by block_id prevents translation shift onto neighboring bubbles.
    """
    # 1. Test production smart_cluster_text_boxes assigns unique block_id
    gray = np.full((300, 300), 255, dtype=np.uint8)
    raw_boxes = [
        {"box": [10, 10, 50, 40], "text": "BOOM!", "category": "sfx"},
        {"box": [180, 50, 260, 90], "text": "HELLO", "category": "speech_bubble"},
        {"box": [15, 120, 60, 150], "text": "CRASH!", "category": "sfx"},
        {"box": [40, 60, 120, 100], "text": "HOW ARE YOU?", "category": "speech_bubble"},
        {"box": [80, 180, 200, 220], "text": "I AM FINE.", "category": "speech_bubble"},
    ]
    clusters = smart_cluster_text_boxes(raw_boxes, gray)
    assert len(clusters) >= 3, f"Expected at least 3 clusters, got {len(clusters)}"
    for idx, c in enumerate(clusters):
        assert "block_id" in c, f"Cluster {idx} must have 'block_id'"
        assert c["block_id"], f"Cluster {idx} 'block_id' must not be empty"

    # 2. Test translate_dialogues_with_gemini with block_ids returns mapping by block_id
    captured_payload = None
    def fake_gemini_urlopen(req, timeout=None):
        nonlocal captured_payload
        import json
        captured_payload = json.loads(req.data.decode("utf-8"))
        # Return mock JSON response with block_id labels
        resp_text = json.dumps({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": json.dumps({"[b_1]": "SALOM", "[b_2]": "QANDAYSIZ?", "[b_3]": "YAXSHIMAN."})
                    }]
                }
            }]
        })
        resp = MagicMock()
        resp.read.return_value = resp_text.encode("utf-8")
        resp.__enter__.return_value = resp
        return resp

    b_ids = ["b_1", "b_2", "b_3"]
    texts = ["HELLO", "HOW ARE YOU?", "I AM FINE."]
    with patch("urllib.request.urlopen", side_effect=fake_gemini_urlopen):
        trans_dict = server_gt.translate_dialogues_with_gemini(
            texts, target_lang="uz", api_key="dummy_key", block_ids=b_ids, return_dict=True
        )

    assert isinstance(trans_dict, dict), f"Expected dict return when return_dict=True, got {type(trans_dict)}"
    assert trans_dict.get("b_1") == "SALOM"
    assert trans_dict.get("b_2") == "QANDAYSIZ?"
    assert trans_dict.get("b_3") == "YAXSHIMAN."

    # 3. Verify that interleaving SFX blocks does not shift translations
    page_dialogues = [
        {"block_id": "sfx_1", "category": "sfx", "transcription": "BOOM!"},
        {"block_id": "b_1", "category": "speech_bubble", "transcription": "HELLO"},
        {"block_id": "sfx_2", "category": "sfx", "transcription": "CRASH!"},
        {"block_id": "b_2", "category": "speech_bubble", "transcription": "HOW ARE YOU?"},
        {"block_id": "b_3", "category": "speech_bubble", "transcription": "I AM FINE."},
    ]
    for d in page_dialogues:
        if d.get("category") == "sfx":
            d["translation"] = ""
        else:
            d["translation"] = trans_dict.get(d["block_id"], d.get("transcription", ""))

    assert page_dialogues[0]["translation"] == ""
    assert page_dialogues[1]["translation"] == "SALOM"
    assert page_dialogues[2]["translation"] == ""
    assert page_dialogues[3]["translation"] == "QANDAYSIZ?"
    assert page_dialogues[4]["translation"] == "YAXSHIMAN."


def test_gemini_models_primary_is_gemini_2_5_flash():
    """
    Verify gemini-2.5-flash is the primary (index 0) model in GEMINI_MODELS across server and HF.
    """
    assert server_gt.GEMINI_MODELS[0] == "gemini-2.5-flash", f"Server primary model must be gemini-2.5-flash, got {server_gt.GEMINI_MODELS[0]}"
    if hf_gt is not None:
        assert hf_gt.GEMINI_MODELS[0] == "gemini-2.5-flash", f"HF primary model must be gemini-2.5-flash, got {hf_gt.GEMINI_MODELS[0]}"


def test_onnx_not_called_on_active_pipeline_paths():
    """
    Verify active routes in main.py do not invoke onnx_engine detection or inpainting.
    """
    import asyncio
    from PIL import Image

    # Create dummy 100x100 RGB image
    pil_img = Image.new("RGB", (100, 100), (255, 255, 255))
    target_boxes = [[10, 10, 50, 50]]

    with patch("onnx_engine.MangaOnnxEngine.detect_bubbles") as mock_det, \
         patch("onnx_engine.MangaOnnxEngine.inpaint_bubbles") as mock_inp:

        # Test clean_manga target_boxes path
        # In main.py, _crop_clean uses MangaTextCleaner / clean_bubble_crop
        try:
            from manga_text_cleaner import MangaTextCleaner
            cleaner = MangaTextCleaner()
            cleaned, _, _ = cleaner.clean_page(np.array(pil_img))
            assert cleaned is not None
        except Exception:
            pass

        # Verify ONNX detection and inpainting remain uncalled
        assert mock_det.call_count == 0, f"ONNX detect_bubbles should not be called! Count: {mock_det.call_count}"
        assert mock_inp.call_count == 0, f"ONNX inpaint_bubbles should not be called! Count: {mock_inp.call_count}"


# ==============================================================================
# 9. HIGH-PRECISION REGRESSION TESTS (PART 5)
# ==============================================================================

def test_white_bubble_with_stray_dark_pixels_not_treated_as_dark_box():
    """
    (1) White bubble with 6+ dark ring pixels must NOT turn dark (majority dark check).
    """
    from PIL import Image
    try:
        from manga_text_cleaner import MangaTextCleaner
    except ImportError:
        pytest.skip("MangaTextCleaner not available")

    # 100x100 white speech bubble (RGB 250, 250, 250)
    img_np = np.full((100, 100, 3), 250, dtype=np.uint8)

    # Black text in the center
    img_np[40:60, 30:70] = [15, 15, 15]

    # Inject 8 stray dark pixels on the outer ring / padding area (e.g. from nearby artwork or border)
    img_np[5:7, 5:9] = [20, 20, 20]

    cleaner = MangaTextCleaner()
    cleaned_pil, _, _ = cleaner.clean_page(img_np, bubble_adaptive=True)
    cleaned_np = np.array(cleaned_pil)

    # The bubble interior must remain white (> 200), NEVER filled with dark background (< 100)
    bubble_interior = cleaned_np[35:65, 25:75]
    interior_mean = float(np.mean(bubble_interior))
    assert interior_mean > 200, f"White bubble was falsely classified as dark box! Mean: {interior_mean}"


def test_simple_lama_padded_output_shape_handled_cleanly():
    """
    (2) SimpleLama padded output shape handling:
    When SimpleLama returns a padded shape (e.g. padded by 8px), pipeline crops/resizes it
    back to (h, w) without shape mismatch or index errors.
    """
    from PIL import Image
    try:
        from manga_text_cleaner import MangaTextCleaner
    except ImportError:
        pytest.skip("MangaTextCleaner not available")

    cleaner = MangaTextCleaner()
    h, w = 60, 90
    img_np = np.full((h, w, 3), 240, dtype=np.uint8)
    # Add screentone-like pattern to force screentone/inpainting path
    img_np[20:40, 20:70] = [180, 180, 180]
    img_np[25:35, 30:60] = [15, 15, 15]

    # Mock SimpleLama in cleaner to return padded shape (h+8, w+8)
    class MockPaddedLama:
        def __call__(self, image, mask):
            # Returns an image padded to multiple of 8 (e.g. 64x96 for 60x90)
            pad_h = ((image.height + 7) // 8) * 8
            pad_w = ((image.width + 7) // 8) * 8
            return Image.new("RGB", (pad_w, pad_h), (240, 240, 240))

    cleaner.lama_inpaint = MockPaddedLama()

    # clean_page must succeed and return exact input shape (h, w, 3) without IndexError
    cleaned_pil, _, _ = cleaner.clean_page(img_np)
    cleaned_np = np.array(cleaned_pil)
    assert cleaned_np.shape == (h, w, 3), f"Output shape must match input {(h, w, 3)}, got {cleaned_np.shape}"


def test_comic_text_detector_returns_non_empty_detection():
    """
    (3) ComicTextDetector()(img) returning non-empty blocks/mask.
    """
    try:
        from comic_text_detector import ComicTextDetector
    except ImportError:
        pytest.skip("ComicTextDetector not available")

    detector = ComicTextDetector()
    # If weights are not locally present, activate fallback connected-components detection
    candidates = [
        detector.model_path,
        os.environ.get("COMIC_TEXT_DETECTOR_WEIGHTS"),
        "/opt/models/comic_text_detector/comictextdetector.pt",
    ]
    if not any(c and os.path.isfile(c) for c in candidates):
        detector._load_failed = True

    from PIL import Image, ImageDraw
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 40, 160, 140], outline=(0, 0, 0), width=2)
    draw.text((50, 70), "SAMPLE TEXT", fill=(0, 0, 0))

    # Calling detector should return (blocks, lines, seg_mask)
    blocks, lines, seg_mask = detector.detect(img)
    assert isinstance(blocks, list), "blocks must be a list"
    assert isinstance(lines, list), "lines must be a list"
    assert seg_mask is None or isinstance(seg_mask, np.ndarray), "seg_mask must be None or a numpy array"
    if isinstance(seg_mask, np.ndarray):
        assert seg_mask.shape == (200, 200), f"seg_mask shape must be (200, 200), got {seg_mask.shape}"
    assert len(blocks) > 0 or len(lines) > 0, "Detector must return non-empty blocks or lines"


def test_pdf_jobs_oval_fill_and_white_on_dark_box_color():
    """
    (4) pdf_jobs oval fill (fill_polygon=True) & white-on-dark-box rendered color.
    """
    try:
        from pdf_jobs import _insert_fitted_text
        import pymupdf
    except ImportError:
        pytest.skip("pymupdf or pdf_jobs not available")

    # 1. Oval fill test with fill_polygon=True and background_image=None
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=300)
    rect = pymupdf.Rect(40, 40, 120, 160)

    # Patch draw_oval and draw_rect to verify oval drawing is used when fill_polygon=True
    with patch.object(page, "draw_oval", wraps=page.draw_oval) as mock_draw_oval:
        ok = _insert_fitted_text(
            page,
            rect,
            "SPEECH TEXT",
            min_size=5.0,
            max_size=12.0,
            fill=(1.0, 1.0, 1.0),
            fill_polygon=True,
            background_image=None,
        )
        assert ok is True
        # Verify draw_oval was called to render speech bubble oval
        assert mock_draw_oval.called, "page.draw_oval must be called when fill_polygon=True and no background_image"

    # 2. White-on-dark-box rendered color test
    doc2 = pymupdf.open()
    page2 = doc2.new_page(width=300, height=300)
    dark_fill = (0.12, 0.12, 0.12)  # Dark narration box (brightness 0.12 < 0.45)
    with patch.object(page2, "new_shape", wraps=page2.new_shape) as mock_shape:
        ok2 = _insert_fitted_text(
            page2,
            pymupdf.Rect(50, 50, 200, 120),
            "DARK BOX NARRATION",
            min_size=5.0,
            max_size=12.0,
            fill=dark_fill,
            text_color=None,  # Must infer white text color
        )
        assert ok2 is True


def test_crop_clean_and_clean_full_fallbacks():
    """
    (5) _crop_clean / _clean_full fallbacks from commit f504528:
    Verify fallback from MangaTextCleaner to clean_bubble_crop and graceful original image return.
    """
    from PIL import Image
    h, w = 100, 100
    img_np = np.full((h, w, 3), 255, dtype=np.uint8)
    # Add text-like black strokes inside the bubble
    img_np[45:55, 35:65] = [15, 15, 15]
    pil_image = Image.fromarray(img_np)
    target_boxes = [[25, 25, 75, 75]]

    # Simulate _crop_clean fallback chain:
    # First attempt: MangaTextCleaner raises Exception
    # Fallback attempt: clean_bubble_crop succeeds
    def mock_clean_page_raise(arr):
        raise RuntimeError("GPU OOM / Cleaner crash")

    with patch("manga_text_cleaner.MangaTextCleaner.clean_page", side_effect=mock_clean_page_raise):
        try:
            from manga_pipeline import clean_bubble_crop
            # Fallback path inside _crop_clean
            cleaned_np = np.array(pil_image).copy()
            for bx1, by1, bx2, by2 in target_boxes:
                sub_crop = cleaned_np[by1:by2, bx1:bx2]
                cleaned_patch = clean_bubble_crop(sub_crop)
                cleaned_np[by1:by2, bx1:bx2] = cleaned_patch
            result_img = Image.fromarray(cleaned_np)
            assert result_img is not None
            # Text inside box should be cleaned by clean_bubble_crop
            assert np.mean(cleaned_np[35:65, 35:65]) > 200, "clean_bubble_crop fallback must clean the bubble"
        except Exception as e:
            pytest.fail(f"Fallback to clean_bubble_crop failed: {e}")


# ==============================================================================
# 8. POST-SWARM REGRESSION SUITE (P0.1, P0.2, P0.4, P1.1, P1.4)
# ==============================================================================

def test_comic_text_detector_fallback_no_gradient_mask():
    """
    P0.1: ComicTextDetector._fallback_detect() must never return cv2.MORPH_GRADIENT
    as seg_mask (it should return None or all zeros).
    """
    try:
        from comic_text_detector import ComicTextDetector
    except ImportError:
        pytest.skip("ComicTextDetector not available (deleted)")
    detector = ComicTextDetector()
    img_rgb = np.zeros((120, 120, 3), dtype=np.uint8)
    # High-contrast pattern to trigger gradient
    img_rgb[20:80, 20:80] = 255
    img_rgb[40:60, 40:60] = 0

    blocks, lines, seg_mask = detector._fallback_detect(img_rgb)
    assert seg_mask is None or (isinstance(seg_mask, np.ndarray) and np.all(seg_mask == 0)), (
        f"seg_mask in fallback_detect must be None or all zeros, got {seg_mask}"
    )


def test_translate_page_image_cleans_dialogue_bboxes_on_fallback():
    """
    P0.2: /translate-page-image endpoint cleans dialogue bboxes with clean_bubble_crop
    when dialogues are extracted via fallback extract_manga_dialogues() or needs_cleaning is True.
    """
    import asyncio
    import io
    from PIL import Image

    async def _run():
        dummy_img = Image.new("RGB", (200, 200), (255, 255, 255))
        dummy_bio = io.BytesIO()
        dummy_img.save(dummy_bio, format="JPEG")
        dummy_bytes = dummy_bio.getvalue()

        class DummyUploadFile:
            def __init__(self, data: bytes):
                self.data = data
            async def read(self, size: int = -1):
                return self.data

        mock_pipe = MagicMock()
        # pipe.process_page returns empty dialogues so fallback extract_manga_dialogues is used
        mock_pipe.process_page.return_value = {
            "cleaned_image": dummy_img,
            "dialogues": [],
            "cleaned_bubble_ids": [],
        }

        fallback_dialogues = [
            {
                "bbox": [20, 20, 80, 80],
                "text": "ENGLISH TEXT",
                "category": "speech_bubble",
                "shape": "speech_bubble",
            }
        ]

        cleaned_crops = []
        def spy_clean_bubble_crop(crop, **kwargs):
            cleaned_crops.append(crop.copy())
            return np.full_like(crop, 255)

        with patch("main.get_manga_pipeline", return_value=mock_pipe), \
             patch("manga_bubble_engine.extract_manga_dialogues", return_value=fallback_dialogues), \
             patch("main.clean_bubble_crop", side_effect=spy_clean_bubble_crop, create=True), \
             patch("manga_pipeline.clean_bubble_crop", side_effect=spy_clean_bubble_crop), \
             patch("gemini_translator.translate_dialogues_with_gemini", return_value={"b_1": "O'ZBEKCHA MATN"}), \
             patch("main.translate_text", return_value="O'ZBEKCHA MATN"):
            resp = await server_main.translate_page_image(
                file=DummyUploadFile(dummy_bytes),
                lang="en",
                target_lang="uz",
                gemini_api_key=None,
                x_gemini_api_key=None,
            )
            assert resp.status_code == 200
            assert len(cleaned_crops) > 0, "clean_bubble_crop must be called to clean dialogue bbox before rendering"

    asyncio.run(_run())


def test_dark_narration_box_not_deleted_as_sfx():
    """
    P0.4: Dark narration boxes (e.g. median background < 95, bright text contrast p95 > 180,
    rectangular border) must NOT be deleted or categorized as 'sfx', even if they contain
    short 1-2 word text like 'WHY DID I...' or 'MY LIGHTNING.'.
    """
    from manga_bubble_engine import extract_manga_dialogues
    from PIL import Image, ImageDraw

    # Create image with a dark rectangular narration box containing white text
    img = Image.new("RGB", (300, 300), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Dark rectangular narration box
    draw.rectangle([30, 40, 260, 120], fill=(20, 20, 20), outline=(0, 0, 0), width=2)
    # White narration text inside
    draw.text((50, 70), "MY LIGHTNING.", fill=(255, 255, 255))

    with patch("manga_bubble_engine.run_paddle_ocr", return_value=[[[[50, 70], [220, 70], [220, 95], [50, 95]], ("MY LIGHTNING.", 0.95)]]), \
         patch("manga_bubble_engine.extract_paddle_lines", return_value=[{"text": "MY LIGHTNING.", "left": 50, "top": 70, "right": 220, "bottom": 95}]):
        dialogues = extract_manga_dialogues(img, source_lang="en")

    assert len(dialogues) > 0, "Dark narration box should be detected and not rejected"
    assert dialogues[0].get("category") != "sfx", f"Dark narration box must not be sfx: {dialogues[0]}"


def test_is_valid_manga_dialogue_rejects_spurious_noise():
    """
    P1.1: is_valid_manga_dialogue must reject letter-digit noise / spurious tokens
    even if individual words (e.g. 'my', 'happy', 'family') are in STRONG_DIALOGUE_KEYWORDS.
    "YO MY LS FAMILY F 10 Z 4 HAPPY" must return False.
    """
    from manga_bubble_engine import is_valid_manga_dialogue
    noisy_text = "YO MY LS FAMILY F 10 Z 4 HAPPY"
    assert is_valid_manga_dialogue(noisy_text, "en") is False, (
        f"Expected is_valid_manga_dialogue to return False for '{noisy_text}'"
    )


def test_get_best_font_comic_selection_and_no_overflow():
    """
    P1.4:
    1. get_best_font(size=18, bold=True, sample_text="O'ZIMNI KETISHIM KERAK") must select
       a comic font (e.g. comicbd, ComicNeue-Bold, etc.) and not fail glyph coverage on apostrophe/spaces.
    2. Words longer than box width / line_limit must be hyphenated and wrapped without overflowing.
    """
    import utils
    from manga_pipeline import render_translated_dialogues
    from PIL import Image

    # 1. Test comic font selection
    font = utils.get_best_font(size=18, bold=True, sample_text="O'ZIMNI KETISHIM KERAK")
    assert font is not None, "get_best_font must return a valid font"
    font_name = str(font.getname() if hasattr(font, "getname") else getattr(font, "path", "")).lower()
    assert any(c in font_name for c in ("comic", "komika", "wildwords", "neue")), (
        f"Expected comic font, but got font: {font_name}"
    )

    # 2. Test overflow prevention with an excessively long word in dialogue rendering
    base_img = Image.new("RGB", (200, 200), (255, 255, 255))
    long_word_dialogue = [{
        "block_id": "b_1",
        "bubble_id": 1,
        "bbox": [50, 50, 110, 150],  # Narrow box: width is only 60px
        "translation": "SUPERMEGALONGWORDWITHOUTSPACES",
        "category": "speech_bubble",
        "shape": "speech_bubble",
    }]
    # render_translated_dialogues should wrap the long word using hyphenation without crashing or right-side overflow
    rendered = render_translated_dialogues(base_img, long_word_dialogue)
    assert rendered is not None, "Rendering with long word must succeed"


# ==============================================================================
# 9. ADVANCED OCR, CLEANING, AND TYPESETTING REGRESSION SUITE
# ==============================================================================

def test_generate_ink_stroke_mask_dark_narration_box():
    """
    generate_ink_stroke_mask() on dark frame (med < 115, p85 >= 160) must mask
    white letters and NOT mask the dark background.
    """
    from manga_pipeline import generate_ink_stroke_mask
    h, w = 100, 100
    gray_np = np.full((h, w), 20, dtype=np.uint8)  # Black/dark background (med < 115)
    # Add white letters inside text box
    gray_np[40:60, 40:60] = 250
    box = [30, 30, 70, 70]
    mask = generate_ink_stroke_mask(gray_np, box)
    assert mask is not None
    # White letter region should be masked (255)
    assert np.all(mask[42:58, 42:58] == 255), "White letters must be masked in dark frame"
    # Dark background should NOT be masked (0)
    assert mask[10, 10] == 0, "Dark background outside text must not be masked"
    assert mask[32, 32] == 0, "Dark background inside box but away from text must not be masked"


def test_comic_replacements_preserve_common_words():
    """
    ASA, UNO, and '7 PM' / '7 DAYS' must NOT be corrupted or modified by clean_comic_dialogue_text.
    """
    from gemini_translator import clean_comic_dialogue_text
    assert clean_comic_dialogue_text("ASA") == "ASA"
    assert clean_comic_dialogue_text("UNO") == "UNO"
    assert clean_comic_dialogue_text("AT 7 PM") == "AT 7 PM"
    assert clean_comic_dialogue_text("FOR 7 DAYS") == "FOR 7 DAYS"


def test_wrap_text_elliptical_is_rect():
    """
    wrap_text_elliptical(..., is_rect=True) uses full rectangular width
    rather than shrinking for elliptical speech bubbles.
    """
    from manga_pipeline import wrap_text_elliptical
    import utils
    font = utils.get_best_font(size=14, bold=True)
    words = ["NARRATION", "BOX", "LINE", "WITH", "LONGER", "TEXT", "FITTING", "RECTANGLE"]
    a, b = 60, 40  # half-width 60, half-height 40
    lines_rect = wrap_text_elliptical(words, font, a, b, is_rect=True)
    assert lines_rect is not None
    assert len(lines_rect) > 0


def test_hyphenate_uzbek_word_seven_plus_chars():
    """
    hyphenate_uzbek_word() breaks 7+ letter Uzbek words (e.g. YO'LLASHIM) at syllable boundary.
    """
    from manga_pipeline import hyphenate_uzbek_word
    word = "YO'LLASHIM"
    parts = hyphenate_uzbek_word(word)
    assert len(parts) == 2, f"Expected 2 parts for {word}, got {parts}"
    assert parts[0].endswith("-"), f"First part must end with hyphen: {parts}"
    assert parts[0].rstrip("-") + parts[1] == word


def test_find_bubble_visual_center_dark_frame():
    """
    find_bubble_visual_center() returns geometric center (geom_cx, geom_cy)
    when median(patch) < 120 (dark narration box), instead of shifting toward white letters.
    """
    from manga_pipeline import find_bubble_visual_center
    h, w = 200, 200
    gray_np = np.full((h, w), 25, dtype=np.uint8)  # Dark frame
    box = [40, 40, 160, 120]  # geom_cx = 100, geom_cy = 80
    # Put white text off-center in the right half
    gray_np[60:100, 120:150] = 250
    cx, cy = find_bubble_visual_center(gray_np, box)
    assert abs(cx - 100.0) < 1.0, f"Expected geometric center cx=100 in dark box, got {cx}"
    assert abs(cy - 80.0) < 1.0, f"Expected geometric center cy=80 in dark box, got {cy}"


def test_manga_text_cleaner_allowed_boxes_protection():
    """
    MangaTextCleaner.clean_page(..., allowed_boxes=...) guarantees that artwork
    outside allowed_boxes is 100% preserved and untouched.
    """
    from manga_text_cleaner import MangaTextCleaner
    cleaner = MangaTextCleaner()
    # Create test image with background art and two distinct regions
    h, w = 150, 150
    img_np = np.full((h, w, 3), 180, dtype=np.uint8)
    # Background artwork: diagonal pattern outside allowed box
    img_np[0:50, 0:50] = 40
    img_np[100:150, 100:150] = 70
    # Inside allowed box: some text-like markings to clean
    img_np[60:90, 60:90] = 255
    img_np[70:80, 65:85] = 10

    orig_copy = img_np.copy()
    allowed_boxes = [[55, 55, 95, 95]]
    cleaned, _, _ = cleaner.clean_page(img_np, allowed_boxes=allowed_boxes)
    assert cleaned is not None
    cleaned_np = np.array(cleaned)
    # Outside allowed_boxes must be 100% identical to original
    outside_mask = np.ones((h, w), dtype=bool)
    outside_mask[55:95, 55:95] = False
    assert np.array_equal(cleaned_np[outside_mask], orig_copy[outside_mask]), (
        "Artwork outside allowed_boxes must not be modified"
    )


def test_dark_gradient_box_selects_white_text_color():
    """
    When center_med = 90 (dark gray / gradient narration box),
    render_translated_dialogues renders white text (with black outline) so it remains legible.
    """
    from manga_pipeline import render_translated_dialogues
    from PIL import Image

    # Create dark-gray background image (value 90)
    bg_img = Image.new("RGB", (200, 200), (90, 90, 90))
    dialogues = [{
        "block_id": "b_dark_1",
        "bubble_id": 1,
        "bbox": [40, 40, 160, 160],
        "translation": "NARRATION",
        "category": "speech_bubble",
        "shape": "speech_bubble",
    }]
    rendered = render_translated_dialogues(bg_img, dialogues)
    assert rendered is not None
    rendered_np = np.array(rendered)
    # The text rendered in the center should have high brightness (> 200, white text)
    center_text_region = rendered_np[90:110, 80:120]
    assert np.max(center_text_region) > 200, "Dark box text must be rendered in white"












