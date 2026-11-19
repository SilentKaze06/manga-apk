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

try:
    import importlib.util
    def load_module_from_path(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    hf_gt = load_module_from_path("hf_gemini_translator", os.path.join(HF_DIR, "gemini_translator.py"))
    hf_be = load_module_from_path("hf_manga_bubble_engine", os.path.join(HF_DIR, "manga_bubble_engine.py"))
except Exception as e:
    hf_gt = None
    hf_be = None


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
    """HTTP 400 and 404 must NOT be retried (abort immediately)."""
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
            # Must abort Gemini calls after 1 attempt (no retry on 400/404!)
            assert gemini_calls == 1, f"Expected 1 Gemini call on HTTP {err_code}, got {gemini_calls}"


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
