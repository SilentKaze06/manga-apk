import sys
from unittest.mock import MagicMock
if "gradio" not in sys.modules:
    try:
        import gradio  # noqa: F401
    except ImportError:
        sys.modules["gradio"] = MagicMock()

import numpy as np
from PIL import Image, ImageDraw

from app import update_bubble_boundary, get_default_session, recompute_inpaint_and_annotated


def create_test_manga_image():
    # White background manga page with some black text/ink strokes
    img = Image.new("RGB", (400, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Draw text-like ink lines in region [100, 100, 150, 150]
    draw.rectangle([110, 110, 140, 140], fill=(0, 0, 0))
    # Draw text-like ink lines in expanded region [150, 100, 190, 140]
    draw.rectangle([155, 110, 185, 140], fill=(0, 0, 0))
    # Draw artwork/feature in region [50, 50, 80, 80]
    draw.rectangle([55, 55, 75, 75], fill=(50, 50, 50))
    return img


def test_update_bubble_boundary_syncs_text_box_and_cleans_expanded_region():
    raw_img = create_test_manga_image()
    session = get_default_session()
    session["orig_img"] = raw_img
    session["dialogues"] = [{
        "bubble_id": 1,
        "category": "speech_bubble",
        "bbox": [100, 100, 150, 150],
        "text_box": [100, 100, 150, 150],
        "transcription": "Sample text",
    }]
    session["crops"] = {1: raw_img.crop((94, 94, 156, 156))}

    # Initial compute
    recompute_inpaint_and_annotated(session, target_bid=1, active_bid=1)
    cleaned_initial = np.array(session["cleaned_img"])
    # The first box [110, 110, 140, 140] should be inpainted/white
    assert np.mean(cleaned_initial[115:135, 115:135]) > 200
    # The expanded area [155, 110, 185, 140] should still be black initially
    assert np.mean(cleaned_initial[115:135, 160:180]) < 50

    # Expand boundary to include the second text block: [100, 100, 200, 150]
    update_bubble_boundary("#1", 100, 100, 200, 150, session)

    d = session["dialogues"][0]
    assert d["bbox"] == [100, 100, 200, 150]
    assert d["text_box"] == [100, 100, 200, 150], "text_box must be synchronized with new bbox!"

    cleaned_after_expand = np.array(session["cleaned_img"])
    # Now the expanded region [155, 110, 185, 140] must also be inpainted/cleaned!
    assert np.mean(cleaned_after_expand[115:135, 160:180]) > 200


def test_update_bubble_boundary_shrink_restores_original_art():
    raw_img = create_test_manga_image()
    session = get_default_session()
    session["orig_img"] = raw_img
    session["dialogues"] = [{
        "bubble_id": 1,
        "category": "speech_bubble",
        "bbox": [100, 100, 200, 150],
        "text_box": [100, 100, 200, 150],
        "transcription": "Sample text",
    }]
    session["crops"] = {1: raw_img.crop((94, 94, 206, 156))}

    # Initial compute with large boundary
    recompute_inpaint_and_annotated(session, target_bid=1, active_bid=1)
    cleaned_initial = np.array(session["cleaned_img"])
    # Both regions are cleaned
    assert np.mean(cleaned_initial[115:135, 115:135]) > 200
    assert np.mean(cleaned_initial[115:135, 160:180]) > 200

    # Shrink boundary back to [100, 100, 150, 150]
    update_bubble_boundary("#1", 100, 100, 150, 150, session)
    d = session["dialogues"][0]
    assert d["bbox"] == [100, 100, 150, 150]
    assert d["text_box"] == [100, 100, 150, 150]

    cleaned_after_shrink = np.array(session["cleaned_img"])
    # Shrunk region [155, 110, 185, 140] must be restored to original dark ink!
    assert np.mean(cleaned_after_shrink[115:135, 160:180]) < 50


def test_update_bubble_boundary_move_restores_old_and_cleans_new():
    raw_img = create_test_manga_image()
    session = get_default_session()
    session["orig_img"] = raw_img
    session["dialogues"] = [{
        "bubble_id": 1,
        "category": "speech_bubble",
        "bbox": [100, 100, 150, 150],
        "text_box": [100, 100, 150, 150],
        "transcription": "Sample text",
    }]
    session["crops"] = {1: raw_img.crop((94, 94, 156, 156))}
    recompute_inpaint_and_annotated(session, target_bid=1, active_bid=1)

    # Move bubble to [50, 50, 80, 80] where dark artwork is
    update_bubble_boundary("#1", 50, 50, 80, 80, session)
    d = session["dialogues"][0]
    assert d["bbox"] == [50, 50, 80, 80]
    assert d["text_box"] == [50, 50, 80, 80]

    cleaned = np.array(session["cleaned_img"])
    # Old location [115:135, 115:135] must be restored with original black ink!
    assert np.mean(cleaned[115:135, 115:135]) < 50
    # New location [55:75, 55:75] must be cleaned/inpainted!
    assert np.mean(cleaned[58:72, 58:72]) > 180


def test_neighbor_bubble_remains_clean_when_boundary_changes():
    raw_img = create_test_manga_image()
    session = get_default_session()
    session["orig_img"] = raw_img
    session["dialogues"] = [
        {
            "bubble_id": 1,
            "category": "speech_bubble",
            "bbox": [100, 100, 150, 150],
            "text_box": [100, 100, 150, 150],
            "transcription": "Bubble 1",
        },
        {
            "bubble_id": 2,
            "category": "speech_bubble",
            "bbox": [150, 100, 200, 150],
            "text_box": [150, 100, 200, 150],
            "transcription": "Bubble 2",
        }
    ]
    session["crops"] = {
        1: raw_img.crop((94, 94, 156, 156)),
        2: raw_img.crop((144, 94, 206, 156))
    }
    recompute_inpaint_and_annotated(session, active_bid=1)

    cleaned_initial = np.array(session["cleaned_img"])
    assert np.mean(cleaned_initial[115:135, 160:180]) > 200  # Bubble 2 cleaned

    # Update Bubble 1 boundary (shrink it slightly)
    update_bubble_boundary("#1", 105, 105, 145, 145, session)

    cleaned_after = np.array(session["cleaned_img"])
    # Bubble 2 must still remain cleaned, not damaged by neighbor update!
    assert np.mean(cleaned_after[115:135, 160:180]) > 200
