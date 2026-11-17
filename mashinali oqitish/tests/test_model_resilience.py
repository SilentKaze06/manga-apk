import torch
import pytest

from models import MangaBubbleDetector, MangaOCRModel, MangaInpaintingModel

def test_ocr_load_checkpoint_weights_vocab_expansion():
    """Verify that expanding vocabulary and loading an older checkpoint does not crash."""
    old_vocab_size = 100
    new_vocab_size = 120
    
    old_model = MangaOCRModel(vocab_size=old_vocab_size, hidden_dim=64)
    old_state = old_model.state_dict()
    
    new_model = MangaOCRModel(vocab_size=new_vocab_size, hidden_dim=64)
    # Should not raise RuntimeError: size mismatch for seq_fc.weight or fc
    new_model.load_checkpoint_weights(old_state)
    
    assert new_model.vocab_size == new_vocab_size
    # Verify weights up to old_vocab_size match
    assert torch.allclose(new_model.seq_fc.weight[:old_vocab_size], old_state["seq_fc.weight"])
    assert torch.allclose(new_model.seq_fc.bias[:old_vocab_size], old_state["seq_fc.bias"])


@pytest.mark.parametrize("missing_key", ["fc.4.bias", "seq_fc.bias"])
def test_ocr_vocab_expansion_rejects_incomplete_checkpoint(missing_key):
    old_model = MangaOCRModel(vocab_size=100, hidden_dim=64)
    old_state = old_model.state_dict()
    del old_state[missing_key]
    new_model = MangaOCRModel(vocab_size=120, hidden_dim=64)
    with pytest.raises(ValueError, match="Incomplete OCR checkpoint"):
        new_model.load_checkpoint_weights(old_state)

def test_inpainting_arbitrary_dimensions_padding():
    """Verify that dimensions not divisible by 8 don't crash U-Net down/up sampling."""
    model = MangaInpaintingModel()
    model.eval()
    
    # Non-multiple of 8 dimensions (e.g. 125 x 137)
    dummy_img = torch.randn(1, 3, 125, 137)
    dummy_mask = torch.ones(1, 1, 125, 137)
    
    with torch.no_grad():
        out, comp = model(dummy_img, dummy_mask)
        
    assert out.shape == (1, 3, 125, 137)
    assert comp.shape == (1, 3, 125, 137)

def test_bubble_detector_prediction():
    """Verify that MangaBubbleDetector predict returns valid boxes."""
    model = MangaBubbleDetector(num_classes=4)
    model.eval()
    dummy_img = torch.randn(1, 3, 384, 256)
    preds = model.predict(dummy_img, conf_thresh=0.0)
    assert isinstance(preds, list)
    for p in preds:
        assert "box" in p
        assert len(p["box"]) == 4
        x1, y1, x2, y2 = p["box"]
        assert 0.0 <= x1 <= x2 <= 1.0
        assert 0.0 <= y1 <= y2 <= 1.0
        assert "conf" in p or "score" in p
        conf_val = p.get("conf", p.get("score"))
        assert 0.0 <= conf_val <= 1.0
        assert "class_id" in p
