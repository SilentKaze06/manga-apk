"""
Export trained PyTorch models (MangaBubbleDetector, MangaOCRModel, MangaInpaintingModel)
to highly optimized ONNX format for 4x faster CPU inference and 70% lower RAM.
"""

import os
import sys
import json
import argparse
import shutil
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import ml_paths
ml_paths.ensure_ml_paths()

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = ml_paths.SERVER_ONNX_DIR
VOCAB_PATH = ml_paths.resolve_vocab_path(BASE_DIR)

from models import MangaBubbleDetector, MangaOCRModel, MangaInpaintingModel


def export_detector(output_dir=OUTPUT_DIR, checkpoint_dir=CHECKPOINT_DIR, device=None):
    if device is None:
        device = torch.device("cpu")
    os.makedirs(output_dir, exist_ok=True)
    print("\n[1/3] MangaBubbleDetector eksport qilinmoqda...")
    detector_path = os.path.join(checkpoint_dir, "best_manga_detector.pt")
    if not os.path.exists(detector_path):
        print(f"  ✗ Topilmadi: {detector_path}")
        return None

    detector = MangaBubbleDetector(num_classes=4).to(device)
    detector.load_state_dict(torch.load(detector_path, map_location=device, weights_only=True))
    detector.eval()

    dummy_input = torch.randn(1, 3, 384, 256, device=device)
    detector_onnx_path = os.path.join(output_dir, "manga_detector.onnx")

    try:
        torch.onnx.export(
            detector,
            dummy_input,
            detector_onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            dynamo=False,
            input_names=["image_input"],
            output_names=["box_predictions"],
            dynamic_axes={
                "image_input": {0: "batch_size", 2: "height", 3: "width"},
                "box_predictions": {0: "batch_size", 2: "grid_h", 3: "grid_w"},
            }
        )
    except TypeError:
        torch.onnx.export(
            detector,
            dummy_input,
            detector_onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["image_input"],
            output_names=["box_predictions"],
            dynamic_axes={
                "image_input": {0: "batch_size", 2: "height", 3: "width"},
                "box_predictions": {0: "batch_size", 2: "grid_h", 3: "grid_w"},
            }
        )
    print(f"  ✓ Muvaffaqiyatli saqlandi: {detector_onnx_path} ({os.path.getsize(detector_onnx_path) / (1024*1024):.2f} MB)")
    return detector_onnx_path


def export_ocr(output_dir=OUTPUT_DIR, checkpoint_dir=CHECKPOINT_DIR, vocab_path=VOCAB_PATH, device=None):
    if device is None:
        device = torch.device("cpu")
    os.makedirs(output_dir, exist_ok=True)
    print("\n[2/3] MangaOCRModel eksport qilinmoqda...")
    ocr_path = os.path.join(checkpoint_dir, "best_manga_ocr.pt")
    if not (os.path.exists(ocr_path) and os.path.exists(vocab_path)):
        print(f"  ✗ Topilmadi: {ocr_path} yoki {vocab_path}")
        return None

    with open(vocab_path, encoding="utf-8") as f:
        vocab_data = json.load(f)
    vocab_size = max(vocab_data["char_to_idx"].values()) + 1

    ocr = MangaOCRModel(vocab_size=vocab_size, hidden_dim=128).to(device)
    state = torch.load(ocr_path, map_location=device, weights_only=True)
    ocr.load_checkpoint_weights(state)
    ocr.eval()

    class OcrSequenceWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            feats = self.model.features(x)
            pooled = feats.mean(dim=2)  # ONNX-compatible height collapse
            seq_in = pooled.permute(2, 0, 1)
            recurrent, _ = self.model.rnn(seq_in)
            logits = self.model.seq_fc(recurrent)
            return torch.nn.functional.log_softmax(logits, dim=-1)

    wrapped_ocr = OcrSequenceWrapper(ocr)
    dummy_ocr_input = torch.randn(1, 3, 64, 256, device=device)
    ocr_onnx_path = os.path.join(output_dir, "manga_ocr.onnx")

    try:
        torch.onnx.export(
            wrapped_ocr,
            dummy_ocr_input,
            ocr_onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            dynamo=False,
            input_names=["line_crop_input"],
            output_names=["ctc_log_probs"],
            dynamic_axes={
                "line_crop_input": {0: "batch_size", 3: "crop_width"},
                "ctc_log_probs": {0: "time_steps", 1: "batch_size"}
            }
        )
    except TypeError:
        torch.onnx.export(
            wrapped_ocr,
            dummy_ocr_input,
            ocr_onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["line_crop_input"],
            output_names=["ctc_log_probs"],
            dynamic_axes={
                "line_crop_input": {0: "batch_size", 3: "crop_width"},
                "ctc_log_probs": {0: "time_steps", 1: "batch_size"}
            }
        )
    print(f"  ✓ Muvaffaqiyatli saqlandi: {ocr_onnx_path} ({os.path.getsize(ocr_onnx_path) / (1024*1024):.2f} MB)")
    return ocr_onnx_path


def export_inpainter(output_dir=OUTPUT_DIR, checkpoint_dir=CHECKPOINT_DIR, device=None):
    if device is None:
        device = torch.device("cpu")
    os.makedirs(output_dir, exist_ok=True)
    print("\n[3/3] MangaInpaintingModel eksport qilinmoqda...")
    inpaint_path = os.path.join(checkpoint_dir, "best_manga_inpaint.pt")
    if not os.path.exists(inpaint_path):
        print(f"  ✗ Topilmadi: {inpaint_path}")
        return None

    inpainter = MangaInpaintingModel().to(device)
    inpainter.load_state_dict(torch.load(inpaint_path, map_location=device, weights_only=True))
    inpainter.eval()

    class InpaintExportWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, image, mask):
            # Unconditional slice keeps channel reduction in the exported ONNX graph.
            mask = mask[:, :1]
            h = image.shape[-2]
            w = image.shape[-1]
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            image_pad = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
            mask_pad = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h), mode="replicate")
            infilled, comp = self.model(image_pad, mask_pad)
            return infilled[..., :h, :w], comp[..., :h, :w]

    wrapped_inpainter = InpaintExportWrapper(inpainter)
    dummy_img = torch.randn(1, 3, 256, 256, device=device)
    dummy_mask = torch.zeros(1, 1, 256, 256, device=device)
    inpaint_onnx_path = os.path.join(output_dir, "manga_inpaint.onnx")

    try:
        torch.onnx.export(
            wrapped_inpainter,
            (dummy_img, dummy_mask),
            inpaint_onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            dynamo=False,
            input_names=["image", "mask"],
            output_names=["infilled", "composite"],
            dynamic_axes={
                "image": {0: "batch", 2: "h", 3: "w"},
                "mask": {0: "batch", 1: "mask_channels", 2: "h", 3: "w"},
                "infilled": {0: "batch", 2: "h", 3: "w"},
                "composite": {0: "batch", 2: "h", 3: "w"},
            }
        )
    except TypeError:
        torch.onnx.export(
            wrapped_inpainter,
            (dummy_img, dummy_mask),
            inpaint_onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["image", "mask"],
            output_names=["infilled", "composite"],
            dynamic_axes={
                "image": {0: "batch", 2: "h", 3: "w"},
                "mask": {0: "batch", 1: "mask_channels", 2: "h", 3: "w"},
                "infilled": {0: "batch", 2: "h", 3: "w"},
                "composite": {0: "batch", 2: "h", 3: "w"},
            }
        )
    print(f"  ✓ Muvaffaqiyatli saqlandi: {inpaint_onnx_path} ({os.path.getsize(inpaint_onnx_path) / (1024*1024):.2f} MB)")
    return inpaint_onnx_path


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description="Export trained Manga PyTorch checkpoints to ONNX")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Destination ONNX directory (default: manga_apk/server/onnx_models)")
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR, help="Source PyTorch checkpoint directory")
    parser.add_argument("--vocab-path", default=VOCAB_PATH, help="OCR vocabulary JSON path")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cpu")

    print("============================================================")
    print("  PyTorch -> ONNX Model Exporter for Manga Server")
    print(f"  Output dir: {args.output_dir}")
    print("============================================================")

    export_detector(output_dir=args.output_dir, checkpoint_dir=args.checkpoint_dir, device=device)
    export_ocr(output_dir=args.output_dir, checkpoint_dir=args.checkpoint_dir, vocab_path=args.vocab_path, device=device)
    export_inpainter(output_dir=args.output_dir, checkpoint_dir=args.checkpoint_dir, device=device)
    if os.path.isfile(args.vocab_path):
        dest_vocab = os.path.join(args.output_dir, "ocr_vocab.json")
        if os.path.abspath(args.vocab_path) != os.path.abspath(dest_vocab):
            shutil.copy2(args.vocab_path, dest_vocab)
    print("\nBarcha ONNX modellari tayyor!")


if __name__ == "__main__":
    main()
