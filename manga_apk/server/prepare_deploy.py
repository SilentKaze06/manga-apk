"""Build an allowlisted deployment context without documents or training data."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

SERVER_FILES = (
    "main.py", "pdf_jobs.py", "gemini_translator.py",
    "manga_bubble_engine.py", "onnx_engine.py",
    "requirements.txt", "Dockerfile", ".env.example",
)
ML_FILES = (
    "manga_pipeline.py", "models.py", "dataset_loader.py",
    "self_learning_collector.py", "self_learning_trainer.py", "validation_gate.py",
    "utils.py", "ocr_vocab.json", "checkpoints/best_manga_detector.pt",
    "checkpoints/best_manga_ocr.pt", "checkpoints/best_manga_inpaint.pt",
)
EASYOCR_FILES = (
    "craft_mlt_25k.pth", "english_g2.pth", "latin_g2.pth",
    "japanese_g2.pth", "korean_g2.pth", "zh_sim_g2.pth",
    "cyrillic_g2.pth", "chinese.pth", "arabic.pth",
)


def _checked_file(root: Path, relative: str) -> Path:
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required regular file is missing: {path}")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Source path escapes its project: {path}") from exc
    return path


def prepare_bundle(server_dir: Path, ml_dir: Path, destination: Path, ocr_model_dir: Path | None = None) -> Path:
    server_dir, ml_dir = server_dir.resolve(), ml_dir.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Choose a new output directory; refusing to overwrite: {destination}")
    sources = [(_checked_file(server_dir, name), Path(name)) for name in SERVER_FILES]
    sources += [(_checked_file(ml_dir, name), Path("ml") / name) for name in ML_FILES]
    if ocr_model_dir is not None:
        for name in EASYOCR_FILES:
            if (ocr_model_dir / name).exists():
                sources.append((_checked_file(ocr_model_dir, name), Path("ml/easyocr_models") / name))
    # Validate all required files before writing any output.
    destination.mkdir(parents=True, exist_ok=False)
    manifest = []
    for source, relative in sources:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest.append({"path": relative.as_posix(), "bytes": target.stat().st_size,
                         "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    (destination / "bundle-manifest.json").write_text(
        json.dumps({"files": manifest}, indent=2) + "\n", encoding="utf-8"
    )
    rules = ["**", "!main.py", "!pdf_jobs.py", "!requirements.txt", "!Dockerfile", "!ml/"]
    # Keep even later additions to ml/ excluded unless they are on this allowlist.
    rules += ["!ml/checkpoints/", "!ml/easyocr_models/"]
    rules += [f"!{relative.as_posix()}" for _, relative in sources if relative.parts[0] == "ml"]
    for name in (".dockerignore", ".gcloudignore"):
        (destination / name).write_text("\n".join(rules) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    server_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-dir", type=Path, default=server_dir.parent.parent / "mashinali oqitish")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ocr-model-dir", type=Path, default=Path.home() / ".EasyOCR/model")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    destination = args.output or server_dir.parent / "build" / "deploy" / f"manga-server-{stamp}"
    try:
        result = prepare_bundle(server_dir, args.ml_dir, destination, args.ocr_model_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Deployment bundle was not created: {exc}\n")
    print(result)


if __name__ == "__main__":
    main()
