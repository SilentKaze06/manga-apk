import os
import sys
from pathlib import Path

ML_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ML_ROOT)
SERVER_DIR = os.path.join(REPO_ROOT, "manga_apk", "server")
SERVER_ML_DIR = os.path.join(SERVER_DIR, "ml")
SERVER_ONNX_DIR = os.path.join(SERVER_DIR, "onnx_models")

_KNOWN_DATA_MARKERS = (
    "self_learning_data",
    "manga_ml_datasets",
    "validation",
    "test_chapter_57",
    "recent_samples",
)


def ensure_ml_paths():
    """Ensure mashinali oqitish/, manga_apk/server/ml/, and manga_apk/server/ are on sys.path."""
    for path in (ML_ROOT, SERVER_ML_DIR, SERVER_DIR, REPO_ROOT):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def resolve_vocab_path(base_dir=None):
    """Resolve ocr_vocab.json across mashinali oqitish/ and manga_apk/server/."""
    candidates = []
    if base_dir is not None:
        candidates.append(os.path.join(os.fspath(base_dir), "ocr_vocab.json"))
    candidates.extend([
        os.path.join(ML_ROOT, "ocr_vocab.json"),
        os.path.join(SERVER_ML_DIR, "ocr_vocab.json"),
        os.path.join(SERVER_ONNX_DIR, "ocr_vocab.json"),
    ])
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return os.path.join(SERVER_ML_DIR, "ocr_vocab.json")


def resolve_data_path(path_str, base_dir=None):
    """Resolve relative or moved desktop dataset paths against base_dir / ML_ROOT."""
    if not path_str:
        return ""
    raw = os.fspath(path_str)
    if os.path.exists(raw):
        return os.path.normpath(raw)

    root = os.fspath(base_dir) if base_dir is not None else ML_ROOT
    rel_cand = os.path.normpath(os.path.join(root, raw))
    if os.path.exists(rel_cand):
        return rel_cand

    parts = Path(raw.replace("\\", "/")).parts
    for marker in _KNOWN_DATA_MARKERS:
        if marker in parts:
            idx = parts.index(marker)
            sub_cand = os.path.normpath(os.path.join(root, *parts[idx:]))
            if os.path.exists(sub_cand):
                return sub_cand
            if root != ML_ROOT:
                ml_cand = os.path.normpath(os.path.join(ML_ROOT, *parts[idx:]))
                if os.path.exists(ml_cand):
                    return ml_cand

    return os.path.normpath(raw)
