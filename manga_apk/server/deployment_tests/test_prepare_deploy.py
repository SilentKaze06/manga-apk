import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("prepare_deploy", Path(__file__).resolve().parents[1] / "prepare_deploy.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class DeploymentBundleTests(unittest.TestCase):
    def make_sources(self, root):
        server, ml = root / "server", root / "ml-source"
        for folder, names in ((server, deploy.SERVER_FILES), (ml, deploy.ML_FILES)):
            for name in names:
                path = folder / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test content", encoding="utf-8")
        return server, ml

    def test_private_documents_and_training_data_never_enter_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server, ml = self.make_sources(root)
            for folder, name in ((server, "pdf_job_data/input.pdf"), (server, ".env"), (ml, "self_learning_data/ocr/feedback.jsonl")):
                path = folder / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("PRIVATE SENTINEL", encoding="utf-8")
            bundle = deploy.prepare_bundle(server, ml, root / "bundle")
            manifest = json.loads((bundle / "bundle-manifest.json").read_text())
            self.assertEqual(len(manifest["files"]), len(deploy.SERVER_FILES) + len(deploy.ML_FILES))
            self.assertTrue((bundle / "ml/checkpoints/best_manga_ocr.pt").is_file())
            self.assertFalse(any(b"PRIVATE SENTINEL" in f.read_bytes() for f in bundle.rglob("*") if f.is_file()))

    def test_missing_checkpoint_fails_before_output_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server, ml = self.make_sources(root)
            (ml / "checkpoints/best_manga_ocr.pt").unlink()
            with self.assertRaises(ValueError):
                deploy.prepare_bundle(server, ml, root / "bundle")
            self.assertFalse((root / "bundle").exists())

    def test_only_known_ocr_weights_are_copied_from_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server, ml = self.make_sources(root)
            cache = root / "ocr-cache"
            cache.mkdir()
            (cache / "japanese_g2.pth").write_bytes(b"known model")
            (cache / "private.pdf").write_bytes(b"private document")
            (cache / "unknown_model.pth").write_bytes(b"unreviewed model")
            bundle = deploy.prepare_bundle(server, ml, root / "bundle", cache)
            copied = list((bundle / "ml/easyocr_models").iterdir())
            self.assertEqual([item.name for item in copied], ["japanese_g2.pth"])
            self.assertEqual(copied[0].read_bytes(), b"known model")
            rules = (bundle / ".dockerignore").read_text()
            self.assertIn("!ml/easyocr_models/japanese_g2.pth", rules)
            self.assertNotIn("!ml/**", rules)
            manifest = json.loads((bundle / "bundle-manifest.json").read_text())
            self.assertEqual(len(manifest["files"]), len(deploy.SERVER_FILES) + len(deploy.ML_FILES) + 1)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server, ml = self.make_sources(root)
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "keep.txt").write_text("keep")
            with self.assertRaises(FileExistsError):
                deploy.prepare_bundle(server, ml, bundle)
            self.assertEqual((bundle / "keep.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
