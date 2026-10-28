"""Offline wiring smoke: real local ML weights/OCR, synthetic art, stub translation.

Run explicitly with the ML virtualenv from the server directory. This is not an
accuracy benchmark; it never trains, collects samples, or opens real PDF jobs.
"""
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont
import pymupdf


def run():
    server_dir = Path(__file__).resolve().parents[1]
    ml_dir = server_dir.parent.parent / "mashinali oqitish"
    sys.path.insert(0, str(server_dir))
    sys.path.insert(0, str(ml_dir))
    with tempfile.TemporaryDirectory(prefix="manga_offline_integration_") as temporary:
        root = Path(temporary)
        environment = {
            "MANGA_PDF_JOB_DIR": str(root / "jobs"),
            "MANGA_ML_DIR": str(ml_dir),
            "MANGA_EASYOCR_MODEL_DIR": str(Path.home() / ".EasyOCR" / "model"),
            "MANGA_ALLOW_UNAUTHENTICATED": "false",
        }
        with patch.dict(os.environ, environment), patch.object(
            socket.socket, "connect", side_effect=AssertionError("Offline smoke forbids network access")
        ):
            import main
            import manga_pipeline
            from pdf_jobs import translate_pdf_file

            image = Image.new("RGB", (600, 800), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((10, 10, 590, 790), outline="black", width=3)
            draw.ellipse((70, 200, 530, 480), outline="black", width=3)
            try:
                font = ImageFont.truetype("arial.ttf", 44)
            except OSError:
                font = ImageFont.load_default(size=44)
            draw.text((160, 300), "HELLO WORLD", fill="black", font=font)
            image_bytes = io.BytesIO()
            image.save(image_bytes, "PNG")
            source_path = root / "source.pdf"
            with pymupdf.open() as source:
                page = source.new_page(width=300, height=400)
                page.insert_image(page.rect, stream=image_bytes.getvalue())
                source.save(source_path)

            translations = []

            def translate(text, target_lang, source_lang, cancel_event=None):
                translations.append(text)
                return "Salom dunyo"

            try:
                with patch.object(manga_pipeline, "SelfLearningCollector", return_value=None), patch.object(
                    manga_pipeline, "SelfLearningTrainer", return_value=None
                ), patch.object(main, "run_paddle_ocr", side_effect=AssertionError("Expected cached EasyOCR path")):
                    translate_pdf_file(
                        input_path=source_path,
                        output_path=root / "translated.pdf",
                        page_cache_path=root / "cache.json",
                        title="Offline integration",
                        source_lang="en",
                        target_lang="uz",
                        output_mode="bilingual",
                        translate_text=translate,
                        progress_callback=lambda *_: None,
                        cancel_event=threading.Event(),
                        ocr_page=main.ocr_pdf_page,
                        document_mode="comic",
                    )
                assert translations, "Real local OCR did not supply any text"
                with pymupdf.open(root / "translated.pdf") as output:
                    assert output.page_count >= 1
                    assert "Salom dunyo" in " ".join(page.get_text() for page in output)
                    result = {"status": "passed", "output_pages": output.page_count,
                              "recognized_blocks": len(translations), "source_lang": "en",
                              "real_checkpoints": True, "real_cached_easyocr": True,
                              "translation": "stub", "network": "disabled", "training": False}
                print(json.dumps(result))
            finally:
                main.pdf_job_manager.shutdown()


if __name__ == "__main__":
    run()
