"""
Qwen3.5-9B (4-bit NF4) Vision-Language Engine for Comic/Manga/Manhwa/Manhua
OCR and direct scanlation-style translation.
Processes each detected text block crop independently (reading-order agnostic).
"""
import json
import logging
import math
import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

logger = logging.getLogger(__name__)

MIN_CROP_PIXELS = 128 * 28 * 28
MAX_CROP_PIXELS = 512 * 28 * 28

_LANG_DISPLAY_NAMES = {
    "uz": "Uzbek (O'zbek tili, Latin script)",
    "ru": "Russian (Русский язык)",
    "en": "English",
    "ja": "Japanese (日本語)",
    "ko": "Korean (한국어)",
    "zh": "Chinese (中文)",
    "es": "Spanish (Español)",
    "de": "German (Deutsch)",
    "fr": "French (Français)",
    "tr": "Turkish (Türkçe)",
    "ar": "Arabic (العربية)",
    "pt": "Portuguese (Português)",
    "it": "Italian (Italiano)",
    "nl": "Dutch (Nederlands)",
    "pl": "Polish (Polski)",
    "auto": "auto-detected source language",
}

_PUNCT_ONLY_RE = re.compile(r"^[?!.…\-—~～]+$")


def _strip_thinking_and_fences(raw: str) -> str:
    text = (raw or "").strip()
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if "<think>" in text:
            text = text.split("<think>", 1)[0].strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def clean_qwen_output_text(text: str, target_lang: Optional[str] = None) -> str:
    """
    Deterministic post-processor:
    - Strips <think> blocks and markdown formatting (_, *, `)
    - Normalizes curly apostrophes/quotes/dashes
    - Rejoins hyphenated English/Latin line-wrap splits
    - Normalizes whitespace
    """
    if not text:
        return ""
    res = _strip_thinking_and_fences(str(text))
    # Strip markdown emphasis/code markers (_, *, `) without removing valid apostrophes (')
    res = re.sub(r"[_*`]+", "", res)
    # Normalize apostrophes, quotes, dashes, ellipses
    res = (
        res.replace("ʻ", "'")
        .replace("ʼ", "'")
        .replace("‘", "'")
        .replace("’", "'")
        .replace("—", "-")
        .replace("–", "-")
        .replace("…", "...")
    )
    # Rejoin hyphenated word breaks across spaces/newlines (e.g. "com- fort" -> "comfort")
    res = re.sub(r"([A-Za-z]{2,})-\s+([A-Za-z']{2,})", r"\1\2", res)
    res = " ".join(res.split()).strip()
    if target_lang and str(target_lang).lower().startswith("uz"):
        # Ensure Uzbek o'/g' uses plain ASCII apostrophe
        res = re.sub(r"([oOgG])\s*['´ʹ]", r"\1'", res)
    return res


def parse_ocr_translate_json(raw_output: str, target_lang: Optional[str] = None) -> Tuple[str, str]:
    """
    Parses strict JSON {"src": "...", "tgt": "..."} returned by Qwen3.5-9B,
    falling back gracefully on minor formatting deviations.
    """
    cleaned = _strip_thinking_and_fences(raw_output)
    if not cleaned:
        return "", ""

    parsed: Optional[dict] = None
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            parsed = obj
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict):
                    parsed = obj
            except Exception:
                parsed = None

    if parsed is not None:
        src = str(parsed.get("src") or parsed.get("source") or parsed.get("original") or "")
        tgt = str(parsed.get("tgt") or parsed.get("target") or parsed.get("translation") or "")
        return clean_qwen_output_text(src), clean_qwen_output_text(tgt, target_lang=target_lang)

    # Regex fallback for slightly malformed JSON
    src_m = re.search(r'"src"\s*:\s*"((?:\\.|[^"\\])*)"', cleaned)
    tgt_m = re.search(r'"tgt"\s*:\s*"((?:\\.|[^"\\])*)"', cleaned)
    if src_m or tgt_m:
        src_raw = src_m.group(1).replace('\\"', '"').replace("\\n", " ") if src_m else ""
        tgt_raw = tgt_m.group(1).replace('\\"', '"').replace("\\n", " ") if tgt_m else ""
        return clean_qwen_output_text(src_raw), clean_qwen_output_text(tgt_raw, target_lang=target_lang)

    return "", ""


def _prepare_crop_image(crop_img: Image.Image) -> Image.Image:
    img = crop_img.convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (64, 64), (255, 255, 255))
    area = w * h
    resample = getattr(Image, "Resampling", Image).BICUBIC
    if min(w, h) < 64:
        scale = min(3.0, 64.0 / float(min(w, h)))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        if nw * nh <= MAX_CROP_PIXELS:
            img = img.resize((nw, nh), resample)
            w, h = img.size
            area = w * h
    if area > MAX_CROP_PIXELS:
        scale = math.sqrt(MAX_CROP_PIXELS / float(area))
        nw, nh = max(32, int(w * scale)), max(32, int(h * scale))
        img = img.resize((nw, nh), resample)
    return img


def _build_ocr_translate_prompt(target_lang: str, source_lang: str = "auto") -> str:
    tgt_code = (target_lang or "en").lower().split("-")[0]
    tgt_name = _LANG_DISPLAY_NAMES.get(tgt_code, target_lang)
    src_code = (source_lang or "auto").lower().split("-")[0]
    src_hint = (
        f"The source language is { _LANG_DISPLAY_NAMES.get(src_code, source_lang) }."
        if src_code != "auto"
        else "Auto-detect the source language (Japanese, Korean, Chinese, English, Russian, etc.)."
    )

    lang_rules = ""
    if tgt_code == "uz":
        lang_rules = (
            "For Uzbek, use natural lively conversational scanlation style in Latin script "
            "(ASCII apostrophe ' for o' and g', never dry academic phrasing)."
        )
    elif tgt_code == "ru":
        lang_rules = (
            "For Russian, use natural, expressive, conversational manga/anime scanlation style "
            "(never dry literal or bureaucratic phrasing)."
        )
    else:
        lang_rules = f"Translate into natural, conversational comic dialogue in {tgt_name}."

    return (
        f"Read the comic/manga/manhwa text in this cropped region and translate it into {tgt_name}.\n"
        f"{src_hint}\n"
        "Rules:\n"
        "1. Transcribe the exact dialogue or narration text cleanly (rejoin words split by a line-break hyphen like LIGHT-\\nNING -> LIGHTNING; read italic fonts and white-on-dark/colored text accurately).\n"
        "2. If the crop contains ONLY sound effects, onomatopoeia, or background noise (e.g. MURM, ZRK, ZRRK, FWP, POP, BAM, SFX) with no speech/narration, return {\"src\": \"\", \"tgt\": \"\"}.\n"
        "3. If the text is standalone punctuation (?, !, ?!, ...), keep the exact punctuation in both src and tgt.\n"
        f"4. {lang_rules} Translate meaning naturally — NEVER phoneticize or transliterate ordinary words (e.g. translate BYE as ПОКА/XAYR, not БАЙ).\n"
        "5. NEVER output markdown formatting (_, *, `), explanations, or extra text.\n"
        'Return ONLY valid JSON: {"src": "...", "tgt": "..."}'
    )


def _build_ocr_only_prompt(source_lang: str = "auto") -> str:
    src_code = (source_lang or "auto").lower().split("-")[0]
    src_hint = (
        f"The text language is {_LANG_DISPLAY_NAMES.get(src_code, source_lang)}."
        if src_code != "auto"
        else "Auto-detect the language."
    )
    return (
        "Transcribe the exact comic/manga/manhwa dialogue or narration text shown in this cropped image.\n"
        f"{src_hint}\n"
        "Rules:\n"
        "1. Join wrapped lines into natural sentences (rejoin hyphenated word splits across lines).\n"
        "2. Read italic text and white-on-black/colored boxes accurately.\n"
        "3. If the crop contains ONLY decorative background SFX or no readable text, return an empty string.\n"
        "4. Output ONLY the plain transcribed text with NO markdown (_, *, `) and NO explanations."
    )


_shared_qwen_engine = None
_shared_qwen_lock = threading.Lock()


def get_shared_qwen_engine(model_dir: Optional[str] = None) -> "QwenVLEngine":
    global _shared_qwen_engine
    if _shared_qwen_engine is not None:
        return _shared_qwen_engine
    with _shared_qwen_lock:
        if _shared_qwen_engine is None:
            _shared_qwen_engine = QwenVLEngine(model_dir=model_dir)
    return _shared_qwen_engine


class QwenVLEngine:
    """
    Thread-safe singleton wrapper for Qwen/Qwen3.5-9B (4-bit NF4 on CUDA T4).
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        use_4bit: Optional[bool] = None,
        min_pixels: int = MIN_CROP_PIXELS,
        max_pixels: int = MAX_CROP_PIXELS,
    ):
        self.model_dir = model_dir or os.environ.get("MANGA_QWEN_MODEL_DIR", "/opt/models/qwen3_5_9b")
        if use_4bit is None:
            self.use_4bit = os.environ.get("MANGA_QWEN_4BIT", "1").lower() in ("1", "true", "yes")
        else:
            self.use_4bit = bool(use_4bit)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self._model = None
        self._processor = None
        self._load_failed = False
        self._lock = threading.RLock()

    def _resolve_model_dir(self) -> Optional[Path]:
        candidates = [
            Path(self.model_dir),
            Path("/opt/models/qwen3_5_9b"),
            Path(__file__).resolve().parent / "checkpoints" / "qwen3_5_9b",
        ]
        for cand in candidates:
            if cand.is_dir() and (cand / "config.json").is_file():
                return cand
        return None

    def load_model(self):
        with self._lock:
            if self._model is not None and self._processor is not None:
                return self._model, self._processor
            if self._load_failed:
                return None, None

            local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            resolved_dir = self._resolve_model_dir()
            if resolved_dir is None:
                if local_only:
                    logger.warning("Qwen3.5-9B model dir missing at %s under HF_HUB_OFFLINE=1", self.model_dir)
                    self._load_failed = True
                    return None, None
                model_source = "Qwen/Qwen3.5-9B"
            else:
                model_source = str(resolved_dir)

            if not torch.cuda.is_available() and self.use_4bit:
                logger.warning("CUDA unavailable; skipping 4-bit Qwen3.5-9B load.")
                self._load_failed = True
                return None, None

            try:
                from transformers import AutoModelForImageTextToText, AutoProcessor

                processor = AutoProcessor.from_pretrained(
                    model_source,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                    local_files_only=local_only or (resolved_dir is not None),
                )
                if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
                    processor.tokenizer.padding_side = "left"

                load_kwargs: Dict = {
                    "device_map": "auto",
                    "local_files_only": local_only or (resolved_dir is not None),
                }
                if self.use_4bit and torch.cuda.is_available():
                    from transformers import BitsAndBytesConfig

                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    load_kwargs["dtype"] = torch.float16
                else:
                    load_kwargs["dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

                try:
                    model = AutoModelForImageTextToText.from_pretrained(model_source, **load_kwargs)
                except TypeError:
                    # Fallback for older transformers accepting torch_dtype instead of dtype
                    dtype_val = load_kwargs.pop("dtype", None)
                    if dtype_val is not None:
                        load_kwargs["torch_dtype"] = dtype_val
                    model = AutoModelForImageTextToText.from_pretrained(model_source, **load_kwargs)

                model.eval()
                self._model = model
                self._processor = processor
                logger.info("Qwen3.5-9B loaded from %s (4bit=%s)", model_source, self.use_4bit)
                return self._model, self._processor
            except Exception as exc:
                logger.error("Failed to load Qwen3.5-9B: %s", exc, exc_info=True)
                self._load_failed = True
                return None, None

    @property
    def is_available(self) -> bool:
        model, processor = self.load_model()
        return model is not None and processor is not None

    def _apply_template_text(self, processor, messages: List[Dict]) -> str:
        try:
            rendered = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
        return rendered

    def _generate_batch(self, crops: List[Image.Image], prompt_text: str, max_new_tokens: int = 160) -> List[str]:
        model, processor = self.load_model()
        if model is None or processor is None:
            return [""] * len(crops)
        if not crops:
            return []

        prepared_imgs = [_prepare_crop_image(c) for c in crops]
        texts = []
        for img in prepared_imgs:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            texts.append(self._apply_template_text(processor, messages))

        with self._lock:
            if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
                processor.tokenizer.padding_side = "left"
            inputs = processor(
                text=texts,
                images=prepared_imgs,
                padding=True,
                return_tensors="pt",
            )
            target_device = getattr(model, "device", None)
            if target_device is None or str(target_device) == "meta":
                target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs = inputs.to(target_device)
            input_len = int(inputs["input_ids"].shape[1])

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                )

        decoded: List[str] = []
        for i in range(len(crops)):
            gen_ids = outputs[i][input_len:]
            decoded.append(processor.decode(gen_ids, skip_special_tokens=True))
        return decoded

    def ocr_and_translate_crop(
        self,
        crop_img: Image.Image,
        target_lang: str,
        source_lang: str = "auto",
    ) -> Tuple[str, str]:
        """Single-crop OCR + direct translation (completely order-independent)."""
        prompt = _build_ocr_translate_prompt(target_lang=target_lang, source_lang=source_lang)
        outputs = self._generate_batch([crop_img], prompt, max_new_tokens=160)
        if not outputs:
            return "", ""
        src, tgt = parse_ocr_translate_json(outputs[0], target_lang=target_lang)
        if src and _PUNCT_ONLY_RE.fullmatch(src) and not tgt:
            tgt = src
        return src, tgt

    def ocr_and_translate_crops(
        self,
        crops: List[Image.Image],
        target_lang: str,
        source_lang: str = "auto",
        batch_size: int = 4,
    ) -> List[Tuple[str, str]]:
        """Mini-batched OCR + translation across unordered crops."""
        if not crops:
            return []
        if not self.is_available:
            return [("", "") for _ in crops]

        prompt = _build_ocr_translate_prompt(target_lang=target_lang, source_lang=source_lang)
        results: List[Tuple[str, str]] = []
        step = max(1, int(batch_size))
        for start in range(0, len(crops), step):
            chunk = crops[start:start + step]
            try:
                raw_outputs = self._generate_batch(chunk, prompt, max_new_tokens=160)
                for raw in raw_outputs:
                    src, tgt = parse_ocr_translate_json(raw, target_lang=target_lang)
                    if src and _PUNCT_ONLY_RE.fullmatch(src) and not tgt:
                        tgt = src
                    results.append((src, tgt))
            except Exception as batch_exc:
                logger.warning("Qwen batch OCR+translate failed (%s); falling back to single crops", batch_exc)
                for single_crop in chunk:
                    try:
                        results.append(self.ocr_and_translate_crop(single_crop, target_lang, source_lang))
                    except Exception as single_exc:
                        logger.warning("Qwen single crop OCR+translate failed: %s", single_exc)
                        results.append(("", ""))
        return results

    def ocr_crop(self, crop_img: Image.Image, source_lang: str = "auto") -> str:
        """Clean OCR-only mode (used when Gemini override is requested)."""
        prompt = _build_ocr_only_prompt(source_lang=source_lang)
        outputs = self._generate_batch([crop_img], prompt, max_new_tokens=128)
        if not outputs:
            return ""
        return clean_qwen_output_text(outputs[0])

    def ocr_crops(
        self,
        crops: List[Image.Image],
        source_lang: str = "auto",
        batch_size: int = 4,
    ) -> List[str]:
        """Mini-batched OCR-only extraction across unordered crops."""
        if not crops:
            return []
        if not self.is_available:
            return ["" for _ in crops]

        prompt = _build_ocr_only_prompt(source_lang=source_lang)
        results: List[str] = []
        step = max(1, int(batch_size))
        for start in range(0, len(crops), step):
            chunk = crops[start:start + step]
            try:
                raw_outputs = self._generate_batch(chunk, prompt, max_new_tokens=128)
                results.extend(clean_qwen_output_text(r) for r in raw_outputs)
            except Exception as batch_exc:
                logger.warning("Qwen batch OCR failed (%s); falling back to single crops", batch_exc)
                for single_crop in chunk:
                    try:
                        results.append(self.ocr_crop(single_crop, source_lang=source_lang))
                    except Exception as single_exc:
                        logger.warning("Qwen single crop OCR failed: %s", single_exc)
                        results.append("")
        return results
