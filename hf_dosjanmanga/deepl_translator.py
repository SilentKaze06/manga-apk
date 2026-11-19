"""
DeepL Translation Engine for Manga & Comics.
Provides high-accuracy European and Asian language translation (especially Russian scanlation)
as a high-speed fallback when Gemini AI limits are reached.
"""
import json
import logging
import os
import re
import urllib.request
import urllib.error
from typing import List, Optional

logger = logging.getLogger("deepl_translator")

DEEPL_SUPPORTED_LANGUAGES = {
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr",
    "hu", "id", "it", "ja", "ko", "lt", "lv", "nl", "pl", "pt",
    "ro", "ru", "sk", "sl", "sv", "tr", "uk", "zh"
}

DEEPL_FORMALITY_LANGUAGES = {
    "de", "es", "fr", "it", "ja", "nl", "pl", "pt", "ru"
}


def colloquialise_manga_text(text: str, target_lang: str) -> str:
    """Softens stiff, formal or bureaucratic translation into lively manga dialogue."""
    if not text:
        return text
    norm = target_lang.lower().split("-")[0].strip()
    if norm == "ru":
        subs = [
            (r"\bВ рамках мероприятия, которое должно было стать празднованием\b", "Мы же празднуем"),
            (r"\bв рамках мероприятия, которое должно было стать празднованием\b", "мы же празднуем"),
            (r"\bнаправленного на укрепление отношений\b", "ради дружбы"),
            (r"\bнаправленного на укрепление\b", "для укрепления"),
            (r"\bнародными традициями\b", "мификами"),
            (r"\bЯ выражаю вам свою благодарность\b", "Спасибо тебе огромное"),
            (r"\bя выражаю вам свою благодарность\b", "спасибо тебе огромное"),
            (r"\bС этого момента я поддерживаю треххразовый режим питания\b", "Теперь я нормально ем три раза в день!"),
            (r"\bподдерживаю треххразовый режим питания\b", "ем три раза в день"),
            (r"\bнеобходимо для выполнения моих задач\b", "чтобы нормально всё сделать"),
            (r"\bявляется необходимым\b", "нужно"),
            (r"\bв соответствии с тем, кто я есть\b", "из-за того, кто я такой"),
            (r"\bв соответствии с\b", "как и"),
            (r"\bосуществлять\b", "делать"),
        ]
        res = text
        for pat, repl in subs:
            res = re.sub(pat, repl, res, flags=re.IGNORECASE)
        # Clean any remaining screentone numbers/symbols
        res = re.sub(r"\b\d{3,}'?\s*(?:i\s*ob'|ob'|io|s)\b", "", res, flags=re.IGNORECASE)
        return " ".join(res.split()).strip()
    return text


def is_deepl_supported(lang_code: str) -> bool:
    """Check if the language code is supported by DeepL (Uzbek 'uz' is NOT supported)."""
    if not lang_code:
        return False
    norm = lang_code.lower().split("-")[0].strip()
    return norm in DEEPL_SUPPORTED_LANGUAGES


def get_deepl_api_key(provided_key: Optional[str] = None) -> str:
    """Retrieve the DeepL auth key from parameter, environment, or persistent file."""
    if provided_key and str(provided_key).strip():
        return str(provided_key).strip()
    env_key = os.environ.get("DEEPL_API_KEY", "").strip()
    if env_key:
        return env_key
    key_file = os.environ.get("DEEPL_API_KEY_FILE", "/data/deepl_api_key.txt")
    if os.path.exists(key_file):
        try:
            with open(key_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
    return ""


def translate_dialogues_with_deepl(
    texts: List[str],
    target_lang: str,
    source_lang: str = "auto",
    api_key: Optional[str] = None,
    timeout: float = 15.0,
) -> Optional[List[str]]:
    """
    Translate a batch of dialogue strings using DeepL API.
    Returns None if DeepL does not support the target language (e.g. Uzbek 'uz')
    or if no valid API key is available.
    """
    if not texts:
        return []

    target_norm = target_lang.lower().split("-")[0].strip()
    if not is_deepl_supported(target_norm):
        # DeepL does not support Uzbek ('uz') or other unsupported languages.
        # Cleanly signal to caller to use current fallback (Google/MyMemory).
        return None

    key = get_deepl_api_key(api_key)
    if not key:
        return None

    # DeepL Free vs Pro endpoints
    endpoint = "https://api-free.deepl.com/v2/translate" if key.endswith(":fx") else "https://api.deepl.com/v2/translate"

    # Map target language to DeepL code
    target_param = target_norm.upper()
    if target_param == "EN":
        target_param = "EN-US"
    elif target_param == "PT":
        target_param = "PT-PT"

    # Pre-clean input texts from OCR screentone garbage tokens (e.g. 8123' i ob')
    cleaned_input = [
        re.sub(r"\b\d{3,}'?\s*(?:i\s*ob'|ob'|io|s)\b", "", t, flags=re.IGNORECASE).strip()
        for t in texts
    ]

    payload = {
        "text": cleaned_input,
        "target_lang": target_param,
    }

    if target_norm in DEEPL_FORMALITY_LANGUAGES:
        payload["formality"] = "prefer_less"

    if source_lang and source_lang != "auto":
        src_norm = source_lang.lower().split("-")[0].strip()
        if is_deepl_supported(src_norm):
            payload["source_lang"] = src_norm.upper()

    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"DeepL-Auth-Key {key}",
                "Content-Type": "application/json",
                "User-Agent": "MangaTranslator/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            translations = data.get("translations", [])
            if translations:
                raw_res = [t.get("text", "") for t in translations]
                while len(raw_res) < len(texts):
                    raw_res.append(texts[len(raw_res)])
                return [colloquialise_manga_text(t, target_lang) for t in raw_res[:len(texts)]]
    except urllib.error.HTTPError as he:
        logger.warning("DeepL HTTP xatosi (%s): %s", he.code, he)
    except Exception as exc:
        logger.warning("DeepL tarjima xatosi: %s", exc)

    return None


def translate_single_with_deepl(
    text: str,
    target_lang: str,
    source_lang: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 8.0,
) -> Optional[str]:
    """Translate a single sentence/dialogue with DeepL."""
    res = translate_dialogues_with_deepl([text], target_lang, source_lang or "auto", api_key=api_key, timeout=timeout)
    if res and len(res) > 0 and res[0]:
        return res[0]
    return None
