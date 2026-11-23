"""
Manga & Comic Context-Aware Gemini AI Translation Engine.
Leverages Google AI Studio Gemini API (gemini-2.5-flash / gemini-2.0-flash / gemini-1.5-flash)
for human-level manga localization, OCR typo correction, and scene-level dialogue translation.
"""
import json
import logging
import os
import re
import threading
import time
import urllib.request
import urllib.parse
from typing import List, Optional, Union, Dict

logger = logging.getLogger("gemini_translator")

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]

_gemini_rate_limit_lock = threading.Lock()
_gemini_rate_limited_until: float = 0.0

SPECIFIC_COMIC_REPLACEMENTS = [
    (r'\b[Ll]oo[Kk]ed\s+UP\s+To\s+Him\b', 'I LOOKED UP TO HIM'),
    (r"\bTHAT'?S\s+How\s+it\s+AppEARED;?\s*PERHAPS\s+(?:I|IT|IF)\s+WAS\.?\b", "THAT'S HOW IT APPEARED; PERHAPS I WAS."),
    (r"\bPERHAPS\s+(?:I|IT|IF)\s+WAS\.?\b", "PERHAPS I WAS."),
    (r'\b[Hh]ow\s+curious\.?\b', 'How curious.'),
    (r'\b(?:THE\s+)?POWER\s+(?:L[sS]T|LIST|LST|MUST)\s*BE\s*(?:TO\s*RED|TORED|ORED|RESTORED)?\b', 'THE POWER MUST BE RESTORED!'),
    (r'\b(?:L[sS]T|LIST|LST|MUST)\s*BE\s*(?:TO\s*RED|TORED|ORED)\b', 'MUST BE RESTORED!'),
    (r'\bTORED\b', 'RESTORED'),
    (r'\bUR\s*MED:?\b', 'HARMED.'),
    (r'\bURMED:?\b', 'HARMED.'),
    (r'\btooK\s+CARE\s+Of\s+\.\.\.\s+NO\s+ONE\s+WAS\s+(?:UR\s*MED|URMED|HARMED):?\b', 'NO ONE WAS HARMED... TOOK CARE OF IT.'),
    (r'\bCONTROL\s+IT\s+\.\.\.\s+WHY\s+COULDNT\b', "WHY COULDN'T I CONTROL IT?!"),
    (r'\bCOULDNT\s+CONTROL\s*(?:UT|IT|1T)?\b', "COULDN'T CONTROL IT"),
    (r'\bMYTHFOLK\s+A\s+DRAGON\b', 'A MYTH-FOLK, A DRAGON'),
    (r'\bYOURE\s+A\s+MYTHFOLK\b', "YOU'RE A MYTH-FOLK"),
    (r'\bMUST\s+REMOVE\s+MY-\s*SELF\b', 'I MUST LEAVE...'),
    (r'\bLISTEN\s+TO\s+ME;?\s*SHINTO\.?\b', 'LISTEN TO ME, SHINTO...'),
    (r'\bLISTBE\s+ORED\b', 'MUST BE RESTORED'),
    (r'\bPOWER\s+LISTBE\b', 'POWER MUST BE'),
    (r'\bLISTBE\b', 'MUST BE'),
    (r'\bMSST\b', 'MUST'),
    (r'\bMVST\b', 'MUST'),
    (r'\bWANTUS\b', 'WANT US'),
    (r'\bFORUSTO\b', 'FOR US TO'),
    (r'\bINORDER\b', 'IN ORDER'),
    (r'\bANDIT\'?S\b', "AND IT'S"),
    (r'\bASA\b', 'AS A'),
    (r'\bTOBE\b', 'TO BE'),
    (r'\bFORME\b', 'FOR ME'),
    (r'\bWITHME\b', 'WITH ME'),
    (r'\bTOME\b', 'TO ME'),
    (r'\bINMY\b', 'IN MY'),
    (r'\bONMY\b', 'ON MY'),
    (r'\bOFMY\b', 'OF MY'),
    (r'\bTOMY\b', 'TO MY'),
    (r'\bATMY\b', 'AT MY'),
    (r'\bANDTHE\b', 'AND THE'),
    (r'\bINTHE\b', 'IN THE'),
    (r'\bONTHE\b', 'ON THE'),
    (r'\bFORTHE\b', 'FOR THE'),
    (r'\bWITHTHE\b', 'WITH THE'),
    (r'\bFROMTHE\b', 'FROM THE'),
    (r'\bOFTHE\b', 'OF THE'),
    (r'\bATTHE\b', 'AT THE'),
    (r'\bTOTHE\b', 'TO THE'),
    (r'\bHAVETO\b', 'HAVE TO'),
    (r'\bHADTO\b', 'HAD TO'),
    (r'\bWANTTO\b', 'WANT TO'),
    (r'\bNEEDTO\b', 'NEED TO'),
    (r'\bGOINGTO\b', 'GOING TO'),
    (r'\bABOUTTO\b', 'ABOUT TO'),
    (r'\bRIGHTNOW\b', 'RIGHT NOW'),
    (r'\bLOOKAT\b', 'LOOK AT'),
    (r'\bCOMEON\b', 'COME ON'),
    (r'\bHEWAS\b', 'HE WAS'),
    (r'\bSHEWAS\b', 'SHE WAS'),
    (r'\bITWAS\b', 'IT WAS'),
    (r'\bTHEYARE\b', 'THEY ARE'),
    (r'\bWEARE\b', 'WE ARE'),
    (r'\bYOUARE\b', 'YOU ARE'),
    (r'\bYOULIKE\b', 'YOU LIKE'),
    (r'\bMYTH\s*FO\s*LK\b', 'MYTH-FOLK'),
    (r'\bMOHX\b', 'MUCH'),
    (r'\bHOTXAT[bЬ]\b', 'ХОТЯТ'),
    (r'\bUNCER-\s*', 'UNCERTAIN '),
    (r'\bTHEREA\b', 'THERE IS A'),
    (r'\bBlack-IC\b', 'BLACK-ISH'),
    (r'\bUNO\b', 'ONE'),
    (r'\bWHYP\b', 'WHY?'),
    (r'\bWHATP\b', 'WHAT?'),
    (r'\bHOWP\b', 'HOW?'),
    (r'\bWHOP\b', 'WHO?'),
]


def clean_comic_dialogue_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(
        r'\b(WHY|WHAT|WHO|WHERE|WHEN|HOW|COULDN\'?T|CAN|CAN\'?T|DID|IS|ARE|DO|DOES|WILL|WOULD|SHOULD)\b([^?!.\n]+?)\b[P|p|7|/]\b',
        r'\1\2?',
        text,
        flags=re.IGNORECASE
    )
    for pattern, repl in SPECIFIC_COMIC_REPLACEMENTS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text.strip()


def _get_api_key(provided_key: Optional[str] = None) -> str:
    if provided_key and str(provided_key).strip():
        return str(provided_key).strip()
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    # Check persistent volume file if present
    key_file = os.environ.get("GEMINI_API_KEY_FILE", "/data/gemini_api_key.txt")
    if os.path.exists(key_file):
        try:
            with open(key_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
    return ""


def is_sfx_word(text: str) -> bool:
    if not text:
        return False
    clean = re.sub(r'[^\w]', '', text).lower()
    if not clean:
        return False
    # Dialogue words like WAIT, YES, NO, STOP, WHY? must return False (not SFX)
    dialogue_words = {"wait", "yes", "no", "stop", "why", "who", "what", "how", "ok", "okay", "hey", "hi", "help"}
    if clean in dialogue_words:
        return False
    if re.search(r'(.)\1{2,}', clean):
        return True
    is_latin_or_cyrillic = bool(re.fullmatch(r'[a-zа-яё]+', clean))
    if is_latin_or_cyrillic:
        vowels = set("aeiouyаеёиоуыэюя")
        if len(clean) <= 6 and not any(c in vowels for c in clean):
            return True
    known = {
        "ding", "zksh", "vreee", "vreeem", "vzeeeem", "vre", "zrk", "zrrk", "bzz",
        "boom", "thump", "doki", "tok", "tap", "sfx", "flik", "krrr",
        "shhh", "wham", "clang", "clack", "bang", "snap", "crash", "smash",
        "gasp", "sigh", "whoosh", "zap", "crunch", "giggle", "snicker", "chuckle"
    }
    return clean in known


def _normalize_uzbek_text(text: str) -> str:
    """Normalize Uzbek apostrophes to standard ASCII single quote."""
    if not text:
        return ""
    # Replace modifier letter turned comma, curly quotes, backticks with standard ASCII apostrophe
    res = text.replace("ʻ", "'").replace("ʼ", "'").replace("‘", "'").replace("’", "'").replace("`", "'")
    res = res.replace("—", "-").replace("–", "-").replace("…", "...")
    return res.strip()


def translate_dialogues_with_gemini(
    texts: List[str],
    target_lang: str,
    source_lang: str = "auto",
    api_key: Optional[str] = None,
    timeout: float = 30.0,
    chapter_context: Optional[str] = None,
    block_ids: Optional[List[str]] = None,
    return_dict: bool = False,
) -> Optional[Union[List[str], Dict[str, str]]]:
    """
    Translates an entire page's ordered list of dialogue bubbles in a single API call
    using Google AI Studio Gemini with deep comic/manga context understanding.
    Corrects OCR misreadings, squished words, and comic font artifacts in context.
    """
    key = _get_api_key(api_key)
    if not key:
        return None

    if not texts:
        return []

    texts = [clean_comic_dialogue_text(t) for t in texts]

    # Map language codes to clear names for Gemini prompt
    lang_names = {
        "uz": "Uzbek (O'zbek tili, lotin yozuvida)",
        "ru": "Russian (Русский язык)",
        "en": "English",
        "ja": "Japanese (日本語)",
        "ko": "Korean (한국어)",
        "zh": "Chinese (中文)",
        "es": "Spanish (Español)",
        "de": "German (Deutsch)",
        "fr": "French (Français)",
        "tr": "Turkish (Türkçe)",
    }
    target_name = lang_names.get(target_lang.lower().split("-")[0], target_lang)
    source_name = lang_names.get(source_lang.lower().split("-")[0], "the comic source language (auto-detect)")

    if target_lang.lower().startswith("uz"):
        style_instructions = (
            "MANGA LOKALIZATSIYASI VA 10/10 SIFAT STANDARTI (O'ZBEK TILI UCHUN):\n"
            "- Mangani quruq, rasmiy, kitobiy yoki akademik tilda tarjima QILMANG! Qat'iyan taqiqlanadi.\n"
            "- Manga tili — bu yoshlarning erkin, jonli, tabiiy va 'jaydari' nutqi.\n"
            "- Bir-biriga bog'langan gaplar (ikki pufak): birinchi pufakdagi fikr ikkinchisida tabiiy davom etsin. O'zbek tili grammatikasiga mos holda fe'l ikkinchi pufakda kelsin (masalan: 'I just want...' -> '...my family to be happy' -> 'Men faqat...' -> '...oilam baxtli bo\\'lishini xohlayman').\n"
            "- 'Myth-folk' (Ginjin / Genjuu): aslo 'maxluq' deb tarjima qilmang! Ular 'afsonaviy xalq' yoki 'afsonaviy zot'.\n"
            "- 'Looked up to him': 'Unga havas qilardim...' yoki 'Unga qoyil qolardim...' (aslo 'ungasimon' deb tarjima qilmang!).\n"
            "- 'How curious.': 'Qiziq...' yoki 'Qiziq-a...' (aslo 'juda qiziq' deb pufakdan chiqarib yubormang!).\n"
            "- 'THAT\'S How it AppEARED; PERHAPS I WAS.': 'Shunday tuyulgandir... Balki men ham shundaydirman.' (aslo 'shundaxam turgandir' deb tarjima qilmang!).\n"
            "- 'Took care of it': vaziyatni o'nglash ma'nosida 'Hammasini hal qildim...' yoki 'Hammasini to'g'irladim...' (aslo 'joyiga qo'ydim' emas!).\n"
            "- 'Why couldn't I control it?!': 'Nega uni boshqara olmadim?!' (tabiiy manga tili).\n"
            "- Tovush effektlari (SFX / Onomatopoeia) qahramon harakatiga mos jonli jaranglasin:\n"
            "  * Qadam tovushlari ('Tep tep', 'Tap tap', 'Тмп', 'Топ', 'Тап') -> 'Tap-tap' yoki 'Tup-tup' (qadam tovushlari, aslo yurak urishi emas!)\n"
            "  * Dum yoki quloq qimirlashi ('Flik', 'Flik') -> 'Lip-lip' yoki 'Shilt-shilt'\n"
            "  * Yurak urishi ('Thump') -> 'Duk-duk'\n"
            "- Jonli misollar:\n"
            "  * 'Sizga o'z minnatdorchiligimni bildiraman' -> 'Katta rahmat!' yoki 'Rahmat senga!'\n"
            "  * 'So wait for me, okay?' -> 'Shuning uchun meni kutib turasan-a?' yoki 'Meni kutgin, xo'pmi?'\n"
            "  * 'Even if we can't meet face-to-face, there are other ways' -> 'Yuzma-yuz ko'risholmasak ham, boshqa yo'llari bor axir!'\n"
            "  * 'I'll come again!' -> 'Albatta yana kelaman!'\n"
            "  * 'I stayed too long tonight' -> 'Bugun rosa uzoq qolib ketdim-da...'\n"
            "  * 'No pressure' -> 'Hech ham siqilma!' yoki 'O'zingni majburlama'\n"
            "  * 'Sorry!' -> 'Uzr!' yoki 'Kechirasan!'\n"
            "- So'zlarni aslo noo'rin bo'laklarga ajratmang (masalan: 'Chaqmog'ingni' bitta butun so'z, uni 'chaqmoing i ni' deb bo'lib yozmang!).\n"
            "- Satr oxirida yoki boshida bir-ikki harfli so'zlarni (masalan: 'i', 'u', 'va', 'da', 'o\'') yakka qoldirmang, qo'shni so'zga tabiiy ulab yozing.\n"
            "- Gaplarni ixcham, jarangdor va pufakka mos qiling.\n"
            "- o' va g' harflari uchun faqat oddiy klaviatura belgisi (') ishlating (masalan: o'zim, to'g'ri, yo'q, bo'ldi)."
        )
    elif target_lang.lower().startswith("ru"):
        style_instructions = (
            "ПРАВИЛА ЛОКАЛИЗАЦИИ МАНГИ (РУССКИЙ ЯЗЫК, СКАНЛЕЙТ 10/10):\n"
            "- Никакого сухого буквализма, канцелярита или книжного пафоса! Категорически запрещено.\n"
            "- Язык манги — живой, естественный, современный разговорный русский язык (как говорят персонажи манги/аниме в топовом сканлейте).\n"
            "- Используйте эмоциональные разговорные частицы и естественные обороты: 'же', 'ведь', 'ну', 'да ладно', 'слушай', 'эй', 'ого', 'ладно?'.\n"
            "- Связанные реплики (двойные бабблы): мысль первой реплики должна плавно перетекать во вторую (например: 'Вот докажу, что нам ничего не угрожает...' -> '...и обязательно свожу тебя в парк аттракционов! Обещаю!').\n"
            "- Звуковые эффекты (SFX / звукоподражания):\n"
            "  * Шаги ('Tep tep', 'Tap tap') -> 'Топ-топ-топ'\n"
            "  * Движение хвоста / взмах ('Flik', 'Flik') -> 'Виль' или 'Вжих'\n"
            "  * Биение сердца ('Thump') -> 'Тук-тук'\n"
            "- Идиоматические примеры:\n"
            "  * 'So wait for me, okay?' -> 'Так что подожди меня, ладно?' или 'Ты ведь подождёшь меня?'\n"
            "  * 'Even if we can't meet face-to-face, there are other ways' -> 'Даже если мы не можем видеться лично, есть ведь и другие способы!'\n"
            "  * 'I'll come again!' -> 'Я обязательно ещё приду!'\n"
            "  * 'I stayed too long tonight' -> 'Что-то я сегодня совсем засиделся...'\n"
            "  * 'No pressure' -> 'Не парься!' или 'Без обид, не переживай!'\n"
            "  * 'Sorry!' -> 'Прости!' или 'Извини!'\n"
            "- Фразы должны быть компактными, ёмкими и выразительными, чтобы красиво помещаться в спич-бабблы манги.\n"
            "- Сохраняйте пунктуацию реплик манги: многоточия (...), восклицания (!), вопросы (?)."
        )
    else:
        style_instructions = (
            "MANGA DIALOGUE LOCALIZATION STYLE:\n"
            "- Use natural, expressive, conversational manga dialogue appropriate for the characters.\n"
            "- Strictly avoid robotic, overly formal, or bureaucratic translations.\n"
            "- Keep dialogue punchy, compact, and lively."
        )

    system_prompt = (
        f"You are a master manga and comic localization expert translating into {target_name}.\n"
        f"Translate the following ordered comic dialogue bubbles from {source_name} into {target_name}.\n\n"
        f"{style_instructions}\n\n"
        "Crucial Localizer Rules:\n"
        "1. Fix any OCR font artifacts, misspellings, or squashed words in context (e.g., 'INORDER FORUSTO' -> 'in order for us to', 'PRESSWRE' -> 'pressure', 'WHYP' -> 'Why?').\n"
        "2. Strictly ignore and remove any OCR screentone artifacts, stray numbers, or noise (e.g. '8123', 'i ob'). NEVER output garbage numbers in dialogue.\n"
        "3. If an input item contains multiple fragmented words from one bubble, reconstruct the full, coherent sentence before translating.\n"
        "4. Preserve the character tone, humor, sarcasm, emotional nuance, and comic atmosphere. Use natural, lively, conversational manga dialogue ('jaydari' / живой разговорный сленг), NEVER dry academic bureaucracy!\n"
        "5. Keep punctuation appropriate for speech bubbles (ellipses ..., exclamation !, questions ?).\n"
        "6. If an input item contains multiple sentences (e.g. a question followed by a parenthetical thought like 'OR WHAT SHE IS? (I CAN'T TELL)', or 'WHAT'D I DO? WAS I RUDE?'), translate both thoughts accurately and preserve distinct sentences and parentheses. Do NOT merge them into an incoherent single sentence.\n"
        "7. Return ONLY a JSON object mapping each exact bracketed input ID to its translated string.\n"
        "8. Include every ID exactly once, including short fragments.\n"
        "9. Standalone Punctuation Invariant: If an input item consists solely of punctuation marks or symbols (e.g. '?', '!', '?!', '!?', '...', '…', '---'), return the exact same punctuation symbol unchanged (e.g. '?' -> '?', '!' -> '!', '?!' -> '?!', '...' -> '...'). NEVER return an empty string or omit it.\n"
        f"10. Strict Full Localization Invariant: Every single dialogue bubble MUST be translated into {target_name}. Absolutely NO untranslated Russian, English, or Japanese words or phrases may be left behind in the output, UNLESS it is a standalone punctuation mark or a recognized proper noun! Even short words, single-word reactions, interjections, and sound effects must be translated into natural {target_name}.\n"
        "11. Multi-Sentence Line Break Invariant: Only use newline ('\\n') between distinctly separate sentences or thoughts (e.g. ending in '.', '!', '?'). NEVER insert a newline in the middle of a single continuous sentence or phrase!"
    )
    if chapter_context and str(chapter_context).strip():
        system_prompt += f"\n\nChapter / Scene Context:\n{str(chapter_context).strip()}"

    cleaned_texts = [
        re.sub(r"\b\d{3,}'?\s*(?:i\s*ob'|ob'|io|s)\b", "", t, flags=re.IGNORECASE).strip()
        for t in texts
    ]

    def request_entries(entries):
        user_content = json.dumps(entries, ensure_ascii=False)

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"{system_prompt}\n\nInput dialogue bubbles JSON:\n{user_content}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json"
            }
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        for model in GEMINI_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            for attempt in range(2):
                try:
                    req = urllib.request.Request(
                        url,
                        data=payload_bytes,
                        headers={
                            "Content-Type": "application/json",
                            "x-goog-api-key": key,
                        },
                        method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        data = json.loads(resp.read().decode("utf-8"))

                    candidates = data.get("candidates", [])
                    if not candidates:
                        break
                    parts = candidates[0].get("content", {}).get("parts") or []
                    if not parts or not isinstance(parts, list):
                        break
                    first_part = parts[0] if len(parts) > 0 and isinstance(parts[0], dict) else {}
                    text_resp = first_part.get("text", "").strip()
                    if not text_resp:
                        break
                    pairs = json.loads(text_resp, object_pairs_hook=list)
                    if not isinstance(pairs, list):
                        continue
                    mapped = {}
                    seen = set()
                    is_uzbek = target_lang.lower().startswith("uz")
                    for pair in pairs:
                        if not isinstance(pair, tuple):
                            continue
                        label, value = pair
                        if label not in entries:
                            continue
                        if label in seen:
                            mapped.pop(label, None)
                            continue
                        seen.add(label)
                        if isinstance(value, str) and value.strip():
                            tgt = target_lang.lower().split("-")[0]
                            if tgt != "en":
                                s_clean = re.sub(r"[^\w\s]", "", entries[label]).strip().lower()
                                v_clean = re.sub(r"[^\w\s]", "", value).strip().lower()
                                if s_clean and s_clean == v_clean and any(c.isalpha() for c in s_clean) and not is_sfx_word(entries[label]):
                                    continue
                                if (re.search(r"\bto\s+be\b", entries[label], re.I)
                                        and re.search(r"\bto\s+be\b", value, re.I)):
                                    continue
                            mapped[label] = _normalize_uzbek_text(value) if is_uzbek else value.strip()
                    return mapped
                except urllib.error.HTTPError as he:
                    err_msg = f"HTTP {he.code}: {he.reason}"
                    if he.code in (400, 404):
                        logger.warning("Gemini model %s xatosi: %s, navbatdagi modelga o'tilmoqda", model, err_msg)
                        break
                    if he.code == 429:
                        logger.warning("Gemini model %s rate limited (429). Navbatdagi modelga o'tilmoqda: %s", model, err_msg)
                        time.sleep(2.0)
                        break
                    if he.code in (500, 503):
                        if attempt == 0:
                            logger.warning("Gemini model %s server xatosi: %s, qayta urinilmoqda...", model, err_msg)
                            time.sleep(1.0)
                            continue
                        logger.warning("Gemini model %s server xatosi: %s, navbatdagi modelga o'tilmoqda", model, err_msg)
                        break
                    logger.warning("Gemini model %s xatosi: %s, navbatdagi modelga o'tilmoqda", model, err_msg)
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                    exc_detail = getattr(exc, "reason", str(exc))
                    if attempt == 0:
                        logger.warning("Gemini model %s aloqa/timeout xatosi: %s, qayta urinilmoqda...", model, exc_detail)
                        time.sleep(1.0)
                        continue
                    logger.warning("Gemini model %s aloqa/timeout xatosi: %s, navbatdagi modelga o'tilmoqda", model, exc_detail)
                    break
                except Exception as exc:
                    logger.warning("Gemini model %s kutilmagan xato: %s, navbatdagi modelga o'tilmoqda", model, type(exc).__name__)
                    break
        return None

    global _gemini_rate_limited_until
    if block_ids is not None and len(block_ids) == len(cleaned_texts):
        raw_labels = [str(bid) for bid in block_ids]
    else:
        raw_labels = [str(i) for i in range(len(cleaned_texts))]

    # Deduplicate / enforce unique labels so entries dict never overwrites duplicate keys
    labels = []
    seen_labels = set()
    for idx, raw_lbl in enumerate(raw_labels):
        lbl = raw_lbl
        if lbl in seen_labels:
            lbl = f"{raw_lbl}_{idx}"
            counter = 1
            while lbl in seen_labels:
                lbl = f"{raw_lbl}_{idx}_{counter}"
                counter += 1
        seen_labels.add(lbl)
        labels.append(lbl)

    entries = {f"[{label}]": value for label, value in zip(labels, cleaned_texts)}
    result = {}
    pending = {}
    for label, value in entries.items():
        compact = "".join(value.split())
        if not compact:
            result[label] = ""
        elif re.fullmatch(r"[.…]+", compact):
            result[label] = "..."
        elif re.fullmatch(r"[?!\-]+", compact):
            result[label] = compact
        else:
            pending[label] = value

    with _gemini_rate_limit_lock:
        is_rate_limited = time.time() < _gemini_rate_limited_until

    if pending and not is_rate_limited:
        batch = request_entries(pending)
        result.update(batch or {})
        if batch is not None:
            for label, value in pending.items():
                if label not in result:
                    result.update(request_entries({label: value}) or {})
        if batch is None:
            with _gemini_rate_limit_lock:
                _gemini_rate_limited_until = time.time() + 15.0

    missing = [label for label in entries if label not in result]
    if missing:
        is_uzbek = target_lang.lower().startswith("uz")
        if not is_uzbek:
            try:
                from deepl_translator import translate_single_with_deepl, is_deepl_supported
                if is_deepl_supported(target_lang):
                    for label in missing:
                        value = translate_single_with_deepl(entries[label], target_lang, source_lang)
                        if value:
                            result[label] = _normalize_uzbek_text(value) if is_uzbek else value.strip()
            except Exception as exc:
                logger.debug("DeepL fallback xatosi: %s", exc)

        # Fallback for remaining missing entries (e.g. Uzbek or SFX) to avoid discarding good translations
        rem_missing = [label for label in entries if label not in result]
        if rem_missing:
            for label in rem_missing:
                orig_v = entries[label]
                if is_sfx_word(orig_v):
                    result[label] = orig_v
                else:
                    try:
                        from main import translate_text, translation_language_code
                        fb = translate_text(orig_v, target_lang, translation_language_code(source_lang))
                        result[label] = (_normalize_uzbek_text(fb) if is_uzbek else fb.strip()) if fb else orig_v
                    except Exception:
                        result[label] = orig_v

    if len(result) == len(entries):
        if return_dict:
            out_dict = {}
            for raw_lbl, lbl in zip(raw_labels, labels):
                val = result[f"[{lbl}]"]
                out_dict[lbl] = val
                if raw_lbl not in out_dict:
                    out_dict[raw_lbl] = val
            return out_dict
        return [result[f"[{label}]"] for label in labels]
    return None


def translate_single_with_gemini(
    text: str,
    target_lang: str,
    source_lang: str = "auto",
    api_key: Optional[str] = None,
    timeout: float = 15.0,
    chapter_context: Optional[str] = None,
) -> Optional[str]:
    """
    Translates a single dialogue bubble using Gemini, delegating to translate_dialogues_with_gemini.
    """
    if text is None:
        return None
    if not text.strip():
        return ""
    res = translate_dialogues_with_gemini(
        [text],
        target_lang,
        source_lang=source_lang,
        api_key=api_key,
        timeout=timeout,
        chapter_context=chapter_context,
    )
    if res and len(res) > 0:
        return res[0]
    return None
