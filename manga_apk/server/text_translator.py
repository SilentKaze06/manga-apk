"""Text translation and OCR dialogue text post-processing engine."""

import html
import json
import logging
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Optional, List, Dict, Any

try:
    import wordninja
except ImportError:
    wordninja = None

logger = logging.getLogger("manga-server")

TRANSLATION_TIMEOUT_SECONDS = max(
    1.0, min(30.0, float(os.environ.get("MANGA_TRANSLATION_TIMEOUT_SECONDS", "8")))
)
_google_blocked_until = 0.0
_google_block_lock = threading.Lock()

def normalize_ocr_language(value: Optional[str]) -> str:
    """Normalize UI/API language aliases to the codes supported by this server."""
    code = (value or "").strip().lower()
    if not code:
        return "auto"
    aliases = {
        "jp": "ja", "jpn": "ja", "japanese": "ja",
        "kor": "ko", "korean": "ko",
        "chs": "zh", "cht": "zh", "chinese": "zh", "ch": "zh", "ch_sim": "zh",
        "zh-cn": "zh", "zh-tw": "zh", "zh-hant": "zh", "zh-hans": "zh",
        "eng": "en", "english": "en",
        "rus": "ru", "russian": "ru", "cyrillic": "ru",
        "fra": "fr", "french": "fr",
        "deu": "de", "german": "de",
        "spa": "es", "spanish": "es",
        "por": "pt", "portuguese": "pt",
        "ita": "it", "italian": "it",
        "tur": "tr", "turkish": "tr",
        "nld": "nl", "dutch": "nl",
        "pol": "pl", "polish": "pl",
        "uzb": "uz", "uzbek": "uz",
        "ara": "ar", "arabic": "ar",
    }
    return aliases.get(code, code)


def translation_language_code(value: Optional[str], default: str = "auto") -> str:
    """Map common language aliases to the codes accepted by translation providers."""
    raw = (value or default or "").strip().lower()
    if raw in {"zh-tw", "cht", "zh-hant"}:
        return "zh-TW"
    code = normalize_ocr_language(value or default)
    return {"zh": "zh-CN", "auto": "auto"}.get(code, code)


KEEP_WORDS = {
    "cannot", "into", "onto", "within", "without", "today", "tonight", "tomorrow",
    "maybe", "someone", "anyone", "everyone", "nothing", "something", "anything",
    "everything", "understand", "because", "before", "behind", "already", "myself",
    "yourself", "himself", "herself", "itself", "ourselves", "themselves", "another",
    "together", "forever", "sometimes", "somewhere", "anywhere", "everywhere", "nowhere",
    "always", "almost", "also", "although", "anyway", "become", "became", "becomes",
    "between", "beyond", "inside", "outside", "upon", "towards", "toward", "classroom",
    "blackboard", "blackout", "president", "lightning", "statement", "neighbor",
    "ordinary", "feeling", "expected", "restored", "school", "family", "happy",
    "senpai", "sensei", "tsumiki", "bankai", "kawaii", "sugoi", "nakama", "isekai",
    "ramen", "katana", "shinobi", "wassup", "ketchup", "backup", "pickup", "wakeup",
    "setup", "cleanup", "breakup", "checkup", "holdup", "lineup", "lookup", "makeup",
    "mixup", "startup", "asa", "uno",
}

_COMMON_P_WORDS = {
    "stop", "help", "jump", "sleep", "keep", "deep", "step", "trip", "ship", "shop",
    "drop", "whip", "cup", "top", "cap", "map", "up", "yup", "yep", "wassup", "asap",
    "mvp", "vip", "hp", "mp", "xp", "rip", "op", "group", "troop", "loop", "soup",
    "soap", "slap", "snap", "trap", "clap", "flap", "wrap", "swap", "gasp", "grasp",
    "clasp", "crisp", "dump", "bump", "hump", "lump", "pump", "thump", "stump", "trump",
    "wimp", "simp", "derp", "chump", "burp", "beep", "boop", "creep", "peep", "steep",
    "sweep", "weep", "sheep", "cheap", "heap", "leap", "reap", "chop", "crop", "flop",
    "plop", "prop", "scoop", "stoop", "swoop", "whoop", "sharp", "tarp", "warp", "chirp",
    "slurp", "pulp", "gulp", "scalp", "ketchup", "backup", "pickup", "wakeup", "setup",
    "cleanup", "breakup", "checkup", "holdup", "lineup", "lookup", "makeup", "mixup",
    "startup",
}

_NON_QUESTION_P_LABELS = {
    "class", "group", "grade", "type", "rank", "tier", "level", "vitamin", "option",
    "part", "page", "step", "point", "phase", "plan", "room", "wing", "ward", "block",
    "sector", "deck", "floor", "track", "gate", "mr", "ms", "dr",
}

SPECIFIC_COMIC_REPLACEMENTS = [
    (r'\bMSST\b', 'MUST'),
    (r'\bMVST\b', 'MUST'),
    (r'\bWANTUS\b', 'WANT US'),
    (r'\bFORUSTO\b', 'FOR US TO'),
    (r'\bINORDER\b', 'IN ORDER'),
    (r'\bANDIT\'?S\b', "AND IT'S"),
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
    (r'\bWHYP\b', 'WHY?'),
    (r'\bWHATP\b', 'WHAT?'),
    (r'\bHOWP\b', 'HOW?'),
    (r'\bWHOP\b', 'WHO?'),
]


def clean_comic_dialogue_text(text: str) -> str:
    """Clean common comic OCR punctuation/spacing typos without corrupting real words or names."""
    if not text:
        return ""

    def _fix_trailing_q(match):
        lead = match.group(1)
        mid = match.group(2)
        tail = match.group(3)
        has_space_before_tail = bool(mid) and mid[-1].isspace()
        if not mid:
            if tail == "/":
                return f"{lead}?"
            return match.group(0)
        if has_space_before_tail:
            tokens = mid.strip().split()
            if tail in ("P", "p") and tokens:
                prev = re.sub(r'[^A-Za-z]', '', tokens[-1]).lower()
                if prev in _NON_QUESTION_P_LABELS:
                    return match.group(0)
            return f"{lead}{mid.rstrip()}?"
        if tail == "/":
            if mid[-1].isalpha():
                return f"{lead}{mid}?"
            return match.group(0)
        tokens = mid.strip().split()
        if not tokens:
            return match.group(0)
        stem = re.sub(r'^[^A-Za-z]+', '', tokens[-1])
        if not stem or not stem.isalpha():
            return match.group(0)
        cand = (stem + tail).lower()
        if cand in _COMMON_P_WORDS:
            return match.group(0)
        if wordninja is not None:
            try:
                wc = wordninja.DEFAULT_LANGUAGE_MODEL._wordcost
                if cand in wc:
                    return match.group(0)
                if stem.lower() in wc and wc[stem.lower()] <= 11.5:
                    return f"{lead}{mid}?"
            except Exception:
                pass
        return match.group(0)

    text = re.sub(
        r'\b(WHY|WHAT|WHO|WHERE|WHEN|HOW|COULDN\'?T|CAN|CAN\'?T|DID|IS|ARE|DO|DOES|WILL|WOULD|SHOULD)\b([^?!.\n]*?)([Pp/])(?=[)\]"\'”’]?\s*(?:[\n\r]|$))',
        _fix_trailing_q,
        text,
        flags=re.IGNORECASE,
    )
    for pattern, repl in SPECIFIC_COMIC_REPLACEMENTS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text.strip()


def repair_english_word_spacing(text: str) -> str:
    """Split confidently recognized English words that OCR joined together."""
    if not text:
        return ""
    text = clean_comic_dialogue_text(text)
    if wordninja is None:
        return text

    try:
        word_costs = wordninja.DEFAULT_LANGUAGE_MODEL._wordcost

        def split_token(match):
            token = match.group(0)
            lower = token.lower()
            if lower in KEEP_WORDS or lower in _COMMON_P_WORDS:
                return token
            if lower in word_costs and word_costs[lower] <= 12.5:
                return token
            parts = wordninja.split(lower)
            if len(parts) < 2:
                return token
            if any(len(p) == 1 and p not in {'a', 'i'} for p in parts):
                return token
            if any(len(parts[idx]) == 1 and len(parts[idx + 1]) == 1 for idx in range(len(parts) - 1)):
                return token
            if any(p not in word_costs or word_costs[p] > 11.2 for p in parts if len(p) > 1):
                return token
            token_cost = word_costs.get(lower, 999.0)
            parts_cost = sum(word_costs[p] for p in parts)
            if token_cost > 10.0 or parts_cost < token_cost - 1.0:
                if token.isupper():
                    return " ".join(p.upper() for p in parts)
                elif token[0].isupper() and token[1:].islower():
                    return parts[0].capitalize() + " " + " ".join(parts[1:])
                return " ".join(parts)
            return token

        text = re.sub(r"[A-Za-z]{4,}", split_token, text)
    except Exception as exc:
        logger.debug("wordninja validation exception: %s", exc)
    return text


def _join_dialogue_lines(lines: list[str], lang: str = "en") -> str:
    """Join dialogue lines, properly merging hyphenated words across line breaks."""
    result = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not result:
            result = line
        elif result.endswith("-") and len(result) >= 2 and result[-2].isalpha() and line[0].isalpha():
            result = result[:-1] + line
        else:
            result = result + " " + line
    if lang in {"en", "english", "eng"}:
        return repair_english_word_spacing(result)
    return result

def merge_horizontal_ocr_lines(items, lang: str = "en"):
    """Join word boxes into readable lines, then nearby lines into distinct dialogue blocks.
    Separates adjacent speech bubbles and preserves strict column boundaries."""
    if not items:
        return []

    median_h = sorted(max(1, it["bottom"] - it["top"]) for it in items)[len(items)//2]
    # In manga speech bubbles, words on the exact same line have small spacing (typically 3-7px).
    max_word_gap = max(6, int(median_h * 0.55))

    sorted_items = sorted(items, key=lambda it: (it["top"], it["left"]))
    rows = []
    for it in sorted_items:
        h = max(1, it["bottom"] - it["top"])
        cy = (it["top"] + it["bottom"]) / 2.0

        matched_row = None
        min_gap = 9999
        for row in rows:
            rcy = (row["top"] + row["bottom"]) / 2.0
            rh = row["bottom"] - row["top"]
            # Strict vertical alignment for words on the same line
            if abs(cy - rcy) <= max(3.0, min(h, rh) * 0.35):
                gap_x = it["left"] - row["right"]
                if 0 <= gap_x <= max_word_gap and gap_x < min_gap:
                    min_gap = gap_x
                    matched_row = row

        if matched_row is not None:
            matched_row["items"].append(it)
            matched_row["right"] = max(matched_row["right"], it["right"])
            matched_row["top"] = min(matched_row["top"], it["top"])
            matched_row["bottom"] = max(matched_row["bottom"], it["bottom"])
            matched_row["text"] = matched_row["text"] + " " + it["text"]
            matched_row["confidence"] = sum(item["confidence"] for item in matched_row["items"]) / len(matched_row["items"])
        else:
            rows.append({
                "left": it["left"], "top": it["top"],
                "right": it["right"], "bottom": it["bottom"],
                "text": it["text"],
                "confidence": it["confidence"],
                "items": [it]
            })

    adj = {i: [] for i in range(len(rows))}
    for i in range(len(rows)):
        rA = rows[i]
        wA = max(1, rA["right"] - rA["left"])
        hA = max(1, rA["bottom"] - rA["top"])
        midA = (rA["left"] + rA["right"]) / 2.0
        for j in range(i + 1, len(rows)):
            rB = rows[j]
            wB = max(1, rB["right"] - rB["left"])
            hB = max(1, rB["bottom"] - rB["top"])
            midB = (rB["left"] + rB["right"]) / 2.0

            gap_y = max(0, rB["top"] - rA["bottom"]) if rB["top"] >= rA["bottom"] else max(0, rA["top"] - rB["bottom"])
            max_line_gap = max(10, int(max(hA, hB) * 1.35))
            if gap_y > max_line_gap:
                continue

            x_overlap = min(rA["right"], rB["right"]) - max(rA["left"], rB["left"])
            min_w = min(wA, wB)

            is_short = min_w <= max(20, int(median_h * 1.8))
            centers_aligned = abs(midA - midB) <= max(18.0, min_w * 0.60)

            # Must overlap significantly OR be a centered short word
            if (min_w > 0 and (x_overlap / min_w) >= 0.35 and centers_aligned) or (is_short and centers_aligned and x_overlap >= -4):
                adj[i].append(j)
                adj[j].append(i)

    visited = set()
    blocks = []
    for i in range(len(rows)):
        if i in visited:
            continue
        comp = []
        q = [i]
        visited.add(i)
        while q:
            curr = q.pop(0)
            comp.append(curr)
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append(neighbor)
        comp.sort(key=lambda idx: rows[idx]["top"])
        b_left = min(rows[idx]["left"] for idx in comp)
        b_top = min(rows[idx]["top"] for idx in comp)
        b_right = max(rows[idx]["right"] for idx in comp)
        b_bottom = max(rows[idx]["bottom"] for idx in comp)
        b_text = _join_dialogue_lines([rows[idx]["text"] for idx in comp], lang=lang)
        b_conf = sum(rows[idx]["confidence"] for idx in comp) / len(comp)
        blocks.append({
            "left": b_left, "top": b_top, "right": b_right, "bottom": b_bottom,
            "text": b_text, "confidence": b_conf
        })
    return sorted(blocks, key=lambda block: (block["top"], block["left"]))

class _GoogleResultParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if self.depth and tag not in {"br", "hr", "img", "input", "meta", "link", "wbr"}:
            self.depth += 1
        elif set((dict(attrs).get("class") or "").split()) & {"result-container", "t0"}:
            self.depth = 1

    def handle_endtag(self, tag):
        if tag in {"br", "hr", "img", "input", "meta", "link", "wbr"}:
            return
        if self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if self.depth:
            self.parts.append(data)


def translate_text(
    text: str,
    target_lang: str,
    source_lang: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
    document_mode: str = "comic",
) -> Optional[str]:
    """Use two bounded Google routes and stop retries when cancellation is requested."""
    global _google_blocked_until
    clean_text = text.strip()
    if not clean_text:
        return ""
    if document_mode != "book" and source_lang in {"en", "english", "eng"}:
        clean_text = repair_english_word_spacing(clean_text)

    src_param = translation_language_code(source_lang)
    target_param = translation_language_code(target_lang, default="uz")

    # High-accuracy DeepL translation when available (for Russian, English, Japanese, etc.)
    try:
        from deepl_translator import translate_single_with_deepl, is_deepl_supported
        if is_deepl_supported(target_param):
            deepl_res = translate_single_with_deepl(
                clean_text,
                target_param,
                src_param,
                document_mode=document_mode,
                cancel_event=cancel_event,
            )
            if deepl_res:
                return deepl_res
    except InterruptedError:
        raise
    except Exception as exc:
        logger.debug("DeepL translation exception: %s", exc)

    # If Google was recently rate-limited (429 HTTPError on cloud datacenter IPs),
    # skip directly to the MyMemory fallback to avoid stalling requests.
    with _google_block_lock:
        is_google_blocked = time.monotonic() < _google_blocked_until
    if not is_google_blocked:
        # Both Google routes have finite socket timeouts, a shared retry budget,
        # and bounded responses. Route 0 uses high-limit dict-chrome-ex; Route 1 uses m?.
        query = urllib.parse.urlencode({"sl": src_param, "tl": target_param, "q": clean_text})
        urls = [
            f"https://clients5.google.com/translate_a/t?client=dict-chrome-ex&{query}",
            f"https://translate.google.com/m?{query}",
        ]
        deadline = time.monotonic() + TRANSLATION_TIMEOUT_SECONDS * len(urls)
        for index, url in enumerate(urls):
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Tarjima töxtatildi.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                        "Accept": "*/*",
                    }
                )
                with urllib.request.urlopen(req, timeout=min(TRANSLATION_TIMEOUT_SECONDS, remaining)) as resp:
                    chunks, received = [], 0
                    read_chunk = getattr(resp, "read1", resp.read)
                    while received <= 1024 * 1024:
                        if cancel_event is not None and cancel_event.is_set():
                            raise InterruptedError("Tarjima töxtatildi.")
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Translation deadline exceeded")
                        chunk = read_chunk(min(65536, 1024 * 1024 + 1 - received))
                        if not chunk:
                            break
                        chunks.append(chunk)
                        received += len(chunk)
                    raw = b"".join(chunks)
                if len(raw) > 1024 * 1024:
                    raise ValueError("Translation response exceeds limit")
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Tarjima töxtatildi.")
                if index == 0:
                    data = json.loads(raw.decode("utf-8"))
                    if isinstance(data, list) and data:
                        if isinstance(data[0], list) and data[0]:
                            if isinstance(data[0][0], str):
                                translated = str(data[0][0]).strip()
                            else:
                                translated = "".join(
                                    (segment[0] if isinstance(segment, (list, tuple)) else str(segment))
                                    for segment in data[0]
                                    if segment and (segment[0] if isinstance(segment, (list, tuple)) else segment)
                                ).strip()
                        elif isinstance(data[0], str):
                            translated = str(data[0]).strip()
                        else:
                            translated = ""
                    else:
                        translated = ""
                else:
                    parser = _GoogleResultParser()
                    parser.feed(raw.decode("utf-8"))
                    translated = "".join(parser.parts).strip()
                if translated:
                    return translated
            except InterruptedError:
                raise
            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str or getattr(exc, "code", None) == 429:
                    with _google_block_lock:
                        _google_blocked_until = time.monotonic() + 300.0  # Cool down for 5 minutes
                logger.debug("Google tarjima yo'li %d ishlamadi: %s", index + 1, type(exc).__name__)

    if os.environ.get("MANGA_USE_MYMEMORY_FALLBACK", "false").lower() in ("1", "true", "yes") and len(clean_text) <= 480:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Tarjima töxtatildi.")
        try:
            def _detect_script_language(sample: str) -> str:
                if any('\u3040' <= c <= '\u30ff' for c in sample):
                    return "ja"
                if any('\uac00' <= c <= '\ud7af' or '\u1100' <= c <= '\u11ff' for c in sample):
                    return "ko"
                if any('\u4e00' <= c <= '\u9fff' for c in sample):
                    return "zh"
                if any('\u0400' <= c <= '\u04ff' for c in sample):
                    return "ru"
                return "en"

            mm_src = src_param if src_param != "auto" else _detect_script_language(clean_text)
            if mm_src != target_param:
                mm_query = urllib.parse.urlencode({"q": clean_text, "langpair": f"{mm_src}|{target_param}"})
                mm_url = f"https://api.mymemory.translated.net/get?{mm_query}"
                mm_req = urllib.request.Request(
                    mm_url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
                )
                with urllib.request.urlopen(mm_req, timeout=TRANSLATION_TIMEOUT_SECONDS) as resp:
                    raw_bytes = resp.read(1024 * 1024)
                    mm_data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
                    status = mm_data.get("responseStatus")
                    res_text = mm_data.get("responseData", {}).get("translatedText")
                    res_upper = (res_text or "").strip().upper()
                    is_mm_error = (
                        not res_upper
                        or res_upper.startswith("MYMEMORY WARNING")
                        or res_upper.startswith("INVALID")
                        or "QUERY LENGTH LIMIT EXCEEDED" in res_upper
                        or "PLEASE SELECT TWO DISTINCT LANGUAGES" in res_upper
                        or "NO QUERY SPECIFIED" in res_upper
                        or "TOO MANY REQUESTS" in res_upper
                        or "IS AN INVALID" in res_upper
                        or "INVALID SOURCE LANGUAGE" in res_upper
                        or "INVALID TARGET LANGUAGE" in res_upper
                        or "INVALID EMAIL" in res_upper
                    )
                    if status == 200 and not is_mm_error:
                        return html.unescape(res_text).strip()
        except InterruptedError:
            raise
        except Exception as exc:
            logger.warning("MyMemory fallback xatosi: %s", exc)

    return None


BATCH_DELIMITER = "\n|||\n"


def translate_batch(
    texts: list[str],
    target_lang: str,
    source_lang: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
    document_mode: str = "comic",
    translate_fn: Optional[Any] = None,
) -> list[Optional[str]]:
    """
    Translates a list of texts in batch (using DeepL batch JSON API when configured,
    otherwise bounded '\n|||\n' GET chunks with individual fallback).
    Returns None in failed slots (never substitutes untranslated original text).
    """
    if not texts:
        return []

    non_empty_indices = [i for i, t in enumerate(texts) if t and t.strip()]
    if not non_empty_indices:
        return ["" for _ in texts]

    results: list[Optional[str]] = ["" if not (t and t.strip()) else None for t in texts]

    # Prefer native DeepL batch POST when no custom translate_fn is injected
    if translate_fn is None:
        try:
            from deepl_translator import (
                translate_dialogues_with_deepl,
                is_deepl_supported,
                get_deepl_api_key,
            )
            target_param = translation_language_code(target_lang, default="uz")
            if is_deepl_supported(target_param) and get_deepl_api_key():
                batch_inputs = [texts[i].strip() for i in non_empty_indices]
                if document_mode != "book" and normalize_ocr_language(source_lang) == "en":
                    batch_inputs = [repair_english_word_spacing(t) for t in batch_inputs]
                deepl_res = translate_dialogues_with_deepl(
                    batch_inputs,
                    target_param,
                    translation_language_code(source_lang),
                    document_mode=document_mode,
                    cancel_event=cancel_event,
                )
                if deepl_res and len(deepl_res) == len(non_empty_indices):
                    for idx, val in zip(non_empty_indices, deepl_res):
                        results[idx] = val.strip() if isinstance(val, str) and val.strip() else None
                    return results
        except InterruptedError:
            raise
        except Exception as exc:
            logger.debug("Batch DeepL translation failed: %s", exc)

    tx_fn = translate_fn or translate_text

    def _call_tx(item_text: str) -> Optional[str]:
        try:
            import inspect
            params = inspect.signature(tx_fn).parameters
            kwargs = {}
            if "cancel_event" in params:
                kwargs["cancel_event"] = cancel_event
            if "document_mode" in params:
                kwargs["document_mode"] = document_mode
            res = tx_fn(item_text, target_lang, source_lang, **kwargs)
        except TypeError:
            res = tx_fn(item_text, target_lang, source_lang)
        return res.strip() if isinstance(res, str) and res.strip() else None

    if len(non_empty_indices) == 1:
        idx = non_empty_indices[0]
        results[idx] = _call_tx(texts[idx])
        return results

    # Group into bounded GET chunks; items containing literal delimiter or >900 chars run individually
    chunks: list[list[int]] = []
    current_chunk: list[int] = []
    current_len = 0
    for idx in non_empty_indices:
        item = texts[idx].strip()
        if "|||" in item or len(item) > 900:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk, current_len = [], 0
            chunks.append([idx])
            continue
        extra = len(item) + (len(BATCH_DELIMITER) if current_chunk else 0)
        if current_chunk and (current_len + extra > 1200 or len(current_chunk) >= 12):
            chunks.append(current_chunk)
            current_chunk, current_len = [], 0
        current_chunk.append(idx)
        current_len += len(item) + (len(BATCH_DELIMITER) if len(current_chunk) > 1 else 0)
    if current_chunk:
        chunks.append(current_chunk)

    for chunk in chunks:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Tarjima töxtatildi.")
        if len(chunk) == 1:
            idx = chunk[0]
            results[idx] = _call_tx(texts[idx])
            continue
        joined_text = BATCH_DELIMITER.join(texts[i].strip() for i in chunk)
        translated_joined = None
        try:
            translated_joined = _call_tx(joined_text)
        except InterruptedError:
            raise
        except Exception as exc:
            logger.debug("Batch translation chunk failed: %s", exc)

        if translated_joined:
            parts = re.split(r'\s*\|\|\|\s*', translated_joined)
            if len(parts) == len(chunk) and all(p.strip() for p in parts):
                for idx, part in zip(chunk, parts):
                    results[idx] = part.strip()
                continue
            logger.debug(
                "Batch translation split count mismatch: got %d parts for %d items, falling back to individual",
                len(parts),
                len(chunk),
            )

        for idx in chunk:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Tarjima töxtatildi.")
            results[idx] = _call_tx(texts[idx])

    return results
