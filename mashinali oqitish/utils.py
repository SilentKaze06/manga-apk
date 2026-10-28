import os
import json
import tempfile
import logging
from PIL import ImageFont

logger = logging.getLogger("manga_utils")

MAX_OCR_RENDER_LEN = 24

from functools import lru_cache

@lru_cache(maxsize=32)
def get_best_font(size=20, bold=False):
    """
    Cross-platform font discovery for PIL ImageFont with LRU caching.
    Prioritizes CJK-capable fonts on Windows for multi-lingual manga text,
    falling back to standard system fonts and ImageFont.load_default().
    """
    candidate_paths = []
    
    # 1. Windows fonts (prioritize CJK support: Japanese, Chinese, Korean)
    win_dir = os.environ.get("WINDIR", r"C:\Windows")
    fonts_win = os.path.join(win_dir, "Fonts")
    if bold:
        candidate_paths.extend([
            os.path.join(fonts_win, "msgothic.ttc"),
            os.path.join(fonts_win, "msyh.ttc"),
            os.path.join(fonts_win, "malgunbd.ttf"),
            os.path.join(fonts_win, "arialbd.ttf"),
            os.path.join(fonts_win, "segoeuib.ttf"),
        ])
    else:
        candidate_paths.extend([
            os.path.join(fonts_win, "msgothic.ttc"),
            os.path.join(fonts_win, "msyh.ttc"),
            os.path.join(fonts_win, "malgun.ttf"),
            os.path.join(fonts_win, "simsun.ttc"),
            os.path.join(fonts_win, "arial.ttf"),
            os.path.join(fonts_win, "segoeui.ttf"),
        ])
        
    # 2. Linux / macOS fonts
    candidate_paths.extend([
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf"
    ])
    
    for fpath in candidate_paths:
        if os.path.exists(fpath):
            try:
                return ImageFont.truetype(fpath, size)
            except Exception as e:
                logger.debug("Failed loading font %s: %s", fpath, e)
                continue
                
    return ImageFont.load_default()

def atomik_json_yoz(filepath, data, indent=2, ensure_ascii=False):
    """
    Atomically writes JSON data to a file via a temporary file in the same directory,
    preventing file corruption on process termination or concurrent read.
    """
    dir_name = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(dir_name, exist_ok=True)
    
    # Use same directory so os.replace is atomic across the filesystem
    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
        temp_name = tf.name
        json.dump(data, tf, ensure_ascii=ensure_ascii, indent=indent)
        tf.flush()
        os.fsync(tf.fileno())
        
    try:
        os.replace(temp_name, filepath)
    except Exception as e:
        if os.path.exists(temp_name):
            try:
                os.remove(temp_name)
            except Exception:
                pass
        raise e


def decode_ctc(log_probs_tb, idx_to_char):
    """
    Standard Greedy CTC decode for a single sample: log_probs_tb is (T, C).
    Takes raw argmax across classes, collapses consecutive duplicates,
    and removes blank token (0) and <unk> token (1).
    """
    preds_idx = log_probs_tb.argmax(dim=-1)
    toks = []
    prev = None
    for t in preds_idx.cpu().numpy():
        t = int(t)
        if t != prev and t > 1:
            toks.append(t)
        prev = t
    return "".join(idx_to_char.get(t, "") for t in toks)
