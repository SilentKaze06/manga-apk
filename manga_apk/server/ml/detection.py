"""
Neural detection post-processing: light class-agnostic NMS / containment deduplication,
bubble-to-text metadata matching, and reading-order sorting (RTL manga / LTR manhua).
"""
from typing import Any, Dict, List, Optional

try:
    from geometry import _has_continuous_dark_segment
except (ImportError, ValueError):
    from .geometry import _has_continuous_dark_segment


def _crop_reading_key(r, reading_direction: str = "rtl", col_map: dict = None, line_idx_map: dict = None, coord_scale: float = 2.0):
    """
    Sort key function for OCR items inside a crop, supporting RTL manga reading direction
    and multi-column/line grouping.
    """
    col = col_map.get(id(r), 0) if col_map else 0
    line_idx = line_idx_map.get(id(r), None) if line_idx_map else None
    div = coord_scale if coord_scale and coord_scale > 0 else 1.0

    if isinstance(r, dict) and "box" in r:
        box = r["box"]
        y_c = (box[1] + box[3]) / 2.0
        x1 = box[0]
        x_c = (box[0] + box[2]) / 2.0
    elif isinstance(r, (list, tuple)) and len(r) > 0 and isinstance(r[0], (list, tuple)):
        ys = [pt[1] / div if isinstance(pt, (list, tuple)) else pt for pt in r[0]]
        xs = [pt[0] / div if isinstance(pt, (list, tuple)) else pt for pt in r[0]]
        y_c = (min(ys) + max(ys)) / 2.0
        x1 = min(xs)
        x_c = (min(xs) + max(xs)) / 2.0
    elif isinstance(r, (list, tuple)) and len(r) >= 4 and not isinstance(r[0], (list, tuple)):
        y_c = (r[1] + r[3]) / 2.0
        x1 = r[0]
        x_c = (r[0] + r[2]) / 2.0
    else:
        y_c = 0.0
        x1 = 0.0
        x_c = 0.0

    if col_map is not None or line_idx_map is not None:
        y_val = line_idx if line_idx is not None else y_c
        return (col, y_val, x1)

    if reading_direction == "rtl":
        return (y_c, -x_c)
    return (y_c, x1)


def _box_iou_and_contain(box_a: List[int], box_b: List[int]) -> tuple[float, float]:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0, 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])))
    area_b = float(max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])))
    iou = inter / max(1.0, area_a + area_b - inter)
    contain = inter / min(area_a, area_b)
    return iou, contain


def deduplicate_text_detections(
    detections: List[Dict[str, Any]],
    iou_thresh: float = 0.50,
    contain_thresh: float = 0.75,
) -> List[Dict[str, Any]]:
    """
    Light class-agnostic NMS / containment deduplication across text_bubble and text_free
    so duplicate overlapping boxes are not emitted, without any pixel-distance splitting.
    """
    if not detections or len(detections) <= 1:
        return list(detections) if detections else []

    def _priority(det: Dict[str, Any]) -> tuple[float, float]:
        score = float(det.get("score", det.get("confidence", 0.5)))
        label = det.get("label", "")
        bonus = 0.05 if label == "text_bubble" else 0.0
        b = det.get("text_box") or det.get("box") or [0, 0, 0, 0]
        area = float(max(0, b[2] - b[0]) * max(0, b[3] - b[1]))
        return (score + bonus, area)

    ordered = sorted(detections, key=_priority, reverse=True)
    kept: List[Dict[str, Any]] = []
    for cand in ordered:
        cb = cand.get("text_box") or cand.get("box") or [0, 0, 0, 0]
        is_dup = False
        for prev in kept:
            pb = prev.get("text_box") or prev.get("box") or [0, 0, 0, 0]
            iou, contain = _box_iou_and_contain(cb, pb)
            if iou >= iou_thresh or contain >= contain_thresh:
                is_dup = True
                break
        if not is_dup:
            kept.append(cand)
    return kept


def match_bubbles_to_text_blocks(
    text_detections: List[Dict[str, Any]],
    bubble_detections: List[Dict[str, Any]],
    min_contain: float = 0.50,
) -> List[Dict[str, Any]]:
    """
    Matches RT-DETR-v2 class-0 `bubble` bounds to `text_bubble` detections as layout
    metadata (`bubble_box`). Empty bubbles without text are never emitted.
    """
    if not text_detections:
        return []
    bubbles = [b["box"] for b in (bubble_detections or []) if b.get("box")]
    if not bubbles:
        return [dict(d) for d in text_detections]

    # Count how many text blocks fall into each bubble so shared multi-text bubbles
    # keep tight text_box as bbox while still preserving bubble_box metadata.
    assigned_bubble_idx: List[Optional[int]] = []
    bubble_hit_counts: Dict[int, int] = {}

    for det in text_detections:
        if det.get("label") != "text_bubble":
            assigned_bubble_idx.append(None)
            continue
        tb = det["box"]
        tb_area = float(max(1, (tb[2] - tb[0]) * (tb[3] - tb[1])))
        best_idx = None
        best_area = float("inf")
        for b_idx, bb in enumerate(bubbles):
            ix1, iy1 = max(tb[0], bb[0]), max(tb[1], bb[1])
            ix2, iy2 = min(tb[2], bb[2]), min(tb[3], bb[3])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = float((ix2 - ix1) * (iy2 - iy1))
            if inter / tb_area >= min_contain:
                b_area = float(max(1, (bb[2] - bb[0]) * (bb[3] - bb[1])))
                if b_area < best_area:
                    best_area = b_area
                    best_idx = b_idx
        assigned_bubble_idx.append(best_idx)
        if best_idx is not None:
            bubble_hit_counts[best_idx] = bubble_hit_counts.get(best_idx, 0) + 1

    enriched: List[Dict[str, Any]] = []
    for det, b_idx in zip(text_detections, assigned_bubble_idx):
        out = dict(det)
        if b_idx is not None:
            out["bubble_box"] = list(bubbles[b_idx])
            if bubble_hit_counts.get(b_idx, 0) == 1:
                out["layout_box"] = list(bubbles[b_idx])
        enriched.append(out)
    return enriched


def sort_detections_reading_order(
    detections: List[Dict[str, Any]],
    reading_direction: str = "rtl",
) -> List[Dict[str, Any]]:
    """
    Sort detected text blocks in reading order:
    - 'rtl' (manga): top-to-bottom bands, right-to-left within a band.
    - 'ltr' (manhua/webtoon): top-to-bottom bands, left-to-right within a band.
    """
    if not detections or len(detections) <= 1:
        return list(detections) if detections else []

    direction = (reading_direction or "rtl").strip().lower()
    is_rtl = (direction == "rtl")

    def _coords(det: Dict[str, Any]) -> tuple[float, float, float, float]:
        b = det.get("text_box") or det.get("box") or det.get("bbox") or [0, 0, 0, 0]
        return float(b[0]), float(b[1]), float(b[2]), float(b[3])

    # Sort primarily by vertical top/center into row bands, then horizontally by direction
    heights = [max(1.0, _coords(d)[3] - _coords(d)[1]) for d in detections]
    med_h = sorted(heights)[len(heights) // 2]
    band_tol = max(24.0, med_h * 0.65)

    # First sort top-to-bottom by y_center
    by_y = sorted(detections, key=lambda d: 0.5 * (_coords(d)[1] + _coords(d)[3]))
    bands: List[List[Dict[str, Any]]] = []
    for det in by_y:
        _, y1, _, y2 = _coords(det)
        yc = 0.5 * (y1 + y2)
        placed = False
        for band in bands:
            band_y = sum(0.5 * (_coords(x)[1] + _coords(x)[3]) for x in band) / len(band)
            if abs(yc - band_y) <= band_tol:
                band.append(det)
                placed = True
                break
        if not placed:
            bands.append([det])

    ordered: List[Dict[str, Any]] = []
    for band in bands:
        band.sort(
            key=lambda d: -(0.5 * (_coords(d)[0] + _coords(d)[2])) if is_rtl else (0.5 * (_coords(d)[0] + _coords(d)[2]))
        )
        ordered.extend(band)
    return ordered
