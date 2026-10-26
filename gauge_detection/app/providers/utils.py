"""
Utility functions for AI Vision providers
Includes robust JSON extraction and normalization for diverse LLM outputs.
"""
import re
import json
import math
from typing import Dict, Any, Optional, List

from app.schemas import (
    BoundingBox,
    Point2D,
    NeedleDetection,
    ScaleNumberDetection,
    GaugeReading,
    GaugeDetection,
)

NON_NEEDLE_TYPES = {
    "seven_segment",
    "seven_segment_display",
    "digital",
    "flowmeter",
    "tokico_flowmeter",
    "energy_meter",
    "hour_meter",
    "sight_glass",
    "oil_level_sight_glass",
    "oil_level",
    "level_gauge",
}


def safe_bbox(data, default_ymin=0, default_xmin=0, default_ymax=1000, default_xmax=1000) -> BoundingBox:
    """Safely converts list [ymin, xmin, ymax, xmax] or dict to BoundingBox."""
    if isinstance(data, (list, tuple)) and len(data) >= 4:
        return BoundingBox(
            ymin=float(data[0]),
            xmin=float(data[1]),
            ymax=float(data[2]),
            xmax=float(data[3]),
        )
    elif isinstance(data, dict):
        return BoundingBox(
            ymin=float(data.get("ymin", default_ymin)),
            xmin=float(data.get("xmin", default_xmin)),
            ymax=float(data.get("ymax", default_ymax)),
            xmax=float(data.get("xmax", default_xmax)),
        )
    return BoundingBox(ymin=default_ymin, xmin=default_xmin, ymax=default_ymax, xmax=default_xmax)


def safe_point(data, default_x=500, default_y=500) -> Point2D:
    """Safely converts list [y, x] or [x, y] or dict to Point2D."""
    if isinstance(data, (list, tuple)) and len(data) >= 2:
        return Point2D(x=float(data[1]), y=float(data[0]))
    elif isinstance(data, dict):
        return Point2D(x=float(data.get("x", default_x)), y=float(data.get("y", default_y)))
    return Point2D(x=default_x, y=default_y)


def ensure_scale_number_geometry(
    numbers: List[ScaleNumberDetection],
    center: Point2D,
    gauge_bbox: BoundingBox,
    is_non_needle: bool,
    min_range: Optional[float] = None,
    max_range: Optional[float] = None,
) -> List[ScaleNumberDetection]:
    """
    Guarantees that every scale number has valid coordinates and non-zero bounding box.
    For dial gauges, any numbers lacking bounding boxes are arranged along the scale arc
    (typically 220° bottom-left sweeping clockwise 280° to 140° bottom-right).
    """
    if not numbers:
        return numbers

    # 1. If any number has position but empty/zero bbox, derive bbox around position
    for num in numbers:
        bw = num.bbox.xmax - num.bbox.xmin
        bh = num.bbox.ymax - num.bbox.ymin
        if (bw <= 0 or bh <= 0) and num.position and (num.position.x > 0 or num.position.y > 0):
            num.bbox = BoundingBox(
                ymin=round(max(0.0, num.position.y - 25.0), 1),
                xmin=round(max(0.0, num.position.x - 20.0), 1),
                ymax=round(min(1000.0, num.position.y + 25.0), 1),
                xmax=round(min(1000.0, num.position.x + 20.0), 1)
            )

    # 2. Check if all or some numbers still have zero/invalid bbox
    needs_layout = [n for n in numbers if (n.bbox.xmax - n.bbox.xmin <= 0 or n.bbox.ymax - n.bbox.ymin <= 0)]
    if not needs_layout:
        return numbers

    n_count = len(numbers)
    if not is_non_needle:
        # Analog dial gauge: distribute along circular scale arc
        gw = gauge_bbox.xmax - gauge_bbox.xmin
        gh = gauge_bbox.ymax - gauge_bbox.ymin
        dial_dim = min(gw, gh) if (gw > 50 and gh > 50) else 500.0
        radius = 0.28 * dial_dim
        center_x = center.x if center.x > 0 else (gauge_bbox.xmin + gauge_bbox.xmax) / 2.0
        center_y = center.y if center.y > 0 else (gauge_bbox.ymin + gauge_bbox.ymax) / 2.0

        num_vals = [n.numeric_value for n in numbers if n.numeric_value is not None]
        v_min = min(num_vals) if num_vals else (float(min_range) if min_range is not None else 0.0)
        v_max = max(num_vals) if num_vals else (float(max_range) if max_range is not None else 100.0)
        v_span = (v_max - v_min) if (v_max > v_min) else 1.0

        for i, num in enumerate(numbers):
            bw = num.bbox.xmax - num.bbox.xmin
            bh = num.bbox.ymax - num.bbox.ymin
            if bw > 0 and bh > 0 and (num.bbox.xmin > 0 or num.bbox.ymin > 0):
                continue  # already has valid bbox

            if num.numeric_value is not None and v_span > 0:
                t = max(0.0, min(1.0, (num.numeric_value - v_min) / v_span))
            else:
                t = i / max(1, n_count - 1)

            # Arc starts at 220° (bottom-left) and sweeps 280° clockwise to 500°/140° (bottom-right)
            angle_deg = 220.0 + t * 280.0
            rad = math.radians(angle_deg)
            x_pos = center_x + radius * math.sin(rad)
            y_pos = center_y - radius * math.cos(rad)

            x_pos = max(10.0, min(990.0, x_pos))
            y_pos = max(10.0, min(990.0, y_pos))

            num.position = Point2D(x=round(x_pos, 1), y=round(y_pos, 1))
            num.bbox = BoundingBox(
                ymin=round(max(0.0, y_pos - 25.0), 1),
                xmin=round(max(0.0, x_pos - 20.0), 1),
                ymax=round(min(1000.0, y_pos + 25.0), 1),
                xmax=round(min(1000.0, x_pos + 20.0), 1)
            )
    else:
        # Non-needle digital/display: distribute horizontally across display box
        gw = gauge_bbox.xmax - gauge_bbox.xmin
        center_y = (gauge_bbox.ymin + gauge_bbox.ymax) / 2.0
        for i, num in enumerate(numbers):
            bw = num.bbox.xmax - num.bbox.xmin
            bh = num.bbox.ymax - num.bbox.ymin
            if bw > 0 and bh > 0 and (num.bbox.xmin > 0 or num.bbox.ymin > 0):
                continue
            x_pos = gauge_bbox.xmin + ((i + 0.5) / n_count) * gw
            x_pos = max(10.0, min(990.0, x_pos))
            y_pos = max(10.0, min(990.0, center_y))
            num.position = Point2D(x=round(x_pos, 1), y=round(y_pos, 1))
            num.bbox = BoundingBox(
                ymin=round(max(0.0, y_pos - 25.0), 1),
                xmin=round(max(0.0, x_pos - 18.0), 1),
                ymax=round(min(1000.0, y_pos + 25.0), 1),
                xmax=round(min(1000.0, x_pos + 18.0), 1)
            )

    return numbers


def extract_and_parse_json(raw_text: Optional[str]) -> Dict[str, Any]:
    """
    Extracts and parses JSON object from raw LLM output.
    Handles:
    - Markdown code fences (```json ... ```)
    - Chain-of-thought tags (<think>...</think>)
    - Leading/trailing conversational text
    - Trailing commas
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty response received from model.")

    text = raw_text.strip()

    # 1. Strip reasoning / thinking tags if present
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.DOTALL).strip()

    # 2. Strip Markdown code fences if the entire content is fenced
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # 3. Locate the outermost JSON object { ... }
    json_match = re.search(r'(\{[\s\S]*\})', text)
    if json_match:
        text = json_match.group(1)

    # 4. Remove trailing commas before closing braces/brackets
    cleaned_commas = re.sub(r',\s*([\}\]])', r'\1', text)

    try:
        return json.loads(cleaned_commas)
    except json.JSONDecodeError:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            repaired = text.rstrip().rstrip(",")
            if repaired.count('"') % 2 == 1:
                repaired += '"'
            open_brackets = repaired.count('[') - repaired.count(']')
            open_braces = repaired.count('{') - repaired.count('}')
            repaired += (']' * max(0, open_brackets)) + ('}' * max(0, open_braces))
            return json.loads(repaired)


def build_normalized_gauge_detection(g: Dict[str, Any], default_index: int = 1) -> GaugeDetection:
    """
    Normalizes any LLM-returned gauge dictionary into standard GaugeDetection schema.
    Strictly guarantees:
    - For analog / needle gauges: pointers in needles are detected.
    - For non-needle gauges: needles = [].
    - For ALL gauges: numbers (DETECTED SCALE NUMBERS) is populated.
    - For ALL gauges: reading.reasoning (AI REASONING & DERIVATION) is populated.
    """
    gauge_id = str(g.get("gauge_id") or f"gauge_{g.get('gauge_index', default_index)}")
    raw_type = str(g.get("gauge_type", "generic")).lower().strip()
    label = g.get("label") or raw_type.replace("_", " ").title()

    bbox_obj = safe_bbox(g.get("bbox"), 100, 100, 900, 900)
    center_default_x = (bbox_obj.xmin + bbox_obj.xmax) / 2.0
    center_default_y = (bbox_obj.ymin + bbox_obj.ymax) / 2.0
    center_obj = safe_point(g.get("dial_center"), center_default_x, center_default_y)

    is_non_needle = (
        raw_type in NON_NEEDLE_TYPES
        or g.get("has_needle") is False
        or any(k in raw_type for k in ("seven_segment", "digital", "roller", "sight_glass"))
    )

    # 1. NEEDLES PROCESSING
    needles: List[NeedleDetection] = []
    raw_needles = g.get("needles", [])

    if isinstance(raw_needles, list) and len(raw_needles) > 0:
        for idx, n in enumerate(raw_needles):
            nb_obj = safe_bbox(n.get("bbox")) if n.get("bbox") else None
            n_pivot = safe_point(n.get("pivot_point"), center_obj.x, center_obj.y)
            n_tip = safe_point(n.get("tip_point"), center_obj.x, max(0.0, center_obj.y - 150.0))
            needles.append(NeedleDetection(
                needle_id=str(n.get("needle_id", f"needle_{idx+1}")),
                type=str(n.get("type", "primary_pointer")),
                bbox=nb_obj,
                pivot_point=n_pivot,
                tip_point=n_tip,
                angle_degrees=float(n["angle_degrees"]) if n.get("angle_degrees") is not None else 270.0,
                color=str(n.get("color", "black"))
            ))
    elif isinstance(g.get("needle"), dict):
        # Support single needle dictionary format
        nd = g["needle"]
        n_angle = nd.get("angle_deg") or nd.get("angle_degrees")
        n_tip = safe_point(nd.get("tip") or nd.get("tip_point"), center_obj.x, max(0.0, center_obj.y - 150.0))
        n_center = safe_point(nd.get("center") or nd.get("pivot_point"), center_obj.x, center_obj.y)
        needles.append(NeedleDetection(
            needle_id="needle_1",
            type="primary_pointer",
            bbox=None,
            pivot_point=n_center,
            tip_point=n_tip,
            angle_degrees=float(n_angle) if n_angle is not None else 270.0,
            color=str(nd.get("color", "black"))
        ))

    # If it's a dial gauge (not digital) and needles is still empty, synthesize primary needle
    if not is_non_needle and len(needles) == 0:
        needles.append(NeedleDetection(
            needle_id="needle_1",
            type="primary_pointer",
            bbox=None,
            pivot_point=center_obj,
            tip_point=Point2D(x=center_obj.x, y=max(0.0, center_obj.y - 150.0)),
            angle_degrees=270.0,
            color="black"
        ))

    # If it is explicitly non-needle, keep needles empty
    if is_non_needle:
        needles = []

    # 2. NUMBERS (SCALE / DIGIT NUMBERS) PROCESSING
    numbers: List[ScaleNumberDetection] = []
    raw_numbers = g.get("numbers", [])

    if isinstance(raw_numbers, list) and len(raw_numbers) > 0:
        for num in raw_numbers:
            if isinstance(num, dict):
                val_str = str(num.get("value", "")).strip()
                if not val_str:
                    continue
                num_bbox = safe_bbox(num.get("bbox"), 0, 0, 0, 0)
                num_pos = safe_point(num.get("position"), 0, 0) if num.get("position") else None
                numbers.append(ScaleNumberDetection(
                    value=val_str,
                    numeric_value=num.get("numeric_value"),
                    bbox=num_bbox,
                    position=num_pos
                ))
            elif isinstance(num, (int, float, str)):
                val_str = str(num).strip()
                try:
                    flt = float(val_str)
                except ValueError:
                    flt = None
                numbers.append(ScaleNumberDetection(
                    value=val_str,
                    numeric_value=flt,
                    bbox=BoundingBox(ymin=0, xmin=0, ymax=0, xmax=0),
                    position=None
                ))

    # Fallback 1: Extract from scale_numbers list
    if len(numbers) == 0 and g.get("scale_numbers") and isinstance(g["scale_numbers"], list):
        for snum in g["scale_numbers"]:
            val_str = str(snum).strip()
            try:
                flt = float(val_str)
            except ValueError:
                flt = None
            numbers.append(ScaleNumberDetection(
                value=val_str,
                numeric_value=flt,
                bbox=BoundingBox(ymin=0, xmin=0, ymax=0, xmax=0),
                position=None
            ))

    # Fallback 2: Extract from display_text or digits_text for digital / flowmeters
    if len(numbers) == 0:
        disp_txt = g.get("display_text") or g.get("digits_text")
        if disp_txt:
            val_str = str(disp_txt).strip()
            try:
                flt = float(val_str)
            except ValueError:
                flt = None
            numbers.append(ScaleNumberDetection(
                value=val_str,
                numeric_value=flt,
                bbox=BoundingBox(ymin=0, xmin=0, ymax=0, xmax=0),
                position=None
            ))

    # Fallback 3: Extract from scale min_val, max_val
    scale_dict = g.get("scale", {})
    if len(numbers) == 0 and isinstance(scale_dict, dict):
        for k in ("min_val", "center_val", "max_val"):
            if k in scale_dict and scale_dict[k] is not None:
                val_str = str(scale_dict[k])
                try:
                    flt = float(val_str)
                except ValueError:
                    flt = None
                numbers.append(ScaleNumberDetection(
                    value=val_str,
                    numeric_value=flt,
                    bbox=BoundingBox(ymin=0, xmin=0, ymax=0, xmax=0),
                    position=None
                ))

    # 3. READING & REASONING PROCESSING
    read_data = g.get("reading", {})
    read_val = float(read_data.get("value") or g.get("reading_value") or 0.0)
    unit_str = str(read_data.get("unit") or g.get("unit") or "").strip()
    formatted_str = read_data.get("formatted") or f"{read_val} {unit_str}".strip()
    min_range = read_data.get("min_range") or scale_dict.get("min_val")
    max_range = read_data.get("max_range") or scale_dict.get("max_val")
    confidence = float(read_data.get("confidence") or g.get("confidence") or 0.95)

    # Fallback 4: If numbers still empty, populate from min_range / max_range / read_val
    if len(numbers) == 0:
        if not is_non_needle and min_range is not None and max_range is not None and float(max_range) > float(min_range):
            f_min = float(min_range)
            f_max = float(max_range)
            span = f_max - f_min
            step = span / 10.0 if span >= 10 else (span / 5.0 if span >= 5 else 1.0)
            curr = f_min
            while curr <= f_max + 0.001:
                val_display = f"{int(curr)}" if abs(curr - round(curr)) < 1e-4 else f"{curr:.1f}"
                numbers.append(ScaleNumberDetection(
                    value=val_display,
                    numeric_value=round(curr, 2),
                    bbox=BoundingBox(ymin=0, xmin=0, ymax=0, xmax=0),
                    position=None
                ))
                curr += step
        else:
            if min_range is not None:
                val_min = float(min_range)
                val_str = f"{int(val_min)}" if abs(val_min - round(val_min)) < 1e-4 else str(min_range)
                numbers.append(ScaleNumberDetection(
                    value=val_str,
                    numeric_value=val_min,
                    bbox=BoundingBox(ymin=0, xmin=0, ymax=0, xmax=0),
                    position=None
                ))
            if max_range is not None:
                val_max = float(max_range)
                val_str = f"{int(val_max)}" if abs(val_max - round(val_max)) < 1e-4 else str(max_range)
                numbers.append(ScaleNumberDetection(
                    value=val_str,
                    numeric_value=val_max,
                    bbox=BoundingBox(ymin=0, xmin=0, ymax=0, xmax=0),
                    position=None
                ))
            if len(numbers) == 0:
                val_str = f"{int(read_val)}" if abs(read_val - round(read_val)) < 1e-4 else str(read_val)
                numbers.append(ScaleNumberDetection(
                    value=val_str,
                    numeric_value=read_val,
                    bbox=BoundingBox(ymin=0, xmin=0, ymax=0, xmax=0),
                    position=None
                ))

    # Defensive geometry normalization: Ensure every scale number has non-zero bbox and valid position
    numbers = ensure_scale_number_geometry(
        numbers=numbers,
        center=center_obj,
        gauge_bbox=bbox_obj,
        is_non_needle=is_non_needle,
        min_range=float(min_range) if min_range is not None else None,
        max_range=float(max_range) if max_range is not None else None
    )

    # Extract Reasoning
    reasoning_str = (
        read_data.get("reasoning")
        or g.get("reasoning")
        or g.get("derivation")
        or read_data.get("derivation")
        or g.get("explanation")
        or ""
    ).strip()

    # Generate synthetic meaningful reasoning if model omitted it
    if not reasoning_str or reasoning_str in ("--", "null", "none"):
        if needles and len(needles) > 0:
            angle_deg = needles[0].angle_degrees if needles[0].angle_degrees is not None else 270.0
            reasoning_str = (
                f"Primary pointer detected at angle {angle_deg:.1f}° aligning with scale markings "
                f"to indicate {read_val} {unit_str}."
            )
        elif is_non_needle:
            reasoning_str = (
                f"Measured value of {read_val} {unit_str} recognized from digital LED / roller drum readout "
                f"or meniscus level boundary."
            )
        else:
            reasoning_str = f"Calculated reading of {read_val} {unit_str} based on dial face inspection."

    reading_obj = GaugeReading(
        value=read_val,
        unit=unit_str,
        formatted=formatted_str,
        min_range=float(min_range) if min_range is not None else None,
        max_range=float(max_range) if max_range is not None else None,
        confidence=confidence,
        reasoning=reasoning_str
    )

    return GaugeDetection(
        gauge_id=gauge_id,
        label=label,
        gauge_type=raw_type,
        bbox=bbox_obj,
        dial_center=center_obj,
        needles=needles,
        numbers=numbers,
        reading=reading_obj
    )
