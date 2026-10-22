"""
Image processing utilities for Gauge Detection
Includes OpenCV drawing overlay, base64 encoding, and thumbnail cropping
"""
import io
import cv2
import base64
import numpy as np
from PIL import Image
from typing import List
from app.schemas import GaugeDetection, DetectionResult


def crop_gauge_thumbnail(img_np: np.ndarray, bbox, pad: int = 10) -> str:
    """Crop gauge region and return as base64 JPEG"""
    h, w = img_np.shape[:2]
    ymin = max(0, int((bbox.ymin / 1000.0) * h) - pad)
    xmin = max(0, int((bbox.xmin / 1000.0) * w) - pad)
    ymax = min(h, int((bbox.ymax / 1000.0) * h) + pad)
    xmax = min(w, int((bbox.xmax / 1000.0) * w) + pad)

    crop = img_np[ymin:ymax, xmin:xmax]
    if crop.size == 0:
        return ""

    _, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buf).decode("utf-8")


def render_annotated_image(
    image_bytes: bytes,
    result: DetectionResult
) -> bytes:
    """
    Render professional industrial overlay onto image:
    - Glowing bounding boxes for gauges
    - Needle ray with pivot circle and tip arrow
    - Scale number markers
    - Digital readout HUD badge
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes

    h, w = img.shape[:2]

    # Color palette (BGR)
    COLOR_GAUGE = (255, 180, 0)     # Cyan-Blue
    COLOR_NEEDLE = (50, 50, 255)    # Red-Orange
    COLOR_SECONDARY = (0, 215, 255) # Gold/Bronze
    COLOR_NUMBER = (80, 240, 120)   # Green/Emerald
    COLOR_HUD_BG = (20, 24, 33)     # Dark SCADA slate

    for g_idx, gauge in enumerate(result.gauges):
        # 1. Gauge Bounding Box
        gy1 = int((gauge.bbox.ymin / 1000.0) * h)
        gx1 = int((gauge.bbox.xmin / 1000.0) * w)
        gy2 = int((gauge.bbox.ymax / 1000.0) * h)
        gx2 = int((gauge.bbox.xmax / 1000.0) * w)

        cv2.rectangle(img, (gx1, gy1), (gx2, gy2), COLOR_GAUGE, 2, cv2.LINE_AA)

        # Gauge Label Pill
        label_text = f"[{g_idx+1}] {gauge.label}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (gx1, gy1 - th - 8), (gx1 + tw + 10, gy1), COLOR_GAUGE, -1)
        cv2.putText(img, label_text, (gx1 + 5, gy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # 2. Dial Center
        cx = int((gauge.dial_center.x / 1000.0) * w)
        cy = int((gauge.dial_center.y / 1000.0) * h)
        cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 7, COLOR_GAUGE, 2, cv2.LINE_AA)

        # 3. Dial Numbers
        for num in gauge.numbers:
            nx1 = int((num.bbox.xmin / 1000.0) * w)
            ny1 = int((num.bbox.ymin / 1000.0) * h)
            nx2 = int((num.bbox.xmax / 1000.0) * w)
            ny2 = int((num.bbox.ymax / 1000.0) * h)
            cv2.rectangle(img, (nx1, ny1), (nx2, ny2), COLOR_NUMBER, 1, cv2.LINE_AA)
            cv2.putText(img, num.value, (nx1, max(ny1 - 2, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_NUMBER, 1, cv2.LINE_AA)

        # 4. Needle Rays
        for needle in gauge.needles:
            px = int((needle.pivot_point.x / 1000.0) * w)
            py = int((needle.pivot_point.y / 1000.0) * h)
            tx = int((needle.tip_point.x / 1000.0) * w)
            ty = int((needle.tip_point.y / 1000.0) * h)

            col = COLOR_SECONDARY if needle.type == "secondary_pointer" else COLOR_NEEDLE
            # Draw needle vector line
            cv2.line(img, (px, py), (tx, ty), col, 2, cv2.LINE_AA)
            # Needle tip dot
            cv2.circle(img, (tx, ty), 4, col, -1, cv2.LINE_AA)
            cv2.circle(img, (tx, ty), 6, (255, 255, 255), 1, cv2.LINE_AA)

        # 5. Reading HUD Tag
        read_text = f"Reading: {gauge.reading.formatted}"
        (rw, rh), _ = cv2.getTextSize(read_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        hud_x = gx1
        hud_y = gy2 + rh + 12
        if hud_y > h - 5:
            hud_y = gy1 - 10
        cv2.rectangle(img, (hud_x, hud_y - rh - 6), (hud_x + rw + 12, hud_y + 4), COLOR_HUD_BG, -1)
        cv2.rectangle(img, (hud_x, hud_y - rh - 6), (hud_x + rw + 12, hud_y + 4), (0, 255, 128), 1)
        cv2.putText(img, read_text, (hud_x + 6, hud_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2, cv2.LINE_AA)

    _, out_buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return out_buf.tobytes()
