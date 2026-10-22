"""CPU ONNX instance segmentation with automatic class & mask decoding.
Supports specialized YOLO11-Litter-Seg (aquatic waste) as well as YOLO11n-seg (COCO 80)."""
import os
import threading
import cv2
import numpy as np
import onnxruntime as ort
ort.disable_telemetry_events()
from .config import DATA, ai_config, number
from .yolox import CLASSES as COCO_CLASSES

LITTER_CLASSES = ['Plastic', 'Metal', 'Glass', 'Paper', 'Other']

_model = None
_current_path = None
_lock = threading.Lock()


def model_name():
    configured = os.getenv('DETECTION_MODEL', '').strip().lower()
    if configured in ('gemini', 'gemini-vision', 'gemini_vision'):
        return 'Gemini-Vision'
    if 'litter' in configured or configured in ('yolo11_litter', 'yolo11-litter-seg'):
        return 'YOLO11-Litter-Seg'
    if configured == 'yolox-nano':
        return 'YOLOX-Nano'
    if configured == 'yolo11n-seg':
        return 'YOLO11n-seg'
    if (DATA / 'models' / 'yolo11_litter_seg.onnx').exists():
        return 'YOLO11-Litter-Seg'
    return 'YOLO11n-seg'


def is_vlm():
    return model_name() == 'Gemini-Vision'


def model_path():
    name = model_name()
    if name == 'Gemini-Vision':
        return None
    if name == 'YOLO11-Litter-Seg':
        return DATA / 'models' / 'yolo11_litter_seg.onnx'
    if name == 'YOLOX-Nano':
        return DATA / 'models' / 'yolox_nano.onnx'
    return DATA / 'models' / 'yolo11n-seg.onnx'


def is_ready():
    if is_vlm():
        cfg = ai_config()
        key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY') or (cfg['key'] if cfg['provider'] == 'gemini' else '')
        return bool(key)
    path = model_path()
    return path is not None and path.is_file()


def session():
    global _model, _current_path
    if is_vlm():
        return None
    with _lock:
        target_path = model_path()
        if _model is None or _current_path != target_path:
            if not target_path or not target_path.exists():
                raise RuntimeError(f'Missing ONNX model at {target_path}. Run setup script or scripts/export_model.py.')
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = int(number('CPU_THREADS', 2, 1, 16))
            _model = ort.InferenceSession(str(target_path), sess_options=opts, providers=['CPUExecutionProvider'])
            _current_path = target_path
    return _model


def detect(frame):
    if is_vlm():
        from . import ai
        raw_boxes = ai.detect_debris_gemini(frame)
        threshold = number('DETECTION_CONFIDENCE', 0.15, 0.05, 0.99)
        results = []
        for item in raw_boxes:
            conf = float(item.get('confidence', 0.85))
            if conf < threshold:
                continue
            box_2d = item.get('box_2d') or item.get('box')
            if not box_2d or len(box_2d) != 4:
                continue
            ymin, xmin, ymax, xmax = [float(v) for v in box_2d]
            if max(ymin, xmin, ymax, xmax) > 1.5:
                by1, bx1, by2, bx2 = ymin / 1000.0, xmin / 1000.0, ymax / 1000.0, xmax / 1000.0
            else:
                by1, bx1, by2, bx2 = ymin, xmin, ymax, xmax

            bx1, by1 = max(0.0, min(1.0, bx1)), max(0.0, min(1.0, by1))
            bx2, by2 = max(0.0, min(1.0, bx2)), max(0.0, min(1.0, by2))
            if bx2 <= bx1 or by2 <= by1:
                continue

            bw = bx2 - bx1
            bh = by2 - by1
            area = bw * bh
            aspect = bh / max(0.001, bw)

            # 1. Reject giant background / gate span boxes
            if (bw > 0.40 and bh > 0.35) or area > 0.12:
                continue
            if by1 < 0.02 and by2 > 0.70:
                continue

            # 2. Reject vertical pillar shadows
            if by1 > 0.65 and aspect > 2.0:
                continue
            if area < 0.0002:
                continue

            label = str(item.get('label', 'Trash Cluster')).strip()
            polygon = [
                [round(bx1, 5), round(by1, 5)],
                [round(bx2, 5), round(by1, 5)],
                [round(bx2, 5), round(by2, 5)],
                [round(bx1, 5), round(by2, 5)],
            ]
            results.append({
                'label': label,
                'confidence': round(conf, 3),
                'box': [round(bx1, 5), round(by1, 5), round(bx2, 5), round(by2, 5)],
                'polygons': [polygon],
                'debris_candidate': True,
            })
        return results

    if model_name() == 'YOLOX-Nano':
        from .yolox import detect as yolox_detect
        return yolox_detect(frame)
    model = session()
    height, width = frame.shape[:2]
    size = int(model.get_inputs()[0].shape[2])
    ratio = min(size / height, size / width)
    rw, rh = round(width * ratio), round(height * ratio)
    left, top = (size - rw) // 2, (size - rh) // 2
    image = np.full((size, size, 3), 114, np.uint8)
    image[top:top + rh, left:left + rw] = cv2.resize(frame, (rw, rh))
    tensor = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
    pred, proto = model.run(None, {model.get_inputs()[0].name: tensor})
    pred, proto = pred[0].T, proto[0]

    num_masks = proto.shape[0]  # usually 32
    num_classes = pred.shape[1] - 4 - num_masks

    if num_classes == 5:
        class_names = LITTER_CLASSES
        default_conf = .15
    elif num_classes == 80:
        class_names = COCO_CLASSES
        default_conf = .25
    else:
        class_names = [f'class_{i}' for i in range(num_classes)]
        default_conf = .20

    scores = pred[:, 4:4 + num_classes]
    classes, conf = scores.argmax(1), scores.max(1)
    threshold = number('DETECTION_CONFIDENCE', default_conf, .05, .99)
    candidates = np.flatnonzero(conf >= threshold)
    results = []
    for cls in np.unique(classes[candidates]):
        cls_idx = int(cls)
        cls_label = class_names[cls_idx] if cls_idx < len(class_names) else f'class_{cls_idx}'
        indices = candidates[classes[candidates] == cls]
        boxes, probs = [], []
        for i in indices:
            cx, cy, bw, bh = pred[i, :4]
            boxes.append([float(cx - bw / 2), float(cy - bh / 2), float(bw), float(bh)])
            probs.append(float(conf[i]))
        keep = cv2.dnn.NMSBoxes(boxes, probs, threshold, .45, top_k=100)
        for k in np.asarray(keep).reshape(-1):
            idx = indices[k]
            x, y, bw, bh = boxes[k]
            box = [(x - left) / rw, (y - top) / rh, (x + bw - left) / rw, (y + bh - top) / rh]
            bx1, by1, bx2, by2 = [round(float(np.clip(v, 0, 1)), 5) for v in box]
            bw_norm = bx2 - bx1
            bh_norm = by2 - by1
            area_norm = bw_norm * bh_norm
            aspect = bh_norm / max(0.001, bw_norm)

            # 1. Reject giant background / water surface / gate structure false positives (#15)
            if (bw_norm > 0.35 and bh_norm > 0.30) or area_norm > 0.08:
                continue
            if by1 < 0.02 and by2 > 0.70:
                continue

            # 2. Reject vertical column shadows and water reflection streaks (#3)
            if by1 > 0.65 and aspect > 1.8:
                continue
            if by1 > 0.75 and probs[k] < 0.30:
                continue
            if by2 > 0.85 and aspect > 1.6 and probs[k] < 0.35:
                continue
            if area_norm < 0.0003:
                continue

            logits = pred[idx, 4 + num_classes:] @ proto.reshape(num_masks, -1)
            mask = (logits.reshape(proto.shape[1:]) > 0).astype(np.uint8)
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
            crop = np.zeros_like(mask)
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(size, int(x + bw)), min(size, int(y + bh))
            crop[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            crop = crop[top:top + rh, left:left + rw]
            contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons = []
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
                if cv2.contourArea(contour) < 3:
                    continue
                contour = cv2.approxPolyDP(contour, .003 * cv2.arcLength(contour, True), True)
                polygons.append([[round(float(p[0][0] / rw), 5), round(float(p[0][1] / rh), 5)] for p in contour])
            is_debris = True if num_classes == 5 else (cls_label in ('bottle', 'cup', 'bowl', 'handbag', 'backpack', 'umbrella'))
            results.append({
                'label': cls_label,
                'confidence': round(probs[k], 3),
                'box': [bx1, by1, bx2, by2],
                'polygons': polygons,
                'debris_candidate': is_debris,
            })
    return results

