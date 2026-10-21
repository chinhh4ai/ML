"""CPU ONNX instance segmentation. YOLOX is an optional detection-only backend."""
import os
import threading
import cv2
import numpy as np
import onnxruntime as ort
ort.disable_telemetry_events()
from .config import DATA, number
from .yolox import CLASSES
_model = None
_lock = threading.Lock()


def model_name():
    return 'YOLOX-Nano' if os.getenv('DETECTION_MODEL', 'yolo11n-seg') == 'yolox-nano' else 'YOLO11n-seg'


def model_path():
    return DATA / 'models' / ('yolox_nano.onnx' if model_name() == 'YOLOX-Nano' else 'yolo11n-seg.onnx')


def session():
    global _model
    with _lock:
        if _model is None:
            if not model_path().exists():
                raise RuntimeError('Thiếu model ONNX. Chạy script setup hoặc scripts/export_model.py.')
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = int(number('CPU_THREADS', 2, 1, 16))
            _model = ort.InferenceSession(str(model_path()), sess_options=opts, providers=['CPUExecutionProvider'])
    return _model


def detect(frame):
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
    tensor = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255
    pred, proto = model.run(None, {model.get_inputs()[0].name: tensor})
    pred, proto = pred[0].T, proto[0]
    scores = pred[:, 4:84]
    classes, conf = scores.argmax(1), scores.max(1)
    threshold = number('DETECTION_CONFIDENCE', .25, .05, .99)
    candidates = np.flatnonzero(conf >= threshold)
    results = []
    for cls in np.unique(classes[candidates]):
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
            logits = pred[idx, 84:] @ proto.reshape(proto.shape[0], -1)
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
            box = [(x - left) / rw, (y - top) / rh, (x + bw - left) / rw, (y + bh - top) / rh]
            results.append({'label': CLASSES[int(cls)], 'confidence': round(probs[k], 3),
                            'box': [round(float(np.clip(v, 0, 1)), 5) for v in box], 'polygons': polygons,
                            'debris_candidate': CLASSES[int(cls)] in ('bottle', 'cup', 'bowl', 'handbag', 'backpack', 'umbrella')})
    return results
