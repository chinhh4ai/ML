"""Official YOLOX-Nano ONNX, CPU only, with YOLOX's preprocessing and grid decoding."""
import threading

import cv2
import numpy as np
import onnxruntime as ort
ort.disable_telemetry_events()

from .config import DATA, number

CLASSES = 'person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush'.split(',')
_session = None
_lock = threading.Lock()


def session():
    global _session
    with _lock:
        if _session is None:
            path = DATA / 'models/yolox_nano.onnx'
            if not path.exists():
                raise RuntimeError('Missing YOLOX-Nano model. Use bundled model or download official release; see THIRD_PARTY_NOTICES.md.')
            options = ort.SessionOptions()
            options.intra_op_num_threads = int(number('CPU_THREADS', 2, 1, 16))
            _session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    return _session


def detect(frame):
    model = session()
    h, w = frame.shape[:2]
    size = int(model.get_inputs()[0].shape[2])
    ratio = min(size / h, size / w)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    resized = cv2.resize(frame, (int(w * ratio), int(h * ratio)))
    canvas[:resized.shape[0], :resized.shape[1]] = resized
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32)[None]
    predictions = model.run(None, {model.get_inputs()[0].name: tensor})[0][0].copy()
    grids, strides = [], []
    for stride in (8, 16, 32):
        yy, xx = np.meshgrid(np.arange(size // stride), np.arange(size // stride), indexing='ij')
        grids.append(np.stack([xx, yy], -1).reshape(-1, 2))
        strides.append(np.full((len(grids[-1]), 1), stride))
    grid, stride = np.concatenate(grids), np.concatenate(strides)
    predictions[:, :2] = (predictions[:, :2] + grid) * stride
    predictions[:, 2:4] = np.exp(predictions[:, 2:4]) * stride
    scores = predictions[:, 4:5] * predictions[:, 5:]
    labels, confidences = scores.argmax(1), scores.max(1)
    threshold = number('DETECTION_CONFIDENCE', .35, .05, .99)
    result = []
    for cls in np.unique(labels):
        indices = np.where((labels == cls) & (confidences >= threshold))[0]
        boxes, probs = [], []
        for i in indices:
            cx, cy, bw, bh = predictions[i, :4] / ratio
            boxes.append([float(cx - bw / 2), float(cy - bh / 2), float(bw), float(bh)])
            probs.append(float(confidences[i]))
        keep = cv2.dnn.NMSBoxes(boxes, probs, threshold, .45)
        for k in np.asarray(keep).reshape(-1):
            x, y, bw, bh = boxes[k]
            result.append({'label': CLASSES[int(cls)], 'confidence': round(probs[k], 3),
                           'box': [round(float(np.clip(v, 0, 1)), 5) for v in [x / w, y / h, (x + bw) / w, (y + bh) / h]]})
    return result
