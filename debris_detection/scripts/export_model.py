"""Export official, checksum-verified YOLO11n-seg weights to a CPU ONNX model."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / '.data/models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.data/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.data/matplotlib'))
WEIGHTS_SHA256 = '55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152'
URL = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt'
path = MODEL_DIR / 'yolo11n-seg.pt'
if not path.exists():
    request = urllib.request.Request(URL, headers={'User-Agent': 'Riverlens/0.1'})
    with urllib.request.urlopen(request, timeout=120) as response:
        path.write_bytes(response.read())
if hashlib.sha256(path.read_bytes()).hexdigest() != WEIGHTS_SHA256:
    raise SystemExit('Model checksum mismatch. Do not load these weights. Remove the file and obtain the official artifact.')
import torch
from ultralytics import YOLO

torch.set_num_threads(2)
YOLO(str(path)).export(format='onnx', imgsz=416, opset=17, simplify=False, device='cpu', dynamic=False)
artifact = MODEL_DIR / 'yolo11n-seg.onnx'
# Export metadata includes a build timestamp, so a local export has its own checksum.
# Keep the fixed published-artifact checksums unchanged.
(MODEL_DIR / 'yolo11n-seg.onnx.provenance.json').write_text(json.dumps({
    'weights_sha256': WEIGHTS_SHA256,
    'artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
    'source_url': URL,
    'export': {'imgsz': 416, 'opset': 17, 'device': 'cpu'},
}, indent=2), encoding='utf-8')
print('Exported:', MODEL_DIR / 'yolo11n-seg.onnx')
