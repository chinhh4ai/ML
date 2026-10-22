import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
checks = json.loads((root / 'model-checksums.json').read_text())
for name, expected in checks.items():
    path = root / '.data/models' / name
    if path.exists():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            provenance = path.with_suffix(path.suffix + '.provenance.json')
            local = json.loads(provenance.read_text()) if provenance.exists() else {}
            verified_export = (
                name == 'yolo11n-seg.onnx'
                and local.get('weights_sha256') == '55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152'
                and local.get('artifact_sha256') == actual
            )
            if not verified_export:
                raise SystemExit(f'Model checksum mismatch: {name}. Restore the trusted bundled model or re-export using scripts/export_model.py with verified official weights.')
        print(f'Verified {name}')
