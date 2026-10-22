"""Source + built frontend + verified ONNX models. Never package keys, videos, databases or venvs."""
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / 'riverlens-HIP.zip'
FILES = ['model-checksums.json', '.env.example', '.gitignore', 'README.md', 'requirements.txt', 'requirements-dev.txt',
         'requirements.lock.txt', 'requirements-export.txt', 'THIRD_PARTY_NOTICES.md', 'VALIDATION.md']
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as archive:
    def add(path):
        archive.write(path, 'HIP/' + path.relative_to(ROOT).as_posix())
    for name in FILES:
        add(ROOT / name)
    # Include a fresh template as .env; never read/copy the user's actual .env.
    archive.writestr('HIP/.env', (ROOT / '.env.example').read_bytes())
    for folder in ['backend', 'tests', 'scripts', 'frontend']:
        for path in sorted((ROOT / folder).rglob('*')):
            if path.is_file() and not any(part in {'node_modules', '__pycache__', '.pytest_cache'} for part in path.parts) and not path.name.endswith('.tsbuildinfo'):
                add(path)
    for path in sorted((ROOT / '.data/models').glob('*.onnx')):
        add(path)
        provenance = path.with_suffix(path.suffix + '.provenance.json')
        if provenance.is_file():
            add(provenance)
print(OUT)
print('SHA256:', hashlib.sha256(OUT.read_bytes()).hexdigest())
print('Bytes:', OUT.stat().st_size)
