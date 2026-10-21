import os
import shutil
import tempfile
from pathlib import Path

# Must be set before importing application modules; never touch the review library.
ROOT = Path(__file__).resolve().parents[1]
TEMP = Path(tempfile.mkdtemp(prefix='riverlens-tests-'))
os.environ['VSS_DATA_DIR'] = str(TEMP)
os.environ['VIDEO_SAMPLE_FPS'] = '2'
os.environ['DETECTION_MODEL'] = 'yolo11n-seg'
for key in ('AI_API_KEY', 'GEMINI_API_KEY', 'GOOGLE_API_KEY', 'OPENAI_API_KEY', 'LITELLM_API_KEY'):
    os.environ.pop(key, None)
(TEMP / 'models').mkdir()
shutil.copyfile(ROOT / '.data/models/yolo11n-seg.onnx', TEMP / 'models/yolo11n-seg.onnx')

import pytest
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture(scope='session')
def client():
    # dotenv may contain real user keys; explicitly disable AI in offline tests.
    keys = ['AI_API_KEY', 'GEMINI_API_KEY', 'GOOGLE_API_KEY', 'OPENAI_API_KEY', 'LITELLM_API_KEY']
    saved = {k: os.environ.pop(k, None) for k in keys}
    with TestClient(app) as c:
        yield c
    for key, value in saved.items():
        if value is not None:
            os.environ[key] = value
    shutil.rmtree(TEMP)
