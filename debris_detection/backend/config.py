import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env', override=False)
DATA = Path(os.getenv('VSS_DATA_DIR', str(ROOT / '.data'))).resolve()
for folder in ['videos', 'frames', 'models', 'tmp']:
    (DATA / folder).mkdir(parents=True, exist_ok=True)


def ai_config():
    provider = os.getenv('AI_PROVIDER', 'gemini').lower()
    key = os.getenv('AI_API_KEY') or os.getenv({
        'gemini': 'GEMINI_API_KEY', 'openai': 'OPENAI_API_KEY', 'litellm': 'LITELLM_API_KEY'
    }.get(provider, ''), '')
    if provider == 'gemini':
        key = key or os.getenv('GOOGLE_API_KEY', '')
    return {
        'provider': provider,
        'model': os.getenv('AI_MODEL') or os.getenv(provider.upper() + '_MODEL') or ('gemini-2.5-flash' if provider == 'gemini' else 'gpt-4.1-mini'),
        'key': key,
        'base_url': (os.getenv('AI_BASE_URL') or os.getenv(provider.upper() + '_BASE_URL', '')).rstrip('/'),
    }


def update_env_file(updates: dict[str, str]):
    env_path = ROOT / '.env'
    lines = []
    if env_path.exists():
        lines = env_path.read_text(encoding='utf-8').splitlines()
    updated_keys = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#') and '=' in line:
            key, _ = line.split('=', 1)
            key = key.strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                updated_keys.add(key)
                continue
        new_lines.append(line)
    for key, val in updates.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={val}")
    env_path.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
    for key, val in updates.items():
        os.environ[key] = val


def get_full_config():
    cur = ai_config()
    gemini_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY', '')
    litellm_key = os.getenv('LITELLM_API_KEY', '')
    openai_key = os.getenv('OPENAI_API_KEY', '')
    det_model = os.getenv('DETECTION_MODEL', 'yolo11_litter_seg')
    return {
        'provider': cur['provider'],
        'model': cur['model'],
        'base_url': cur['base_url'],
        'has_key': bool(cur['key']),
        'detection_model': det_model,
        'detection_confidence': number('DETECTION_CONFIDENCE', 0.15, 0.05, 0.99),
        'available_detection_models': [
            {'id': 'yolo11_litter_seg', 'name': 'YOLO11-Litter-Seg (Aquatic Trash/Litter: Plastic, Metal, Glass, Paper)', 'classes': 5, 'type': 'cpu'},
            {'id': 'yolo11n-seg', 'name': 'YOLO11n-seg (General COCO 80)', 'classes': 80, 'type': 'cpu'},
            {'id': 'gemini', 'name': 'Gemini Vision Detector (Multimodal VLM · Zero-shot 2D Bounding Boxes)', 'classes': 'Open Vocabulary', 'type': 'vlm'},
        ],
        'providers': {
            'gemini': {
                'model': os.getenv('GEMINI_MODEL', 'gemini-3.5-flash-lite'),
                'has_key': bool(gemini_key),
                'key_masked': (gemini_key[:4] + '...' + gemini_key[-4:]) if len(gemini_key) > 8 else ('••••' if gemini_key else ''),
                'base_url': os.getenv('GEMINI_BASE_URL', ''),
                'preset_models': ['gemini-3.5-flash-lite', 'gemini-flash-lite-latest', 'gemini-3.6-flash', 'gemini-3.8-flash', 'gemini-2.5-flash', 'gemini-3.1-pro-preview'],
            },
            'litellm': {
                'model': os.getenv('LITELLM_MODEL', 'Llama-3.1-Nemotron-Nano-VL-8B-V1-FP4-QAD-A30.0.121'),
                'has_key': bool(litellm_key),
                'key_masked': (litellm_key[:4] + '...' + litellm_key[-4:]) if len(litellm_key) > 8 else ('••••' if litellm_key else ''),
                'base_url': os.getenv('LITELLM_BASE_URL', 'http://172.18.60.121:4000'),
                'preset_models': ['Llama-3.1-Nemotron-Nano-VL-8B-V1-FP4-QAD-A30.0.121'],
            },
            'openai': {
                'model': os.getenv('OPENAI_MODEL', 'gpt-4.1-mini'),
                'has_key': bool(openai_key),
                'key_masked': (openai_key[:4] + '...' + openai_key[-4:]) if len(openai_key) > 8 else ('••••' if openai_key else ''),
                'base_url': os.getenv('OPENAI_BASE_URL', ''),
                'preset_models': ['gpt-4.1-mini', 'gpt-4o', 'gpt-4o-mini'],
            }
        }
    }


def save_ai_config(provider: str, model: str, key: str | None = None, base_url: str | None = None,
                   detection_model: str | None = None, detection_confidence: float | None = None):
    provider = provider.lower().strip()
    if provider not in ('gemini', 'litellm', 'openai'):
        raise ValueError('AI_PROVIDER must be gemini, litellm, or openai')
    updates = {
        'AI_PROVIDER': provider,
        f"{provider.upper()}_MODEL": model.strip(),
        'AI_MODEL': '',
        'AI_BASE_URL': '',
        'AI_API_KEY': '',
    }
    if key is not None and key.strip():
        updates[f"{provider.upper()}_API_KEY"] = key.strip()
    if base_url is not None:
        updates[f"{provider.upper()}_BASE_URL"] = base_url.strip()
    if detection_model:
        updates['DETECTION_MODEL'] = detection_model.strip()
    if detection_confidence is not None:
        updates['DETECTION_CONFIDENCE'] = str(round(float(detection_confidence), 2))
    update_env_file(updates)


def number(name, default, low, high):
    return min(high, max(low, float(os.getenv(name, default))))
