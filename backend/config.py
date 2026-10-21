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


def number(name, default, low, high):
    return min(high, max(low, float(os.getenv(name, default))))
