"""Provider adapters. No SDK telemetry, key exposure, or implicit provider fallback."""
import base64
import json
import re
import time
import threading

import httpx

from .config import ai_config


class AIUnavailable(Exception):
    pass


SLOTS = threading.BoundedSemaphore(2)


def generate(prompt, images=()):
    with SLOTS:
        return _generate(prompt, images)


def _generate(prompt, images=()):
    cfg = ai_config()
    if not cfg['key']:
        raise AIUnavailable('Chưa cấu hình API key. Thêm khóa vào .env rồi khởi động lại backend.')
    provider = cfg['provider']
    if provider not in ('gemini', 'openai', 'litellm'):
        raise AIUnavailable('AI_PROVIDER phải là gemini, openai hoặc litellm.')
    encoded = [base64.b64encode(path.read_bytes()).decode() for path in images]
    if provider == 'gemini':
        base = cfg['base_url'] or 'https://generativelanguage.googleapis.com/v1beta'
        url = f"{base}/models/{cfg['model']}:generateContent"
        headers = {'x-goog-api-key': cfg['key']}
        body = {'contents': [{'role': 'user', 'parts': [{'text': prompt}] + [
            {'inline_data': {'mime_type': 'image/jpeg', 'data': data}} for data in encoded
        ]}], 'generationConfig': {'temperature': .2, 'maxOutputTokens': 2048}}
    else:
        if provider == 'litellm' and not cfg['base_url']:
            raise AIUnavailable('Cần AI_BASE_URL cho LiteLLM.')
        base = cfg['base_url'] or 'https://api.openai.com/v1'
        url = f'{base}/chat/completions'
        headers = {'Authorization': f"Bearer {cfg['key']}"}
        content = [{'type': 'text', 'text': prompt}] + [
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + data}} for data in encoded
        ]
        body = {'model': cfg['model'], 'messages': [{'role': 'user', 'content': content}]}
    try:
        with httpx.Client(timeout=90) as client:
            for attempt in range(3):
                response = client.post(url, headers=headers, json=body)
                if response.status_code not in (429, 502, 503, 504) or attempt == 2:
                    break
                time.sleep(2 ** attempt)
            if response.is_error:
                # Do not include provider response bodies or request URLs in user-facing errors.
                raise AIUnavailable(f'{provider} trả HTTP {response.status_code}. Kiểm tra model, quyền truy cập và quota.')
            result = response.json()
        if provider == 'gemini':
            text = ''.join(p.get('text', '') for c in result.get('candidates', []) for p in c.get('content', {}).get('parts', []) if not p.get('thought'))
            usage = result.get('usageMetadata', {})
        else:
            text = result['choices'][0]['message']['content']
            usage = result.get('usage', {})
        if not isinstance(text, str) or not text.strip():
            raise AIUnavailable('Model không trả nội dung. Kiểm tra khả năng hỗ trợ ảnh của model.')
        return text.strip(), usage
    except AIUnavailable:
        raise
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
        raise AIUnavailable('Không đọc được phản hồi AI. Kiểm tra kết nối và cấu hình model.') from None


def parse_caption(text):
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict) and isinstance(value.get('description'), str):
            return value['description'][:5000], [str(t)[:100] for t in value.get('tags', [])[:15]] if isinstance(value.get('tags', []), list) else []
    except (ValueError, TypeError):
        pass
    return text[:5000], []
