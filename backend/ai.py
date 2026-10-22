"""Provider adapters. No SDK telemetry, key exposure, or implicit provider fallback."""
import base64
import json
import os
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
        raise AIUnavailable('API key is not configured. Add your key to .env and restart backend.')
    provider = cfg['provider']
    if provider not in ('gemini', 'openai', 'litellm'):
        raise AIUnavailable('AI_PROVIDER must be gemini, openai, or litellm.')
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
            raise AIUnavailable('AI_BASE_URL is required for LiteLLM.')
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
                if response.status_code == 429:
                    raise AIUnavailable(f'{provider} returned HTTP 429: Quota or rate limit exceeded. Check your plan/quota or switch models.')
                elif response.status_code == 503:
                    raise AIUnavailable(f'{provider} returned HTTP 503: Service experiencing high demand. Please try again.')
                elif response.status_code == 404:
                    raise AIUnavailable(f'{provider} returned HTTP 404: Model not found or deprecated for this endpoint.')
                raise AIUnavailable(f'{provider} returned HTTP {response.status_code}. Check model name, access rights, and quota.')
            result = response.json()
        if provider == 'gemini':
            text = ''.join(p.get('text', '') for c in result.get('candidates', []) for p in c.get('content', {}).get('parts', []) if not p.get('thought'))
            usage = result.get('usageMetadata', {})
        else:
            text = result['choices'][0]['message']['content']
            usage = result.get('usage', {})
        if not isinstance(text, str) or not text.strip():
            raise AIUnavailable('Model returned empty response. Check if model supports vision/image inputs.')
        return text.strip(), usage
    except AIUnavailable:
        raise
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
        raise AIUnavailable('Could not read AI response. Check network connectivity and model configuration.') from None


def parse_caption(text):
    if not text:
        return '', []
    cleaned = text.strip()
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                desc = str(value.get('description', '')).strip()
                tags = [str(t).strip()[:100] for t in value.get('tags', []) if str(t).strip()][:15] if isinstance(value.get('tags'), list) else []
                if desc:
                    return desc[:5000], tags
        except (ValueError, TypeError):
            pass
    fallback = re.sub(r'^```(?:json)?\s*|\s*```$', '', cleaned).strip()
    return fallback[:5000], []


def parse_detection_boxes(text):
    if not text:
        return []
    cleaned = text.strip()
    match = re.search(r'\[[\s\S]*\]', cleaned)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
        if isinstance(data, list):
            valid = []
            for item in data:
                if isinstance(item, dict) and ('box_2d' in item or 'box' in item):
                    valid.append(item)
            return valid
    except (ValueError, TypeError):
        pass
    return []


def detect_debris_gemini(frame):
    cfg = ai_config()
    key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY') or (cfg['key'] if cfg['provider'] == 'gemini' else '')
    if not key:
        raise AIUnavailable('Gemini API key is required for Gemini Vision Detection. Please set GEMINI_API_KEY.')

    configured_model = os.getenv('GEMINI_DETECTION_MODEL') or (cfg['model'] if cfg['provider'] == 'gemini' else 'gemini-3.5-flash-lite')
    if configured_model == 'gemini-3.6-flash':
        models_to_try = ['gemini-3.5-flash-lite', 'gemini-flash-lite-latest', 'gemini-3.6-flash']
    else:
        models_to_try = [configured_model]
        if 'gemini-3.5-flash-lite' not in models_to_try:
            models_to_try.append('gemini-3.5-flash-lite')

    import cv2
    h, w = frame.shape[:2]
    max_dim = 1024
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame_to_send = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        frame_to_send = frame

    ok, buf = cv2.imencode('.jpg', frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return []
    b64_data = base64.b64encode(buf).decode('utf-8')

    prompt = (
        "Locate all floating aquatic debris, garbage, plastic containers/bottles, bags, styrofoam, "
        "cans, organic debris clusters, or water waste in this river intake surveillance frame.\n"
        "Return a JSON array of objects with the exact schema:\n"
        "[\n"
        "  {\n"
        '    "box_2d": [ymin, xmin, ymax, xmax],\n'
        '    "label": "Plastic | Metal | Glass | Paper | Organic Debris | Trash Cluster",\n'
        '    "confidence": 0.95\n'
        "  }\n"
        "]\n"
        "Coordinates ymin, xmin, ymax, xmax MUST be integers between 0 and 1000 representing normalized box positions.\n"
        "Do NOT detect static river banks, concrete structures, water gate metal pillars, dark water shadows, or reflections. Only real floating debris. If no floating debris is present, return [].\n"
        "Return ONLY valid JSON."
    )

    base = (os.getenv('GEMINI_BASE_URL') or (cfg['base_url'] if cfg['provider'] == 'gemini' else '') or 'https://generativelanguage.googleapis.com/v1beta').rstrip('/')

    with SLOTS:
        for model_name in models_to_try:
            url = f"{base}/models/{model_name}:generateContent"
            headers = {'x-goog-api-key': key}
            body = {
                'contents': [{'role': 'user', 'parts': [
                    {'text': prompt},
                    {'inline_data': {'mime_type': 'image/jpeg', 'data': b64_data}}
                ]}],
                'generationConfig': {
                    'temperature': 0.1,
                    'maxOutputTokens': 2048,
                    'responseMimeType': 'application/json'
                }
            }
            try:
                with httpx.Client(timeout=45) as client:
                    for attempt in range(2):
                        resp = client.post(url, headers=headers, json=body)
                        if resp.status_code != 429 or attempt == 1:
                            break
                        time.sleep(1.5 * (attempt + 1))
                    if resp.status_code == 429:
                        continue
                    if resp.is_error:
                        continue
                    res_json = resp.json()
                    raw_text = ''.join(
                        p.get('text', '') for c in res_json.get('candidates', [])
                        for p in c.get('content', {}).get('parts', []) if not p.get('thought')
                    )
                    boxes = parse_detection_boxes(raw_text)
                    return boxes
            except Exception:
                continue

    return []

