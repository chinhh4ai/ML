import json

import httpx
import pytest

from backend import ai
from backend.main import rank_segments
from backend.prompts import DOMAIN


@pytest.mark.parametrize('provider', ['gemini', 'openai', 'litellm'])
def test_provider_wire_format(provider, monkeypatch, tmp_path):
    monkeypatch.setenv('AI_PROVIDER', provider)
    monkeypatch.setenv('AI_API_KEY', 'unit-test-key')
    monkeypatch.setenv('AI_MODEL', 'vision-test')
    monkeypatch.setenv('AI_BASE_URL', 'https://provider.test/v1')
    frame = tmp_path / 'frame.jpg'
    frame.write_bytes(b'fake-image-for-wire-format-test')
    captured = []
    def handler(request):
        captured.append(request)
        body = json.loads(request.content)
        if provider == 'gemini':
            assert request.headers['x-goog-api-key'] == 'unit-test-key'
            assert body['contents'][0]['parts'][1]['inline_data']['mime_type'] == 'image/jpeg'
            return httpx.Response(200, json={'candidates': [{'content': {'parts': [{'text': 'Kết quả thử nghiệm'}]}}]})
        assert request.headers['authorization'] == 'Bearer unit-test-key'
        assert body['messages'][0]['content'][1]['type'] == 'image_url'
        return httpx.Response(200, json={'choices': [{'message': {'content': 'Kết quả thử nghiệm'}}]})
    original = httpx.Client
    monkeypatch.setattr(ai.httpx, 'Client', lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    text, _ = ai.generate(DOMAIN + '\nPhân tích rác nổi.', [frame])
    assert text == 'Kết quả thử nghiệm'
    assert len(captured) == 1


def test_provider_error_does_not_expose_response_or_secret(monkeypatch):
    monkeypatch.setenv('AI_PROVIDER', 'openai')
    monkeypatch.setenv('AI_API_KEY', 'private-test-secret')
    original = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(401, text='sensitive provider diagnostics'))
    monkeypatch.setattr(ai.httpx, 'Client', lambda **kw: original(transport=transport, **kw))
    with pytest.raises(ai.AIUnavailable) as error:
        ai.generate('test')
    assert '401' in str(error.value)
    assert 'private-test-secret' not in str(error.value)
    assert 'sensitive' not in str(error.value)


def test_debris_search_and_prompt():
    row = {'segments': [{'id': 0, 'start': 0, 'labels': ['bottle'], 'description': '', 'tags': []}]}
    assert rank_segments(row, 'rác nhựa')[0]['id'] == 0
    assert rank_segments(row, 'chai nhựa')[0]['id'] == 0
    assert rank_segments(row, 'elephant') == []
    assert 'bèo' in DOMAIN and 'phản xạ' in DOMAIN and 'Không coi mọi vật thể' in DOMAIN
