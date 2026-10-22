import time
import json
from pathlib import Path

import cv2

from backend import store
from backend import ai
from backend.tracking import Tracker


def test_video_upload_segmentation_tracking_and_retrieval(client, tmp_path):
    image = cv2.imread(str(Path(__file__).parent / 'assets/segmentation-reference.jpg'))
    path = tmp_path / 'reference.mp4'
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 12, (image.shape[1], image.shape[0]))
    for _ in range(36):
        out.write(image)
    out.release()
    with path.open('rb') as file:
        response = client.post('/api/videos', files={'file': ('Functional test - not a river scene.mp4', file, 'video/mp4')})
    assert response.status_code == 202, response.text
    vid = response.json()['id']
    for _ in range(120):
        row = client.get(f'/api/videos/{vid}').json()
        if row['status'] in ('ready', 'failed'):
            break
        time.sleep(.1)
    assert row['status'] == 'ready', row
    assert row['processed_frames'] == 6
    assert row['summary'] == ''  # Never pass off detector output as an AI summary.
    assert row['warning']
    assert row['detection_model'] == 'YOLO11n-seg'
    objects = [o for obs in row['observations'] for o in obs['objects']]
    assert any(o['label'] == 'person' for o in objects)
    assert any(len(p) >= 3 for o in objects for p in o.get('polygons', []))
    assert all(o['track_id'] >= 1 for o in objects)
    first_ids = {o['track_id'] for o in row['observations'][0]['objects']}
    last_ids = {o['track_id'] for o in row['observations'][-1]['objects']}
    assert first_ids & last_ids
    result = client.get(f'/api/videos/{vid}/search', params={'q': 'tìm người'}).json()
    assert result['results'][0]['start'] == 0
    assert result['method'] == 'keyword'
    assert client.get(f'/api/videos/{vid}/frames/0-0.jpg').headers['content-type'] == 'image/jpeg'
    assert client.get(f'/api/videos/{vid}/media', headers={'Range': 'bytes=0-99'}).status_code == 206
    assert client.post(f'/api/videos/{vid}/ask', json={'question': 'Có rác không?'}).status_code == 503
    store.init()  # Reopen the on-disk database; result remains persisted.
    assert store.get(vid)['track_count'] > 0


def test_reject_invalid_upload_and_path(client):
    response = client.post('/api/videos', files={'file': ('fake.mp4', b'not video', 'video/mp4')})
    assert response.status_code == 400
    assert client.get('/api/videos/not-a-valid-id').status_code == 404
    assert client.get('/api/health').json()['device'] == 'cpu'


def test_tracker_expiry_class_and_one_to_one_matching():
    tracker = Tracker(max_age=2)
    make = lambda name='bottle': {'label': name, 'box': [.1, .1, .3, .3]}
    first = tracker.update([make()], 0)[0]['track_id']
    assert tracker.update([make()], .5)[0]['track_id'] == first
    assert tracker.update([make('cup')], 1)[0]['track_id'] != first
    matches = tracker.update([make(), make()], 1.5)
    assert len({o['track_id'] for o in matches}) == 2
    assert tracker.update([make()], 10)[0]['track_id'] != first


def test_ai_pipeline_and_grounded_sources_with_mock_model(client, monkeypatch):
    row = store.all_videos()[0]
    vid = row['id']
    monkeypatch.setenv('AI_API_KEY', 'mock-only-key')
    prompts = []
    def fake_generate(prompt, images=()):
        prompts.append(prompt)
        if 'segment_ids' in prompt:
            return json.dumps({'answer': 'Không đủ bằng chứng xác nhận rác trên mặt nước.', 'segment_ids': [0, 999]}), {'test': True}
        if '"description"' in prompt:
            assert images and images[0].exists()
            return json.dumps({'description': 'Cảnh đường phố; không có bằng chứng về mặt nước.', 'tags': ['đường phố']}), {'test': True}
        return 'Không đủ bằng chứng để đánh giá rác sông hồ từ video này.', {'test': True}
    monkeypatch.setattr(ai, 'generate', fake_generate)
    assert client.post(f'/api/videos/{vid}/retry').status_code == 202
    for _ in range(120):
        row = client.get(f'/api/videos/{vid}').json()
        if row['status'] in ('ready', 'failed'):
            break
        time.sleep(.1)
    assert row['status'] == 'ready'
    assert row['segments'][0]['ai'] is True
    assert row['summary'].startswith('Không đủ bằng chứng')
    response = client.post(f'/api/videos/{vid}/ask', json={'question': 'Có rác nổi không?'})
    assert response.status_code == 200
    assert [s['id'] for s in response.json()['sources']] == [0]  # Reject invented source ID.
    assert all('quan trắc rác nổi' in p for p in prompts)
