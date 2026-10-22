import logging
import os
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import cv2

from . import ai, detection, store
from .config import DATA, ai_config, number
from .tracking import Tracker
from .prompts import DOMAIN

log = logging.getLogger(__name__)
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='video-cpu')


def submit(vid):
    EXECUTOR.submit(process, vid)


def process(vid):
    started = time.monotonic()
    try:
        row = store.get(vid)
        store.update(vid, status='processing', stage='Normalizing video', progress=3, error=None, warning=None)
        source = DATA / 'videos' / f'{vid}.source'
        target = DATA / 'videos' / f'{vid}.mp4'
        if not target.exists():
            temporary = DATA / 'videos' / f'{vid}.processing.mp4'
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-threads', '2', '-i', str(source),
                            '-map', '0:v:0', '-an', '-vf', 'scale=min(1280\\,iw):-2',
                            '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast', '-crf', '25',
                            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)],
                           check=True, capture_output=True, timeout=900)
            temporary.replace(target)
        store.update(vid, stage='Segmentation & tracking (CPU)', progress=8)
        cap = cv2.VideoCapture(str(target))
        frames_dir = DATA / 'frames' / vid
        frames_dir.mkdir(exist_ok=True)
        for old_frame in frames_dir.glob('*.jpg'):
            old_frame.unlink()
        store.update(vid, width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        duration = row['duration']
        is_vlm_det = detection.is_vlm()
        fps = number('VLM_SAMPLE_FPS', 0.25, .05, 1.0) if is_vlm_det else number('VIDEO_SAMPLE_FPS', 1, .1, 5)
        chunk = int(number('VIDEO_CHUNK_SECONDS', 30, 5, 120))
        observations, segments, counts = [], [], Counter()
        tracker = Tracker(max_age=max(10.0, 4.0 / fps) if is_vlm_det else 3)
        total = max(1, int(duration * fps))
        initial_stage = 'Gemini Vision detection & tracking' if is_vlm_det else 'Segmentation & tracking (CPU)'
        store.update(vid, stage=initial_stage, progress=8)

        for index in range(total):
            timestamp = index / fps
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            objects = tracker.update(detection.detect(frame), timestamp)
            counts.update(o['label'] for o in objects)
            observations.append({'time': round(timestamp, 3), 'objects': objects})
            # Keep at most four frames per segment
            slot = int((timestamp % chunk) / (chunk / 4))
            segment_id = int(timestamp // chunk)
            image = frames_dir / f'{segment_id}-{slot}.jpg'
            if not image.exists():
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (int(w * min(1, 768 / w)), int(h * min(1, 768 / w))))
                cv2.imwrite(str(image), small, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if is_vlm_det:
                time.sleep(0.2)  # Gentle pacing to respect API quota
                store.update(vid, progress=min(65, 8 + int(57 * (index + 1) / total)),
                             stage=f"Gemini Vision detecting frame {index + 1}/{total} ({round(timestamp, 1)}s)")
            elif index % max(1, int(fps * 3)) == 0:
                store.update(vid, progress=min(65, 8 + int(57 * (index + 1) / total)),
                             stage=f"Segmentation & tracking ({round(timestamp, 1)}s)")
        cap.release()
        if not observations:
            raise RuntimeError('Could not read frames from video.')
        for i, start in enumerate(range(0, max(1, int(duration + .999)), chunk)):
            observed = [o for o in observations if start <= o['time'] < start + chunk]
            labels = sorted({obj['label'] for o in observed for obj in o['objects']})
            segments.append({'id': i, 'start': start, 'end': round(min(duration, start + chunk), 2),
                             'labels': labels, 'description': '', 'tags': [], 'ai': False,
                             'thumbnail': f'/api/videos/{vid}/frames/{i}-0.jpg'})
        store.update(vid, observations=observations, segments=segments, object_counts=dict(counts),
                     progress=67, stage='Analyzing content')
        summary = ''
        warning = None
        usages = []
        cfg = ai_config()
        if cfg['key']:
            try:
                limit = int(number('AI_MAX_SEGMENTS', 60, 1, 200))
                for i, seg in enumerate(segments[:limit]):
                    images = sorted(frames_dir.glob(f"{seg['id']}-*.jpg"))
                    seg_obs = [o for o in observations if seg['start'] <= o['time'] < seg['end'] and o['objects']]
                    det_details = []
                    for o in seg_obs:
                        det_details.append(f"at {o['time']}s: " + ', '.join(f"{obj['label']} ({obj['confidence']})" for obj in o['objects']))
                    if det_details:
                        labels_hint = f"Detected items in this segment: {'; '.join(det_details[:6])}."
                    elif seg['labels']:
                        labels_hint = f"Detected labels: {', '.join(seg['labels'])}."
                    else:
                        labels_hint = "No floating debris detected in this segment."

                    prompt = (
                        f"{DOMAIN}\n\n"
                        f"Analyze this river monitoring video segment ({seg['start']}s–{seg['end']}s) from the provided frame(s).\n"
                        f"{labels_hint}\n"
                        "Inspect the water surface, intake gates, and riverbanks for floating debris (plastics, bottles, bags, styrofoam, cans, organic debris, trash clusters).\n"
                        "Respond with ONLY a valid JSON object in this format:\n"
                        '{"description": "Clear and detailed observation of floating debris, water condition, and scene structure.", "tags": ["debris", "plastic", "river gate"]}'
                    )
                    text, usage = ai.generate(prompt, images)
                    seg['description'], seg['tags'] = ai.parse_caption(text)
                    seg['ai'] = True
                    usages.append(usage)
                    store.update(vid, segments=segments, progress=68 + int(24 * (i + 1) / min(len(segments), limit)),
                                 stage=f'AI analyzing segment {i + 1}/{min(len(segments), limit)}')
                if len(segments) > limit:
                    warning = f'Cost limit: AI analyzed only {limit}/{len(segments)} segments.'
                context = '\n'.join(f"[{s['start']}–{s['end']}s] {s['description']}" for s in segments if s['ai'])
                summary, usage = ai.generate(
                    f"{DOMAIN}\n\n"
                    "Provide a comprehensive river monitoring and floating debris assessment in English summarizing the observations from the video segments below.\n"
                    "Highlight timestamps where debris or trash clusters were detected, describe the water condition at the gates, and note any potential intake risks.\n\n"
                    f"<segment_observations>\n{context}\n</segment_observations>"
                )
                usages.append(usage)
            except ai.AIUnavailable as exc:
                warning = str(exc)
        else:
            warning = 'No API key configured: detection completed; AI summarization and Q&A are disabled.'
        store.update(vid, status='ready', stage='Completed', progress=100, segments=segments,
                     summary=summary, warning=warning, ai_usage=usages,
                     elapsed_seconds=round(time.monotonic() - started, 2),
                     sample_fps=fps, ai_provider=cfg['provider'], ai_model=cfg['model'],
                     processed_frames=len(observations), detection_model=detection.model_name(),
                     detection_device='cloud-vlm' if is_vlm_det else 'cpu',
                     tracking='class-aware IoU', track_count=len({o['track_id'] for obs in observations for o in obs['objects']}))
    except Exception as exc:
        log.exception('Video processing failed: %s', vid)
        message = str(exc) if isinstance(exc, RuntimeError) else 'Video processing failed. Check backend logs and try again.'
        store.update(vid, status='failed', stage='Processing failed', error=message)
