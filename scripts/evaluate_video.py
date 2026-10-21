"""Evaluate a local MOV/MP4 without upload limits. No AI traffic unless --with-ai is passed."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import ai, detection
from backend.config import ai_config, number
from backend.prompts import DOMAIN
from backend.tracking import Tracker


def annotate(image, objects):
    out = image.copy()
    height, width = out.shape[:2]
    for obj in objects:
        color = (65, 170, 250) if obj.get('debris_candidate') else (120, 220, 160)
        for polygon in obj.get('polygons', []):
            points = np.array([[int(x * width), int(y * height)] for x, y in polygon], np.int32)
            tint = out.copy()
            cv2.fillPoly(tint, [points], color)
            out = cv2.addWeighted(tint, .25, out, .75, 0)
            cv2.polylines(out, [points], True, color, 1)
        x1, y1, x2, y2 = [int(v * (width if i % 2 == 0 else height)) for i, v in enumerate(obj['box'])]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"#{obj['track_id']} {obj['label']} {obj['confidence']:.2f}"
        cv2.putText(out, text, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fps', type=float, default=2)
    parser.add_argument('--max-seconds', type=float, default=60, help='Analyze first 60s by default; 0 means whole video.')
    parser.add_argument('--with-ai', action='store_true', help='Explicitly send up to four sampled images to the configured provider.')
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error('Video file does not exist.')
    if not .1 <= args.fps <= 10 or args.max_seconds < 0:
        parser.error('fps must be 0.1..10 and max-seconds must be nonnegative.')
    if args.with_ai and not ai_config()['key']:
        parser.error('Configure API key in .env before --with-ai.')
    out = args.output or ROOT / '.data/evaluations' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error('Output directory must be empty to avoid overwriting an earlier evaluation.')
    probe = subprocess.run(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(args.video)], capture_output=True, check=True, text=True)
    metadata = json.loads(probe.stdout)
    duration = float(metadata['format']['duration'])
    duration = min(duration, args.max_seconds) if args.max_seconds else duration
    normalized = out / 'normalized.mp4'
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-threads', '2', '-i', str(args.video)]
    if args.max_seconds:
        command += ['-t', str(args.max_seconds)]
    command += ['-map', '0:v:0', '-an', '-vf', 'scale=min(1280\\,iw):-2', '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast', '-crf', '24', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(normalized)]
    print('Normalizing video (orientation/HDR conversion quality may require manual inspection)...', flush=True)
    subprocess.run(command, check=True)
    cap = cv2.VideoCapture(str(normalized))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    preview_path = out / 'overlay-temp.mp4'
    writer = cv2.VideoWriter(str(preview_path), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError('Cannot open video writer.')
    tracker = Tracker()
    counts, observations, inference_times = Counter(), [], []
    frame_count = max(1, int(duration * args.fps))
    keyframe_indices = set(int(i) for i in np.linspace(0, frame_count - 1, min(4, frame_count)))
    samples = []
    start = time.monotonic()
    for index in range(frame_count):
        timestamp = index / args.fps
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, image = cap.read()
        if not ok:
            break
        begin = time.monotonic()
        objects = tracker.update(detection.detect(image), timestamp)
        inference_times.append(time.monotonic() - begin)
        observations.append({'time': round(timestamp, 3), 'objects': objects})
        counts.update(o['label'] for o in objects)
        annotated = annotate(image, objects)
        cv2.putText(annotated, f'{timestamp:.1f}s | CPU | sampled {args.fps:g} FPS', (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
        writer.write(annotated)
        if index in keyframe_indices:
            path = out / f'frame-{index:06d}.jpg'
            cv2.imwrite(str(path), image)
            cv2.imwrite(str(out / f'overlay-{index:06d}.jpg'), annotated)
            samples.append(path)
        if index % max(1, int(args.fps * 5)) == 0:
            print(f'{timestamp:.1f}/{duration:.1f}s analyzed', flush=True)
    writer.release()
    cap.release()
    if not observations:
        raise RuntimeError('No frames could be decoded.')
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(preview_path), '-c:v', 'libx264', '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'overlay.mp4')], check=True)
    preview_path.unlink()
    report = {
        'input_name': args.video.name, 'input_bytes': args.video.stat().st_size,
        'analyzed_seconds': duration, 'sample_fps': args.fps, 'sampled_frames': len(observations),
        'model': detection.model_name(), 'confidence_threshold': number('DETECTION_CONFIDENCE', .25, .05, .99),
        'keyframe_times': [round(i / args.fps, 3) for i in sorted(keyframe_indices)],
        'device': 'CPUExecutionProvider', 'frame_detection_counts': dict(counts),
        'tracks_created': tracker.next_id - 1,
        'debris_candidate_detections': sum(bool(o.get('debris_candidate')) for obs in observations for o in obs['objects']),
        'detections_with_mask': sum(bool(o.get('polygons')) for obs in observations for o in obs['objects']),
        'analysis_wall_seconds': round(time.monotonic() - start, 2),
        'median_inference_ms': round(float(np.median(inference_times)) * 1000, 2),
        'observations': observations,
        'limitations': [
            'COCO pretrained is not a river-debris model. Zero detections does not mean no debris.',
            'No ground truth: precision, recall, mAP and segmentation IoU cannot be estimated.',
            'Candidate labels are a heuristic, not water localization or confirmed trash.',
            'IoU track IDs may split or swap; track count is not a unique object count.',
            'Overlay video is sampled FPS, not original full-frame-rate tracking.'
        ],
        'ai_status': 'not_requested',
    }
    if args.with_ai:
        try:
            prompt = DOMAIN + f"\nẢnh theo thứ tự ở các thời điểm: {report['keyframe_times']} giây. " + '\nĐánh giá các ảnh lấy mẫu theo thứ tự từ video. Nêu rác nghi vấn, vị trí, yếu tố dễ nhầm và giới hạn. Không có ground truth, không báo precision/recall. Chỉ đánh giá các ảnh đã cung cấp.'
            report['ai_assessment'], report['ai_usage'] = ai.generate(prompt, samples)
            report['ai_status'] = 'completed'
        except ai.AIUnavailable as exc:
            report['ai_status'], report['ai_error'] = 'failed', str(exc)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'observations'}, ensure_ascii=False, indent=2))
    print(f'Results: {out.resolve()}')


if __name__ == '__main__':
    main()
