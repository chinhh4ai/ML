# Validation — 2026-10-21

## Executed in the current Linux development environment

- `python -m pytest -q`: **9 passed**. Includes actual CPU ONNX segmentation, mask polygons, track association, video upload/transcode, image/media retrieval and HTTP Range, persisted SQLite results, invalid uploads, provider request formats, sanitized errors, Vietnamese object search, mocked AI caption/summary pipeline and rejection of invented citation IDs.
- One non-failing Starlette TestClient deprecation warning about its future HTTP client backend. No assertions disabled.
- `npm ci && npm run build`: passed (TypeScript and production Vite build).
- `bash scripts/setup.sh`: passed after moving pip/npm cache directories under writable `.data`; a prior run failed because the container home directory is read-only. Package integrity checks remain enabled.
- `bash scripts/install-ubuntu.sh --skip-system`: tested on the current Linux machine. This tests project setup, not Ubuntu apt installation.
- `bash -n scripts/*.sh`: passed.
- Generated `riverlens.service` passed `systemd-analyze verify`. The service was **not installed** into this container.
- Playwright with system Chromium: library page, analysis page, actual segmentation overlay, object list, unavailable-AI chat state, settings modal and mobile horizontal overflow checks passed; no browser JavaScript errors.
- API server started on port **5076**, health endpoint reports CPU / YOLO11n-seg / segmentation enabled. Service restart and retained library are checked before delivery.
- Bundled ONNX models pass SHA256 verification. YOLO11n-seg was exported on CPU from checksum-verified official weights.

## Functional video check (not a debris benchmark)

The reference is an official Ultralytics street-scene image repeated as a four-second video, explicitly labeled as a technical test. Current-instance processing: 8 sampled frames at 2 FPS, 5 track IDs, about 1.24 seconds wall time for transcode/analysis in that run. This tiny repeated-frame result is **not** a throughput guarantee or an accuracy estimate for real videos.

Standalone `evaluate_video.py` tested with a two-second excerpt: 4 sampled frames, 16 detections with masks, overlay MP4 and JSON report produced. It correctly reported zero debris candidates for that test and explicitly did not equate that with absence of real debris. Results vary with codec/transcoding and confidence.

## Unverified / blocked

- Actual Gemini/OpenAI/LiteLLM calls: no live API key was configured during these checks. Adapter and AI pipeline tests use mocks; they do not prove account permissions, selected-model availability, quota or real answer quality.
- User upload `20260915_010416000_iOS.MOV`: transfer rejected because the attached file exceeds the executor's **32 MiB** download limit. Its frames were not read and no claims about its contents, masks or debris accuracy are made. CLI supports evaluating the original file on the user's machine without this attachment limit.
- River/lake/pond debris precision, recall, segmentation IoU and tracking quality: no accessible user clip or labeled ground truth yet. COCO weights are general-purpose; additional domain data/fine-tuning is needed for reliable deployment.
- Ubuntu 24.04 apt installation, actual systemd enable/start, Windows PowerShell execution, and iOS HDR color correctness: not executed on those target systems.
- No source was written to a Windows D: drive; that machine is not attached. ZIP is prepared for extraction into `D:\work_space\2026\HIP`.
- These validation runs preceded the initial GitHub push. No public deployment or environment publication was performed.

Search is keyword retrieval, tracking is basic class-aware IoU, and neither is presented as an embedding search engine or a verified unique-object counter.
