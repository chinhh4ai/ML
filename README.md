# Machine Learning Services (HIP ML)

This repository contains computer vision and machine learning applications:

## Projects

### 1. [`debris_detection`](./debris_detection)
- **Riverlens Aquatic Debris Intelligence**: Real-time river intake surveillance and floating debris detection.
- **Tech Stack**: Fast ONNX YOLO11 instance segmentation, VLM zero-shot multimodal detection (Gemini Vision / Nemotron), FastAPI backend, React / Vite frontend.
- **Port**: 5050 (default)

### 2. [`gauge_detection`](./gauge_detection)
- **GaugeVision AI**: Industrial analog gauge meter detection, needle pointer angle estimation, scale tick recognition, and value prediction.
- **Tech Stack**: Multimodal Vision Language Models (Gemini / OpenAI / LiteLLM), OpenCV vector geometry, FastAPI backend, SCADA cockpit web UI.
- **Port**: 5070 (default)
