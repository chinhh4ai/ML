# Industrial Analog Gauge Detection & Reading AI System

An advanced computer vision system designed for analog dial gauge meters:
- **Dial Face Object Detection**: Accurately bounds circular and square meter housings and pinpoints the needle pivot center.
- **Needle / Pointer Localization**: Detects pointer bounding box, rotation base, pointer tip point, deflection angle ($0^\circ - 360^\circ$), and distinguishes primary vs. secondary/setpoint indicators.
- **Scale Number & Tick Recognition**: Extracts scale numbers ($0, 10, 20 \dots 600$) with spatial bounding boxes and scale range minimums/maximums.
- **Reading Prediction**: Interactively interpolates needle deflection relative to adjacent tick marks, returning numeric readings with units ($^\circ\text{C}$, $\text{V}$, $\text{A}$, $\text{bar}$, $\dots$) and detailed AI step-by-step reasoning.
- **Multi-Gauge Panel Detection**: Segregates and parses multiple dials on industrial switchboards in a single frame.

---

## 1. Directory Structure

```
HIP/gauge_detection/
├── app/
│   ├── config.py                 # Pydantic Settings & .env loader
│   ├── main.py                   # FastAPI backend & REST API endpoints
│   ├── schemas.py                # Data models (BoundingBox, Needle, Reading, etc.)
│   ├── providers/
│   │   ├── base.py               # Abstract BaseGaugeDetector
│   │   ├── gemini_provider.py    # Google Gemini Vision (official google-genai SDK)
│   │   ├── openai_provider.py    # OpenAI Vision (GPT-4o, GPT-4o-mini)
│   │   ├── litellm_provider.py   # LiteLLM Proxy / Local vLLM (http://172.18.60.121:4000)
│   │   └── mock_provider.py      # Pre-calibrated ground-truth engine for instant offline testing
│   ├── services/
│   │   ├── detector_factory.py   # Dynamic provider switching factory
│   │   └── image_utils.py        # OpenCV bounding box drawing & thumbnail cropping
│   └── static/
│       ├── index.html            # SCADA Cockpit Dashboard UI
│       ├── style.css             # Dark Industrial Glassmorphism Theme
│       └── app.js                # Canvas rendering, interactive layers, analog dial SVG
├── data/
│   └── samples/                  # Real-world industrial gauge sample images
├── .env                          # Active environment configuration
├── .env.example                  # Environment configuration template
├── requirements.txt              # Python package requirements
├── run.sh                        # 1-click startup shell script
└── README.md                     # Documentation
```

---

## 2. Configuration & Provider Switching in [`.env`](file:///home/chinhh/workspace/HIP/gauge_detection/.env)

You can select the active AI provider in `.env` or dynamically via the **"Configure AI"** modal in the web interface:

```env
# Provider choice: 'gemini', 'openai', 'litellm', or 'mock'
AI_PROVIDER=gemini

# ----------------- 1. Google Gemini (Default) ------------------
# Obtain a free key at: https://aistudio.google.com/
GEMINI_API_KEY=AIzaSy...
GEMINI_MODEL=gemini-2.5-flash

# ----------------- 2. OpenAI Vision ----------------------------
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

# ----------------- 3. LiteLLM Proxy / Local vLLM ---------------
LITELLM_API_KEY=sk-1AdUrg6SJB5the4AmtWniQ
LITELLM_BASE_URL=http://172.18.60.121:4000
LITELLM_MODEL=Nemotron-3-Nano-30B-A3B-FP8-H200.0.124

# ----------------- Server Settings -----------------------------
HOST=0.0.0.0
PORT=5070
```

> **Offline / Demo Fallback**: When `GEMINI_API_KEY` is not set, the application automatically activates the **Mock / Pre-calibrated Engine** for the included samples, allowing immediate testing without crashes or timeouts. Once you provide an API key, live multimodal vision inference executes seamlessly.

---

## 3. How to Run

Activate your conda environment and run:

```bash
cd /home/chinhh/workspace/HIP/gauge_detection
conda activate hip1
./run.sh
```

Or run directly with Python:

```bash
/home/chinhh/miniconda3/envs/hip1/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 5070 --reload
```

Then open your browser at:
- **Web Dashboard**: `http://localhost:5070` (or `http://<YOUR_IP>:5070`)
- **Interactive Swagger API Docs**: `http://localhost:5070/docs`

---

## 4. Key UI Features

1. **1-Click Demo Samples**:
   - **Sample 1**: Wire-mesh reinforced glass ammeter (0-15 A).
   - **Sample 2**: Dual-needle Dial thermometer (0-100 °C) with primary black needle pointing at 20 °C and setpoint needle at ~77 °C.
   - **Sample 3**: Industrial panel with 3 meters (Voltmeter at 420 V, 2 Ammeters at 0 A).
   - **Sample 4**: Close-up ammeter resting at 0 A.
2. **Interactive Canvas Viewport**:
   - Toggle individual overlay layers with checkboxes:
     - 🟢 **Gauges**: Housing bounding boxes & rotation pivot crosshairs.
     - 🔴 **Pointers**: Needle rays from pivot to tip, with deflection angles.
     - 🟢 **Numbers**: OCR bounding boxes and recognized values.
     - 🟡 **HUD Badges**: Floating value callout pills directly next to the meters.
     - 🟣 **Angle Arcs**: Radial arc indicating deflection relative to zero.
   - Zoom in/out, fit to viewport, pan around high-resolution images.
   - Hover and click on multi-gauge elements to highlight specific meters.
3. **Animated Radial Dial Simulator**:
   - An interactive SVG analog meter mirroring the detected pointer deflection in real time.
4. **Custom Upload & Live Webcam**:
   - Drag & drop local images or capture snapshots directly from a connected webcam.
5. **Data Export**:
   - Download the annotated high-resolution image with visual bounding boxes.
   - Copy or inspect raw JSON response schemas for SCADA/IoT integrations.
