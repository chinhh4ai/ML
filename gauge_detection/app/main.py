"""
FastAPI application for Industrial Gauge Detection & Reading
Supports Instant Precomputed Caching for Demo Samples
"""
import asyncio
import io
import os
import json
import cv2
import httpx
import numpy as np
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Body, Response, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

from app.config import settings, update_settings, SAMPLES_DIR, BASE_DIR
from app.schemas import DetectionResult, ConfigUpdateRequest, SampleInfo, ModelListRequest, ModelListResponse
from app.services.detector_factory import get_detector
from app.services.image_utils import crop_gauge_thumbnail, render_annotated_image
from app.prompts import GaugeTemplate, get_all_templates, get_template_by_id

app = FastAPI(
    title="Industrial Gauge Detection & Reading AI",
    description="Multi-gauge object detection, pointer angle analysis, scale numbers localization, and reading prediction.",
    version="1.0.0"
)

# Enable CORS for external dashboards / microservices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prevent browser from caching outdated JS/CSS/HTML during development
@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

PRECOMPUTED_DIR = BASE_DIR / "data" / "precomputed"
PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)


# Pre-defined sample metadata (English)
SAMPLE_REGISTRY = [
    SampleInfo(
        id="sample_1",
        filename="sample_1_mesh_glass_gauge.png",
        title="Wire Mesh Glass Ammeter",
        description="Ammeter dial gauge (0-15 A) located behind protective wire-mesh reinforced security glass.",
        expected_type="ammeter",
        thumbnail_url="/api/samples/sample_1_mesh_glass_gauge.png"
    ),
    SampleInfo(
        id="sample_2",
        filename="sample_2_hitachi_dial_thermometer.png",
        title="Dial Thermometer",
        description="Dial thermometer (0-100 °C) equipped with dual needles: primary black pointer (~20 °C) and secondary bronze limit/setpoint indicator (~77 °C).",
        expected_type="thermometer",
        thumbnail_url="/api/samples/sample_2_hitachi_dial_thermometer.png"
    ),
    SampleInfo(
        id="sample_3",
        filename="sample_3_panel_multigauge.png",
        title="Industrial Switchboard Panel",
        description="Multi-gauge electrical panel with 3 analog meters: 1 Voltmeter V (0-600V, pointing at ~420V) and 2 Ammeters A (0-30A, resting at 0A).",
        expected_type="multigauge",
        thumbnail_url="/api/samples/sample_3_panel_multigauge.png"
    ),
    SampleInfo(
        id="sample_4",
        filename="sample_4_panel_ammeter_closeup.png",
        title="Panel Ammeter Close-up",
        description="High-resolution close-up of an analog Ammeter (0-30 A) with black needle resting accurately at 0 A mark.",
        expected_type="ammeter",
        thumbnail_url="/api/samples/sample_4_panel_ammeter_closeup.png"
    ),
]




class Base64DetectRequest(BaseModel):
    image_base64: Optional[str] = None
    sample_id: Optional[str] = None
    filename: Optional[str] = None
    provider: Optional[str] = None
    force_live: Optional[bool] = False
    custom_prompt: Optional[str] = None
    gauge_type: Optional[str] = None


def load_precomputed_data(sample_id: str) -> Optional[DetectionResult]:
    """Load pre-cached detection result from disk for instant demonstration"""
    cache_file = PRECOMPUTED_DIR / f"{sample_id}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return DetectionResult(**data)
        except Exception as e:
            print(f"Warning: Failed to load precomputed cache for {sample_id}: {e}")
    return None


@app.get("/api/prompts/templates", response_model=List[GaugeTemplate])
async def list_prompt_templates():
    """List all gauge prompt templates based on real-world industrial catalog"""
    return get_all_templates()


@app.get("/")
async def root():
    """Serve the modern Gauge Detection dashboard UI"""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"status": "running", "docs": "/docs", "message": "Gauge Detection Backend is active"})


@app.get("/api/samples", response_model=List[SampleInfo])
async def list_samples():
    """List preloaded demonstration sample images with embedded precomputed detections"""
    results = []
    for s in SAMPLE_REGISTRY:
        s_copy = s.model_copy()
        s_copy.detection_data = load_precomputed_data(s.id)
        results.append(s_copy)
    return results


@app.get("/api/samples/{filename}")
async def get_sample_image(filename: str):
    """Retrieve sample image file"""
    target = SAMPLES_DIR / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Sample '{filename}' not found")
    return FileResponse(str(target))


@app.get("/api/samples/{sample_id}/detection", response_model=DetectionResult)
async def get_sample_detection(sample_id: str):
    """Retrieve instant precomputed detection for a sample, or fallback to live AI"""
    cached = load_precomputed_data(sample_id)
    if cached:
        return cached
    matched = next((s for s in SAMPLE_REGISTRY if s.id == sample_id or s.filename == sample_id), None)
    if not matched:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found")
    target = SAMPLES_DIR / matched.filename
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Sample image for '{sample_id}' not found")
    with open(target, "rb") as f:
        image_bytes = f.read()
    detector = get_detector()
    return await detector.detect(image_bytes=image_bytes, filename=matched.filename)


@app.get("/api/config")
async def get_config():
    """Get active provider status and runtime settings"""
    def mask_key(k: str):
        if not k or len(k) < 6:
            return "Not Set" if not k else "******"
        return f"{k[:3]}...{k[-3:]}"

    return {
        "provider": settings.AI_PROVIDER,
        "available_providers": ["hybrid", "gemini", "openai", "litellm"],
        "hybrid": {
            "model": f"YOLO + {settings.LITELLM_MODEL}",
            "litellm_model": settings.LITELLM_MODEL,
            "configured": bool(settings.LITELLM_BASE_URL)
        },
        "gemini": {
            "model": settings.GEMINI_MODEL,
            "configured": bool(settings.GEMINI_API_KEY),
            "key_preview": mask_key(settings.GEMINI_API_KEY)
        },
        "openai": {
            "model": settings.OPENAI_MODEL,
            "base_url": settings.OPENAI_BASE_URL,
            "configured": bool(settings.OPENAI_API_KEY),
            "key_preview": mask_key(settings.OPENAI_API_KEY)
        },
        "litellm": {
            "model": settings.LITELLM_MODEL,
            "base_url": settings.LITELLM_BASE_URL,
            "configured": bool(settings.LITELLM_BASE_URL)
        }
    }


@app.post("/api/config")
async def set_config(req: ConfigUpdateRequest):
    """Update active provider and API configurations"""
    updates = {}
    if req.provider is not None:
        updates["ai_provider"] = req.provider
    if req.gemini_api_key is not None:
        updates["gemini_api_key"] = req.gemini_api_key
    if req.gemini_model is not None:
        updates["gemini_model"] = req.gemini_model
    if req.openai_api_key is not None:
        updates["openai_api_key"] = req.openai_api_key
    if req.openai_base_url is not None:
        updates["openai_base_url"] = req.openai_base_url
    if req.openai_model is not None:
        updates["openai_model"] = req.openai_model
    if req.litellm_api_key is not None:
        updates["litellm_api_key"] = req.litellm_api_key
    if req.litellm_base_url is not None:
        updates["litellm_base_url"] = req.litellm_base_url
    if req.litellm_model is not None:
        updates["litellm_model"] = req.litellm_model

    update_settings(updates)
    return {"status": "success", "message": "Configuration updated successfully", "current_provider": settings.AI_PROVIDER}


@app.post("/api/models", response_model=ModelListResponse)
async def get_provider_models(req: ModelListRequest):
    """
    Fetch dynamically available models from the specified provider using the provided or configured credentials.
    """
    provider = req.provider.lower().strip()
    models = []
    error_msg = None

    if provider == "gemini":
        key = req.api_key.strip() if req.api_key and req.api_key.strip() else settings.GEMINI_API_KEY
        current_model = settings.GEMINI_MODEL or "gemini-flash-lite-latest"

        # Curated list of verified working Gemini vision models with active quota.
        # Tested and validated: models that support multimodal gauge detection without 429/404 errors.
        VERIFIED_WORKING_GEMINI_MODELS = [
            "gemini-flash-lite-latest",
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-3.1-flash-lite-preview",
            "gemini-3-flash-preview",
        ]

        if not key:
            models = VERIFIED_WORKING_GEMINI_MODELS
            error_msg = "Please enter Gemini API Key to verify access."
        else:
            try:
                def _fetch():
                    cl = genai.Client(api_key=key)
                    catalog = set()
                    for m in cl.models.list():
                        name = (m.name or "").replace("models/", "")
                        catalog.add(name)
                    # Filter: Only keep models that are confirmed working with active quota
                    available = [m for m in VERIFIED_WORKING_GEMINI_MODELS if m in catalog]
                    return available if available else VERIFIED_WORKING_GEMINI_MODELS

                fetched = await asyncio.to_thread(_fetch)
                models = fetched
            except Exception as e:
                error_msg = f"Failed to fetch Gemini models: {str(e)}"
                models = VERIFIED_WORKING_GEMINI_MODELS

        return ModelListResponse(provider="gemini", models=models, current_model=current_model, error=error_msg)

    elif provider in ("litellm", "hybrid"):
        base_url = (req.base_url.strip() if req.base_url and req.base_url.strip() else settings.LITELLM_BASE_URL) or "http://172.18.60.121:4000"
        base_url = base_url.rstrip("/")
        api_key = req.api_key.strip() if req.api_key and req.api_key.strip() else settings.LITELLM_API_KEY
        current_model = settings.LITELLM_MODEL or "Llama-3.1-Nemotron-Nano-VL-8B-V1-FP4-QAD-A30.0.121"
        endpoint = f"{base_url}/models" if base_url.endswith("/v1") else f"{base_url}/v1/models"
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(endpoint, headers=headers)
                if r.status_code == 200:
                    data = r.json()
                    models = [m["id"] for m in data.get("data", []) if "id" in m]
                else:
                    error_msg = f"LiteLLM proxy returned status {r.status_code}"
                    models = ["Llama-3.1-Nemotron-Nano-VL-8B-V1-FP4-QAD-A30.0.121", "Gemma-4-31B-IT-NVFP4-L40.0.127", "Gemma-26B-A4B-It-L40.1.127"]
        except Exception as e:
            error_msg = f"Could not connect to LiteLLM: {str(e)}"
            models = ["Llama-3.1-Nemotron-Nano-VL-8B-V1-FP4-QAD-A30.0.121", "Gemma-4-31B-IT-NVFP4-L40.0.127", "Gemma-26B-A4B-It-L40.1.127"]

        return ModelListResponse(provider=provider, models=models, current_model=current_model, error=error_msg)

    elif provider == "openai":
        base_url = (req.base_url.strip() if req.base_url and req.base_url.strip() else settings.OPENAI_BASE_URL) or "https://api.openai.com/v1"
        base_url = base_url.rstrip("/")
        api_key = req.api_key.strip() if req.api_key and req.api_key.strip() else settings.OPENAI_API_KEY
        current_model = settings.OPENAI_MODEL or "gpt-4o-mini"
        endpoint = f"{base_url}/models" if base_url.endswith("/v1") else f"{base_url}/v1/models"

        if not api_key:
            models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1-mini", "o1-preview"]
            error_msg = "Please enter OpenAI API Key to fetch live model list."
        else:
            headers = {"Authorization": f"Bearer {api_key}"}
            try:
                async with httpx.AsyncClient(timeout=6.0) as client:
                    r = await client.get(endpoint, headers=headers)
                    if r.status_code == 200:
                        data = r.json()
                        all_ids = [m["id"] for m in data.get("data", []) if "id" in m]
                        vision_ids = [m for m in all_ids if any(k in m.lower() for k in ["gpt-4", "o1", "chatgpt"])]
                        models = sorted(vision_ids) if vision_ids else sorted(all_ids)
                    else:
                        error_msg = f"OpenAI returned status {r.status_code}"
                        models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1-mini"]
            except Exception as e:
                error_msg = f"Failed to fetch OpenAI models: {str(e)}"
                models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1-mini"]

        return ModelListResponse(provider="openai", models=models, current_model=current_model, error=error_msg)

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider '{provider}'")


@app.post("/api/detect", response_model=DetectionResult)
async def detect_gauge(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    provider_override: Optional[str] = Form(None),
    force_live: Optional[bool] = Form(False),
    custom_prompt: Optional[str] = Form(None),
    gauge_type: Optional[str] = Form(None),
):
    """
    Detect gauges, pointers, dial numbers, and predict readings from uploaded image or sample.
    Instantly returns precomputed cache for demo samples if no provider override, force_live, or custom prompt is requested.
    """
    # 1. Fast Path for demo samples
    if sample_id and not force_live and not provider_override and not custom_prompt and (not gauge_type or gauge_type.strip().lower() == "default"):
        cached = load_precomputed_data(sample_id)
        if cached:
            return cached

    image_bytes = None
    filename = None

    if file and file.filename:
        image_bytes = await file.read()
        filename = file.filename
    elif sample_id:
        matched = next((s for s in SAMPLE_REGISTRY if s.id == sample_id or s.filename == sample_id), None)
        if matched:
            target = SAMPLES_DIR / matched.filename
            if target.exists():
                with open(target, "rb") as f:
                    image_bytes = f.read()
                filename = matched.filename
        else:
            raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image provided. Upload a file or specify a sample_id.")

    detector = get_detector(provider_override=provider_override)
    result = await detector.detect(
        image_bytes=image_bytes,
        filename=filename,
        custom_prompt=custom_prompt,
        gauge_type=gauge_type
    )

    # Generate cropped thumbnails for each gauge
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_np is not None:
            for g in result.gauges:
                g.crop_base64 = crop_gauge_thumbnail(img_np, g.bbox)
    except Exception as e:
        print(f"Warning: Failed to generate thumbnail crops: {e}")

    return result


@app.post("/api/detect/json", response_model=DetectionResult)
async def detect_gauge_json(payload: Base64DetectRequest):
    """Detect gauges using JSON payload (base64 image or sample_id)"""
    import base64

    # Fast path for demo samples
    if payload.sample_id and not payload.force_live and not payload.provider and not payload.custom_prompt and (not payload.gauge_type or payload.gauge_type.strip().lower() == "default"):
        cached = load_precomputed_data(payload.sample_id)
        if cached:
            return cached

    image_bytes = None
    filename = payload.filename

    if payload.sample_id:
        matched = next((s for s in SAMPLE_REGISTRY if s.id == payload.sample_id or s.filename == payload.sample_id), None)
        if matched:
            target = SAMPLES_DIR / matched.filename
            if target.exists():
                with open(target, "rb") as f:
                    image_bytes = f.read()
                filename = matched.filename

    elif payload.image_base64:
        raw_b64 = payload.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[1]
        image_bytes = base64.b64decode(raw_b64)

    if not image_bytes:
        raise HTTPException(status_code=400, detail="No valid image data or sample_id provided.")

    detector = get_detector(provider_override=payload.provider)
    result = await detector.detect(
        image_bytes=image_bytes,
        filename=filename,
        custom_prompt=payload.custom_prompt,
        gauge_type=payload.gauge_type
    )

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_np is not None:
            for g in result.gauges:
                g.crop_base64 = crop_gauge_thumbnail(img_np, g.bbox)
    except Exception as e:
        print(f"Warning: Failed to generate thumbnail crops: {e}")

    return result


@app.post("/api/detect/annotated")
async def detect_annotated_image(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    provider_override: Optional[str] = Form(None)
):
    """
    Run detection and return JPEG image with visual bounding boxes, needle vectors, and HUD tags rendered.
    """
    image_bytes = None
    filename = None

    if file and file.filename:
        image_bytes = await file.read()
        filename = file.filename
    elif sample_id:
        matched = next((s for s in SAMPLE_REGISTRY if s.id == sample_id or s.filename == sample_id), None)
        if matched:
            target = SAMPLES_DIR / matched.filename
            if target.exists():
                with open(target, "rb") as f:
                    image_bytes = f.read()
                filename = matched.filename

    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image provided")

    result = None
    if sample_id and (not provider_override or provider_override in ("precomputed", "mock")):
        result = load_precomputed_data(sample_id)

    if not result:
        detector = get_detector(provider_override=provider_override)
        result = await detector.detect(image_bytes=image_bytes, filename=filename)

    annotated_bytes = render_annotated_image(image_bytes, result)
    return Response(content=annotated_bytes, media_type="image/jpeg")
