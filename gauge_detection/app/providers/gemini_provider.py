"""
Google Gemini Vision Provider for Analog Gauge Detection and Reading
Uses the official google-genai SDK
"""
import asyncio
import io
import json
import time
from typing import Optional
from PIL import Image
from google import genai
from google.genai import types

from app.providers.base import BaseGaugeDetector
from app.providers.mock_provider import MockGaugeDetector
from app.schemas import (
    DetectionResult,
    GaugeDetection,
    NeedleDetection,
    ScaleNumberDetection,
    GaugeReading,
    BoundingBox,
    Point2D,
)


from app.prompts import GeminiGaugePrompt
from app.providers.utils import extract_and_parse_json


class GeminiGaugeDetector(BaseGaugeDetector):
    def __init__(self, api_key: str, model_name: str = "gemini-flash-lite-latest"):
        super().__init__(provider_name="gemini", model_name=model_name)
        self.api_key = api_key
        self.client = None
        self.prompt_builder = GeminiGaugePrompt(model_name=model_name)
        if api_key and api_key.strip():
            try:
                self.client = genai.Client(api_key=api_key.strip())
            except Exception as e:
                print(f"Warning: Failed to initialize Gemini Client: {e}")

    async def detect(
        self,
        image_bytes: bytes,
        filename: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        gauge_type: Optional[str] = None,
    ) -> DetectionResult:
        start_time = time.time()

        # Load image for dimension inspection
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size

        # If no client or API key is provided, use mock detector as fallback
        if not self.client or not self.api_key:
            mock_detector = MockGaugeDetector(model_name="mock-fallback-no-api-key")
            result = await mock_detector.detect(image_bytes, filename=filename, custom_prompt=custom_prompt, gauge_type=gauge_type)
            result.provider = "mock (Gemini API key not configured)"
            result.raw_explanation = (
                "GEMINI_API_KEY is not configured. The system is operating in mock/pre-calibrated mode. "
                "Please configure an API key in .env or via the Settings modal to run live Gemini Vision inference."
            )
            return result

        try:
            # Prepare image content
            img_format = img.format if img.format else "JPEG"
            mime_type = f"image/{img_format.lower()}"
            if mime_type == "image/jpg":
                mime_type = "image/jpeg"

            if custom_prompt and custom_prompt.strip():
                user_prompt = custom_prompt.strip()
                system_prompt = (
                    "You are an expert industrial computer vision inspector specializing in industrial meters.\n"
                    "Carefully follow the user prompt instructions and output strictly valid raw JSON starting with '{' and ending with '}'.\n"
                    "Do NOT include markdown fences, preambles, or conversational text outside the JSON."
                )
            else:
                system_prompt = self.prompt_builder.build_system_prompt()
                user_prompt = self.prompt_builder.build_user_prompt(filename=filename)

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    user_prompt
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    temperature=0.1,
                )
            )

            raw_text = response.text or "{}"
            parsed_data = extract_and_parse_json(raw_text)

            # Robust conversion helpers for both list and dict formats
            def safe_bbox(data, default_ymin=0, default_xmin=0, default_ymax=1000, default_xmax=1000) -> BoundingBox:
                if isinstance(data, (list, tuple)) and len(data) >= 4:
                    return BoundingBox(ymin=float(data[0]), xmin=float(data[1]), ymax=float(data[2]), xmax=float(data[3]))
                elif isinstance(data, dict):
                    return BoundingBox(
                        ymin=float(data.get("ymin", default_ymin)),
                        xmin=float(data.get("xmin", default_xmin)),
                        ymax=float(data.get("ymax", default_ymax)),
                        xmax=float(data.get("xmax", default_xmax))
                    )
                return BoundingBox(ymin=default_ymin, xmin=default_xmin, ymax=default_ymax, xmax=default_xmax)

            def safe_point(data, default_x=500, default_y=500) -> Point2D:
                if isinstance(data, (list, tuple)) and len(data) >= 2:
                    return Point2D(x=float(data[0]), y=float(data[1]))
                elif isinstance(data, dict):
                    return Point2D(x=float(data.get("x", default_x)), y=float(data.get("y", default_y)))
                return Point2D(x=default_x, y=default_y)

            # Convert to schema
            gauges_data = parsed_data.get("gauges", [])
            gauges = []
            for g in gauges_data:
                bbox_obj = safe_bbox(g.get("bbox"))
                center_obj = safe_point(g.get("dial_center"), 500, 500)
                
                needles = []
                for n in g.get("needles", []):
                    nb_obj = safe_bbox(n.get("bbox")) if n.get("bbox") else None
                    needles.append(NeedleDetection(
                        needle_id=n.get("needle_id", "needle_1"),
                        type=n.get("type", "primary_pointer"),
                        bbox=nb_obj,
                        pivot_point=safe_point(n.get("pivot_point"), center_obj.x, center_obj.y),
                        tip_point=safe_point(n.get("tip_point"), 500, 200),
                        angle_degrees=n.get("angle_degrees"),
                        color=n.get("color", "black")
                    ))

                numbers = []
                for num in g.get("numbers", []):
                    n_bbox_obj = safe_bbox(num.get("bbox"), 0, 0, 0, 0)
                    pos_obj = safe_point(num.get("position"), 0, 0) if num.get("position") else None
                    numbers.append(ScaleNumberDetection(
                        value=str(num.get("value", "")),
                        numeric_value=num.get("numeric_value"),
                        bbox=n_bbox_obj,
                        position=pos_obj
                    ))

                read_data = g.get("reading", {})
                reading = GaugeReading(
                    value=float(read_data.get("value", 0.0)),
                    unit=read_data.get("unit", ""),
                    formatted=read_data.get("formatted", f"{read_data.get('value', 0.0)} {read_data.get('unit', '')}"),
                    min_range=read_data.get("min_range"),
                    max_range=read_data.get("max_range"),
                    confidence=float(read_data.get("confidence", 0.9)),
                    reasoning=read_data.get("reasoning", "")
                )

                gauges.append(GaugeDetection(
                    gauge_id=g.get("gauge_id", f"gauge_{len(gauges)+1}"),
                    label=g.get("label", "Analog Gauge"),
                    gauge_type=g.get("gauge_type", "generic"),
                    bbox=bbox_obj,
                    dial_center=center_obj,
                    needles=needles,
                    numbers=numbers,
                    reading=reading
                ))

            # Token usage extraction
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", None) if usage else None
            completion_tokens = getattr(usage, "candidates_token_count", None) if usage else None
            total_tokens = getattr(usage, "total_token_count", None) if usage else None
            if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens

            execution_ms = round((time.time() - start_time) * 1000, 1)

            return DetectionResult(
                success=True,
                provider="gemini",
                model=self.model_name,
                image_width=width,
                image_height=height,
                gauge_count=len(gauges),
                gauges=gauges,
                execution_time_ms=execution_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                raw_explanation=parsed_data.get("explanation", "Detection finished successfully with Gemini Vision.")
            )

        except Exception as e:
            execution_ms = round((time.time() - start_time) * 1000, 1)
            print(f"Gemini detection error: {e}. Falling back to pre-calibrated mock detector.")
            # Fallback gracefully so the app stays functional
            mock_detector = MockGaugeDetector(model_name="mock-fallback-on-error")
            fallback_res = await mock_detector.detect(image_bytes, filename=filename)
            fallback_res.provider = f"mock (Gemini Error: {str(e)})"
            fallback_res.error = str(e)
            return fallback_res
