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
from app.providers.utils import extract_and_parse_json, build_normalized_gauge_detection


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

        if not self.client or not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in .env or via Settings."
            )

        try:
            # Prepare image content
            img_format = img.format if img.format else "JPEG"
            mime_type = f"image/{img_format.lower()}"
            if mime_type == "image/jpg":
                mime_type = "image/jpeg"

            system_prompt = self.prompt_builder.build_system_prompt()
            if custom_prompt and custom_prompt.strip():
                user_prompt = (
                    f"{custom_prompt.strip()}\n\n"
                    "CRITICAL MANDATORY INSTRUCTIONS:\n"
                    "1. For dial/needle gauges, detect all needles in `needles` with exact tip_point and pivot_point.\n"
                    "2. Detect ALL printed scale numbers in `numbers` with exact `bbox` [ymin, xmin, ymax, xmax] in 0-1000 scale and `position`.\n"
                    "3. Provide `reading.reasoning` explaining needle angle and scale numbers.\n"
                    "4. Output strictly valid raw JSON matching the required system schema."
                )
            else:
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

            # Convert to schema using normalized builder
            gauges_data = parsed_data.get("gauges", [])
            gauges = []
            for idx, g in enumerate(gauges_data, start=1):
                gauges.append(build_normalized_gauge_detection(g, default_index=idx))

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
            print(f"Gemini detection error: {e}")
            raise RuntimeError(f"Gemini detection error: {e}")
