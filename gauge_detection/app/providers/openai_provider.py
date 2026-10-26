"""
OpenAI Vision Provider for Analog Gauge Detection and Reading
Uses openai.AsyncOpenAI with GPT-4o / GPT-4o-mini
"""
import io
import json
import time
import base64
from typing import Optional
from PIL import Image
from openai import AsyncOpenAI

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
from app.prompts import OpenAIGaugePrompt
from app.providers.utils import extract_and_parse_json, build_normalized_gauge_detection


class OpenAIGaugeDetector(BaseGaugeDetector):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o-mini"
    ):
        super().__init__(provider_name="openai", model_name=model_name)
        self.api_key = api_key
        self.base_url = base_url
        self.client = None
        self.prompt_builder = OpenAIGaugePrompt(model_name=model_name)
        if api_key and api_key.strip():
            try:
                self.client = AsyncOpenAI(api_key=api_key.strip(), base_url=base_url)
            except Exception as e:
                print(f"Warning: Failed to init AsyncOpenAI: {e}")

    async def detect(
        self,
        image_bytes: bytes,
        filename: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        gauge_type: Optional[str] = None,
    ) -> DetectionResult:
        start_time = time.time()
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size

        if not self.client or not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. Please set OPENAI_API_KEY in .env or via Settings."
            )

        try:
            b64_img = base64.b64encode(image_bytes).decode("utf-8")
            fmt = (img.format or "JPEG").lower()
            if fmt == "jpg":
                fmt = "jpeg"
            data_uri = f"data:image/{fmt};base64,{b64_img}"

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

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": data_uri}}
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            raw_text = response.choices[0].message.content or "{}"
            parsed_data = extract_and_parse_json(raw_text)

            # Convert to schema using normalized builder
            gauges_data = parsed_data.get("gauges", [])
            gauges = []
            for idx, g in enumerate(gauges_data, start=1):
                gauges.append(build_normalized_gauge_detection(g, default_index=idx))

            # Token usage extraction
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
            total_tokens = getattr(usage, "total_tokens", None) if usage else None
            if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens

            execution_ms = round((time.time() - start_time) * 1000, 1)

            return DetectionResult(
                success=True,
                provider="openai",
                model=self.model_name,
                image_width=width,
                image_height=height,
                gauge_count=len(gauges),
                gauges=gauges,
                execution_time_ms=execution_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                raw_explanation=parsed_data.get("explanation", "Detection finished with OpenAI Vision.")
            )

        except Exception as e:
            print(f"OpenAI detection error: {e}")
            raise RuntimeError(f"OpenAI detection error: {e}")
