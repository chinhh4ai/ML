"""
LiteLLM Proxy Provider for Analog Gauge Detection and Reading
Connects to LiteLLM proxy server (http://172.18.60.121:4000 or custom)
Uses dedicated LiteLLMGaugePrompt and robust JSON parsing for diverse open-source vision models.
"""
import io
import time
import base64
from typing import Optional
from PIL import Image
from openai import AsyncOpenAI

from app.providers.base import BaseGaugeDetector
from app.prompts import LiteLLMGaugePrompt
from app.providers.utils import extract_and_parse_json, build_normalized_gauge_detection
from app.schemas import (
    DetectionResult,
    GaugeDetection,
    NeedleDetection,
    ScaleNumberDetection,
    GaugeReading,
    BoundingBox,
    Point2D,
)


class LiteLLMGaugeDetector(BaseGaugeDetector):
    def __init__(
        self,
        api_key: str,
        base_url: str = "http://172.18.60.121:4000",
        model_name: str = "Gemma-26B-A4B-It-L40.1.127"
    ):
        super().__init__(provider_name="litellm", model_name=model_name)
        self.api_key = api_key or "sk-dummy-litellm"
        clean_base = base_url.rstrip("/") if base_url else "http://172.18.60.121:4000"
        if not clean_base.endswith("/v1"):
            self.base_url = f"{clean_base}/v1"
        else:
            self.base_url = clean_base

        self.client = None
        try:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            print(f"Warning: Failed to init LiteLLM AsyncOpenAI client: {e}")

        # Dedicated prompt builder instance for this model
        self.prompt_builder = LiteLLMGaugePrompt(model_name=self.model_name)

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

        if not self.client:
            raise RuntimeError(
                "LiteLLM client is not initialized. Please verify LITELLM_BASE_URL and configuration."
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


            # Request vision inference via LiteLLM proxy
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
                max_tokens=2500,
                temperature=0.1
            )

            choice = response.choices[0]
            raw_text = choice.message.content or ""

            # Some reasoning models put the text in reasoning_content when content is empty
            if not raw_text.strip():
                raw_text = getattr(choice.message, "reasoning_content", "") or ""

            # Robust JSON extraction handling markdown fences, <think> tags, etc.
            parsed_data = extract_and_parse_json(raw_text)

            # Convert to schema using normalized builder
            gauges_data = parsed_data.get("gauges", [])
            gauges = []
            for idx, g in enumerate(gauges_data, start=1):
                gauges.append(build_normalized_gauge_detection(g, default_index=idx))

            # If model produced 0 gauges, fallback gracefully
            if not gauges:
                raise ValueError("Model completed inference but returned 0 valid gauges in schema.")

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
                provider="litellm",
                model=self.model_name,
                image_width=width,
                image_height=height,
                gauge_count=len(gauges),
                gauges=gauges,
                execution_time_ms=execution_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                raw_explanation=parsed_data.get("explanation", f"Detection completed with LiteLLM ({self.model_name}).")
            )

        except Exception as e:
            print(f"LiteLLM detection error: {e}")
            raise RuntimeError(f"LiteLLM detection error: {e}")
