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
from app.prompts import OpenAIGaugePrompt
from app.providers.utils import extract_and_parse_json


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
            mock = MockGaugeDetector(model_name="mock-fallback-no-openai-key")
            res = await mock.detect(image_bytes, filename=filename, custom_prompt=custom_prompt, gauge_type=gauge_type)
            res.provider = "mock (OpenAI API key not configured)"
            res.raw_explanation = "OPENAI_API_KEY is not configured. Running mock detector."
            return res

        try:
            b64_img = base64.b64encode(image_bytes).decode("utf-8")
            fmt = (img.format or "JPEG").lower()
            if fmt == "jpg":
                fmt = "jpeg"
            data_uri = f"data:image/{fmt};base64,{b64_img}"

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
            execution_ms = round((time.time() - start_time) * 1000, 1)
            print(f"OpenAI error: {e}. Falling back to mock detector.")
            mock = MockGaugeDetector(model_name="mock-fallback-on-error")
            fallback = await mock.detect(image_bytes, filename=filename)
            fallback.provider = f"mock (OpenAI Error: {str(e)})"
            fallback.error = str(e)
            return fallback
