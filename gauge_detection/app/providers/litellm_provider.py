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
from app.providers.mock_provider import MockGaugeDetector
from app.prompts import LiteLLMGaugePrompt
from app.providers.utils import extract_and_parse_json
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
            mock = MockGaugeDetector(model_name="mock-fallback-no-litellm-client")
            res = await mock.detect(image_bytes, filename=filename, custom_prompt=custom_prompt, gauge_type=gauge_type)
            res.provider = "mock (LiteLLM client not initialized)"
            res.raw_explanation = "LiteLLM client is not initialized. Running mock detector."
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
                raw_needles = g.get("needles")
                if raw_needles is None and "needle" in g:
                    needle_val = g.get("needle")
                    raw_needles = [needle_val] if isinstance(needle_val, dict) else []
                elif not isinstance(raw_needles, list):
                    raw_needles = []

                for n in raw_needles:
                    if not isinstance(n, dict):
                        continue
                    nb_obj = safe_bbox(n.get("bbox")) if n.get("bbox") else None
                    pivot_p = n.get("pivot_point") or n.get("center")
                    tip_p = n.get("tip_point") or n.get("tip")
                    angle_v = n.get("angle_degrees") if n.get("angle_degrees") is not None else n.get("angle_deg")
                    needles.append(NeedleDetection(
                        needle_id=n.get("needle_id", "needle_1"),
                        type=n.get("type", "primary_pointer"),
                        bbox=nb_obj,
                        pivot_point=safe_point(pivot_p, center_obj.x, center_obj.y),
                        tip_point=safe_point(tip_p, 500, 200),
                        angle_degrees=float(angle_v) if angle_v is not None else None,
                        color=n.get("color", "black")
                    ))

                numbers = []
                for num in g.get("numbers", []):
                    if not isinstance(num, dict):
                        continue
                    n_bbox_obj = safe_bbox(num.get("bbox"), 0, 0, 0, 0)
                    pos_obj = safe_point(num.get("position"), 0, 0) if num.get("position") else None
                    numbers.append(ScaleNumberDetection(
                        value=str(num.get("value", "")),
                        numeric_value=num.get("numeric_value"),
                        bbox=n_bbox_obj,
                        position=pos_obj
                    ))

                raw_reading = g.get("reading", {})
                raw_scale = g.get("scale", {})
                if isinstance(raw_reading, (int, float)):
                    read_data = {"value": float(raw_reading), "unit": str(g.get("unit") or "")}
                elif isinstance(raw_reading, dict):
                    read_data = raw_reading
                else:
                    read_data = {"value": g.get("reading_value", 0.0), "unit": str(g.get("unit") or "")}

                min_r = read_data.get("min_range")
                max_r = read_data.get("max_range")
                if isinstance(raw_scale, dict):
                    if min_r is None:
                        min_r = raw_scale.get("min_val")
                    if max_r is None:
                        max_r = raw_scale.get("max_val")

                rval = float(read_data.get("value", 0.0)) if read_data.get("value") is not None else 0.0
                runit = str(read_data.get("unit") or g.get("unit") or "")
                clean_unit = runit if runit.lower() not in ["null", "none", "unknown", ""] else ""
                r_fmt = read_data.get("formatted") or (f"{rval:g} {clean_unit}".strip() if clean_unit else f"{rval:g}")

                reading = GaugeReading(
                    value=rval,
                    unit=clean_unit,
                    formatted=r_fmt,
                    min_range=float(min_r) if min_r is not None else None,
                    max_range=float(max_r) if max_r is not None else None,
                    confidence=float(read_data.get("confidence", 0.9)),
                    reasoning=read_data.get("reasoning", "")
                )

                raw_label = g.get("label")
                gtype = (g.get("gauge_type") or "generic").lower().replace(" ", "_")
                if not raw_label:
                    clean_l = gtype.replace("analog_", "").replace("_", " ").title()
                    raw_label = f"Industrial {clean_l}"

                gauges.append(GaugeDetection(
                    gauge_id=g.get("gauge_id", f"gauge_{len(gauges)+1}"),
                    label=raw_label,
                    gauge_type=gtype,
                    bbox=bbox_obj,
                    dial_center=center_obj,
                    needles=needles,
                    numbers=numbers,
                    reading=reading
                ))

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
            execution_ms = round((time.time() - start_time) * 1000, 1)
            print(f"LiteLLM error: {e}. Falling back to mock detector.")
            mock = MockGaugeDetector(model_name="mock-fallback-on-litellm-error")
            fallback = await mock.detect(image_bytes, filename=filename)
            fallback.provider = f"mock (LiteLLM Error: {str(e)})"
            fallback.error = str(e)
            return fallback
