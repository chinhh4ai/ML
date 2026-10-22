"""
OpenAI Vision Prompt Definition for Analog Gauge Detection
Inherits from BaseGaugePrompt and adds OpenAI-specific optimizations (GPT-4o, GPT-4o-mini).
"""
from typing import Optional
from app.prompts.base_prompt import BaseGaugePrompt


class OpenAIGaugePrompt(BaseGaugePrompt):
    """
    Prompt configuration tailored for OpenAI Vision (GPT-4o, GPT-4o-mini).
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        super().__init__(model_name=model_name)

    def build_system_prompt(self) -> str:
        base_prompt = super().build_system_prompt()
        openai_directives = (
            "\n\nOPENAI SPECIFIC DIRECTIVES:\n"
            "- Output valid JSON object only. Ensure all numeric coordinates are integers or floats, never strings.\n"
            "- If a needle is resting at 0, accurately record `angle_degrees` and `reading.value: 0.0`.\n"
            "- Provide step-by-step angle and scale tick derivation in `reading.reasoning`."
        )
        return base_prompt + openai_directives
