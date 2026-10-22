"""
Gemini Vision Prompt Definition for Analog Gauge Detection
Inherits from BaseGaugePrompt and adds Gemini-specific optimizations.
"""
from typing import Optional
from app.prompts.base_prompt import BaseGaugePrompt


class GeminiGaugePrompt(BaseGaugePrompt):
    """
    Prompt configuration tailored for Google Gemini Multimodal Models (Gemini 2.5 Flash, 1.5 Pro).
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        super().__init__(model_name=model_name)

    def build_system_prompt(self) -> str:
        base_prompt = super().build_system_prompt()
        gemini_directives = (
            "\n\nGEMINI SPECIFIC DIRECTIVES:\n"
            "- Return strictly valid JSON without any markdown formatting or commentary.\n"
            "- For multiple meters on electrical panels, detect and populate all distinct gauges into the `gauges` array.\n"
            "- Ensure `reading.value` accurately interpolates the needle position between dial scale ticks."
        )
        return base_prompt + gemini_directives
