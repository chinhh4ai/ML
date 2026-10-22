"""
Prompt Architecture Package for Gauge Detection
Base prompt with provider-specific extensions for Gemini, OpenAI, and LiteLLM.
"""
from app.prompts.base_prompt import BaseGaugePrompt
from app.prompts.gemini_prompt import GeminiGaugePrompt
from app.prompts.openai_prompt import OpenAIGaugePrompt
from app.prompts.litellm_prompt import LiteLLMGaugePrompt
from app.prompts.catalog import (
    GaugeTemplate,
    get_all_templates,
    get_template_by_id,
    get_prompt_for_type,
)

__all__ = [
    "BaseGaugePrompt",
    "GeminiGaugePrompt",
    "OpenAIGaugePrompt",
    "LiteLLMGaugePrompt",
    "GaugeTemplate",
    "get_all_templates",
    "get_template_by_id",
    "get_prompt_for_type",
]

