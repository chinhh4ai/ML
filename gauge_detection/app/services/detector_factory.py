"""
Detector Factory for dynamically selecting the active Gauge Detection provider
"""
from app.config import settings
from app.providers.base import BaseGaugeDetector
from app.providers.gemini_provider import GeminiGaugeDetector
from app.providers.openai_provider import OpenAIGaugeDetector
from app.providers.hybrid_provider import HybridGaugeDetector
from app.providers.mock_provider import MockGaugeDetector


def get_detector(provider_override: str = None) -> BaseGaugeDetector:
    """Return an instance of the configured AI provider"""
    provider = (provider_override or settings.AI_PROVIDER or "gemini").lower().strip()

    if provider == "hybrid":
        return HybridGaugeDetector(
            litellm_api_key=settings.LITELLM_API_KEY,
            litellm_base_url=settings.LITELLM_BASE_URL,
            litellm_model=settings.LITELLM_MODEL
        )
    elif provider == "gemini":
        return GeminiGaugeDetector(
            api_key=settings.GEMINI_API_KEY,
            model_name=settings.GEMINI_MODEL
        )
    elif provider == "openai":
        return OpenAIGaugeDetector(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model_name=settings.OPENAI_MODEL
        )
    elif provider == "litellm":
        return LiteLLMGaugeDetector(
            api_key=settings.LITELLM_API_KEY,
            base_url=settings.LITELLM_BASE_URL,
            model_name=settings.LITELLM_MODEL
        )
    elif provider in ("mock", "offline", "demo"):
        return MockGaugeDetector(model_name="mock-precalibrated")
    else:
        # Default fallback to Gemini
        return GeminiGaugeDetector(
            api_key=settings.GEMINI_API_KEY,
            model_name=settings.GEMINI_MODEL
        )
