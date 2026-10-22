"""
Base detector interface for Gauge Detection
"""
from abc import ABC, abstractmethod
from typing import Optional
from app.schemas import DetectionResult


class BaseGaugeDetector(ABC):
    """Abstract Base Class for Gauge Detection and Reading Providers"""

    def __init__(self, provider_name: str, model_name: str):
        self.provider_name = provider_name
        self.model_name = model_name

    @abstractmethod
    async def detect(
        self,
        image_bytes: bytes,
        filename: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        gauge_type: Optional[str] = None,
    ) -> DetectionResult:
        """
        Detect gauges, needles, numbers, and predict readings from image bytes.

        Args:
            image_bytes: Raw binary image data (JPEG/PNG)
            filename: Optional original filename (useful for sample identification)
            custom_prompt: Optional user-customized inspection prompt
            gauge_type: Optional gauge category identifier


        Returns:
            DetectionResult containing list of detected gauges and reading predictions.
        """
        pass
