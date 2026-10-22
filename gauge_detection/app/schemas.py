"""
Pydantic schemas for Gauge Detection and Reading System
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Normalized coordinates 0 to 1000 (standard for multimodal vision models)"""
    ymin: float = Field(..., description="Top coordinate (0-1000)")
    xmin: float = Field(..., description="Left coordinate (0-1000)")
    ymax: float = Field(..., description="Bottom coordinate (0-1000)")
    xmax: float = Field(..., description="Right coordinate (0-1000)")


class Point2D(BaseModel):
    """Normalized coordinate 0 to 1000"""
    x: float = Field(..., description="X coordinate (0-1000)")
    y: float = Field(..., description="Y coordinate (0-1000)")


class NeedleDetection(BaseModel):
    """Detected needle/pointer inside a gauge"""
    needle_id: str = Field(..., description="Identifier for this needle")
    type: str = Field("primary_pointer", description="primary_pointer, secondary_pointer, setpoint, or marker")
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box around the needle")
    pivot_point: Point2D = Field(..., description="Center pivot where the needle rotates")
    tip_point: Point2D = Field(..., description="Pointy tip of the needle")
    angle_degrees: Optional[float] = Field(None, description="Angle in degrees (0 to 360 clockwise)")
    color: Optional[str] = Field("black", description="Needle color e.g. black, red, silver, bronze")


class ScaleNumberDetection(BaseModel):
    """Detected number/scale marking on the gauge dial"""
    value: str = Field(..., description="Recognized text e.g. '0', '10', '20', '400'")
    numeric_value: Optional[float] = Field(None, description="Numeric parsed value")
    bbox: BoundingBox = Field(..., description="Bounding box of the number")
    position: Optional[Point2D] = Field(None, description="Center position of the number")


class GaugeReading(BaseModel):
    """Predicted reading of the gauge"""
    value: float = Field(..., description="Numeric predicted reading value")
    unit: str = Field("", description="Unit of measurement e.g. °C, V, A, bar, psi, rpm")
    formatted: str = Field(..., description="Formatted reading string e.g. '20 °C', '420 V'")
    min_range: Optional[float] = Field(None, description="Minimum scale value")
    max_range: Optional[float] = Field(None, description="Maximum scale value")
    confidence: float = Field(0.9, description="Confidence score 0.0 - 1.0")
    reasoning: str = Field("", description="Explanation of how the reading was derived from needle angle and scale numbers")


class GaugeDetection(BaseModel):
    """Complete detection for a single gauge dial"""
    gauge_id: str = Field(..., description="Unique ID for gauge e.g. gauge_1")
    label: str = Field("Analog Gauge", description="Human readable label e.g. Dial Thermometer, Voltmeter, Ammeter")
    gauge_type: str = Field("generic", description="thermometer, voltmeter, ammeter, pressure_gauge, or generic")
    bbox: BoundingBox = Field(..., description="Bounding box around the entire gauge dial")
    dial_center: Point2D = Field(..., description="Center of the circular dial")
    needles: List[NeedleDetection] = Field(default_factory=list, description="Detected needle(s)")
    numbers: List[ScaleNumberDetection] = Field(default_factory=list, description="Detected numbers on dial")
    reading: GaugeReading = Field(..., description="Predicted gauge reading")
    crop_base64: Optional[str] = Field(None, description="Base64 encoded thumbnail of this gauge")


class DetectionResult(BaseModel):
    """Top-level response for image detection"""
    success: bool = True
    provider: str = Field("gemini", description="AI provider used: gemini, openai, litellm, mock")
    model: str = Field("", description="Model name used")
    image_width: int = Field(0, description="Original image width in pixels")
    image_height: int = Field(0, description="Original image height in pixels")
    gauge_count: int = Field(0, description="Number of gauges detected")
    gauges: List[GaugeDetection] = Field(default_factory=list, description="Detected gauges")
    execution_time_ms: float = Field(0.0, description="Processing time in milliseconds")
    prompt_tokens: Optional[int] = Field(None, description="Prompt tokens used")
    completion_tokens: Optional[int] = Field(None, description="Completion tokens used")
    total_tokens: Optional[int] = Field(None, description="Total tokens used (prompt + completion)")
    raw_explanation: Optional[str] = Field(None, description="Raw model explanation / CoT")
    error: Optional[str] = Field(None, description="Error message if failed")


class ConfigUpdateRequest(BaseModel):
    """Request to update runtime configuration"""
    provider: Optional[str] = None
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_model: Optional[str] = None
    litellm_api_key: Optional[str] = None
    litellm_base_url: Optional[str] = None
    litellm_model: Optional[str] = None


class SampleInfo(BaseModel):
    """Sample image metadata"""
    id: str
    filename: str
    title: str
    description: str
    expected_type: str
    thumbnail_url: str
    detection_data: Optional[DetectionResult] = Field(None, description="Precomputed detection result for instant display")


class ModelListRequest(BaseModel):
    """Request to fetch available models for a provider"""
    provider: str = Field(..., description="Provider name: gemini, openai, litellm")
    api_key: Optional[str] = Field(None, description="Optional API key override")
    base_url: Optional[str] = Field(None, description="Optional Base URL override")


class ModelListResponse(BaseModel):
    """Response containing available model IDs"""
    provider: str
    models: List[str]
    current_model: Optional[str] = None
    error: Optional[str] = None

