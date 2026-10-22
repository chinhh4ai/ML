"""
Base Prompt Architecture for Analog Gauge Detection & Reading AI
Provides base system instructions, normalized coordinate specifications,
common JSON schemas, and methods that provider-specific prompts inherit from.
"""
from typing import Dict, Any, Optional


class BaseGaugePrompt:
    """
    Base prompt definition for analog gauge detection.
    Subclasses customize and extend for specific model providers (Gemini, OpenAI, LiteLLM).
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or "default"

    def get_role_description(self) -> str:
        return (
            "You are an expert industrial computer vision AI system specializing in "
            "analog gauge detection, dial segmentation, needle/pointer kinematic tracking, "
            "dial scale tick & number reading, and physical measurement prediction."
        )

    def get_coordinate_rules(self) -> str:
        return (
            "COORDINATE SYSTEM RULES:\n"
            "1. All bounding boxes (`bbox`) and points (`dial_center`, `pivot_point`, `tip_point`, `position`) "
            "MUST be normalized to the range [0, 1000] where:\n"
            "   - (0, 0) is the top-left corner of the image.\n"
            "   - (1000, 1000) is the bottom-right corner of the image.\n"
            "2. Bounding boxes use format: {\"ymin\": int, \"xmin\": int, \"ymax\": int, \"xmax\": int}.\n"
            "3. Points use format: {\"x\": int, \"y\": int}."
        )

    def get_detection_tasks(self) -> str:
        return (
            "DETECTION & REASONING TASKS:\n"
            "1. Locate each analog gauge meter frame (`bbox`).\n"
            "2. Determine dial center pivot point (`dial_center` [x, y]).\n"
            "3. Identify all needles/pointers (`needles`):\n"
            "   - `needle_id`: e.g. 'needle_1'\n"
            "   - `type`: 'primary_pointer' or 'secondary_pointer' or 'setpoint'\n"
            "   - `bbox`: bounding box of needle\n"
            "   - `pivot_point`: rotation base [x, y]\n"
            "   - `tip_point`: pointing tip [x, y]\n"
            "   - `angle_degrees`: rotation angle in degrees (0 to 360 clockwise where 0 is 12 o'clock, or standard polar)\n"
            "   - `color`: pointer color e.g. 'black', 'red', 'bronze'\n"
            "4. Recognize visible scale numbers printed on dial (`numbers`):\n"
            "   - `value`: text string e.g. '0', '10', '20', '100'\n"
            "   - `numeric_value`: float or null\n"
            "   - `bbox`: bounding box around number\n"
            "   - `position`: center [x, y] of number\n"
            "5. Predict exact pointer reading (`reading`):\n"
            "   - `value`: predicted reading number that needle points to\n"
            "   - `unit`: unit of measurement (e.g. '°C', 'V', 'A', 'bar', 'psi', 'MPa', 'rpm')\n"
            "   - `formatted`: e.g. '20 °C', '0 A', '420 V'\n"
            "   - `min_range`: dial scale minimum value\n"
            "   - `max_range`: dial scale maximum value\n"
            "   - `confidence`: confidence score between 0.0 and 1.0\n"
            "   - `reasoning`: concise explanation of how the needle tip aligns with neighboring tick marks and numbers."
        )

    def get_json_example(self) -> str:
        return """{
  "gauges": [
    {
      "gauge_id": "gauge_1",
      "label": "Analog Gauge Meter",
      "gauge_type": "thermometer",
      "bbox": {"ymin": 160, "xmin": 250, "ymax": 830, "xmax": 840},
      "dial_center": {"x": 545, "y": 495},
      "needles": [
        {
          "needle_id": "needle_1",
          "type": "primary_pointer",
          "bbox": {"ymin": 465, "xmin": 375, "ymax": 525, "xmax": 550},
          "pivot_point": {"x": 545, "y": 495},
          "tip_point": {"x": 375, "y": 495},
          "angle_degrees": 270.0,
          "color": "black"
        }
      ],
      "numbers": [
        {
          "value": "20",
          "numeric_value": 20.0,
          "bbox": {"ymin": 460, "xmin": 405, "ymax": 520, "xmax": 440},
          "position": {"x": 422, "y": 490}
        }
      ],
      "reading": {
        "value": 20.0,
        "unit": "°C",
        "formatted": "20 °C",
        "min_range": 0.0,
        "max_range": 100.0,
        "confidence": 0.95,
        "reasoning": "Primary black pointer points directly at the 20 mark."
      }
    }
  ],
  "explanation": "Detected 1 analog gauge."
}"""

    def build_system_prompt(self) -> str:
        """Construct full system prompt."""
        return (
            f"{self.get_role_description()}\n\n"
            f"{self.get_coordinate_rules()}\n\n"
            f"{self.get_detection_tasks()}\n\n"
            f"REQUIRED OUTPUT FORMAT:\n"
            f"You must return strictly valid JSON matching this structure:\n"
            f"{self.get_json_example()}"
        )

    def build_user_prompt(self, filename: Optional[str] = None) -> str:
        """Construct user prompt accompanying the image."""
        hint = f" Image filename: {filename}." if filename else ""
        return (
            f"Detect all analog gauge meters, pointer needles, and scale numbers in this image.{hint} "
            f"Predict the precise reading that each needle points to and output strictly valid JSON."
        )
