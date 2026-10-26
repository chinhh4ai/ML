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
            "DETECTION & REASONING RULES BY GAUGE CATEGORY:\n\n"
            "A. FOR ANALOG / DIAL GAUGES (WITH NEEDLES / POINTERS):\n"
            "   1. Frame & Dial: Detect `bbox` [ymin, xmin, ymax, xmax] and `dial_center` [x, y].\n"
            "   2. MANDATORY POINTERS (`needles`):\n"
            "      - Identify EVERY pointer needle: `needle_id`, `type` ('primary_pointer', 'secondary_pointer', 'setpoint'),\n"
            "        `pivot_point` [x, y], `tip_point` [x, y], `angle_degrees` (0-360° clockwise from 12 o'clock), `color`.\n"
            "      - Distinguish active operating pointer (usually black) from stationary limit/setpoint marker (red/bronze).\n"
            "   3. MANDATORY SCALE NUMBERS (`numbers`):\n"
            "      - Detect ALL printed numbers along the scale arc.\n"
            "      - For each number: `value` (e.g. '0', '10', '20', '400'), `numeric_value`, `bbox`, `position`.\n"
            "   4. MANDATORY AI REASONING & DERIVATION (`reading.reasoning`):\n"
            "      - Explain how the pointer tip aligns with the dial numbers and tick marks to derive `reading.value`.\n\n"
            "B. FOR NON-NEEDLE GAUGES (DIGITAL 7-SEGMENT, ROLLER DRUM COUNTERS, SIGHT GLASSES):\n"
            "   1. `needles`: Set to empty list `[]` (pointers are NOT required for non-needle meters).\n"
            "   2. MANDATORY DETECTED SCALE NUMBERS (`numbers`):\n"
            "      - For 7-segment LED / LCD: extract each displayed digit or full number (e.g. '5', '6', '.', '0' or '56.0').\n"
            "      - For Tokico / mechanical roller drums: extract each visible roller digit wheel and sub-dial numbers.\n"
            "      - For Dual-Counter Hour Meters (e.g. Mitsubishi HS-72B with 'TOTAL TIME' and 'TIME'):\n"
            "        * Identify BOTH counter rows: Upper row is non-resettable cumulative 'TOTAL TIME' (e.g. 267.6 h), Lower row is resettable 'TIME' (e.g. 257.9 h) with bottom reset knob.\n"
            "        * Detect both rows as two distinct gauge entries in `gauges` (or report both in `reading.formatted` as 'TOTAL: 267.6 h | TRIP: 257.9 h') and include all digits of BOTH rows in `numbers`.\n"
            "      - For oil level sight glass: extract reference marks ('MIN', 'MAX', 0, 50, 100%).\n"
            "      - For each number: provide `value`, `numeric_value`, `bbox`, `position`.\n"
            "   3. MANDATORY AI REASONING & DERIVATION (`reading.reasoning`):\n"
            "      - Explain step-by-step how the value was recognized from the LED segments/decimal points, roller wheels (including both TOTAL and TRIP rows on dual-counter hour meters), or oil meniscus boundary.\n\n"
            "5. READING PREDICTION (`reading` - REQUIRED FOR ALL GAUGES):\n"
            "   - `value`: predicted numeric reading.\n"
            "   - `unit`: unit of measurement ('°C', 'V', 'A', 'bar', 'psi', 'MPa', 'L', 'cos_phi', '%').\n"
            "   - `formatted`: formatted text (e.g. '20 °C', '420 V', '56.0 V', '130477.9 L', '50.0 %').\n"
            "   - `min_range`: lowest scale value or 0.0.\n"
            "   - `max_range`: highest scale value.\n"
            "   - `confidence`: confidence score between 0.0 and 1.0.\n"
            "   - `reasoning`: detailed derivation explanation (MANDATORY FOR ALL GAUGES)."
        )

    def get_json_example(self) -> str:
        return """{
  "gauges": [
    {
      "gauge_id": "gauge_1",
      "label": "Dial Thermometer",
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
          "value": "0",
          "numeric_value": 0.0,
          "bbox": {"ymin": 630, "xmin": 460, "ymax": 690, "xmax": 495},
          "position": {"x": 477, "y": 660}
        },
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
        "confidence": 0.98,
        "reasoning": "Primary black pointer pivots at (545, 495) and points horizontally (270.0°) directly aligned with the 20 °C scale number mark between 0 and 100 °C."
      }
    }
  ],
  "explanation": "Detected 1 dial gauge with needle kinematics, scale numbers, and derivation reasoning."
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
            f"Detect all meters, dial needles/pointers, and scale numbers in this image.{hint}\n"
            f"- For needle/pointer dial gauges: pointers in `needles`, detected scale numbers in `numbers`, and derivation reasoning in `reading.reasoning` are ALL MANDATORY.\n"
            f"- For other meters (digital, roller drum, sight glass): pointers are not required (`needles: []`), but detected numbers/digits in `numbers` and derivation reasoning in `reading.reasoning` are MANDATORY.\n"
            f"Output strictly valid raw JSON."
        )
