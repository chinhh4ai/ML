"""
LiteLLM / Local vLLM Prompt Definition for Industrial Gauge Detection & Reading
Inherits from BaseGaugePrompt and adds specialized optimizations for open-source Vision-Language
models (Llama-3.1-Nemotron-VL, Gemma-4, Qwen2-VL, etc.) across 12 industrial gauge categories.
Supports both Hybrid local-crop inspection and full-image meter inspection.
"""
from typing import Optional, Dict
from app.prompts.base_prompt import BaseGaugePrompt


class LiteLLMGaugePrompt(BaseGaugePrompt):
    """
    Prompt configuration specifically tailored for LiteLLM proxy and open-source models.
    Provides specialized inspection rules for 12 industrial gauge families and
    seamless compatibility with Hybrid local YOLO/YOLOX detector crops.
    """

    def __init__(self, model_name: str = "Llama-3.1-Nemotron-Nano-VL-8B-V1-FP4-QAD-A30.0.121"):
        super().__init__(model_name=model_name)

    @staticmethod
    def get_template_domain_rules(template_id: Optional[str] = None) -> str:
        """
        Return domain-specific inspection rules for each of the 12 industrial gauge templates.
        """
        rules_map: Dict[str, str] = {
            "dial_standard": (
                "GAUGE TYPE: STANDARD CIRCULAR PRESSURE / VACUUM DIAL GAUGE\n"
                "1. SCALE MARKINGS & UNIT:\n"
                "   - Physical unit: 'MPa', 'bar', 'psi', 'kgf/cm2', or 'kPa'.\n"
                "   - min_range: scale start (usually 0 or -0.1 for vacuum).\n"
                "   - max_range: scale end (e.g. 0.4, 0.6, 1.0, 1.6, 2.5, 10, 25, 100).\n"
                "2. POINTER TRACING:\n"
                "   - Trace the dark/black mechanical pointer from center pivot to perimeter pointing tip.\n"
                "   - Ignore the short counterweight tail behind the central pivot hub.\n"
                "   - Linearly interpolate the pointed position between nearest scale ticks."
            ),
            "dial_dual_pointer": (
                "GAUGE TYPE: DUAL-POINTER / SETPOINT LIMIT GAUGE\n"
                "1. DUAL-POINTER DISAMBIGUATION (CRITICAL):\n"
                "   - SOLID BLACK POINTER: Dynamic active live measured reading. This needle deflects according to operating load.\n"
                "   - RED / BRONZE / ORANGE POINTER: Adjustable limit marker or alarm setpoint threshold. Stationary during normal operation.\n"
                "   - CRITICAL: Read the live value strictly from the solid BLACK active pointer. NEVER report the red/bronze setpoint marker as the primary reading!\n"
                "2. SCALE & UNIT: Check unit symbol ('A', 'V', '°C'). Determine min_range and max_range."
            ),
            "dial_power_factor": (
                "GAUGE TYPE: POWER FACTOR METER (cos φ)\n"
                "1. BIDIRECTIONAL SYMMETRIC SCALE:\n"
                "   - The center of the scale arc corresponds to 1.0 (Unity power factor).\n"
                "   - 'LEAD' zone (Capacitive, C): ranges between 0.5 and 1.0 on one side.\n"
                "   - 'LAG' zone (Inductive, L): ranges between 1.0 and 0.5 on the other side.\n"
                "   - The physical unit is 'cos_phi' between 0.50 and 1.00.\n"
                "2. READING DEDUCTION: Determine if needle points into LEAD or LAG side, and read exact value (e.g. 0.88, 0.94, 0.98, 1.00)."
            ),
            "dial_control_panel": (
                "GAUGE TYPE: CONTROL PANEL AC METER WITH GLASS GLARE / WIRE-MESH\n"
                "1. REFLECTION & GLARE FILTERING:\n"
                "   - Ignore specular lighting glare streaks, glass reflections, and wire-mesh patterns.\n"
                "   - Identify real pointer needle: a continuous, sharp dark line pivoting at dial center.\n"
                "2. METER TYPE: 'A' = AC Ammeter (0-15A, 0-30A, 0-50A); 'V' = AC Voltmeter (0-150V, 0-300V, 0-600V).\n"
                "3. PROPORTIONAL READING: Interpolate needle tip between nearest scale divisions."
            ),
            "dial_extended_scale": (
                "GAUGE TYPE: EXTENDED OVERLOAD SCALE AMMETER (MOTOR STARTING INRUSH)\n"
                "1. NON-LINEAR COMPRESSED SCALE GRADUATIONS:\n"
                "   - Operating Zone (left half, e.g. 0 to 30A or 50A): Widely spaced linear ticks with fine divisions.\n"
                "   - Overload Zone (right half, e.g. 50 to 250A): Heavily compressed ticks to absorb motor starting inrush surge.\n"
                "2. LOCAL INTERPOLATION (CRITICAL):\n"
                "   - Identify the two printed numbers immediately flanking the needle tip.\n"
                "   - Count subdivision ticks between those two numbers. DO NOT assume uniform tick spacing across the whole arc!"
            ),
            "stem_thermometer": (
                "GAUGE TYPE: INDUSTRIAL STEM BIMETAL THERMOMETER\n"
                "1. DIAL LOCALIZATION & TEMPERATURE READING:\n"
                "   - Circular dial head mounted on machinery piping or engine thermowell.\n"
                "   - Physical unit is Celsius degrees ('°C'). Scale is typically 0-100°C, 0-120°C, or 0-150°C.\n"
                "2. NEEDLE OBSERVATION:\n"
                "   - Trace active dark needle from center pivot to perimeter scale tip.\n"
                "   - If a secondary bronze/red setpoint marker is present, ignore it and read ONLY the active dark needle.\n"
                "   - Compute indicated temperature."
            ),
            "digital_flowmeter_tokico": (
                "GAUGE TYPE: TOKICO MECHANICAL ROLLER DRUM FLOW METER\n"
                "1. ROLLER COUNTER WINDOW:\n"
                "   - Rectangular window with odometer-style roller digit drums (Liters, 'L', or 'm3').\n"
                "2. DIGIT READING (LEFT TO RIGHT):\n"
                "   - Read consecutive whole digits (white on black background).\n"
                "   - Rightmost drum (or red drum) represents tenths (0.1 L).\n"
                "   - Combine into full float reading (e.g. 130477.9 or 130478.2).\n"
                "3. AUXILIARY SUB-DIAL: Circular sub-dial (0.0-1.0 L) indicates fine fractional flow."
            ),
            "digital_7segment": (
                "GAUGE TYPE: 7-SEGMENT LED DIGITAL DISPLAY PANEL\n"
                "1. LED DIGIT RECOGNITION:\n"
                "   - Inspect illuminated bar segments to recognize digits 0 through 9.\n"
                "   - CRITICAL: Locate illuminated DECIMAL POINT (DP) dots to correctly format decimal fractions.\n"
                "2. CHANNELS: Top row with 'V' = Voltage (e.g. 56.0 V), bottom row with 'A' = Current (e.g. 0.00 A)."
            ),
            "digital_energy_hour": (
                "GAUGE TYPE: KILOWATT-HOUR (kWh) OR OPERATING RUNTIME (Hour) METER\n"
                "1. DISCRIMINATION: 'kWh' = Active Energy Meter; 'h' or 'hrs' = Operating Hour Meter.\n"
                "2. DIGITS: Read consecutive roller digits left-to-right. Rightmost red digit represents 0.1."
            ),
            "oil_level_sight_glass": (
                "GAUGE TYPE: GEARBOX OIL LEVEL SIGHT GLASS\n"
                "1. LOCATE SIGHT GLASS: Circular bullseye window or vertical sight tube on gearbox casing.\n"
                "2. OIL-AIR MENISCUS: Horizontal boundary line between dark amber lubricating oil and air.\n"
                "3. PERCENT & STATUS: 0% (empty) to 100% (full). Normal level is centered (40-60%). Status: 'NORMAL', 'LOW', or 'HIGH'."
            ),
            "gauge_board_multi": (
                "GAUGE TYPE: INDUSTRIAL MULTI-GAUGE PANEL BOARD\n"
                "1. Scan the image to detect every meter on the board.\n"
                "2. For each meter, classify family (pressure, thermometer, vacuum, voltage), locate pointer, and read indicated value."
            ),
            "default": (
                "GAUGE TYPE: INDUSTRIAL METER (AUTO-DETECT)\n"
                "1. Classify instrument family (circular dial, dual pointer, flowmeter, stem thermometer, 7-segment LED, or sight glass).\n"
                "2. Identify scale boundaries (min_range, max_range) and printed physical unit ('A', 'V', 'bar', 'MPa', 'psi', '°C', 'L', 'cos_phi', '%').\n"
                "3. Read primary active indicator tip or consecutive digit readout."
            ),
        }
        tid = (template_id or "default").lower().strip()
        return rules_map.get(tid, rules_map["default"])

    def get_inspection_protocol(self) -> str:
        return (
            "SYSTEMATIC 5-STEP DIAL INSPECTION PROTOCOL:\n"
            "1. UNIT & GAUGE TYPE IDENTIFICATION:\n"
            "   - Prominent unit symbols: 'A', 'V', 'bar', 'psi', 'MPa', 'kPa', 'kg/cm2', '°C', 'L', 'cos_phi'.\n"
            "2. SCALE BOUNDARIES & TICK MARKS:\n"
            "   - min_range: lowest scale number at arc start (usually 0).\n"
            "   - max_range: highest scale number at arc end.\n"
            "3. NEEDLE IDENTIFICATION & DUAL-POINTER DISAMBIGUATION:\n"
            "   - Primary active needle: solid dark/black pointer pivoting at center hub.\n"
            "   - CAUTION: If there is a secondary bronze/red setpoint marker, ALWAYS read the active dark needle!\n"
            "4. PROPORTIONAL INTERPOLATION:\n"
            "   - Interpolate needle tip position between flanking scale ticks to deduce exact reading value.\n"
            "5. OUTPUT FORMAT:\n"
            "   - Strict valid raw JSON starting with '{' and ending with '}'."
        )

    def build_system_prompt(self) -> str:
        base_prompt = super().build_system_prompt()
        return (
            f"{base_prompt}\n\n"
            f"{self.get_inspection_protocol()}\n\n"
            "CRITICAL OUTPUT DIRECTIVES FOR OPEN-SOURCE VLM:\n"
            "- Output valid raw JSON ONLY. Do NOT include markdown code fences (no ```json or ```).\n"
            "- Do NOT output greetings, conversational text, or preambles outside the JSON.\n"
            "- All numbers must be numeric types (e.g. 0.0, 10, 25.5), not stringified numbers."
        )

    def build_user_prompt(self, filename: Optional[str] = None, template_id: Optional[str] = None) -> str:
        hint = f" [Image: {filename}]" if filename else ""
        domain_rules = self.get_template_domain_rules(template_id)
        return (
            f"Carefully analyze this industrial gauge meter image.{hint}\n"
            f"{domain_rules}\n\n"
            "REQUIREMENTS:\n"
            "- Dial/Needle gauges: pointers in `needles`, detected scale numbers in `numbers`, and derivation in `reading.reasoning` are ALL MANDATORY.\n"
            "- Digital / Roller / Sight glass: pointers not required (`needles: []`), but detected digits/marks in `numbers` and derivation in `reading.reasoning` are MANDATORY.\n\n"
            "Perform precision visual inspection and output strictly valid raw JSON:\n"
            "{\n"
            '  "gauges": [\n'
            "    {\n"
            '      "gauge_id": "gauge_1",\n'
            '      "label": "string",\n'
            '      "gauge_type": "string",\n'
            '      "bbox": {"ymin": int, "xmin": int, "ymax": int, "xmax": int},\n'
            '      "dial_center": {"x": int, "y": int},\n'
            '      "needles": [\n'
            "        {\n"
            '          "needle_id": "needle_1",\n'
            '          "type": "primary_pointer",\n'
            '          "bbox": {"ymin": int, "xmin": int, "ymax": int, "xmax": int},\n'
            '          "pivot_point": {"x": int, "y": int},\n'
            '          "tip_point": {"x": int, "y": int},\n'
            '          "angle_degrees": float,\n'
            '          "color": "black"\n'
            "        }\n"
            "      ],\n"
            '      "numbers": [\n'
            '        {"value": "string", "numeric_value": float, "bbox": {"ymin": int, "xmin": int, "ymax": int, "xmax": int}, "position": {"x": int, "y": int}}\n'
            "      ],\n"
            '      "reading": {\n'
            '        "value": float,\n'
            '        "unit": "string",\n'
            '        "formatted": "string",\n'
            '        "min_range": float,\n'
            '        "max_range": float,\n'
            '        "confidence": float,\n'
            '        "reasoning": "Detailed derivation of reading from needle angle or displayed digits"\n'
            "      }\n"
            "    }\n"
            "  ]\n"
            "}"
        )

    def build_cropped_dial_prompt(
        self,
        gauge_idx: int = 1,
        angle_degrees: Optional[float] = None,
        template_id: Optional[str] = None,
        custom_instructions: Optional[str] = None,
    ) -> str:
        """
        Focused prompt for a single high-resolution cropped gauge dial in Hybrid mode.
        Combines specialized domain rules for the gauge family, computer vision needle angle assistance,
        and a clean, unambiguous JSON output structure.
        """
        if custom_instructions and len(custom_instructions.strip()) > 30:
            domain_section = f"INSPECTION PROTOCOL:\n{custom_instructions.strip()}"
        else:
            domain_section = self.get_template_domain_rules(template_id)

        geom_hint = ""
        if angle_degrees is not None:
            clock_hour = (angle_degrees / 30.0) % 12.0
            if clock_hour < 0.7 or clock_hour >= 11.3:
                direction_desc = "pointing UPWARDS towards 12 o'clock"
            elif 2.3 <= clock_hour <= 3.7:
                direction_desc = "pointing RIGHT towards 3 o'clock"
            elif 5.3 <= clock_hour <= 6.7:
                direction_desc = "pointing DOWNWARDS towards 6 o'clock"
            elif 8.3 <= clock_hour <= 9.7:
                direction_desc = "pointing LEFT towards 9 o'clock"
            elif 0.7 <= clock_hour < 2.3:
                direction_desc = "pointing to upper-right (1-2 o'clock)"
            elif 3.7 < clock_hour < 5.3:
                direction_desc = "pointing to lower-right (4-5 o'clock)"
            elif 6.7 < clock_hour < 8.3:
                direction_desc = "pointing to lower-left (7-8 o'clock)"
            else:
                direction_desc = "pointing to upper-left (10-11 o'clock)"

            geom_hint = (
                f"- Computer Vision pointer analysis detected the primary needle at deflection angle {angle_degrees:.1f}° ({direction_desc}).\n"
                f"- Carefully verify which scale tick or printed number aligns with this angle direction.\n"
            )

        return (
            f"You are inspecting a cropped close-up of Gauge #{gauge_idx}.\n\n"
            f"{domain_section}\n\n"
            "PRECISION VISUAL ANALYSIS INSTRUCTIONS:\n"
            "1. Identify the printed unit symbol on the dial face ('°C', 'bar', 'psi', 'A', 'V', 'kPa', 'MPa', 'L', 'cos_phi').\n"
            "2. Identify scale boundaries: lowest printed number (min_range) and highest printed number (max_range).\n"
            "3. Pointer observation:\n"
            f"{geom_hint}"
            "   - Follow the primary active dark needle from center pivot to perimeter pointing tip.\n"
            "   - CAUTION FOR DUAL-POINTERS / THERMOMETERS: If a stationary bronze or red setpoint marker is present, ignore it! ALWAYS read the active dark pointer.\n"
            "4. Calculate exact indicated reading value by proportional interpolation between the nearest flanking ticks.\n"
            "5. Detect scale numbers visible on the dial face.\n\n"
            "Output strictly valid raw JSON:\n"
            "{\n"
            '  "gauge_type": "pressure_gauge | thermometer | ammeter | voltmeter | power_factor_meter | flowmeter",\n'
            '  "reading_value": float,\n'
            '  "unit": "string",\n'
            '  "min_range": float,\n'
            '  "max_range": float,\n'
            '  "confidence": float (0.0 to 1.0),\n'
            '  "scale_numbers": [float],\n'
            '  "reasoning": "Clear explanation of needle tip alignment with ticks and reading derivation"\n'
            "}"
        )

    def build_full_image_prompt(
        self,
        template_id: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Prompt for full image inspection (e.g. digital flowmeters, 7-segment displays, multi-gauge boards, sight glasses).
        """
        hint = f" [Image: {filename}]" if filename else ""
        if custom_instructions and len(custom_instructions.strip()) > 30:
            domain_section = custom_instructions.strip()
        else:
            domain_section = self.get_template_domain_rules(template_id)

        return (
            f"Inspect this industrial image carefully.{hint}\n"
            f"{domain_section}\n\n"
            "Locate each gauge, meter, digital counter, or sight glass device present.\n"
            "For each device detected:\n"
            "1. Bounding box [ymin, xmin, ymax, xmax] in normalized coordinates [0, 1000].\n"
            "2. Gauge type ('flowmeter', 'pressure_gauge', 'thermometer', 'ammeter', 'voltmeter', 'seven_segment', 'sight_glass').\n"
            "3. Exact reading value and unit.\n"
            "4. Dial pointers in `needles`: MANDATORY for dial/pointer meters; empty list `[]` for digital/drum/sight glass.\n"
            "5. Scale numbers or digits in `numbers`: MANDATORY for ALL devices.\n"
            "6. Derivation explanation in `reading.reasoning`: MANDATORY for ALL devices.\n\n"
            "Output strictly valid raw JSON:\n"
            "{\n"
            '  "gauge_count": int,\n'
            '  "gauges": [\n'
            "    {\n"
            '      "gauge_index": 1,\n'
            '      "gauge_type": "string",\n'
            '      "label": "string",\n'
            '      "bbox": [ymin, xmin, ymax, xmax],\n'
            '      "dial_center": [500, 500],\n'
            '      "needles": [],\n'
            '      "numbers": [\n'
            '        {"value": "string", "numeric_value": float, "bbox": [ymin, xmin, ymax, xmax], "position": [x, y]}\n'
            '      ],\n'
            '      "reading": {\n'
            '        "value": float,\n'
            '        "unit": "string",\n'
            '        "min_range": float,\n'
            '        "max_range": float,\n'
            '        "confidence": float,\n'
            '        "reasoning": "Step-by-step derivation reasoning"\n'
            '      }\n'
            "    }\n"
            "  ]\n"
            "}"
        )
