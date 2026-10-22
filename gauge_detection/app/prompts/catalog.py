"""
Gauge Prompt Template Catalog & Registry (English Edition)
Loads and serves prompt templates customized for real-world industrial gauges.
Based on sample data from /home/chinhh/workspace/HIP/Gauge_Photos
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

TEMPLATES_DIR = Path(__file__).parent / "templates"


class GaugeTemplate(BaseModel):
    id: str
    name: str
    category: str
    description: str
    sample_reference: str
    shape: str
    scale_info: str
    indicator_type: str
    prompt: str


DEFAULT_TEMPLATES: List[Dict[str, Any]] = [
    {
        "id": "default",
        "name": "Default (Auto-Detect)",
        "category": "Auto-Detect",
        "description": "Automatically analyzes the full image to classify the instrument family (circular dial, dual-pointer, Tokico flowmeter, stem thermometer, 7-segment LED, energy/hour meter, oil sight glass, or multi-gauge array) and executes the optimal inspection protocol.",
        "sample_reference": "All real-world gauge categories in Gauge_Photos",
        "shape": "Diverse (Circular, Square Bezel, Rectangular Display, Sight Tube)",
        "scale_info": "Auto-extracted from dial ticks and face markings",
        "indicator_type": "Pointer Needle / Mechanical Drum / 7-Segment LED / Liquid Meniscus",
        "prompt": (
            "You are an expert industrial computer vision inspector specializing in analog dial and digital meters.\n"
            "SYSTEMATIC 3-STAGE INSPECTION PROTOCOL:\n"
            "1. INSTRUMENT FAMILY CLASSIFICATION:\n"
            "   - Circular Dial Gauge: Round dial face, central needle pivot, circular arc scale (bar, MPa, psi, V, A, °C).\n"
            "   - Dual/Multi-Pointer: Dial with active black pointer + secondary red/bronze limit or setpoint marker.\n"
            "   - Power Factor Meter (cos φ): Symmetric bidirectional scale centered at 1.0 with LEAD and LAG zones.\n"
            "   - Tokico Mechanical Flowmeter: Rectangular window with odometer roller digit wheels + auxiliary needle dial.\n"
            "   - 7-Segment LED Display: Glowing red/green numeric segments on dark background with decimal points.\n"
            "   - Stem Thermometer: Dial mounted on stainless steel immersion stem/thermowell on engine pipes (°C).\n"
            "   - Oil Level Sight Glass: Cylindrical glass tube or bullseye window indicating oil meniscus level (MIN/MAX).\n"
            "   - Multi-Gauge Board: Multiple meters mounted simultaneously on an industrial control panel.\n\n"
            "2. DETAILED MORPHOLOGY & READING DEDUCTION:\n"
            "   - Identify min_range, max_range, and unit printed on the face ('A', 'V', 'bar', 'MPa', 'psi', '°C', 'L', 'm3', 'cos_phi', '%').\n"
            "   - Locate active indicator tip or consecutive digits.\n"
            "   - For dual-pointer meters: Read ONLY the active black needle for current operating value. Do NOT confuse with stationary red setpoint needle.\n\n"
            "3. STRICT RAW JSON OUTPUT:\n"
            "Start with '{' and end with '}'. Do NOT wrap in markdown fences:\n"
            "{\n"
            '  "gauges": [\n'
            "    {\n"
            '      "gauge_index": 1,\n'
            '      "gauge_type": "<pressure_gauge | ammeter | voltmeter | thermometer | flowmeter | seven_segment | sight_glass>",\n'
            '      "bbox": [ymin, xmin, ymax, xmax],\n'
            '      "reading": {\n'
            '        "value": <float>,\n'
            '        "unit": "<unit>",\n'
            '        "confidence": <0.0-1.0>\n'
            "      },\n"
            '      "scale": {\n'
            '        "min_val": <float>,\n'
            '        "max_val": <float>,\n'
            '        "unit": "<unit>"\n'
            "      },\n"
            '      "needle": {\n'
            '        "angle_deg": <float 0-360>,\n'
            '        "tip": [y, x],\n'
            '        "center": [y, x]\n'
            "      }\n"
            "    }\n"
            "  ]\n"
            "}"
        )
    },
    {
        "id": "dial_standard",
        "name": "Standard Pressure Dial Gauge",
        "category": "Dial Gauges",
        "description": "Standard circular pressure or vacuum gauge mounted on compressors, pumps, or pressure vessels. Common engineering ranges: 0-1.0 MPa, 0-1.6 MPa, 0-10 bar, 0-100 psi.",
        "sample_reference": "1_Dial Gauges/1_Standard Cases/1-1_Air Compressor & 3_Mounted Gauge_Vibrating Pointer",
        "shape": "Circular metal/plastic bezel, 60-150mm diameter",
        "scale_info": "270-degree circular arc, min typically 0 (or -0.1 for vacuum), max 0.4, 1.0, 1.6, 2.5, 10, 25, 100",
        "indicator_type": "Single mechanical black pointer pivoting at dial center",
        "prompt": (
            "You are inspecting a STANDARD CIRCULAR PRESSURE DIAL GAUGE.\n"
            "VISUAL INSPECTION PROTOCOL:\n"
            "1. CASING & DIAL FACE:\n"
            "   - Circular face with scale markings distributed around ~270 degree perimeter arc.\n"
            "   - Locate central needle pivot at dial center.\n"
            "2. UNIT & SCALE BOUNDARIES (MIN - MAX):\n"
            "   - Identify unit symbol: 'MPa', 'bar', 'kgf/cm2', 'psi', or 'kPa'.\n"
            "   - min_range: scale start point (usually 0 or needle stop pin).\n"
            "   - max_range: scale end point (e.g. 0.4, 0.6, 1.0, 1.6, 2.5, 10, 25, 100).\n"
            "   - Count major division increments (step of 0.1, 0.2, 1, 2, or 5).\n"
            "3. NEEDLE POINTER IDENTIFICATION:\n"
            "   - Follow the black pointer from central pivot to pointing tip at scale perimeter.\n"
            "   - Ignore the short counterweight tail behind pivot.\n"
            "   - If needle has vibration motion blur, trace the centerline of the pointer body.\n"
            "   - Linearly interpolate the pointed position between nearest tick marks.\n"
            "4. OUTPUT FORMAT (RAW JSON ONLY):\n"
            "{\n"
            '  "gauges": [{\n'
            '    "gauge_index": 1,\n'
            '    "gauge_type": "pressure_gauge",\n'
            '    "bbox": [ymin, xmin, ymax, xmax],\n'
            '    "reading": {"value": <float>, "unit": "<MPa|bar|psi|kgf/cm2>", "confidence": 0.95},\n'
            '    "scale": {"min_val": <float>, "max_val": <float>, "unit": "<unit>"},\n'
            '    "needle": {"angle_deg": <float>, "tip": [y, x], "center": [y, x]}\n'
            "  }]\n"
            "}"
        )
    },
    {
        "id": "dial_dual_pointer",
        "name": "Dual-Pointer / Setpoint Limit Gauge",
        "category": "Dial Gauges",
        "description": "Control panel dial gauge featuring two needles: a dynamic black needle indicating active operating measurement, and a stationary red/bronze needle indicating alarm setpoint or trip threshold.",
        "sample_reference": "1_Dial Gauges/2_Gauges with Multiple Pointers (Motor Control Center, Ventilation Fan, Pump Panel)",
        "shape": "Square bezel (72x72mm, 96x96mm) or round panel meter",
        "scale_info": "Voltage 0-150V, 0-300V, 0-600V or Current 0-15A, 0-30A, 0-50A",
        "indicator_type": "Two co-axial needles: Black (live reading) and Red/Bronze (alarm setpoint)",
        "prompt": (
            "You are inspecting a DUAL-POINTER / MULTI-NEEDLE GAUGE.\n"
            "CRITICAL DUAL-POINTER RULES:\n"
            "1. NEEDLE DISAMBIGUATION:\n"
            "   - SOLID BLACK POINTER: Dynamic active live measured reading. This needle deflects according to operating load.\n"
            "   - RED / BRONZE / ORANGE POINTER: Adjustable limit marker or alarm setpoint threshold. Stationary during normal operation.\n"
            "   - DO NOT report the red setpoint pointer as the primary operating value!\n"
            "2. SCALE BOUNDARIES & UNIT:\n"
            "   - Identify unit: 'A' (Ammeter), 'V' (Voltmeter), or '°C' (Thermometer).\n"
            "   - min_range: scale start point (usually 0).\n"
            "   - max_range: scale upper boundary (e.g. 15, 30, 50, 150, 300, 600).\n"
            "3. READING DEDUCTION:\n"
            "   - Primary reading 'value' MUST be derived strictly from the BLACK active pointer tip.\n"
            "   - Secondary setpoint reading 'setpoint_value' derived from the RED limit marker.\n"
            "4. OUTPUT FORMAT (RAW JSON ONLY):\n"
            "{\n"
            '  "gauges": [{\n'
            '    "gauge_index": 1,\n'
            '    "gauge_type": "dual_pointer_gauge",\n'
            '    "bbox": [ymin, xmin, ymax, xmax],\n'
            '    "reading": {"value": <float black needle>, "unit": "<unit>", "confidence": 0.95},\n'
            '    "setpoint_reading": {"value": <float red needle>, "unit": "<unit>"},\n'
            '    "scale": {"min_val": <float>, "max_val": <float>, "unit": "<unit>"},\n'
            '    "needle": {"angle_deg": <float>, "tip": [y, x], "center": [y, x]}\n'
            "  }]\n"
            "}"
        )
    },
    {
        "id": "dial_power_factor",
        "name": "Power Factor Meter (cos φ)",
        "category": "Dial Gauges",
        "description": "Switchboard cosine phi power factor meter. Features a symmetric bidirectional arc centered at 1.0 (Unity), with capacitive LEAD zone on one side and inductive LAG zone on the other side.",
        "sample_reference": "1_Dial Gauges/1_Standard Cases/1-3_Power Factor Meter (Service Entrance & Generator Panel)",
        "shape": "Square switchboard bezel 96x96mm or 110x110mm",
        "scale_info": "Center is 1.0 (Unity). LEAD zone: 0.5 - 0.7 - 0.9 - 1.0. LAG zone: 1.0 - 0.9 - 0.7 - 0.5",
        "indicator_type": "Center-pivot black mechanical needle with bidirectional arc",
        "prompt": (
            "You are inspecting a POWER FACTOR METER (cos phi meter).\n"
            "TECHNICAL SPECIFICATION & READING RULES:\n"
            "1. BIDIRECTIONAL SYMMETRIC SCALE:\n"
            "   - The center of the scale arc corresponds to 1.0 (Unity power factor).\n"
            "   - 'LEAD' zone (Capacitive, C): ranges between 0.5 and 1.0.\n"
            "   - 'LAG' zone (Inductive, L): ranges between 1.0 and 0.5.\n"
            "   - The physical unit is dimensionless 'cos_phi' between 0.50 and 1.00.\n"
            "2. NEEDLE OBSERVATION:\n"
            "   - Determine whether needle tip points into the LEAD side or LAG side.\n"
            "   - Read the exact numerical cos phi value (e.g. 0.88, 0.94, 0.98, 1.00).\n"
            "   - Filter out glass reflections from ambient fluorescent room lighting.\n"
            "3. OUTPUT FORMAT (RAW JSON ONLY):\n"
            "{\n"
            '  "gauges": [{\n'
            '    "gauge_index": 1,\n'
            '    "gauge_type": "power_factor_meter",\n'
            '    "bbox": [ymin, xmin, ymax, xmax],\n'
            '    "reading": {"value": <float 0.5-1.0>, "unit": "cos_phi", "mode": "<LEAD|LAG|UNITY>", "confidence": 0.95},\n'
            '    "scale": {"min_val": 0.5, "center_val": 1.0, "max_val": 0.5, "unit": "cos_phi"},\n'
            '    "needle": {"angle_deg": <float>, "tip": [y, x], "center": [y, x]}\n'
            "  }]\n"
            "}"
        )
    },
    {
        "id": "dial_control_panel",
        "name": "Control Panel AC Meter with Reflection",
        "category": "Dial Gauges",
        "description": "Panel-mounted AC Ammeter or Voltmeter inside electrical switchgear cabinets, viewed through protective cabinet door glass with glare, lighting reflections, or safety wire-mesh.",
        "sample_reference": "1_Dial Gauges/1_Standard Cases/1-2_Control Panel_with Reflections & Wire-Mesh",
        "shape": "Square panel meter 72x72mm or 96x96mm",
        "scale_info": "AC Ammeter: 0-15A, 0-30A, 0-50A; AC Voltmeter: 0-150V, 0-300V, 0-600V",
        "indicator_type": "Slender black needle moving across white dial background",
        "prompt": (
            "You are inspecting a CONTROL PANEL METER WITH GLASS GLARE / REFLECTIONS.\n"
            "GLARE SUPPRESSION & NEEDLE LOCALIZATION:\n"
            "1. REFLECTION FILTERING:\n"
            "   - Cabinet door glass may exhibit specular glare from overhead lights, photographer reflection, or wire-mesh grid.\n"
            "   - Real pointer needle is a continuous, sharp mechanical line casting a subtle shadow on the dial face.\n"
            "   - Ignore light glare streaks and wire-mesh lines.\n"
            "2. METER TYPE & SCALE:\n"
            "   - Check capital letters: 'A' = AC Ammeter, 'V' = AC Voltmeter.\n"
            "   - min_range: scale start point (usually 0).\n"
            "   - max_range: rightmost scale boundary (e.g. 15, 30, 50, 150, 300, 600).\n"
            "3. PROPORTIONAL INTERPOLATION:\n"
            "   - Trace needle axis from pivot to scale perimeter and read indicated value.\n"
            "4. OUTPUT FORMAT (RAW JSON ONLY):\n"
            "{\n"
            '  "gauges": [{\n'
            '    "gauge_index": 1,\n'
            '    "gauge_type": "<ammeter|voltmeter>",\n'
            '    "bbox": [ymin, xmin, ymax, xmax],\n'
            '    "reading": {"value": <float>, "unit": "<A|V>", "confidence": 0.95},\n'
            '    "scale": {"min_val": <float>, "max_val": <float>, "unit": "<A|V>"},\n'
            '    "needle": {"angle_deg": <float>, "tip": [y, x], "center": [y, x]}\n'
            "  }]\n"
            "}"
        )
    },
    {
        "id": "dial_extended_scale",
        "name": "Extended Overload Scale Ammeter",
        "category": "Dial Gauges",
        "description": "AC Ammeter designed for motor starting inrush currents (Cooling Water Pump Panel, Generator). Features non-linear scale divisions: linear low-end for normal operating current and compressed high-end (2x-5x) for overload.",
        "sample_reference": "1_Dial Gauges/4_Extended Scale and Others/Cooling Water Pump Panel",
        "shape": "Square switchgear bezel 96x96mm",
        "scale_info": "Example: 0 to 50A expanded linear scale, 50 to 250A compressed overload scale",
        "indicator_type": "Fine pointer needle with non-uniform scale graduations",
        "prompt": (
            "You are inspecting an EXTENDED / EXPANDED NON-LINEAR SCALE AMMETER.\n"
            "NON-UNIFORM SCALE CALCULATION RULES:\n"
            "1. SCALE COMPRESSION IDENTIFICATION:\n"
            "   - Tick spacing is NOT uniform across the dial arc!\n"
            "   - Operating Zone (left half): Ticks are widely spaced with fine subdivisions.\n"
            "   - Overload Inrush Zone (right half): Ticks are heavily compressed to absorb motor startup surge.\n"
            "2. LOCAL INTERPOLATION (CRITICAL):\n"
            "   - Identify the TWO nearest printed numbers flanking the needle tip.\n"
            "   - Count the subdivision ticks between those two numbers to determine local tick value.\n"
            "   - DO NOT use linear interpolation across the whole dial; interpolate strictly within the local sector.\n"
            "3. OUTPUT FORMAT (RAW JSON ONLY):\n"
            "{\n"
            '  "gauges": [{\n'
            '    "gauge_index": 1,\n'
            '    "gauge_type": "ammeter_extended_scale",\n'
            '    "bbox": [ymin, xmin, ymax, xmax],\n'
            '    "reading": {"value": <float>, "unit": "A", "confidence": 0.95},\n'
            '    "scale": {"min_val": <float>, "rated_val": <float>, "max_val": <float>, "unit": "A"},\n'
            '    "needle": {"angle_deg": <float>, "tip": [y, x], "center": [y, x]}\n'
            "  }]\n"
            "}"
        )
    },
    {
        "id": "stem_thermometer",
        "name": "Industrial Stem Bimetal Thermometer",
        "category": "Stem Thermometers",
        "description": "Industrial bimetallic stem dial thermometer mounted on engine cooling water pipes, lube oil lines, or generator bearing wells. Temperature scale in °C (0-100°C, 0-120°C, or 0-150°C).",
        "sample_reference": "4_Stem Thermometers/1_Pump Engine, 2_Generator & 3_Room Thermometer",
        "shape": "Circular dial head atop stainless steel stem probe and hex mounting fitting",
        "scale_info": "0 to 100 °C, 0 to 120 °C, or 0 to 150 °C (sometimes dual °C / °F)",
        "indicator_type": "Center-pivot mechanical pointer with teardrop tail",
        "prompt": (
            "You are inspecting an INDUSTRIAL STEM BIMETAL THERMOMETER.\n"
            "LOCALIZATION & TEMPERATURE READING RULES:\n"
            "1. DIAL FACE LOCALIZATION ON PIPING:\n"
            "   - Locate the circular dial head mounted on machinery piping or engine casing.\n"
            "   - Ignore the background pipe, flange bolts, and mounting hex fitting.\n"
            "2. TEMPERATURE SCALE & UNIT:\n"
            "   - Physical unit is Celsius degrees ('°C' or 'deg C'). Outer secondary scale may be '°F'.\n"
            "   - min_range: typically 0°C (or -20°C).\n"
            "   - max_range: typically 100°C, 120°C, or 150°C.\n"
            "3. NEEDLE POINTER DEFLECTION:\n"
            "   - Follow the pointer from center pivot to perimeter scale tip.\n"
            "   - Read indicated temperature (e.g. 45°C, 62°C, 78°C).\n"
            "4. OUTPUT FORMAT (RAW JSON ONLY):\n"
            "{\n"
            '  "gauges": [{\n'
            '    "gauge_index": 1,\n'
            '    "gauge_type": "stem_thermometer",\n'
            '    "bbox": [ymin, xmin, ymax, xmax],\n'
            '    "reading": {"value": <float>, "unit": "°C", "confidence": 0.95},\n'
            '    "scale": {"min_val": <float>, "max_val": <float>, "unit": "°C"},\n'
            '    "needle": {"angle_deg": <float>, "tip": [y, x], "center": [y, x]}\n'
            "  }]\n"
            "}"
        )
    },
    {
        "id": "digital_flowmeter_tokico",
        "name": "Tokico Mechanical Drum Flow Meter",
        "category": "Digital Meters",
        "description": "Heavy industrial fuel/oil flow meter (TOKICO brand). Features a prominent rectangular odometer-style roller digit counter indicating total accumulated Liters/m3 plus an auxiliary fractional needle dial.",
        "sample_reference": "3_Digital Meters/3_Mounted Flow Meter",
        "shape": "Heavy cast-iron casing with rectangular roller drum window",
        "scale_info": "5 to 7 mechanical roller digits (e.g. 130477.9 or 130478.2), unit Liters (L) or m3",
        "indicator_type": "Odometer-style mechanical drum wheels (white on black + red decimal) & fractional needle",
        "prompt": (
            "You are inspecting a TOKICO MECHANICAL ROLLER DRUM FLOW METER.\n"
            "ODOMETER DRUM READING PROTOCOL:\n"
            "1. LOCATE ROLLER COUNTER WINDOW:\n"
            "   - Identify the rectangular counter window showing odometer-style mechanical roller digit wheels.\n"
            "   - Face plate often displays 'TOKICO' or 'OIL METER'.\n"
            "2. READ DIGIT WHEELS SEQUENTIALLY (LEFT TO RIGHT):\n"
            "   - Read consecutive whole digits (white digits on black background).\n"
            "   - The rightmost roller drum (or red-colored wheel) represents tenths of a Liter (0.1 L).\n"
            "   - If a roller drum is mid-transition, reference the auxiliary sub-dial or determine incoming digit.\n"
            "   - Combine into complete float reading (e.g. '130477.9' or '130478.2').\n"
            "3. AUXILIARY FRACTIONAL DIAL:\n"
            "   - An auxiliary circular sub-dial (0.0 - 1.0 L in 0.05 increments) confirms fine fractional flow.\n"
            "4. OUTPUT FORMAT (RAW JSON ONLY):\n"
            "{\n"
            '  "gauges": [{\n'
            '    "gauge_index": 1,\n'
            '    "gauge_type": "flowmeter",\n'
            '    "bbox": [ymin, xmin, ymax, xmax],\n'
            '    "reading": {"value": <float e.g. 130477.9>, "unit": "L", "confidence": 0.98},\n'
            '    "display_type": "roller_counter",\n'
            '    "digits_text": "<digit string e.g. 130477.9>"\n'
            "  }]\n"
            "}"
        )
    },
    {
        "id": "digital_7segment",
        "name": "7-Segment LED Display Panel",
        "category": "7-Segment Displays",
        "description": "Digital electronic indicator panel with illuminated red, green, or amber 7-segment LED digits on DC Power Supply Units or Minigra process panels. Displays voltage (V), current (A), or temperature.",
        "sample_reference": "2_7-Segment Displays/1_Minigra Panel & 2_DC Power Supply Unit",
        "shape": "Rectangular dark panel face with bright LED digit segments",
        "scale_info": "Digital numeric readout with active decimal points (e.g. 56.0 V, 12.50 A, 220.4 V)",
        "indicator_type": "7-segment illuminated LED bar digits and decimal dots",
        "prompt": (
            "You are inspecting a 7-SEGMENT LED DIGITAL DISPLAY PANEL.\n"
            "DIGIT SEGMENT RECOGNITION RULES:\n"
            "1. IDENTIFY ILLUMINATED SEGMENTS:\n"
            "   - Inspect the on/off state of standard 7 segments (a-g) to recognize digits 0 through 9.\n"
            "   - CRITICAL: Locate illuminated DECIMAL POINT (DP) dots to correctly format decimal fractions.\n"
            "2. CHANNEL & UNIT IDENTIFICATION:\n"
            "   - If multiple rows/channels are present:\n"
            "     * Row with 'V' indicates Voltage (e.g. 56.0 V).\n"
            "     * Row with 'A' indicates Current (e.g. 0.00 A).\n"
            "   - Extract exact numerical reading for each channel.\n"
            "3. OUTPUT FORMAT (RAW JSON ONLY):\n"
            "{\n"
            '  "gauges": [{\n'
            '    "gauge_index": 1,\n'
            '    "gauge_type": "seven_segment_display",\n'
            '    "bbox": [ymin, xmin, ymax, xmax],\n'
            '    "reading": {"value": <float e.g. 56.0>, "unit": "<V|A|°C>", "confidence": 0.98},\n'
            '    "display_text": "<string e.g. 56.0>"\n'
            "  }]\n"
            "}"
        )
    },
    {
        "id": "digital_energy_hour",
        "name": "Kilowatt-Hour & Operating Hour Meter",
        "category": "Digital Meters",
        "description": "Electromechanical or digital active energy meter (kWh) and cumulative runtime hour meter (h) on industrial Service Entrance Panels.",
        "sample_reference": "3_Digital Meters/2_Energy Meter_Operating Hour Meter & 1_Service Entrance Panel",
        "shape": "DIN-rail or panel-mount modular enclosure",
        "scale_info": "5 to 7 digits (e.g. 00022.1 kWh or 01450.8 h)",
        "indicator_type": "Mechanical roller drums or segmented LCD display",
        "prompt": (
            "You are inspecting an ENERGY (kWh) OR OPERATING RUNTIME (Hour) METER.\n"
            "DIGIT READOUT PROTOCOL:\n"
            "1. METER TYPE DISCRIMINATION:\n"
            "   - Energy Meter: Unit is 'kWh' or 'MWh'.\n"
            "   - Operating Hour Meter: Unit is 'h' (hours) or 'hrs'.\n"
            "2. READ SEQUENTIAL DIGITS:\n"
            "   - Read digits from left to right.\n"
            "   - Rightmost digit (often red-bordered or after a decimal comma) represents 1/10th (0.1 kWh or 0.1 h).\n"
            "3. OUTPUT FORMAT (RAW JSON ONLY):\n"
            "{\n"
            '  "gauges": [{\n'
            '    "gauge_index": 1,\n'
            '    "gauge_type": "<energy_meter|hour_meter>",\n'
            '    "bbox": [ymin, xmin, ymax, xmax],\n'
            '    "reading": {"value": <float>, "unit": "<kWh|h>", "confidence": 0.95},\n'
            '    "display_text": "<digit string>"\n'
            "  }]\n"
            "}"
        )
    },
    {
        "id": "oil_level_sight_glass",
        "name": "Gearbox Oil Level Sight Glass",
        "category": "Level Gauges",
        "description": "Lubricating oil level sight glass (bullseye circular window or tubular glass column) on heavy industrial gear reducers. Evaluates liquid meniscus level against MIN/MAX reference marks or red center bullseye.",
        "sample_reference": "5_Oil Level Gauges/Gear Reducer No.1, No.2, No.3",
        "shape": "Threaded brass/steel circular bullseye window or vertical sight tube",
        "scale_info": "MIN line, MAX line, or red center bullseye circle",
        "indicator_type": "Horizontal oil meniscus liquid boundary line (amber/gold liquid)",
        "prompt": (
            "You are inspecting a GEARBOX OIL LEVEL SIGHT GLASS.\n"
            "MENISCUS EVALUATION PROTOCOL:\n"
            "1. LOCATE SIGHT GLASS ON CASING:\n"
            "   - Locate the circular sight glass window or tubular glass column threaded into the gearbox housing.\n"
            "2. OIL-AIR MENISCUS LINE:\n"
            "   - Lower portion contains dark amber/yellow lubricating oil.\n"
            "   - Upper portion contains air.\n"
            "   - The horizontal boundary line between liquid and air is the oil level.\n"
            "3. STATUS ASSESSMENT:\n"
            "   - Reference against etched MIN/MAX lines or center red target circle.\n"
            "   - Target normal level is centered (40% to 60%).\n"
            "   - Determine status: 'NORMAL' (within bounds), 'LOW' (below MIN), or 'HIGH' (overfill).\n"
            "4. OUTPUT FORMAT (RAW JSON ONLY):\n"
            "{\n"
            '  "gauges": [{\n'
            '    "gauge_index": 1,\n'
            '    "gauge_type": "oil_level_sight_glass",\n'
            '    "bbox": [ymin, xmin, ymax, xmax],\n'
            '    "reading": {"value": <float percent e.g. 50.0>, "unit": "%", "status": "<NORMAL|LOW|HIGH>", "confidence": 0.95},\n'
            '    "scale": {"min_val": 0, "max_val": 100, "unit": "%"}\n'
            "  }]\n"
            "}"
        )
    },
    {
        "id": "gauge_board_multi",
        "name": "Industrial Multi-Gauge Panel Board",
        "category": "Multi-Gauge Panels",
        "description": "Multi-gauge industrial board or wall mounting an array of pressure gauges, thermometers, vacuum meters, or water level rulers simultaneously.",
        "sample_reference": "6_gauge board",
        "shape": "Large panel/wall array containing 2 to 10 meters",
        "scale_info": "Individual scale per detected meter",
        "indicator_type": "Multiple independent mechanical needles and scale rulers",
        "prompt": (
            "You are inspecting an INDUSTRIAL MULTI-GAUGE PANEL BOARD.\n"
            "MULTI-OBJECT DETECTION PROTOCOL:\n"
            "1. LOCATE ALL INDEPENDENT GAUGES:\n"
            "   - Scan the full image and generate a distinct bounding box [ymin, xmin, ymax, xmax] for EVERY meter on the board.\n"
            "   - Number them gauge_index 1, 2, 3... in top-to-bottom, left-to-right order.\n"
            "2. INSPECT EACH GAUGE INDIVIDUALLY:\n"
            "   - Determine gauge family (pressure, thermometer, vacuum, water level ruler).\n"
            "   - Read min_range, max_range, and unit for each meter.\n"
            "   - Determine needle pointer angle and deduce exact reading value.\n"
            "3. OUTPUT FORMAT (RAW JSON ONLY):\n"
            "{\n"
            '  "gauges": [\n'
            "    {\n"
            '      "gauge_index": 1,\n'
            '      "gauge_type": "<type 1>",\n'
            '      "bbox": [ymin, xmin, ymax, xmax],\n'
            '      "reading": {"value": <float>, "unit": "<unit 1>", "confidence": 0.95},\n'
            '      "scale": {"min_val": <float>, "max_val": <float>, "unit": "<unit 1>"},\n'
            '      "needle": {"angle_deg": <float>, "tip": [y, x], "center": [y, x]}\n'
            "    },\n"
            "    {\n"
            '      "gauge_index": 2,\n'
            '      "gauge_type": "<type 2>",\n'
            '      "bbox": [ymin, xmin, ymax, xmax],\n'
            '      "reading": {"value": <float>, "unit": "<unit 2>", "confidence": 0.95},\n'
            '      "scale": {"min_val": <float>, "max_val": <float>, "unit": "<unit 2>"},\n'
            '      "needle": {"angle_deg": <float>, "tip": [y, x], "center": [y, x]}\n'
            "    }\n"
            "  ]\n"
            "}"
        )
    }
]


def init_templates_directory(force_overwrite: bool = False):
    """Ensure the templates directory exists and contains all JSON template files"""
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    for tpl in DEFAULT_TEMPLATES:
        file_path = TEMPLATES_DIR / f"{tpl['id']}.json"
        if not file_path.exists() or force_overwrite:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(tpl, f, ensure_ascii=False, indent=2)


def get_all_templates() -> List[GaugeTemplate]:
    """Retrieve all gauge templates, loading from disk if available or falling back to defaults"""
    init_templates_directory()
    templates: List[GaugeTemplate] = []
    
    # Load order matching DEFAULT_TEMPLATES
    id_order = [t["id"] for t in DEFAULT_TEMPLATES]
    seen_ids = set()

    for tpl_id in id_order:
        file_path = TEMPLATES_DIR / f"{tpl_id}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    templates.append(GaugeTemplate(**data))
                    seen_ids.add(tpl_id)
            except Exception as e:
                print(f"Warning: Failed to load template {file_path}: {e}")

    # Fallback to in-memory for any missing
    for tpl in DEFAULT_TEMPLATES:
        if tpl["id"] not in seen_ids:
            templates.append(GaugeTemplate(**tpl))

    return templates


def get_template_by_id(template_id: str) -> Optional[GaugeTemplate]:
    """Get a specific template by its ID"""
    init_templates_directory()
    clean_id = (template_id or "default").strip().lower()
    file_path = TEMPLATES_DIR / f"{clean_id}.json"
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return GaugeTemplate(**json.load(f))
        except Exception as e:
            print(f"Warning: Error reading template file {file_path}: {e}")

    matched = next((t for t in DEFAULT_TEMPLATES if t["id"] == clean_id), None)
    if matched:
        return GaugeTemplate(**matched)
    return None


def get_prompt_for_type(template_id: Optional[str] = None) -> str:
    """Retrieve prompt text for a given template ID, defaulting to 'default'"""
    tpl = get_template_by_id(template_id or "default")
    if tpl:
        return tpl.prompt
    return DEFAULT_TEMPLATES[0]["prompt"]
