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

TEMPLATE_IDS_ORDER = [
    "default",
    "dial_standard",
    "dial_dual_pointer",
    "dial_control_panel",
    "dial_power_factor",
    "dial_extended_scale",
    "stem_thermometer",
    "gauge_board_multi",
    "digital_7segment",
    "digital_flowmeter_tokico",
    "digital_energy_hour",
    "oil_level_sight_glass",
]


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


def _load_disk_templates() -> List[Dict[str, Any]]:
    """Load all templates from JSON files on disk in specified order"""
    loaded = []
    for tid in TEMPLATE_IDS_ORDER:
        file_path = TEMPLATES_DIR / f"{tid}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    loaded.append(json.load(f))
            except Exception as e:
                print(f"Warning: Failed to load template from {file_path}: {e}")
    return loaded


# Loaded at module import from standardized JSON files
DEFAULT_TEMPLATES: List[Dict[str, Any]] = _load_disk_templates()


def init_templates_directory(force_overwrite: bool = False):
    """Ensure the templates directory exists and contains all JSON template files"""
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    for tpl in DEFAULT_TEMPLATES:
        file_path = TEMPLATES_DIR / f"{tpl['id']}.json"
        if not file_path.exists() or force_overwrite:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(tpl, f, ensure_ascii=False, indent=2)


def get_all_templates() -> List[GaugeTemplate]:
    """Retrieve all gauge templates, loading from disk or memory"""
    templates: List[GaugeTemplate] = []
    seen_ids = set()

    for tid in TEMPLATE_IDS_ORDER:
        file_path = TEMPLATES_DIR / f"{tid}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    templates.append(GaugeTemplate(**data))
                    seen_ids.add(tid)
            except Exception as e:
                print(f"Warning: Failed to load template {file_path}: {e}")

    # Fallback to in-memory for any missing
    for tpl in DEFAULT_TEMPLATES:
        if tpl["id"] not in seen_ids:
            templates.append(GaugeTemplate(**tpl))
            seen_ids.add(tpl["id"])

    return templates


def get_template_by_id(template_id: str) -> Optional[GaugeTemplate]:
    """Get a specific template by its ID"""
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
    if DEFAULT_TEMPLATES:
        return DEFAULT_TEMPLATES[0]["prompt"]
    return ""
