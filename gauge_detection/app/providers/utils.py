"""
Utility functions for AI Vision providers
Includes robust JSON extraction and normalization for diverse LLM outputs.
"""
import re
import json
from typing import Dict, Any, Optional


def extract_and_parse_json(raw_text: Optional[str]) -> Dict[str, Any]:
    """
    Extracts and parses JSON object from raw LLM output.
    Handles:
    - Markdown code fences (```json ... ```)
    - Chain-of-thought tags (<think>...</think>)
    - Leading/trailing conversational text
    - Trailing commas
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty response received from model.")

    text = raw_text.strip()

    # 1. Strip reasoning / thinking tags if present
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.DOTALL).strip()

    # 2. Strip Markdown code fences if the entire content is fenced
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (``` or ```json)
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove last line (```)
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # 3. Locate the outermost JSON object { ... }
    json_match = re.search(r'(\{[\s\S]*\})', text)
    if json_match:
        text = json_match.group(1)

    # 4. Remove trailing commas before closing braces/brackets
    cleaned_commas = re.sub(r',\s*([\}\]])', r'\1', text)

    try:
        return json.loads(cleaned_commas)
    except json.JSONDecodeError:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # If JSON was truncated, attempt to close open strings and brackets
            repaired = text.rstrip().rstrip(",")
            if repaired.count('"') % 2 == 1:
                repaired += '"'
            open_brackets = repaired.count('[') - repaired.count(']')
            open_braces = repaired.count('{') - repaired.count('}')
            repaired += (']' * max(0, open_brackets)) + ('}' * max(0, open_braces))
            return json.loads(repaired)
