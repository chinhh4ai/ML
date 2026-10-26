"""
Application configuration for Gauge Detection System
Supports Gemini, OpenAI, and LiteLLM providers
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    # Server config
    HOST: str = "0.0.0.0"
    PORT: int = 5070
    DEBUG: bool = True

    # AI Provider: 'hybrid', 'gemini', 'openai', 'litellm'
    AI_PROVIDER: str = "gemini"

    # Gemini config
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.8-flash"

    # OpenAI config
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"

    # LiteLLM config
    LITELLM_API_KEY: str = ""
    LITELLM_BASE_URL: str = "http://172.18.60.121:4000"
    LITELLM_MODEL: str = "Gemma-4-31B-IT-NVFP4-L40.0.127"

    # Confidence threshold
    CONFIDENCE_THRESHOLD: float = 0.5

    class Config:
        env_file = str(ENV_FILE)
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()


def update_settings(new_vals: dict):
    """Update settings in-memory and write changes to .env file"""
    global settings
    for key, val in new_vals.items():
        if val is not None and hasattr(settings, key.upper()):
            setattr(settings, key.upper(), val)

    # Persist to .env file
    try:
        lines = []
        if ENV_FILE.exists():
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()

        existing_keys = set()
        new_lines = []
        for line in lines:
            line_str = line.strip()
            if line_str and not line_str.startswith("#") and "=" in line_str:
                k = line_str.split("=")[0].strip()
                existing_keys.add(k)
                if k.lower() in new_vals and new_vals[k.lower()] is not None:
                    new_lines.append(f"{k}={new_vals[k.lower()]}\n")
                    continue
            new_lines.append(line)

        for k, v in new_vals.items():
            upper_k = k.upper()
            if upper_k not in existing_keys and v is not None:
                new_lines.append(f"{upper_k}={v}\n")

        with open(ENV_FILE, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    except Exception as e:
        print(f"Warning: Could not persist settings to .env: {e}")
