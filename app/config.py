"""
Qwen3-TTS API Configuration
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Model settings
    model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    device: str = "auto"  # auto, cuda:0, mps, cpu
    dtype: str = "float32"  # float32, float16, bfloat16

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Audio settings
    default_sample_rate: int = 24000
    max_text_length: int = 5000

    class Config:
        env_file = ".env"


settings = Settings()
