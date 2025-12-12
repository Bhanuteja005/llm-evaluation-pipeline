"""Configuration management for LLM evaluation pipeline."""

import os
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM Configuration
    llm_provider: Literal["mock", "gemini"] = "mock"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_max_tokens: int = 500

    # Cost Configuration (USD per token)
    cost_per_input_token: float = 0.0000015
    cost_per_output_token: float = 0.000002

    # Evaluation Configuration
    top_k_contexts: int = 5
    relevance_threshold: float = 0.7
    factual_verification_threshold: float = 0.8
    factual_weak_threshold: float = 0.5
    max_latency_ms: int = 2000
    max_cost_usd: float = 0.01

    # Database
    database_url: str = "sqlite:///./evaluations.db"

    # Logging
    log_level: str = "INFO"

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"

    def __init__(self, **kwargs):  # type: ignore
        """Initialize settings."""
        super().__init__(**kwargs)
        # Validate Gemini settings if provider is gemini
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY must be set when LLM_PROVIDER is 'gemini'")


# Global settings instance
settings = Settings()
