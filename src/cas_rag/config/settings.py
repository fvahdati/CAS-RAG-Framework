"""Application settings loaded from YAML and environment."""

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="CAS_RAG_",
        extra="ignore",
    )

    llm_provider: str = Field(
        default="openai",
        description="openai | gemini | ollama (mock only if allow_mock=True)",
    )
    allow_mock: bool = Field(
        default=False,
        description="Allow rule-based mock LLM (tests/CI only)",
    )
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    retrieval_top_k: int = Field(default=3, alias="RETRIEVAL_TOP_K")
    prompt_version: str = Field(default="v1", alias="PROMPT_VERSION")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", validation_alias="GEMINI_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2", validation_alias="OLLAMA_MODEL")
    offline_mode: bool = Field(default=False, validation_alias="OFFLINE_MODE")
    huggingface_token: str = Field(default="", validation_alias="HF_TOKEN")
    youtube_api_key: str = Field(default="", validation_alias="YOUTUBE_API_KEY")

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def config_path(self) -> Path:
        return PROJECT_ROOT / "config" / "default.yaml"

    def load_yaml(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f) or {}
        return {}


def _apply_hub_token(settings: Settings) -> None:
    """Expose HF token to sentence-transformers / huggingface_hub."""
    token = settings.huggingface_token.strip()
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    _apply_hub_token(settings)
    return settings
