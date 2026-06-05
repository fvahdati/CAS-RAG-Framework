"""Tests for LLM provider validation."""

import pytest

from cas_rag.config.settings import Settings
from cas_rag.llm.validation import validate_llm_provider


def test_mock_blocked_without_allow_mock():
    settings = Settings.model_validate({"llm_provider": "mock", "allow_mock": False})
    with pytest.raises(ValueError, match="Mock LLM is disabled"):
        validate_llm_provider(settings, "mock")


def test_openai_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(
        llm_provider="openai",
        allow_mock=False,
        openai_api_key="",
        _env_file=None,
    )
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        validate_llm_provider(settings, "openai")
