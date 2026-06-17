"""Tests for LLM provider probing and Gemini fallback."""

from unittest.mock import patch

import pytest

from cas_rag.experiments.comparison_runner import _resolve_llm_providers
from cas_rag.llm.probe import resolve_providers_with_gemini_fallback


def test_resolve_providers_with_gemini_fallback_uses_ollama():
    with patch("cas_rag.llm.probe.probe_llm_provider") as probe:
        probe.side_effect = [
            (False, "quota exceeded"),
            (True, ""),
        ]
        resolved, messages = resolve_providers_with_gemini_fallback(["openai", "gemini"])
    assert resolved == ["openai", "ollama"]
    assert messages
    assert "Ollama" in messages[0] or "ollama" in messages[0]


def test_resolve_providers_with_gemini_fallback_keeps_gemini_when_ok():
    with patch("cas_rag.llm.probe.probe_llm_provider", return_value=(True, "")):
        resolved, messages = resolve_providers_with_gemini_fallback(["openai", "gemini"])
    assert resolved == ["openai", "gemini"]
    assert messages == []


def test_resolve_providers_with_gemini_fallback_raises_when_both_fail():
    with patch("cas_rag.llm.probe.probe_llm_provider", return_value=(False, "down")):
        with pytest.raises(ValueError, match="fallback ollama failed"):
            resolve_providers_with_gemini_fallback(["openai", "gemini"])


def test_comparison_runner_applies_fallback_for_default_providers():
    with patch("cas_rag.llm.probe.probe_llm_provider") as probe:
        probe.side_effect = [(False, "quota"), (True, "")]
        providers = _resolve_llm_providers(None, None)
    assert providers == ["openai", "ollama"]
