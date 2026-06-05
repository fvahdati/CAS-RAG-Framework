"""Factory for multi-provider LLM selection."""

from cas_rag.config.settings import get_settings
from cas_rag.llm.base import BaseLLMProvider
from cas_rag.llm.mock_provider import MockLLMProvider
from cas_rag.llm.validation import validate_llm_provider


def create_llm_provider(provider: str | None = None) -> BaseLLMProvider:
    settings = get_settings()
    raw = provider or settings.llm_provider
    name = validate_llm_provider(settings, raw)

    if name == "mock":
        return MockLLMProvider()
    if name == "openai":
        from cas_rag.llm.openai_provider import OpenAIProvider

        return OpenAIProvider()
    if name == "gemini":
        from cas_rag.llm.gemini_provider import GeminiProvider

        return GeminiProvider()
    if name == "ollama":
        from cas_rag.llm.ollama_provider import OllamaProvider

        return OllamaProvider()
    raise ValueError(f"Unknown LLM provider: {name}")
