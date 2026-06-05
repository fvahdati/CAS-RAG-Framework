"""Validate LLM provider configuration before pipeline runs."""

from __future__ import annotations

from cas_rag.config.settings import Settings

SUPPORTED_PROVIDERS = ("openai", "gemini", "ollama", "llama", "meta", "mock")


def validate_llm_provider(settings: Settings, provider: str) -> str:
    """Return normalized provider name or raise ValueError with setup instructions."""
    name = provider.lower().strip()
    if name in ("llama", "meta"):
        name = "ollama"

    if name not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Use one of: openai (ChatGPT), gemini, ollama (local Llama/Meta). "
            f"Set CAS_RAG_LLM_PROVIDER in .env or pass --llm."
        )

    if name == "mock":
        if not settings.allow_mock:
            raise ValueError(
                "Mock LLM is disabled. CAS-RAG is configured to use real models only. "
                "Set CAS_RAG_LLM_PROVIDER=openai|gemini|ollama and the matching API key in .env. "
                "For automated tests only, set CAS_RAG_ALLOW_MOCK=1."
            )
        return name

    if name == "openai" and not settings.openai_api_key.strip():
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to .env (see .env.example). "
            "Install: pip install -e '.[openai]'"
        )

    if name == "gemini" and not settings.gemini_api_key.strip():
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to .env (see .env.example). "
            "Install: pip install -e '.[gemini]'"
        )

    return name
