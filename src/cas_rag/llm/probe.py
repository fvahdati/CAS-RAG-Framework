"""Lightweight live checks for LLM providers before batch runs."""

from __future__ import annotations

import httpx

from cas_rag.config.settings import Settings, get_settings
from cas_rag.llm.factory import create_llm_provider
from cas_rag.llm.validation import validate_llm_provider


def _normalize_provider(provider: str) -> str:
    name = provider.lower().strip()
    if name in ("llama", "meta"):
        return "ollama"
    return name


def _probe_ollama(settings: Settings) -> tuple[bool, str]:
    base = settings.ollama_base_url.rstrip("/")
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.get(f"{base}/api/tags")
            r.raise_for_status()
            models = [m.get("name", "") for m in r.json().get("models", [])]
            wanted = settings.ollama_model
            if not any(m == wanted or m.startswith(f"{wanted}:") for m in models):
                return (
                    False,
                    f"model {wanted!r} not found locally; run: ollama pull {wanted}",
                )
        resp = create_llm_provider("ollama").complete("Reply with exactly: OK.")
        if not (resp.text or "").strip():
            return False, "empty response from Ollama"
        return True, ""
    except httpx.HTTPError as e:
        return False, f"Ollama not reachable at {base} ({e})"
    except Exception as e:
        return False, str(e)[:300]


def probe_llm_provider(provider: str, *, settings: Settings | None = None) -> tuple[bool, str]:
    """Return (ok, error_message). Performs a minimal live request when possible."""
    settings = settings or get_settings()
    name = _normalize_provider(provider)
    try:
        validate_llm_provider(settings, name)
    except ValueError as e:
        return False, str(e)

    if name == "ollama":
        return _probe_ollama(settings)

    try:
        resp = create_llm_provider(name).complete("Reply with exactly: OK.")
        if not (resp.text or "").strip():
            return False, "empty response"
        return True, ""
    except Exception as e:
        return False, str(e)[:300]


def resolve_providers_with_gemini_fallback(
    providers: list[str],
    *,
    fallback: str = "ollama",
    settings: Settings | None = None,
) -> tuple[list[str], list[str]]:
    """Swap gemini for fallback provider when Gemini probe fails.

    Returns (resolved_providers, info_messages).
    """
    settings = settings or get_settings()
    resolved: list[str] = []
    messages: list[str] = []
    fallback_name = _normalize_provider(fallback)

    for provider in providers:
        name = _normalize_provider(provider)
        if name != "gemini":
            resolved.append(name)
            continue

        ok, err = probe_llm_provider("gemini", settings=settings)
        if ok:
            resolved.append("gemini")
            continue

        fb_ok, fb_err = probe_llm_provider(fallback_name, settings=settings)
        if fb_ok:
            messages.append(
                f"Gemini unavailable ({err}); using {fallback_name} for second LLM columns."
            )
            resolved.append(fallback_name)
            continue

        raise ValueError(
            f"Gemini unavailable ({err}) and fallback {fallback_name} failed ({fb_err}). "
            f"Fix Gemini billing/quota or start Ollama "
            f"(ollama serve && ollama pull {settings.ollama_model})."
        )

    return resolved, messages
