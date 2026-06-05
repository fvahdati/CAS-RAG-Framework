"""Ollama provider for local Llama / Meta models."""

import httpx

from cas_rag.config.settings import get_settings
from cas_rag.llm.base import BaseLLMProvider, LLMResponse


class OllamaProvider(BaseLLMProvider):
    @property
    def name(self) -> str:
        return "ollama"

    def complete(self, prompt: str, *, temperature: float = 0.1) -> LLMResponse:
        settings = get_settings()
        url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
        payload = {
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        with httpx.Client(timeout=120.0) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        return LLMResponse(text=data.get("response", ""), model=settings.ollama_model, provider=self.name)
