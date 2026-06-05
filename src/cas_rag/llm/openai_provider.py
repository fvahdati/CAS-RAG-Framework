"""OpenAI / ChatGPT provider."""

from cas_rag.config.settings import get_settings
from cas_rag.llm.base import BaseLLMProvider, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    @property
    def name(self) -> str:
        return "openai"

    def complete(self, prompt: str, *, temperature: float = 0.1) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Install openai: pip install 'cas-rag[openai]'") from e

        settings = get_settings()
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
        return LLMResponse(text=text, model=settings.openai_model, provider=self.name)
