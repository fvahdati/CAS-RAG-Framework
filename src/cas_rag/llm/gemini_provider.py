"""Google Gemini provider."""

from cas_rag.config.settings import get_settings
from cas_rag.llm.base import BaseLLMProvider, LLMResponse


class GeminiProvider(BaseLLMProvider):
    @property
    def name(self) -> str:
        return "gemini"

    def complete(self, prompt: str, *, temperature: float = 0.1) -> LLMResponse:
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError("Install google-generativeai: pip install 'cas-rag[gemini]'") from e

        settings = get_settings()
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_model)
        resp = model.generate_content(prompt, generation_config={"temperature": temperature})
        return LLMResponse(text=resp.text or "", model=settings.gemini_model, provider=self.name)
