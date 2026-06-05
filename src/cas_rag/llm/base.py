"""LLM provider abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str


class BaseLLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, *, temperature: float = 0.1) -> LLMResponse:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
