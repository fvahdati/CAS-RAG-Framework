"""Deterministic mock LLM for tests and offline demos."""

import re

from cas_rag.core.enums import EmotionLabel
from cas_rag.llm.base import BaseLLMProvider, LLMResponse


class MockLLMProvider(BaseLLMProvider):
    """Rule-based responses aligned with thesis prompt output formats."""

    @property
    def name(self) -> str:
        return "mock"

    def complete(self, prompt: str, *, temperature: float = 0.1) -> LLMResponse:
        text = self._infer(prompt)
        return LLMResponse(text=text, model="mock-v1", provider=self.name)

    def _infer(self, prompt: str) -> str:
        lower = prompt.lower()
        comment_match = re.search(r"input comment:\s*\n?\s*(.+?)(?:\n\n|\nrespond|\Z)", prompt, re.I | re.S)
        comment = (comment_match.group(1).strip() if comment_match else "").lower()

        if "unknown reference" in lower[:200]:
            if any(p in comment for p in ("she ", "he ", "they ", "that kid", "when ", "that part")):
                return "Unknown Reference: Yes\n\nExplanation: Unresolved referent detected."
            return "Unknown Reference: No\n\nExplanation: References resolvable from scenario."

        if "semantic similarity label" in lower or "explicit prototypes" in lower:
            vague_cues = ("uneasy", "mixed", "not sure", "wow", "amazing", "hard to tell", "performative")
            if any(c in comment for c in vague_cues):
                return "Semantic Similarity Label: Ambiguous\n\nExplanation: Weak or uncertain reaction."
            return "Semantic Similarity Label: Explicit\n\nExplanation: Clear emotional alignment."

        if "ambiguous" in lower and "verify" in lower:
            if comment in ("wow", "amazing", "wow,amazing") or len(comment.split()) <= 2:
                return f"Emotion Label: {EmotionLabel.AMBIGUOUS.value}\n\nExplanation: Too generic."
            return f"Emotion Label: {EmotionLabel.EMPATHY.value}\n\nExplanation: Clear supportive reaction."

        if "retrieved" in lower or "rag knowledge" in lower:
            if "harsh" in comment or "frustrat" in comment:
                label = EmotionLabel.FRUSTRATION
            elif "empathy" in comment or "kids" in comment or "help" in comment:
                label = EmotionLabel.EMPATHY
            else:
                label = EmotionLabel.NONE
            return f"Emotion Label: {label.value}\n\nExplanation: Classified with retrieved context."

        if ("ambiguous" in lower or "vague" in lower) and "predefined" in lower:
            if len(comment.split()) <= 2:
                return f"Emotion Label: {EmotionLabel.AMBIGUOUS.value}\n\nExplanation: Generic comment."
            return f"Emotion Label: {EmotionLabel.EMPATHY.value}\n\nExplanation: Dominant emotion explicit."

        label = EmotionLabel.EMPATHY if "empathy" in comment or "kids" in comment else EmotionLabel.NONE
        return f"Emotion Label: {label.value}\n\nExplanation: No-RAG classification."
