"""Emotion classification and ambiguity subsystems."""

from cas_rag.core.enums import EmotionLabel, PromptId
from cas_rag.llm.base import BaseLLMProvider
from cas_rag.llm.parser import parse_emotion_label, parse_unknown_reference, parse_semantic_similarity
from cas_rag.prompts.manager import PromptManager
from cas_rag.subsystems.human_verification import HumanVerificationService


class UnknownReferenceDetection:
    def __init__(self, llm: BaseLLMProvider, prompts: PromptManager):
        self.llm = llm
        self.prompts = prompts

    def detect(self, comment: str, scenario: str) -> bool:
        prompt = self.prompts.render(PromptId.P1_UNKNOWN_REFERENCE, comment=comment, scenario=scenario)
        resp = self.llm.complete(prompt)
        return parse_unknown_reference(resp.text)


class RAGEmotionClassifier:
    def __init__(self, llm: BaseLLMProvider, prompts: PromptManager):
        self.llm = llm
        self.prompts = prompts

    def classify(self, comment: str, scenario: str, retrieved: str) -> tuple[EmotionLabel, str]:
        prompt = self.prompts.render(
            PromptId.P2_RAG_EMOTION, comment=comment, scenario=scenario, retrieved_context=retrieved
        )
        resp = self.llm.complete(prompt)
        return parse_emotion_label(resp.text), resp.text


class NoRAGEmotionClassifier:
    def __init__(self, llm: BaseLLMProvider, prompts: PromptManager):
        self.llm = llm
        self.prompts = prompts

    def classify_p3(self, comment: str, scenario: str) -> tuple[EmotionLabel, str]:
        prompt = self.prompts.render(PromptId.P3_NO_RAG_EMOTION, comment=comment, scenario=scenario)
        resp = self.llm.complete(prompt)
        return parse_emotion_label(resp.text), resp.text

    def classify_p7(self, comment: str, scenario: str) -> tuple[EmotionLabel, str]:
        prompt = self.prompts.render(PromptId.P7_NO_RAG_NO_VAGUE, comment=comment, scenario=scenario)
        resp = self.llm.complete(prompt)
        return parse_emotion_label(resp.text), resp.text


class SemanticAmbiguityDetection:
    def __init__(self, llm: BaseLLMProvider, prompts: PromptManager, human: HumanVerificationService):
        self.llm = llm
        self.prompts = prompts
        self.human = human

    def classify_p4(self, comment: str, scenario: str, comment_id: str, algorithm: str) -> tuple[EmotionLabel | None, bool]:
        prompt = self.prompts.render(PromptId.P4_AMBIGUITY, comment=comment, scenario=scenario)
        resp = self.llm.complete(prompt)
        label = parse_emotion_label(resp.text)
        return self._handle_ambiguity(label, comment, comment_id, algorithm, "P4")

    def classify_p6(self, comment: str, scenario: str, comment_id: str, algorithm: str) -> tuple[EmotionLabel | None, bool]:
        prompt = self.prompts.render(PromptId.P6_AMBIGUITY_AWARE, comment=comment, scenario=scenario)
        resp = self.llm.complete(prompt)
        label = parse_emotion_label(resp.text)
        return self._handle_ambiguity(label, comment, comment_id, algorithm, "P6")

    def _handle_ambiguity(
        self, label: EmotionLabel, comment: str, comment_id: str, algorithm: str, prompt_name: str
    ) -> tuple[EmotionLabel | None, bool]:
        if label != EmotionLabel.AMBIGUOUS:
            return label, False
        existing = self.human.resolve_for_comment(comment_id)
        if existing:
            result = self.human.apply_review(existing)
            return (None if result == EmotionLabel.DISCARDED else result), False
        self.human.enqueue(comment_id, comment, f"{prompt_name}: Ambiguous", algorithm)
        return None, True


class SemanticSimilarityLLM:
    def __init__(self, llm: BaseLLMProvider, prompts: PromptManager):
        self.llm = llm
        self.prompts = prompts

    def classify(self, comment: str, scenario: str) -> str:
        prompt = self.prompts.render(PromptId.P5_SEMANTIC_SIMILARITY, comment=comment, scenario=scenario)
        resp = self.llm.complete(prompt)
        return parse_semantic_similarity(resp.text)
