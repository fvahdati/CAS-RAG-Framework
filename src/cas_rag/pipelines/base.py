"""Base pipeline context shared by all algorithms."""

from dataclasses import dataclass, field

from cas_rag.core.enums import AlgorithmId
from cas_rag.llm.base import BaseLLMProvider
from cas_rag.prompts.manager import PromptManager
from cas_rag.retrieval.rag_engine import RAGEngine
from cas_rag.semantic.similarity import SemanticSimilarityEngine
from cas_rag.linguistic.spacy_analyzer import SpacyLinguisticAnalyzer
from cas_rag.subsystems.emotion_services import (
    UnknownReferenceDetection,
    RAGEmotionClassifier,
    NoRAGEmotionClassifier,
    SemanticAmbiguityDetection,
    SemanticSimilarityLLM,
)
from cas_rag.subsystems.human_verification import HumanVerificationService


@dataclass
class PipelineContext:
    algorithm: AlgorithmId
    scenario: str
    script_corpus_path: str
    llm: BaseLLMProvider
    prompts: PromptManager
    rag: RAGEngine
    semantic: SemanticSimilarityEngine
    spacy: SpacyLinguisticAnalyzer
    human: HumanVerificationService
    ref_detection: UnknownReferenceDetection = field(init=False)
    rag_emotion: RAGEmotionClassifier = field(init=False)
    no_rag: NoRAGEmotionClassifier = field(init=False)
    ambiguity: SemanticAmbiguityDetection = field(init=False)
    semantic_llm: SemanticSimilarityLLM = field(init=False)

    def __post_init__(self):
        self.ref_detection = UnknownReferenceDetection(self.llm, self.prompts)
        self.rag_emotion = RAGEmotionClassifier(self.llm, self.prompts)
        self.no_rag = NoRAGEmotionClassifier(self.llm, self.prompts)
        self.ambiguity = SemanticAmbiguityDetection(self.llm, self.prompts, self.human)
        self.semantic_llm = SemanticSimilarityLLM(self.llm, self.prompts)
        self.rag.load_corpus(self.script_corpus_path)
