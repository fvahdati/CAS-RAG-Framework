"""Domain enumerations for CAS-RAG."""

from enum import Enum


class AlgorithmId(str, Enum):
    ALGORITHM_1 = "algorithm_1"
    ALGORITHM_2 = "algorithm_2"
    ALGORITHM_3 = "algorithm_3"
    ALGORITHM_4 = "algorithm_4"
    CAS_RAG = "cas_rag"


class PromptId(str, Enum):
    P1_UNKNOWN_REFERENCE = "P1"
    P2_RAG_EMOTION = "P2"
    P3_NO_RAG_EMOTION = "P3"
    P4_AMBIGUITY = "P4"
    P5_SEMANTIC_SIMILARITY = "P5"
    P6_AMBIGUITY_AWARE = "P6"
    P7_NO_RAG_NO_VAGUE = "P7"


class EmotionLabel(str, Enum):
    SHOCK = "Shock"
    DISBELIEF = "Disbelief"
    ANGER = "Anger"
    FRUSTRATION = "Frustration"
    EMPATHY = "Empathy"
    NONE = "None"
    VAGUE = "Vague"
    AMBIGUOUS = "Ambiguous"
    DISCARDED = "DISCARDED"


class SemanticCategory(str, Enum):
    EXPLICIT = "Explicit"
    VAGUE = "Vague"
    CONTEXT_DEPENDENT = "Context-Dependent"


class PipelineStatus(str, Enum):
    SUCCESS = "success"
    DISCARDED = "discarded"
    PENDING_HUMAN = "pending_human"


EMOTION_LABELS = {
    EmotionLabel.SHOCK,
    EmotionLabel.DISBELIEF,
    EmotionLabel.ANGER,
    EmotionLabel.FRUSTRATION,
    EmotionLabel.EMPATHY,
    EmotionLabel.NONE,
}
