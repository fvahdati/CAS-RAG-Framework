"""LLM-based linguistic analysis (Prompt P1 — unknown reference detection)."""

from cas_rag.subsystems.emotion_services import UnknownReferenceDetection


class LLMLinguisticAnalysis:
    """Wrapper aligning subsystem naming with thesis terminology."""

    def __init__(self, ref_detection: UnknownReferenceDetection):
        self._ref = ref_detection

    def has_unknown_reference(self, comment: str, scenario: str) -> bool:
        return self._ref.detect(comment, scenario)
