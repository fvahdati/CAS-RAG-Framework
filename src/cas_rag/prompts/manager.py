"""Prompt loading, versioning, and template rendering."""

from __future__ import annotations

from pathlib import Path

from cas_rag.config.settings import get_settings
from cas_rag.core.enums import PromptId


class PromptManager:
    """Loads versioned prompts from resources/prompts/{version}/."""

    FILENAME_ALIASES = {
        PromptId.P1_UNKNOWN_REFERENCE: ["P1_unknown_reference_detection.txt", "P1_prompt.txt"],
        PromptId.P2_RAG_EMOTION: ["P2_rag_emotion_classification.txt", "P2_prompt.txt"],
        PromptId.P3_NO_RAG_EMOTION: ["P3_no_rag_emotion_classification.txt", "P3_prompt.txt"],
        PromptId.P4_AMBIGUITY: ["P4_semantic_ambiguity_verification.txt", "P4_prompt.txt"],
        PromptId.P5_SEMANTIC_SIMILARITY: ["P5_semantic_similarity_explicit_vague.txt", "P5_prompt.txt"],
        PromptId.P6_AMBIGUITY_AWARE: ["P6_semantic_ambiguity_aware_emotion.txt", "P6_prompt.txt"],
        PromptId.P7_NO_RAG_NO_VAGUE: ["P7_no_rag_emotion_without_vagueness_check.txt", "P7_prompt.txt"],
    }

    def __init__(self, version: str | None = None, base_dir: Path | None = None):
        settings = get_settings()
        self.version = version or settings.prompt_version
        self.base_dir = base_dir or (settings.project_root / "resources" / "prompts" / self.version)
        self._cache: dict[PromptId, str] = {}

    def load(self, prompt_id: PromptId) -> str:
        if prompt_id not in self._cache:
            path = None
            for name in self.FILENAME_ALIASES[prompt_id]:
                candidate = self.base_dir / name
                if candidate.exists():
                    path = candidate
                    break
            if path is None:
                raise FileNotFoundError(
                    f"Prompt not found for {prompt_id.value} in {self.base_dir}"
                )
            self._cache[prompt_id] = path.read_text(encoding="utf-8")
        return self._cache[prompt_id]

    def render(
        self,
        prompt_id: PromptId,
        *,
        comment: str,
        scenario: str,
        retrieved_context: str = "",
        prototypes_block: str = "",
    ) -> str:
        template = self.load(prompt_id)
        return (
            template.replace("{comment}", comment)
            .replace("{retrieved context}", retrieved_context)
            .replace("{retrieved_context}", retrieved_context)
            .replace("{video scenario}", scenario)
            .replace("{scenario}", scenario)
            .replace("{prototypes}", prototypes_block)
        )

    def list_versions(self) -> list[str]:
        root = get_settings().project_root / "resources" / "prompts"
        return [p.name for p in root.iterdir() if p.is_dir()]
