"""Pydantic domain models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from cas_rag.core.datetime_utils import utc_now

from pydantic import BaseModel, Field

from cas_rag.core.enums import (
    AlgorithmId,
    EmotionLabel,
    PipelineStatus,
    PromptId,
    SemanticCategory,
)


class CommentInput(BaseModel):
    comment_id: str
    comment: str
    scenario_id: str = "tutoring_social_experiment"
    gold_label: EmotionLabel | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HumanReviewDecision(BaseModel):
    comment_id: str
    confirms_ambiguous: bool = False
    resolvable: bool = False
    assigned_label: EmotionLabel | None = None
    reviewer_id: str = "reviewer"
    reviewed_at: datetime = Field(default_factory=utc_now)
    notes: str = ""


class PipelineResult(BaseModel):
    comment_id: str
    algorithm: AlgorithmId
    status: PipelineStatus
    emotion_label: EmotionLabel | None = None
    secondary_emotion_label: EmotionLabel | None = None
    semantic_category: SemanticCategory | None = None
    rag_used: bool = False
    prompts_invoked: list[PromptId] = Field(default_factory=list)
    retrieved_context: str | None = None
    explanation: str = ""
    trace: list[str] = Field(default_factory=list)
    pending_human: bool = False
    human_queue_id: str | None = None

    @property
    def discarded(self) -> bool:
        return self.status == PipelineStatus.DISCARDED
