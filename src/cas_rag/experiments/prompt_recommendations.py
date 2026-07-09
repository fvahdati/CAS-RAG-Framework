"""Detailed offline prompt-fix recommendations for failure analysis reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cas_rag.core.enums import AlgorithmId, PromptId

PROMPT_FILE_NAMES: dict[PromptId, str] = {
    PromptId.P1_UNKNOWN_REFERENCE: "P1_unknown_reference_detection.txt",
    PromptId.P2_RAG_EMOTION: "P2_rag_emotion_classification.txt",
    PromptId.P3_NO_RAG_EMOTION: "P3_no_rag_emotion_classification.txt",
    PromptId.P4_AMBIGUITY: "P4_semantic_ambiguity_verification.txt",
    PromptId.P5_SEMANTIC_SIMILARITY: "P5_semantic_similarity_explicit_vague.txt",
    PromptId.P6_AMBIGUITY_AWARE: "P6_semantic_ambiguity_aware_emotion.txt",
    PromptId.P7_NO_RAG_NO_VAGUE: "P7_no_rag_emotion_without_vagueness_check.txt",
}

EMOTION_PROMPTS = frozenset(
    {
        PromptId.P2_RAG_EMOTION,
        PromptId.P3_NO_RAG_EMOTION,
        PromptId.P4_AMBIGUITY,
        PromptId.P6_AMBIGUITY_AWARE,
        PromptId.P7_NO_RAG_NO_VAGUE,
    }
)

ROUTING_PROMPT_BY_ALGORITHM: dict[AlgorithmId, str] = {
    AlgorithmId.ALGORITHM_1: "P1",
    AlgorithmId.ALGORITHM_2: "spaCy routing (not a prompt file)",
    AlgorithmId.ALGORITHM_3: "P5 + spaCy",
    AlgorithmId.ALGORITHM_4: "P5 + spaCy",
    AlgorithmId.CAS_RAG: "Layer-1 semantic routing (+ P5 on borderline)",
}


@dataclass(frozen=True)
class PromptFixRecommendation:
    contributing_prompt: str
    prompt_file: str
    section_to_edit: str
    suggested_prompt_fix: str
    example_to_add: str


def _last_emotion_prompt(prompts_invoked: list[PromptId]) -> PromptId | None:
    for prompt_id in reversed(prompts_invoked):
        if prompt_id in EMOTION_PROMPTS:
            return prompt_id
    return None


def _prompt_label(prompt_id: PromptId | None) -> str:
    return prompt_id.value if prompt_id else ""


def _prompt_file(prompt_id: PromptId | None) -> str:
    if prompt_id is None:
        return ""
    return PROMPT_FILE_NAMES.get(prompt_id, "")


def identify_contributing_prompt(
    algorithm: AlgorithmId,
    prompts_invoked: list[PromptId],
    reasons: list[str],
) -> str:
    routing_reasons = {
        "rag_detection_error",
        "rag_under_triggered",
        "rag_over_triggered",
        "semantic_routing_mismatch",
        "ambiguous_under_detected",
        "context_dependent_under_detected",
        "generic_comment_misrouted",
        "borderline_semantic_routing",
    }
    emotion_reasons = {
        "emotion_mismatch",
        "empathy_frustration_confusion",
        "pending_human_review",
        "discarded_by_pipeline",
    }

    if any(r in routing_reasons for r in reasons):
        if algorithm == AlgorithmId.CAS_RAG:
            if "borderline_semantic_routing" in reasons:
                return "P5 (borderline arbitration)"
            if any(r in reasons for r in ("ambiguous_under_detected", "generic_comment_misrouted")):
                return "Layer-1 prototypes + P5/P6"
            return "Layer-1 semantic routing"
        if algorithm == AlgorithmId.ALGORITHM_1 and any(
            r in reasons for r in ("rag_detection_error", "rag_under_triggered", "rag_over_triggered")
        ):
            return "P1"
        if algorithm in (AlgorithmId.ALGORITHM_3, AlgorithmId.ALGORITHM_4) and any(
            r in reasons
            for r in (
                "ambiguous_under_detected",
                "generic_comment_misrouted",
                "semantic_routing_mismatch",
                "borderline_semantic_routing",
            )
        ):
            return "P5"
        return ROUTING_PROMPT_BY_ALGORITHM.get(algorithm, "routing layer")

    if any(r in emotion_reasons for r in reasons):
        emotion_prompt = _last_emotion_prompt(prompts_invoked)
        if emotion_prompt:
            return _prompt_label(emotion_prompt)
        if "pending_human_review" in reasons:
            return "P4 or P6"
    return _prompt_label(_last_emotion_prompt(prompts_invoked)) or "unknown"


def explain_why_misclassified(
    *,
    expected_emotion: str,
    predicted_emotion: str,
    expected_category: str,
    predicted_category: str,
    rag_used: bool,
    emotion_correct: bool | None,
    routing_correct: bool | None,
    reasons: list[str],
    algorithm: str,
) -> str:
    if "correct" in reasons and emotion_correct is not False and routing_correct is not False:
        return "Classification matches available gold labels for this row."

    parts: list[str] = []
    if emotion_correct is False:
        parts.append(
            f"Emotion mismatch: expected '{expected_emotion}' but pipeline returned '{predicted_emotion}'."
        )
    if routing_correct is False:
        if algorithm == "CAS-RAG" and expected_category and predicted_category:
            parts.append(
                f"Routing mismatch: expected semantic category '{expected_category}' "
                f"but pipeline routed to '{predicted_category}'."
            )
        elif expected_category:
            expected_rag = expected_category == "context_dependent"
            parts.append(
                f"RAG routing mismatch for {algorithm}: expected rag_used={expected_rag} "
                f"but pipeline used rag_used={rag_used}."
            )
    if "empathy_frustration_confusion" in reasons:
        parts.append(
            "The comment likely expresses solidarity or admiration, but the model treated harsh-tutor "
            "language as Frustration instead of Empathy."
        )
    if "generic_comment_misrouted" in reasons:
        parts.append(
            "Short generic praise was not treated as Ambiguous, so emotion classification proceeded "
            "without enough video-specific evidence."
        )
    if "pending_human_review" in reasons:
        parts.append("Pipeline stopped at an ambiguous verification step and requires human review.")
    if not parts:
        parts.append("Failure signals were detected, but no gold labels were available to confirm the target.")
    return " ".join(parts)


def build_prompt_fix_recommendation(
    *,
    algorithm: AlgorithmId,
    prompts_invoked: list[PromptId],
    reasons: list[str],
    comment_text: str,
    expected_emotion: str,
    predicted_emotion: str,
    expected_category: str,
    predicted_category: str,
) -> PromptFixRecommendation:
    contributing = identify_contributing_prompt(algorithm, prompts_invoked, reasons)
    emotion_prompt = _last_emotion_prompt(prompts_invoked)
    snippet = comment_text.strip().replace("\n", " ")
    if len(snippet) > 90:
        snippet = snippet[:87] + "..."

    if "pending_human_review" in reasons:
        verify_prompt = (
            PromptId.P4_AMBIGUITY
            if PromptId.P4_AMBIGUITY in prompts_invoked
            else PromptId.P6_AMBIGUITY_AWARE
        )
        return PromptFixRecommendation(
            contributing_prompt=_prompt_label(verify_prompt),
            prompt_file=_prompt_file(verify_prompt),
            section_to_edit="Ambiguous definition + Instructions",
            suggested_prompt_fix=(
                "Clarify when generic praise should remain Ambiguous versus when enough video-specific "
                "evidence exists to assign a concrete emotion label."
            ),
            example_to_add=(
                f'If gold is ambiguous: keep "{snippet}" as Ambiguous with a short explanation.\n'
                f"If gold is {expected_emotion or 'a concrete emotion'}: add a clear non-ambiguous example near this pattern."
            ),
        )

    if "empathy_frustration_confusion" in reasons:
        prompt_id = emotion_prompt or PromptId.P2_RAG_EMOTION
        return PromptFixRecommendation(
            contributing_prompt=_prompt_label(prompt_id),
            prompt_file=_prompt_file(prompt_id),
            section_to_edit="Empathy vs Frustration (important)",
            suggested_prompt_fix=(
                "Clarify that hypothetical harsh-tutor wording still maps to Empathy when the dominant "
                "reaction is admiration for the helping children or compassion for the struggling child."
            ),
            example_to_add=(
                f'Example (Empathy): "{snippet}" -> Empathy when praise/solidarity toward the children '
                "or struggling student is the dominant reaction."
            ),
        )

    if "rag_over_triggered" in reasons:
        return PromptFixRecommendation(
            contributing_prompt="P1",
            prompt_file=PROMPT_FILE_NAMES[PromptId.P1_UNKNOWN_REFERENCE],
            section_to_edit='Examples + "A reference should be considered resolved"',
            suggested_prompt_fix=(
                "Add a No example showing that scenario-known entities do not require script retrieval."
            ),
            example_to_add=(
                f'Comment: "{snippet}"\nUnknown Reference: No\n'
                "Explanation: References the tutor, children, or scenario events without needing script context."
            ),
        )

    if "rag_under_triggered" in reasons:
        return PromptFixRecommendation(
            contributing_prompt="P1",
            prompt_file=PROMPT_FILE_NAMES[PromptId.P1_UNKNOWN_REFERENCE],
            section_to_edit='Examples + "A reference should be considered unknown or unresolved"',
            suggested_prompt_fix=(
                "Add a Yes example for scene-specific names, moments, or pronouns that require script grounding."
            ),
            example_to_add=(
                f'Comment: "{snippet}"\nUnknown Reference: Yes\n'
                "Explanation: Refers to a specific person or scene moment that cannot be resolved from the scenario summary alone."
            ),
        )

    if "ambiguous_under_detected" in reasons or "generic_comment_misrouted" in reasons:
        return PromptFixRecommendation(
            contributing_prompt=contributing,
            prompt_file=(
                f"{PROMPT_FILE_NAMES[PromptId.P5_SEMANTIC_SIMILARITY]}; "
                f"{PROMPT_FILE_NAMES[PromptId.P3_NO_RAG_EMOTION]}; prototypes_semantic.json"
            ),
            section_to_edit="Ambiguous prototypes / Ambiguous label examples",
            suggested_prompt_fix=(
                "Add this comment (or a close variant) to ambiguous prototypes and to P4/P5/P6 generic-comment "
                "examples so short praise is routed away from explicit emotion classification."
            ),
            example_to_add=(
                f'Ambiguous example: "{snippet}"\n'
                "Also add to prototypes_semantic.json -> ambiguous."
            ),
        )

    if "context_dependent_under_detected" in reasons:
        return PromptFixRecommendation(
            contributing_prompt=contributing,
            prompt_file="prototypes_semantic.json",
            section_to_edit="context_dependent prototypes",
            suggested_prompt_fix=(
                "Add scene-cue prototypes containing names, temporal phrases, or specific diner moments."
            ),
            example_to_add=(
                f'Context-dependent prototype: "{snippet}"'
            ),
        )

    if "borderline_semantic_routing" in reasons:
        return PromptFixRecommendation(
            contributing_prompt="P5",
            prompt_file=PROMPT_FILE_NAMES[PromptId.P5_SEMANTIC_SIMILARITY],
            section_to_edit="Ambiguous Emotional Reaction Prototypes + Definitions",
            suggested_prompt_fix=(
                "Review close-score comments in prototypes before changing runtime logic. Add clearer "
                "ambiguous or explicit prototypes that resemble this comment."
            ),
            example_to_add=(
                f'Add a prototype in the category you want Layer-1 to prefer for: "{snippet}"'
            ),
        )

    if "emotion_mismatch" in reasons and emotion_prompt:
        section = "Label Definitions (video-anchored)"
        if emotion_prompt == PromptId.P2_RAG_EMOTION:
            section = "Label Definitions (video-anchored) + Empathy vs Frustration (important)"
        return PromptFixRecommendation(
            contributing_prompt=_prompt_label(emotion_prompt),
            prompt_file=_prompt_file(emotion_prompt),
            section_to_edit=section,
            suggested_prompt_fix=(
                f"Add or refine a video-anchored example for expected label '{expected_emotion}' that matches "
                f"comments like this one and contrasts with predicted label '{predicted_emotion}'."
            ),
            example_to_add=(
                f'Example ({expected_emotion}): "{snippet}"'
            ),
        )

    return PromptFixRecommendation(
        contributing_prompt=contributing,
        prompt_file=_prompt_file(emotion_prompt),
        section_to_edit="Review prompts_used and routing_path",
        suggested_prompt_fix="Inspect the invoked prompt chain for this algorithm and align examples with the gold label.",
        example_to_add=f'Add a tutoring-video example for: "{snippet}"',
    )


def recommendation_to_row(rec: PromptFixRecommendation) -> dict[str, str]:
    return {
        "contributing_prompt": rec.contributing_prompt,
        "prompt_file": rec.prompt_file,
        "prompt_section_to_edit": rec.section_to_edit,
        "suggested_prompt_fix": rec.suggested_prompt_fix,
        "example_to_add": rec.example_to_add,
    }


def build_prompt_fix_guide(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate detailed prompt-fix guidance for failing comments only."""
    guide: list[dict[str, Any]] = []
    for row in rows:
        if row.get("emotion_correct") is not False and row.get("routing_correct") is not False:
            if "pending_human_review" not in str(row.get("likely_failure_reason", "")):
                continue
        guide.append(
            {
                "comment_id": row.get("comment_id", ""),
                "algorithm": row.get("algorithm", ""),
                "comment_text": row.get("comment_text", ""),
                "why_misclassified": row.get("why_misclassified", ""),
                "contributing_prompt": row.get("contributing_prompt", ""),
                "prompt_file": row.get("prompt_file", ""),
                "prompt_section_to_edit": row.get("prompt_section_to_edit", ""),
                "suggested_prompt_fix": row.get("suggested_prompt_fix", ""),
                "example_to_add": row.get("example_to_add", ""),
                "likely_failure_reason": row.get("likely_failure_reason", ""),
            }
        )
    if not guide:
        guide.append(
            {
                "comment_id": "",
                "algorithm": "all",
                "comment_text": "",
                "why_misclassified": "No failing comments in this subset.",
                "contributing_prompt": "",
                "prompt_file": "",
                "prompt_section_to_edit": "",
                "suggested_prompt_fix": "Expand the subset or include expected gold labels.",
                "example_to_add": "",
                "likely_failure_reason": "none",
            }
        )
    return guide
