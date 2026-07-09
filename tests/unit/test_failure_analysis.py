"""Tests for offline failure analysis helpers."""

from pathlib import Path

from cas_rag.core.enums import AlgorithmId, EmotionLabel, PipelineStatus, PromptId, SemanticCategory
from cas_rag.core.models import CommentInput, PipelineResult
from cas_rag.experiments.failure_analysis import (
    build_evaluation_row,
    build_failure_summary,
    build_iteration_recommendations,
    build_prompt_fix_guide,
    infer_failure_reasons,
    normalize_category,
)
from cas_rag.experiments.prompt_recommendations import (
    build_prompt_fix_recommendation,
    explain_why_misclassified,
    identify_contributing_prompt,
)


def test_normalize_category_maps_vague_to_ambiguous():
    assert normalize_category("Vague") == "ambiguous"
    assert normalize_category("Context-Dependent") == "context_dependent"


def test_infer_emotion_mismatch_and_empathy_frustration():
    reasons = infer_failure_reasons(
        algorithm=AlgorithmId.ALGORITHM_1,
        expected_emotion="Empathy",
        expected_category="explicit",
        predicted_emotion="Frustration",
        predicted_category="explicit",
        rag_used=False,
        status=PipelineStatus.SUCCESS.value,
        semantic_borderline=False,
        comment_text="If I had a tutor that harsh I would feel so stupid.",
    )
    assert "emotion_mismatch" in reasons
    assert "empathy_frustration_confusion" in reasons


def test_infer_rag_detection_error():
    reasons = infer_failure_reasons(
        algorithm=AlgorithmId.ALGORITHM_1,
        expected_emotion="Empathy",
        expected_category="context_dependent",
        predicted_emotion="Empathy",
        predicted_category="",
        rag_used=False,
        status=PipelineStatus.SUCCESS.value,
        semantic_borderline=False,
        comment_text="When Harper walked over I melted.",
    )
    assert "rag_detection_error" in reasons
    assert "rag_under_triggered" in reasons


def test_build_evaluation_row_offline(tmp_path: Path):
    prototypes = tmp_path / "prototypes_semantic.json"
    prototypes.write_text(
        '{"explicit":["Those kids were sweet"],"ambiguous":["Wow"],"context_dependent":["When Harper helped"]}',
        encoding="utf-8",
    )
    inp = CommentInput(comment_id="c1", comment="Wow amazing")
    result = PipelineResult(
        comment_id="c1",
        algorithm=AlgorithmId.CAS_RAG,
        status=PipelineStatus.SUCCESS,
        emotion_label=EmotionLabel.NONE,
        semantic_category=SemanticCategory.EXPLICIT,
        prompts_invoked=[PromptId.P7_NO_RAG_NO_VAGUE],
        trace=["semantic=Explicit", "scores={'explicit': 0.7, 'ambiguous': 0.65, 'context_dependent': 0.3}"],
    )
    row = build_evaluation_row(
        inp,
        result,
        gold_row={"expected_category": "ambiguous", "gold_label": "None"},
        prototypes_path=prototypes,
    )
    assert row["expected_category"] == "ambiguous"
    assert row["prompts_used"] == "P7"
    assert row["confidence"] is not None
    assert "ambiguous_under_detected" in row["likely_failure_reason"]
    assert row["why_misclassified"]
    assert row["contributing_prompt"]
    assert row["suggested_prompt_fix"]


def test_build_failure_summary_and_recommendations():
    rows = [
        {
            "emotion_correct": False,
            "routing_correct": False,
            "likely_failure_reason": "emotion_mismatch; empathy_frustration_confusion",
            "algorithm": "Alg1",
            "comment_id": "c1",
            "contributing_prompt": "P2",
            "prompt_file": "P2_rag_emotion_classification.txt",
            "prompt_section_to_edit": "Empathy vs Frustration (important)",
            "suggested_prompt_fix": "Clarify empathy rule",
            "example_to_add": 'Example (Empathy): "test"',
        }
    ]
    summary = build_failure_summary(rows)
    assert summary[0]["key"] == "emotion_mismatch"
    recs = build_iteration_recommendations(rows)
    assert recs[0]["prompt_file"] == "P2_rag_emotion_classification.txt"
    guide = build_prompt_fix_guide(rows)
    assert guide[0]["contributing_prompt"] == "P2"


def test_prompt_recommendation_for_rag_over_trigger():
    rec = build_prompt_fix_recommendation(
        algorithm=AlgorithmId.ALGORITHM_1,
        prompts_invoked=[PromptId.P1_UNKNOWN_REFERENCE, PromptId.P2_RAG_EMOTION],
        reasons=["rag_detection_error", "rag_over_triggered"],
        comment_text="Those kids were so brave.",
        expected_emotion="Empathy",
        predicted_emotion="None",
        expected_category="explicit",
        predicted_category="",
    )
    assert rec.contributing_prompt == "P1"
    assert "P1_unknown_reference_detection.txt" in rec.prompt_file
    assert "resolved" in rec.section_to_edit.lower()


def test_explain_why_misclassified_emotion_and_routing():
    text = explain_why_misclassified(
        expected_emotion="Empathy",
        predicted_emotion="Frustration",
        expected_category="explicit",
        predicted_category="explicit",
        rag_used=True,
        emotion_correct=False,
        routing_correct=False,
        reasons=["emotion_mismatch", "rag_over_triggered"],
        algorithm="Alg1",
    )
    assert "Emotion mismatch" in text
    assert "RAG routing mismatch" in text


def test_identify_contributing_prompt_for_p1():
    prompt = identify_contributing_prompt(
        AlgorithmId.ALGORITHM_1,
        [PromptId.P1_UNKNOWN_REFERENCE, PromptId.P2_RAG_EMOTION],
        ["rag_over_triggered"],
    )
    assert prompt == "P1"
