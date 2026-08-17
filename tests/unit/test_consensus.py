"""Tests for offline multi-LLM consensus evaluation (RQ1)."""

from cas_rag.evaluation.consensus import (
    compute_consensus,
    emotion_matches_gold,
    normalize_emotion_label,
)


def test_normalize_emotion_strips_parenthetical():
    assert normalize_emotion_label("Frustration(Borderline Case)") == "Frustration"


def test_unanimous_high_consensus():
    result = compute_consensus(
        {"openai": "Empathy", "gemini": "Empathy"},
        gold_label="Empathy",
    )
    assert result.consensus_label == "Empathy"
    assert result.reliability == "HIGH"
    assert result.unanimous is True
    assert result.disagreement_flag is False
    assert result.hallucination_proxy is False


def test_disagreement_split_with_two_providers():
    result = compute_consensus(
        {"openai": "Empathy", "gemini": "Anger"},
        gold_label="Empathy",
    )
    assert result.reliability == "SPLIT"
    assert result.consensus_label is None
    assert result.disagreement_flag is True
    assert result.hallucination_proxy is True


def test_excludes_non_votable_labels():
    result = compute_consensus(
        {"openai": "DISCARDED", "gemini": "PENDING_HUMAN"},
    )
    assert result.reliability == "NON_LABELLED"
    assert result.n_valid_votes == 0


def test_single_provider_high():
    result = compute_consensus({"openai": "None"}, gold_label="None")
    assert result.reliability == "HIGH"
    assert result.consensus_label == "None"


def test_emotion_matches_gold_case_insensitive():
    assert emotion_matches_gold("empathy", "Empathy") is True
    assert emotion_matches_gold("Anger", "Empathy") is False
    assert emotion_matches_gold("", "Empathy") is False
