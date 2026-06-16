"""Tests for multi-LLM comparison column helpers."""

from cas_rag.experiments.llm_columns import (
    emotion_label_column,
    filter_base_dataframe_columns,
    is_legacy_algorithm_output_column,
    provider_column_suffix,
)


def test_provider_column_suffix():
    assert provider_column_suffix("openai") == "ChatGpt"
    assert provider_column_suffix("gemini") == "Gemini"


def test_emotion_label_column_name():
    assert emotion_label_column("Alg1", "openai") == "Alg1_emotion_label_ChatGpt"
    assert emotion_label_column("CAS-RAG", "gemini") == "CAS-RAG_emotion_label_Gemini"


def test_legacy_columns_are_removed():
    assert is_legacy_algorithm_output_column("Alg1_status")
    assert is_legacy_algorithm_output_column("CAS-RAG_rag_used")
    assert is_legacy_algorithm_output_column("algorithm_rag_needed")
    assert not is_legacy_algorithm_output_column("Alg1_emotion_label_ChatGpt")


def test_filter_base_dataframe_columns():
    cols = filter_base_dataframe_columns(
        [
            "comment_id",
            "comment_text",
            "gold_label",
            "Alg1_status",
            "Alg1_emotion_label",
            "Alg1_emotion_label_ChatGpt",
            "algorithm_rag_needed",
        ]
    )
    assert cols == ["comment_id", "comment_text", "gold_label", "Alg1_emotion_label_ChatGpt"]
