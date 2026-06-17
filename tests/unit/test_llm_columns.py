"""Tests for multi-LLM comparison column helpers."""

from cas_rag.experiments.llm_columns import (
    emotion_label_column,
    filter_base_dataframe_columns,
    is_legacy_algorithm_output_column,
    provider_column_suffix,
    rag_used_column,
    routing_correct_column,
    secondary_emotion_label_column,
    semantic_category_column,
)


def test_provider_column_suffix():
    assert provider_column_suffix("openai") == "ChatGpt"
    assert provider_column_suffix("gemini") == "Gemini"


def test_emotion_label_column_name():
    assert emotion_label_column("Alg1", "openai") == "Alg1_emotion_label_ChatGpt"
    assert emotion_label_column("CAS-RAG", "gemini") == "CAS-RAG_emotion_label_Gemini"
    assert secondary_emotion_label_column("Alg1", "openai") == "Alg1_secondary_emotion_label_ChatGpt"
    assert rag_used_column("Alg1", "openai") == "Alg1_rag_used_ChatGpt"
    assert semantic_category_column("CAS-RAG", "openai") == "CAS-RAG_semantic_category_ChatGpt"
    assert routing_correct_column("CAS-RAG", "gemini") == "CAS-RAG_routing_correct_Gemini"


def test_legacy_columns_are_removed():
    assert is_legacy_algorithm_output_column("Alg1_status")
    assert is_legacy_algorithm_output_column("Alg1_status_ChatGpt")
    assert is_legacy_algorithm_output_column("CAS-RAG_rag_used")
    assert is_legacy_algorithm_output_column("algorithm_rag_needed")
    assert not is_legacy_algorithm_output_column("Alg1_emotion_label_ChatGpt")
    assert not is_legacy_algorithm_output_column("Alg1_secondary_emotion_label_ChatGpt")
    assert not is_legacy_algorithm_output_column("Alg1_rag_used_ChatGpt")
    assert not is_legacy_algorithm_output_column("CAS-RAG_semantic_category_ChatGpt")
    assert not is_legacy_algorithm_output_column("CAS-RAG_routing_correct_ChatGpt")


def test_filter_base_dataframe_columns():
    cols = filter_base_dataframe_columns(
        [
            "comment_id",
            "comment_text",
            "gold_label",
            "Alg1_status",
            "Alg1_emotion_label",
            "Alg1_emotion_label_ChatGpt",
            "Alg1_secondary_emotion_label_ChatGpt",
            "algorithm_rag_needed",
        ]
    )
    assert cols == [
        "comment_id",
        "comment_text",
        "gold_label",
        "Alg1_emotion_label_ChatGpt",
        "Alg1_secondary_emotion_label_ChatGpt",
    ]

    cols_with_metadata = filter_base_dataframe_columns(
        [
            "comment_id",
            "Alg1_rag_used_ChatGpt",
            "CAS-RAG_semantic_category_ChatGpt",
            "CAS-RAG_routing_correct_ChatGpt",
            "Alg1_status_ChatGpt",
        ]
    )
    assert cols_with_metadata == [
        "comment_id",
        "Alg1_rag_used_ChatGpt",
        "CAS-RAG_semantic_category_ChatGpt",
        "CAS-RAG_routing_correct_ChatGpt",
    ]
