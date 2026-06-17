"""LLM provider names for multi-model comparison export columns."""

from __future__ import annotations

import re

PROVIDER_COLUMN_SUFFIX: dict[str, str] = {
    "openai": "ChatGpt",
    "gemini": "Gemini",
    "ollama": "Ollama",
    "mock": "Mock",
}

ALGORITHM_SHORT_NAMES: list[tuple[str, str]] = [
    ("Alg1", "algorithm_1"),
    ("Alg2", "algorithm_2"),
    ("Alg3", "algorithm_3"),
    ("Alg4", "algorithm_4"),
    ("CAS-RAG", "cas_rag"),
]

# Legacy / pipeline metadata columns stripped from input and output workbooks.
DROP_EXACT_COLUMNS = frozenset(
    {
        "algorithm_rag_needed",
        "algorithm_infeasible",
        "algorithm_rag_outcome",
    }
)


def provider_column_suffix(provider: str) -> str:
    key = provider.lower().strip()
    if key in ("llama", "meta"):
        key = "ollama"
    return PROVIDER_COLUMN_SUFFIX.get(key, provider.strip().title().replace(" ", ""))


def emotion_label_column(algorithm_short: str, provider: str) -> str:
    return f"{algorithm_short}_emotion_label_{provider_column_suffix(provider)}"


def secondary_emotion_label_column(algorithm_short: str, provider: str) -> str:
    return f"{algorithm_short}_secondary_emotion_label_{provider_column_suffix(provider)}"


def rag_used_column(algorithm_short: str, provider: str) -> str:
    return f"{algorithm_short}_rag_used_{provider_column_suffix(provider)}"


def semantic_category_column(algorithm_short: str, provider: str) -> str:
    return f"{algorithm_short}_semantic_category_{provider_column_suffix(provider)}"


def routing_correct_column(algorithm_short: str, provider: str) -> str:
    return f"{algorithm_short}_routing_correct_{provider_column_suffix(provider)}"


_EXPORT_COLUMN_MARKERS = (
    "_emotion_label_",
    "_secondary_emotion_label_",
    "_rag_used_",
    "_semantic_category_",
    "_routing_correct_",
)


def is_legacy_algorithm_output_column(column: str) -> bool:
    if column in DROP_EXACT_COLUMNS:
        return True
    if column.startswith("algorithm_"):
        return True
    if re.match(r"^(Alg\d|CAS-RAG)_", column):
        if "_status" in column:
            return True
        return not any(marker in column for marker in _EXPORT_COLUMN_MARKERS)
    return False


def filter_base_dataframe_columns(columns: list[str]) -> list[str]:
    """Keep source dataset columns; drop prior algorithm run outputs."""
    return [c for c in columns if not is_legacy_algorithm_output_column(c)]
