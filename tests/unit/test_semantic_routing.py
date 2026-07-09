"""Tests for semantic routing (Layer 1)."""

import json
from pathlib import Path

import pytest

from cas_rag.core.enums import SemanticCategory
from cas_rag.semantic.similarity import SemanticSimilarityEngine


@pytest.fixture
def engine(tmp_path: Path) -> SemanticSimilarityEngine:
    prototypes = {
        "explicit": ["Those kids were so sweet and brave to help."],
        "ambiguous": ["Wow just wow.", "Love this video."],
        "context_dependent": ["When that specific moment happened, the reaction changed."],
    }
    path = tmp_path / "prototypes_semantic.json"
    path.write_text(json.dumps(prototypes), encoding="utf-8")
    return SemanticSimilarityEngine(prototypes_path=path)


def test_generic_comment_routes_ambiguous(engine: SemanticSimilarityEngine):
    category, _, borderline = engine.classify_three_way("Wow amazing")
    assert category == SemanticCategory.AMBIGUOUS


def test_context_cue_routes_context_dependent(engine: SemanticSimilarityEngine):
    category, _, _ = engine.classify_three_way(
        "When that specific moment happened, the reaction changed."
    )
    assert category == SemanticCategory.CONTEXT_DEPENDENT


def test_no_ambiguous_downgrade_to_explicit(engine: SemanticSimilarityEngine):
    category, scores, _ = engine.classify_three_way("Mixed feelings about hidden camera setups")
    assert scores["ambiguous"] >= scores["explicit"]
    assert category == SemanticCategory.AMBIGUOUS


def test_explicit_comment_routes_explicit(engine: SemanticSimilarityEngine):
    category, _, _ = engine.classify_three_way(
        "Those kids were so sweet and brave to help that struggling student."
    )
    assert category == SemanticCategory.EXPLICIT
