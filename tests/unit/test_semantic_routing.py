"""Tests for semantic routing (Layer 1) matching thesis margins m=0.12, b=0.08."""

import json
from pathlib import Path

from cas_rag.core.enums import SemanticCategory
from cas_rag.semantic.similarity import SemanticSimilarityEngine


def _engine(tmp_path: Path) -> SemanticSimilarityEngine:
    prototypes = {
        "explicit": ["Those kids were so sweet and brave to help."],
        "ambiguous": ["Wow just wow.", "Love this video."],
        "context_dependent": ["When that specific moment happened, the reaction changed."],
    }
    path = tmp_path / "prototypes_semantic.json"
    path.write_text(json.dumps(prototypes), encoding="utf-8")
    return SemanticSimilarityEngine(prototypes_path=path)


def test_margin_rule_flips_weak_ambiguous_to_explicit(tmp_path: Path):
    engine = _engine(tmp_path)
    category, _ = engine.apply_margin_rules(
        {"explicit": 0.60, "ambiguous": 0.65, "context_dependent": 0.20}
    )
    assert category == SemanticCategory.EXPLICIT


def test_margin_keeps_strong_ambiguous(tmp_path: Path):
    engine = _engine(tmp_path)
    category, _ = engine.apply_margin_rules(
        {"explicit": 0.40, "ambiguous": 0.55, "context_dependent": 0.20}
    )
    assert category == SemanticCategory.AMBIGUOUS


def test_context_dependent_argmax_unchanged_by_exp_amb_margins(tmp_path: Path):
    engine = _engine(tmp_path)
    category, _ = engine.apply_margin_rules(
        {"explicit": 0.30, "ambiguous": 0.20, "context_dependent": 0.80}
    )
    assert category == SemanticCategory.CONTEXT_DEPENDENT


def test_two_way_ignores_context_dependent_class(tmp_path: Path):
    engine = _engine(tmp_path)
    engine._model = False
    engine._proto_emb = {}
    label = engine.classify_two_way("When that specific moment happened, the reaction changed.")
    assert label in ("Explicit", "Ambiguous")


def test_offline_overlap_uses_prototypes_not_phrase_lists(tmp_path: Path):
    engine = _engine(tmp_path)
    engine._model = False
    engine._proto_emb = {}
    category, scores, _ = engine.classify_three_way("Those kids were so sweet and brave to help.")
    assert scores["explicit"] >= scores["ambiguous"]
    assert category in (
        SemanticCategory.EXPLICIT,
        SemanticCategory.AMBIGUOUS,
        SemanticCategory.CONTEXT_DEPENDENT,
    )
