"""Tests for spaCy linguistic heuristics."""

from cas_rag.linguistic.spacy_analyzer import SpacyLinguisticAnalyzer


def test_resolved_demonstrative_does_not_force_rag():
    analyzer = SpacyLinguisticAnalyzer()
    comment = "Those kids were so sweet and brave to help that struggling student."
    assert analyzer.needs_rag(comment, "tutoring diner experiment") is False


def test_unresolved_pronoun_still_needs_rag():
    analyzer = SpacyLinguisticAnalyzer()
    comment = "When she walked over I knew exactly what she meant."
    assert analyzer.needs_rag(comment, "tutoring diner experiment") is True
