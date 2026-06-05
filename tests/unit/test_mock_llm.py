"""Unit tests for mock LLM and parsers."""

from cas_rag.llm.mock_provider import MockLLMProvider
from cas_rag.llm.parser import parse_emotion_label, parse_unknown_reference


def test_unknown_reference_yes():
    p = MockLLMProvider()
    out = p.complete("unknown reference detection\nInput Comment:\nShe handled that well.")
    assert parse_unknown_reference(out.text)


def test_emotion_label_empathy():
    p = MockLLMProvider()
    out = p.complete("emotion classification\nempathy\nInput Comment:\nThe kids showed empathy")
    assert parse_emotion_label(out.text).value == "Empathy"
