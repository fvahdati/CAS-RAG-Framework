"""Tests for emotion label parsing."""

from cas_rag.core.enums import EmotionLabel
from cas_rag.llm.parser import parse_emotion_label, parse_emotion_labels


def test_parse_single_emotion_label():
    text = "Emotion Label: Empathy\nSecondary Emotion Label: None\nExplanation: kind comment"
    primary, secondary = parse_emotion_labels(text)
    assert primary == EmotionLabel.EMPATHY
    assert secondary is None
    assert parse_emotion_label(text) == EmotionLabel.EMPATHY


def test_parse_primary_and_secondary_fields():
    text = (
        "Emotion Label: Anger\n"
        "Secondary Emotion Label: Frustration\n"
        "Explanation: outrage at unfair treatment"
    )
    primary, secondary = parse_emotion_labels(text)
    assert primary == EmotionLabel.ANGER
    assert secondary == EmotionLabel.FRUSTRATION


def test_parse_slash_separated_labels():
    text = "Emotion Label: Empathy / Anger\nExplanation: mixed reaction"
    primary, secondary = parse_emotion_labels(text)
    assert primary == EmotionLabel.EMPATHY
    assert secondary == EmotionLabel.ANGER
