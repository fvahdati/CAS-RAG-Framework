"""Parse structured LLM outputs (Emotion Label, Unknown Reference, etc.)."""

import re

from cas_rag.core.enums import EmotionLabel


def parse_unknown_reference(text: str) -> bool:
    m = re.search(r"unknown\s*reference\s*:\s*(yes|no)", text, re.I)
    return bool(m and m.group(1).lower() == "yes")


def parse_semantic_similarity(text: str) -> str:
    m = re.search(r"semantic\s*similarity\s*label\s*:\s*(explicit|ambiguous|vague)", text, re.I)
    if not m:
        return "Explicit"
    label = m.group(1).lower()
    return "Ambiguous" if label in ("ambiguous", "vague") else "Explicit"


def parse_emotion_label(text: str) -> EmotionLabel:
    m = re.search(
        r"emotion\s*label\s*:\s*"
        r"(shock|disbelief|anger|frustration|empathy|none|vague|ambiguous|discarded)",
        text,
        re.I,
    )
    if not m:
        return EmotionLabel.NONE
    key = m.group(1).lower()
    mapping = {
        "shock": EmotionLabel.SHOCK,
        "disbelief": EmotionLabel.DISBELIEF,
        "anger": EmotionLabel.ANGER,
        "frustration": EmotionLabel.FRUSTRATION,
        "empathy": EmotionLabel.EMPATHY,
        "none": EmotionLabel.NONE,
        "vague": EmotionLabel.AMBIGUOUS,
        "ambiguous": EmotionLabel.AMBIGUOUS,
        "discarded": EmotionLabel.DISCARDED,
    }
    return mapping.get(key, EmotionLabel.NONE)
