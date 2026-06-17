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


_EMOTION_LABEL_MAP = {
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

_NONE_SECONDARY_TOKENS = frozenset({"none", "n/a", "na", "-", ""})


def _token_to_emotion_label(token: str) -> EmotionLabel:
    key = token.strip().lower()
    if key in _EMOTION_LABEL_MAP:
        return _EMOTION_LABEL_MAP[key]
    m = re.search(
        r"(shock|disbelief|anger|frustration|empathy|none|vague|ambiguous|discarded)",
        key,
        re.I,
    )
    if not m:
        return EmotionLabel.NONE
    return _EMOTION_LABEL_MAP.get(m.group(1).lower(), EmotionLabel.NONE)


def parse_emotion_labels(text: str) -> tuple[EmotionLabel, EmotionLabel | None]:
    """Parse primary and optional secondary emotion labels from LLM output."""
    primary = EmotionLabel.NONE
    secondary: EmotionLabel | None = None

    primary_match = re.search(r"(?<!secondary\s)emotion\s*label\s*:\s*([^\n]+)", text, re.I)
    if primary_match:
        primary_text = primary_match.group(1).strip()
        if "/" in primary_text:
            parts = [p.strip() for p in primary_text.split("/") if p.strip()]
            if parts:
                primary = _token_to_emotion_label(parts[0])
            if len(parts) > 1 and parts[1].lower() not in _NONE_SECONDARY_TOKENS:
                secondary = _token_to_emotion_label(parts[1])
        else:
            primary = _token_to_emotion_label(primary_text)

    secondary_match = re.search(r"secondary\s*emotion\s*label\s*:\s*([^\n]+)", text, re.I)
    if secondary_match:
        secondary_text = secondary_match.group(1).strip()
        if secondary_text.lower() not in _NONE_SECONDARY_TOKENS:
            secondary = _token_to_emotion_label(secondary_text)

    return primary, secondary


def parse_emotion_label(text: str) -> EmotionLabel:
    primary, _ = parse_emotion_labels(text)
    return primary
