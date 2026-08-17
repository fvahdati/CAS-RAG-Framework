"""Offline multi-LLM consensus and reliability signalling for RQ1 evaluation.

This module does not alter pipeline runtime behaviour. It aggregates emotion labels
from existing multi-provider comparison exports (OpenAI, Gemini, optional Ollama).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

ReliabilityTier = Literal["HIGH", "LOW", "SPLIT", "NON_LABELLED"]

NON_VOTABLE_LABELS = frozenset(
    {
        "",
        "discarded",
        "pending_human",
        "none_labelled",
    }
)


def normalize_emotion_label(value: str | None) -> str:
    """Normalise emotion labels for voting (matches evaluation workbook rules)."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "?"}:
        return ""
    text = re.sub(r"\(.*?\)", "", text).strip()
    return text


def is_votable_emotion(label: str) -> bool:
    norm = normalize_emotion_label(label).lower()
    return bool(norm) and norm not in NON_VOTABLE_LABELS


@dataclass
class ConsensusResult:
    providers: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    consensus_label: str | None = None
    agreement_ratio: float = 0.0
    unanimous: bool = False
    reliability: ReliabilityTier = "NON_LABELLED"
    disagreement_flag: bool = False
    hallucination_proxy: bool | None = None
    n_valid_votes: int = 0


def compute_consensus(
    provider_labels: dict[str, str | None],
    *,
    gold_label: str | None = None,
) -> ConsensusResult:
    """Majority vote across provider emotion labels with reliability tiers."""
    providers: list[str] = []
    labels: list[str] = []
    for provider, raw in provider_labels.items():
        norm = normalize_emotion_label(raw)
        if not is_votable_emotion(norm):
            continue
        providers.append(provider)
        labels.append(norm)

    n = len(labels)
    if n == 0:
        return ConsensusResult(reliability="NON_LABELLED", disagreement_flag=False)

    if n == 1:
        return ConsensusResult(
            providers=providers,
            labels=labels,
            consensus_label=labels[0],
            agreement_ratio=1.0,
            unanimous=True,
            reliability="HIGH",
            disagreement_flag=False,
            hallucination_proxy=_hallucination_proxy(labels, gold_label),
            n_valid_votes=1,
        )

    counts: dict[str, int] = {}
    for label in labels:
        key = label.lower()
        counts[key] = counts.get(key, 0) + 1

    winner_key, winner_count = max(counts.items(), key=lambda item: item[1])
    agreement_ratio = winner_count / n
    unanimous = len(counts) == 1
    unique_labels = {label.lower() for label in labels}
    disagreement_flag = len(unique_labels) > 1

    if winner_count <= (n / 2):
        reliability: ReliabilityTier = "SPLIT"
        consensus_label = None
    elif unanimous:
        reliability = "HIGH"
        consensus_label = _label_for_key(labels, winner_key)
    else:
        reliability = "LOW"
        consensus_label = _label_for_key(labels, winner_key)

    return ConsensusResult(
        providers=providers,
        labels=labels,
        consensus_label=consensus_label,
        agreement_ratio=round(agreement_ratio, 4),
        unanimous=unanimous,
        reliability=reliability,
        disagreement_flag=disagreement_flag,
        hallucination_proxy=_hallucination_proxy(labels, gold_label),
        n_valid_votes=n,
    )


def _label_for_key(labels: list[str], key: str) -> str:
    for label in labels:
        if label.lower() == key:
            return label
    return key


def _hallucination_proxy(labels: list[str], gold_label: str | None) -> bool | None:
    """Disagreement-based proxy: providers disagree and at least one mismatches gold."""
    gold = normalize_emotion_label(gold_label)
    if not gold or not labels:
        return None
    if len({label.lower() for label in labels}) <= 1:
        return False
    return any(label.lower() != gold.lower() for label in labels)


def emotion_matches_gold(predicted: str | None, gold: str | None) -> bool | None:
    pred = normalize_emotion_label(predicted)
    g = normalize_emotion_label(gold)
    if not g:
        return None
    if not pred:
        return False
    return pred.lower() == g.lower()
