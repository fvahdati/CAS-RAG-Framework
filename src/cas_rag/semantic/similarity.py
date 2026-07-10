"""Semantic similarity: Explicit / Ambiguous / Context-Dependent (thesis Layer 1)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from cas_rag.config.settings import get_settings
from cas_rag.core.enums import SemanticCategory


def _load_prototype_groups(data: dict) -> dict[str, list[str]]:
    """Support legacy JSON key ``vague`` and current ``ambiguous``."""
    ambiguous = data.get("ambiguous") or data.get("vague") or []
    return {
        "explicit": data["explicit"],
        "ambiguous": ambiguous,
        "context_dependent": data["context_dependent"],
    }


_GENERIC_TOKENS = frozenset(
    {"wow", "amazing", "love", "cute", "sweet", "good", "nice", "great", "awesome", "lol", "wholesome"}
)

_EXPLICIT_TOPIC_CUES = (
    "kids",
    "kid",
    "tutor",
    "bullying",
    "kindness",
    "empathy",
    "parents",
    "struggling student",
    "classroom",
    "help that",
    "stood up",
    "faith",
)


def _has_explicit_topic(comment: str) -> bool:
    lower = comment.lower()
    return any(cue in lower for cue in _EXPLICIT_TOPIC_CUES)


class SemanticSimilarityEngine:
    """Embedding-based three-way or two-way prototype classification."""

    WEIGHTS = {
        "cosine": 0.22,
        "euclidean": 0.14,
        "dot": 0.12,
        "manhattan": 0.10,
        "dice": 0.18,
        "pearson": 0.14,
        "exp_euclidean": 0.10,
    }

    def __init__(self, prototypes_path: Path | None = None, model_name: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        path = prototypes_path or (settings.project_root / "resources" / "prototypes" / "prototypes.json")
        data = json.loads(path.read_text())
        self.prototypes = _load_prototype_groups(data)
        self._model = None
        self._proto_emb: dict[str, np.ndarray] | None = None
        yaml_cfg = settings.load_yaml()
        sem = yaml_cfg.get("semantic", {})
        self.margin = sem.get("margin_explicit_vague", sem.get("margin_explicit_ambiguous", 0.10))

    def _ensure_encoder(self):
        if self._model is not None or self._proto_emb is not None:
            return
        if get_settings().offline_mode:
            self._model = False
            self._proto_emb = {}
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._proto_emb = {}
            for cat, texts in self.prototypes.items():
                self._proto_emb[cat] = self._model.encode(texts, normalize_embeddings=True)
        except Exception:
            self._model = False  # sentinel: use heuristic fallback
            self._proto_emb = {}

    def _metric_scores(self, c_emb: np.ndarray, p_emb: np.ndarray) -> float:
        cos = float(np.dot(c_emb, p_emb.mean(axis=0)))
        euc = 1.0 / (1.0 + float(np.linalg.norm(c_emb - p_emb.mean(axis=0))))
        dot = float(np.dot(c_emb, p_emb.mean(axis=0)))
        man = 1.0 / (1.0 + float(np.sum(np.abs(c_emb - p_emb.mean(axis=0)))))
        dice = 2 * cos / (1 + cos + 1e-9)
        pear = cos
        exp_euc = float(np.exp(-np.linalg.norm(c_emb - p_emb.mean(axis=0))))
        parts = {
            "cosine": cos,
            "euclidean": euc,
            "dot": dot,
            "manhattan": man,
            "dice": dice,
            "pearson": pear,
            "exp_euclidean": exp_euc,
        }
        return sum(self.WEIGHTS[k] * parts[k] for k in self.WEIGHTS)

    def _looks_generic(self, comment: str) -> bool:
        lower = comment.lower().strip()
        tokens = re.findall(r"[a-z']+", lower)
        if len(tokens) > 8 and _has_explicit_topic(comment):
            return False
        if len(tokens) <= 5 and any(t in _GENERIC_TOKENS for t in tokens):
            return True
        return lower in {"i love it", "so good", "wow", "amazing", "love this", "great video"}

    def _heuristic_scores(self, comment: str) -> dict[str, float]:
        lower = comment.lower()
        ambiguous_cues = (
            "uneasy",
            "mixed feelings",
            "not sure",
            "hard to tell",
            "performative",
            "staged",
            "scripted",
            "hidden camera",
            "editing",
        )
        ctx_cues = (
            "harper",
            "aubrey",
            "waitress",
            "when ",
            "that part",
            "that moment",
            "that scene",
            "the tutor",
            "that kid",
            "wwyd",
            "tick tock",
        )
        scores = {"explicit": 0.5, "ambiguous": 0.3, "context_dependent": 0.3}
        if self._looks_generic(comment):
            scores["ambiguous"] = 0.90
            scores["explicit"] = 0.25
        elif any(c in lower for c in ambiguous_cues):
            scores["ambiguous"] = 0.85
            scores["explicit"] = 0.2
        elif any(c in lower for c in ctx_cues):
            scores["context_dependent"] = 0.85
            scores["explicit"] = 0.4
        else:
            scores["explicit"] = 0.75
        return scores

    def _apply_linguistic_cues(self, comment: str, scores: dict[str, float]) -> dict[str, float]:
        adjusted = dict(scores)
        lower = comment.lower()
        ambiguous_cues = (
            "uneasy",
            "mixed feelings",
            "not sure",
            "hard to tell",
            "performative",
            "staged",
            "scripted",
            "hidden camera",
            "editing",
        )
        ctx_cues = (
            "harper",
            "aubrey",
            "waitress",
            "when ",
            "that part",
            "that moment",
            "that scene",
            "the tutor",
            "that kid",
            "wwyd",
            "tick tock",
        )

        if self._looks_generic(comment):
            adjusted["ambiguous"] = max(adjusted.get("ambiguous", 0.0), 0.90)
            adjusted["explicit"] = min(adjusted.get("explicit", 0.0), 0.35)

        if any(c in lower for c in ambiguous_cues):
            adjusted["ambiguous"] = max(adjusted.get("ambiguous", 0.0), 0.85)

        if any(c in lower for c in ctx_cues):
            adjusted["context_dependent"] = max(adjusted.get("context_dependent", 0.0), 0.80)

        if _has_explicit_topic(comment) and not self._looks_generic(comment):
            adjusted["explicit"] = max(adjusted.get("explicit", 0.0), 0.82)
            adjusted["ambiguous"] = min(adjusted.get("ambiguous", 0.0), 0.40)

        return adjusted

    def _top_two_gap(self, scores: dict[str, float]) -> tuple[str, str, float]:
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_key, top_score = ordered[0]
        second_key, second_score = ordered[1]
        return top_key, second_key, top_score - second_score

    def classify_three_way(self, comment: str) -> tuple[SemanticCategory, dict[str, float], bool]:
        self._ensure_encoder()
        if self._model is False:
            scores = self._heuristic_scores(comment)
        else:
            c = self._model.encode([comment], normalize_embeddings=True)[0]
            scores = {cat: self._metric_scores(c, self._proto_emb[cat]) for cat in self._proto_emb}

        scores = self._apply_linguistic_cues(comment, scores)
        top_key, second_key, gap = self._top_two_gap(scores)
        borderline = gap < self.margin

        cat_map = {
            "explicit": SemanticCategory.EXPLICIT,
            "ambiguous": SemanticCategory.AMBIGUOUS,
            "context_dependent": SemanticCategory.CONTEXT_DEPENDENT,
        }
        category = cat_map[top_key]

        if borderline and {top_key, second_key} == {"explicit", "ambiguous"}:
            if self._looks_generic(comment) or top_key == "ambiguous":
                category = SemanticCategory.AMBIGUOUS

        return category, scores, borderline

    def classify_two_way(self, comment: str) -> str:
        cat, _, _ = self.classify_three_way(comment)
        if cat == SemanticCategory.AMBIGUOUS:
            return "Ambiguous"
        return "Explicit"
