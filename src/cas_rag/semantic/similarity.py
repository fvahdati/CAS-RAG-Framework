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
    {"wow", "amazing", "love", "cute", "good", "nice", "great", "awesome", "lol", "wholesome"}
)

_UNCERTAINTY_CUES = (
    "not sure",
    "hard to tell",
    "unclear",
    "maybe",
    "perhaps",
    "might be",
    "could be",
    "mixed feelings",
    "i guess",
    "seems like",
)

_PRODUCTION_DOUBT_CUES = (
    "staged",
    "scripted",
    "performative",
    "fake",
    "edited",
    "editing",
    "acted",
    "acting",
    "setup",
)

_CONTEXT_REFERENCE_CUES = (
    "when that",
    "that moment",
    "that part",
    "this part",
    "that scene",
    "in that scene",
    "at that point",
    "the interaction",
    "the situation",
    "everyone reacted",
    "if i had",
    "who goes",
    "used to",
    "reminds me",
    "that line",
)

_REPORTING_VERBS = ("said", "told", "asked", "replied", "was like", "goes", "went")
_ROLE_WORDS = ("teacher", "student", "kid", "child", "class", "tutor", "person", "someone")
_OFF_TOPIC_ANALOGY_CUES = (
    "on the show",
    "in that show",
    "in that movie",
    "like in",
    "reminds me of",
    "what about",
    "at their wedding",
)


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

    def _is_low_information_comment(self, comment: str) -> bool:
        lower = comment.lower().strip()
        tokens = re.findall(r"[a-z']+", lower)
        has_generic_token = any(t in _GENERIC_TOKENS for t in tokens)
        has_uncertainty = any(cue in lower for cue in _UNCERTAINTY_CUES)
        if len(tokens) <= 4 and (has_generic_token or has_uncertainty):
            return True
        # Repeated character noise like "woooowww" is usually low-information.
        if re.search(r"([a-z])\1{4,}", lower):
            return True
        return False

    def _is_timestamp_only_reaction(self, comment: str) -> bool:
        if not re.search(r"\b\d+:\d+\b", comment):
            return False
        remainder = re.sub(r"\b\d+:\d+\b", "", comment).strip().lower()
        tokens = re.findall(r"[a-z']+", remainder)
        # Keep this intentionally conservative to avoid over-triggering.
        return len(tokens) <= 6

    def _is_uncertain_or_doubtful(self, comment: str) -> bool:
        lower = comment.lower()
        if any(cue in lower for cue in _UNCERTAINTY_CUES):
            return True
        return any(cue in lower for cue in _PRODUCTION_DOUBT_CUES)

    def _is_context_reference(self, comment: str) -> bool:
        lower = comment.lower()
        has_context_cue = any(cue in lower for cue in _CONTEXT_REFERENCE_CUES)
        tokens = re.findall(r"[a-z']+", lower)
        has_deictic = bool(re.search(r"\b(when|while|then|that|this|there)\b", lower))
        has_event_word = bool(re.search(r"\b(moment|scene|interaction|part|situation)\b", lower))
        has_role = any(w in lower for w in _ROLE_WORDS)
        # Quoted/reported speech usually requires external scene context.
        has_quote = '"' in comment or "“" in comment or "”" in comment
        has_reporting = any(v in lower for v in _REPORTING_VERBS)
        if has_quote and (has_reporting or has_role):
            return True
        if has_context_cue and (has_event_word or has_role or has_reporting):
            return True
        if has_deictic and has_event_word and has_role and len(tokens) >= 10:
            return True
        return False

    def _is_entity_unresolved(self, comment: str) -> bool:
        lower = comment.lower()
        pronouns = re.findall(r"\b(he|she|they|him|her|them|that one|this one)\b", lower)
        return len(pronouns) >= 1 and len(re.findall(r"[a-z']+", lower)) >= 8

    def _is_personal_anecdote_contextual(self, comment: str) -> bool:
        lower = comment.lower()
        has_first_person = bool(re.search(r"\b(i|my|me|we|our)\b", lower))
        has_past = bool(re.search(r"\b(was|were|had|did|used to|remember|happened)\b", lower))
        has_role = any(w in lower for w in _ROLE_WORDS)
        return has_first_person and has_past and has_role

    def _is_off_topic_reference(self, comment: str) -> bool:
        lower = comment.lower()
        return any(cue in lower for cue in _OFF_TOPIC_ANALOGY_CUES)

    def _is_ambiguous_signal(self, comment: str) -> bool:
        return (
            self._is_uncertain_or_doubtful(comment)
            or self._is_low_information_comment(comment)
            or self._is_timestamp_only_reaction(comment)
            or self._is_off_topic_reference(comment)
        )

    def _looks_generic(self, comment: str) -> bool:
        return self._is_low_information_comment(comment)

    def _heuristic_scores(self, comment: str) -> dict[str, float]:
        scores = {"explicit": 0.5, "ambiguous": 0.3, "context_dependent": 0.3}
        if self._is_ambiguous_signal(comment):
            scores["ambiguous"] = 0.90
            scores["explicit"] = 0.25
        else:
            scores["explicit"] = 0.75
        return scores

    def _apply_linguistic_cues(self, comment: str, scores: dict[str, float]) -> dict[str, float]:
        adjusted = dict(scores)

        if self._is_uncertain_or_doubtful(comment):
            adjusted["ambiguous"] = max(adjusted.get("ambiguous", 0.0), 0.85)
            adjusted["explicit"] = min(adjusted.get("explicit", 0.0), 0.45)
        if self._is_context_reference(comment):
            adjusted["context_dependent"] = max(adjusted.get("context_dependent", 0.0), 0.82)
        if self._is_entity_unresolved(comment):
            adjusted["context_dependent"] = max(adjusted.get("context_dependent", 0.0), 0.76)
        if self._is_personal_anecdote_contextual(comment):
            adjusted["context_dependent"] = max(adjusted.get("context_dependent", 0.0), 0.84)
        if self._is_low_information_comment(comment):
            adjusted["ambiguous"] = max(adjusted.get("ambiguous", 0.0), 0.82)
            adjusted["explicit"] = min(adjusted.get("explicit", 0.0), 0.50)
        if self._is_timestamp_only_reaction(comment):
            adjusted["ambiguous"] = max(adjusted.get("ambiguous", 0.0), 0.78)
        if self._is_off_topic_reference(comment):
            adjusted["ambiguous"] = max(adjusted.get("ambiguous", 0.0), 0.84)
            adjusted["explicit"] = min(adjusted.get("explicit", 0.0), 0.45)

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
            if self._is_ambiguous_signal(comment) or top_key == "ambiguous":
                category = SemanticCategory.AMBIGUOUS

        return category, scores, borderline

    def classify_two_way(self, comment: str) -> str:
        cat, _, _ = self.classify_three_way(comment)
        if cat == SemanticCategory.AMBIGUOUS:
            return "Ambiguous"
        return "Explicit"
