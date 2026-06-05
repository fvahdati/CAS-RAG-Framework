"""Semantic similarity: Explicit / Vague / Context-Dependent (thesis Layer 1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cas_rag.config.settings import get_settings
from cas_rag.core.enums import SemanticCategory


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
        self.prototypes = {k: data[k] for k in ("explicit", "vague", "context_dependent")}
        self._model = None
        self._proto_emb: dict[str, np.ndarray] | None = None
        yaml_cfg = settings.load_yaml()
        sem = yaml_cfg.get("semantic", {})
        self.margin = sem.get("margin_explicit_vague", 0.12)
        self.band = sem.get("band_explicit_vague", 0.08)

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

    def _heuristic_scores(self, comment: str) -> dict[str, float]:
        lower = comment.lower()
        vague_cues = ("uneasy", "mixed", "not sure", "wow", "amazing", "hard to tell", "performative")
        ctx_cues = ("harper", "aubrey", "waitress", "when ", "that part", "tutor", "kid's", "wwyd")
        scores = {"explicit": 0.5, "vague": 0.3, "context_dependent": 0.3}
        if any(c in lower for c in vague_cues):
            scores["vague"] = 0.85
            scores["explicit"] = 0.2
        elif any(c in lower for c in ctx_cues):
            scores["context_dependent"] = 0.85
            scores["explicit"] = 0.4
        else:
            scores["explicit"] = 0.75
        return scores

    def classify_three_way(self, comment: str) -> tuple[SemanticCategory, dict[str, float]]:
        self._ensure_encoder()
        if self._model is False:
            scores = self._heuristic_scores(comment)
        else:
            c = self._model.encode([comment], normalize_embeddings=True)[0]
            scores = {cat: self._metric_scores(c, self._proto_emb[cat]) for cat in self._proto_emb}
        winner = max(scores, key=scores.get)
        cat_map = {
            "explicit": SemanticCategory.EXPLICIT,
            "vague": SemanticCategory.VAGUE,
            "context_dependent": SemanticCategory.CONTEXT_DEPENDENT,
        }
        category = cat_map[winner]
        s_exp, s_vag = scores["explicit"], scores["vague"]
        if category == SemanticCategory.VAGUE and s_exp >= s_vag - self.band:
            category = SemanticCategory.EXPLICIT
        return category, scores

    def classify_two_way(self, comment: str) -> str:
        cat, _ = self.classify_three_way(comment)
        if cat == SemanticCategory.VAGUE:
            return "Vague"
        return "Explicit"
