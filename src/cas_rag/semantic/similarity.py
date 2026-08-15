"""Semantic similarity: Explicit / Ambiguous / Context-Dependent (thesis Layer 1).

Layer 1 uses SentenceTransformer embeddings and a seven-metric weighted fusion.
No dataset-specific phrase lists are applied on the production path.
Margins m=0.12, b=0.08 follow config/default.yaml and Chapter 5.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cas_rag.config.settings import get_settings
from cas_rag.core.enums import SemanticCategory


def _load_prototype_groups(data: dict) -> dict[str, list[str]]:
    """Support legacy JSON key ``vague`` and thesis key ``ambiguous``."""
    ambiguous = data.get("ambiguous") or data.get("vague") or []
    return {
        "explicit": data["explicit"],
        "ambiguous": ambiguous,
        "context_dependent": data["context_dependent"],
    }


class SemanticSimilarityEngine:
    """Embedding-based three-way (CAS-RAG) or two-way (Algorithms 3--4) classification."""

    DEFAULT_WEIGHTS = {
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
        emb = yaml_cfg.get("embedding", {})
        self.WEIGHTS = dict(emb.get("metric_weights") or self.DEFAULT_WEIGHTS)
        self.margin_m = float(sem.get("margin_m", 0.12))
        self.margin_b = float(sem.get("margin_b", 0.08))
        self.margin = self.margin_b

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
            self._model = False
            self._proto_emb = {}

    def _metric_scores(self, c_emb: np.ndarray, p_emb: np.ndarray) -> float:
        p_mean = p_emb.mean(axis=0)
        cos = float(np.dot(c_emb, p_mean))
        euc = 1.0 / (1.0 + float(np.linalg.norm(c_emb - p_mean)))
        dot = float(np.dot(c_emb, p_mean))
        man = 1.0 / (1.0 + float(np.sum(np.abs(c_emb - p_mean))))
        dice = 2 * cos / (1 + cos + 1e-9)
        pear = cos
        if c_emb.size > 1:
            corr = np.corrcoef(c_emb, p_mean)[0, 1]
            if np.isfinite(corr):
                pear = float(corr)
        exp_euc = float(np.exp(-np.linalg.norm(c_emb - p_mean)))
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

    def _prototype_overlap_scores(self, comment: str) -> dict[str, float]:
        """Offline fallback: bag-of-words overlap with prototype sentences only."""
        cl = set(comment.lower().split())
        scores: dict[str, float] = {}
        for cat, texts in self.prototypes.items():
            proto = set(" ".join(texts).lower().split())
            scores[cat] = len(cl & proto) / max(len(cl | proto), 1)
        return scores

    def _raw_scores(self, comment: str) -> dict[str, float]:
        self._ensure_encoder()
        if self._model is False:
            return self._prototype_overlap_scores(comment)
        c = self._model.encode([comment], normalize_embeddings=True)[0]
        return {cat: self._metric_scores(c, self._proto_emb[cat]) for cat in self._proto_emb}

    def apply_margin_rules(self, scores: dict[str, float]) -> tuple[SemanticCategory, bool]:
        """Thesis Layer 1 post-processing: argmax then Explicit-Ambiguous margins m, b."""
        s_exp = scores.get("explicit", 0.0)
        s_amb = scores.get("ambiguous", 0.0)
        s_cd = scores.get("context_dependent", 0.0)
        ranked = sorted(
            (
                ("explicit", s_exp),
                ("ambiguous", s_amb),
                ("context_dependent", s_cd),
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        k0 = ranked[0][0]
        delta = s_amb - s_exp
        m, b = self.margin_m, self.margin_b
        key = k0
        if k0 == "explicit" and s_amb > s_exp + m:
            key = "ambiguous"
        elif k0 == "ambiguous" and s_exp >= s_amb - b:
            key = "explicit"
        cat_map = {
            "explicit": SemanticCategory.EXPLICIT,
            "ambiguous": SemanticCategory.AMBIGUOUS,
            "context_dependent": SemanticCategory.CONTEXT_DEPENDENT,
        }
        borderline = abs(delta) <= b or (ranked[0][1] - ranked[1][1]) < b
        return cat_map[key], borderline

    def classify_three_way(self, comment: str) -> tuple[SemanticCategory, dict[str, float], bool]:
        scores = self._raw_scores(comment)
        category, borderline = self.apply_margin_rules(scores)
        return category, scores, borderline

    def classify_two_way(self, comment: str) -> str:
        """Algorithms 3--4: Explicit vs Ambiguous prototypes only (no Context-Dependent class)."""
        scores = self._raw_scores(comment)
        if scores.get("ambiguous", 0.0) > scores.get("explicit", 0.0):
            return "Ambiguous"
        return "Explicit"
