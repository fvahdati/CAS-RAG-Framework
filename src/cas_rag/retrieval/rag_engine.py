"""Vector retrieval over video script corpus."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class RAGEngine:
    """Lightweight dense retrieval (sentence-transformers when available)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", top_k: int = 3):
        self.model_name = model_name
        self.top_k = top_k
        self._model = None
        self._chunks: list[str] = []
        self._embeddings: np.ndarray | None = None

    def _ensure_model(self):
        if self._model is not None:
            return
        from cas_rag.config.settings import get_settings

        if get_settings().offline_mode:
            self._model = False
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        except Exception:
            self._model = False  # offline fallback

    def load_corpus(self, path: Path | str) -> None:
        text = Path(path).read_text(encoding="utf-8")
        self._chunks = [c.strip() for c in text.split("\n") if c.strip()]
        self._ensure_model()
        if self._model not in (None, False) and self._chunks:
            self._embeddings = self._model.encode(self._chunks, normalize_embeddings=True)
        else:
            self._embeddings = None

    def retrieve(self, query: str) -> str:
        if not self._chunks:
            return ""
        self._ensure_model()
        if self._model in (None, False) or self._embeddings is None:
            return "\n".join(self._chunks[: self.top_k])
        q = self._model.encode([query], normalize_embeddings=True)[0]
        scores = self._embeddings @ q
        idx = np.argsort(scores)[::-1][: self.top_k]
        return "\n".join(self._chunks[i] for i in idx)
