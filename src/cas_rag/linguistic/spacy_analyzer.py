"""spaCy-based linguistic analysis for reference and grounding checks."""

from __future__ import annotations

import re

PRONOUNS = {"this", "that", "it", "they", "he", "she", "him", "her", "them"}


class SpacyLinguisticAnalyzer:
    def __init__(self, model: str = "en_core_web_sm"):
        self.model_name = model
        self._nlp = None

    def _ensure_nlp(self):
        if self._nlp is None:
            try:
                import spacy

                self._nlp = spacy.load(self.model_name)
            except OSError:
                self._nlp = None

    def needs_rag(self, comment: str, scenario: str) -> bool:
        """Algorithm 2/3: unresolved person-like reference."""
        self._ensure_nlp()
        if self._nlp is None:
            return bool(re.search(r"\b(she|he|they|it|that|this)\b", comment, re.I))
        doc = self._nlp(comment)
        for tok in doc:
            if tok.text.lower() in PRONOUNS:
                return True
        for chunk in doc.noun_chunks:
            if chunk.root.text.lower() in ("guy", "kid", "lady", "person", "child"):
                if chunk.root.text.lower() not in scenario.lower():
                    return True
        return False

    def can_ground_with_script(self, comment: str, scenario: str, script: str) -> bool:
        """CAS-RAG context-dependent: resolvable using video script."""
        if self.needs_rag(comment, scenario):
            comment_lemmas = set(re.findall(r"[a-z]{3,}", comment.lower()))
            script_lemmas = set(re.findall(r"[a-z]{3,}", (scenario + " " + script).lower()))
            overlap = comment_lemmas & script_lemmas
            temporal = bool(re.search(r"\b(when|that part|then)\b", comment, re.I))
            return len(overlap) >= 2 or temporal
        return True
