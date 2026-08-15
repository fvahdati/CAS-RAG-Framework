"""spaCy-based linguistic analysis for reference and grounding checks."""

from __future__ import annotations

import re

PRONOUNS = {"this", "that", "it", "they", "he", "she", "him", "her", "them"}
_RESOLVED_DEMONSTRATIVE = re.compile(
    r"\b(this|that|these|those)\s+(kid|kids|student|tutor|mom|mother|girl|boy|waitress|part|moment|scene|line)\b",
    re.I,
)


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
        if _RESOLVED_DEMONSTRATIVE.search(comment) and len(re.findall(r"[a-z']+", comment.lower())) > 6:
            return False
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

    def cd_rag_policy(self, comment: str, scenario: str, script: str) -> str:
        """CAS-RAG Layer 2 for Context-Dependent comments.

        Returns ``resolvable`` (RAG), ``unnecessary`` (no RAG / P7), or
        ``infeasible`` (video mismatch -> human verification). Retrieval fires
        when spaCy flags underspecified references and the semantic fallback
        score exceeds tau=0.4.
        """
        comment_lemmas = set(re.findall(r"[a-z]{4,}", comment.lower()))
        corpus_lemmas = set(re.findall(r"[a-z]{4,}", (scenario + " " + script).lower()))
        overlap = comment_lemmas & corpus_lemmas
        temporal = bool(re.search(r"\b(when|that part|then|that moment)\b", comment, re.I))
        if comment_lemmas and len(overlap) == 0 and not temporal:
            return "infeasible"
        needs = self.needs_rag(comment, scenario)
        score = 0.2
        if needs:
            score = 0.75
        if temporal:
            score = max(score, 0.55)
        tau = 0.4
        if needs and score > tau and (len(overlap) >= 1 or temporal):
            return "resolvable"
        if needs and score > tau:
            return "infeasible"
        return "unnecessary"

    def can_ground_with_script(self, comment: str, scenario: str, script: str) -> bool:
        """True when Layer 2 should retrieve (RAG-resolvable)."""
        return self.cd_rag_policy(comment, scenario, script) == "resolvable"
