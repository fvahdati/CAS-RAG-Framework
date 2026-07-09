"""Post-hoc failure analysis for prompt iteration (no inference changes)."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cas_rag.core.enums import AlgorithmId, EmotionLabel, PipelineStatus, PromptId, SemanticCategory
from cas_rag.core.models import CommentInput, PipelineResult
from cas_rag.experiments.prompt_recommendations import (
    build_prompt_fix_guide,
    build_prompt_fix_recommendation,
    explain_why_misclassified,
    recommendation_to_row,
)
from cas_rag.semantic.similarity import SemanticSimilarityEngine

ALGORITHM_SHORT = {
    AlgorithmId.ALGORITHM_1: "Alg1",
    AlgorithmId.ALGORITHM_2: "Alg2",
    AlgorithmId.ALGORITHM_3: "Alg3",
    AlgorithmId.ALGORITHM_4: "Alg4",
    AlgorithmId.CAS_RAG: "CAS-RAG",
}

SHORT_TO_ALGORITHM = {v: k for k, v in ALGORITHM_SHORT.items()}
SHORT_TO_ALGORITHM.update({k.value: k for k in AlgorithmId})


def normalize_category(value: str | None) -> str:
    if value is None or str(value).strip().lower() in {"", "nan", "none"}:
        return ""
    s = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if s == "vague":
        return "ambiguous"
    return s


def normalize_emotion(value: str | None) -> str:
    if value is None or str(value).strip().lower() in {"", "nan", "none"}:
        return ""
    return str(value).strip()


def resolve_gold_emotion(row: dict[str, Any]) -> str:
    for col in ("gold_label", "final_label", "expected_emotion", "emotion_label"):
        if col in row and normalize_emotion(row.get(col)):
            return normalize_emotion(row[col])
    return ""


def resolve_gold_category(row: dict[str, Any]) -> str:
    for col in ("expected_category", "gold_category", "routing_label"):
        if col in row and normalize_category(row.get(col)):
            return normalize_category(row[col])
    return ""


def expected_rag_used(algorithm: AlgorithmId, expected_category: str) -> bool | None:
    """Heuristic gold RAG expectation for Alg1/Alg2-style routing experiments."""
    cat = normalize_category(expected_category)
    if not cat:
        return None
    if cat == "context_dependent":
        return True
    if cat in ("explicit", "ambiguous"):
        return False
    return None


def _result_emotion(result: PipelineResult) -> str:
    if result.status == PipelineStatus.DISCARDED:
        return EmotionLabel.DISCARDED.value
    if result.pending_human:
        return "PENDING_HUMAN"
    if result.emotion_label:
        return result.emotion_label.value
    return ""


def _prompts_used(result: PipelineResult) -> str:
    if not result.prompts_invoked:
        return ""
    return ", ".join(p.value for p in result.prompts_invoked)


def _routing_path(result: PipelineResult) -> str:
    if result.trace:
        return " -> ".join(result.trace)
    parts: list[str] = []
    if result.rag_used:
        parts.append("RAG")
    if result.prompts_invoked:
        parts.append(_prompts_used(result))
    if result.semantic_category:
        parts.append(f"semantic={result.semantic_category.value}")
    return " -> ".join(parts)


def _parse_scores_from_trace(trace: list[str]) -> dict[str, float]:
    for item in trace:
        if not item.startswith("scores="):
            continue
        payload = item.split("=", 1)[1].strip()
        try:
            parsed = ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            continue
        if isinstance(parsed, dict):
            return {str(k): float(v) for k, v in parsed.items()}
    return {}


@dataclass
class SemanticConfidence:
    category: str = ""
    top_score: float | None = None
    margin: float | None = None
    borderline: bool = False
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def confidence(self) -> float | None:
        """Routing confidence proxy: margin between top-1 and top-2 semantic scores."""
        return self.margin


def compute_semantic_confidence(
    comment: str,
    *,
    prototypes_path: Path | None = None,
    trace: list[str] | None = None,
) -> SemanticConfidence:
    trace_scores = _parse_scores_from_trace(trace or [])
    if trace_scores:
        ordered = sorted(trace_scores.items(), key=lambda item: item[1], reverse=True)
        margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else None
        winner = ordered[0][0]
        cat_map = {
            "explicit": SemanticCategory.EXPLICIT.value,
            "ambiguous": SemanticCategory.AMBIGUOUS.value,
            "context_dependent": SemanticCategory.CONTEXT_DEPENDENT.value,
        }
        return SemanticConfidence(
            category=cat_map.get(winner, winner),
            top_score=round(ordered[0][1], 4),
            margin=round(margin, 4) if margin is not None else None,
            borderline=bool(margin is not None and margin < 0.10),
            scores=trace_scores,
        )

    engine = SemanticSimilarityEngine(prototypes_path=prototypes_path)
    category, scores, borderline = engine.classify_three_way(comment)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else None
    return SemanticConfidence(
        category=category.value,
        top_score=round(ordered[0][1], 4),
        margin=round(margin, 4) if margin is not None else None,
        borderline=borderline,
        scores={k: round(v, 4) for k, v in scores.items()},
    )


def infer_failure_reasons(
    *,
    algorithm: AlgorithmId,
    expected_emotion: str,
    expected_category: str,
    predicted_emotion: str,
    predicted_category: str,
    rag_used: bool,
    status: str,
    semantic_borderline: bool,
    comment_text: str,
) -> list[str]:
    reasons: list[str] = []

    if status == PipelineStatus.PENDING_HUMAN.value:
        reasons.append("pending_human_review")
    if status == PipelineStatus.DISCARDED.value:
        reasons.append("discarded_by_pipeline")

    exp_emotion = normalize_emotion(expected_emotion)
    pred_emotion = normalize_emotion(predicted_emotion)
    if exp_emotion and pred_emotion and exp_emotion.lower() != pred_emotion.lower():
        reasons.append("emotion_mismatch")
        pair = {exp_emotion, pred_emotion}
        if pair == {"Empathy", "Frustration"}:
            reasons.append("empathy_frustration_confusion")

    exp_cat = normalize_category(expected_category)
    pred_cat = normalize_category(predicted_category)
    if exp_cat and pred_cat and exp_cat != pred_cat:
        reasons.append("semantic_routing_mismatch")
        if exp_cat == "ambiguous" and pred_cat == "explicit":
            reasons.append("ambiguous_under_detected")
        if exp_cat == "context_dependent" and pred_cat == "explicit":
            reasons.append("context_dependent_under_detected")

    expected_rag = expected_rag_used(algorithm, exp_cat)
    if expected_rag is not None and rag_used != expected_rag:
        reasons.append("rag_detection_error")
        if expected_rag and not rag_used:
            reasons.append("rag_under_triggered")
        if not expected_rag and rag_used:
            reasons.append("rag_over_triggered")

    if semantic_borderline:
        reasons.append("borderline_semantic_routing")

    tokens = re.findall(r"[a-z']+", comment_text.lower())
    if len(tokens) <= 5 and any(t in {"wow", "amazing", "love", "cute", "sweet", "good"} for t in tokens):
        if exp_cat == "ambiguous" and pred_cat and pred_cat != "ambiguous":
            reasons.append("generic_comment_misrouted")

    if not reasons and exp_emotion and pred_emotion and exp_emotion.lower() == pred_emotion.lower():
        reasons.append("correct")

    return reasons


def build_evaluation_row(
    inp: CommentInput,
    result: PipelineResult,
    *,
    gold_row: dict[str, Any],
    prototypes_path: Path | None = None,
) -> dict[str, Any]:
    expected_emotion = resolve_gold_emotion(gold_row)
    expected_category = resolve_gold_category(gold_row)
    predicted_emotion = _result_emotion(result)
    predicted_category = normalize_category(
        result.semantic_category.value if result.semantic_category else ""
    )

    sem = compute_semantic_confidence(
        inp.comment,
        prototypes_path=prototypes_path,
        trace=result.trace,
    )
    if not predicted_category and sem.category:
        predicted_category = normalize_category(sem.category)

    reasons = infer_failure_reasons(
        algorithm=result.algorithm,
        expected_emotion=expected_emotion,
        expected_category=expected_category,
        predicted_emotion=predicted_emotion,
        predicted_category=predicted_category,
        rag_used=result.rag_used,
        status=result.status.value,
        semantic_borderline=sem.borderline,
        comment_text=inp.comment,
    )

    emotion_correct: bool | None = None
    if expected_emotion and predicted_emotion and predicted_emotion != "PENDING_HUMAN":
        emotion_correct = expected_emotion.lower() == predicted_emotion.lower()

    routing_correct: bool | None = None
    if expected_category:
        if result.algorithm == AlgorithmId.CAS_RAG:
            routing_correct = bool(predicted_category) and expected_category == predicted_category
        else:
            expected_rag = expected_rag_used(result.algorithm, expected_category)
            if expected_rag is not None:
                routing_correct = result.rag_used == expected_rag

    why_misclassified = explain_why_misclassified(
        expected_emotion=expected_emotion,
        predicted_emotion=predicted_emotion,
        expected_category=expected_category,
        predicted_category=predicted_category,
        rag_used=result.rag_used,
        emotion_correct=emotion_correct,
        routing_correct=routing_correct,
        reasons=reasons,
        algorithm=ALGORITHM_SHORT[result.algorithm],
    )
    fix = build_prompt_fix_recommendation(
        algorithm=result.algorithm,
        prompts_invoked=result.prompts_invoked,
        reasons=reasons,
        comment_text=inp.comment,
        expected_emotion=expected_emotion,
        predicted_emotion=predicted_emotion,
        expected_category=expected_category,
        predicted_category=predicted_category,
    )
    fix_fields = recommendation_to_row(fix)

    return {
        "comment_id": inp.comment_id,
        "comment_text": inp.comment,
        "algorithm": ALGORITHM_SHORT[result.algorithm],
        "expected_emotion": expected_emotion,
        "expected_category": expected_category,
        "predicted_emotion": predicted_emotion,
        "predicted_category": predicted_category,
        "emotion_correct": emotion_correct,
        "routing_correct": routing_correct,
        "why_misclassified": why_misclassified,
        "contributing_prompt": fix_fields["contributing_prompt"],
        "prompt_file": fix_fields["prompt_file"],
        "prompt_section_to_edit": fix_fields["prompt_section_to_edit"],
        "suggested_prompt_fix": fix_fields["suggested_prompt_fix"],
        "example_to_add": fix_fields["example_to_add"],
        "confidence": sem.confidence,
        "semantic_top_score": sem.top_score,
        "semantic_margin": sem.margin,
        "semantic_borderline": sem.borderline,
        "routing_path": _routing_path(result),
        "prompts_used": _prompts_used(result),
        "rag_used": result.rag_used,
        "status": result.status.value,
        "likely_failure_reason": "; ".join(reasons),
        "explanation_excerpt": (result.explanation or "")[:240],
    }


def build_failure_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reason_counts: dict[str, int] = {}
    prompt_counts: dict[str, int] = {}
    for row in rows:
        emotion_ok = row.get("emotion_correct") is not False
        routing_ok = row.get("routing_correct") is not False
        pending = "pending_human_review" in str(row.get("likely_failure_reason", ""))
        if emotion_ok and routing_ok and not pending:
            continue
        for reason in str(row.get("likely_failure_reason", "")).split("; "):
            reason = reason.strip()
            if reason and reason != "correct":
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        prompt = str(row.get("contributing_prompt", "")).strip()
        if prompt:
            prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1

    summary = [
        {"metric_type": "failure_reason", "key": reason, "count": count}
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    summary.extend(
        {"metric_type": "contributing_prompt", "key": prompt, "count": count}
        for prompt, count in sorted(prompt_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    if not summary:
        summary.append({"metric_type": "failure_reason", "key": "none", "count": 0})
    return summary


def build_algorithm_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_alg: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_alg.setdefault(row["algorithm"], []).append(row)

    out: list[dict[str, Any]] = []
    for alg, alg_rows in sorted(by_alg.items()):
        n = len(alg_rows)
        emotion_gold = [r for r in alg_rows if r.get("expected_emotion")]
        routing_gold = [r for r in alg_rows if r.get("expected_category")]
        out.append(
            {
                "algorithm": alg,
                "n_comments": n,
                "emotion_accuracy": round(
                    sum(1 for r in emotion_gold if r.get("emotion_correct")) / len(emotion_gold), 4
                )
                if emotion_gold
                else None,
                "routing_accuracy": round(
                    sum(1 for r in routing_gold if r.get("routing_correct")) / len(routing_gold), 4
                )
                if routing_gold
                else None,
                "failures": sum(
                    1
                    for r in alg_rows
                    if r.get("emotion_correct") is False or r.get("routing_correct") is False
                ),
            }
        )
    return out


def build_iteration_recommendations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate detailed prompt-fix guidance by algorithm and failure reason."""
    recs: list[dict[str, Any]] = []
    failing = [
        r
        for r in rows
        if r.get("emotion_correct") is False
        or r.get("routing_correct") is False
        or "pending_human_review" in str(r.get("likely_failure_reason", ""))
    ]
    seen: set[tuple[str, str, str]] = set()
    for row in failing:
        for reason in str(row.get("likely_failure_reason", "")).split("; "):
            reason = reason.strip()
            if not reason or reason == "correct":
                continue
            key = (row["algorithm"], reason, row.get("contributing_prompt", ""))
            if key in seen:
                continue
            seen.add(key)
            recs.append(
                {
                    "algorithm": row["algorithm"],
                    "failure_reason": reason,
                    "contributing_prompt": row.get("contributing_prompt", ""),
                    "prompt_file": row.get("prompt_file", ""),
                    "prompt_section_to_edit": row.get("prompt_section_to_edit", ""),
                    "suggested_prompt_fix": row.get("suggested_prompt_fix", ""),
                    "example_to_add": row.get("example_to_add", ""),
                    "example_comment_id": row["comment_id"],
                }
            )
    if not recs:
        recs.append(
            {
                "algorithm": "all",
                "failure_reason": "none",
                "contributing_prompt": "",
                "prompt_file": "",
                "prompt_section_to_edit": "",
                "suggested_prompt_fix": "No failures in this subset — expand to more comments or algorithms.",
                "example_to_add": "",
                "example_comment_id": "",
            }
        )
    return recs


__all__ = [
    "ALGORITHM_SHORT",
    "SHORT_TO_ALGORITHM",
    "build_algorithm_summary",
    "build_evaluation_row",
    "build_failure_summary",
    "build_iteration_recommendations",
    "build_prompt_fix_guide",
    "compute_semantic_confidence",
    "infer_failure_reasons",
    "normalize_category",
    "normalize_emotion",
]
