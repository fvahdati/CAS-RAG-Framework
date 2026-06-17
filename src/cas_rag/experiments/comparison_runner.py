"""Run Algorithms 1–4 and CAS-RAG on one dataset; export combined Excel with KPIs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from cas_rag.config.settings import get_settings
from cas_rag.core.enums import AlgorithmId, EmotionLabel, PipelineStatus
from cas_rag.core.models import CommentInput, HumanReviewDecision, PipelineResult
from cas_rag.experiments.llm_columns import (
    emotion_label_column,
    filter_base_dataframe_columns,
    provider_column_suffix,
    rag_used_column,
    routing_correct_column,
    secondary_emotion_label_column,
    semantic_category_column,
)
from cas_rag.llm.probe import resolve_providers_with_gemini_fallback
from cas_rag.llm.validation import validate_llm_provider
from cas_rag.orchestration.engine import PipelineEngine
from cas_rag.subsystems.human_verification import HumanVerificationService

DEFAULT_MULTI_LLM_PROVIDERS = ("openai", "gemini")

ALGORITHMS = [
    (AlgorithmId.ALGORITHM_1, "Alg1"),
    (AlgorithmId.ALGORITHM_2, "Alg2"),
    (AlgorithmId.ALGORITHM_3, "Alg3"),
    (AlgorithmId.ALGORITHM_4, "Alg4"),
    (AlgorithmId.CAS_RAG, "CAS-RAG"),
]


def load_experiment_comments(path: Path) -> tuple[pd.DataFrame, list[CommentInput]]:
    """Load comments from Excel or CSV."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    text_col = "comment_text" if "comment_text" in df.columns else "comment"
    id_col = "comment_id" if "comment_id" in df.columns else None

    comments: list[CommentInput] = []
    for idx, row in df.iterrows():
        cid = str(row[id_col]) if id_col else str(idx)
        gold_emotion = None
        if "gold_label" in df.columns and pd.notna(row.get("gold_label")):
            try:
                gold_emotion = EmotionLabel(str(row["gold_label"]).strip())
            except ValueError:
                gold_emotion = None
        comments.append(
            CommentInput(
                comment_id=cid,
                comment=str(row[text_col]),
                gold_label=gold_emotion,
                metadata={"row_index": int(idx)},
            )
        )
    return df, comments


def _norm_category(value: str | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if s == "vague":
        s = "ambiguous"
    if s == "context_dependent":
        return "context_dependent"
    return s


def _extract_emotion_label_text(text: str) -> str:
    match = re.search(r"emotion\s*label\s*:\s*([^\n]+)", text, re.I)
    if match:
        return match.group(1).strip()
    return ""


def _result_label(result: PipelineResult) -> str:
    if result.status == PipelineStatus.DISCARDED:
        return "DISCARDED"
    if result.pending_human:
        return "PENDING_HUMAN"
    if result.emotion_label:
        return result.emotion_label.value
    if result.explanation:
        label_text = _extract_emotion_label_text(result.explanation)
        if label_text:
            return label_text.split("/")[0].strip()
    return ""


def _result_secondary_label(result: PipelineResult) -> str:
    if result.status in (PipelineStatus.DISCARDED,) or result.pending_human:
        return ""
    if result.secondary_emotion_label:
        return result.secondary_emotion_label.value
    if result.explanation:
        from cas_rag.llm.parser import parse_emotion_labels

        _, secondary = parse_emotion_labels(result.explanation)
        if secondary:
            return secondary.value
    return ""


def _semantic_label(result: PipelineResult) -> str:
    if result.semantic_category:
        return _norm_category(result.semantic_category.value)
    return ""


def auto_resolve_pending_human(
    human: HumanVerificationService,
    expected_by_comment: dict[str, str],
    default_emotion: str = "Empathy",
) -> int:
    """Resolve batch human queue using gold routing category (required human verification)."""
    resolved = 0
    for item in list(human.get_pending()):
        cid = item["comment_id"]
        exp = _norm_category(expected_by_comment.get(cid, ""))
        if exp in ("ambiguous", "vague"):
            decision = HumanReviewDecision(
                comment_id=cid,
                confirms_ambiguous=True,
                resolvable=False,
                assigned_label=None,
                reviewer_id="batch_gold_router",
                notes="Auto-resolved: gold expected_category=ambiguous",
            )
        else:
            try:
                label = EmotionLabel(default_emotion)
            except ValueError:
                label = EmotionLabel.NONE
            decision = HumanReviewDecision(
                comment_id=cid,
                confirms_ambiguous=False,
                resolvable=True,
                assigned_label=label,
                reviewer_id="batch_gold_router",
                notes=f"Auto-resolved: gold expected_category={exp or 'not_ambiguous'}",
            )
        human.submit_review(item["queue_id"], decision)
        resolved += 1
    return resolved


def run_algorithm_pass(
    algorithm: AlgorithmId,
    comments: list[CommentInput],
    video_slug: str,
    human_queue_path: Path,
    llm_provider: str | None,
    expected_by_comment: dict[str, str],
    max_passes: int = 2,
) -> list[PipelineResult]:
    """Run one algorithm; re-process pending_human after auto-resolution."""
    results_map: dict[str, PipelineResult] = {}
    for pass_num in range(max_passes):
        engine = PipelineEngine(
            algorithm,
            llm_provider=llm_provider,
            video_slug=video_slug,
            human_queue_path=human_queue_path,
        )
        for inp in comments:
            if pass_num == 0 or results_map[inp.comment_id].pending_human:
                results_map[inp.comment_id] = engine.process(inp)
        auto_resolve_pending_human(engine.ctx.human, expected_by_comment)
        if not engine.ctx.human.get_pending():
            break
    return [results_map[c.comment_id] for c in comments]


def compute_routing_kpi(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }


def _resolve_llm_providers(
    llm_provider: str | None,
    llm_providers: list[str] | None,
    *,
    gemini_fallback: str | None = "ollama",
) -> list[str]:
    if llm_providers:
        requested = llm_providers
    elif llm_provider:
        requested = [llm_provider]
    else:
        requested = list(DEFAULT_MULTI_LLM_PROVIDERS)

    if gemini_fallback and "gemini" in [_normalize_provider_name(p) for p in requested]:
        resolved, messages = resolve_providers_with_gemini_fallback(
            requested,
            fallback=gemini_fallback,
            settings=get_settings(),
        )
        for msg in messages:
            print(msg)
        return resolved
    return requested


def _normalize_provider_name(provider: str) -> str:
    name = provider.lower().strip()
    if name in ("llama", "meta"):
        return "ollama"
    return name


def _prepare_output_frame(df: pd.DataFrame) -> pd.DataFrame:
    keep = filter_base_dataframe_columns(list(df.columns))
    return df[keep].copy()


def _reset_human_queue(settings) -> Path:
    human_queue_path = settings.project_root / "data" / "human_verification_batch_run.json"
    human_queue_path.write_text(json.dumps({"pending": [], "completed": []}), encoding="utf-8")
    return human_queue_path


def run_five_algorithm_comparison(
    input_path: Path,
    output_path: Path,
    video_slug: str = "video_02_education_tutoring_diner",
    llm_provider: str | None = None,
    llm_providers: list[str] | None = None,
    include_kpi: bool = False,
    gemini_fallback: str | None = "ollama",
) -> Path:
    settings = get_settings()
    df, comments = load_experiment_comments(input_path)
    providers = _resolve_llm_providers(
        llm_provider,
        llm_providers,
        gemini_fallback=gemini_fallback,
    )
    print("LLM providers for this run:", ", ".join(providers))

    for provider in providers:
        validate_llm_provider(settings, provider)

    gold_col = "expected_category" if "expected_category" in df.columns else None
    expected_by_comment = {}
    if gold_col:
        for _, row in df.iterrows():
            cid = str(row["comment_id"])
            expected_by_comment[cid] = _norm_category(row[gold_col])

    out = _prepare_output_frame(df)
    all_results: dict[str, dict[str, list[PipelineResult]]] = {
        provider_column_suffix(p): {} for p in providers
    }

    for provider in providers:
        provider_key = provider_column_suffix(provider)
        human_queue_path = _reset_human_queue(settings)
        for alg_id, short in ALGORITHMS:
            results = run_algorithm_pass(
                alg_id,
                comments,
                video_slug=video_slug,
                human_queue_path=human_queue_path,
                llm_provider=provider,
                expected_by_comment=expected_by_comment,
            )
            all_results[provider_key][short] = results
            col = emotion_label_column(short, provider)
            out[col] = [_result_label(r) for r in results]
            sec_col = secondary_emotion_label_column(short, provider)
            out[sec_col] = [_result_secondary_label(r) for r in results]
            rag_col = rag_used_column(short, provider)
            out[rag_col] = [r.rag_used for r in results]
            if short == "CAS-RAG":
                sem_col = semantic_category_column(short, provider)
                out[sem_col] = [_semantic_label(r) for r in results]
                if gold_col:
                    route_col = routing_correct_column(short, provider)
                    out[route_col] = [
                        _norm_category(r.semantic_category.value if r.semantic_category else None)
                        == expected_by_comment.get(str(cid), "")
                        for r, cid in zip(results, out["comment_id"].astype(str))
                    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="Comparison", index=False)
        if include_kpi:
            kpi_rows = []
            if gold_col:
                for provider in providers:
                    provider_key = provider_column_suffix(provider)
                    sem_col = semantic_category_column("CAS-RAG", provider)
                    if sem_col not in out.columns:
                        continue
                    y_true = [_norm_category(v) for v in out[gold_col]]
                    cas_sem = out[sem_col].tolist()
                    mask = [bool(t) and bool(p) for t, p in zip(y_true, cas_sem)]
                    if any(mask):
                        yt = [y_true[i] for i in range(len(y_true)) if mask[i]]
                        yp = [cas_sem[i] for i in range(len(cas_sem)) if mask[i]]
                        m = compute_routing_kpi(yt, yp)
                        kpi_rows.append(
                            {
                                "metric_group": "CAS-RAG Layer-1 routing vs expected_category",
                                "llm_provider": provider,
                                "llm_column_suffix": provider_key,
                                "algorithm": "CAS-RAG",
                                "n": m["n"],
                                "accuracy": m["accuracy"],
                                "precision_macro": m["precision_macro"],
                                "recall_macro": m["recall_macro"],
                                "f1_macro": m["f1_macro"],
                            }
                        )
            for provider in providers:
                provider_key = provider_column_suffix(provider)
                for short, results in all_results[provider_key].items():
                    kpi_rows.append(
                        {
                            "metric_group": "Pipeline execution",
                            "llm_provider": provider,
                            "llm_column_suffix": provider_key,
                            "algorithm": short,
                            "n": len(results),
                            "labelled": sum(1 for r in results if r.status == PipelineStatus.SUCCESS),
                            "discarded": sum(1 for r in results if r.status == PipelineStatus.DISCARDED),
                            "pending_human": sum(1 for r in results if r.pending_human),
                            "rag_used_count": sum(1 for r in results if r.rag_used),
                            "rag_used_pct": round(
                                100 * sum(1 for r in results if r.rag_used) / len(results), 2
                            )
                            if results
                            else 0.0,
                        }
                    )
                    if short == "CAS-RAG":
                        route_col = routing_correct_column(short, provider)
                        if route_col in out.columns:
                            kpi_rows.append(
                                {
                                    "metric_group": "Routing accuracy",
                                    "llm_provider": provider,
                                    "llm_column_suffix": provider_key,
                                    "algorithm": short,
                                    "n": len(out),
                                    "accuracy": float(out[route_col].mean()),
                                }
                            )
            pd.DataFrame(kpi_rows).to_excel(writer, sheet_name="KPI_Summary", index=False)

    return output_path
