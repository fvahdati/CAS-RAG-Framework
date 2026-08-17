"""Read multi-provider comparison workbooks and produce RQ1 consensus audit sheets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cas_rag.evaluation.consensus import (
    ConsensusResult,
    compute_consensus,
    emotion_matches_gold,
    normalize_emotion_label,
)
from cas_rag.experiments.llm_columns import (
    ALGORITHM_SHORT_NAMES,
    emotion_label_column,
    provider_column_suffix,
)

DEFAULT_PROVIDERS = ("openai", "gemini")


def resolve_gold_emotion_column(columns: list[str]) -> str | None:
    for name in ("final label", "final_label", "gold_label", "Emotion label", "emotion_label"):
        if name in columns:
            return name
    return None


def resolve_comment_id_column(columns: list[str]) -> str:
    if "comment_id" in columns:
        return "comment_id"
    return columns[0]


def discover_provider_columns(columns: list[str], algorithm_short: str) -> dict[str, str]:
    """Map provider key -> workbook column for an algorithm's emotion label."""
    found: dict[str, str] = {}
    prefix = f"{algorithm_short}_emotion_label_"
    for col in columns:
        if col.startswith(prefix):
            suffix = col[len(prefix) :]
            for provider in ("openai", "gemini", "ollama", "mock"):
                if provider_column_suffix(provider) == suffix:
                    found[provider] = col
                    break
    legacy = f"{algorithm_short}_emotion_label"
    if legacy in columns and "openai" not in found:
        found["openai"] = legacy
    return found


def build_consensus_audit_rows(
    df: pd.DataFrame,
    *,
    algorithms: list[str] | None = None,
    providers: tuple[str, ...] = DEFAULT_PROVIDERS,
    gold_col: str | None = None,
) -> list[dict]:
    columns = list(df.columns)
    gold_col = gold_col or resolve_gold_emotion_column(columns)
    id_col = resolve_comment_id_column(columns)
    alg_list = algorithms or [short for short, _ in ALGORITHM_SHORT_NAMES]

    rows: list[dict] = []
    for _, record in df.iterrows():
        comment_id = str(record[id_col])
        gold = normalize_emotion_label(record.get(gold_col)) if gold_col else ""

        for short in alg_list:
            provider_cols = discover_provider_columns(columns, short)
            provider_labels = {
                provider: record.get(provider_cols[provider])
                for provider in providers
                if provider in provider_cols
            }
            if not provider_labels:
                continue

            consensus: ConsensusResult = compute_consensus(provider_labels, gold_label=gold or None)
            row = {
                "comment_id": comment_id,
                "algorithm": short,
                "gold_emotion": gold,
                "consensus_label": consensus.consensus_label or "",
                "agreement_ratio": consensus.agreement_ratio,
                "reliability": consensus.reliability,
                "disagreement_flag": consensus.disagreement_flag,
                "hallucination_proxy": consensus.hallucination_proxy,
                "n_valid_votes": consensus.n_valid_votes,
                "unanimous": consensus.unanimous,
            }
            for provider in providers:
                col = provider_cols.get(provider)
                label = normalize_emotion_label(record.get(col)) if col else ""
                row[f"{provider}_label"] = label
                row[f"{provider}_correct"] = emotion_matches_gold(label, gold)
            row["consensus_correct"] = emotion_matches_gold(consensus.consensus_label, gold)
            rows.append(row)
    return rows


def summarize_rq1(audit_rows: list[dict], *, dataset_name: str = "") -> list[dict]:
    if not audit_rows:
        return [
            {
                "dataset": dataset_name,
                "algorithm": "none",
                "n_comments": 0,
                "note": "no dual-provider emotion columns found",
            }
        ]

    summary: list[dict] = []
    df = pd.DataFrame(audit_rows)
    for algorithm, group in df.groupby("algorithm"):
        gold_mask = group["gold_emotion"].astype(str).str.strip() != ""
        eval_rows = group[gold_mask]
        n = len(group)
        n_gold = len(eval_rows)

        def _accuracy(col: str) -> float | None:
            if n_gold == 0:
                return None
            series = eval_rows[col]
            valid = series.notna()
            if not valid.any():
                return None
            return round(float(series[valid].mean()), 4)

        summary.append(
            {
                "dataset": dataset_name,
                "algorithm": algorithm,
                "n_comments": n,
                "n_gold_evaluable": n_gold,
                "pct_unanimous_high": round(100 * (group["reliability"] == "HIGH").mean(), 2),
                "pct_disagreement": round(100 * group["disagreement_flag"].mean(), 2),
                "pct_split": round(100 * (group["reliability"] == "SPLIT").mean(), 2),
                "openai_accuracy": _accuracy("openai_correct"),
                "gemini_accuracy": _accuracy("gemini_correct"),
                "consensus_accuracy": _accuracy("consensus_correct"),
                "pct_hallucination_proxy": round(
                    100 * eval_rows["hallucination_proxy"].fillna(False).mean(), 2
                )
                if n_gold
                else None,
                "pct_gold_errors_under_disagreement": round(
                    100
                    * (
                        (eval_rows["disagreement_flag"])
                        & (eval_rows["consensus_correct"] == False)  # noqa: E712
                    ).mean(),
                    2,
                )
                if n_gold
                else None,
            }
        )
    return summary


def run_consensus_audit(
    input_path: Path,
    output_path: Path,
    *,
    algorithms: list[str] | None = None,
    providers: tuple[str, ...] = DEFAULT_PROVIDERS,
    dataset_name: str = "",
    gold_col: str | None = None,
) -> Path:
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path, sheet_name=0)
    else:
        df = pd.read_csv(input_path)

    audit_rows = build_consensus_audit_rows(
        df,
        algorithms=algorithms,
        providers=providers,
        gold_col=gold_col,
    )
    summary_rows = summarize_rq1(audit_rows, dataset_name=dataset_name or input_path.stem)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(audit_rows).to_excel(writer, sheet_name="Consensus_Audit", index=False)
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="RQ1_Summary", index=False)
    return output_path
