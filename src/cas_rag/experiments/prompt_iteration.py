"""Offline prompt iteration workflow (Option A): evaluate, diagnose, report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cas_rag.config.settings import get_settings
from cas_rag.core.datetime_utils import utc_now_iso
from cas_rag.core.enums import AlgorithmId
from cas_rag.core.models import CommentInput
from cas_rag.experiments.comparison_runner import load_experiment_comments, run_algorithm_pass
from cas_rag.experiments.failure_analysis import (
    SHORT_TO_ALGORITHM,
    build_algorithm_summary,
    build_evaluation_row,
    build_failure_summary,
    build_iteration_recommendations,
    build_prompt_fix_guide,
    normalize_category,
)
from cas_rag.llm.validation import validate_llm_provider

DEFAULT_ALGORITHMS = ("Alg1", "Alg2", "Alg3", "Alg4", "CAS-RAG")

EVALUATION_COLUMN_ORDER = (
    "comment_id",
    "comment_text",
    "algorithm",
    "expected_emotion",
    "predicted_emotion",
    "expected_category",
    "predicted_category",
    "emotion_correct",
    "routing_correct",
    "why_misclassified",
    "contributing_prompt",
    "prompt_file",
    "prompt_section_to_edit",
    "suggested_prompt_fix",
    "example_to_add",
    "confidence",
    "semantic_top_score",
    "semantic_margin",
    "semantic_borderline",
    "routing_path",
    "prompts_used",
    "rag_used",
    "status",
    "likely_failure_reason",
    "explanation_excerpt",
)


def _order_columns(df: pd.DataFrame, preferred: tuple[str, ...]) -> pd.DataFrame:
    cols = [c for c in preferred if c in df.columns]
    cols.extend(c for c in df.columns if c not in cols)
    return df[cols]


def _filter_comments(
    df: pd.DataFrame,
    comments: list[CommentInput],
    *,
    comment_ids: list[str] | None,
    max_comments: int | None,
) -> tuple[pd.DataFrame, list[CommentInput]]:
    if comment_ids:
        wanted = {str(cid).strip() for cid in comment_ids}
        mask = df["comment_id"].astype(str).isin(wanted)
        df = df[mask].copy()
        comments = [c for c in comments if c.comment_id in wanted]
    if max_comments is not None and max_comments > 0:
        df = df.head(max_comments).copy()
        keep_ids = set(df["comment_id"].astype(str))
        comments = [c for c in comments if c.comment_id in keep_ids][:max_comments]
    return df, comments


def _gold_row_for_comment(df: pd.DataFrame, comment_id: str) -> dict:
    rows = df[df["comment_id"].astype(str) == str(comment_id)]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def run_prompt_iteration(
    input_path: Path,
    output_path: Path,
    *,
    video_slug: str = "video_02_education_tutoring_diner",
    algorithms: list[str] | None = None,
    llm_provider: str | None = None,
    comment_ids: list[str] | None = None,
    max_comments: int | None = None,
    experiments_root: Path | None = None,
) -> Path:
    """
    Run a prompt-iteration evaluation pass without changing inference pipelines.

    Produces an Excel workbook with evaluation detail, failure analysis, and
    iteration recommendations for manual prompt editing.
    """
    settings = get_settings()
    if llm_provider:
        validate_llm_provider(settings, llm_provider)

    df, comments = load_experiment_comments(input_path)
    df, comments = _filter_comments(df, comments, comment_ids=comment_ids, max_comments=max_comments)
    if not comments:
        raise ValueError("No comments selected for prompt iteration run.")

    exp_root = experiments_root or (settings.project_root / "resources" / "datasets" / "Experiments-data")
    video_dir = exp_root / video_slug
    prototypes_path = video_dir / "prototypes_semantic.json"

    alg_names = algorithms or list(DEFAULT_ALGORITHMS)
    alg_ids: list[AlgorithmId] = []
    for name in alg_names:
        if name not in SHORT_TO_ALGORITHM:
            raise ValueError(f"Unknown algorithm '{name}'. Use one of: {', '.join(DEFAULT_ALGORITHMS)}")
        alg_ids.append(SHORT_TO_ALGORITHM[name])

    gold_col = "expected_category" if "expected_category" in df.columns else None
    expected_by_comment: dict[str, str] = {}
    if gold_col:
        for _, row in df.iterrows():
            expected_by_comment[str(row["comment_id"])] = normalize_category(row[gold_col])

    human_queue_path = settings.project_root / "data" / "prompt_iteration_human_queue.json"
    human_queue_path.parent.mkdir(parents=True, exist_ok=True)
    human_queue_path.write_text('{"pending": [], "completed": []}', encoding="utf-8")

    evaluation_rows: list[dict] = []
    for alg_id in alg_ids:
        results = run_algorithm_pass(
            alg_id,
            comments,
            video_slug=video_slug,
            human_queue_path=human_queue_path,
            llm_provider=llm_provider,
            expected_by_comment=expected_by_comment,
        )
        for inp, result in zip(comments, results):
            gold_row = _gold_row_for_comment(df, inp.comment_id)
            evaluation_rows.append(
                build_evaluation_row(
                    inp,
                    result,
                    gold_row=gold_row,
                    prototypes_path=prototypes_path if prototypes_path.exists() else None,
                )
            )

    failure_summary = build_failure_summary(evaluation_rows)
    algorithm_summary = build_algorithm_summary(evaluation_rows)
    recommendations = build_iteration_recommendations(evaluation_rows)
    prompt_fix_guide = build_prompt_fix_guide(evaluation_rows)

    workflow_steps = [
        {
            "step": 1,
            "action": "Run prompt iteration",
            "command": f"python scripts/run_prompt_iteration.py --input {input_path} --output {output_path}",
        },
        {
            "step": 2,
            "action": "Open Prompt_Fix_Guide and Iteration_Recommendations",
            "command": "For each failing comment, read why_misclassified, contributing_prompt, and suggested_prompt_fix.",
        },
        {
            "step": 3,
            "action": "Edit prompts for video",
            "command": f"Update files under {video_dir / 'prompts'} using prompt_file and prompt_section_to_edit.",
        },
        {
            "step": 4,
            "action": "Re-run on the same comment subset",
            "command": "Repeat step 1 with the same --comment-ids or --max-comments before scaling up.",
        },
    ]

    meta = pd.DataFrame(
        [
            {
                "generated_at_utc": utc_now_iso(),
                "input_path": str(input_path),
                "video_slug": video_slug,
                "llm_provider": llm_provider or settings.llm_provider,
                "n_comments": len(comments),
                "algorithms": ", ".join(alg_names),
            }
        ]
    )

    evaluation_df = _order_columns(pd.DataFrame(evaluation_rows), EVALUATION_COLUMN_ORDER)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        evaluation_df.to_excel(writer, sheet_name="Evaluation_Detail", index=False)
        pd.DataFrame(prompt_fix_guide).to_excel(writer, sheet_name="Prompt_Fix_Guide", index=False)
        pd.DataFrame(failure_summary).to_excel(writer, sheet_name="Failure_Summary", index=False)
        pd.DataFrame(algorithm_summary).to_excel(writer, sheet_name="Algorithm_Summary", index=False)
        pd.DataFrame(recommendations).to_excel(writer, sheet_name="Iteration_Recommendations", index=False)
        pd.DataFrame(workflow_steps).to_excel(writer, sheet_name="Iteration_Workflow", index=False)
        meta.to_excel(writer, sheet_name="Run_Metadata", index=False)

    return output_path
