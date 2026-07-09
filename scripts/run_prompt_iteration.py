#!/usr/bin/env python3
"""Offline prompt iteration workflow: evaluate subset, analyze failures, export report."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cas_rag.experiments.prompt_iteration import run_prompt_iteration  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Prompt iteration workflow (Option A): failure analysis without runtime loops."
    )
    parser.add_argument(
        "--input",
        default=str(ROOT / "resources/datasets/Experiments-data/comments.csv"),
        help="Excel/CSV with comment_text, comment_id, and optional gold columns.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data/evaluation_outputs/prompt_iteration_report.xlsx"),
    )
    parser.add_argument("--video-slug", default="video_02_education_tutoring_diner")
    parser.add_argument(
        "--algorithm",
        action="append",
        dest="algorithms",
        help="Algorithm short name (Alg1, Alg2, Alg3, Alg4, CAS-RAG). Repeatable.",
    )
    parser.add_argument("--llm", default=None, help="LLM provider (e.g. openai, mock).")
    parser.add_argument(
        "--comment-ids",
        default=None,
        help="Comma-separated comment_id values to evaluate (supervisor subset).",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=None,
        help="Limit number of comments (useful for 2-3 comment iteration).",
    )
    args = parser.parse_args()

    comment_ids = None
    if args.comment_ids:
        comment_ids = [part.strip() for part in args.comment_ids.split(",") if part.strip()]

    out = run_prompt_iteration(
        Path(args.input),
        Path(args.output),
        video_slug=args.video_slug,
        algorithms=args.algorithms,
        llm_provider=args.llm,
        comment_ids=comment_ids,
        max_comments=args.max_comments,
    )
    print(f"Wrote prompt iteration report: {out}")


if __name__ == "__main__":
    main()
