#!/usr/bin/env python3
"""Run Alg1–4 + CAS-RAG on an experiment Excel file; write combined comparison workbook."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cas_rag.experiments.comparison_runner import run_five_algorithm_comparison  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Five-algorithm comparison export")
    parser.add_argument(
        "--input",
        default=str(ROOT / "resources/datasets/Experiments-data/comments_education_tutoring_diner.xlsx"),
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT
            / "data/evaluation_outputs/education_tutoring_diner_five_algorithm_comparison.xlsx"
        ),
    )
    parser.add_argument("--video-slug", default="video_02_education_tutoring_diner")
    parser.add_argument(
        "--llm",
        default=None,
        help="Run a single LLM provider (e.g. openai). Overrides --llms.",
    )
    parser.add_argument(
        "--llms",
        nargs="+",
        default=None,
        help="Run multiple LLM providers sequentially (default: openai gemini).",
    )
    parser.add_argument(
        "--kpi",
        action="store_true",
        help="Include KPI_Summary sheet (after gold emotion labels are available)",
    )
    parser.add_argument(
        "--no-gemini-fallback",
        action="store_true",
        help="Do not fall back to Ollama when Gemini is unavailable.",
    )
    parser.add_argument(
        "--gemini-fallback",
        default="ollama",
        help="Provider to use when Gemini probe fails (default: ollama).",
    )
    args = parser.parse_args()

    llm_providers = None
    llm_provider = None
    if args.llm:
        llm_provider = args.llm
    else:
        llm_providers = args.llms

    out = run_five_algorithm_comparison(
        Path(args.input),
        Path(args.output),
        video_slug=args.video_slug,
        llm_provider=llm_provider,
        llm_providers=llm_providers,
        include_kpi=args.kpi,
        gemini_fallback=None if args.no_gemini_fallback else args.gemini_fallback,
    )
    print(f"Wrote comparison workbook: {out}")


if __name__ == "__main__":
    main()
