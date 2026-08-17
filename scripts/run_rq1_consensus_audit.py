#!/usr/bin/env python3
"""Offline RQ1 consensus audit on multi-provider comparison workbooks.

Requires emotion label columns for at least two providers (e.g. from
``run_five_algorithm_comparison.py --llms openai gemini``).

Example:
  python scripts/run_rq1_consensus_audit.py \\
    --input Results/comments_bullying_Models_Run_results.xlsx \\
    --output Results/rq1_consensus_bullying.xlsx \\
    --dataset bullying \\
    --algorithms CAS-RAG Alg1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cas_rag.experiments.consensus_audit import run_consensus_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ1 offline multi-LLM consensus audit")
    parser.add_argument("--input", required=True, help="Comparison or results workbook (.xlsx)")
    parser.add_argument(
        "--output",
        default=str(ROOT / "data/evaluation_outputs/rq1_consensus_audit.xlsx"),
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Dataset label for RQ1_Summary (e.g. tutoring, bullying)",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["CAS-RAG", "Alg1", "Alg2", "Alg3", "Alg4"],
        help="Algorithm short names to audit",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["openai", "gemini"],
        help="LLM providers whose emotion columns must be present",
    )
    parser.add_argument(
        "--gold-col",
        default=None,
        help="Gold emotion column override (auto-detected if omitted)",
    )
    args = parser.parse_args()

    out = run_consensus_audit(
        Path(args.input),
        Path(args.output),
        algorithms=args.algorithms,
        providers=tuple(args.providers),
        dataset_name=args.dataset,
        gold_col=args.gold_col,
    )
    print(f"Wrote RQ1 consensus audit: {out}")
    print("Sheets: Consensus_Audit, RQ1_Summary")


if __name__ == "__main__":
    main()
