#!/usr/bin/env python3
"""Run CAS-RAG experiments from command line."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cas_rag.experiments.runner import ExperimentRunner  # noqa: E402
from cas_rag.core.enums import AlgorithmId  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", default="cas_rag")
    p.add_argument("--input", default=str(ROOT / "resources/datasets/sample/comments.csv"))
    p.add_argument("--compare-all", action="store_true")
    p.add_argument("--llm", default=None)
    args = p.parse_args()
    runner = ExperimentRunner()
    if args.compare_all:
        out = runner.compare_all(Path(args.input))
        print(out)
    else:
        out = runner.run(AlgorithmId(args.algorithm), Path(args.input), args.llm)
        print(out)


if __name__ == "__main__":
    main()
