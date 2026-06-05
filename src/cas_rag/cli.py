"""CLI entry point."""

import argparse
from pathlib import Path

from cas_rag.config.settings import get_settings
from cas_rag.core.enums import AlgorithmId
from cas_rag.core.models import CommentInput, HumanReviewDecision
from cas_rag.experiments.runner import ExperimentRunner
from cas_rag.orchestration.engine import PipelineEngine
from cas_rag.subsystems.human_verification import HumanVerificationService


def main():
    parser = argparse.ArgumentParser(description="CAS-RAG research platform CLI")
    parser.add_argument("--algorithm", default="cas_rag", choices=[a.value for a in AlgorithmId])
    parser.add_argument("--comment", default=None)
    parser.add_argument("--input", default=None, help="CSV dataset path")
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--llm", default=None)
    sub = parser.add_subparsers(dest="cmd")

    human_p = sub.add_parser("human-list")
    human_s = sub.add_parser("human-submit")
    human_s.add_argument("--queue-id", required=True)
    human_s.add_argument("--confirms-ambiguous", action="store_true")
    human_s.add_argument("--label", default=None)

    args = parser.parse_args()
    settings = get_settings()

    if args.cmd == "human-list":
        for item in HumanVerificationService().get_pending():
            print(item)
        return

    if args.cmd == "human-submit":
        svc = HumanVerificationService()
        decision = HumanReviewDecision(
            comment_id="",
            confirms_ambiguous=args.confirms_ambiguous,
            resolvable=not args.confirms_ambiguous,
            assigned_label=None if args.confirms_ambiguous else args.label,
        )
        svc.submit_review(args.queue_id, decision)
        print("Review submitted.")
        return

    if args.compare_all and args.input:
        out = ExperimentRunner().compare_all(Path(args.input))
        print(f"Comparison saved: {out}")
        return

    if args.input:
        out = ExperimentRunner().run(AlgorithmId(args.algorithm), Path(args.input), args.llm)
        print(f"Results: {out}")
        return

    comment = args.comment or "The kids standing up for him was beautiful."
    engine = PipelineEngine(AlgorithmId(args.algorithm), llm_provider=args.llm)
    result = engine.process(CommentInput(comment_id="cli-1", comment=comment))
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
