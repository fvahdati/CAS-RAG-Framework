"""Algorithm 3: spaCy + Semantic Similarity Pipeline."""

from cas_rag.core.enums import AlgorithmId, EmotionLabel, PipelineStatus, PromptId
from cas_rag.core.models import CommentInput, PipelineResult
from cas_rag.pipelines.base import PipelineContext


def _route_rag(sigma: str, needs_spacy: bool) -> bool:
    return needs_spacy or sigma == "Explicit"


def run_algorithm3(ctx: PipelineContext, inp: CommentInput) -> PipelineResult:
    sigma = ctx.semantic.classify_two_way(inp.comment)
    needs = ctx.spacy.needs_rag(inp.comment, ctx.scenario)
    prompts = [PromptId.P5_SEMANTIC_SIMILARITY]

    if _route_rag(sigma, needs):
        tau = ctx.rag.retrieve(inp.comment)
        label, secondary, expl = ctx.rag_emotion.classify(inp.comment, ctx.scenario, tau)
        return PipelineResult(
            comment_id=inp.comment_id,
            algorithm=AlgorithmId.ALGORITHM_3,
            status=PipelineStatus.SUCCESS,
            emotion_label=label,
            secondary_emotion_label=secondary,
            rag_used=True,
            prompts_invoked=prompts + [PromptId.P2_RAG_EMOTION],
            retrieved_context=tau,
            explanation=expl,
            trace=[f"sigma={sigma}, needs_rag=True -> P2"],
        )

    if sigma == "Ambiguous":
        prompts.append(PromptId.P6_AMBIGUITY_AWARE)
        label, secondary, pending, expl = ctx.ambiguity.classify_p6(
            inp.comment, ctx.scenario, inp.comment_id, "algorithm_3"
        )
        if pending:
            return PipelineResult(
                comment_id=inp.comment_id,
                algorithm=AlgorithmId.ALGORITHM_3,
                status=PipelineStatus.PENDING_HUMAN,
                pending_human=True,
                prompts_invoked=prompts,
                explanation=expl,
                trace=["sigma=Ambiguous -> P6=Ambiguous -> human verification"],
            )
        if label is None:
            return PipelineResult(
                comment_id=inp.comment_id,
                algorithm=AlgorithmId.ALGORITHM_3,
                status=PipelineStatus.DISCARDED,
                emotion_label=EmotionLabel.DISCARDED,
                prompts_invoked=prompts,
                explanation=expl,
            )
        return PipelineResult(
            comment_id=inp.comment_id,
            algorithm=AlgorithmId.ALGORITHM_3,
            status=PipelineStatus.SUCCESS,
            emotion_label=label,
            secondary_emotion_label=secondary,
            prompts_invoked=prompts,
            explanation=expl,
            trace=["sigma=Ambiguous -> P6 clear"],
        )

    label, secondary, expl = ctx.no_rag.classify_p7(inp.comment, ctx.scenario)
    return PipelineResult(
        comment_id=inp.comment_id,
        algorithm=AlgorithmId.ALGORITHM_3,
        status=PipelineStatus.SUCCESS,
        emotion_label=label,
        secondary_emotion_label=secondary,
        prompts_invoked=prompts + [PromptId.P7_NO_RAG_NO_VAGUE],
        explanation=expl,
        trace=["sigma=Explicit -> P7"],
    )
