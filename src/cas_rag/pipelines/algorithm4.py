"""Algorithm 4: LLM + Semantic Similarity Pipeline."""

from cas_rag.core.enums import AlgorithmId, EmotionLabel, PipelineStatus, PromptId
from cas_rag.core.models import CommentInput, PipelineResult
from cas_rag.pipelines.base import PipelineContext


def run_algorithm4(ctx: PipelineContext, inp: CommentInput) -> PipelineResult:
    sigma = ctx.semantic.classify_two_way(inp.comment)

    if sigma == "Ambiguous":
        label, pending = ctx.ambiguity.classify_p6(inp.comment, ctx.scenario, inp.comment_id, "algorithm_4")
        if pending:
            return PipelineResult(
                comment_id=inp.comment_id,
                algorithm=AlgorithmId.ALGORITHM_4,
                status=PipelineStatus.PENDING_HUMAN,
                pending_human=True,
                prompts_invoked=[PromptId.P6_AMBIGUITY_AWARE],
                trace=["sigma=Ambiguous -> P6=Ambiguous -> human verification"],
            )
        if label is None:
            return PipelineResult(
                comment_id=inp.comment_id,
                algorithm=AlgorithmId.ALGORITHM_4,
                status=PipelineStatus.DISCARDED,
                emotion_label=EmotionLabel.DISCARDED,
                prompts_invoked=[PromptId.P6_AMBIGUITY_AWARE],
            )
        return PipelineResult(
            comment_id=inp.comment_id,
            algorithm=AlgorithmId.ALGORITHM_4,
            status=PipelineStatus.SUCCESS,
            emotion_label=label,
            prompts_invoked=[PromptId.P6_AMBIGUITY_AWARE],
            trace=["sigma=Ambiguous -> P6 clear"],
        )

    # ROUTE_TO_RAG: explicit comments that need contextual grounding
    if "when " in inp.comment.lower() or "that " in inp.comment.lower():
        tau = ctx.rag.retrieve(inp.comment)
        label, expl = ctx.rag_emotion.classify(inp.comment, ctx.scenario, tau)
        return PipelineResult(
            comment_id=inp.comment_id,
            algorithm=AlgorithmId.ALGORITHM_4,
            status=PipelineStatus.SUCCESS,
            emotion_label=label,
            rag_used=True,
            prompts_invoked=[PromptId.P2_RAG_EMOTION],
            retrieved_context=tau,
            explanation=expl,
            trace=["sigma=Explicit, ROUTE_TO_RAG -> P2"],
        )

    label, expl = ctx.no_rag.classify_p7(inp.comment, ctx.scenario)
    return PipelineResult(
        comment_id=inp.comment_id,
        algorithm=AlgorithmId.ALGORITHM_4,
        status=PipelineStatus.SUCCESS,
        emotion_label=label,
        prompts_invoked=[PromptId.P7_NO_RAG_NO_VAGUE],
        explanation=expl,
        trace=["sigma=Explicit -> P7"],
    )
