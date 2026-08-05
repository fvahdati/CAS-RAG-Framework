"""Algorithm 4: LLM + Semantic Similarity Pipeline."""

from cas_rag.core.enums import AlgorithmId, EmotionLabel, PipelineStatus, PromptId, SemanticCategory
from cas_rag.core.models import CommentInput, PipelineResult
from cas_rag.pipelines.base import PipelineContext


def _semantic_two_way(
    ctx: PipelineContext,
    inp: CommentInput,
) -> tuple[str, list[PromptId]]:
    """Explicit vs Vague (Ambiguous) routing per thesis design diagram."""
    _, _, borderline = ctx.semantic.classify_three_way(inp.comment)
    if borderline:
        sigma = ctx.semantic_llm.classify(inp.comment, ctx.scenario)
        return sigma, [PromptId.P5_SEMANTIC_SIMILARITY]
    return ctx.semantic.classify_two_way(inp.comment), []


def run_algorithm4(ctx: PipelineContext, inp: CommentInput) -> PipelineResult:
    # Step 1: Unknown reference detection with LLM (P1)
    if ctx.ref_detection.detect(inp.comment, ctx.scenario):
        tau = ctx.rag.retrieve(inp.comment)
        label, secondary, expl = ctx.rag_emotion.classify(inp.comment, ctx.scenario, tau)
        return PipelineResult(
            comment_id=inp.comment_id,
            algorithm=AlgorithmId.ALGORITHM_4,
            status=PipelineStatus.SUCCESS,
            emotion_label=label,
            secondary_emotion_label=secondary,
            rag_used=True,
            prompts_invoked=[PromptId.P1_UNKNOWN_REFERENCE, PromptId.P2_RAG_EMOTION],
            retrieved_context=tau,
            explanation=expl,
            trace=["P1 unknown reference -> P2"],
        )

    # Step 2: Semantic similarity (Explicit / Vague)
    sigma, routing_prompts = _semantic_two_way(ctx, inp)

    if sigma == "Ambiguous":
        prompts = routing_prompts + [PromptId.P6_AMBIGUITY_AWARE]
        label, secondary, pending, expl = ctx.ambiguity.classify_p6(
            inp.comment, ctx.scenario, inp.comment_id, "algorithm_4"
        )
        if pending:
            return PipelineResult(
                comment_id=inp.comment_id,
                algorithm=AlgorithmId.ALGORITHM_4,
                status=PipelineStatus.PENDING_HUMAN,
                semantic_category=SemanticCategory.AMBIGUOUS,
                pending_human=True,
                prompts_invoked=prompts,
                explanation=expl,
                trace=["sigma=Vague -> P6=Ambiguous -> human verification"],
            )
        if label is None:
            return PipelineResult(
                comment_id=inp.comment_id,
                algorithm=AlgorithmId.ALGORITHM_4,
                status=PipelineStatus.DISCARDED,
                emotion_label=EmotionLabel.DISCARDED,
                semantic_category=SemanticCategory.AMBIGUOUS,
                prompts_invoked=prompts,
                explanation=expl,
                trace=["sigma=Vague -> P6 -> discarded"],
            )
        return PipelineResult(
            comment_id=inp.comment_id,
            algorithm=AlgorithmId.ALGORITHM_4,
            status=PipelineStatus.SUCCESS,
            emotion_label=label,
            secondary_emotion_label=secondary,
            semantic_category=SemanticCategory.AMBIGUOUS,
            prompts_invoked=prompts,
            explanation=expl,
            trace=["sigma=Vague -> P6 clear"],
        )

    label, secondary, expl = ctx.no_rag.classify_p7(inp.comment, ctx.scenario)
    return PipelineResult(
        comment_id=inp.comment_id,
        algorithm=AlgorithmId.ALGORITHM_4,
        status=PipelineStatus.SUCCESS,
        emotion_label=label,
        secondary_emotion_label=secondary,
        semantic_category=SemanticCategory.EXPLICIT,
        prompts_invoked=routing_prompts + [PromptId.P7_NO_RAG_NO_VAGUE],
        explanation=expl,
        trace=["sigma=Explicit -> P7"],
    )
