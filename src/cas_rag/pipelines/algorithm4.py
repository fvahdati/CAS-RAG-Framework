"""Algorithm 4: LLM + Semantic Similarity Pipeline."""

from cas_rag.core.enums import AlgorithmId, EmotionLabel, PipelineStatus, PromptId, SemanticCategory
from cas_rag.core.models import CommentInput, PipelineResult
from cas_rag.pipelines.base import PipelineContext


def _resolve_borderline_category(
    ctx: PipelineContext,
    inp: CommentInput,
    category: SemanticCategory,
    borderline: bool,
) -> tuple[SemanticCategory, list[PromptId]]:
    if not borderline:
        return category, []
    if category == SemanticCategory.CONTEXT_DEPENDENT:
        return category, [PromptId.P5_SEMANTIC_SIMILARITY]

    p5_label = ctx.semantic_llm.classify(inp.comment, ctx.scenario)
    prompts = [PromptId.P5_SEMANTIC_SIMILARITY]
    if p5_label == "Ambiguous":
        return SemanticCategory.AMBIGUOUS, prompts
    if category == SemanticCategory.AMBIGUOUS and p5_label == "Explicit":
        return SemanticCategory.EXPLICIT, prompts
    return category, prompts


def run_algorithm4(ctx: PipelineContext, inp: CommentInput) -> PipelineResult:
    category, scores, borderline = ctx.semantic.classify_three_way(inp.comment)
    category, prompts = _resolve_borderline_category(ctx, inp, category, borderline)

    if category == SemanticCategory.AMBIGUOUS:
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
                prompts_invoked=prompts + [PromptId.P6_AMBIGUITY_AWARE],
                explanation=expl,
                trace=["category=Ambiguous -> P6=Ambiguous -> human verification"],
            )
        if label is None:
            return PipelineResult(
                comment_id=inp.comment_id,
                algorithm=AlgorithmId.ALGORITHM_4,
                status=PipelineStatus.DISCARDED,
                emotion_label=EmotionLabel.DISCARDED,
                semantic_category=SemanticCategory.AMBIGUOUS,
                prompts_invoked=prompts + [PromptId.P6_AMBIGUITY_AWARE],
                explanation=expl,
            )
        return PipelineResult(
            comment_id=inp.comment_id,
            algorithm=AlgorithmId.ALGORITHM_4,
            status=PipelineStatus.SUCCESS,
            emotion_label=label,
            secondary_emotion_label=secondary,
            semantic_category=SemanticCategory.AMBIGUOUS,
            prompts_invoked=prompts + [PromptId.P6_AMBIGUITY_AWARE],
            explanation=expl,
            trace=["category=Ambiguous -> P6 clear"],
        )

    if category == SemanticCategory.CONTEXT_DEPENDENT:
        tau = ctx.rag.retrieve(inp.comment)
        label, secondary, expl = ctx.rag_emotion.classify(inp.comment, ctx.scenario, tau)
        return PipelineResult(
            comment_id=inp.comment_id,
            algorithm=AlgorithmId.ALGORITHM_4,
            status=PipelineStatus.SUCCESS,
            emotion_label=label,
            secondary_emotion_label=secondary,
            semantic_category=SemanticCategory.CONTEXT_DEPENDENT,
            rag_used=True,
            prompts_invoked=prompts + [PromptId.P2_RAG_EMOTION],
            retrieved_context=tau,
            explanation=expl,
            trace=["category=Context-Dependent -> P2"],
        )

    label, secondary, expl = ctx.no_rag.classify_p7(inp.comment, ctx.scenario)
    return PipelineResult(
        comment_id=inp.comment_id,
        algorithm=AlgorithmId.ALGORITHM_4,
        status=PipelineStatus.SUCCESS,
        emotion_label=label,
        secondary_emotion_label=secondary,
        semantic_category=SemanticCategory.EXPLICIT,
        prompts_invoked=prompts + [PromptId.P7_NO_RAG_NO_VAGUE],
        explanation=expl,
        trace=["category=Explicit -> P7"],
    )
