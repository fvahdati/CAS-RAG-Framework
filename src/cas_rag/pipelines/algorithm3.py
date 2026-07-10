"""Algorithm 3: spaCy + Semantic Similarity Pipeline."""

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


def run_algorithm3(ctx: PipelineContext, inp: CommentInput) -> PipelineResult:
    category, scores, borderline = ctx.semantic.classify_three_way(inp.comment)
    category, prompts = _resolve_borderline_category(ctx, inp, category, borderline)
    needs = ctx.spacy.needs_rag(inp.comment, ctx.scenario)

    if category == SemanticCategory.AMBIGUOUS:
        prompts = prompts + [PromptId.P6_AMBIGUITY_AWARE]
        label, secondary, pending, expl = ctx.ambiguity.classify_p6(
            inp.comment, ctx.scenario, inp.comment_id, "algorithm_3"
        )
        if pending:
            return PipelineResult(
                comment_id=inp.comment_id,
                algorithm=AlgorithmId.ALGORITHM_3,
                status=PipelineStatus.PENDING_HUMAN,
                semantic_category=SemanticCategory.AMBIGUOUS,
                pending_human=True,
                prompts_invoked=prompts,
                explanation=expl,
                trace=["category=Ambiguous -> P6=Ambiguous -> human verification"],
            )
        if label is None:
            return PipelineResult(
                comment_id=inp.comment_id,
                algorithm=AlgorithmId.ALGORITHM_3,
                status=PipelineStatus.DISCARDED,
                emotion_label=EmotionLabel.DISCARDED,
                semantic_category=SemanticCategory.AMBIGUOUS,
                prompts_invoked=prompts,
                explanation=expl,
            )
        return PipelineResult(
            comment_id=inp.comment_id,
            algorithm=AlgorithmId.ALGORITHM_3,
            status=PipelineStatus.SUCCESS,
            emotion_label=label,
            secondary_emotion_label=secondary,
            semantic_category=SemanticCategory.AMBIGUOUS,
            prompts_invoked=prompts,
            explanation=expl,
            trace=["category=Ambiguous -> P6 clear"],
        )

    if category == SemanticCategory.CONTEXT_DEPENDENT or needs:
        tau = ctx.rag.retrieve(inp.comment)
        label, secondary, expl = ctx.rag_emotion.classify(inp.comment, ctx.scenario, tau)
        routed_category = (
            SemanticCategory.CONTEXT_DEPENDENT
            if category == SemanticCategory.CONTEXT_DEPENDENT
            else SemanticCategory.EXPLICIT
        )
        return PipelineResult(
            comment_id=inp.comment_id,
            algorithm=AlgorithmId.ALGORITHM_3,
            status=PipelineStatus.SUCCESS,
            emotion_label=label,
            secondary_emotion_label=secondary,
            semantic_category=routed_category,
            rag_used=True,
            prompts_invoked=prompts + [PromptId.P2_RAG_EMOTION],
            retrieved_context=tau,
            explanation=expl,
            trace=[f"category={routed_category.value}, needs_rag={needs} -> P2"],
        )

    label, secondary, expl = ctx.no_rag.classify_p7(inp.comment, ctx.scenario)
    return PipelineResult(
        comment_id=inp.comment_id,
        algorithm=AlgorithmId.ALGORITHM_3,
        status=PipelineStatus.SUCCESS,
        emotion_label=label,
        secondary_emotion_label=secondary,
        semantic_category=SemanticCategory.EXPLICIT,
        prompts_invoked=prompts + [PromptId.P7_NO_RAG_NO_VAGUE],
        explanation=expl,
        trace=["category=Explicit -> P7"],
    )
