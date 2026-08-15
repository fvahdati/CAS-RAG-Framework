"""Algorithm 3: spaCy + Semantic Similarity Pipeline."""

from cas_rag.core.enums import AlgorithmId, EmotionLabel, PipelineStatus, PromptId, SemanticCategory
from cas_rag.core.models import CommentInput, PipelineResult
from cas_rag.pipelines.base import PipelineContext


def _semantic_two_way(
    ctx: PipelineContext,
    inp: CommentInput,
) -> tuple[str, list[PromptId]]:
    """Explicit vs Ambiguous two-way routing. P5 only on the Explicit--Ambiguous borderline."""
    scores = ctx.semantic._raw_scores(inp.comment)
    sigma = "Ambiguous" if scores.get("ambiguous", 0.0) > scores.get("explicit", 0.0) else "Explicit"
    gap = abs(scores.get("ambiguous", 0.0) - scores.get("explicit", 0.0))
    if gap <= ctx.semantic.margin_b:
        sigma = ctx.semantic_llm.classify(inp.comment, ctx.scenario)
        return sigma, [PromptId.P5_SEMANTIC_SIMILARITY]
    return sigma, []


def run_algorithm3(ctx: PipelineContext, inp: CommentInput) -> PipelineResult:
    # Step 1: Unknown reference detection with spaCy
    if ctx.spacy.needs_rag(inp.comment, ctx.scenario):
        tau = ctx.rag.retrieve(inp.comment)
        label, secondary, expl = ctx.rag_emotion.classify(inp.comment, ctx.scenario, tau)
        return PipelineResult(
            comment_id=inp.comment_id,
            algorithm=AlgorithmId.ALGORITHM_3,
            status=PipelineStatus.SUCCESS,
            emotion_label=label,
            secondary_emotion_label=secondary,
            rag_used=True,
            prompts_invoked=[PromptId.P2_RAG_EMOTION],
            retrieved_context=tau,
            explanation=expl,
            trace=["spaCy unknown reference -> P2"],
        )

    # Step 2: Semantic similarity (Explicit / Ambiguous)
    sigma, routing_prompts = _semantic_two_way(ctx, inp)

    if sigma == "Ambiguous":
        prompts = routing_prompts + [PromptId.P6_AMBIGUITY_AWARE]
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
                trace=["sigma=Ambiguous -> P6=Ambiguous -> human verification"],
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
                trace=["sigma=Ambiguous -> P6 -> discarded"],
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
            trace=["sigma=Ambiguous -> P6 clear"],
        )

    label, secondary, expl = ctx.no_rag.classify_p7(inp.comment, ctx.scenario)
    return PipelineResult(
        comment_id=inp.comment_id,
        algorithm=AlgorithmId.ALGORITHM_3,
        status=PipelineStatus.SUCCESS,
        emotion_label=label,
        secondary_emotion_label=secondary,
        semantic_category=SemanticCategory.EXPLICIT,
        prompts_invoked=routing_prompts + [PromptId.P7_NO_RAG_NO_VAGUE],
        explanation=expl,
        trace=["sigma=Explicit -> P7"],
    )
