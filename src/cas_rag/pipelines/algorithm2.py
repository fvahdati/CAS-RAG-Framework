"""Algorithm 2: spaCy–LLM Pipeline."""

from cas_rag.core.enums import AlgorithmId, EmotionLabel, PipelineStatus, PromptId, SemanticCategory
from cas_rag.core.models import CommentInput, PipelineResult
from cas_rag.pipelines.base import PipelineContext


def run_algorithm2(ctx: PipelineContext, inp: CommentInput) -> PipelineResult:
    if ctx.spacy.needs_rag(inp.comment, ctx.scenario):
        tau = ctx.rag.retrieve(inp.comment)
        label, secondary, expl = ctx.rag_emotion.classify(inp.comment, ctx.scenario, tau)
        return PipelineResult(
            comment_id=inp.comment_id,
            algorithm=AlgorithmId.ALGORITHM_2,
            status=PipelineStatus.SUCCESS,
            emotion_label=label,
            secondary_emotion_label=secondary,
            semantic_category=SemanticCategory.CONTEXT_DEPENDENT,
            rag_used=True,
            prompts_invoked=[PromptId.P2_RAG_EMOTION],
            retrieved_context=tau,
            explanation=expl,
            trace=["spaCy needs_rag -> P2"],
        )

    prompts = [PromptId.P3_NO_RAG_EMOTION]
    y, secondary, expl = ctx.no_rag.classify_p3(inp.comment, ctx.scenario)
    if y == EmotionLabel.AMBIGUOUS:
        prompts.append(PromptId.P4_AMBIGUITY)
        label, p4_secondary, pending, p4_expl = ctx.ambiguity.classify_p4(
            inp.comment, ctx.scenario, inp.comment_id, "algorithm_2"
        )
        if pending:
            return PipelineResult(
                comment_id=inp.comment_id,
                algorithm=AlgorithmId.ALGORITHM_2,
                status=PipelineStatus.PENDING_HUMAN,
                semantic_category=SemanticCategory.AMBIGUOUS,
                pending_human=True,
                prompts_invoked=prompts,
                explanation=p4_expl,
                trace=["spaCy no RAG; P3=Ambiguous -> P4=Ambiguous -> human verification"],
            )
        if label is None:
            return PipelineResult(
                comment_id=inp.comment_id,
                algorithm=AlgorithmId.ALGORITHM_2,
                status=PipelineStatus.DISCARDED,
                emotion_label=EmotionLabel.DISCARDED,
                semantic_category=SemanticCategory.AMBIGUOUS,
                prompts_invoked=prompts,
                explanation=p4_expl,
            )
        y = label
        secondary = p4_secondary
        expl = p4_expl
    category = SemanticCategory.AMBIGUOUS if PromptId.P4_AMBIGUITY in prompts else SemanticCategory.EXPLICIT
    return PipelineResult(
        comment_id=inp.comment_id,
        algorithm=AlgorithmId.ALGORITHM_2,
        status=PipelineStatus.SUCCESS,
        emotion_label=y,
        secondary_emotion_label=secondary,
        semantic_category=category,
        prompts_invoked=prompts,
        explanation=expl,
        trace=["spaCy no RAG -> P3"],
    )
