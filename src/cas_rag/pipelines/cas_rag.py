"""CAS-RAG unified context-aware selective retrieval framework."""

from cas_rag.core.enums import AlgorithmId, EmotionLabel, PipelineStatus, PromptId, SemanticCategory
from cas_rag.core.models import CommentInput, PipelineResult
from cas_rag.pipelines.base import PipelineContext


def run_cas_rag(ctx: PipelineContext, inp: CommentInput) -> PipelineResult:
    category, scores = ctx.semantic.classify_three_way(inp.comment)
    trace = [f"semantic={category.value}", f"scores={scores}"]

    if category == SemanticCategory.EXPLICIT:
        label, expl = ctx.no_rag.classify_p7(inp.comment, ctx.scenario)
        return PipelineResult(
            comment_id=inp.comment_id,
            algorithm=AlgorithmId.CAS_RAG,
            status=PipelineStatus.SUCCESS,
            emotion_label=label,
            semantic_category=category,
            prompts_invoked=[PromptId.P7_NO_RAG_NO_VAGUE],
            explanation=expl,
            trace=trace + ["Explicit -> P7"],
        )

    if category == SemanticCategory.VAGUE:
        label, pending = ctx.ambiguity.classify_p6(inp.comment, ctx.scenario, inp.comment_id, "cas_rag")
        if pending:
            return PipelineResult(
                comment_id=inp.comment_id,
                algorithm=AlgorithmId.CAS_RAG,
                status=PipelineStatus.PENDING_HUMAN,
                semantic_category=category,
                pending_human=True,
                prompts_invoked=[PromptId.P6_AMBIGUITY_AWARE],
                trace=trace + ["Vague -> P6=Ambiguous -> human verification"],
            )
        if label is None:
            return PipelineResult(
                comment_id=inp.comment_id,
                algorithm=AlgorithmId.CAS_RAG,
                status=PipelineStatus.DISCARDED,
                emotion_label=EmotionLabel.DISCARDED,
                semantic_category=category,
                trace=trace + ["discarded"],
            )
        return PipelineResult(
            comment_id=inp.comment_id,
            algorithm=AlgorithmId.CAS_RAG,
            status=PipelineStatus.SUCCESS,
            emotion_label=label,
            semantic_category=category,
            prompts_invoked=[PromptId.P6_AMBIGUITY_AWARE],
            trace=trace + ["Vague -> P6 clear"],
        )

    # Context-Dependent
    script = "\n".join(ctx.rag._chunks) if ctx.rag._chunks else ""
    resolvable = ctx.spacy.can_ground_with_script(inp.comment, ctx.scenario, script)
    if resolvable:
        tau = ctx.rag.retrieve(inp.comment)
        label, expl = ctx.rag_emotion.classify(inp.comment, ctx.scenario, tau)
        return PipelineResult(
            comment_id=inp.comment_id,
            algorithm=AlgorithmId.CAS_RAG,
            status=PipelineStatus.SUCCESS,
            emotion_label=label,
            semantic_category=category,
            rag_used=True,
            prompts_invoked=[PromptId.P2_RAG_EMOTION],
            retrieved_context=tau,
            explanation=expl,
            trace=trace + ["Context-Dependent resolvable -> P2"],
        )

    existing = ctx.human.resolve_for_comment(inp.comment_id)
    if existing:
        label = ctx.human.apply_review(existing)
        if label == EmotionLabel.DISCARDED:
            return PipelineResult(
                comment_id=inp.comment_id,
                algorithm=AlgorithmId.CAS_RAG,
                status=PipelineStatus.DISCARDED,
                emotion_label=EmotionLabel.DISCARDED,
                semantic_category=category,
                trace=trace + ["human: not resolvable"],
            )
        return PipelineResult(
            comment_id=inp.comment_id,
            algorithm=AlgorithmId.CAS_RAG,
            status=PipelineStatus.SUCCESS,
            emotion_label=label,
            semantic_category=category,
            trace=trace + ["human: assigned label"],
        )

    ctx.human.enqueue(
        inp.comment_id,
        inp.comment,
        "Context-Dependent: not resolvable via script",
        "cas_rag",
        {"category": category.value},
    )
    return PipelineResult(
        comment_id=inp.comment_id,
        algorithm=AlgorithmId.CAS_RAG,
        status=PipelineStatus.PENDING_HUMAN,
        semantic_category=category,
        pending_human=True,
        trace=trace + ["Context-Dependent -> human verification"],
    )
