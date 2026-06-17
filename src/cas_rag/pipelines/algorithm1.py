"""Algorithm 1: LLM–LLM Pipeline."""

from cas_rag.core.enums import AlgorithmId, EmotionLabel, PipelineStatus, PromptId
from cas_rag.core.models import CommentInput, PipelineResult
from cas_rag.pipelines.base import PipelineContext


def run_algorithm1(ctx: PipelineContext, inp: CommentInput) -> PipelineResult:
    trace, prompts = [], []
    if ctx.ref_detection.detect(inp.comment, ctx.scenario):
        prompts.append(PromptId.P1_UNKNOWN_REFERENCE)
        tau = ctx.rag.retrieve(inp.comment)
        prompts.append(PromptId.P2_RAG_EMOTION)
        label, secondary, expl = ctx.rag_emotion.classify(inp.comment, ctx.scenario, tau)
        return PipelineResult(
            comment_id=inp.comment_id,
            algorithm=AlgorithmId.ALGORITHM_1,
            status=PipelineStatus.SUCCESS,
            emotion_label=label,
            secondary_emotion_label=secondary,
            rag_used=True,
            prompts_invoked=prompts,
            retrieved_context=tau,
            explanation=expl,
            trace=trace + ["P1=Yes -> RAG -> P2"],
        )

    prompts.append(PromptId.P1_UNKNOWN_REFERENCE)
    trace.append("P1=No -> P3")
    prompts.append(PromptId.P3_NO_RAG_EMOTION)
    y, secondary, expl = ctx.no_rag.classify_p3(inp.comment, ctx.scenario)
    if y == EmotionLabel.AMBIGUOUS:
        prompts.append(PromptId.P4_AMBIGUITY)
        label, p4_secondary, pending, p4_expl = ctx.ambiguity.classify_p4(
            inp.comment, ctx.scenario, inp.comment_id, "algorithm_1"
        )
        if pending:
            return PipelineResult(
                comment_id=inp.comment_id,
                algorithm=AlgorithmId.ALGORITHM_1,
                status=PipelineStatus.PENDING_HUMAN,
                pending_human=True,
                prompts_invoked=prompts,
                explanation=p4_expl,
                trace=trace + ["P3=Ambiguous -> P4=Ambiguous -> human verification"],
            )
        if label is None:
            return PipelineResult(
                comment_id=inp.comment_id,
                algorithm=AlgorithmId.ALGORITHM_1,
                status=PipelineStatus.DISCARDED,
                emotion_label=EmotionLabel.DISCARDED,
                prompts_invoked=prompts,
                explanation=p4_expl,
                trace=trace + ["discarded after human review"],
            )
        y = label
        secondary = p4_secondary
        expl = p4_expl
    return PipelineResult(
        comment_id=inp.comment_id,
        algorithm=AlgorithmId.ALGORITHM_1,
        status=PipelineStatus.SUCCESS,
        emotion_label=y,
        secondary_emotion_label=secondary,
        prompts_invoked=prompts,
        explanation=expl,
        trace=trace,
    )
