"""Algorithm 2: spaCy–LLM Pipeline."""

from cas_rag.core.enums import AlgorithmId, EmotionLabel, PipelineStatus, PromptId
from cas_rag.core.models import CommentInput, PipelineResult
from cas_rag.pipelines.base import PipelineContext


def run_algorithm2(ctx: PipelineContext, inp: CommentInput) -> PipelineResult:
    if ctx.spacy.needs_rag(inp.comment, ctx.scenario):
        tau = ctx.rag.retrieve(inp.comment)
        label, expl = ctx.rag_emotion.classify(inp.comment, ctx.scenario, tau)
        return PipelineResult(
            comment_id=inp.comment_id,
            algorithm=AlgorithmId.ALGORITHM_2,
            status=PipelineStatus.SUCCESS,
            emotion_label=label,
            rag_used=True,
            prompts_invoked=[PromptId.P2_RAG_EMOTION],
            retrieved_context=tau,
            explanation=expl,
            trace=["spaCy needs_rag -> P2"],
        )

    prompts = [PromptId.P3_NO_RAG_EMOTION]
    y, expl = ctx.no_rag.classify_p3(inp.comment, ctx.scenario)
    if y == EmotionLabel.AMBIGUOUS:
        prompts.append(PromptId.P4_AMBIGUITY)
        label, pending = ctx.ambiguity.classify_p4(inp.comment, ctx.scenario, inp.comment_id, "algorithm_2")
        if pending:
            return PipelineResult(
                comment_id=inp.comment_id,
                algorithm=AlgorithmId.ALGORITHM_2,
                status=PipelineStatus.PENDING_HUMAN,
                pending_human=True,
                prompts_invoked=prompts,
                trace=["spaCy no RAG; P3=Ambiguous -> P4=Ambiguous -> human verification"],
            )
        if label is None:
            return PipelineResult(
                comment_id=inp.comment_id,
                algorithm=AlgorithmId.ALGORITHM_2,
                status=PipelineStatus.DISCARDED,
                emotion_label=EmotionLabel.DISCARDED,
                prompts_invoked=prompts,
            )
        y = label
    return PipelineResult(
        comment_id=inp.comment_id,
        algorithm=AlgorithmId.ALGORITHM_2,
        status=PipelineStatus.SUCCESS,
        emotion_label=y,
        prompts_invoked=prompts,
        explanation=expl,
        trace=["spaCy no RAG -> P3"],
    )
