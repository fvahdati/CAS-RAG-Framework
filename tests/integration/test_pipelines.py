"""Integration tests for pipelines (mock LLM)."""

from cas_rag.core.enums import AlgorithmId
from cas_rag.core.models import CommentInput
from cas_rag.orchestration.engine import PipelineEngine


def test_cas_rag_explicit():
    engine = PipelineEngine(AlgorithmId.CAS_RAG, llm_provider="mock")
    r = engine.process(CommentInput(comment_id="t1", comment="The kids jumping in proves empathy wins."))
    assert r.status.value in ("success", "pending_human", "discarded")


def test_algorithm1_runs():
    engine = PipelineEngine(AlgorithmId.ALGORITHM_1, llm_provider="mock")
    r = engine.process(CommentInput(comment_id="t2", comment="Amazing"))
    assert r.algorithm == AlgorithmId.ALGORITHM_1
