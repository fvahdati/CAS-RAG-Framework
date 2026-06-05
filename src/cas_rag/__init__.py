"""CAS-RAG: Context-Aware Semantic Retrieval-Augmented Generation."""

__version__ = "1.0.0"

from cas_rag.core.models import PipelineResult, CommentInput
from cas_rag.orchestration.engine import PipelineEngine

__all__ = ["__version__", "PipelineResult", "CommentInput", "PipelineEngine"]
