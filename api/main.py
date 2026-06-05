"""FastAPI backend for CAS-RAG research platform."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from cas_rag.core.enums import AlgorithmId
from cas_rag.core.models import CommentInput, HumanReviewDecision, PipelineResult
from cas_rag.experiments.runner import ExperimentRunner
from cas_rag.orchestration.engine import PipelineEngine
from cas_rag.subsystems.human_verification import HumanVerificationService
from cas_rag.analytics.dashboard_data import AnalyticsEngine
from cas_rag.config.settings import get_settings
from cas_rag.llm.validation import validate_llm_provider

app = FastAPI(
    title="CAS-RAG API",
    description="Context-Aware Semantic Retrieval-Augmented Generation",
    version="1.0.0",
)


class ProcessRequest(BaseModel):
    comment_id: str
    comment: str
    algorithm: str = "cas_rag"
    llm_provider: str | None = None


class HumanSubmitRequest(BaseModel):
    queue_id: str
    confirms_ambiguous: bool = False
    assigned_label: str | None = None
    reviewer_id: str = "api_reviewer"


@app.get("/health")
def health():
    settings = get_settings()
    provider = settings.llm_provider
    llm_ready = True
    llm_error = None
    try:
        validate_llm_provider(settings, provider)
    except ValueError as e:
        llm_ready = False
        llm_error = str(e)
    return {
        "status": "ok",
        "service": "CAS-RAG",
        "llm_provider": provider,
        "llm_ready": llm_ready,
        "llm_error": llm_error,
        "offline_mode": settings.offline_mode,
    }


@app.get("/algorithms")
def list_algorithms():
    return {"algorithms": [a.value for a in AlgorithmId]}


@app.post("/process", response_model=PipelineResult)
def process_comment(req: ProcessRequest):
    try:
        engine = PipelineEngine(req.algorithm, llm_provider=req.llm_provider)
        return engine.process(CommentInput(comment_id=req.comment_id, comment=req.comment))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/compare")
def compare_algorithms(req: ProcessRequest):
    engine = PipelineEngine("cas_rag", llm_provider=req.llm_provider)
    inp = CommentInput(comment_id=req.comment_id, comment=req.comment)
    results = engine.compare_algorithms(inp)
    return {k: v.model_dump() for k, v in results.items()}


@app.get("/human/pending")
def human_pending():
    return HumanVerificationService().get_pending()


@app.post("/human/submit")
def human_submit(req: HumanSubmitRequest):
    svc = HumanVerificationService()
    decision = HumanReviewDecision(
        comment_id="",
        confirms_ambiguous=req.confirms_ambiguous,
        resolvable=not req.confirms_ambiguous,
        assigned_label=req.assigned_label,
        reviewer_id=req.reviewer_id,
    )
    try:
        svc.submit_review(req.queue_id, decision)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"status": "submitted"}


@app.post("/experiments/run")
def run_experiment(algorithm: str = "cas_rag", dataset: str = "resources/datasets/sample/comments.csv"):
    from pathlib import Path

    path = ExperimentRunner().run(AlgorithmId(algorithm), Path(dataset))
    return {"output": str(path)}


@app.post("/analytics/report")
def analytics_report(algorithm: str = "cas_rag", dataset: str = "resources/datasets/sample/comments.csv"):
    from pathlib import Path
    from cas_rag.datasets.loader import DatasetLoader

    engine = PipelineEngine(algorithm)
    comments = DatasetLoader().load_comments(Path(dataset))
    gold = {c.comment_id: c.gold_label.value for c in comments if c.gold_label}
    results = engine.process_dataset(dataset)
    return AnalyticsEngine().build_report(results, gold)
