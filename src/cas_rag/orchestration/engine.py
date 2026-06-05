"""Pipeline orchestration and algorithm switching."""

from pathlib import Path

from cas_rag.config.settings import get_settings
from cas_rag.core.enums import AlgorithmId
from cas_rag.core.models import CommentInput, PipelineResult
from cas_rag.datasets.loader import DatasetLoader
from cas_rag.llm.factory import create_llm_provider
from cas_rag.logging_utils import get_logger
from cas_rag.prompts.manager import PromptManager
from cas_rag.retrieval.rag_engine import RAGEngine
from cas_rag.semantic.similarity import SemanticSimilarityEngine
from cas_rag.linguistic.spacy_analyzer import SpacyLinguisticAnalyzer
from cas_rag.subsystems.human_verification import HumanVerificationService
from cas_rag.pipelines.base import PipelineContext
from cas_rag.pipelines import algorithm1, algorithm2, algorithm3, algorithm4, cas_rag

logger = get_logger(__name__)

RUNNERS = {
    AlgorithmId.ALGORITHM_1: algorithm1.run_algorithm1,
    AlgorithmId.ALGORITHM_2: algorithm2.run_algorithm2,
    AlgorithmId.ALGORITHM_3: algorithm3.run_algorithm3,
    AlgorithmId.ALGORITHM_4: algorithm4.run_algorithm4,
    AlgorithmId.CAS_RAG: cas_rag.run_cas_rag,
}


class PipelineEngine:
    """Central orchestrator for all CAS-RAG research pipelines."""

    def __init__(
        self,
        algorithm: AlgorithmId | str,
        llm_provider: str | None = None,
        video_slug: str | None = "video_02_education_tutoring_diner",
        experiments_root: Path | None = None,
        human_queue_path: Path | None = None,
    ):
        settings = get_settings()
        self.algorithm = AlgorithmId(algorithm) if isinstance(algorithm, str) else algorithm
        self.settings = settings
        exp_root = experiments_root or (settings.project_root / "resources" / "datasets" / "Experiments-data")

        if video_slug:
            video_dir = exp_root / video_slug
            scenario_path = video_dir / "scenario_full.txt"
            script_path = video_dir / "script_corpus.txt"
            prototypes_path = video_dir / "prototypes_semantic.json"
            prompts_dir = video_dir / "prompts"
        else:
            scenario_path = settings.project_root / "resources" / "scenarios" / "tutoring_social_experiment.txt"
            script_path = settings.project_root / "resources" / "datasets" / "sample" / "script_corpus.txt"
            prototypes_path = settings.project_root / "resources" / "prototypes" / "prototypes.json"
            prompts_dir = settings.project_root / "resources" / "prompts" / settings.prompt_version

        self.scenario = scenario_path.read_text(encoding="utf-8")
        self.ctx = PipelineContext(
            algorithm=self.algorithm,
            scenario=self.scenario,
            script_corpus_path=str(script_path),
            llm=create_llm_provider(llm_provider),
            prompts=PromptManager(base_dir=prompts_dir),
            rag=RAGEngine(model_name=settings.embedding_model, top_k=settings.retrieval_top_k),
            semantic=SemanticSimilarityEngine(prototypes_path=prototypes_path),
            spacy=SpacyLinguisticAnalyzer(),
            human=HumanVerificationService(queue_path=human_queue_path),
        )

    def process(self, inp: CommentInput) -> PipelineResult:
        runner = RUNNERS[self.algorithm]
        result = runner(self.ctx, inp)
        logger.info("pipeline_complete", algorithm=self.algorithm.value, comment_id=inp.comment_id, status=result.status.value)
        return result

    def process_dataset(self, csv_path: str | Path) -> list[PipelineResult]:
        loader = DatasetLoader()
        comments = loader.load_comments(csv_path)
        return [self.process(c) for c in comments]

    def compare_algorithms(self, inp: CommentInput) -> dict[str, PipelineResult]:
        out = {}
        for alg in AlgorithmId:
            engine = PipelineEngine(alg, llm_provider=self.settings.llm_provider)
            out[alg.value] = engine.process(inp)
        return out
