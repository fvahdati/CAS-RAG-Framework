"""Batch experiment runner for algorithm comparison."""

import json
from pathlib import Path

from cas_rag.core.datetime_utils import utc_now

from cas_rag.config.settings import get_settings
from cas_rag.core.enums import AlgorithmId
from cas_rag.datasets.loader import DatasetLoader
from cas_rag.evaluation.metrics import KPIProcessor
from cas_rag.orchestration.engine import PipelineEngine


class ExperimentRunner:
    def __init__(self, output_dir: Path | None = None):
        settings = get_settings()
        self.output_dir = output_dir or (settings.project_root / "data" / "evaluation_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, algorithm: AlgorithmId, dataset_path: Path, llm_provider: str | None = None) -> Path:
        engine = PipelineEngine(algorithm, llm_provider=llm_provider)
        results = engine.process_dataset(dataset_path)
        loader = DatasetLoader()
        comments = loader.load_comments(dataset_path)
        gold = {c.comment_id: c.gold_label.value for c in comments if c.gold_label}
        kpi = KPIProcessor().compute(results, gold)
        stamp = utc_now().strftime("%Y%m%d_%H%M%S")
        out = self.output_dir / f"{algorithm.value}_{stamp}.json"
        payload = {
            "algorithm": algorithm.value,
            "kpi": kpi,
            "results": [r.model_dump() for r in results],
        }
        out.write_text(json.dumps(payload, indent=2, default=str))
        return out

    def compare_all(self, dataset_path: Path) -> Path:
        summary = {}
        for alg in AlgorithmId:
            path = self.run(alg, dataset_path)
            summary[alg.value] = json.loads(path.read_text())["kpi"]
        out = self.output_dir / f"comparison_{utc_now().strftime('%Y%m%d_%H%M%S')}.json"
        out.write_text(json.dumps(summary, indent=2))
        return out
