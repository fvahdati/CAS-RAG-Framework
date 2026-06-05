"""Analytics aggregations for dashboard visualizations."""

from __future__ import annotations

import pandas as pd

from cas_rag.core.models import PipelineResult
from cas_rag.evaluation.metrics import KPIProcessor


class AnalyticsEngine:
    def build_report(self, results: list[PipelineResult], gold: dict[str, str] | None = None) -> dict:
        kpi = KPIProcessor().compute(results, gold)
        df = pd.DataFrame([r.model_dump() for r in results])
        report = {"kpi": kpi, "charts": {}}
        if df.empty:
            return report

        if "emotion_label" in df.columns:
            dist = df["emotion_label"].value_counts().to_dict()
            report["charts"]["emotion_distribution"] = {str(k): int(v) for k, v in dist.items()}
            total = max(len(df), 1)
            empathy_n = int(dist.get("Empathy", 0))
            report["charts"]["empathy_percentage"] = round(100 * empathy_n / total, 2)
            anger = int(dist.get("Anger", 0)) + int(dist.get("Frustration", 0))
            report["charts"]["anger_frustration_share"] = round(100 * anger / total, 2)

        report["charts"]["ambiguous_pending"] = int(df.get("pending_human", pd.Series([False])).sum())
        report["charts"]["rag_vs_no_rag"] = {
            "rag": int(df.get("rag_used", pd.Series([False])).sum()),
            "no_rag": int((~df.get("rag_used", pd.Series([False]))).sum()),
        }
        if "semantic_category" in df.columns:
            report["charts"]["semantic_categories"] = df["semantic_category"].value_counts().to_dict()
        return report
