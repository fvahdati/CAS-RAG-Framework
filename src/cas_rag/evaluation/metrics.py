"""KPI processing: Accuracy, Precision, Recall, F1."""

from __future__ import annotations

from collections import Counter

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from cas_rag.core.models import PipelineResult


class KPIProcessor:
    def compute(self, results: list[PipelineResult], gold_labels: dict[str, str] | None = None) -> dict:
        labelled = [r for r in results if r.emotion_label and r.status.value == "success"]
        df = pd.DataFrame([r.model_dump() for r in results])
        metrics = {
            "total": len(results),
            "labelled": len(labelled),
            "discarded": sum(1 for r in results if r.discarded),
            "pending_human": sum(1 for r in results if r.pending_human),
            "rag_used_count": sum(1 for r in results if r.rag_used),
        }
        if gold_labels and labelled:
            y_true, y_pred = [], []
            for r in labelled:
                if r.comment_id in gold_labels:
                    y_true.append(gold_labels[r.comment_id])
                    y_pred.append(r.emotion_label.value if r.emotion_label else "None")
            if y_true:
                labels = sorted(set(y_true) | set(y_pred))
                metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
                metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels))
                metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels))
                metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels))
        if not df.empty and "emotion_label" in df.columns:
            counts = Counter(str(x) for x in df["emotion_label"].dropna())
            metrics["emotion_distribution"] = dict(counts)
        return metrics
