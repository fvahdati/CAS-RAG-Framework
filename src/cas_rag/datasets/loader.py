"""Dataset loading and management."""

from pathlib import Path

import pandas as pd

from cas_rag.core.enums import EmotionLabel
from cas_rag.core.models import CommentInput


class DatasetLoader:
    def load_comments(self, path: str | Path) -> list[CommentInput]:
        df = pd.read_csv(path)
        rows = []
        for idx, r in df.iterrows():
            gold = None
            if "gold_label" in r and pd.notna(r["gold_label"]):
                try:
                    gold = EmotionLabel(str(r["gold_label"]).strip())
                except ValueError:
                    gold = None
            rows.append(
                CommentInput(
                    comment_id=str(r.get("comment_id", idx)),
                    comment=str(r["comment"]),
                    gold_label=gold,
                )
            )
        return rows

    def save_results(self, results: list, path: str | Path) -> None:
        records = [r.model_dump() if hasattr(r, "model_dump") else r for r in results]
        pd.DataFrame(records).to_csv(path, index=False)
