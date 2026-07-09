"""Required human-in-the-loop verification workflow."""

from __future__ import annotations

import json
import uuid
from cas_rag.core.datetime_utils import utc_now_iso
from pathlib import Path

from cas_rag.config.settings import get_settings
from cas_rag.core.enums import EmotionLabel
from cas_rag.core.models import HumanReviewDecision


class HumanVerificationService:
    """Persistent queue for ambiguous / unresolvable comments."""

    def __init__(self, queue_path: Path | None = None):
        settings = get_settings()
        cfg = settings.load_yaml().get("human_verification", {})
        self.queue_path = queue_path or (settings.project_root / cfg.get("queue_file", "data/human_verification_queue.json"))
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.queue_path.exists():
            self._save({"pending": [], "completed": []})

    def _load(self) -> dict:
        return json.loads(self.queue_path.read_text())

    def _save(self, data: dict) -> None:
        self.queue_path.write_text(json.dumps(data, indent=2, default=str))

    def enqueue(
        self,
        comment_id: str,
        comment: str,
        reason: str,
        algorithm: str,
        context: dict | None = None,
    ) -> str:
        data = self._load()
        qid = str(uuid.uuid4())
        data["pending"].append(
            {
                "queue_id": qid,
                "comment_id": comment_id,
                "comment": comment,
                "reason": reason,
                "algorithm": algorithm,
                "context": context or {},
                "created_at": utc_now_iso(),
            }
        )
        self._save(data)
        return qid

    def submit_review(self, queue_id: str, decision: HumanReviewDecision) -> HumanReviewDecision:
        data = self._load()
        pending = data["pending"]
        item = next((p for p in pending if p["queue_id"] == queue_id), None)
        if not item:
            raise KeyError(f"Queue item not found: {queue_id}")
        data["pending"] = [p for p in pending if p["queue_id"] != queue_id]
        record = {**item, "decision": decision.model_dump()}
        data["completed"].append(record)
        self._save(data)
        return decision

    def get_pending(self) -> list[dict]:
        return self._load().get("pending", [])

    def resolve_for_comment(self, comment_id: str) -> HumanReviewDecision | None:
        """Return completed review if available."""
        for item in self._load().get("completed", []):
            if item.get("comment_id") == comment_id and "decision" in item:
                return HumanReviewDecision(**item["decision"])
        return None

    def apply_review(self, decision: HumanReviewDecision) -> EmotionLabel | None:
        """Thesis logic: confirm ambiguous -> discard; else assign label."""
        if decision.confirms_ambiguous and not decision.resolvable:
            return EmotionLabel.DISCARDED
        return decision.assigned_label
