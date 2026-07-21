"""Grounding evaluation log (Module 9).

An append-only audit trail of every answer the assistant produces: whether it
stayed grounded in the user's videos, which sources it cited, and how confident
retrieval was. Controlling hallucination is the core of this project — this log
is the evidence that the grounding actually works, and the data behind the
"Grounding" inspection page.

Stored as JSONL (one JSON object per line) at `grounding_log_path`: a simple,
human-readable, append-only format. Aggregate metrics are computed on read.
Writing is best-effort and never raised into the answer path — logging must
never break answering.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone

from app.config import get_settings

logger = logging.getLogger(__name__)


class GroundingLog:
    def __init__(self, path: str | None = None) -> None:
        self._path = path or get_settings().grounding_log_path
        self._lock = threading.Lock()
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    # ── write ────────────────────────────────────────────────────────────
    def record(self, response) -> None:
        """Append one answer's grounding record. Best-effort; never raises."""
        try:
            mode = ("no_match" if response.needs_confirmation
                    else "general_knowledge" if response.from_general_knowledge
                    else "grounded")
            retrieval = list(response.retrieval or [])
            best = retrieval[0]["similarity"] if retrieval else None
            record = {
                "id": uuid.uuid4().hex,
                "ts": datetime.now(timezone.utc).isoformat(),
                "question": response.question,
                "mode": mode,
                "num_sources": len(response.sources),
                "transcript_sources": sum(1 for s in response.sources if s.transcript_used),
                "best_similarity": round(best, 4) if best is not None else None,
                "answer_chars": len(response.answer or ""),
                "sources": [
                    {"video_id": s.video_id, "title": s.title,
                     "similarity": s.similarity, "transcript_used": s.transcript_used}
                    for s in response.sources
                ],
                "retrieval": retrieval,
            }
            line = json.dumps(record, ensure_ascii=False)
            with self._lock:
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:  # noqa: BLE001 — logging must never break answering
            logger.debug("Failed to write grounding log", exc_info=True)

    # ── read ─────────────────────────────────────────────────────────────
    def _read_all(self) -> list[dict]:
        if not os.path.exists(self._path):
            return []
        rows: list[dict] = []
        with self._lock:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # skip a corrupt line rather than fail
        return rows

    def entries(self, limit: int = 100) -> list[dict]:
        """Most recent records first."""
        return list(reversed(self._read_all()))[:limit]

    def metrics(self) -> dict:
        """Aggregate grounding metrics across all logged answers."""
        rows = self._read_all()
        total = len(rows)
        grounded = [r for r in rows if r.get("mode") == "grounded"]
        general = [r for r in rows if r.get("mode") == "general_knowledge"]
        no_match = [r for r in rows if r.get("mode") == "no_match"]

        def pct(n: int) -> float:
            return round(100.0 * n / total, 1) if total else 0.0

        total_sources = sum(r.get("num_sources", 0) for r in grounded)
        transcript_sources = sum(r.get("transcript_sources", 0) for r in grounded)
        sims = [r["best_similarity"] for r in grounded if r.get("best_similarity") is not None]

        # Which of the user's videos most often actually answer their questions.
        counter: Counter = Counter()
        titles: dict[str, str] = {}
        for r in grounded:
            for s in r.get("sources", []):
                counter[s["video_id"]] += 1
                titles[s["video_id"]] = s.get("title", s["video_id"])
        top_sources = [
            {"video_id": vid, "title": titles.get(vid, vid), "count": n}
            for vid, n in counter.most_common(5)
        ]

        return {
            "total_questions": total,
            "grounded": len(grounded), "grounded_pct": pct(len(grounded)),
            "general_knowledge": len(general), "general_knowledge_pct": pct(len(general)),
            "no_match": len(no_match), "no_match_pct": pct(len(no_match)),
            "avg_sources": round(total_sources / len(grounded), 1) if grounded else 0.0,
            "avg_best_similarity": round(sum(sims) / len(sims), 3) if sims else 0.0,
            "transcript_coverage_pct": (
                round(100.0 * transcript_sources / total_sources, 1) if total_sources else 0.0
            ),
            "top_sources": top_sources,
        }


# Process-wide singleton.
grounding_log = GroundingLog()
