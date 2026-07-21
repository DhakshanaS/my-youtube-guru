"""Grounding evaluation endpoints (Module 9).

  GET /api/evaluation/metrics   aggregate grounding metrics across all answers
  GET /api/evaluation/log       the recent answer audit trail (what sources each
                                answer used, its mode, and retrieval confidence)

These power the "Grounding" page, which is how hallucination control is
inspected and verified.
"""

from fastapi import APIRouter, Query

from app.models.schemas import (
    EvalLogEntry,
    EvaluationLogResponse,
    EvaluationMetrics,
)
from app.services.grounding_log import grounding_log

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


@router.get("/metrics", response_model=EvaluationMetrics)
def metrics() -> EvaluationMetrics:
    return EvaluationMetrics(**grounding_log.metrics())


@router.get("/log", response_model=EvaluationLogResponse)
def log(limit: int = Query(50, ge=1, le=500)) -> EvaluationLogResponse:
    entries = grounding_log.entries(limit=limit)
    return EvaluationLogResponse(
        count=len(entries),
        entries=[EvalLogEntry(**e) for e in entries],
    )
