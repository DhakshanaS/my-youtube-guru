"""Knowledge-base endpoints — data for the visualisation page (Module 8).

  GET /api/knowledge-base/stats        total videos indexed
  GET /api/knowledge-base/categories   per-category counts (feeds Chart.js)
  GET /api/knowledge-base/videos       browse stored videos (optionally by category)
"""

from fastapi import APIRouter, Query

from app.models.schemas import (
    CategoriesResponse,
    CategoryCount,
    KBStatsResponse,
    VideoModel,
    VideosResponse,
)
from app.services.vector_store import vector_store

router = APIRouter(prefix="/api/knowledge-base", tags=["knowledge-base"])


@router.get("/stats", response_model=KBStatsResponse)
def kb_stats() -> KBStatsResponse:
    return KBStatsResponse(total_videos=vector_store.count())


@router.get("/categories", response_model=CategoriesResponse)
def kb_categories() -> CategoriesResponse:
    """Video counts per category, sorted desc — the visualisation's data source."""
    counts = vector_store.category_counts()
    return CategoriesResponse(
        total=sum(counts.values()),
        categories=[CategoryCount(category=c, count=n) for c, n in counts.items()],
    )


@router.get("/videos", response_model=VideosResponse)
def kb_videos(
    category: str | None = Query(None, description="Filter to a single category"),
    limit: int = Query(100, ge=1, le=1000),
) -> VideosResponse:
    """List stored videos (metadata only — transcripts are never returned here)."""
    rows = vector_store.list_videos(category=category, limit=limit)
    videos = [
        VideoModel(
            video_id=r.get("video_id", ""),
            title=r.get("title", ""),
            url=r.get("url", ""),
            channel=r.get("channel") or None,
            category=r.get("category"),
            watch_count=int(r.get("watch_count", 1) or 1),
            transcript_fetched=bool(r.get("transcript_fetched", False)),
            transcript_available=r.get("transcript_available"),
        )
        for r in rows
    ]
    return VideosResponse(count=len(videos), videos=videos)
