"""Upload endpoints — ingest a Google Takeout export into the knowledge base.

Two-step design:
  POST /api/upload/takeout   validates + parses the file immediately (so bad
                             uploads fail fast with 422), then kicks off the
                             slow ingest on a background thread and returns a
                             job id right away.
  GET  /api/upload/status/{job_id}   poll for progress until status == "done".
"""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import JobStatusResponse, UploadStartResponse
from app.services.ingestion import ingest_videos
from app.services.llm_service import llm_service
from app.services.takeout_parser import TakeoutParseError, parse_takeout
from app.utils.jobs import job_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["upload"])


@router.post("/takeout", response_model=UploadStartResponse)
async def upload_takeout(
    file: UploadFile = File(..., description="Google Takeout .zip (or watch-history .html/.json)"),
    categorize: bool = True,
) -> UploadStartResponse:
    """Parse a Takeout upload and start ingesting it in the background."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # Validate + parse now, so malformed files are rejected immediately rather
    # than failing silently inside a background job.
    try:
        parse_result = parse_takeout(data)
    except TakeoutParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Categorisation needs a key; if none is set we still ingest (uncategorised)
    # so the user isn't blocked — they can add a key and re-ask later.
    use_llm = categorize and llm_service.is_configured()

    job = job_manager.create(total=parse_result.stats.unique_videos)

    def worker(progress_cb) -> dict:
        stats = ingest_videos(parse_result.videos, use_llm=use_llm,
                              progress_cb=progress_cb)
        return {
            "parse_stats": parse_result.stats.to_dict(),
            "ingest_stats": stats.to_dict(),
        }

    job_manager.run(job, worker)
    logger.info("Upload accepted: %d unique videos, job %s (categorize=%s)",
                parse_result.stats.unique_videos, job.id, use_llm)

    return UploadStartResponse(
        job_id=job.id, status=job.status,
        unique_videos=parse_result.stats.unique_videos,
        categorization_enabled=use_llm,
    )


@router.get("/status/{job_id}", response_model=JobStatusResponse)
def upload_status(job_id: str) -> JobStatusResponse:
    """Poll the progress/result of a background ingest job."""
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return JobStatusResponse(**job.to_dict())
