"""Settings endpoints — configure the LLM provider / API key at runtime.

The key entered here lives only in memory (in the LLM service singleton) for
the life of the process; it is never written to disk or returned in any
response. This is what lets the user supply their key through the UI instead of
editing files.
"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import SettingsRequest, SettingsResponse
from app.services.llm_service import llm_service

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("/llm", response_model=SettingsResponse)
def get_llm_settings() -> SettingsResponse:
    """Return the active provider/model and whether a key is configured."""
    return SettingsResponse(**llm_service.status())


@router.post("/llm", response_model=SettingsResponse)
def update_llm_settings(req: SettingsRequest) -> SettingsResponse:
    """Set/replace the API key (and optionally provider, model, base URL)."""
    if not any([req.api_key, req.base_url, req.model, req.provider]):
        raise HTTPException(status_code=400,
                            detail="Provide at least one field to update.")
    llm_service.configure(
        api_key=req.api_key, base_url=req.base_url,
        model=req.model, provider=req.provider,
    )
    return SettingsResponse(**llm_service.status())
