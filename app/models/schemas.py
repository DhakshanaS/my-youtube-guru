"""Pydantic request/response models for the REST API (Module 7).

Centralising these gives three things:
  * validated, typed request bodies (FastAPI rejects bad input automatically);
  * stable, documented response shapes; and
  * a rich auto-generated OpenAPI page at /docs — which doubles as living
    documentation of the API for the portfolio.

These API models are kept separate from the internal service dataclasses
(WatchedVideo, RAGResponse, …) so the wire format can evolve independently of
the implementation. Routers convert between the two.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Meta ─────────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    embedding_model: str


# ── Settings (API key) ───────────────────────────────────────────────────────
class SettingsRequest(BaseModel):
    """Update the active LLM provider. All fields optional; send what changes."""
    api_key: str | None = Field(None, description="Provider API key (stored in memory only)")
    base_url: str | None = Field(None, description="OpenAI-compatible base URL")
    model: str | None = Field(None, description="Model name, e.g. 'deepseek-chat'")
    provider: str | None = Field(None, description="Provider label, e.g. 'deepseek'")


class SettingsResponse(BaseModel):
    """Safe view of provider config — never includes the key itself."""
    provider: str
    model: str
    base_url: str
    configured: bool = Field(..., description="True once an API key is set")


# ── Upload / ingestion ───────────────────────────────────────────────────────
class UploadStartResponse(BaseModel):
    job_id: str
    status: str
    unique_videos: int = Field(..., description="Unique videos parsed from the upload")
    categorization_enabled: bool = Field(
        ..., description="False if no API key is set (videos ingested uncategorised)"
    )


class JobStatusResponse(BaseModel):
    job_id: str
    status: str = Field(..., description="running | done | error")
    phase: str
    done: int
    total: int
    result: dict | None = Field(None, description="parse_stats + ingest_stats when done")
    error: str | None = None


# ── Chat / RAG ───────────────────────────────────────────────────────────────
class ChatTurn(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    top_k: int | None = Field(None, ge=1, le=20, description="Override max videos used")
    history: list[ChatTurn] | None = Field(
        None, description="Recent conversation turns, for follow-up context"
    )


class ConfirmRequest(BaseModel):
    """Sent after the user agrees to a general-knowledge answer for a question
    that wasn't found in their watch history."""
    question: str = Field(..., min_length=1)
    history: list[ChatTurn] | None = None


class SourceModel(BaseModel):
    video_id: str
    title: str
    url: str
    similarity: float
    transcript_used: bool
    note: str | None = None


class RetrievalItem(BaseModel):
    video_id: str
    title: str
    similarity: float


class AskResponse(BaseModel):
    answer: str
    grounded: bool = Field(..., description="Answered from watch-history transcripts")
    from_general_knowledge: bool = Field(..., description="Answered from the LLM's own knowledge")
    needs_confirmation: bool = Field(..., description="Nothing relevant found; awaiting user's OK for general knowledge")
    question: str
    sources: list[SourceModel] = []
    retrieval: list[RetrievalItem] = Field(
        default=[], description="What was retrieved and its similarity (grounding trail)"
    )


# ── Knowledge base ───────────────────────────────────────────────────────────
class CategoryCount(BaseModel):
    category: str
    count: int


class CategoriesResponse(BaseModel):
    total: int
    categories: list[CategoryCount]


class KBStatsResponse(BaseModel):
    total_videos: int


class VideoModel(BaseModel):
    video_id: str
    title: str
    url: str
    channel: str | None = None
    category: str | None = None
    watch_count: int = 1
    transcript_fetched: bool = False
    transcript_available: bool | None = None


class VideosResponse(BaseModel):
    count: int
    videos: list[VideoModel]


# ── Chat sessions (conversation history) ─────────────────────────────────────
from typing import Literal  # noqa: E402  (kept local to this feature block)


class SessionModel(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class SessionsResponse(BaseModel):
    sessions: list[SessionModel]


class CreateSessionRequest(BaseModel):
    title: str | None = Field(None, description="Optional initial title")


class RenameSessionRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class MessageModel(BaseModel):
    id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    data: dict | None = Field(None, description="Assistant turn payload (sources, flags) for re-rendering")
    created_at: str


class SessionDetailResponse(SessionModel):
    messages: list[MessageModel] = []


class AddMessageRequest(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    data: dict | None = None


class DeletedResponse(BaseModel):
    deleted: bool


# ── Grounding evaluation (Module 9) ──────────────────────────────────────────
class TopSource(BaseModel):
    video_id: str
    title: str
    count: int


class EvaluationMetrics(BaseModel):
    total_questions: int
    grounded: int
    grounded_pct: float
    general_knowledge: int
    general_knowledge_pct: float
    no_match: int
    no_match_pct: float
    avg_sources: float
    avg_best_similarity: float
    transcript_coverage_pct: float
    top_sources: list[TopSource]


class EvalSource(BaseModel):
    video_id: str
    title: str
    similarity: float
    transcript_used: bool


class EvalLogEntry(BaseModel):
    id: str
    ts: str
    question: str
    mode: str = Field(..., description="grounded | general_knowledge | no_match")
    num_sources: int
    transcript_sources: int = 0
    best_similarity: float | None = None
    answer_chars: int = 0
    sources: list[EvalSource] = []


class EvaluationLogResponse(BaseModel):
    count: int
    entries: list[EvalLogEntry]
