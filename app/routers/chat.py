"""Chat endpoints — the RAG question-answering surface.

  POST /api/chat/ask          grounded answer (single JSON response)
  POST /api/chat/ask/stream   same, but streams live progress events (SSE) so
                              the UI can show a "thinking" trace during the wait
  POST /api/chat/confirm[/stream]   general-knowledge fallback (after user OK)

The /stream variants run the (synchronous) RAG pipeline on a worker thread and
forward its progress events to the client as Server-Sent Events. The pipeline's
own steps (search → retrieve → fetch transcripts → generate) are the events —
nothing is faked.
"""

import asyncio
import json
import threading
from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import (
    AskRequest,
    AskResponse,
    ConfirmRequest,
    RetrievalItem,
    SourceModel,
)
from app.services.llm_service import LLMNotConfiguredError
from app.services.rag_pipeline import RAGResponse, rag_pipeline

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _to_response(r: RAGResponse) -> AskResponse:
    """Map the internal RAG dataclass onto the API response model."""
    return AskResponse(
        answer=r.answer,
        grounded=r.grounded,
        from_general_knowledge=r.from_general_knowledge,
        needs_confirmation=r.needs_confirmation,
        question=r.question,
        sources=[SourceModel(**asdict(s)) for s in r.sources],
        retrieval=[RetrievalItem(**item) for item in r.retrieval],
    )


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Answer a question grounded strictly in the user's watch history."""
    history = [t.model_dump() for t in req.history] if req.history else None
    try:
        result = rag_pipeline.answer_question(req.question, history=history, top_k=req.top_k)
    except LLMNotConfiguredError as exc:
        # 400: the client must set an API key on the Settings page first.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _to_response(result)


@router.post("/confirm", response_model=AskResponse)
def confirm_general_knowledge(req: ConfirmRequest) -> AskResponse:
    """Answer from the LLM's general knowledge (post user confirmation)."""
    history = [t.model_dump() for t in req.history] if req.history else None
    try:
        result = rag_pipeline.answer_from_general_knowledge(req.question, history=history)
    except LLMNotConfiguredError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _to_response(result)


# ── Streaming (Server-Sent Events) ───────────────────────────────────────────

async def _sse_from_pipeline(run) -> StreamingResponse:
    """Drive a synchronous pipeline call on a worker thread and stream its
    progress events to the client as SSE.

    `run(emit)` performs the work and must call emit("final", <dict>) at the
    end. Errors are converted into an "error" event so the client always gets
    a clean terminal frame. A thread-safe bridge (call_soon_threadsafe) moves
    events from the worker thread onto the event loop's queue.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(kind: str, data: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, {"kind": kind, "data": data})

    def worker() -> None:
        try:
            run(emit)
        except LLMNotConfiguredError as exc:
            emit("error", {"detail": str(exc), "code": "no_key"})
        except Exception as exc:  # noqa: BLE001 — report any failure to the client
            emit("error", {"detail": str(exc)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # end sentinel

    threading.Thread(target=worker, daemon=True).start()

    async def event_stream():
        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """Grounded answer with a live progress stream (SSE)."""
    history = [t.model_dump() for t in req.history] if req.history else None
    def run(emit):
        result = rag_pipeline.answer_question(req.question, history=history,
                                              top_k=req.top_k, on_event=emit)
        emit("final", _to_response(result).model_dump())
    return await _sse_from_pipeline(run)


@router.post("/confirm/stream")
async def confirm_stream(req: ConfirmRequest):
    """General-knowledge answer with a live progress stream (SSE)."""
    history = [t.model_dump() for t in req.history] if req.history else None
    def run(emit):
        result = rag_pipeline.answer_from_general_knowledge(req.question, history=history,
                                                            on_event=emit)
        emit("final", _to_response(result).model_dump())
    return await _sse_from_pipeline(run)
