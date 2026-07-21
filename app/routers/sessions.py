"""Chat session endpoints — conversation history for the sidebar.

  POST   /api/sessions              create a new chat
  GET    /api/sessions              list chats (newest first)
  GET    /api/sessions/{id}         open a chat (with its messages)
  PATCH  /api/sessions/{id}         rename a chat
  DELETE /api/sessions/{id}         delete a chat (and its messages)
  POST   /api/sessions/{id}/messages   append a message

The frontend persists each turn here after streaming an answer, so the full
conversation — including each answer's cited sources — can be reloaded later.
"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    AddMessageRequest,
    CreateSessionRequest,
    DeletedResponse,
    MessageModel,
    RenameSessionRequest,
    SessionDetailResponse,
    SessionModel,
    SessionsResponse,
)
from app.services.chat_store import chat_store

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=SessionModel)
def create_session(req: CreateSessionRequest) -> SessionModel:
    return SessionModel(**chat_store.create_session(req.title))


@router.get("", response_model=SessionsResponse)
def list_sessions() -> SessionsResponse:
    return SessionsResponse(
        sessions=[SessionModel(**s) for s in chat_store.list_sessions()]
    )


@router.get("/{session_id}", response_model=SessionDetailResponse)
def get_session(session_id: str) -> SessionDetailResponse:
    session = chat_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return SessionDetailResponse(**session)


@router.patch("/{session_id}", response_model=SessionModel)
def rename_session(session_id: str, req: RenameSessionRequest) -> SessionModel:
    updated = chat_store.rename_session(session_id, req.title)
    if updated is None:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return SessionModel(**updated)


@router.delete("/{session_id}", response_model=DeletedResponse)
def delete_session(session_id: str) -> DeletedResponse:
    if not chat_store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Chat not found.")
    return DeletedResponse(deleted=True)


@router.post("/{session_id}/messages", response_model=MessageModel)
def add_message(session_id: str, req: AddMessageRequest) -> MessageModel:
    msg = chat_store.add_message(session_id, req.role, req.content, req.data)
    if msg is None:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return MessageModel(**msg)
