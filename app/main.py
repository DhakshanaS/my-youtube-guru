"""FastAPI entrypoint for My YouTube Guru.

Run locally with:

    uvicorn app.main:app --reload

Architecture note: this file only wires things together (app factory,
middleware, routers, static frontend). All real logic lives in
`app/services/*`, which are deliberately framework-free so they can be
reused from scripts and unit tests. Routers in `app/routers/*` are thin
HTTP adapters over those services.

Build order (routers are included here as each module lands):
    Module 2  → services/takeout_parser.py          (done)
    Module 3  → services/{embeddings,vector_store,llm_service}.py + ingestion
    Module 5  → services/{transcripts,rag_pipeline}.py
    Module 7  → routers/{upload,chat,knowledge_base,settings}.py
    Module 8  → frontend/ (served below as static files)
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.models.schemas import HealthResponse
from app.routers import chat, evaluation, knowledge_base, sessions, settings as settings_router, upload

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def create_app() -> FastAPI:
    """Application factory — keeps app creation testable and import-safe."""
    settings = get_settings()

    app = FastAPI(
        title="My YouTube Guru",
        version="0.1.0",
        description=(
            "RAG over your own YouTube watch history: Takeout parsing → LLM "
            "categorisation → sentence-transformers embeddings → ChromaDB → "
            "grounded question answering with lazy transcript fetching."
        ),
    )

    # Dev-friendly CORS. The frontend is normally served by this same process
    # (below), so this mainly helps when running the UI from another origin
    # during development. Lock this down before any real deployment.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Stop the browser caching the frontend (HTML/CSS/JS). Without this, editing
    # a file and reloading can keep running the OLD cached version — a classic
    # "I changed it but nothing happened" trap during development. For a local
    # single-user app there's no downside to always serving fresh assets.
    @app.middleware("http")
    async def no_cache_frontend(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    @app.get("/api/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        """Liveness probe; also confirms which model config is active."""
        return HealthResponse(
            status="ok",
            llm_provider=settings.llm_provider,
            embedding_model=settings.embedding_model_name,
        )

    # REST API routers (Module 7).
    app.include_router(settings_router.router)
    app.include_router(upload.router)
    app.include_router(chat.router)
    app.include_router(knowledge_base.router)
    app.include_router(sessions.router)
    app.include_router(evaluation.router)

    # Serve the vanilla-JS frontend from the same process — one command runs
    # the whole demo. Mounted LAST so it doesn't shadow the /api/* routes.
    # Guarded so the API works before the frontend (Module 8) exists.
    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    return app


app = create_app()
