"""Application configuration for My YouTube Guru.

All secrets and tunables come from environment variables (or a local `.env`
file, see `.env.example`). Nothing sensitive is ever hardcoded — the LLM API
key in particular is normally supplied at runtime through the Settings page
in the UI; the environment value only acts as an optional fallback.

`pydantic-settings` gives us typed, validated, self-documenting configuration
(12-factor style), which also makes every setting easy to override in tests.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central typed settings object. Field names map 1:1 to env variables."""

    # ── LLM provider (DeepSeek exposes an OpenAI-compatible API) ─────────
    llm_provider: str = "deepseek"
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    # Optional fallback only — the primary way to set the key is the UI.
    llm_api_key: str = ""

    # ── Embeddings & vector store ────────────────────────────────────────
    embedding_model_name: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "youtube_history"
    # SQLite database for chat session history (sidebar of past conversations).
    chat_db_path: str = "./data/chat.db"
    # Append-only JSONL audit log of every answer's grounding (Module 9).
    grounding_log_path: str = "./data/grounding_log.jsonl"

    # ── Retrieval tuning (RAG query flow) ────────────────────────────────
    # How many candidates to pull from the vector store before filtering.
    retrieval_candidates: int = 15
    # Max videos actually used in an answer (after threshold filtering). The
    # number used is ADAPTIVE: every candidate above the threshold is included,
    # up to this cap. Raise for more sources per answer (slower + more tokens).
    top_k_results: int = 6
    # Minimum cosine similarity for a retrieved video to count as "relevant".
    # Below this for the best hit, the app declines to answer from the KB and
    # offers general knowledge instead. Tunable heuristic — see README.
    similarity_threshold: float = 0.25
    # Max characters of each transcript included in the grounding prompt
    # (keeps token cost bounded; the full transcript is still cached).
    transcript_char_budget: int = 6000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # unknown env vars are simply ignored
    )


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so the .env file is read once per process.

    Using a function (rather than a module-level instance) keeps imports
    side-effect free and lets tests call `get_settings.cache_clear()` after
    monkeypatching the environment.
    """
    return Settings()
