"""Embedding service — sentence-transformers `all-MiniLM-L6-v2` (Module 3).

Turns video content (and later, the user's questions) into 384-dimensional
vectors for semantic similarity search in ChromaDB.

Two deliberate choices:
  * Lazy loading — the model (~80 MB) is downloaded/loaded on first use, not
    at import. This keeps API startup fast and lets modules that only need the
    class (e.g. tests injecting a fake) import it without pulling in torch.
  * L2-normalised outputs — combined with ChromaDB's cosine space, this makes
    similarity scores well-behaved and comparable across queries.
"""

from __future__ import annotations

import logging
import threading

from app.config import get_settings

logger = logging.getLogger(__name__)


def build_embedding_text(title: str, channel: str | None,
                         transcript: str | None = None) -> str:
    """Compose the text that represents a video for embedding.

    Title + channel captures the topic well for the initial index. Once a
    transcript is fetched (Module 5) it can be folded in for a richer vector;
    we cap its length so one long video can't dominate the representation.
    """
    parts = [title or ""]
    if channel:
        parts.append(f"Channel: {channel}")
    if transcript:
        parts.append(transcript[:2000])
    return "\n".join(p for p in parts if p)


class EmbeddingModel:
    """Thin wrapper around a SentenceTransformer with lazy loading."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_settings().embedding_model_name
        self._model = None  # loaded on first embed
        self._lock = threading.Lock()  # guards lazy load against concurrent calls

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:  # double-checked locking
                    # Imported here (not at module top) so importing this file is
                    # cheap and torch is only required when embeddings run.
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model '%s' (first use may "
                                "download ~80 MB)...", self.model_name)
                    self._model = SentenceTransformer(self.model_name)
                    logger.info("Embedding model ready (dim=%d).",
                                _embedding_dim(self._model))
        return self._model

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Embed many texts at once. Returns plain Python lists for ChromaDB."""
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,   # unit vectors → clean cosine scores
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text (e.g. a user's question in the RAG query flow)."""
        return self.embed_texts([text])[0]

    @property
    def dimension(self) -> int:
        return _embedding_dim(self._load())


def _embedding_dim(model) -> int:
    """Return the embedding dimension across sentence-transformers versions.

    v5 renamed `get_sentence_embedding_dimension` → `get_embedding_dimension`.
    Try the new name first, fall back to the old one so both work.
    """
    getter = getattr(model, "get_embedding_dimension", None) or \
        model.get_sentence_embedding_dimension
    return getter()


# Process-wide singleton, configured from settings.
embedding_model = EmbeddingModel()
