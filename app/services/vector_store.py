"""Vector store — persistent ChromaDB wrapper (Module 3).

Owns all interaction with ChromaDB: storing video embeddings + metadata,
running similarity search for the RAG loop (Module 5), de-duplicating videos
on re-upload (Module 4), caching fetched transcripts (Module 5), and reporting
category counts for the knowledge-base visualisation (Module 8).

Design notes
------------
* We provide our OWN embeddings (from sentence-transformers) on every add and
  query, and never rely on Chroma's built-in embedding function. This keeps
  the RAG pipeline explicit and provider-controlled.
* The collection uses cosine space; Chroma returns distance = 1 - cosine
  similarity, so we expose `similarity = 1 - distance` for easy thresholding.
* Chroma metadata values must be str/int/float/bool (no None, no lists), so
  everything is sanitised through `_clean_metadata` before it is stored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import chromadb

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class VideoRecord:
    """One item to store: identity, its embedding, and metadata."""
    video_id: str
    embedding: list[float]
    document: str                      # the text that was embedded
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedVideo:
    """A single search hit returned to the RAG layer."""
    video_id: str
    document: str
    metadata: dict
    distance: float
    similarity: float                  # 1 - distance (cosine), higher = closer


class VectorStore:
    """Persistent local vector database keyed by YouTube video ID."""

    def __init__(self, persist_dir: str | None = None, collection: str | None = None) -> None:
        s = get_settings()
        self._persist_dir = persist_dir or s.chroma_persist_dir
        self._collection_name = collection or s.chroma_collection
        self._client = None
        self._col = None

    def _collection(self):
        """Lazily open the persistent client + collection (cosine space)."""
        if self._col is None:
            self._client = chromadb.PersistentClient(path=self._persist_dir)
            self._col = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},  # cosine matches MiniLM
            )
            logger.info("ChromaDB ready at '%s' (collection '%s', %d items).",
                        self._persist_dir, self._collection_name, self._col.count())
        return self._col

    # ── writes ────────────────────────────────────────────────────────────
    def add_videos(self, records: list[VideoRecord]) -> int:
        """Upsert a batch of video records. Returns how many were written.

        `upsert` (rather than `add`) makes retries idempotent: re-writing the
        same video ID overwrites rather than erroring.
        """
        if not records:
            return 0
        col = self._collection()
        col.upsert(
            ids=[r.video_id for r in records],
            embeddings=[r.embedding for r in records],
            documents=[r.document for r in records],
            metadatas=[_clean_metadata(r.metadata) for r in records],
        )
        return len(records)

    def update_transcript(self, video_id: str, transcript: str,
                          *, category: str | None = None,
                          embedding: list[float] | None = None,
                          document: str | None = None) -> None:
        """Cache a fetched transcript back onto a video (Module 5, lazy load).

        Sets `transcript_fetched=True` so it is never fetched again, and
        optionally refreshes the category and embedding now that we have the
        real content. Chroma's `update` merges the provided metadata keys.
        """
        col = self._collection()
        meta_update = {
            "transcript_fetched": True,
            "transcript_available": True,
            "transcript": transcript[:200_000],
        }
        if category:
            meta_update["category"] = category
        kwargs = {"ids": [video_id], "metadatas": [meta_update]}
        if embedding is not None:
            kwargs["embeddings"] = [embedding]
        if document is not None:
            kwargs["documents"] = [document]
        col.update(**kwargs)

    def mark_transcript_unavailable(self, video_id: str, reason: str) -> None:
        """Record that a video has no usable transcript, so we don't retry it.

        Sets `transcript_fetched=True` (the fetch attempt happened) but
        `transcript_available=False` with a reason the UI can show. On later
        queries the RAG layer sees this and falls back to the title without
        hitting YouTube again.
        """
        self._collection().update(
            ids=[video_id],
            metadatas=[{
                "transcript_fetched": True,
                "transcript_available": False,
                "transcript_note": reason,
            }],
        )

    # ── dedup (Module 4) ────────────────────────────────────────────────
    def get_existing_ids(self, ids: list[str]) -> set[str]:
        """Return which of `ids` are already stored (empty `include` = ids only)."""
        if not ids:
            return set()
        got = self._collection().get(ids=ids, include=[])
        return set(got["ids"])

    # ── reads / search ───────────────────────────────────────────────────
    def query(self, query_embedding: list[float], top_k: int = 4,
              where: dict | None = None) -> list[RetrievedVideo]:
        """Cosine similarity search — the retrieval step of the RAG loop."""
        col = self._collection()
        if col.count() == 0:
            return []
        res = col.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, col.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[RetrievedVideo] = []
        for vid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append(RetrievedVideo(
                video_id=vid, document=doc or "", metadata=meta or {},
                distance=float(dist), similarity=1.0 - float(dist),
            ))
        return hits

    def get_video(self, video_id: str) -> dict | None:
        """Fetch one video's document + metadata (used after retrieval)."""
        got = self._collection().get(ids=[video_id], include=["documents", "metadatas"])
        if not got["ids"]:
            return None
        return {"video_id": got["ids"][0],
                "document": got["documents"][0],
                "metadata": got["metadatas"][0]}

    # ── knowledge-base visualisation (Module 8) ─────────────────────────
    def count(self) -> int:
        return self._collection().count()

    def category_counts(self) -> dict[str, int]:
        """Video counts per category, for the Chart.js visualisation."""
        got = self._collection().get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in got["metadatas"]:
            cat = (meta or {}).get("category") or "Uncategorized"
            counts[cat] = counts.get(cat, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))

    def list_videos(self, category: str | None = None,
                    limit: int | None = None) -> list[dict]:
        """List stored videos (optionally by category) for the KB browser.

        The (potentially huge) cached `transcript` text is stripped out — it's
        never needed for browsing/visualisation and would bloat responses.
        """
        where = {"category": category} if category else None
        got = self._collection().get(where=where, include=["metadatas"], limit=limit)
        rows = []
        for vid, meta in zip(got["ids"], got["metadatas"]):
            m = dict(meta or {})
            m.pop("transcript", None)  # don't ship transcript bodies to the client
            rows.append({"video_id": vid, **m})
        return rows


def _clean_metadata(meta: dict) -> dict:
    """Coerce metadata to Chroma-safe scalars (str/int/float/bool, no None)."""
    clean: dict = {}
    for key, value in meta.items():
        if value is None:
            continue  # Chroma rejects None — omit the key entirely
        if isinstance(value, bool) or isinstance(value, (int, float, str)):
            clean[key] = value
        else:
            clean[key] = str(value)  # last resort, keep it a scalar
    return clean


# Process-wide singleton.
vector_store = VectorStore()
