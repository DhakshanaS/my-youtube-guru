"""Ingestion pipeline (Module 3) — parsed videos → categorised, embedded, stored.

Orchestrates the other services into one resumable, observable operation:

    parsed videos
        → drop ones already in ChromaDB   (dedup, Module 4)
        → for each batch:
              categorise via the LLM       (title + channel → one category)
              embed via sentence-transformers
              upsert into ChromaDB          (transcript_fetched = False)
        → report progress after every batch

Why dedup happens BEFORE categorisation: LLM calls cost money and time, so on
a re-upload we must never re-categorise the thousands of videos we already
have — we only spend tokens on genuinely new ones.

The embedder and LLM are injectable parameters (defaulting to the module
singletons) so this orchestration can be unit-tested with deterministic fakes
and no network / no API key.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Callable, Protocol

from app.services.embeddings import build_embedding_text, embedding_model
from app.services.llm_service import LLMNotConfiguredError, llm_service
from app.services.takeout_parser import WatchedVideo
from app.services.vector_store import VideoRecord, vector_store

logger = logging.getLogger(__name__)

# progress_cb(done, total, phase) — lets the API expose a polling endpoint.
ProgressCallback = Callable[[int, int, str], None]


# Structural typing: anything with these methods can stand in for the real
# services in tests (a fake embedder / fake categoriser).
class _Embedder(Protocol):
    def embed_texts(self, texts: list[str], batch_size: int = ...) -> list[list[float]]: ...


class _Categoriser(Protocol):
    def is_configured(self) -> bool: ...
    def categorize_batch(self, items: list[dict]) -> list[str]: ...


@dataclass
class IngestStats:
    """Everything the UI needs to report what an ingest did."""
    total_input: int = 0
    already_present: int = 0      # skipped by dedup
    new_videos: int = 0
    added: int = 0                # successfully written to ChromaDB
    categorised: int = 0          # got a real (non-uncategorised) category
    uncategorised: int = 0
    batches: int = 0
    used_llm: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def ingest_videos(
    videos: list[WatchedVideo],
    *,
    use_llm: bool = True,
    batch_size: int = 20,
    skip_existing: bool = True,
    progress_cb: ProgressCallback | None = None,
    store=vector_store,
    embedder: _Embedder = embedding_model,
    llm: _Categoriser = llm_service,
) -> IngestStats:
    """Categorise, embed and store parsed videos. See module docstring."""
    stats = IngestStats(total_input=len(videos), used_llm=use_llm)

    # ── 1. Dedup against what's already stored (Module 4) ────────────────
    if skip_existing and videos:
        existing = store.get_existing_ids([v.video_id for v in videos])
        stats.already_present = len(existing)
        videos = [v for v in videos if v.video_id not in existing]
    stats.new_videos = len(videos)

    if not videos:
        logger.info("Ingest: nothing new (%d already present).", stats.already_present)
        if progress_cb:
            progress_cb(0, 0, "done")
        return stats

    # Fail fast with a friendly message if categorisation is requested but no
    # key is set — better than silently storing everything as uncategorised.
    if use_llm and not llm.is_configured():
        raise LLMNotConfiguredError(
            "Categorisation needs an LLM API key. Set one on the Settings "
            "page, or ingest with categorisation disabled."
        )

    total = len(videos)
    done = 0

    # ── 2. Process in batches: categorise → embed → store ────────────────
    for start in range(0, total, batch_size):
        batch = videos[start : start + batch_size]
        stats.batches += 1

        # (a) categories
        if use_llm:
            categories = llm.categorize_batch(
                [{"title": v.title, "channel": v.channel} for v in batch]
            )
        else:
            categories = ["Uncategorized"] * len(batch)

        # (b) embeddings (one batched encode call per batch)
        texts = [build_embedding_text(v.title, v.channel) for v in batch]
        vectors = embedder.embed_texts(texts)

        # (c) assemble records + store
        records: list[VideoRecord] = []
        for video, category, vector, text in zip(batch, categories, vectors, texts):
            if category and category != "Uncategorized":
                stats.categorised += 1
            else:
                stats.uncategorised += 1
            records.append(VideoRecord(
                video_id=video.video_id,
                embedding=vector,
                document=text,
                metadata={
                    "title": video.title,
                    "url": video.url,
                    "video_id": video.video_id,
                    "channel": video.channel or "",
                    "product": video.product,
                    "category": category,
                    "watch_count": video.watch_count,
                    "last_watched": video.last_watched or "",
                    # Lazy-loading flag: transcripts are fetched only at query
                    # time for videos that actually match a question (Module 5).
                    "transcript_fetched": False,
                },
            ))

        stats.added += store.add_videos(records)
        done += len(batch)
        if progress_cb:
            progress_cb(done, total, "ingesting")
        logger.info("Ingest progress: %d/%d videos", done, total)

    if progress_cb:
        progress_cb(total, total, "done")
    logger.info("Ingest complete: %s", stats.to_dict())
    return stats
