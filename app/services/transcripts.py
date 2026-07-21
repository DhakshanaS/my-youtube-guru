"""Transcript fetching (Module 5, lazy loading).

Fetches a video's transcript on demand — only for videos that actually match a
question — using `youtube-transcript-api`. Fetched transcripts are cached back
into ChromaDB by the RAG layer so a video is never fetched twice.

Two things this module gets right:

1. Permanent vs transient failures.
   Many videos simply have no transcript (captions disabled, live/removed,
   age-restricted). Retrying those is pointless, so they are caught and
   reported as `available=False` with a human-readable reason — the RAG layer
   then falls back to the title/channel for that video.
   Transient failures (YouTube rate-limiting the IP, a flaky request) ARE
   worth retrying, so those go through the shared exponential-backoff helper
   (this is the Module 6 rate-limit strategy applied to transcripts).

2. API version.
   youtube-transcript-api 1.x is instance-based: `YouTubeTranscriptApi().fetch(id)`
   returns a `FetchedTranscript` whose `.to_raw_data()` yields snippet dicts.
   (The old 0.x `YouTubeTranscriptApi.get_transcript` staticmethod is gone.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from youtube_transcript_api import (
    AgeRestricted,
    InvalidVideoId,
    NoTranscriptFound,
    RequestBlocked,       # base class of IpBlocked → covers both
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
    YouTubeRequestFailed,
    YouTubeTranscriptApi,
)

from app.utils.retry import with_retries

logger = logging.getLogger(__name__)

# Transient → retry with backoff. YouTube throttles by IP under load, and
# requests can fail intermittently; both usually succeed on a later attempt.
_TRANSIENT = (RequestBlocked, YouTubeRequestFailed, ConnectionError, TimeoutError)

# Permanent → do NOT retry; report a clear reason and move on. Mapping the
# exception type to a short message keeps the UI/notes informative.
_PERMANENT: dict[type[Exception], str] = {
    TranscriptsDisabled: "captions are disabled for this video",
    NoTranscriptFound: "no transcript available in the requested language",
    VideoUnavailable: "the video is unavailable (removed or private)",
    VideoUnplayable: "the video is not playable",
    AgeRestricted: "the video is age-restricted",
    InvalidVideoId: "invalid video id",
}

# Preferred caption languages, in order. English variants first, then a few
# common ones; youtube-transcript-api falls back within this list.
_DEFAULT_LANGUAGES = ("en", "en-US", "en-GB", "hi", "ta")


@dataclass
class TranscriptResult:
    """Outcome of a transcript fetch (success or a graceful failure)."""
    video_id: str
    available: bool
    text: str | None = None
    language: str | None = None
    reason: str | None = None       # why it's unavailable (when available=False)


class TranscriptService:
    """Wrapper around youtube-transcript-api with retry + graceful degradation."""

    def __init__(self, languages: tuple[str, ...] = _DEFAULT_LANGUAGES,
                 max_attempts: int = 4) -> None:
        self._languages = languages
        self._api = YouTubeTranscriptApi()
        # Bind the backoff policy once; reused for every fetch. Longer base
        # delay than the LLM because YouTube IP-throttling needs more cool-off.
        self._fetch_with_retry = with_retries(
            retry_on=_TRANSIENT, max_attempts=max_attempts,
            base_delay=2.0, max_delay=30.0, label="transcript fetch",
        )(self._raw_fetch)

    def _raw_fetch(self, video_id: str):
        """One raw fetch attempt (wrapped by the retry decorator above)."""
        return self._api.fetch(video_id, languages=list(self._languages))

    def fetch(self, video_id: str) -> TranscriptResult:
        """Fetch and flatten a transcript, or return a graceful 'unavailable'.

        Never raises for the normal 'this video has no transcript' cases —
        those are expected and returned as `available=False`.
        """
        try:
            fetched = self._fetch_with_retry(video_id)
        except tuple(_PERMANENT) as exc:
            reason = _PERMANENT[type(exc)]
            logger.info("No transcript for %s: %s", video_id, reason)
            return TranscriptResult(video_id, available=False, reason=reason)
        except _TRANSIENT as exc:
            # Exhausted retries on a transient error — degrade gracefully so
            # one blocked video can't sink the whole answer.
            logger.warning("Transcript for %s failed after retries: %s", video_id, exc)
            return TranscriptResult(
                video_id, available=False,
                reason="temporarily unavailable (rate-limited or network error)",
            )
        except Exception as exc:  # noqa: BLE001 — unknown lib error, stay resilient
            logger.warning("Unexpected transcript error for %s: %s", video_id, exc)
            return TranscriptResult(video_id, available=False,
                                    reason="could not be retrieved")

        # Flatten snippet dicts ([{text,start,duration}, ...]) into one string.
        snippets = fetched.to_raw_data()
        text = " ".join(s["text"].strip() for s in snippets if s.get("text"))
        text = " ".join(text.split())  # collapse whitespace/newlines
        language = getattr(fetched, "language_code", None) or getattr(fetched, "language", None)
        if not text:
            return TranscriptResult(video_id, available=False,
                                    reason="transcript was empty")
        logger.info("Fetched transcript for %s (%d chars, lang=%s)",
                    video_id, len(text), language)
        return TranscriptResult(video_id, available=True, text=text, language=language)


# Process-wide singleton.
transcript_service = TranscriptService()
