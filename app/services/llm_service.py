"""LLM service — the ONLY place the app talks to a language model (Module 3).

Every LLM interaction in the project goes through this module: video
categorisation (Module 3), category refinement (Module 5), and grounded
question answering (Module 5). Isolating it here means:

  * the provider is swappable in one place (DeepSeek today; OpenAI or an
    Anthropic-compatible gateway tomorrow) — the rest of the app is agnostic;
  * the API key can be injected at RUNTIME from the Settings page, not baked
    into code or even required at startup;
  * retries / rate-limit handling live in one spot.

DeepSeek exposes an OpenAI-compatible REST API, so we reuse the official
`openai` client and simply point `base_url` at the provider. Swapping to
OpenAI proper is just a different base_url + model + key.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from app.config import get_settings
from app.utils.retry import with_retries

logger = logging.getLogger(__name__)

# Transient failures worth retrying with backoff. 4xx auth/validation errors
# are NOT here — retrying a bad API key just wastes time.
_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


# A bounded, curated taxonomy. We force the model to choose from this fixed
# list (rather than inventing labels) so the knowledge-base visualisation
# stays clean — free-form categories would explode into hundreds of near
# -duplicates ("AI", "A.I.", "Artificial Intelligence", ...).
DEFAULT_CATEGORIES: list[str] = [
    "Programming", "AI & Machine Learning", "Data & Databases",
    "Cloud & DevOps", "Web Development", "Cybersecurity", "Technology",
    "Design (UI/UX)", "Business & Entrepreneurship", "Finance & Investing",
    "Cryptocurrency", "Productivity", "Self-Improvement", "Psychology",
    "Health & Fitness", "Meditation & Mindfulness", "Science", "Education",
    "Music", "Movies & TV", "Gaming", "Entertainment", "News & Politics",
    "Cooking", "Travel", "Sports", "Vlogs & Lifestyle", "Other",
]
_UNCATEGORISED = "Uncategorized"  # used when the LLM is disabled or a batch fails
# Lookup for snapping the model's answer back onto a canonical label.
_CANON = {c.lower(): c for c in DEFAULT_CATEGORIES}


@dataclass
class LLMConfig:
    """Mutable runtime configuration for the active provider."""
    provider: str
    base_url: str
    model: str
    api_key: str


class LLMNotConfiguredError(RuntimeError):
    """Raised when an LLM call is attempted before an API key is set."""


class LLMService:
    """Thread-safe, runtime-reconfigurable wrapper around an OpenAI-compatible API."""

    def __init__(self) -> None:
        s = get_settings()
        # Seed from environment; the UI (Settings page) can override any of
        # these later via `configure()`. An empty key is allowed at startup.
        self._config = LLMConfig(
            provider=s.llm_provider,
            base_url=s.llm_base_url,
            model=s.llm_model,
            api_key=s.llm_api_key,
        )
        self._client: OpenAI | None = None
        self._lock = threading.Lock()  # guards config + cached client

    # ── configuration ────────────────────────────────────────────────────
    def configure(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        """Update provider settings at runtime (called by the Settings endpoint)."""
        with self._lock:
            if api_key is not None:
                self._config.api_key = api_key.strip()
            if base_url is not None:
                self._config.base_url = base_url.strip()
            if model is not None:
                self._config.model = model.strip()
            if provider is not None:
                self._config.provider = provider.strip()
            self._client = None  # force rebuild on next use
        logger.info("LLM reconfigured: provider=%s model=%s configured=%s",
                    self._config.provider, self._config.model, self.is_configured())

    def is_configured(self) -> bool:
        return bool(self._config.api_key)

    def status(self) -> dict:
        """Safe status for the UI — never leaks the key itself."""
        return {
            "provider": self._config.provider,
            "model": self._config.model,
            "base_url": self._config.base_url,
            "configured": self.is_configured(),
        }

    def _get_client(self) -> OpenAI:
        with self._lock:
            if not self._config.api_key:
                raise LLMNotConfiguredError(
                    "No LLM API key set. Add one on the Settings page "
                    "(or in your .env file) before asking questions."
                )
            if self._client is None:
                self._client = OpenAI(
                    api_key=self._config.api_key,
                    base_url=self._config.base_url,
                    timeout=60.0,
                )
            return self._client, self._config.model

    # ── low-level chat primitive (used by RAG in Module 5) ───────────────
    def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        """Single-turn completion. Retries transient errors with backoff."""
        client, model = self._get_client()

        @with_retries(retry_on=_RETRYABLE, label="LLM chat")
        def _call() -> str:
            kwargs = {}
            if json_mode:
                # OpenAI-compatible JSON mode; the prompt must mention "JSON".
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            choice = resp.choices[0]
            content = (choice.message.content or "").strip()
            if not content:
                # Diagnose the blank-answer case. A common cause is the output
                # token budget being consumed before any answer text is emitted
                # (e.g. a reasoning-style model spending it on hidden reasoning),
                # which surfaces as finish_reason="length". Some providers also
                # place text in a non-standard `reasoning_content` field.
                finish = getattr(choice, "finish_reason", "?")
                reasoning = getattr(choice.message, "reasoning_content", None)
                logger.warning("LLM returned empty content (finish_reason=%s, "
                               "model=%s, max_tokens=%d)", finish, model, max_tokens)
                if reasoning:  # fall back to reasoning text if that's all we got
                    return reasoning.strip()
            return content

        return _call()

    def chat_stream(self, *, system: str, user: str, temperature: float = 0.35,
                    max_tokens: int = 3072, on_delta) -> str:
        """Streaming completion for token-by-token answers.

        Calls `on_delta(text_piece)` for each content chunk as it arrives from
        the provider, and returns the full accumulated text at the end.

        Retry policy: we retry transient failures ONLY if they happen before
        any text has been emitted, so a mid-stream reconnect can never
        duplicate tokens the user has already seen.
        """
        client, model = self._get_client()
        attempt = 0
        while True:
            attempt += 1
            emitted = False
            parts: list[str] = []
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    piece = getattr(chunk.choices[0].delta, "content", None)
                    if piece:
                        emitted = True
                        parts.append(piece)
                        on_delta(piece)
                text = "".join(parts).strip()
                if not text:
                    logger.warning("LLM stream produced no content (model=%s, "
                                   "max_tokens=%d)", model, max_tokens)
                return text
            except _RETRYABLE as exc:
                if emitted or attempt >= 4:
                    raise
                delay = min(30.0, 2.0 ** (attempt - 1))
                logger.info("LLM stream attempt %d failed before output (%s); "
                            "retrying in %.1fs", attempt, type(exc).__name__, delay)
                time.sleep(delay)

    # ── categorisation (Module 3 ingestion) ──────────────────────────────
    def categorize_batch(self, items: list[dict]) -> list[str]:
        """Assign one category to each video from title + channel.

        `items` is a list of {"title": str, "channel": str|None}. Returns a
        list of canonical category strings, one per item, in the same order.

        Robustness: the whole batch is wrapped so that ANY failure (bad key
        surfaced upstream, malformed JSON, wrong length) degrades to
        `_UNCATEGORISED` for that batch instead of crashing the ingest. The
        video is still embedded and stored — it just starts uncategorised and
        can be refined later (Module 5) once its transcript is fetched.
        """
        if not items:
            return []

        numbered = "\n".join(
            f'{i}. title: "{it.get("title", "")}"'
            + (f' | channel: "{it["channel"]}"' if it.get("channel") else "")
            for i, it in enumerate(items)
        )
        allowed = ", ".join(DEFAULT_CATEGORIES)
        system = (
            "You are a precise content classifier for a user's YouTube watch "
            "history. Assign each video to exactly ONE category from the "
            "allowed list. Respond ONLY with a JSON object mapping each "
            'item number (as a string) to its category, e.g. {"0": "Music", '
            '"1": "Programming"}. Use only categories from the allowed list; '
            'if nothing fits, use "Other".'
        )
        user = (
            f"Allowed categories: {allowed}\n\n"
            f"Videos:\n{numbered}\n\n"
            "Return the JSON object now."
        )

        try:
            raw = self.chat(system=system, user=user, temperature=0.0,
                            max_tokens=2048, json_mode=True)
            mapping = self._parse_json_object(raw)
            return [self._canonical(mapping.get(str(i))) for i in range(len(items))]
        except Exception:  # noqa: BLE001 — never let categorisation kill an ingest
            logger.warning("Categorisation batch failed; marking %d videos "
                           "uncategorised", len(items), exc_info=True)
            return [_UNCATEGORISED] * len(items)

    def categorize_one(self, *, title: str, channel: str | None,
                       transcript_excerpt: str | None = None) -> str:
        """Refine a single video's category, optionally using its transcript.

        Called in Module 5 after a transcript is lazily fetched, to sharpen
        the initial title-only guess. Falls back to the uncategorised label
        on any error rather than raising.
        """
        allowed = ", ".join(DEFAULT_CATEGORIES)
        system = (
            "Classify this single YouTube video into exactly ONE category "
            "from the allowed list. Respond ONLY as JSON: "
            '{"category": "<one allowed category>"}.'
        )
        excerpt = f'\nTranscript excerpt: "{transcript_excerpt[:1500]}"' if transcript_excerpt else ""
        user = (f"Allowed categories: {allowed}\n\n"
                f'Title: "{title}"' + (f'\nChannel: "{channel}"' if channel else "")
                + excerpt + "\n\nReturn the JSON now.")
        try:
            raw = self.chat(system=system, user=user, temperature=0.0,
                            max_tokens=64, json_mode=True)
            return self._canonical(self._parse_json_object(raw).get("category"))
        except Exception:  # noqa: BLE001
            logger.debug("Single-video categorisation failed", exc_info=True)
            return _UNCATEGORISED

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _parse_json_object(raw: str) -> dict:
        """Parse a JSON object from the model, tolerating ```json fences."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            # drop an optional leading "json" language tag
            if "\n" in text:
                first, rest = text.split("\n", 1)
                if first.strip().lower() in ("json", ""):
                    text = rest
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        return json.loads(text)

    @staticmethod
    def _canonical(value: str | None) -> str:
        """Snap a model-returned label onto the canonical taxonomy."""
        if not value:
            return _UNCATEGORISED
        return _CANON.get(value.strip().lower(), "Other")


# Process-wide singleton. (For multi-worker deployments you'd move runtime
# config to a shared store; a single uvicorn worker is assumed here.)
llm_service = LLMService()
