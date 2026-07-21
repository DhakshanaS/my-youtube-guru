"""Exponential-backoff retry helper (shared utility).

Introduced in Module 3 because the batched LLM categorisation calls must
survive transient rate-limit / network errors without aborting a multi-minute
ingest. Module 6 reuses this same primitive for transcript fetching and tunes
the delays.

Deliberately generic: the caller passes which exception types are retryable
(so this module never imports `openai` or `youtube_transcript_api`), keeping it
dependency-free and unit-testable.
"""

from __future__ import annotations

import logging
import random
import time
from functools import wraps
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def with_retries(
    *,
    retry_on: tuple[type[Exception], ...],
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.3,
    label: str = "operation",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: retry a call on the given exceptions with exponential backoff.

    Delay before attempt *n* (1-indexed) is::

        min(max_delay, base_delay * 2**(n-1))  ± up to `jitter` fraction

    Jitter spreads out retries so parallel workers don't all wake at once
    (the "thundering herd" problem). After `max_attempts` the last exception
    is re-raised so the caller can decide how to degrade.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            attempt = 0
            while True:
                attempt += 1
                try:
                    return func(*args, **kwargs)
                except retry_on as exc:  # type: ignore[misc]
                    if attempt >= max_attempts:
                        logger.warning(
                            "%s failed after %d attempts: %s", label, attempt, exc
                        )
                        raise
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    delay += delay * random.uniform(-jitter, jitter)
                    delay = max(0.0, delay)
                    logger.info(
                        "%s attempt %d/%d hit %s; retrying in %.1fs",
                        label, attempt, max_attempts, type(exc).__name__, delay,
                    )
                    time.sleep(delay)

        return wrapper

    return decorator
