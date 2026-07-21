#!/usr/bin/env python3
"""Manual test for Module 3: parse a Takeout export and ingest it into ChromaDB.

Examples
--------
Cheap smoke test — no API key, no tokens spent (everything stored as
"Uncategorized"), just proves parsing + embedding + ChromaDB work:

    python scripts/ingest.py path/to/takeout.zip --no-llm --limit 50

Real categorisation with DeepSeek (needs LLM_API_KEY in .env or the Settings
page), on a small slice first so you don't spend much:

    python scripts/ingest.py path/to/takeout.zip --limit 100

Re-run the SAME command to see deduplication in action (already_present > 0,
new_videos == 0).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.ingestion import ingest_videos  # noqa: E402
from app.services.llm_service import LLMNotConfiguredError  # noqa: E402
from app.services.takeout_parser import TakeoutParseError, parse_takeout  # noqa: E402
from app.services.vector_store import vector_store  # noqa: E402


def _progress(done: int, total: int, phase: str) -> None:
    if phase == "done":
        print(f"\r  ...{total}/{total} done            ")
    else:
        pct = (done / total * 100) if total else 100
        print(f"\r  ...{done}/{total} ({pct:.0f}%)", end="", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("takeout", help="Path to Takeout .zip (or watch-history.html/.json)")
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip LLM categorisation (no key/tokens needed)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Ingest only the first N videos (keep test costs low)")
    ap.add_argument("--batch-size", type=int, default=20)
    args = ap.parse_args()

    print("Parsing Takeout...")
    try:
        result = parse_takeout(args.takeout)
    except TakeoutParseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    videos = result.videos
    if args.limit:
        videos = videos[: args.limit]
    print(f"  {len(videos)} videos to ingest "
          f"(of {result.stats.unique_videos} unique parsed).")

    print(f"Ingesting {'without' if args.no_llm else 'with'} LLM categorisation...")
    t0 = time.perf_counter()
    try:
        stats = ingest_videos(
            videos, use_llm=not args.no_llm,
            batch_size=args.batch_size, progress_cb=_progress,
        )
    except LLMNotConfiguredError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print("Hint: re-run with --no-llm to test the pipeline without a key.",
              file=sys.stderr)
        return 2
    elapsed = time.perf_counter() - t0

    s = stats.to_dict()
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  input={s['total_input']}  already_present={s['already_present']}  "
          f"new={s['new_videos']}  added={s['added']}")
    print(f"  categorised={s['categorised']}  uncategorised={s['uncategorised']}  "
          f"batches={s['batches']}")

    print(f"\nChromaDB now holds {vector_store.count()} videos.")
    counts = vector_store.category_counts()
    print("Top categories:")
    for cat, n in list(counts.items())[:12]:
        print(f"  {n:>5}  {cat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
