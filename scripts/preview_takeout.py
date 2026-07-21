#!/usr/bin/env python3
"""Manual test for Module 2: parse a Takeout export and print what was found.

Usage:
    python scripts/preview_takeout.py path/to/takeout-XXXX.zip [--limit 8]

This exists so the parser can be verified against your real export before
the upload API (Module 7) is built on top of it.
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.takeout_parser import TakeoutParseError, parse_takeout  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("takeout", help="Path to the Takeout .zip (or a bare watch-history.html/.json)")
    ap.add_argument("--limit", type=int, default=8, help="How many sample videos to print")
    args = ap.parse_args()

    t0 = time.perf_counter()
    try:
        result = parse_takeout(args.takeout)
    except TakeoutParseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - t0

    s = result.stats
    print(f"Source file    : {s.source_file}  (format: {s.source_format})")
    print(f"Parsed in      : {elapsed:.2f}s")
    print(f"Total entries  : {s.total_entries:,}")
    print(f"Unique videos  : {s.unique_videos:,}  (re-watches merged: {s.rewatches_merged:,})")
    print(f"Skipped        : {s.skipped_ads} ads · {s.skipped_no_link} no-link · "
          f"{s.skipped_non_video} non-video · {s.skipped_malformed} malformed")

    products = Counter(v.product for v in result.videos)
    print("By product     : " + ", ".join(f"{k}: {n:,}" for k, n in products.most_common()))

    most_watched = sorted(result.videos, key=lambda v: v.watch_count, reverse=True)[:3]
    print("\nMost re-watched:")
    for v in most_watched:
        print(f"  {v.watch_count:>3}x  [{v.video_id}] {v.title[:70]}")

    print(f"\nFirst {args.limit} videos (newest first):")
    for v in result.videos[: args.limit]:
        ch = f" — {v.channel}" if v.channel else ""
        seen = f" (last {v.last_watched[:10]})" if v.last_watched else ""
        print(f"  [{v.video_id}] {v.title[:60]}{ch}{seen}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
