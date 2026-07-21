"""Google Takeout → structured watch history (Module 2).

Turns a raw Takeout export into a clean list of unique watched videos, ready
for LLM categorisation, embedding and storage in ChromaDB (Module 3).

Written against a real export, whose structure is:

    Takeout/YouTube and YouTube Music/history/watch-history.html   ← we want this
    Takeout/YouTube and YouTube Music/history/search-history.html  ← ignored
    Takeout/YouTube and YouTube Music/playlists/*.csv              ← ignored

Design notes
------------
* Framework-free: no FastAPI imports, so this module is reusable from
  scripts/tests. The upload endpoint (Module 7) just calls `parse_takeout()`.
* Takeout localises folder names and lets users export history as HTML
  (default) or JSON, so we *search* the archive for the watch-history file
  instead of hardcoding a path, and we support both formats. Direct uploads
  of a bare `watch-history.html` / `.json` (no zip) also work.
* Real-world quirks handled (all observed in actual exports):
    - Ad impressions carry "From Google Ads" in their details → skipped.
    - "Viewed <community post>" entries link to /post/... (not videos) → skipped.
    - Entries with no link at all ("Viewed Ads On YouTube Homepage",
      "Used Shorts creation tools", removed/private videos) → skipped.
    - Timestamps contain narrow no-break spaces (\\u202f) → normalised.
    - The same video appears once per (re)watch. We merge repeats into one
      record with `watch_count` / `last_watched`, because ChromaDB keys
      records by video ID. Re-watch count is also a nice relevance signal.
* Every entry is parsed inside try/except: one malformed record can never
  abort an import — it is just counted in `ParseStats.skipped_malformed`.
"""

from __future__ import annotations

import io
import json
import logging
import re
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Iterator

from lxml import html as lxml_html

logger = logging.getLogger(__name__)


class TakeoutParseError(Exception):
    """Raised when the upload cannot be parsed at all (bad zip, no history file).

    The message is written to be user-facing: the upload endpoint returns it
    verbatim in a 422 response.
    """


# Matches every YouTube video URL shape seen in exports (regular watch pages,
# music.youtube.com watch pages, youtu.be short links, Shorts). Video IDs are
# always exactly 11 chars of [A-Za-z0-9_-].
VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?(?:[^\"'\s#]*&)?v=|youtu\.be/|youtube\.com/shorts/)"
    r"([A-Za-z0-9_-]{11})"
)

# JSON exports prefix titles with the action ("Watched Foo"); HTML anchor text
# is already clean. \xa0 = non-breaking space, seen in some exports.
_WATCHED_PREFIXES = ("Watched ", "Watched\xa0")

_AD_MARKER = "From Google Ads"

# Normalise the exotic whitespace Google puts in timestamps (\u202f narrow
# no-break space between time and AM/PM, \xa0 elsewhere).
_WS_RE = re.compile(r"[\s\u00a0\u202f]+")

# Timestamp formats after stripping the trailing timezone token.
# Observed: "Jul 17, 2026, 7:25:36 PM IST" (en-US style, 12h clock).
_TIME_FORMATS = (
    "%b %d, %Y, %I:%M:%S %p",  # Jul 17, 2026, 7:25:36 PM
    "%d %b %Y, %H:%M:%S",      # 17 Jul 2026, 19:25:36 (en-GB style exports)
)


# ──────────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class WatchedVideo:
    """One unique video from the history (re-watches merged)."""

    video_id: str
    title: str
    url: str                      # canonicalised https://www.youtube.com/watch?v=<id>
    channel: str | None = None
    channel_url: str | None = None
    product: str = "YouTube"      # "YouTube" or "YouTube Music" (header cell)
    watch_count: int = 1
    last_watched: str | None = None  # ISO-8601, local time as written in the export

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParseStats:
    """Import observability — surfaced by the upload endpoint and the UI."""

    source_file: str = ""
    source_format: str = ""       # "html" | "json"
    total_entries: int = 0
    unique_videos: int = 0
    rewatches_merged: int = 0
    skipped_ads: int = 0
    skipped_no_link: int = 0      # removed/private videos, "Viewed Ads", Shorts tools
    skipped_non_video: int = 0    # community posts, external links
    skipped_malformed: int = 0
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParseResult:
    videos: list[WatchedVideo] = field(default_factory=list)
    stats: ParseStats = field(default_factory=ParseStats)


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────

def parse_takeout(source: str | Path | bytes | BinaryIO) -> ParseResult:
    """Parse a Takeout export into unique watched videos.

    `source` may be a filesystem path, raw bytes, or a binary file object
    (e.g. FastAPI's `UploadFile.file`). Accepts a full Takeout .zip, or a
    bare watch-history .html/.json file uploaded directly.
    """
    fileobj = _as_fileobj(source)

    if zipfile.is_zipfile(fileobj):
        fileobj.seek(0)
        with zipfile.ZipFile(fileobj) as zf:
            member = _find_history_member(zf)
            data = zf.read(member)
            return _parse_history_bytes(data, source_file=member)

    # Not a zip → maybe the user uploaded the history file itself.
    fileobj.seek(0)
    data = fileobj.read()
    sniff = data.lstrip()[:1]
    if sniff in (b"[", b"<"):
        return _parse_history_bytes(data, source_file="(uploaded directly)")

    raise TakeoutParseError(
        "The uploaded file is neither a Takeout .zip nor a watch-history "
        ".html/.json file. Export it from https://takeout.google.com with "
        "'YouTube and YouTube Music → history' selected."
    )


# ──────────────────────────────────────────────────────────────────────────
# Locating the history file inside the archive
# ──────────────────────────────────────────────────────────────────────────

def _as_fileobj(source: str | Path | bytes | BinaryIO) -> BinaryIO:
    if isinstance(source, (str, Path)):
        return open(source, "rb")
    if isinstance(source, (bytes, bytearray)):
        return io.BytesIO(source)
    return source  # already a binary file object


def _find_history_member(zf: zipfile.ZipFile) -> str:
    """Find the watch-history file without assuming a fixed path.

    Takeout localises the folder names ("YouTube and YouTube Music" is
    language-dependent), so we rank candidates instead:

      priority 0 — basename contains "watch-history" / "watch_history"
      priority 1 — any .json/.html whose first bytes *look like* watch
                   history (content sniff), excluding search-history

    JSON is preferred over HTML when both exist (richer, cheaper to parse),
    and larger files win ties. If nothing matches we raise a user-facing
    error rather than silently parsing the wrong file (e.g. search history).
    """
    candidates: list[tuple[int, int, int, str]] = []

    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename
        base = Path(name).name.lower()
        if name.startswith("__MACOSX") or base.startswith("."):
            continue  # macOS zip artefacts
        if not base.endswith((".json", ".html")):
            continue
        if "search-history" in base or "search_history" in base:
            continue  # explicitly never fall back to search history

        if "watch-history" in base or "watch_history" in base:
            priority = 0
        else:
            # Content sniff: read the first 4 KB and look for markers that
            # only appear in activity-history files.
            head = zf.open(info).read(4096)
            if b'"titleUrl"' in head or b"outer-cell" in head or b"header-cell" in head:
                priority = 1
            else:
                continue

        fmt_pref = 0 if base.endswith(".json") else 1  # prefer JSON
        candidates.append((priority, fmt_pref, -info.file_size, name))

    if not candidates:
        raise TakeoutParseError(
            "No watch-history file found in this archive. When creating the "
            "export on takeout.google.com, make sure 'YouTube and YouTube "
            "Music' → 'history' is included."
        )

    candidates.sort()
    chosen = candidates[0][3]
    logger.info("Using history file from archive: %s", chosen)
    return chosen


def _parse_history_bytes(data: bytes, source_file: str) -> ParseResult:
    """Dispatch on content: JSON array vs Takeout HTML page."""
    stripped = data.lstrip()
    if stripped.startswith(b"["):
        entries = _iter_json_entries(data)
        fmt = "json"
    elif stripped.startswith(b"<"):
        entries = _iter_html_entries(data)
        fmt = "html"
    else:
        raise TakeoutParseError("Unrecognised watch-history format.")
    return _merge_entries(entries, source_file=source_file, source_format=fmt)


# ──────────────────────────────────────────────────────────────────────────
# Format-specific extractors
# Each yields (skip_reason | None, WatchedVideo | None) per raw entry, so the
# shared merge loop below owns all counting/dedup logic exactly once.
# ──────────────────────────────────────────────────────────────────────────

def _iter_html_entries(data: bytes) -> Iterator[tuple[str | None, WatchedVideo | None]]:
    """Extract entries from Takeout's HTML export (the default format).

    lxml (C-based) parses the whole 10–100 MB document in a couple of
    seconds; each history record is one `div.outer-cell`:

        <div class="outer-cell ...">
          <div class="header-cell ..."><p ...>YouTube</p></div>       product
          <div class="content-cell ... body-1">
              Watched <a href="...watch?v=ID">Title</a><br>
              <a href="...channel/...">Channel</a><br>
              Jul 17, 2026, 7:25:36 PM IST<br>
          </div>
          <div class="content-cell ... caption">Products: ... </div>   ad marker
        </div>
    """
    # Takeout's HTML omits a charset declaration, so lxml falls back to
    # Latin-1 and non-ASCII titles become mojibake ("Hereâs", broken Tamil
    # script, …). Exports are always UTF-8 — decode explicitly.
    tree = lxml_html.fromstring(data.decode("utf-8", errors="replace"))

    for outer in tree.xpath('//div[contains(@class, "outer-cell")]'):
        try:
            # Ads first: ad impressions can still contain a watch?v= link,
            # so this check must happen before URL extraction.
            caption = " ".join(
                outer.xpath('.//div[contains(@class, "mdl-typography--caption")]//text()')
            )
            if _AD_MARKER in caption:
                yield "ad", None
                continue

            body_cells = outer.xpath('.//div[contains(@class, "mdl-typography--body-1")]')
            if not body_cells:
                yield "malformed", None
                continue
            body = body_cells[0]  # second body-1 cell is an empty right-spacer

            anchors = body.xpath("./a")
            if not anchors:
                # "Viewed Ads On YouTube Homepage", "Used Shorts creation
                # tools", removed/private videos — nothing watchable here.
                yield "no_link", None
                continue

            href = anchors[0].get("href", "")
            m = VIDEO_ID_RE.search(href)
            if not m:
                # Community posts (/post/...), external links, channel links.
                yield "non_video", None
                continue

            video_id = m.group(1)
            title = _clean_title(anchors[0].text_content(), video_id)

            channel = channel_url = None
            if len(anchors) > 1:
                channel = anchors[1].text_content().strip() or None
                channel_url = anchors[1].get("href")

            product = "YouTube"
            header = outer.xpath('.//div[contains(@class, "header-cell")]//p/text()')
            if header:
                product = header[0].strip() or product

            # The timestamp is the last bare text node in the body cell.
            texts = [t.strip() for t in body.itertext() if t.strip()]
            watched_at = _parse_html_time(texts[-1]) if texts else None

            yield None, WatchedVideo(
                video_id=video_id,
                title=title,
                url=f"https://www.youtube.com/watch?v={video_id}",
                channel=channel,
                channel_url=channel_url,
                product=product,
                last_watched=watched_at,
            )
        except Exception:  # noqa: BLE001 — one bad record must never kill an import
            logger.debug("Malformed HTML history entry", exc_info=True)
            yield "malformed", None


def _iter_json_entries(data: bytes) -> Iterator[tuple[str | None, WatchedVideo | None]]:
    """Extract entries from Takeout's JSON export (if the user selected it).

    Shape per entry:
        {"header": "YouTube", "title": "Watched Foo",
         "titleUrl": "https://www.youtube.com/watch?v=...",
         "subtitles": [{"name": "Channel", "url": "..."}],
         "time": "2026-07-17T13:55:36.123Z",
         "details": [{"name": "From Google Ads"}]}   ← only on ads
    """
    try:
        entries = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TakeoutParseError(f"watch-history.json could not be parsed: {exc}") from exc
    if not isinstance(entries, list):
        raise TakeoutParseError("watch-history.json is not a list of activity entries.")

    for entry in entries:
        try:
            details = entry.get("details") or []
            if any(d.get("name") == _AD_MARKER for d in details):
                yield "ad", None
                continue

            url = entry.get("titleUrl")
            if not url:
                yield "no_link", None  # removed/private videos have no URL
                continue

            m = VIDEO_ID_RE.search(url)
            if not m:
                yield "non_video", None
                continue

            video_id = m.group(1)
            title = entry.get("title") or ""
            for prefix in _WATCHED_PREFIXES:
                if title.startswith(prefix):
                    title = title[len(prefix):]
                    break
            title = _clean_title(title, video_id)

            subtitles = entry.get("subtitles") or []
            channel = subtitles[0].get("name") if subtitles else None
            channel_url = subtitles[0].get("url") if subtitles else None

            yield None, WatchedVideo(
                video_id=video_id,
                title=title,
                url=f"https://www.youtube.com/watch?v={video_id}",
                channel=channel,
                channel_url=channel_url,
                product=entry.get("header") or "YouTube",
                last_watched=(entry.get("time") or None),
            )
        except Exception:  # noqa: BLE001
            logger.debug("Malformed JSON history entry", exc_info=True)
            yield "malformed", None


# ──────────────────────────────────────────────────────────────────────────
# Shared merge / dedup loop
# ──────────────────────────────────────────────────────────────────────────

_SKIP_FIELD = {
    "ad": "skipped_ads",
    "no_link": "skipped_no_link",
    "non_video": "skipped_non_video",
    "malformed": "skipped_malformed",
}


def _merge_entries(
    entries: Iterator[tuple[str | None, WatchedVideo | None]],
    source_file: str,
    source_format: str,
) -> ParseResult:
    """Count skips and merge re-watches into unique videos.

    Exports are ordered newest-first, so the first occurrence of a video ID
    carries its most recent title/timestamp — we keep that as canonical and
    only bump `watch_count` for older repeats (with a timestamp comparison
    as a safety net in case ordering ever changes).
    """
    stats = ParseStats(source_file=source_file, source_format=source_format)
    merged: dict[str, WatchedVideo] = {}

    for skip_reason, video in entries:
        stats.total_entries += 1

        if skip_reason is not None:
            setattr(stats, _SKIP_FIELD[skip_reason],
                    getattr(stats, _SKIP_FIELD[skip_reason]) + 1)
            continue

        assert video is not None  # by contract of the extractors
        existing = merged.get(video.video_id)
        if existing is None:
            merged[video.video_id] = video
            continue

        existing.watch_count += 1
        stats.rewatches_merged += 1
        # Safety net: keep the newest timestamp/title regardless of order.
        # ISO-8601 strings compare correctly as plain strings.
        if video.last_watched and (
            existing.last_watched is None or video.last_watched > existing.last_watched
        ):
            existing.last_watched = video.last_watched
            existing.title = video.title

    stats.unique_videos = len(merged)
    logger.info(
        "Parsed %s: %d entries → %d unique videos (%d re-watches merged; "
        "skipped %d ads, %d no-link, %d non-video, %d malformed)",
        source_file, stats.total_entries, stats.unique_videos,
        stats.rewatches_merged, stats.skipped_ads, stats.skipped_no_link,
        stats.skipped_non_video, stats.skipped_malformed,
    )
    return ParseResult(videos=list(merged.values()), stats=stats)


# ──────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────

def _clean_title(raw: str, video_id: str) -> str:
    """Normalise whitespace; never return an empty title (fallback to the ID)."""
    title = _WS_RE.sub(" ", raw or "").strip()
    return title or f"(untitled video {video_id})"


def _parse_html_time(raw: str) -> str | None:
    """'Jul 17, 2026, 7:25:36 PM IST' → '2026-07-17T19:25:36' (best effort).

    The trailing timezone abbreviation is dropped: abbreviations like 'IST'
    are ambiguous (India/Israel/Ireland), and for our metadata purposes the
    user's local wall-clock time is exactly what we want to display. Returns
    None rather than guessing when the format is unfamiliar.
    """
    cleaned = _WS_RE.sub(" ", raw).strip()
    parts = cleaned.rsplit(" ", 1)
    if len(parts) == 2 and parts[1].isalpha() and len(parts[1]) <= 5:
        cleaned = parts[0]  # drop 'IST' / 'PM'-less tz token
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).isoformat()
        except ValueError:
            continue
    logger.debug("Unparsed timestamp: %r", raw)
    return None
