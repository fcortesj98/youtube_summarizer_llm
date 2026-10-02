"""Fetching and formatting YouTube transcripts."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from youtube_transcript_api import NoTranscriptFound, YouTubeTranscriptApi

# Matches watch?v=, youtu.be/, /shorts/, /embed/, /live/ and /v/ URLs (plus a bare 11-char ID).
_VIDEO_ID_RE = re.compile(
    r"(?:v=|youtu\.be/|/shorts/|/embed/|/live/|/v/)([A-Za-z0-9_-]{11})|^([A-Za-z0-9_-]{11})$"
)


class TranscriptError(Exception):
    """Raised when a transcript cannot be obtained for a video."""


def extract_video_id(url: str) -> str | None:
    """Return the 11-character YouTube video ID from a URL, or None if none is found."""
    match = _VIDEO_ID_RE.search(url.strip())
    if not match:
        return None
    return match.group(1) or match.group(2)


def fetch_transcript(video_id: str, languages: Sequence[str] = ("en",)):
    """Fetch a transcript, preferring a manually created one over an auto-generated one."""
    try:
        transcripts = YouTubeTranscriptApi().list(video_id)
        try:
            transcript = transcripts.find_manually_created_transcript(languages)
        except NoTranscriptFound:
            transcript = transcripts.find_generated_transcript(languages)
        return transcript.fetch()
    except NoTranscriptFound as exc:
        raise TranscriptError(
            f"No transcript in {', '.join(languages)} is available for this video."
        ) from exc
    except Exception as exc:  # TranscriptsDisabled, VideoUnavailable, network errors...
        raise TranscriptError(f"Could not fetch the transcript: {exc.__class__.__name__}.") from exc


def format_transcript(snippets: Iterable) -> str:
    """Flatten transcript snippets into 'Text: ... Start: ...' lines for the LLM."""
    return "\n".join(f"Text: {s.text} Start: {s.start}" for s in snippets)
