"""Offline tests: no network, no watsonx credentials required."""

from types import SimpleNamespace

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListLLM

from ytbot import TranscriptError, VideoAssistant, extract_video_id
from ytbot.transcript import format_transcript

VID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VID}",
        f"https://youtube.com/watch?v={VID}&t=42s",
        f"https://m.youtube.com/watch?feature=share&v={VID}",
        f"https://youtu.be/{VID}?si=abc",
        f"https://www.youtube.com/shorts/{VID}",
        f"https://www.youtube.com/embed/{VID}",
        VID,
    ],
)
def test_extract_video_id(url):
    assert extract_video_id(url) == VID


@pytest.mark.parametrize("url", ["", "https://example.com", "not a url at all"])
def test_extract_video_id_invalid(url):
    assert extract_video_id(url) is None


def test_format_transcript():
    snippets = [SimpleNamespace(text="hello", start=0.0), SimpleNamespace(text="world", start=1.5)]
    assert format_transcript(snippets) == "Text: hello Start: 0.0\nText: world Start: 1.5"


def _assistant(responses, transcripts):
    loads = []

    def loader(video_id):
        loads.append(video_id)
        return transcripts[video_id]

    bot = VideoAssistant(
        llm=FakeListLLM(responses=responses),
        embeddings=DeterministicFakeEmbedding(size=32),
        transcript_loader=loader,
    )
    return bot, loads


def test_summarize_and_answer_use_cached_transcript():
    bot, loads = _assistant(["a summary", "an answer"], {VID: "Text: hi Start: 0.0"})
    url = f"https://youtu.be/{VID}"
    assert bot.summarize(url) == "a summary"
    assert bot.answer(url, "what?") == "an answer"
    assert loads == [VID]  # fetched once, reused for Q&A


def test_switching_videos_uses_new_transcript():
    other = "abcdefghijk"
    bot, loads = _assistant(["a1", "a2"], {VID: "Text: one Start: 0", other: "Text: two Start: 0"})
    bot.answer(VID, "q")
    bot.answer(other, "q")
    assert loads == [VID, other]


def test_invalid_inputs_raise_friendly_errors():
    bot, _ = _assistant([], {})
    with pytest.raises(TranscriptError, match="valid YouTube"):
        bot.summarize("https://example.com")
    with pytest.raises(TranscriptError, match="question"):
        bot.answer(VID, "   ")
