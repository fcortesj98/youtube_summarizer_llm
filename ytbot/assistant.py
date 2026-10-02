"""Summarization and retrieval-augmented Q&A over a single YouTube video."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import models
from .config import settings
from .prompts import QA_PROMPT, SUMMARY_PROMPT
from .transcript import TranscriptError, extract_video_id, fetch_transcript, format_transcript


class VideoAssistant:
    """Summarizes a video and answers questions about it.

    Transcripts and vector indexes are cached per video ID, so switching URLs always
    uses the right video, and repeat questions about the same video skip re-embedding.
    Dependencies are injectable to make the class easy to test.
    """

    def __init__(
        self,
        llm=None,
        embeddings=None,
        transcript_loader: Callable[[str], str] | None = None,
        cache_size: int = 8,
    ):
        self._llm = llm
        self._embeddings = embeddings
        self._load = transcript_loader or self._default_loader
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        self._transcript = lru_cache(maxsize=cache_size)(self._load)
        self._index = lru_cache(maxsize=cache_size)(self._build_index)

    # Lazily created so the UI can start without contacting watsonx.
    @property
    def llm(self):
        if self._llm is None:
            self._llm = models.get_llm()
        return self._llm

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = models.get_embeddings()
        return self._embeddings

    @staticmethod
    def _default_loader(video_id: str) -> str:
        return format_transcript(fetch_transcript(video_id, settings.transcript_languages))

    def _build_index(self, video_id: str) -> FAISS:
        chunks = self._splitter.split_text(self._transcript(video_id))
        return FAISS.from_texts(chunks, self.embeddings)

    @staticmethod
    def _video_id(url: str) -> str:
        video_id = extract_video_id(url or "")
        if not video_id:
            raise TranscriptError("Please enter a valid YouTube video URL.")
        return video_id

    def summarize(self, url: str) -> str:
        transcript = self._transcript(self._video_id(url))
        chain = SUMMARY_PROMPT | self.llm | StrOutputParser()
        return chain.invoke({"transcript": transcript}).strip()

    def answer(self, url: str, question: str) -> str:
        if not question or not question.strip():
            raise TranscriptError("Please enter a question about the video.")
        docs = self._index(self._video_id(url)).similarity_search(question, k=settings.top_k)
        context = "\n".join(doc.page_content for doc in docs)
        chain = QA_PROMPT | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question}).strip()
