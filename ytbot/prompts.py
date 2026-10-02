"""Prompt templates for summarization and question answering."""

from langchain_core.prompts import PromptTemplate

_TRANSCRIPT_NOTE = (
    'In the transcript, "Text" is the spoken content and "Start" is the timestamp '
    "(in seconds) at which that line begins."
)

SUMMARY_PROMPT = PromptTemplate.from_template(
    f"""You are an AI assistant that summarizes YouTube video transcripts.
Provide a concise, informative summary that captures the main points of the video.

Instructions:
1. Summarize the transcript in a single concise paragraph.
2. Ignore timestamps; focus on the spoken content.

{_TRANSCRIPT_NOTE}

Transcript:
{{transcript}}

Summary:"""
)

QA_PROMPT = PromptTemplate.from_template(
    f"""You are an expert assistant answering questions about a video, using only the
excerpts of its transcript provided below. Your answer should be:
1. Precise and free from repetition
2. Consistent with the video content
3. Well organized and easy to understand
4. Focused directly on the user's question
If the excerpts conflict, use context to give the most likely correct answer.
If they do not contain the answer, say so.

{_TRANSCRIPT_NOTE}

Relevant video excerpts:
{{context}}

Question: {{question}}

Answer:"""
)
