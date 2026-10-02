"""Gradio web interface."""

from __future__ import annotations

import gradio as gr

from .assistant import VideoAssistant
from .transcript import TranscriptError


def build_interface(assistant: VideoAssistant | None = None) -> gr.Blocks:
    assistant = assistant or VideoAssistant()

    def _safe(fn, *args) -> str:
        try:
            return fn(*args)
        except TranscriptError as exc:
            return f"⚠️ {exc}"
        except Exception as exc:  # surface model/network errors without crashing the UI
            return f"⚠️ Something went wrong: {exc}"

    def summarize(url):
        return _safe(assistant.summarize, url)

    def answer(url, question):
        return _safe(assistant.answer, url, question)

    with gr.Blocks(title="YouTube Summarizer & Q&A") as demo:
        gr.Markdown("## 🎬 YouTube Video Summarizer & Q&A")

        url = gr.Textbox(
            label="YouTube video URL", placeholder="https://www.youtube.com/watch?v=..."
        )

        with gr.Row():
            with gr.Column():
                summarize_btn = gr.Button("Summarize video", variant="primary")
                summary = gr.Textbox(label="Summary", lines=8)
            with gr.Column():
                question = gr.Textbox(
                    label="Ask a question about the video",
                    placeholder="What are the main takeaways?",
                )
                ask_btn = gr.Button("Ask", variant="primary")
                answer_box = gr.Textbox(label="Answer", lines=6)

        summarize_btn.click(summarize, inputs=url, outputs=summary)
        ask_btn.click(answer, inputs=[url, question], outputs=answer_box)
        question.submit(answer, inputs=[url, question], outputs=answer_box)

    return demo
