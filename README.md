# 🎬 YouTube Summarizer & Q&A

Paste a YouTube link to get a one-paragraph summary of the video, then ask questions about it. Answers are grounded in the video's own transcript through retrieval-augmented generation (RAG).

**Stack:** Gradio · LangChain · FAISS · IBM watsonx.ai (Granite LLM + Slate embeddings)

## How it works

```
YouTube URL ──► transcript ──┬──► LLM ──────────────────────────► summary
                             └──► chunks ──► embeddings ──► FAISS
                                                              │
                                     question ──► top-k chunks ┴──► LLM ──► answer
```

1. **Transcript**: fetched with `youtube-transcript-api`. A manual English transcript is used when one exists; otherwise the auto-generated one.
2. **Summary**: the full transcript goes to the LLM with a summarization prompt.
3. **Q&A**: the transcript is split into chunks, embedded and indexed in FAISS. The most relevant chunks are passed to the LLM as context for each question.

Transcripts and indexes are cached per video, so follow-up questions don't re-fetch or re-embed anything.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # optional: add your watsonx credentials
python app.py           # → http://localhost:7860
```

In the IBM Skills Network lab, the defaults work as-is with no API key. Anywhere else, set `WATSONX_APIKEY` and `WATSONX_PROJECT_ID` in `.env`.

## Configuration

All settings are environment variables (see [`.env.example`](.env.example)):

| Variable | Default | Purpose |
|---|---|---|
| `WATSONX_URL` | `https://us-south.ml.cloud.ibm.com` | watsonx.ai endpoint |
| `WATSONX_PROJECT_ID` | `skills-network` | watsonx project |
| `WATSONX_APIKEY` | *(none)* | IBM Cloud API key |
| `LLM_MODEL_ID` | `ibm/granite-8b-code-instruct` | Generation model |
| `EMBEDDING_MODEL_ID` | `ibm/slate-30m-english-rtrvr-v2` | Embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `200` / `20` | Transcript chunking (characters) |
| `TOP_K` | `7` | Chunks retrieved per question |
| `TRANSCRIPT_LANGUAGES` | `en` | Preferred languages, comma-separated |

## Project structure

```
├── app.py               # Entry point
├── ytbot/
│   ├── config.py        # Settings from environment
│   ├── transcript.py    # URL parsing, fetching, formatting
│   ├── prompts.py       # Summary and Q&A prompt templates
│   ├── models.py        # watsonx LLM + embeddings
│   ├── assistant.py     # VideoAssistant: summarize() and answer()
│   └── ui.py            # Gradio interface
└── tests/               # Offline tests (no network or credentials needed)
```

## Tests

```bash
pytest
```

## Acknowledgements

Built as part of the [IBM RAG and Agentic AI Professional Certificate](https://www.coursera.org/professional-certificates/ibm-rag-and-agentic-ai), offered by IBM through Coursera. The certificate covers LangChain, LangGraph, RAG pipelines, vector databases, multimodal AI, and agentic frameworks such as CrewAI, AG2, BeeAI, and the Model Context Protocol.

Model access is provided by [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai).
