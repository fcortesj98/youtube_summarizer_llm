"""Runtime settings, read from environment variables (or a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:  # python-dotenv is optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # watsonx.ai connection
    watsonx_url: str = _env("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    watsonx_project_id: str = _env("WATSONX_PROJECT_ID", "skills-network")
    watsonx_apikey: str | None = os.getenv("WATSONX_APIKEY")

    # Models
    llm_model_id: str = _env("LLM_MODEL_ID", "ibm/granite-8b-code-instruct")
    embedding_model_id: str = _env("EMBEDDING_MODEL_ID", "ibm/slate-30m-english-rtrvr-v2")
    max_new_tokens: int = _env_int("MAX_NEW_TOKENS", 900)

    # Retrieval
    chunk_size: int = _env_int("CHUNK_SIZE", 200)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 20)
    top_k: int = _env_int("TOP_K", 7)

    # Transcript
    transcript_languages: tuple[str, ...] = tuple(
        _env("TRANSCRIPT_LANGUAGES", "en").replace(" ", "").split(",")
    )

    # Web server
    server_name: str = _env("SERVER_NAME", "0.0.0.0")
    server_port: int = _env_int("SERVER_PORT", 7860)


settings = Settings()
