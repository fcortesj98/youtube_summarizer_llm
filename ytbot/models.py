"""watsonx.ai LLM and embedding clients (created once and reused)."""

from __future__ import annotations

from functools import lru_cache

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM

from .config import settings


def _connection() -> dict:
    conn = {"url": settings.watsonx_url, "project_id": settings.watsonx_project_id}
    if settings.watsonx_apikey:
        conn["apikey"] = settings.watsonx_apikey
    return conn


@lru_cache(maxsize=1)
def get_llm() -> WatsonxLLM:
    return WatsonxLLM(
        model_id=settings.llm_model_id,
        params={
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MAX_NEW_TOKENS: settings.max_new_tokens,
        },
        **_connection(),
    )


@lru_cache(maxsize=1)
def get_embeddings() -> WatsonxEmbeddings:
    return WatsonxEmbeddings(model_id=settings.embedding_model_id, **_connection())
