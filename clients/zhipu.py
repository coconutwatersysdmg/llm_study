"""Zhipu (智谱) GLM client via OpenAI-compatible API."""

from __future__ import annotations

from clients.openai_compatible import OpenAICompatibleClient
from config import MODELS, get_api_key


def make_zhipu_client(model: str | None = None) -> OpenAICompatibleClient:
    meta = MODELS["zhipu"]
    return OpenAICompatibleClient(
        provider="zhipu",
        model=model or meta["model"],
        base_url=meta["base_url"],
        api_key=get_api_key("zhipu"),
        enable_thinking=None,
    )
