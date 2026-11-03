"""Bailian / DashScope Qwen client via OpenAI-compatible API."""

from __future__ import annotations

from clients.openai_compatible import OpenAICompatibleClient
from config import MODELS, get_api_key


def make_bailian_client(model: str | None = None) -> OpenAICompatibleClient:
    meta = MODELS["bailian"]
    return OpenAICompatibleClient(
        provider="bailian",
        model=model or meta["model"],
        base_url=meta["base_url"],
        api_key=get_api_key("bailian"),
        enable_thinking=False,  # required: disable thinking for structured study
    )
