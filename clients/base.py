"""Base LLM client types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LLMResponse:
    request_id: str
    provider: str
    model: str
    raw_text: str
    latency_sec: float
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    success: bool
    error: Optional[str] = None
    attempts: int = 1
    raw: dict[str, Any] = field(default_factory=dict)


class LLMClient(ABC):
    provider: str
    model: str

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        request_id: str,
    ) -> LLMResponse:
        raise NotImplementedError
