"""OpenAI-compatible client shared by Zhipu and Bailian."""

from __future__ import annotations

import time
from typing import Any, Optional

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from clients.base import LLMClient, LLMResponse
from config import (
    MAX_RETRIES,
    MAX_TOKENS,
    RETRY_BACKOFF_SEC,
    TEMPERATURE,
    TIMEOUT_SEC,
    mask_secret,
)


def _redact(text: str, api_key: str) -> str:
    if not text:
        return text
    if api_key and api_key in text:
        return text.replace(api_key, mask_secret(api_key))
    return text


class OpenAICompatibleClient(LLMClient):
    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str,
        api_key: str,
        enable_thinking: Optional[bool] = None,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        timeout_sec: float = TIMEOUT_SEC,
        max_retries: int = MAX_RETRIES,
        retry_backoff_sec: Optional[tuple[int, ...]] = None,
    ) -> None:
        if not api_key:
            raise ValueError(f"API key missing for provider={provider}")
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.enable_thinking = enable_thinking
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec or RETRY_BACKOFF_SEC
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout_sec,
            max_retries=0,  # we handle retries ourselves
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        request_id: str,
    ) -> LLMResponse:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        # Qwen thinking must be off for this structured study.
        if self.enable_thinking is False:
            kwargs["extra_body"] = {"enable_thinking": False}

        last_error: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(**kwargs)
                latency = time.perf_counter() - t0
                text = ""
                if resp.choices:
                    content = resp.choices[0].message.content
                    text = content if isinstance(content, str) else ("" if content is None else str(content))
                usage = getattr(resp, "usage", None)
                pt = getattr(usage, "prompt_tokens", None) if usage else None
                ct = getattr(usage, "completion_tokens", None) if usage else None
                tt = getattr(usage, "total_tokens", None) if usage else None
                if tt is None and pt is not None and ct is not None:
                    tt = pt + ct
                return LLMResponse(
                    request_id=request_id,
                    provider=self.provider,
                    model=self.model,
                    raw_text=text,
                    latency_sec=latency,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=tt,
                    success=True,
                    error=None,
                    attempts=attempt,
                    raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
                )
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_error = _redact(str(e), self.api_key)
                if attempt >= self.max_retries:
                    break
                time.sleep(
                    self.retry_backoff_sec[
                        min(attempt - 1, len(self.retry_backoff_sec) - 1)
                    ]
                )
            except APIStatusError as e:
                last_error = _redact(f"HTTP {e.status_code}: {e}", self.api_key)
                if e.status_code in {429} or e.status_code >= 500:
                    if attempt >= self.max_retries:
                        break
                    time.sleep(
                        self.retry_backoff_sec[
                            min(attempt - 1, len(self.retry_backoff_sec) - 1)
                        ]
                    )
                    continue
                # Permanent client error — do not retry.
                break
            except Exception as e:  # noqa: BLE001
                last_error = _redact(str(e), self.api_key)
                break

        return LLMResponse(
            request_id=request_id,
            provider=self.provider,
            model=self.model,
            raw_text="",
            latency_sec=0.0,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            success=False,
            error=last_error or "unknown_error",
            attempts=self.max_retries,
            raw={},
        )
