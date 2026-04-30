"""Single async OpenAI-compatible streaming client.

All LLM calls in this project go through `stream_chat`. Adapters/scripts must
not call httpx directly — keeps the request shape, error handling, and timing
math in one place.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from app.core.config import Config


@dataclass
class RequestRecord:
    model: str
    prompt_chars: int
    max_tokens: int
    ok: bool = False
    error: str | None = None
    ttft_ms: float | None = None         # send -> first content token
    total_ms: float | None = None        # send -> last token
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    decode_tps: float | None = None      # completion_tokens / (total - ttft)
    output_chars: int = 0
    chunks: int = 0
    timings: dict = field(default_factory=dict)  # raw runtime-reported timings
    output_text: str = ""  # populated only when stream_chat(..., capture_output=True)


class LLMClient:
    """Streaming OpenAI-compat chat client. One per script run."""

    def __init__(self, cfg: Config, timeout: float = 600.0):
        self.cfg = cfg
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url,
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()

    async def stream_chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int | None = None,
        model: str | None = None,
        extra: dict | None = None,
        capture_output: bool = False,
    ) -> RequestRecord:
        """Send one chat request, stream tokens, return timing record."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": model or self.cfg.model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # ask runtimes that support it for token usage in the final chunk
            "stream_options": {"include_usage": True},
        }
        if seed is not None:
            payload["seed"] = seed
        if extra:
            payload.update(extra)

        rec = RequestRecord(
            model=payload["model"],
            prompt_chars=len(prompt) + (len(system) if system else 0),
            max_tokens=max_tokens,
        )

        t0 = time.perf_counter()
        first_token_t: float | None = None
        last_token_t: float | None = None
        out_parts: list[str] = []

        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as r:
                if r.status_code >= 400:
                    body = (await r.aread()).decode("utf-8", "replace")
                    rec.error = f"HTTP {r.status_code}: {body[:500]}"
                    return rec

                async for line in r.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    # usage typically arrives in a final chunk with empty choices
                    if usage := obj.get("usage"):
                        rec.prompt_tokens = usage.get("prompt_tokens")
                        rec.completion_tokens = usage.get("completion_tokens")

                    # llama.cpp / ollama may include extra timings
                    if timings := obj.get("timings"):
                        rec.timings = timings

                    for ch in obj.get("choices", []) or []:
                        delta = ch.get("delta", {}) or {}
                        # Some runtimes/models stream reasoning tokens (deepseek-r1
                        # via Ollama) on a separate field. Count those for TTFT
                        # but only emit `content` to out_parts.
                        piece = delta.get("content")
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                        if piece or reasoning:
                            now = time.perf_counter()
                            if first_token_t is None:
                                first_token_t = now
                            last_token_t = now
                            rec.chunks += 1
                            if piece:
                                out_parts.append(piece)

            t_end = time.perf_counter()
            rec.ok = True
            rec.output_chars = sum(len(p) for p in out_parts)
            if capture_output:
                rec.output_text = "".join(out_parts)
            if first_token_t is not None:
                rec.ttft_ms = (first_token_t - t0) * 1000.0
            rec.total_ms = ((last_token_t or t_end) - t0) * 1000.0
            if (
                rec.completion_tokens
                and first_token_t is not None
                and last_token_t is not None
                and last_token_t > first_token_t
            ):
                rec.decode_tps = rec.completion_tokens / (last_token_t - first_token_t)
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            rec.error = f"{type(e).__name__}: {e}"
        return rec
