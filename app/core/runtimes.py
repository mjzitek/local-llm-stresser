"""Runtime presets and model discovery.

A runtime preset = (default base_url, how to list models). All three runtimes
expose `/v1/models` via OpenAI compat; Ollama additionally has `/api/tags`
with richer info, but we stick to the OpenAI surface for portability.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class Runtime:
    name: str
    default_base_url: str
    description: str


RUNTIMES: dict[str, Runtime] = {
    "ollama": Runtime("ollama", "http://localhost:11434/v1", "Ollama (default port 11434)"),
    "llamacpp": Runtime("llamacpp", "http://localhost:8080/v1", "llama.cpp llama-server (default 8080)"),
    "lmstudio": Runtime("lmstudio", "http://localhost:1234/v1", "LM Studio (default 1234)"),
    "custom": Runtime("custom", "", "Use --base-url"),
}


def resolve_base_url(runtime: str, base_url: str | None) -> str:
    if base_url:
        return base_url.rstrip("/")
    if runtime not in RUNTIMES:
        raise ValueError(f"unknown runtime: {runtime}")
    rt = RUNTIMES[runtime]
    if not rt.default_base_url:
        raise ValueError("runtime=custom requires --base-url")
    return rt.default_base_url


def list_models(base_url: str, api_key: str = "not-needed", timeout: float = 5.0) -> list[str]:
    """Hit /models on an OpenAI-compat endpoint. Returns sorted model ids."""
    with httpx.Client(timeout=timeout) as c:
        r = c.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        r.raise_for_status()
        data = r.json()
    items = data.get("data", []) if isinstance(data, dict) else []
    return sorted({m.get("id") for m in items if m.get("id")})
