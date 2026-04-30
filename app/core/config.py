"""Centralized config: env defaults overridable per-call by CLI flags."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from app.core.runtimes import resolve_base_url

load_dotenv()


@dataclass(frozen=True)
class Config:
    base_url: str
    api_key: str
    model: str
    runtime: str  # "ollama" | "llamacpp" | "lmstudio" | "custom"


def load_config(
    *,
    runtime: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> Config:
    env_runtime = os.environ.get("LLM_RUNTIME", "ollama")
    env_base = os.environ.get("LLM_BASE_URL")
    env_model = os.environ.get("LLM_MODEL", "llama3.1:8b")
    env_key = os.environ.get("LLM_API_KEY", "not-needed")

    final_runtime = runtime or env_runtime
    # explicit > env > runtime default
    final_base = base_url or env_base or resolve_base_url(final_runtime, None)
    return Config(
        base_url=final_base.rstrip("/"),
        api_key=api_key or env_key,
        model=model or env_model,
        runtime=final_runtime,
    )
