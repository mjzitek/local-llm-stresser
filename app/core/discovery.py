"""Probe localhost for running LLM runtimes and enumerate their models.

Each runtime preset has a default base URL. We hit `/models` on each in
parallel; whatever responds quickly is "available". Used both for the
`detect` subcommand and as a startup banner.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import httpx

from app.core.runtimes import RUNTIMES


@dataclass
class Detected:
    name: str
    base_url: str
    available: bool
    models: list[str] = field(default_factory=list)
    error: str | None = None
    latency_ms: float | None = None


async def _probe_one(name: str, base_url: str, timeout: float = 1.5) -> Detected:
    import time
    out = Detected(name=name, base_url=base_url, available=False)
    if not base_url:
        out.error = "no default url"
        return out
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(
                f"{base_url}/models",
                headers={"Authorization": "Bearer not-needed"},
            )
        out.latency_ms = (time.perf_counter() - t0) * 1000.0
        if r.status_code >= 400:
            out.error = f"HTTP {r.status_code}"
            return out
        data = r.json()
        items = data.get("data", []) if isinstance(data, dict) else []
        out.models = sorted({m.get("id") for m in items if m.get("id")})
        out.available = True
    except (httpx.HTTPError, ValueError) as e:
        out.error = f"{type(e).__name__}: {e}"
    return out


async def detect_all(timeout: float = 1.5) -> list[Detected]:
    """Probe every known runtime preset (excluding 'custom') in parallel."""
    targets = [(name, rt.default_base_url) for name, rt in RUNTIMES.items() if rt.default_base_url]
    results = await asyncio.gather(*(_probe_one(n, u, timeout) for n, u in targets))
    return list(results)


def format_detection(results: list[Detected], *, show_models: bool = True, max_models: int = 20) -> str:
    lines = []
    for d in results:
        if d.available:
            head = f"  ✓ {d.name:<10} {d.base_url}  ({len(d.models)} models, {d.latency_ms:.0f}ms)"
        else:
            head = f"  ✗ {d.name:<10} {d.base_url}  [{d.error}]"
        lines.append(head)
        if d.available and show_models and d.models:
            shown = d.models[:max_models]
            for m in shown:
                lines.append(f"      - {m}")
            if len(d.models) > max_models:
                lines.append(f"      ... and {len(d.models) - max_models} more")
    return "\n".join(lines)
