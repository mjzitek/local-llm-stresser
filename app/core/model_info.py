"""Best-effort model footprint lookup so the run banner can show what's loaded.

Tries `ollama ps` first (works for Ollama), falls back to nvidia-smi process
memory. Returns a short string like "8.2 GB GPU" or empty when unknown.
"""
from __future__ import annotations

import json
import shutil
import subprocess


def _ollama_ps_for(model: str) -> str | None:
    if shutil.which("ollama") is None:
        return None
    try:
        out = subprocess.check_output(
            ["ollama", "ps", "--format", "json"],
            text=True, timeout=2.0, stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    # `ollama ps --format json` outputs one JSON per running model on newer versions,
    # or a JSON array. Be tolerant.
    rows: list[dict] = []
    out = out.strip()
    if not out:
        return None
    try:
        data = json.loads(out)
        rows = data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        for line in out.splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    for r in rows:
        name = r.get("name") or r.get("model") or ""
        if name == model or name.split(":")[0] == model.split(":")[0]:
            size_bytes = r.get("size_vram") or r.get("size") or 0
            if size_bytes:
                gb = int(size_bytes) / (1024**3)
                processor = r.get("processor", "")
                tag = " GPU" if processor and "gpu" in processor.lower() else ""
                return f"{gb:.1f} GB{tag}"
    return None


def _nvidia_total_used() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=2.0, stderr=subprocess.DEVNULL,
        )
    except subprocess.SubprocessError:
        return None
    try:
        used_mib = float(out.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None
    if used_mib < 100:  # idle desktop only
        return None
    return f"~{used_mib / 1024:.1f} GB GPU (total)"


def model_footprint(model: str) -> str:
    """Return short footprint string, or empty when unknown."""
    return _ollama_ps_for(model) or _nvidia_total_used() or ""
