"""Background system metrics sampler.

Samples CPU/RAM always; samples NVIDIA GPU via `nvidia-smi` if present and
Apple Silicon GPU via `powermetrics` if present (powermetrics needs sudo so
it's opt-in via env LLM_USE_POWERMETRICS=1).
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field

import psutil


@dataclass
class Sample:
    t: float
    cpu_pct: float
    ram_used_gb: float
    ram_pct: float
    gpu_util_pct: float | None = None
    gpu_mem_used_gb: float | None = None
    gpu_temp_c: float | None = None
    gpu_power_w: float | None = None


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _read_nvidia() -> tuple[float, float, float, float] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2.0,
        )
        # take first GPU
        first = out.strip().splitlines()[0]
        util, mem_mib, temp, power = [p.strip() for p in first.split(",")]
        return float(util), float(mem_mib) / 1024.0, float(temp), float(power)
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


class SysMonitor:
    """Async sampler. Use `async with` then read `.samples` after it's stopped."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.samples: list[Sample] = []
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._has_nvidia = _have("nvidia-smi")

    async def __aenter__(self):
        self._stop.clear()
        # prime cpu_percent (first call returns 0)
        psutil.cpu_percent(None)
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *_):
        self._stop.set()
        if self._task:
            await self._task

    async def _run(self):
        while not self._stop.is_set():
            s = Sample(
                t=time.time(),
                cpu_pct=psutil.cpu_percent(None),
                ram_used_gb=psutil.virtual_memory().used / (1024**3),
                ram_pct=psutil.virtual_memory().percent,
            )
            if self._has_nvidia:
                if g := _read_nvidia():
                    s.gpu_util_pct, s.gpu_mem_used_gb, s.gpu_temp_c, s.gpu_power_w = g
            self.samples.append(s)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                pass

    def summary(self) -> dict:
        if not self.samples:
            return {}

        def stats(vals: list[float]) -> dict:
            if not vals:
                return {}
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            return {
                "avg": sum(vals_sorted) / n,
                "max": vals_sorted[-1],
                "p95": vals_sorted[min(n - 1, int(n * 0.95))],
            }

        out: dict = {
            "n_samples": len(self.samples),
            "cpu_pct": stats([s.cpu_pct for s in self.samples]),
            "ram_pct": stats([s.ram_pct for s in self.samples]),
            "ram_used_gb": stats([s.ram_used_gb for s in self.samples]),
        }
        gpu_util = [s.gpu_util_pct for s in self.samples if s.gpu_util_pct is not None]
        if gpu_util:
            out["gpu_util_pct"] = stats(gpu_util)
            out["gpu_mem_used_gb"] = stats(
                [s.gpu_mem_used_gb for s in self.samples if s.gpu_mem_used_gb is not None]
            )
            out["gpu_temp_c"] = stats(
                [s.gpu_temp_c for s in self.samples if s.gpu_temp_c is not None]
            )
            out["gpu_power_w"] = stats(
                [s.gpu_power_w for s in self.samples if s.gpu_power_w is not None]
            )
        return out
