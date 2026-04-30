"""Best-effort host introspection for the run banner.

Returns short strings — keep the banner compact. Anything we can't determine
gracefully degrades to a sensible fallback.
"""
from __future__ import annotations

import platform
import shutil
import socket
import subprocess
from functools import lru_cache

import psutil


def _gb(bytes_: int) -> float:
    return bytes_ / (1024**3)


@lru_cache(maxsize=1)
def hostname() -> str:
    return socket.gethostname()


@lru_cache(maxsize=1)
def os_string() -> str:
    sys_ = platform.system()
    rel = platform.release()
    mach = platform.machine()
    if sys_ == "Darwin":
        # rel is the kernel version (e.g. 23.2.0); show macOS marketing version too
        try:
            mac = platform.mac_ver()[0]
            return f"macOS {mac} ({mach})"
        except Exception:
            return f"Darwin {rel} ({mach})"
    return f"{sys_} {rel} ({mach})"


@lru_cache(maxsize=1)
def cpu_string() -> str:
    """e.g. 'Apple M3 Max — 14 cores'  or  'NVIDIA Grace — 80 cores'."""
    sys_ = platform.system()
    cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 0
    name = ""
    if sys_ == "Darwin":
        try:
            name = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True, timeout=1.0, stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    elif sys_ == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name") or line.startswith("Model"):
                        name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    if not name:
        name = platform.processor() or platform.machine() or "CPU"
    return f"{name} — {cores} cores"


@lru_cache(maxsize=1)
def ram_string() -> str:
    total_gb = _gb(psutil.virtual_memory().total)
    return f"{total_gb:.0f} GB"


@lru_cache(maxsize=1)
def gpu_string() -> str:
    """NVIDIA via nvidia-smi; Apple Silicon via system_profiler; otherwise empty."""
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=2.0, stderr=subprocess.DEVNULL,
            ).strip().splitlines()
            if out:
                name, mem_mib, drv = [p.strip() for p in out[0].split(",")]
                return f"{name} ({float(mem_mib)/1024:.0f} GB VRAM, driver {drv})"
        except (subprocess.SubprocessError, ValueError):
            pass
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                text=True, timeout=3.0, stderr=subprocess.DEVNULL,
            )
            chip = ""
            cores = ""
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("Chipset Model:"):
                    chip = line.split(":", 1)[1].strip()
                elif line.startswith("Total Number of Cores:"):
                    cores = line.split(":", 1)[1].strip()
            if chip:
                return f"{chip}" + (f" ({cores} GPU cores)" if cores else "") + " — unified memory"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    return ""


def host_summary() -> dict[str, str]:
    """Return banner fields. Keys are stable; values are short strings."""
    fields = {
        "HOST": f"{hostname()}  ({os_string()})",
        "CPU": cpu_string(),
        "RAM": ram_string(),
    }
    g = gpu_string()
    if g:
        fields["GPU"] = g
    return fields
