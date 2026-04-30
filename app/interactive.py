"""Interactive wizard: pick test, pick model, quick or detailed, run."""
from __future__ import annotations

import asyncio
from pathlib import Path

from app.benchmarks.workloads import REGISTRY as WORKLOADS, SIZE_FILES, list_workloads
from app.core.config import load_config
from app.core.discovery import detect_all, format_detection
from app.core.runtimes import RUNTIMES

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "files" / "prompts"


# ---------- prompt helpers ---------- #

def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        v = input(f"{prompt}{suffix}: ").strip()
        if v:
            return v
        if default is not None:
            return default


def _ask_int(prompt: str, default: int) -> int:
    while True:
        v = input(f"{prompt} [{default}]: ").strip()
        if not v:
            return default
        try:
            return int(v)
        except ValueError:
            print("  not a number, try again")


def _menu(prompt: str, options: list[str], default_idx: int = 0) -> int:
    print(prompt)
    for i, o in enumerate(options, 1):
        marker = "*" if (i - 1) == default_idx else " "
        print(f"  {marker} {i:>2}) {o}")
    while True:
        v = input(f"choice [{default_idx + 1}]: ").strip()
        if not v:
            return default_idx
        try:
            n = int(v)
            if 1 <= n <= len(options):
                return n - 1
        except ValueError:
            pass
        print("  invalid; pick a number from the list")


def _ask_yn(prompt: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        v = input(f"{prompt} [{d}]: ").strip().lower()
        if not v:
            return default
        if v in ("y", "yes"):
            return True
        if v in ("n", "no"):
            return False


# ---------- defaults per test ---------- #

QUICK_TASK_N = 3
QUICK_TASK_CONCURRENCY = 1

DECODE_DEFAULT_N = 5
DECODE_DEFAULT_MAX_TOKENS = 1024

PREFILL_DEFAULT_SIZES = "512,2048,8192,16384"
PREFILL_DEFAULT_REPS = 3

CONCURRENCY_DEFAULT_LEVELS = "1,2,4,8,16"
CONCURRENCY_DEFAULT_REQS = 16
CONCURRENCY_DEFAULT_MAX_TOKENS = 256


# ---------- wizard ---------- #

def run_wizard() -> int:
    print("== Local LLM Stress Tester ==")
    print("Detecting runtimes on this machine...\n")
    detections = asyncio.run(detect_all())
    print(format_detection(detections, show_models=False))
    print()

    avail = [d for d in detections if d.available]
    if not avail:
        print("No runtimes detected on default ports.")
        print("Start Ollama / llama-server / LM Studio and re-run.")
        return 1

    # 1. runtime
    if len(avail) == 1:
        chosen = avail[0]
        print(f"Using only available runtime: {chosen.name} ({chosen.base_url})\n")
    else:
        idx = _menu(
            "Pick a runtime:",
            [f"{d.name}  ({d.base_url}, {len(d.models)} models)" for d in avail],
        )
        chosen = avail[idx]
        print()

    if not chosen.models:
        print(f"Runtime {chosen.name} reports no models loaded. Load one first.")
        return 1

    # 2. test (workloads + focused benchmarks in one flat menu)
    workloads = list_workloads()
    options: list[tuple[str, str, str]] = []
    for name, desc in workloads:
        options.append(("workload", name, f"{name:<14} — {desc}"))
    options.extend([
        ("benchmark", "decode",      f"{'decode':<14} — Decode-throughput benchmark (long output)"),
        ("benchmark", "prefill",     f"{'prefill':<14} — Prefill-speed sweep across prompt sizes"),
        ("benchmark", "concurrency", f"{'concurrency':<14} — Concurrency sweep — find the throughput knee"),
    ])
    tidx = _menu("Pick a test:", [o[2] for o in options])
    kind, key, _ = options[tidx]
    print()

    # 3. model
    if len(chosen.models) == 1:
        model = chosen.models[0]
        print(f"Using only loaded model: {model}\n")
    else:
        midx = _menu("Pick a model:", chosen.models)
        model = chosen.models[midx]
        print()

    # 4. quick vs detailed
    mode_idx = _menu(
        "Run mode:",
        ["Quick     — sensible defaults, just run it",
         "Detailed  — set request count, concurrency, max tokens, etc."],
    )
    detailed = mode_idx == 1
    print()

    cfg = load_config(runtime=chosen.name, base_url=chosen.base_url, model=model)

    if kind == "workload":
        return _run_task(cfg, workload_name=key, detailed=detailed)
    if key == "decode":
        return _run_decode(cfg, detailed=detailed)
    if key == "prefill":
        return _run_prefill(cfg, detailed=detailed)
    if key == "concurrency":
        return _run_concurrency(cfg, detailed=detailed)
    return 1


# ---------- per-test runners ---------- #

def _pick_size() -> str:
    """Pick a text size for summarize/qa workloads."""
    labels = []
    keys = list(SIZE_FILES)
    descriptions = {
        "short":  "~85 tokens   (one paragraph)",
        "medium": "~650 tokens  (article excerpt)",
        "large":  "~1.6K tokens (full article)",
        "xlarge": "~4.7K tokens (long-form essay)",
    }
    for k in keys:
        labels.append(f"{k:<7} — {descriptions.get(k, '')}")
    idx = _menu("Pick a text size:", labels, default_idx=1)  # default medium
    return keys[idx]


def _run_task(cfg, *, workload_name: str, detailed: bool) -> int:
    from app.benchmarks import task

    size_arg: str | None = None
    if workload_name in ("summarize", "qa"):
        size_arg = _pick_size()
        print()

    workload_max_tokens = WORKLOADS[workload_name]().max_tokens

    if detailed:
        n = _ask_int("How many requests?", QUICK_TASK_N)
        concurrency = _ask_int("Concurrency (parallel in-flight)?", QUICK_TASK_CONCURRENCY)
        max_tokens = _ask_int("Max output tokens (override workload default)?", workload_max_tokens)
        show = _ask_yn("Show LLM outputs (in addition to stats)?", default=False)
        print()
    else:
        n, concurrency, max_tokens = QUICK_TASK_N, QUICK_TASK_CONCURRENCY, workload_max_tokens
        # quick mode: show outputs by default for content-y workloads
        show = workload_name in ("simple", "chat", "code-function", "code-refactor",
                                 "summarize", "qa", "extraction", "compare-contrast")

    return asyncio.run(task.run(
        cfg, workload=workload_name, size=size_arg,
        n=n, concurrency=concurrency, max_tokens=max_tokens, warmup=1,
        show_outputs=show,
    ))


def _run_decode(cfg, *, detailed: bool) -> int:
    from app.benchmarks import decode_speed
    if detailed:
        n = _ask_int("How many decode runs?", DECODE_DEFAULT_N)
        max_tokens = _ask_int("Max output tokens per run?", DECODE_DEFAULT_MAX_TOKENS)
        print()
    else:
        n, max_tokens = DECODE_DEFAULT_N, DECODE_DEFAULT_MAX_TOKENS
    return asyncio.run(decode_speed.run(cfg, n=n, max_tokens=max_tokens, warmup=1))


def _run_prefill(cfg, *, detailed: bool) -> int:
    from app.benchmarks import prefill
    if detailed:
        sizes_str = _ask("Prompt sizes (chars, comma-separated)", PREFILL_DEFAULT_SIZES)
        reps = _ask_int("Repetitions per size?", PREFILL_DEFAULT_REPS)
        print()
    else:
        sizes_str, reps = PREFILL_DEFAULT_SIZES, PREFILL_DEFAULT_REPS
    sizes = [int(x) for x in sizes_str.split(",") if x.strip()]
    return asyncio.run(prefill.run(cfg, sizes=sizes, reps=reps))


def _run_concurrency(cfg, *, detailed: bool) -> int:
    from app.benchmarks import concurrency
    if detailed:
        levels_str = _ask("Concurrency levels (comma-separated)", CONCURRENCY_DEFAULT_LEVELS)
        reqs = _ask_int("Requests per level?", CONCURRENCY_DEFAULT_REQS)
        max_tokens = _ask_int("Max output tokens per request?", CONCURRENCY_DEFAULT_MAX_TOKENS)
        print()
    else:
        levels_str = CONCURRENCY_DEFAULT_LEVELS
        reqs = CONCURRENCY_DEFAULT_REQS
        max_tokens = CONCURRENCY_DEFAULT_MAX_TOKENS
    levels = [int(x) for x in levels_str.split(",") if x.strip()]
    return asyncio.run(concurrency.run(
        cfg, levels=levels, reqs_per_level=reqs, max_tokens=max_tokens
    ))
