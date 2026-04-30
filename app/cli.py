"""stresser — CLI entrypoint.

    stresser                        # banner: runtimes + models found on this machine
    stresser detect                 # same, full model lists
    stresser runtimes               # list runtime presets
    stresser models                 # list models exposed by a chosen runtime
    stresser workloads              # list available workloads
    stresser task simple            # run a single workload
    stresser task summarize --size large --concurrency 4
    stresser decode                 # decode-throughput benchmark
    stresser prefill                # prefill speed sweep
    stresser concurrency            # concurrency sweep

Common flags (all benchmark commands):
    --runtime {ollama,llamacpp,lmstudio,custom}   default: env or auto-detected
    --base-url URL                                override runtime default
    --model NAME                                  default: env or first available
    --no-banner                                   skip the startup detection banner
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from app.core.config import Config, load_config
from app.core.discovery import Detected, detect_all, format_detection
from app.core.runtimes import RUNTIMES, list_models, resolve_base_url


# ---------------- detection / banner ---------------- #

_DETECTION_CACHE: list[Detected] | None = None


def _detect_sync() -> list[Detected]:
    global _DETECTION_CACHE
    if _DETECTION_CACHE is None:
        _DETECTION_CACHE = asyncio.run(detect_all())
    return _DETECTION_CACHE


def _print_banner(show_models: bool = False, max_models: int = 6) -> list[Detected]:
    results = _detect_sync()
    print("Detected LLM runtimes on this machine:")
    print(format_detection(results, show_models=show_models, max_models=max_models))
    return results


def _auto_resolve(args) -> Config:
    """Resolve runtime + model. Prompts the user for anything not pre-specified
    on the CLI (or in env) when stdin is a TTY. When stdin isn't a TTY (piped /
    CI), falls back to first-available auto-pick to keep scripts working.
    """
    from app.interactive import _menu  # reuse the same menu helper

    interactive = sys.stdin.isatty()

    env_runtime = os.environ.get("LLM_RUNTIME")
    env_model = os.environ.get("LLM_MODEL")
    runtime = args.runtime or env_runtime
    model = args.model or env_model
    base_url = args.base_url

    results = _detect_sync()
    avail = [d for d in results if d.available]

    if not avail:
        print("[warn] no local LLM runtime detected on default ports; "
              "request will likely fail", file=sys.stderr)
        return load_config(runtime=runtime, base_url=base_url, model=model, api_key=args.api_key)

    # 1. Resolve runtime
    chosen: Detected | None = None
    if base_url:
        # explicit url wins; trust the user
        pass
    elif runtime:
        chosen = next((d for d in avail if d.name == runtime), None)
        if not chosen:
            print(f"[auto] runtime={runtime} not responding; falling back")
            chosen = avail[0]
    elif len(avail) == 1:
        chosen = avail[0]
    elif interactive:
        idx = _menu(
            "Pick a runtime:",
            [f"{d.name}  ({d.base_url}, {len(d.models)} models)" for d in avail],
        )
        chosen = avail[idx]
        print()
    else:
        chosen = avail[0]
        print(f"[auto] using runtime={chosen.name} ({chosen.base_url})")

    if chosen:
        runtime = chosen.name
        base_url = chosen.base_url

    # 2. Resolve model
    if chosen and chosen.models:
        if model and model in chosen.models:
            pass
        elif model:
            # tolerate missing :latest suffix
            alt = next((m for m in chosen.models if m.split(":")[0] == model.split(":")[0]), None)
            if alt:
                print(f"[auto] model={model} not loaded; using {alt} (same family)")
                model = alt
            elif interactive:
                print(f"[!] model '{model}' is not loaded on {chosen.name}.")
                idx = _menu("Pick a model:", chosen.models)
                model = chosen.models[idx]
                print()
            else:
                model = chosen.models[0]
                print(f"[auto] model not loaded; using {model}")
        elif len(chosen.models) == 1:
            model = chosen.models[0]
        elif interactive:
            idx = _menu("Pick a model:", chosen.models)
            model = chosen.models[idx]
            print()
        else:
            model = chosen.models[0]
            print(f"[auto] using model={model}")

    return load_config(runtime=runtime, base_url=base_url, model=model, api_key=args.api_key)


# ---------------- subcommands ---------------- #

def cmd_detect(_args) -> int:
    _print_banner(show_models=True, max_models=200)
    return 0


def cmd_runtimes(_args) -> int:
    print(f"{'name':<10} {'default base_url':<35} description")
    for r in RUNTIMES.values():
        print(f"{r.name:<10} {r.default_base_url:<35} {r.description}")
    return 0


def cmd_models(args) -> int:
    base = resolve_base_url(args.runtime or "ollama", args.base_url)
    try:
        models = list_models(base, api_key=args.api_key or "not-needed")
    except Exception as e:
        print(f"error querying {base}/models: {e}", file=sys.stderr)
        return 2
    if not models:
        print(f"(no models reported by {base})")
        return 0
    print(f"# models at {base}")
    for m in models:
        print(m)
    return 0


def cmd_workloads(_args) -> int:
    from app.benchmarks.workloads import list_workloads
    print(f"{'name':<14} description")
    for name, desc in list_workloads():
        print(f"{name:<14} {desc}")
    return 0


def cmd_task(args) -> int:
    from app.benchmarks import task
    cfg = _auto_resolve(args)
    return asyncio.run(task.run(
        cfg,
        workload=args.workload,
        file=args.file,
        size=args.size,
        n=args.n,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        warmup=args.warmup,
        show_outputs=args.show_outputs,
    ))


def cmd_decode(args) -> int:
    from app.benchmarks import decode_speed
    cfg = _auto_resolve(args)
    return asyncio.run(decode_speed.run(
        cfg, n=args.n, max_tokens=args.max_tokens, warmup=args.warmup
    ))


def cmd_prefill(args) -> int:
    from app.benchmarks import prefill
    cfg = _auto_resolve(args)
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    return asyncio.run(prefill.run(cfg, sizes=sizes, reps=args.reps))


def cmd_concurrency(args) -> int:
    from app.benchmarks import concurrency
    cfg = _auto_resolve(args)
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    return asyncio.run(concurrency.run(
        cfg, levels=levels, reqs_per_level=args.reqs_per_level, max_tokens=args.max_tokens
    ))


def cmd_suite(args) -> int:
    from app.benchmarks import suite
    cfg = _auto_resolve(args)
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    return asyncio.run(suite.run(cfg, warmup=args.warmup, only=only))


def cmd_context_stress(args) -> int:
    from app.benchmarks import context_stress
    cfg = _auto_resolve(args)
    tiers = [int(x) for x in args.tiers.split(",") if x.strip()]
    return asyncio.run(context_stress.run(
        cfg, tiers_k=tiers, depth_pct=args.depth, auto_max=not args.no_auto_max,
        answer_max_tokens=args.answer_max_tokens,
    ))


def cmd_vision(args) -> int:
    from app.benchmarks import vision
    cfg = _auto_resolve(args)
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    return asyncio.run(vision.run(cfg, only=only))


# ---------------- argparse wiring ---------------- #

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--runtime", choices=list(RUNTIMES), default=None,
                   help="runtime preset (default: $LLM_RUNTIME or auto-detect)")
    p.add_argument("--base-url", default=None, help="override OpenAI-compat base URL")
    p.add_argument("--model", default=None, help="model id (default: $LLM_MODEL or auto-pick)")
    p.add_argument("--api-key", default=None)
    p.add_argument("--no-banner", action="store_true", help="skip startup detection banner")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stresser",
        description="Stress-test local LLM runtimes (Ollama / llama.cpp / LM Studio).",
    )
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("detect", help="probe localhost for LLM runtimes + models").set_defaults(func=cmd_detect)
    sub.add_parser("runtimes", help="list runtime presets").set_defaults(func=cmd_runtimes)
    sub.add_parser("workloads", help="list workload presets").set_defaults(func=cmd_workloads)

    pm = sub.add_parser("models", help="list models the runtime is serving")
    _add_common(pm); pm.set_defaults(func=cmd_models)

    pt = sub.add_parser("task", help="run a single workload")
    _add_common(pt)
    pt.add_argument("workload", help="workload name (see `stresser workloads`)")
    pt.add_argument("--file", default=None, help="filename under files/prompts/ for summarize/qa")
    pt.add_argument("--size", choices=["short", "medium", "large", "xlarge"], default=None,
                    help="for summarize/qa: text size preset (overridden by --file)")
    pt.add_argument("--n", type=int, default=5)
    pt.add_argument("--concurrency", type=int, default=1)
    pt.add_argument("--max-tokens", type=int, default=None)
    pt.add_argument("--warmup", type=int, default=1)
    pt.add_argument("--show-outputs", action="store_true",
                    help="print each LLM response in addition to stats")
    pt.set_defaults(func=cmd_task)

    pd = sub.add_parser("decode", help="decode-throughput benchmark")
    _add_common(pd)
    pd.add_argument("--n", type=int, default=5)
    pd.add_argument("--max-tokens", type=int, default=1024)
    pd.add_argument("--warmup", type=int, default=1)
    pd.set_defaults(func=cmd_decode)

    pp = sub.add_parser("prefill", help="prefill-speed sweep across prompt sizes")
    _add_common(pp)
    pp.add_argument("--sizes", default="512,2048,8192,16384")
    pp.add_argument("--reps", type=int, default=3)
    pp.set_defaults(func=cmd_prefill)

    pc = sub.add_parser("concurrency", help="concurrency sweep")
    _add_common(pc)
    pc.add_argument("--levels", default="1,2,4,8,16")
    pc.add_argument("--reqs-per-level", type=int, default=16)
    pc.add_argument("--max-tokens", type=int, default=256)
    pc.set_defaults(func=cmd_concurrency)

    ps = sub.add_parser("suite", help="run the fixed scenario suite (cross-machine comparable)")
    _add_common(ps)
    ps.add_argument("--warmup", type=int, default=1)
    ps.add_argument("--only", default=None,
                    help="comma-separated subset of scenario names")
    ps.set_defaults(func=cmd_suite)

    pcs = sub.add_parser("context-stress",
                          help="needle-in-a-haystack: how well does the model use its context?")
    _add_common(pcs)
    pcs.add_argument("--tiers", default="1,4,16,32,64,128",
                     help="comma-separated context-token tiers in K (e.g. 1,4,16,64)")
    pcs.add_argument("--depth", type=float, default=0.5,
                     help="needle position as fraction of prompt (0=start, 1=end)")
    pcs.add_argument("--no-auto-max", action="store_true",
                     help="don't auto-cap tiers at the model's reported max context")
    pcs.add_argument("--answer-max-tokens", type=int, default=4096,
                     help="max tokens for the model's answer (default 4096; "
                          "reasoning models burn lots of these on <think> blocks)")
    pcs.set_defaults(func=cmd_context_stress)

    pv = sub.add_parser("vision",
                        help="vision/OCR benchmark — needs a multimodal model (moondream, gemma4, llava)")
    _add_common(pv)
    pv.add_argument("--only", default=None,
                    help="comma-separated substrings; only test images whose name contains one")
    pv.set_defaults(func=cmd_vision)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # No subcommand => launch interactive wizard.
    if not args.cmd:
        from app.interactive import run_wizard
        try:
            return run_wizard()
        except (KeyboardInterrupt, EOFError):
            print("\naborted.")
            return 130

    # Banner before benchmark commands (skip for pure-info commands).
    benchmark_cmds = {"task", "decode", "prefill", "concurrency"}
    if args.cmd in benchmark_cmds and not getattr(args, "no_banner", False):
        _print_banner(show_models=False)
        print()

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
