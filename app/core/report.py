"""Shared stat/printing helpers for benchmark scripts."""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

from app.core.client import RequestRecord
from app.core.host_info import host_summary

BANNER_WIDTH = 78


def print_run_banner(fields: dict[str, str]) -> None:
    """Print the loud run header/footer block.

    Includes host details (computer name, OS, CPU, RAM, GPU) ahead of the
    run-specific fields, so a screenshot tells you both *what* ran and
    *where* it ran.
    """
    host = host_summary()
    print("=" * BANNER_WIDTH)
    all_fields = {**host, **fields}
    label_width = max(len(k) for k in all_fields)
    for k, v in host.items():
        print(f"  {k:<{label_width}} : {v}")
    print("-" * BANNER_WIDTH)
    for k, v in fields.items():
        print(f"  {k:<{label_width}} : {v}")
    print("=" * BANNER_WIDTH)


def percentiles(vals: list[float], pcts=(50, 90, 95, 99)) -> dict[int, float]:
    if not vals:
        return {p: float("nan") for p in pcts}
    s = sorted(vals)
    n = len(s)
    out = {}
    for p in pcts:
        k = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
        out[p] = s[k]
    return out


def summarize_records(records: list[RequestRecord]) -> dict:
    ok = [r for r in records if r.ok]
    errs = [r for r in records if not r.ok]
    ttft = [r.ttft_ms for r in ok if r.ttft_ms is not None]
    total = [r.total_ms for r in ok if r.total_ms is not None]
    tps = [r.decode_tps for r in ok if r.decode_tps is not None]
    comp = [r.completion_tokens for r in ok if r.completion_tokens]
    return {
        "n": len(records),
        "ok": len(ok),
        "err": len(errs),
        "ttft_ms": percentiles(ttft) if ttft else None,
        "total_ms": percentiles(total) if total else None,
        "decode_tps": percentiles(tps) if tps else None,
        "completion_tokens_avg": (sum(comp) / len(comp)) if comp else None,
    }


def print_summary(title: str, records: list[RequestRecord], extra: dict | None = None) -> None:
    s = summarize_records(records)
    c = Console()
    c.rule(f"[bold]{title}")
    c.print(f"requests: {s['n']}  ok: {s['ok']}  err: {s['err']}")
    if s["completion_tokens_avg"]:
        c.print(f"avg completion tokens: {s['completion_tokens_avg']:.0f}")

    # Per-request table — always shown so individual runs are visible.
    per = Table(title="Per-request results", show_header=True, header_style="bold")
    per.add_column("#", justify="right")
    per.add_column("status")
    per.add_column("ttft (ms)", justify="right")
    per.add_column("total (ms)", justify="right")
    per.add_column("decode tps", justify="right")
    per.add_column("out tok", justify="right")
    for i, r in enumerate(records, 1):
        per.add_row(
            str(i),
            "OK" if r.ok else "ERR",
            f"{r.ttft_ms:.0f}" if r.ttft_ms else "—",
            f"{r.total_ms:.0f}" if r.total_ms else "—",
            f"{r.decode_tps:.1f}" if r.decode_tps else "—",
            str(r.completion_tokens) if r.completion_tokens else "—",
        )
    c.print(per)

    # Percentile tables — always shown, but caveat at low n.
    def row(name: str, d: dict | None, unit: str):
        if not d:
            return
        t = Table(title=name, show_header=True, header_style="bold")
        for p in d:
            t.add_column(f"p{p}")
        t.add_row(*[f"{v:.1f}{unit}" for v in d.values()])
        c.print(t)

    row("TTFT", s["ttft_ms"], " ms")
    row("Total latency", s["total_ms"], " ms")
    row("Decode tokens/sec", s["decode_tps"], "")
    if s["n"] < 10:
        c.print(f"[dim](note: only {s['n']} samples — tail percentiles "
                f"collapse; run with --n 10+ for meaningful p95/p99)[/dim]")

    if extra:
        c.rule("[bold]Extras")
        for k, v in extra.items():
            c.print(f"{k}: {v}")

    errs = [r.error for r in records if not r.ok and r.error]
    if errs:
        c.rule("[red]Errors (first 3)")
        for e in errs[:3]:
            c.print(e)
