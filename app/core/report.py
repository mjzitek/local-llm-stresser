"""Shared stat/printing helpers for benchmark scripts."""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

from app.core.client import RequestRecord


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

    if extra:
        c.rule("[bold]Extras")
        for k, v in extra.items():
            c.print(f"{k}: {v}")

    errs = [r.error for r in records if not r.ok and r.error]
    if errs:
        c.rule("[red]Errors (first 3)")
        for e in errs[:3]:
            c.print(e)
