"""Concurrency sweep: aggregate tokens/sec + per-request p95 vs in-flight count."""
from __future__ import annotations

import asyncio
import time

from app.core.client import LLMClient, RequestRecord
from app.core.config import Config
from app.core.recorder import finish_run, new_run, save_records, save_samples
from app.core.report import percentiles, summarize_records
from app.core.sysmon import SysMonitor

PROMPT = "Explain how HTTP works in about 200 words."


async def _worker(client: LLMClient, sem: asyncio.Semaphore, max_tokens: int, out: list[RequestRecord]):
    async with sem:
        out.append(await client.stream_chat(PROMPT, max_tokens=max_tokens))


async def _run_level(client: LLMClient, concurrency: int, reqs: int, max_tokens: int):
    sem = asyncio.Semaphore(concurrency)
    records: list[RequestRecord] = []
    t0 = time.perf_counter()
    await asyncio.gather(*[_worker(client, sem, max_tokens, records) for _ in range(reqs)])
    return records, time.perf_counter() - t0


async def run(cfg: Config, *, levels: list[int], reqs_per_level: int = 16, max_tokens: int = 256) -> int:
    run_id = new_run(
        "concurrency", cfg.runtime, cfg.base_url, cfg.model,
        {"levels": levels, "reqs_per_level": reqs_per_level, "max_tokens": max_tokens},
    )
    print(f"run_id={run_id}  runtime={cfg.runtime}  model={cfg.model}  url={cfg.base_url}")

    all_records: list[RequestRecord] = []
    rows: list[dict] = []
    async with LLMClient(cfg) as client, SysMonitor(interval=1.0) as mon:
        await client.stream_chat("hi", max_tokens=1)  # warmup

        print(
            f"{'concurrency':>11} {'reqs':>5} {'wall_s':>7} "
            f"{'ok':>4} {'err':>4} {'agg_tps':>9} "
            f"{'ttft_p50':>9} {'ttft_p95':>9} {'lat_p50':>8} {'lat_p95':>8}"
        )
        for level in levels:
            recs, wall = await _run_level(client, level, reqs_per_level, max_tokens)
            all_records.extend(recs)
            ok = [r for r in recs if r.ok]
            total_tokens = sum(r.completion_tokens or 0 for r in ok)
            agg_tps = total_tokens / wall if wall > 0 else 0.0
            ttft = percentiles([r.ttft_ms for r in ok if r.ttft_ms is not None], (50, 95))
            lat = percentiles([r.total_ms for r in ok if r.total_ms is not None], (50, 95))
            row = {
                "concurrency": level, "reqs": len(recs), "wall_s": wall,
                "ok": len(ok), "err": len(recs) - len(ok), "agg_tps": agg_tps,
                "ttft_p50": ttft[50], "ttft_p95": ttft[95],
                "lat_p50": lat[50], "lat_p95": lat[95],
            }
            rows.append(row)
            print(
                f"{level:>11} {row['reqs']:>5} {wall:>7.2f} "
                f"{row['ok']:>4} {row['err']:>4} {agg_tps:>9.1f} "
                f"{ttft[50]:>9.0f} {ttft[95]:>9.0f} {lat[50]:>8.0f} {lat[95]:>8.0f}"
            )

    save_records(run_id, all_records)
    save_samples(run_id, mon.samples)
    finish_run(run_id, {"records": summarize_records(all_records), "system": mon.summary(), "levels": rows})
    return 0 if all(r.ok for r in all_records) else 1
