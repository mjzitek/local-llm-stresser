"""Generic task runner — runs any named workload N times, optionally concurrent."""
from __future__ import annotations

import asyncio
import time

from app.benchmarks.workloads import Workload, get_workload
from app.core.client import LLMClient, RequestRecord
from app.core.config import Config
from app.core.model_info import model_footprint
from app.core.recorder import finish_run, new_run, save_records, save_samples
from app.core.report import print_run_banner, print_summary, summarize_records
from app.core.sysmon import SysMonitor


async def _one(
    client: LLMClient, wl: Workload, max_tokens: int | None, capture: bool
) -> RequestRecord:
    return await client.stream_chat(
        wl.user, system=wl.system,
        max_tokens=max_tokens or wl.max_tokens,
        capture_output=capture,
    )


async def run(
    cfg: Config,
    *,
    workload: str,
    file: str | None = None,
    size: str | None = None,
    n: int = 5,
    concurrency: int = 1,
    max_tokens: int | None = None,
    warmup: int = 1,
    show_outputs: bool = False,
) -> int:
    wl = get_workload(workload, file=file, size=size)
    run_id = new_run(
        f"task:{wl.name}", cfg.runtime, cfg.base_url, cfg.model,
        {"n": n, "concurrency": concurrency, "max_tokens": max_tokens or wl.max_tokens,
         "warmup": warmup, "file": file, "size": size},
    )
    fp = model_footprint(cfg.model)
    banner = {
        "MODEL": f"{cfg.model}" + (f"   [{fp}]" if fp else ""),
        "RUNTIME": f"{cfg.runtime}  ({cfg.base_url})",
        "WORKLOAD": f"{wl.name}  — {wl.description}",
        "PARAMS": (f"n={n}  concurrency={concurrency}  "
                   f"max_tokens={max_tokens or wl.max_tokens}  prompt_chars={len(wl.user)}"),
        "RUN ID": run_id,
    }
    print_run_banner(banner)

    async with LLMClient(cfg) as client, SysMonitor(interval=1.0) as mon:
        for i in range(warmup):
            print(f"warmup {i+1}/{warmup}...")
            await client.stream_chat("hi", max_tokens=1)

        records: list[RequestRecord] = []
        sem = asyncio.Semaphore(concurrency)

        async def slot(i: int):
            async with sem:
                r = await _one(client, wl, max_tokens, capture=show_outputs)
                tag = "OK" if r.ok else f"ERR {r.error}"
                ttft = f"{r.ttft_ms:.0f}ms" if r.ttft_ms else "?"
                tps = f"{r.decode_tps:.1f}" if r.decode_tps else "?"
                print(f"  [{i+1}/{n}] {tag}  ttft={ttft}  decode_tps={tps}  out_tok={r.completion_tokens}")
                if show_outputs and r.output_text:
                    print(f"  --- output [{i+1}/{n}] ---")
                    for line in r.output_text.rstrip().splitlines():
                        print(f"  | {line}")
                    print(f"  --- end output [{i+1}/{n}] ---")
                records.append(r)

        t0 = time.perf_counter()
        await asyncio.gather(*(slot(i) for i in range(n)))
        wall = time.perf_counter() - t0

    save_records(run_id, records)
    save_samples(run_id, mon.samples)
    total_completion = sum(r.completion_tokens or 0 for r in records if r.ok)
    agg_tps = total_completion / wall if wall > 0 else 0.0
    summary = {
        "records": summarize_records(records),
        "system": mon.summary(),
        "wall_s": wall,
        "agg_decode_tps": agg_tps,
    }
    finish_run(run_id, summary)
    print_summary(
        f"Task: {wl.name}", records,
        extra={"wall_s": f"{wall:.2f}", "aggregate_decode_tps": f"{agg_tps:.1f}"},
    )
    print_run_banner(banner)
    return 0 if all(r.ok for r in records) else 1
