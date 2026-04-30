"""Prompt-processing (prefill) speed.

Sweep prompt sizes with max_tokens=1 so TTFT is dominated by prefill.
"""
from __future__ import annotations

from app.core.client import LLMClient
from app.core.config import Config
from app.core.recorder import finish_run, new_run, save_records, save_samples
from app.core.report import print_summary, summarize_records
from app.core.sysmon import SysMonitor

FILLER_WORD = "lorem ipsum dolor sit amet consectetur adipiscing elit. "


def _make_prompt(target_chars: int) -> str:
    reps = max(1, target_chars // len(FILLER_WORD) + 1)
    body = (FILLER_WORD * reps)[:target_chars]
    return f"{body}\n\nQ: Reply with the single word OK."


async def run(cfg: Config, *, sizes: list[int], reps: int = 3) -> int:
    run_id = new_run(
        "prefill", cfg.runtime, cfg.base_url, cfg.model,
        {"sizes": sizes, "reps": reps},
    )
    print(f"run_id={run_id}  runtime={cfg.runtime}  model={cfg.model}  url={cfg.base_url}")

    all_records = []
    async with LLMClient(cfg) as client, SysMonitor(interval=1.0) as mon:
        await client.stream_chat("hi", max_tokens=1)  # warmup

        print(f"{'size_chars':>10} {'rep':>4} {'ttft_ms':>9} {'ptokens':>8} {'prefill_tps':>12}")
        for size in sizes:
            prompt = _make_prompt(size)
            for rep in range(reps):
                r = await client.stream_chat(prompt, max_tokens=1)
                all_records.append(r)
                pt = r.prompt_tokens or 0
                tps = (pt * 1000.0 / r.ttft_ms) if (r.ttft_ms and pt) else None
                print(
                    f"{size:>10} {rep+1:>4} "
                    f"{(r.ttft_ms or 0):>9.0f} {pt:>8} "
                    f"{(tps or 0):>12.1f}"
                )

    save_records(run_id, all_records)
    save_samples(run_id, mon.samples)
    finish_run(run_id, {"records": summarize_records(all_records), "system": mon.summary()})
    print_summary("Prefill", all_records)
    return 0 if all(r.ok for r in all_records) else 1
