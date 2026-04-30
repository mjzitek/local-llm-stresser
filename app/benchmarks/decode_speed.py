"""Pure decode throughput.

Short prompt, force long output; measure tokens/sec from first token to last.
This isolates "how fast does this GPU/CPU decode" from prefill cost.
"""
from __future__ import annotations

from app.core.client import LLMClient
from app.core.config import Config
from app.core.model_info import model_footprint
from app.core.recorder import finish_run, new_run, save_records, save_samples
from app.core.report import print_run_banner, print_summary, summarize_records
from app.core.sysmon import SysMonitor

PROMPT = (
    "Write a long, continuous monologue about the history of computing. "
    "Do not stop early. Keep going until you are cut off."
)


async def run(cfg: Config, *, n: int = 5, max_tokens: int = 1024, warmup: int = 1) -> int:
    run_id = new_run(
        "decode_speed", cfg.runtime, cfg.base_url, cfg.model,
        {"n": n, "max_tokens": max_tokens, "warmup": warmup},
    )
    fp = model_footprint(cfg.model)
    banner = {
        "MODEL": f"{cfg.model}" + (f"   [{fp}]" if fp else ""),
        "RUNTIME": f"{cfg.runtime}  ({cfg.base_url})",
        "TEST": f"decode-throughput  (n={n}, max_tokens={max_tokens})",
        "RUN ID": run_id,
    }
    print_run_banner(banner)

    async with LLMClient(cfg) as client, SysMonitor(interval=1.0) as mon:
        for i in range(warmup):
            print(f"warmup {i+1}/{warmup}...")
            await client.stream_chat(PROMPT, max_tokens=max_tokens)

        records = []
        for i in range(n):
            r = await client.stream_chat(PROMPT, max_tokens=max_tokens)
            tag = "OK" if r.ok else f"ERR {r.error}"
            tps = f"{r.decode_tps:.1f} tok/s" if r.decode_tps else "?"
            ttft = f"{r.ttft_ms:.0f}ms" if r.ttft_ms else "?"
            print(f"  [{i+1}/{n}] {tag}  ttft={ttft}  decode={tps}  out_tok={r.completion_tokens}")
            records.append(r)

    save_records(run_id, records)
    save_samples(run_id, mon.samples)
    finish_run(run_id, {"records": summarize_records(records), "system": mon.summary()})
    print_summary("Decode Speed", records, extra={"system": mon.summary()})
    print_run_banner(banner)
    return 0 if all(r.ok for r in records) else 1
