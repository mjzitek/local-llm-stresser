"""Fixed scenario suite — comparable across machines.

Each scenario is a fixed (prompt, max_tokens, concurrency) tuple. Run the suite
on machine A and machine B with the same `--model` and the per-scenario rows
are directly comparable.

This is *not* a score. No composite number, no leaderboard. Just the same set
of named scenarios producing the same per-scenario metrics every time.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from app.benchmarks.workloads import (
    workload_chat, workload_coding_function, workload_qa_file,
    workload_simple, workload_summarize_file,
)
from app.core.client import LLMClient, RequestRecord
from app.core.config import Config
from app.core.host_info import host_summary
from app.core.model_info import model_footprint
from app.core.recorder import finish_run, new_run, save_records, save_samples
from app.core.report import print_run_banner, summarize_records
from app.core.sysmon import SysMonitor


@dataclass
class Scenario:
    name: str
    description: str
    system: str | None
    user: str
    max_tokens: int
    concurrency: int = 1
    repeats: int = 1   # how many requests to run in this scenario


def _decode_burst_user() -> str:
    return ("Continue this monologue about distributed systems for as long as "
            "possible. Do not stop early.")


def _scenarios() -> list[Scenario]:
    """Built from existing workloads — no new prompt corpora needed."""
    s_simple = workload_simple()
    s_chat = workload_chat()
    s_sum_med = workload_summarize_file("sample_medium.txt")
    s_sum_xlg = workload_summarize_file("sample_xlarge.txt")
    s_qa_lrg = workload_qa_file("sample_large.txt")
    s_code = workload_coding_function()

    return [
        Scenario("chat-short",        "short prompt, short output (fast Q&A)",
                 s_simple.system, s_simple.user, max_tokens=80, repeats=1),
        Scenario("chat-long",         "short prompt, ~400-token output",
                 s_chat.system, s_chat.user, max_tokens=400, repeats=1),
        Scenario("decode-burst",      "short prompt, 1024-token output (decode stress)",
                 None, _decode_burst_user(), max_tokens=1024, repeats=1),
        Scenario("summarize-medium",  "~650-tok prompt, 250-tok output",
                 s_sum_med.system, s_sum_med.user, max_tokens=250, repeats=1),
        Scenario("summarize-xlarge",  "~4.7K-tok prompt, 250-tok output (long context)",
                 s_sum_xlg.system, s_sum_xlg.user, max_tokens=250, repeats=1),
        Scenario("qa-large",          "~1.6K-tok prompt, short output (RAG-shape)",
                 s_qa_lrg.system, s_qa_lrg.user, max_tokens=120, repeats=1),
        Scenario("code-gen",          "mid prompt, ~600-tok code output",
                 s_code.system, s_code.user, max_tokens=600, repeats=1),
        Scenario("concurrency-light", "short prompt × 4 in flight (batch sanity)",
                 s_simple.system, s_simple.user, max_tokens=80, concurrency=4, repeats=4),
    ]


@dataclass
class ScenarioOutcome:
    scenario: Scenario
    records: list[RequestRecord]
    wall_s: float


async def _run_scenario(client: LLMClient, s: Scenario) -> ScenarioOutcome:
    sem = asyncio.Semaphore(s.concurrency)
    records: list[RequestRecord] = []

    async def one():
        async with sem:
            r = await client.stream_chat(s.user, system=s.system, max_tokens=s.max_tokens)
            records.append(r)

    t0 = time.perf_counter()
    await asyncio.gather(*(one() for _ in range(s.repeats)))
    return ScenarioOutcome(s, records, time.perf_counter() - t0)


def _summarize_outcome(o: ScenarioOutcome) -> dict:
    ok = [r for r in o.records if r.ok]
    if not ok:
        return {"ttft_ms": None, "prefill_tps": None, "decode_tps": None, "total_ms": None}
    n = len(ok)
    avg_ttft = sum(r.ttft_ms for r in ok if r.ttft_ms) / max(1, sum(1 for r in ok if r.ttft_ms))
    avg_total = sum(r.total_ms for r in ok if r.total_ms) / max(1, sum(1 for r in ok if r.total_ms))
    avg_decode = (
        sum(r.decode_tps for r in ok if r.decode_tps)
        / max(1, sum(1 for r in ok if r.decode_tps))
    ) if any(r.decode_tps for r in ok) else None
    prefill_vals = [
        r.prompt_tokens * 1000.0 / r.ttft_ms
        for r in ok if r.prompt_tokens and r.ttft_ms
    ]
    avg_prefill = (sum(prefill_vals) / len(prefill_vals)) if prefill_vals else None
    return {
        "n": n,
        "prompt_tokens": ok[0].prompt_tokens,
        "completion_tokens_avg": sum(r.completion_tokens or 0 for r in ok) / n,
        "ttft_ms":     avg_ttft,
        "total_ms":    avg_total,
        "prefill_tps": avg_prefill,
        "decode_tps":  avg_decode,
    }


async def run(cfg: Config, *, warmup: int = 1, only: list[str] | None = None) -> int:
    scenarios = [s for s in _scenarios() if not only or s.name in only]
    fp = model_footprint(cfg.model)
    banner = {
        "MODEL":   f"{cfg.model}" + (f"   [{fp}]" if fp else ""),
        "RUNTIME": f"{cfg.runtime}  ({cfg.base_url})",
        "TEST":    f"scenario suite ({len(scenarios)} scenarios, fixed-comparable)",
    }
    run_id = new_run(
        "suite", cfg.runtime, cfg.base_url, cfg.model,
        {"scenarios": [s.name for s in scenarios], "warmup": warmup},
    )
    banner["RUN ID"] = run_id
    print_run_banner(banner)

    print(f"\n{'scenario':<19} {'pp_tok':>7} {'tg_tok':>7} {'ttft_ms':>9} "
          f"{'total_ms':>9} {'prefill_tps':>12} {'decode_tps':>11}")
    print("-" * 80)

    outcomes: list[ScenarioOutcome] = []
    summaries: list[dict] = []
    all_records: list[RequestRecord] = []

    async with LLMClient(cfg) as client, SysMonitor(interval=1.0) as mon:
        for _ in range(warmup):
            await client.stream_chat("hi", max_tokens=1)

        for s in scenarios:
            o = await _run_scenario(client, s)
            summary = _summarize_outcome(o)
            outcomes.append(o)
            summaries.append({"name": s.name, **summary})
            all_records.extend(o.records)

            print(
                f"{s.name:<19} "
                f"{(summary.get('prompt_tokens') or 0):>7} "
                f"{int(summary.get('completion_tokens_avg') or 0):>7} "
                f"{(summary.get('ttft_ms') or 0):>9.0f} "
                f"{(summary.get('total_ms') or 0):>9.0f} "
                f"{(summary.get('prefill_tps') or 0):>12.1f} "
                f"{(summary.get('decode_tps') or 0):>11.1f}"
            )

    save_records(run_id, all_records)
    save_samples(run_id, mon.samples)
    finish_run(run_id, {
        "records":   summarize_records(all_records),
        "system":    mon.summary(),
        "scenarios": summaries,
        "host":      host_summary(),
    })

    print()
    print_run_banner(banner)
    return 0 if all(r.ok for o in outcomes for r in o.records) else 1
