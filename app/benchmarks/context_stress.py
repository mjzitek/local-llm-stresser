"""Context-stress + needle-in-a-haystack (NIAH).

Tests how well a model uses its claimed context window — both numerically
(can the runtime even prefill that many tokens?) and behaviorally (does the
model actually retrieve a fact buried deep in the context?).

For each tier (in tokens), we:
  1. Build a long prompt with a unique "needle" inserted at depth_pct
  2. Ask the model to retrieve the needle
  3. Record TTFT, prefill TPS, and whether the response contains the needle

Auto-detects the model's max context length via Ollama's `/api/show` when
available, and caps tiers accordingly. For Ollama, also sets `num_ctx` on
each request so the runtime allocates enough KV cache (Ollama defaults to
2048 regardless of what the model supports).
"""
from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass

import httpx

from app.core.client import LLMClient
from app.core.config import Config
from app.core.host_info import host_summary
from app.core.model_info import model_footprint
from app.core.recorder import finish_run, new_run, save_records, save_samples
from app.core.report import print_run_banner, summarize_records
from app.core.sysmon import SysMonitor

CHARS_PER_TOKEN = 4  # rough English approximation; we use the runtime's reported token count for math

DEFAULT_TIERS_K = [1, 4, 16, 32, 64, 128, 256]

FILLER = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do "
    "eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim "
    "ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut "
    "aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit "
    "in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui "
    "officia deserunt mollit anim id est laborum. "
)


@dataclass
class NeedleResult:
    tier_k: int
    target_tokens: int
    actual_tokens: int
    depth_pct: float
    needle: str
    found: bool
    response: str
    ttft_ms: float | None
    total_ms: float | None
    prefill_tps: float | None
    ok: bool
    error: str | None


def _generate_passcode() -> str:
    word = "".join(random.choices(string.ascii_lowercase, k=6))
    num = random.randint(1000, 9999)
    return f"{word}-{num}"


def _build_haystack(target_tokens: int, depth_pct: float, needle_phrase: str) -> str:
    target_chars = target_tokens * CHARS_PER_TOKEN
    body_reps = (target_chars // len(FILLER)) + 1
    body = (FILLER * body_reps)[:target_chars]
    insert_at = int(len(body) * depth_pct)
    needle = f"\n\n--- IMPORTANT FACT ---\n{needle_phrase}\n--- END IMPORTANT FACT ---\n\n"
    return body[:insert_at] + needle + body[insert_at:]


def _detect_context_limit(base_url: str, model: str) -> int | None:
    """Ollama-only: ask `/api/show` for the model's max context length."""
    api_root = base_url.rstrip("/")
    if api_root.endswith("/v1"):
        api_root = api_root[: -len("/v1")]
    try:
        with httpx.Client(timeout=5.0) as c:
            r = c.post(f"{api_root}/api/show", json={"name": model})
            if r.status_code != 200:
                return None
            data = r.json()
    except (httpx.HTTPError, ValueError):
        return None
    # model_info dict has keys like "gemma2.context_length"
    info = data.get("model_info", {}) if isinstance(data, dict) else {}
    for k, v in info.items():
        if k.endswith(".context_length") and isinstance(v, int):
            return v
    # fallback: explicit num_ctx in modelfile parameters
    params = data.get("parameters") or ""
    m = re.search(r"num_ctx\s+(\d+)", params)
    if m:
        return int(m.group(1))
    return None


async def run(
    cfg: Config,
    *,
    tiers_k: list[int] | None = None,
    depth_pct: float = 0.5,
    auto_max: bool = True,
    answer_max_tokens: int = 64,
) -> int:
    tiers_k = sorted(tiers_k or DEFAULT_TIERS_K)
    fp = model_footprint(cfg.model)

    max_ctx = _detect_context_limit(cfg.base_url, cfg.model) if auto_max else None
    if max_ctx:
        # Allow tiers strictly within the limit, plus include the largest tier
        # that fits. Drop tiers that would obviously exceed it.
        tiers_k = [t for t in tiers_k if t * 1024 + answer_max_tokens + 256 <= max_ctx]
        if not tiers_k:
            tiers_k = [max(1, max_ctx // 2 // 1024)]

    # Set num_ctx for Ollama once for the whole run (changing it forces reload)
    target_num_ctx = max(tiers_k) * 1024 + answer_max_tokens + 512
    if max_ctx:
        target_num_ctx = min(target_num_ctx, max_ctx)
    request_extra = {"options": {"num_ctx": target_num_ctx}}

    banner = {
        "MODEL":   f"{cfg.model}" + (f"   [{fp}]" if fp else ""),
        "RUNTIME": f"{cfg.runtime}  ({cfg.base_url})",
        "TEST":    f"context-stress + NIAH "
                   f"(tiers={tiers_k} K, depth={int(depth_pct*100)}%, "
                   f"num_ctx={target_num_ctx:,})",
    }
    if max_ctx:
        banner["MAX CTX"] = f"{max_ctx:,} tokens (model metadata)"

    run_id = new_run(
        "context_stress", cfg.runtime, cfg.base_url, cfg.model,
        {"tiers_k": tiers_k, "depth_pct": depth_pct, "max_ctx": max_ctx,
         "num_ctx": target_num_ctx},
    )
    banner["RUN ID"] = run_id
    print_run_banner(banner)

    print(f"\n{'tier':<8} {'tgt_tok':>9} {'real_tok':>9} {'ttft_ms':>10} "
          f"{'prefill_tps':>12} {'found?':>10}")
    print("-" * 70)

    results: list[NeedleResult] = []
    all_records = []

    async with LLMClient(cfg) as client, SysMonitor(interval=2.0) as mon:
        await client.stream_chat("hi", max_tokens=1, extra=request_extra)  # warmup + load with num_ctx

        for tier in tiers_k:
            target_tokens = tier * 1024
            passcode = _generate_passcode()
            needle_phrase = f"The secret passcode for this conversation is {passcode}."

            haystack = _build_haystack(target_tokens, depth_pct, needle_phrase)
            full_prompt = (
                "Below is a long text. Somewhere inside it there is an important fact. "
                "Read carefully and answer the question after the text.\n\n"
                f"=== TEXT START ===\n{haystack}\n=== TEXT END ===\n\n"
                "Question: What is the secret passcode? "
                "Reply with only the passcode and nothing else."
            )

            r = await client.stream_chat(
                full_prompt,
                max_tokens=answer_max_tokens,
                capture_output=True,
                extra=request_extra,
            )
            all_records.append(r)

            prefill_tps = None
            if r.ok and r.ttft_ms and r.prompt_tokens:
                prefill_tps = r.prompt_tokens * 1000.0 / r.ttft_ms

            response = (r.output_text or "").strip()
            found = passcode.lower() in response.lower() if r.ok else False

            status_str = ("✓ YES" if found else "✗ NO") if r.ok else "ERR"
            print(
                f"{tier:>3}K     "
                f"{target_tokens:>9} "
                f"{(r.prompt_tokens or 0):>9} "
                f"{(r.ttft_ms or 0):>10.0f} "
                f"{(prefill_tps or 0):>12.1f} "
                f"{status_str:>10}"
            )
            if not r.ok:
                print(f"     error: {r.error}")
            elif not found:
                # Show truncated response so the user can see what the model said
                snippet = response.replace("\n", " ")[:80]
                print(f"     answered: {snippet!r}")

            results.append(NeedleResult(
                tier_k=tier,
                target_tokens=target_tokens,
                actual_tokens=r.prompt_tokens or 0,
                depth_pct=depth_pct,
                needle=passcode,
                found=found,
                response=response,
                ttft_ms=r.ttft_ms,
                total_ms=r.total_ms,
                prefill_tps=prefill_tps,
                ok=r.ok,
                error=r.error,
            ))

    save_records(run_id, all_records)
    save_samples(run_id, mon.samples)
    finish_run(run_id, {
        "records": summarize_records(all_records),
        "system":  mon.summary(),
        "tiers": [
            {
                "tier_k": x.tier_k, "target_tokens": x.target_tokens,
                "actual_tokens": x.actual_tokens, "depth_pct": x.depth_pct,
                "needle": x.needle, "found": x.found,
                "ttft_ms": x.ttft_ms, "prefill_tps": x.prefill_tps,
                "ok": x.ok, "error": x.error,
            }
            for x in results
        ],
        "host": host_summary(),
    })

    found_count = sum(1 for x in results if x.found)
    error_count = sum(1 for x in results if not x.ok)
    largest_pass = max((x.target_tokens for x in results if x.found), default=0)
    largest_attempted = max((x.target_tokens for x in results), default=0)

    print()
    print("=" * 78)
    print(f"  NIAH retrieval         : {found_count}/{len(results)} tiers")
    if largest_pass:
        print(f"  Largest passing tier   : {largest_pass:,} tokens "
              f"({largest_pass // 1024}K)")
    if error_count:
        print(f"  Tiers with errors      : {error_count}")
    if max_ctx and largest_attempted < max_ctx:
        print(f"  (model claims {max_ctx:,} tokens; we only tested up to "
              f"{largest_attempted:,})")
    print("=" * 78)

    print_run_banner(banner)
    return 0
