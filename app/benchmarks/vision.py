"""Vision / OCR benchmark for multimodal local models.

For each bundled test image we ask the model two questions:
  1. "Describe this image" — perception test
  2. "What text appears in the image?" — OCR test (when the image has text)

Per request, we record TTFT, decode TPS, total latency, *and* a simple
correctness signal: did the model's response mention the expected text
or key facts. The correctness check is deliberately lenient (substring +
case-insensitive) — it's a sanity test, not a leaderboard score.

Vision-capable Ollama models we know work:
  - moondream:latest          (small, fast, basic captioning)
  - gemma4:e4b / gemma4:latest (multimodal, good at OCR)
  - llava family
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.client import LLMClient
from app.core.config import Config
from app.core.host_info import host_summary
from app.core.model_info import model_footprint
from app.core.recorder import finish_run, new_run, save_records, save_samples
from app.core.report import print_run_banner, summarize_records
from app.core.sysmon import SysMonitor

IMAGES_DIR = Path(__file__).resolve().parents[2] / "files" / "images"


@dataclass
class VisionResult:
    image: str
    task_kind: str
    question: str
    expected_hits: list[str]
    response: str
    hits: int
    total_expected: int
    ttft_ms: float | None
    total_ms: float | None
    decode_tps: float | None
    ok: bool
    error: str | None


def _load_manifest() -> dict:
    p = IMAGES_DIR / "manifest.json"
    if not p.exists():
        raise SystemExit(
            f"missing {p} — run `python scripts/generate_test_images.py` first"
        )
    return json.loads(p.read_text())


def _score(response: str, expected: list[str]) -> tuple[int, int]:
    """Lenient: case-insensitive substring matches."""
    if not expected:
        return 0, 0
    lower = response.lower()
    hits = sum(1 for e in expected if e.lower() in lower)
    return hits, len(expected)


def _describe_question(meta: dict) -> str:
    return "Describe this image in 1-2 sentences."


def _ocr_question(meta: dict) -> str:
    return ("What text appears in this image? List every readable word or "
            "number you can see. Be exhaustive.")


def _scene_question(meta: dict) -> str:
    return "Describe the scene. What objects do you see and where are they?"


async def _ask(client: LLMClient, image_path: Path, question: str,
               max_tokens: int = 250):
    return await client.stream_chat(
        question,
        system="You are a vision assistant. Be precise and concise.",
        max_tokens=max_tokens,
        capture_output=True,
        images=[str(image_path)],
    )


async def run(cfg: Config, *, only: list[str] | None = None) -> int:
    manifest = _load_manifest()
    images = sorted(manifest.keys())
    if only:
        images = [n for n in images if any(o in n for o in only)]

    fp = model_footprint(cfg.model)
    banner = {
        "MODEL":   f"{cfg.model}" + (f"   [{fp}]" if fp else ""),
        "RUNTIME": f"{cfg.runtime}  ({cfg.base_url})",
        "TEST":    f"vision benchmark ({len(images)} images, perception + OCR)",
    }
    run_id = new_run(
        "vision", cfg.runtime, cfg.base_url, cfg.model,
        {"images": images},
    )
    banner["RUN ID"] = run_id
    print_run_banner(banner)

    print(f"\n{'image':<16} {'task':<14} {'ttft_ms':>9} {'total_ms':>10} "
          f"{'decode_tps':>11} {'hits':>9}")
    print("-" * 80)

    results: list[VisionResult] = []
    all_records = []

    async with LLMClient(cfg) as client, SysMonitor(interval=2.0) as mon:
        # Quick warmup with no image to load the model
        await client.stream_chat("hi", max_tokens=1)

        for name in images:
            meta = manifest[name]
            path = IMAGES_DIR / name
            kind = meta.get("task", "describe")

            # Build the questions to ask for this image
            if kind == "ocr":
                questions = [("ocr", _ocr_question(meta), meta.get("expected_text", []))]
            elif kind == "code-ocr":
                questions = [
                    ("ocr", _ocr_question(meta), meta.get("expected_text", [])),
                    ("describe", "What does this code do? "
                                 "Reply in one sentence.", meta.get("key_facts", [])),
                ]
            elif kind == "chart":
                questions = [
                    ("ocr", _ocr_question(meta), meta.get("expected_text", [])),
                    ("describe", "Describe the chart. Include the title and "
                                 "the highest and lowest values.",
                     meta.get("key_facts", [])),
                ]
            elif kind == "scene":
                questions = [
                    ("describe", _scene_question(meta), meta.get("key_facts", [])),
                ]
            else:
                questions = [
                    ("describe", _describe_question(meta), meta.get("key_facts", [])),
                ]

            for q_kind, q_text, expected in questions:
                t0 = time.perf_counter()
                rec = await _ask(client, path, q_text)
                all_records.append(rec)
                response = (rec.output_text or "").strip()
                hits, total = _score(response, expected)

                hits_str = f"{hits}/{total}" if total else "n/a"
                print(
                    f"{name:<16} {q_kind:<14} "
                    f"{(rec.ttft_ms or 0):>9.0f} "
                    f"{(rec.total_ms or 0):>10.0f} "
                    f"{(rec.decode_tps or 0):>11.1f} "
                    f"{hits_str:>9}"
                )
                if response:
                    snippet = response.replace("\n", " ")[:90]
                    print(f"   ↳ {snippet}")
                if not rec.ok and rec.error:
                    print(f"   ↳ ERROR: {rec.error}")

                results.append(VisionResult(
                    image=name, task_kind=q_kind, question=q_text,
                    expected_hits=expected, response=response,
                    hits=hits, total_expected=total,
                    ttft_ms=rec.ttft_ms, total_ms=rec.total_ms,
                    decode_tps=rec.decode_tps,
                    ok=rec.ok, error=rec.error,
                ))

    save_records(run_id, all_records)
    save_samples(run_id, mon.samples)

    total_hits    = sum(r.hits for r in results)
    total_expected = sum(r.total_expected for r in results)
    success_rate = (total_hits / total_expected * 100) if total_expected else 0

    finish_run(run_id, {
        "records": summarize_records(all_records),
        "system":  mon.summary(),
        "vision": {
            "total_questions":  len(results),
            "total_hits":       total_hits,
            "total_expected":   total_expected,
            "success_rate_pct": success_rate,
            "items": [
                {"image": r.image, "task": r.task_kind,
                 "hits": r.hits, "expected": r.total_expected,
                 "response": r.response[:500]}
                for r in results
            ],
        },
        "host": host_summary(),
    })

    print()
    print("=" * 78)
    print(f"  Total questions   : {len(results)}")
    print(f"  Expected matches  : {total_expected}")
    print(f"  Actual matches    : {total_hits}  ({success_rate:.0f}%)")
    print("=" * 78)
    print_run_banner(banner)
    return 0
