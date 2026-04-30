"""Reusable workload presets used by the `task` benchmark.

A workload is just (system_prompt, user_prompt, suggested_max_tokens). Files
referenced live under `files/prompts/`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "files" / "prompts"

SIZE_FILES = {
    "short":  "sample_short.txt",    # ~350 chars  / ~85 tokens
    "medium": "sample_medium.txt",   # ~2.6KB     / ~650 tokens
    "large":  "sample_large.txt",    # ~6.4KB     / ~1600 tokens
    "xlarge": "sample_xlarge.txt",   # ~19KB      / ~4750 tokens
}


@dataclass
class Workload:
    name: str
    description: str
    system: str | None
    user: str
    max_tokens: int


def _read(name: str) -> str:
    p = PROMPTS_DIR / name
    if not p.exists():
        return f"[missing file: {p}]"
    return p.read_text(encoding="utf-8")


def workload_simple() -> Workload:
    return Workload(
        name="simple",
        description="One-shot factual Q&A — short prompt, short answer.",
        system="You are a concise assistant. Answer in one or two sentences.",
        user="What is the capital of France, and what river runs through it?",
        max_tokens=80,
    )


def workload_chat() -> Workload:
    return Workload(
        name="chat",
        description="Open-ended chat reply — medium length.",
        system="You are a friendly assistant.",
        user="I'm planning a 3-day trip to Tokyo. Give me a rough itinerary.",
        max_tokens=400,
    )


def workload_long_output() -> Workload:
    return Workload(
        name="long-output",
        description="Force a long generation to stress decode throughput.",
        system=None,
        user=(
            "Write a long, continuous monologue about the history of computing. "
            "Do not stop early."
        ),
        max_tokens=1024,
    )


def workload_coding_function() -> Workload:
    return Workload(
        name="code-function",
        description="Write a single non-trivial function in Python.",
        system="You are a senior Python engineer. Output only code in a fenced block.",
        user=(
            "Write a function `merge_intervals(intervals: list[tuple[int, int]]) "
            "-> list[tuple[int, int]]` that merges overlapping intervals. "
            "Include a short docstring and 3 doctest examples."
        ),
        max_tokens=600,
    )


def workload_coding_refactor() -> Workload:
    return Workload(
        name="code-refactor",
        description="Refactor a sample file (read from disk).",
        system="You are a senior Python engineer.",
        user=(
            "Refactor the following Python code for readability. Keep behavior identical. "
            "Output only the refactored code in a fenced block.\n\n"
            "```python\n" + _read("sample_code.py") + "\n```"
        ),
        max_tokens=900,
    )


def workload_summarize_file(filename: str = "sample_medium.txt") -> Workload:
    text = _read(filename)
    return Workload(
        name=f"summarize:{filename}",
        description="Summarize an article loaded from a file.",
        system="You are a precise summarizer. Output 5 bullet points then a 2-sentence TL;DR.",
        user=f"Summarize the following article.\n\n---\n{text}\n---",
        max_tokens=500,
    )


def workload_qa_file(filename: str = "sample_medium.txt") -> Workload:
    text = _read(filename)
    return Workload(
        name=f"qa:{filename}",
        description="Answer a question grounded in a file's contents.",
        system="Answer the question using only the provided context. Cite the relevant sentence.",
        user=(
            f"Context:\n---\n{text}\n---\n\n"
            "Question: What is the central claim of the article, and what evidence is offered for it?"
        ),
        max_tokens=350,
    )


def workload_compare_contrast() -> Workload:
    return Workload(
        name="compare-contrast",
        description="Open-ended compare/contrast essay — Star Trek vs Star Wars.",
        system="You are a thoughtful writer. Use clear structure with sections.",
        user=(
            "Compare and contrast Star Trek and Star Wars. Cover at minimum: "
            "tone and worldview, technology vs mysticism, episodic vs serialized "
            "storytelling, the role of the Federation/Empire, and what each "
            "franchise says about the future. Aim for ~500 words."
        ),
        max_tokens=800,
    )


def workload_extraction() -> Workload:
    return Workload(
        name="extraction",
        description="Structured JSON extraction from prose.",
        system="Output ONLY valid JSON. No prose.",
        user=(
            "Extract people, organizations, and dates from this text as JSON with keys "
            "`people`, `orgs`, `dates`.\n\n"
            "Text: On March 14, 2024, Dr. Alice Chen of Acme Robotics met with senator Bob Lee "
            "to discuss the OpenLab initiative announced by MIT in January."
        ),
        max_tokens=200,
    )


REGISTRY: dict[str, callable] = {
    "simple": workload_simple,
    "chat": workload_chat,
    "long-output": workload_long_output,
    "code-function": workload_coding_function,
    "code-refactor": workload_coding_refactor,
    "summarize": workload_summarize_file,
    "qa": workload_qa_file,
    "extraction": workload_extraction,
    "compare-contrast": workload_compare_contrast,
}


def list_workloads() -> list[tuple[str, str]]:
    return [(name, fn().description) for name, fn in REGISTRY.items()]


def resolve_file(file: str | None, size: str | None) -> str | None:
    """Resolve --file / size flag to a real filename under files/prompts/."""
    if file:
        return file
    if size:
        if size not in SIZE_FILES:
            raise ValueError(f"unknown size: {size}. options: {list(SIZE_FILES)}")
        return SIZE_FILES[size]
    return None


def get_workload(name: str, *, file: str | None = None, size: str | None = None) -> Workload:
    if name not in REGISTRY:
        raise ValueError(f"unknown workload: {name}. options: {list(REGISTRY)}")
    fn = REGISTRY[name]
    resolved = resolve_file(file, size)
    if name in ("summarize", "qa") and resolved:
        return fn(resolved)
    return fn()
