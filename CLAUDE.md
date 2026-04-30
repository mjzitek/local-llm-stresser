# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Stress-test harness for local LLM runtimes running on this machine — Ollama, llama.cpp (`llama-server`), and LM Studio. The goal is to push a local inference server with controlled load (concurrency, prompt size, output length, mixed model sizes) and capture throughput / latency / resource metrics so the host computer's limits can be characterized.

This is a brand-new empty repo — no code yet. The first implementation should establish the architecture below.

## Architecture (to be implemented)

The harness should be backend-agnostic via a thin adapter layer, since the three target runtimes all expose OpenAI-compatible HTTP APIs but with quirks:

- **Ollama** — `http://localhost:11434` (native `/api/generate`, `/api/chat`) and `/v1/chat/completions` (OpenAI-compat). Model pull/list via `/api/tags`.
- **llama.cpp** — `llama-server` exposes `/completion` (native) and `/v1/chat/completions`. Single model per process.
- **LM Studio** — OpenAI-compatible at `http://localhost:1234/v1`. Models loaded via the LM Studio UI or `lms` CLI.

Each adapter should normalize: model listing, streaming chat completion, token usage (when available), and error shape.

Core components expected:
- `app/backend/services/runner.py` — orchestrates concurrent workloads (asyncio + httpx, not threads)
- `app/backend/services/workloads.py` — workload definitions: short-prompt high-QPS, long-context, long-output, mixed, code-gen, RAG-shaped
- `app/backend/services/metrics.py` — captures TTFT, inter-token latency, tokens/sec, total latency, error rate; system metrics (CPU, GPU, RAM, VRAM) sampled in parallel
- `app/backend/adapters/{ollama,llamacpp,lmstudio}.py` — backend adapters
- `app/backend/api/` — FastAPI control surface for kicking off runs and viewing results
- `app/frontend/` — Next.js dashboard for live metrics + historical run comparison
- `data/` — SQLite for run history
- `files/prompts/` — prompt corpora used by workloads

Metrics worth capturing per request: queue wait, time-to-first-token, time-to-last-token, prompt tokens, completion tokens, tokens/sec (decode), success/error. System-side: GPU utilization + VRAM (Apple Metal via `powermetrics` or `ioreg`; NVIDIA via `nvidia-smi`), CPU, RAM, thermal state.

## User conventions (from global CLAUDE.md, applied here)

- venv directory is `env/` (not `.venv` or `venv`).
- Avoid common dev ports (3000, 8000, 8080, 8765). Pick something like 8421 (backend) / 3421 (frontend).
- Do not assume a process from `ps aux` belongs to this project — many local LLM servers may be running; verify by full path/cwd before acting on a PID.
- Centralize LLM-calling code (DRY) — the adapter layer is the single place HTTP calls to runtimes happen.
- Light mode by default in any UI.
- Use `gpt-5` family via the OpenAI **responses** API if any cloud LLM is needed (e.g., for judging output quality); never `gpt-4*`. No `temperature` param on gpt-5 models.

## Commands

To be filled in once the stack is chosen. Expected once scaffolded:

```
./dev_start.sh                 # iTerm tabs: backend (FastAPI) + frontend (Next.js)
python -m app.backend.api.main # backend only
```
