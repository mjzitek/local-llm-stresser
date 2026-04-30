# local-llm-stresser

Stress-test harness for local LLM runtimes — **Ollama**, **llama.cpp** (`llama-server`), and **LM Studio**. All three speak the OpenAI-compatible HTTP API, so the only thing that changes between runtimes is the base URL.

One CLI, several workload shapes (simple Q&A, coding, file summarization, file-grounded Q&A, JSON extraction, long-output decode), plus dedicated decode/prefill/concurrency benchmarks. Every run is recorded to a local SQLite file at `data/runs.db`.

## Quick start

```bash
git clone https://github.com/<your-username>/local-llm-stresser
cd local-llm-stresser
./install.sh
source env/bin/activate
stresser           # interactive wizard: pick test → pick model → quick or detailed
```

The wizard auto-detects which runtimes are running on `localhost`, lists their loaded models, and walks you through picking a workload. Hit Enter to accept defaults at every prompt.

## Install

```bash
git clone <this-repo> local-llm-stresser
cd local-llm-stresser
./install.sh
source env/bin/activate
```

That creates `env/`, installs the package in editable mode, and puts a `stresser` command on your PATH.

## What's running on this box?

```bash
stresser              # shows detected runtimes + models, then prints help
stresser detect       # full model lists for every runtime that responded
```

On startup it probes the default ports for Ollama (11434), llama.cpp (8080), and LM Studio (1234) in parallel, lists which are alive, and shows the models each one is serving. Benchmark commands run the same probe and use it to pick a sane default if you didn't specify `--runtime` or `--model`.

## Pick a runtime explicitly

```bash
stresser runtimes        # show presets and default URLs
stresser models --runtime ollama
stresser models --runtime lmstudio --base-url http://localhost:1234/v1
```

You can either pass `--runtime` / `--model` on every call, or set defaults in `.env`:

```
LLM_RUNTIME=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.1:8b
```

Runtime defaults:

| runtime    | default base URL                  |
|------------|-----------------------------------|
| `ollama`   | `http://localhost:11434/v1`       |
| `llamacpp` | `http://localhost:8080/v1`        |
| `lmstudio` | `http://localhost:1234/v1`        |
| `custom`   | requires `--base-url`             |

## Run tests

```bash
# show available workloads
stresser workloads

# single-shot tasks (one prompt, run N times, optionally concurrent)
stresser task simple --n 10
stresser task chat --n 5 --concurrency 2
stresser task code-function --n 3
stresser task code-refactor --n 3
stresser task summarize --size medium --n 5
stresser task summarize --size xlarge --show-outputs --n 2
stresser task qa --size large --n 5
stresser task extraction --n 10
stresser task long-output --max-tokens 2048

# focused performance benchmarks
stresser decode --n 5 --max-tokens 1024              # pure decode tok/s
stresser prefill --sizes 512,2048,8192,16384         # prompt-processing speed
stresser concurrency --levels 1,2,4,8,16             # throughput vs in-flight
```

For `summarize` and `qa`, pick a text size with `--size {short,medium,large,xlarge}` or point at your own file with `--file your_file.txt` (drop it under `files/prompts/`).

## Workloads

| name           | what it does                                                       |
|----------------|--------------------------------------------------------------------|
| `simple`       | one-shot factual Q&A, short out                                    |
| `chat`         | open-ended ~400-token reply                                        |
| `long-output`  | force a long monologue — stresses decode                           |
| `code-function`| ask the model to write a function with doctests                    |
| `code-refactor`| refactor `files/prompts/sample_code.py`                            |
| `summarize`    | summarize a file from `files/prompts/`                             |
| `qa`           | answer a question grounded in a file's contents                    |
| `extraction`   | structured JSON extraction (people / orgs / dates)                 |

## What's measured

Per request: TTFT, total latency, prompt/completion tokens (when the runtime returns usage), decode tok/s (computed from time between first and last streamed token).

Per run: CPU%, RAM, plus — if `nvidia-smi` is on PATH — GPU util / VRAM / temp / power, sampled at 1 Hz.

Everything is appended to `data/runs.db` (tables: `runs`, `requests`, `samples`).

## Layout

```
app/
  cli.py             # `stresser` entrypoint
  core/              # client, sysmon, recorder, report, runtimes, config
  benchmarks/        # decode, prefill, concurrency, task, workloads
files/prompts/       # text/code corpora used by file-based workloads
data/runs.db         # SQLite history
```
