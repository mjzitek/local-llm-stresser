#!/usr/bin/env bash
# One-shot installer: creates ./env, installs the package + CLI.
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d env ]; then
  python3 -m venv env
fi

# shellcheck disable=SC1091
source env/bin/activate
pip install --upgrade pip >/dev/null
pip install -e .

if [ ! -f .env ]; then
  cp .env.example .env
  echo "[install] wrote .env from .env.example — edit it to point at your runtime"
fi

# Generate vision test images (idempotent; safe to re-run)
python scripts/generate_test_images.py

echo
echo "Done. Activate with:  source env/bin/activate"
echo "Then try:             stresser runtimes"
echo "                      stresser models --runtime ollama"
echo "                      stresser workloads"
echo "                      stresser task simple"
