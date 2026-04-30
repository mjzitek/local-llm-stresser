#!/usr/bin/env python3
"""Top-level entrypoint so you can run `python stresser.py ...` without pip install."""
from app.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
