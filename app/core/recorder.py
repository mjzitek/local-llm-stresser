"""SQLite recorder for run history."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from app.core.client import RequestRecord
from app.core.sysmon import Sample

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "runs.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    finished_at REAL,
    workload TEXT NOT NULL,
    runtime TEXT,
    base_url TEXT,
    model TEXT,
    config_json TEXT,
    summary_json TEXT
);
CREATE TABLE IF NOT EXISTS requests (
    run_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    record_json TEXT NOT NULL,
    PRIMARY KEY (run_id, idx)
);
CREATE TABLE IF NOT EXISTS samples (
    run_id TEXT NOT NULL,
    t REAL NOT NULL,
    sample_json TEXT NOT NULL
);
"""


@contextmanager
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    try:
        c.executescript(SCHEMA)
        yield c
        c.commit()
    finally:
        c.close()


def new_run(workload: str, runtime: str, base_url: str, model: str, config: dict) -> str:
    run_id = uuid.uuid4().hex[:12]
    with _conn() as c:
        c.execute(
            "INSERT INTO runs(run_id, started_at, workload, runtime, base_url, model, config_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (run_id, time.time(), workload, runtime, base_url, model, json.dumps(config)),
        )
    return run_id


def save_records(run_id: str, records: list[RequestRecord]) -> None:
    with _conn() as c:
        c.executemany(
            "INSERT OR REPLACE INTO requests(run_id, idx, record_json) VALUES (?,?,?)",
            [(run_id, i, json.dumps(asdict(r))) for i, r in enumerate(records)],
        )


def save_samples(run_id: str, samples: list[Sample]) -> None:
    with _conn() as c:
        c.executemany(
            "INSERT INTO samples(run_id, t, sample_json) VALUES (?,?,?)",
            [(run_id, s.t, json.dumps(asdict(s))) for s in samples],
        )


def finish_run(run_id: str, summary: dict) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE runs SET finished_at=?, summary_json=? WHERE run_id=?",
            (time.time(), json.dumps(summary), run_id),
        )
