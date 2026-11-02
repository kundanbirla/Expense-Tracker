"""SQLite connection + schema.

Single-user local app: authentication is deliberately deferred (see README).
All SQL uses bound parameters, so descriptions can never inject SQL.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_date     TEXT    NOT NULL,                       -- ISO date YYYY-MM-DD
    amount_paise         INTEGER NOT NULL CHECK (amount_paise > 0),
    transaction_type     TEXT    NOT NULL CHECK (transaction_type IN ('expense','income','refund')),
    category             TEXT    NOT NULL,
    description          TEXT    NOT NULL DEFAULT '',
    wasteful_label       TEXT    CHECK (wasteful_label IN ('Yes','No','Not sure') OR wasteful_label IS NULL),
    need_want            TEXT    CHECK (need_want IN ('Need','Want','Both') OR need_want IS NULL),
    payment_method       TEXT,
    context              TEXT,
    notes                TEXT    NOT NULL DEFAULT '',
    predicted_category   TEXT,                                   -- what the ML model suggested (if used)
    predicted_confidence REAL,
    created_at           TEXT    NOT NULL,
    updated_at           TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tx_date     ON transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_tx_category ON transactions(category);
"""


@contextmanager
def connect(path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open a connection (creating the DB/schema if needed); commit on success."""
    db_file = Path(path) if path else config.db_path()
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
