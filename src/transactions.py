"""Transaction CRUD, filtering and export helpers."""
from __future__ import annotations

import sqlite3
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .validation import validate_transaction

COLUMNS = [
    "id", "transaction_date", "amount_paise", "transaction_type", "category", "description",
    "wasteful_label", "need_want", "payment_method", "context", "notes",
    "predicted_category", "predicted_confidence", "created_at", "updated_at",
]
_WRITE_FIELDS = [c for c in COLUMNS if c not in ("id", "created_at", "updated_at")]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ------------------------------------------------------------------ write
def add_transaction(conn: sqlite3.Connection, data: Mapping[str, Any], today: date | None = None) -> int:
    clean = validate_transaction(data, today)
    now = _now()
    cols = ", ".join(_WRITE_FIELDS + ["created_at", "updated_at"])
    marks = ", ".join(f":{c}" for c in _WRITE_FIELDS + ["created_at", "updated_at"])
    cur = conn.execute(f"INSERT INTO transactions ({cols}) VALUES ({marks})",
                       {**clean, "created_at": now, "updated_at": now})
    return int(cur.lastrowid)


def update_transaction(conn: sqlite3.Connection, tx_id: int, data: Mapping[str, Any],
                       today: date | None = None) -> None:
    if get_transaction(conn, tx_id) is None:
        raise KeyError(f"Transaction {tx_id} does not exist.")
    clean = validate_transaction(data, today)
    if "predicted_category" not in data:      # editing must not wipe ML bookkeeping
        clean.pop("predicted_category")
        clean.pop("predicted_confidence")
    sets = ", ".join(f"{k} = :{k}" for k in clean)  # keys are fixed, validated names
    conn.execute(f"UPDATE transactions SET {sets}, updated_at = :updated_at WHERE id = :id",
                 {**clean, "updated_at": _now(), "id": int(tx_id)})


def delete_transaction(conn: sqlite3.Connection, tx_id: int) -> bool:
    cur = conn.execute("DELETE FROM transactions WHERE id = ?", (int(tx_id),))
    return cur.rowcount > 0


def delete_all(conn: sqlite3.Connection) -> int:
    cur = conn.execute("DELETE FROM transactions")
    conn.execute("DELETE FROM sqlite_sequence WHERE name = 'transactions'")
    return cur.rowcount


def bulk_add(conn: sqlite3.Connection, rows: Iterable[Mapping[str, Any]], today: date | None = None) -> int:
    """Validate and insert many rows atomically (all-or-nothing)."""
    n = 0
    for row in rows:
        add_transaction(conn, row, today)
        n += 1
    return n


# ------------------------------------------------------------------- read
def get_transaction(conn: sqlite3.Connection, tx_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM transactions WHERE id = ?", (int(tx_id),)).fetchone()
    return dict(row) if row else None


def load_dataframe(conn: sqlite3.Connection) -> pd.DataFrame:
    """All transactions as a DataFrame. ``amount_paise`` is exact; ``amount`` is rupees for display."""
    rows = conn.execute("SELECT * FROM transactions ORDER BY transaction_date, id").fetchall()
    df = pd.DataFrame([dict(r) for r in rows], columns=COLUMNS)
    return normalise_frame(df)


def normalise_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["id"] = df["id"].astype("int64")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["amount_paise"] = df["amount_paise"].astype("int64")
    df["amount"] = df["amount_paise"] / 100.0
    return df


def filter_transactions(
    df: pd.DataFrame,
    start: date | None = None,
    end: date | None = None,
    categories: Sequence[str] | None = None,
    types: Sequence[str] | None = None,
    wasteful: Sequence[str] | None = None,
    payment_methods: Sequence[str] | None = None,
    search: str | None = None,
    sort: str = "newest",
) -> pd.DataFrame:
    """Filter + sort. ``sort``: newest | oldest | largest | smallest."""
    out = df
    if start is not None:
        out = out[out["transaction_date"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["transaction_date"] <= pd.Timestamp(end)]
    if categories:
        out = out[out["category"].isin(categories)]
    if types:
        out = out[out["transaction_type"].isin(types)]
    if wasteful:
        out = out[out["wasteful_label"].isin(wasteful)]
    if payment_methods:
        out = out[out["payment_method"].isin(payment_methods)]
    if search and search.strip():
        needle = search.strip()
        mask = (out["description"].fillna("").str.contains(needle, case=False, regex=False)
                | out["notes"].fillna("").str.contains(needle, case=False, regex=False))
        out = out[mask]
    order = {
        "newest": (["transaction_date", "id"], [False, False]),
        "oldest": (["transaction_date", "id"], [True, True]),
        "largest": (["amount_paise", "id"], [False, False]),
        "smallest": (["amount_paise", "id"], [True, True]),
    }
    if sort not in order:
        raise ValueError(f"Unknown sort '{sort}'.")
    by, asc = order[sort]
    return out.sort_values(by, ascending=asc, kind="stable")


# ----------------------------------------------------------------- export
def csv_safe(value: Any) -> Any:
    """Neutralise spreadsheet formula injection (cells starting with = + - @)."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def to_export_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """User-friendly, spreadsheet-safe export of their own records."""
    out = pd.DataFrame({
        "date": df["transaction_date"].dt.strftime("%Y-%m-%d"),
        "type": df["transaction_type"],
        "category": df["category"],
        "amount_inr": (df["amount_paise"] / 100).round(2),
        "description": df["description"],
        "wasteful": df["wasteful_label"],
        "need_want": df["need_want"],
        "payment_method": df["payment_method"],
        "context": df["context"],
        "notes": df["notes"],
    })
    for col in out.columns:
        if out[col].dtype == object or str(out[col].dtype).startswith("str"):
            out[col] = out[col].map(csv_safe)
    return out
