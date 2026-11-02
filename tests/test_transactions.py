from datetime import date

import pytest

from src import transactions as tx
from src.validation import ValidationError

TODAY = date(2026, 10, 11)


def row(**kw):
    d = {"transaction_date": "2026-10-10", "amount": "250", "category": "Food", "description": "Lunch with friends"}
    d.update(kw)
    return d


def test_add_and_list(conn):
    tx_id = tx.add_transaction(conn, row(), TODAY)
    df = tx.load_dataframe(conn)
    assert len(df) == 1 and df.loc[0, "id"] == tx_id
    assert df.loc[0, "amount_paise"] == 25000 and df.loc[0, "amount"] == 250.0


def test_empty_database_gives_typed_empty_frame(conn):
    df = tx.load_dataframe(conn)
    assert df.empty and list(df.columns[:3]) == ["id", "transaction_date", "amount_paise"]
    assert str(df["transaction_date"].dtype).startswith("datetime64")


def test_invalid_row_is_not_saved(conn):
    with pytest.raises(ValidationError):
        tx.add_transaction(conn, row(amount="-5"), TODAY)
    assert tx.load_dataframe(conn).empty


def test_sql_injection_text_stored_literally(conn):
    nasty = "x'); DROP TABLE transactions;--"
    tx.add_transaction(conn, row(description=nasty), TODAY)
    assert tx.load_dataframe(conn).loc[0, "description"] == nasty


def test_edit_changes_reports(conn):
    i = tx.add_transaction(conn, row(), TODAY)
    tx.update_transaction(conn, i, row(amount="300", category="Travel", description="Cab"), TODAY)
    r = tx.get_transaction(conn, i)
    assert r["amount_paise"] == 30000 and r["category"] == "Travel" and r["description"] == "Cab"


def test_edit_keeps_ml_bookkeeping_unless_given(conn):
    i = tx.add_transaction(conn, row(predicted_category="Food", predicted_confidence=0.9), TODAY)
    tx.update_transaction(conn, i, row(amount="260"), TODAY)
    assert tx.get_transaction(conn, i)["predicted_category"] == "Food"


def test_edit_missing_row_raises(conn):
    with pytest.raises(KeyError):
        tx.update_transaction(conn, 999, row(), TODAY)


def test_delete(conn):
    i = tx.add_transaction(conn, row(), TODAY)
    assert tx.delete_transaction(conn, i) is True
    assert tx.delete_transaction(conn, i) is False
    assert tx.load_dataframe(conn).empty


def test_bulk_add_is_atomic(tmp_path):
    from src import database as db
    path = tmp_path / "a.db"
    with pytest.raises(ValidationError):
        with db.connect(path) as c:
            tx.bulk_add(c, [row(), row(amount="oops")], TODAY)
    with db.connect(path) as c:
        assert tx.load_dataframe(c).empty


def test_filters_and_sort(conn):
    tx.add_transaction(conn, row(transaction_date="2026-10-01", amount="100", category="Food", description="Chai"), TODAY)
    tx.add_transaction(conn, row(transaction_date="2026-10-05", amount="900", category="Travel", description="Cab to airport",
                                 wasteful_label="Yes"), TODAY)
    tx.add_transaction(conn, row(transaction_date="2026-10-09", amount="50", category="Food", description="Samosa"), TODAY)
    df = tx.load_dataframe(conn)
    assert list(tx.filter_transactions(df, categories=["Food"])["description"]) == ["Samosa", "Chai"]       # newest first
    assert list(tx.filter_transactions(df, sort="largest")["amount_paise"]) == [90000, 10000, 5000]
    assert len(tx.filter_transactions(df, start=date(2026, 10, 2), end=date(2026, 10, 8))) == 1
    assert list(tx.filter_transactions(df, wasteful=["Yes"])["category"]) == ["Travel"]
    assert len(tx.filter_transactions(df, search="AIRPORT")) == 1
    assert len(tx.filter_transactions(df, search="100%")) == 0  # regex characters are literal
    with pytest.raises(ValueError):
        tx.filter_transactions(df, sort="bogus")


def test_export_neutralises_formula_injection(conn):
    tx.add_transaction(conn, row(description="=HYPERLINK(\"http://evil\")"), TODAY)
    out = tx.to_export_dataframe(tx.load_dataframe(conn))
    assert out.loc[0, "description"].startswith("'=")
    assert list(out.columns)[:4] == ["date", "type", "category", "amount_inr"]


def test_delete_all(conn):
    tx.bulk_add(conn, [row(), row()], TODAY)
    assert tx.delete_all(conn) == 2 and tx.load_dataframe(conn).empty
