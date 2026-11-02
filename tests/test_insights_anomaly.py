from datetime import date

import pandas as pd

from src import transactions as tx
from src.anomaly_detection import detect_isolation_forest, detect_rule_based
from src.insights import generate_insights
from src.synthetic_data import generate_transactions
from src.ui import fmt_inr

TODAY = date(2026, 10, 11)


def _load(conn, rows):
    tx.bulk_add(conn, rows, TODAY)
    return tx.load_dataframe(conn)


def e(d, cat, amt, desc="x", **kw):
    return dict(transaction_date=d, category=cat, amount=amt, description=desc, **kw)


def test_empty_and_sparse_data_say_not_enough_data(conn):
    assert generate_insights(tx.load_dataframe(conn), TODAY)[0].title == "No data yet"
    df = _load(conn, [e("2026-10-10", "Food", 100)])
    titles = [i.title for i in generate_insights(df, TODAY)]
    assert "Not enough data yet" in titles
    assert not any("higher" in t or "Larger" in t for t in titles)  # no pretend-personalisation


def test_category_increase_uses_comparable_period(conn):
    rows = [e(f"2026-09-{d:02d}", "Food", 100) for d in (1, 2, 3, 4, 5, 6)]           # 600 in Sep 1-6
    rows += [e("2026-09-25", "Food", 5000)]                                            # outside Sep 1-11 -> ignored
    rows += [e(f"2026-10-{d:02d}", "Food", 200) for d in (1, 2, 3, 4, 5, 6)]           # 1200 in Oct 1-11
    df = _load(conn, rows)
    msgs = [i for i in generate_insights(df, TODAY) if i.kind == "category_increase"]
    assert len(msgs) == 1 and "₹1,200" in msgs[0].message and "₹600" in msgs[0].message
    assert "same 11 days" in msgs[0].rule


def test_no_previous_month_is_explained(conn):
    df = _load(conn, [e(f"2026-10-{d:02d}", "Food", 100) for d in range(1, 11)])
    kinds = [(i.kind, i.title) for i in generate_insights(df, TODAY)]
    assert ("data", "Month comparison: not enough data yet") in kinds


def test_wasteful_and_reminder(conn):
    df = _load(conn, [e("2026-10-10", "Food", 400, wasteful_label="Yes"), e("2026-10-10", "Food", 600, wasteful_label="No")])
    out = {i.kind: i for i in generate_insights(df, TODAY)}
    assert "₹400" in out["wasteful"].message and "40.0%" in out["wasteful"].message
    assert "reminder" in out                                     # nothing dated 2026-10-11
    assert "reminder" not in {i.kind for i in generate_insights(df, TODAY, reminder=False)}


def test_frequent_small_purchases(conn):
    df = _load(conn, [e(f"2026-10-{d:02d}", "Food", 40) for d in (6, 7, 8, 9, 10, 11)] +
               [e("2026-09-01", "Food", 100)] * 10)
    assert any(i.kind == "frequent_small" for i in generate_insights(df, TODAY))


def test_rule_based_flags_large_expense_with_past_data_only(conn):
    rows = [e(f"2026-09-{d:02d}", "Food", 100 + d) for d in range(1, 13)] + [e("2026-09-20", "Food", 5000)]
    df = _load(conn, rows)
    res = detect_rule_based(df).merge(df, on="id")
    assert res.loc[res["amount_paise"] == 500000, "is_anomaly"].item() is True
    assert not res.loc[res["amount_paise"] < 500000, "is_anomaly"].any()
    early = res.sort_values("id").head(8)
    assert (early["reason"] == "not enough history").all()   # first 8 entries can never be flagged


def test_large_expense_shown_in_insights(conn):
    rows = [e(f"2026-09-{d:02d}", "Food", 100 + d) for d in range(1, 13)] + [e("2026-10-10", "Food", 5000, "Wedding feast")]
    df = _load(conn, rows)
    big = [i for i in generate_insights(df, TODAY) if i.kind == "large_expense"]
    assert len(big) == 1 and "Wedding feast" in big[0].message and "not a judgement" in big[0].rule


def test_isolation_forest_needs_enough_data_and_flags_only_high(conn):
    small = _load(conn, [e("2026-10-01", "Food", 100)] * 5)
    assert not detect_isolation_forest(small)["is_anomaly"].any()
    conn.execute("DELETE FROM transactions")
    rows = generate_transactions(end=TODAY, days=200, seed=3, anomaly_rate=0.04)
    df = _load(conn, rows)
    res = detect_isolation_forest(df).merge(df, on="id")
    flagged = res[res["is_anomaly"]]
    med = df.groupby("category")["amount_paise"].median()
    assert len(flagged) > 0
    assert all(r.amount_paise > med[r.category] for r in flagged.itertuples())


def test_fmt_inr_indian_grouping():
    assert fmt_inr(0) == "₹0"
    assert fmt_inr(25000) == "₹250"
    assert fmt_inr(12345678) == "₹1,23,456.78"
    assert fmt_inr(123456700) == "₹12,34,567"
    assert fmt_inr(-99900) == "-₹999"
    assert fmt_inr(100000) == "₹1,000"


def test_synthetic_generator_is_deterministic_and_valid(conn):
    a = generate_transactions(end=TODAY, seed=1)
    assert a == generate_transactions(end=TODAY, seed=1)
    assert a != generate_transactions(end=TODAY, seed=2)
    assert tx.bulk_add(conn, a, TODAY) == len(a)           # every generated row passes validation
    df = tx.load_dataframe(conn)
    assert {"income", "refund", "expense"} <= set(df["transaction_type"])
    assert pd.Timestamp(TODAY) >= df["transaction_date"].max()
