"""Doc section 11.2 scenario: 20+ synthetic transactions over >= 2 months with totals checked by hand."""
from datetime import date

import pandas as pd
import pytest

from src import analytics as an
from src import transactions as tx

TODAY = date(2026, 10, 11)

# (date, type, category, amount ₹, description, wasteful)
FIXTURE = [
    # --- September 2026 ---
    ("2026-09-01", "income", "Salary", 40000, "Salary", None),
    ("2026-09-02", "expense", "Food", 250, "Lunch with friends", "No"),
    ("2026-09-03", "expense", "Travel", 80, "Bus fare", "No"),
    ("2026-09-05", "expense", "Bills", 8000, "Rent", "No"),
    ("2026-09-07", "expense", "Entertainment", 399, "Unplanned subscription", "Yes"),
    ("2026-09-09", "expense", "Food", 500, "Weekly groceries", "No"),
    ("2026-09-10", "expense", "Shopping", 1200, "Sneakers", "Yes"),
    ("2026-09-12", "refund", "Shopping", 400, "Sneakers partial refund", None),
    ("2026-09-15", "expense", "Food", 150, "Zomato", "Not sure"),
    ("2026-09-20", "expense", "Health", 300, "Medicines", "No"),
    ("2026-09-30", "expense", "Food", 100, "Month-end chai", None),
    # --- October 2026 ---
    ("2026-10-01", "income", "Salary", 40000, "Salary", None),
    ("2026-10-01", "income", "Freelance", 5000, "Logo project", None),
    ("2026-10-02", "expense", "Food", 300, "Dinner", "No"),
    ("2026-10-03", "expense", "Travel", 120, "Auto", "No"),
    ("2026-10-05", "expense", "Bills", 8000, "Rent", "No"),
    ("2026-10-06", "expense", "Food", 700, "Weekly groceries", "No"),
    ("2026-10-08", "expense", "Entertainment", 600, "Movie", "Yes"),
    ("2026-10-09", "expense", "Shopping", 2000, "Headphones", "Yes"),
    ("2026-10-10", "expense", "Food", 250, "Lunch with friends", "No"),
    ("2026-10-10", "expense", "Travel", 80, "Bus fare", "No"),
    ("2026-10-10", "expense", "Entertainment", 399, "Unplanned subscription", "Yes"),
    ("2026-10-11", "expense", "Household", 500, "Cleaning supplies", "No"),
]


@pytest.fixture
def df(conn):
    tx.bulk_add(conn, [dict(transaction_date=d, transaction_type=t, category=c, amount=a, description=x, wasteful_label=w)
                       for d, t, c, a, x, w in FIXTURE], TODAY)
    return tx.load_dataframe(conn)


SEP = (date(2026, 9, 1), date(2026, 9, 30))
OCT = (date(2026, 10, 1), date(2026, 10, 31))


def test_monthly_totals_match_hand_calculation(df):
    # Sep spend: 250+80+8000+399+500+1200-400+150+300+100 = 10579 ; Oct spend: 300+120+8000+700+600+2000+250+80+399+500 = 12949
    assert an.total_expenses(df, *SEP) == 1057900
    assert an.total_expenses(df, *OCT) == 1294900


def test_income_not_counted_as_expense_and_net_cash_flow(df):
    assert an.total_income(df, *SEP) == 4000000
    assert an.total_income(df, *OCT) == 4500000
    assert an.net_cash_flow(df, *OCT) == 4500000 - 1294900


def test_refund_reduces_its_category(df):
    cats = an.category_breakdown(df, *SEP).set_index("category")["total_paise"]
    assert cats["Shopping"] == 80000  # 1200 - 400
    assert an.total_expenses(df, date(2026, 9, 10), date(2026, 9, 12)) == 80000  # 1200 on the 10th, -400 on the 12th


def test_category_breakdown_and_shares(df):
    b = an.category_breakdown(df, *OCT)
    assert b.iloc[0]["category"] == "Bills" and b.iloc[0]["total_paise"] == 800000
    assert b["total_paise"].sum() == 1294900
    assert b["share_pct"].sum() == pytest.approx(100, abs=0.2)
    assert dict(zip(b["category"], b["total_paise"]))["Food"] == 125000  # 300+700+250


def test_daily_average_reports_days_used(df):
    avg, n = an.daily_average(df, *OCT)
    assert n == 31 and avg == pytest.approx(1294900 / 31)
    avg, n = an.daily_average(df, date(2026, 10, 10), date(2026, 10, 10))
    assert n == 1 and avg == 25000 + 8000 + 39900  # single-day period: 250 + 80 + 399 rupees


def test_wasteful_summary_uses_only_yes_labels(df):
    w = an.wasteful_summary(df, *OCT)
    assert w["total_paise"] == 60000 + 200000 + 39900 and w["count"] == 3
    assert w["share_pct"] == pytest.approx(299900 / 1294900 * 100, abs=0.06)
    s = an.wasteful_summary(df, *SEP)  # refund is NOT netted off the wasteful total; "Not sure" excluded
    assert s["total_paise"] == 39900 + 120000 and s["count"] == 2


def test_month_over_month(df):
    m = an.month_over_month(df, 2026, 10).set_index("category")
    assert m.loc["Food", "current_paise"] == 125000
    assert m.loc["Food", "previous_paise"] == 25000 + 50000 + 15000 + 10000  # 250+500+150+100 rupees
    assert m.loc["Food", "change_pct"] == pytest.approx((125000 - 100000) / 100000 * 100)
    assert m.loc["Household", "previous_paise"] == 0 and pd.isna(m.loc["Household", "change_pct"])
    assert m.loc["Health", "current_paise"] == 0 and m.loc["Health", "change_paise"] == -30000


def test_monthly_totals_table_fills_gaps(df):
    t = an.monthly_totals(df)
    assert list(t["month"]) == ["2026-09", "2026-10"]
    assert list(t["net_paise"]) == [4000000 - 1057900, 4500000 - 1294900]


def test_top_expenses(df):
    top = an.top_expenses(df, *OCT, n=2)
    assert list(top["amount_paise"]) == [800000, 200000]


def test_daily_series_zero_filled(df):
    s = an.daily_series(df, date(2026, 10, 1), date(2026, 10, 4))
    assert list(s["total_paise"]) == [0, 30000, 12000, 0]


def test_month_boundaries_and_leap_year(conn):
    tx.bulk_add(conn, [
        dict(transaction_date="2028-02-29", amount="100", category="Food"),   # leap day
        dict(transaction_date="2028-03-01", amount="200", category="Food"),
        dict(transaction_date="2028-01-31", amount="50", category="Food"),
    ], date(2028, 12, 31))
    df = tx.load_dataframe(conn)
    assert an.total_expenses(df, *an.month_bounds(2028, 2)) == 10000
    assert an.total_expenses(df, *an.month_bounds(2028, 3)) == 20000
    assert an.month_bounds(2028, 2) == (date(2028, 2, 1), date(2028, 2, 29))
    assert an.month_bounds(2027, 2)[1] == date(2027, 2, 28)
    assert an.month_bounds(2026, 12)[1] == date(2026, 12, 31)
    assert an.total_expenses(df, date(2028, 2, 29), date(2028, 3, 1)) == 30000  # inclusive both ends


def test_empty_database_gives_empty_results(conn):
    df = tx.load_dataframe(conn)
    s, e = date(2026, 10, 1), date(2026, 10, 31)
    assert an.total_expenses(df, s, e) == 0 and an.category_breakdown(df, s, e).empty
    assert an.monthly_totals(df).empty and an.top_expenses(df, s, e).empty
    assert an.wasteful_summary(df, s, e)["share_pct"] is None
    assert an.month_over_month(df, 2026, 10).empty


def test_invalid_period_rejected(df):
    with pytest.raises(ValueError):
        an.total_expenses(df, date(2026, 10, 5), date(2026, 10, 1))
