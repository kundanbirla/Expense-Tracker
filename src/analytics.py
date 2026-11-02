"""Reliable, rule-based reporting (no ML). All money maths is done in integer paise.

Business rules (documented so tests can verify them):
* ``expense``  adds to spending.
* ``refund``   subtracts from spending of its own category (net spending).
* ``income``   is never counted as spending; it only feeds income / net cash-flow.
* Periods are inclusive of both start and end dates.
* "Daily average" = net spending / number of calendar days in the period (returned alongside).
* "Wasteful" totals use only expenses the user labelled ``Yes`` (refunds are not netted off).
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ helpers
def period(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Rows whose date lies in [start, end]."""
    if start > end:
        raise ValueError("start date must not be after end date")
    return df[(df["transaction_date"] >= pd.Timestamp(start)) & (df["transaction_date"] <= pd.Timestamp(end))]


def net_expense_paise(df: pd.DataFrame) -> pd.Series:
    """Per-row signed spending: +amount for expense, -amount for refund, 0 otherwise."""
    amt = df["amount_paise"].astype("int64")
    return amt.where(df["transaction_type"] == "expense", 0) - amt.where(df["transaction_type"] == "refund", 0)


def month_bounds(year: int, month: int) -> tuple[date, date]:
    first = date(year, month, 1)
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    return first, nxt - timedelta(days=1)


# ------------------------------------------------------------------ totals
def total_expenses(df: pd.DataFrame, start: date, end: date) -> int:
    return int(net_expense_paise(period(df, start, end)).sum())


def total_income(df: pd.DataFrame, start: date, end: date) -> int:
    p = period(df, start, end)
    return int(p.loc[p["transaction_type"] == "income", "amount_paise"].sum())


def net_cash_flow(df: pd.DataFrame, start: date, end: date) -> int:
    return total_income(df, start, end) - total_expenses(df, start, end)


def daily_average(df: pd.DataFrame, start: date, end: date) -> tuple[float, int]:
    """(average net spending per day in paise, number of days used)."""
    n_days = (end - start).days + 1
    if n_days <= 0:
        raise ValueError("start date must not be after end date")
    return total_expenses(df, start, end) / n_days, n_days


# --------------------------------------------------------------- breakdowns
def category_breakdown(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Net spending per category with percentage share (categories with net <= 0 are omitted)."""
    p = period(df, start, end)
    cols = ["category", "total_paise", "share_pct"]
    if p.empty:
        return pd.DataFrame(columns=cols)
    g = (p.assign(net=net_expense_paise(p)).groupby("category")["net"].sum().reset_index()
         .rename(columns={"net": "total_paise"}))
    g = g[g["total_paise"] > 0].sort_values("total_paise", ascending=False)
    total = g["total_paise"].sum()
    g["share_pct"] = (g["total_paise"] / total * 100).round(1) if total else 0.0
    return g[cols].reset_index(drop=True)


def daily_series(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Net spending per calendar day (zero-filled)."""
    p = period(df, start, end)
    days = pd.date_range(start, end, freq="D")
    s = pd.Series(net_expense_paise(p).to_numpy(), index=p["transaction_date"].to_numpy())
    s = s.groupby(level=0).sum().reindex(days, fill_value=0)
    return pd.DataFrame({"date": days, "total_paise": s.to_numpy().astype("int64")})


def monthly_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Per calendar month (gaps zero-filled): expenses (net), income and net cash-flow in paise."""
    cols = ["month", "expenses_paise", "income_paise", "net_paise"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    work = df.assign(
        month=df["transaction_date"].dt.to_period("M"),
        spend=net_expense_paise(df),
        inc=df["amount_paise"].where(df["transaction_type"] == "income", 0),
    )
    g = work.groupby("month")[["spend", "inc"]].sum()
    g = g.reindex(pd.period_range(g.index.min(), g.index.max(), freq="M"), fill_value=0)
    out = pd.DataFrame({
        "month": g.index.astype(str),
        "expenses_paise": g["spend"].astype("int64").to_numpy(),
        "income_paise": g["inc"].astype("int64").to_numpy(),
    })
    out["net_paise"] = out["income_paise"] - out["expenses_paise"]
    return out


def month_over_month(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """Category comparison of a month versus the previous calendar month."""
    cs, ce = month_bounds(year, month)
    py, pm = (year - 1, 12) if month == 1 else (year, month - 1)
    ps, pe = month_bounds(py, pm)
    cur = category_breakdown(df, cs, ce).set_index("category")["total_paise"]
    prev = category_breakdown(df, ps, pe).set_index("category")["total_paise"]
    cats = cur.index.union(prev.index)
    out = pd.DataFrame({"current_paise": cur.reindex(cats, fill_value=0),
                        "previous_paise": prev.reindex(cats, fill_value=0)})
    out["change_paise"] = out["current_paise"] - out["previous_paise"]
    out["change_pct"] = np.where(out["previous_paise"] > 0,
                                 out["change_paise"] / out["previous_paise"].replace(0, np.nan) * 100, np.nan)
    return out.reset_index(names="category").sort_values("current_paise", ascending=False).reset_index(drop=True)


def top_expenses(df: pd.DataFrame, start: date, end: date, n: int = 5) -> pd.DataFrame:
    p = period(df, start, end)
    p = p[p["transaction_type"] == "expense"]
    return p.sort_values(["amount_paise", "id"], ascending=[False, False]).head(n)


# ---------------------------------------------------- reflection summaries
def wasteful_summary(df: pd.DataFrame, start: date, end: date) -> dict:
    """What the *user* labelled wasteful. The app makes no judgement of its own."""
    p = period(df, start, end)
    exp = p[p["transaction_type"] == "expense"]
    gross = int(exp["amount_paise"].sum())
    wasteful = exp[exp["wasteful_label"] == "Yes"]
    total = int(wasteful["amount_paise"].sum())
    return {
        "total_paise": total,
        "count": int(len(wasteful)),
        "share_pct": round(total / gross * 100, 1) if gross else None,
        "labelled_count": int(exp["wasteful_label"].notna().sum()),
        "expense_count": int(len(exp)),
    }


def need_want_summary(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    p = period(df, start, end)
    exp = p[(p["transaction_type"] == "expense") & p["need_want"].notna()]
    if exp.empty:
        return pd.DataFrame(columns=["need_want", "total_paise", "count"])
    return (exp.groupby("need_want")["amount_paise"].agg(total_paise="sum", count="count")
            .reset_index().sort_values("total_paise", ascending=False))
