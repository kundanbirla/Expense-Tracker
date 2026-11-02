"""Explainable spending insights. Every insight states the comparison it used.

Wording is deliberately neutral: unusual spending is never called fraud, and the user's own
"wasteful" labels are reported separately from statistical alerts.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from . import analytics, config
from .anomaly_detection import detect_rule_based


@dataclass
class Insight:
    kind: str          # category_increase | large_expense | frequent_small | wasteful | reminder | data
    title: str
    message: str
    rule: str          # the exact comparison/threshold used
    severity: str = "info"   # info | notice


def inr(paise: float) -> str:
    return f"₹{paise / 100:,.0f}"


def _category_increase(df: pd.DataFrame, today: date, min_pct: float = 25.0,
                       min_abs_paise: int = 20000) -> list[Insight]:
    cur_start = today.replace(day=1)
    prev_month_end = cur_start - timedelta(days=1)
    prev_start = prev_month_end.replace(day=1)
    n_days = min(today.day, prev_month_end.day)
    prev_end = prev_start + timedelta(days=n_days - 1)

    prev_total = analytics.total_expenses(df, prev_start, prev_end)
    if prev_total <= 0:
        return [Insight("data", "Month comparison: not enough data yet",
                        "There is no spending recorded for the comparable days of last month, "
                        "so category changes cannot be compared yet.",
                        f"Needs expenses between {prev_start:%d %b} and {prev_end:%d %b}.")]
    cur = analytics.category_breakdown(df, cur_start, today).set_index("category")["total_paise"]
    prev = analytics.category_breakdown(df, prev_start, prev_end).set_index("category")["total_paise"]
    out = []
    for cat, c in cur.items():
        p = int(prev.get(cat, 0))
        if p > 0 and c - p >= min_abs_paise and (c - p) / p * 100 >= min_pct:
            out.append(Insight(
                "category_increase", f"{cat} spending is higher than the comparable period last month",
                f"{cat}: {inr(c)} ({cur_start:%d %b}–{today:%d %b}) vs {inr(p)} "
                f"({prev_start:%d %b}–{prev_end:%d %b}), up {(c - p) / p * 100:.0f}%.",
                f"Month-to-date net {cat} spending compared with the same {n_days} days of the previous "
                f"month; flagged when it is ≥{min_pct:.0f}% and ≥{inr(min_abs_paise)} higher.",
                "notice"))
    return out


def _large_expenses(df: pd.DataFrame, today: date, days: int = 7) -> list[Insight]:
    # Bills (rent, recharge, EMI...) are fixed, heterogeneous and recurring, so a per-category
    # "usual range" would give false alarms; they are excluded from this check.
    res = detect_rule_based(df[df["category"] != "Bills"])
    flagged = res[res["is_anomaly"]].merge(df, on="id")
    flagged = flagged[flagged["transaction_date"] >= pd.Timestamp(today - timedelta(days=days - 1))]
    out = []
    for _, r in flagged.sort_values("amount_paise", ascending=False).iterrows():
        desc = r["description"] or r["category"]
        out.append(Insight(
            "large_expense", f"Larger than usual {r['category']} expense",
            f"{inr(r['amount_paise'])} on '{desc}' ({r['transaction_date']:%d %b}) is larger than your usual "
            f"recorded {r['category']} expenses (usual upper range ≈ {inr(r['threshold_paise'])}).",
            f"Amount above max(Q3 + 3×IQR, 3×median) of the {int(r['history_n'])} earlier "
            f"{r['category']} expenses (Bills are excluded). This is a statistical comparison, not a judgement.",
            "notice"))
    return out


def _frequent_small(df: pd.DataFrame, today: date, small_paise: int = 15000, min_count: int = 5,
                    days: int = 7) -> list[Insight]:
    p = analytics.period(df, today - timedelta(days=days - 1), today)
    p = p[(p["transaction_type"] == "expense") & (p["amount_paise"] < small_paise)]
    out = []
    for cat, g in p.groupby("category"):
        if len(g) >= min_count:
            out.append(Insight(
                "frequent_small", f"Many small {cat} purchases this week",
                f"You recorded {len(g)} small {cat} purchases (under {inr(small_paise)} each) in the last "
                f"{days} days, totalling {inr(g['amount_paise'].sum())}.",
                f"Counts expenses below {inr(small_paise)} per category over the last {days} days; "
                f"shown when there are ≥{min_count}."))
    return out


def _wasteful(df: pd.DataFrame, today: date) -> list[Insight]:
    start = today.replace(day=1)
    s = analytics.wasteful_summary(df, start, today)
    rule = "Sum of this month's expenses that YOU labelled 'Wasteful = Yes'. The app does not judge purchases."
    if s["count"] == 0:
        return [Insight("wasteful", "Wasteful-labelled spending", "No expenses are marked wasteful this month.", rule)]
    return [Insight("wasteful", "Wasteful-labelled spending",
                    f"You marked {inr(s['total_paise'])} ({s['count']} purchases) as wasteful this month"
                    f"{'' if s['share_pct'] is None else f', {s['share_pct']}% of recorded expenses'}.", rule)]


def generate_insights(df: pd.DataFrame, today: date | None = None, reminder: bool = True) -> list[Insight]:
    today = today or date.today()
    if df.empty:
        return [Insight("data", "No data yet", "Add a few expenses to start seeing insights.",
                        "Insights need recorded transactions.")]
    out: list[Insight] = []
    n_exp = int((df["transaction_type"] == "expense").sum())
    if n_exp < config.MIN_EXPENSES_FOR_INSIGHTS:
        out.append(Insight("data", "Not enough data yet",
                           f"You have {n_exp} expenses recorded. Comparisons and unusual-spending checks "
                           f"switch on after {config.MIN_EXPENSES_FOR_INSIGHTS}.",
                           "Avoids pretending to personalise from very little history."))
    else:
        out += _category_increase(df, today)
        out += _large_expenses(df, today)
        out += _frequent_small(df, today)
    out += _wasteful(df, today)
    if reminder and not (df["transaction_date"] == pd.Timestamp(today)).any():
        out.append(Insight("reminder", "No entry for today yet",
                           "You have not added a transaction for today.",
                           "Optional daily logging reminder (shown when nothing is dated today)."))
    return out
