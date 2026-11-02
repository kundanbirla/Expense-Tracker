"""Dashboard: totals, category split, trends, month-over-month, wasteful-labelled spend."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import date, timedelta  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src import analytics  # noqa: E402
from src.ui import fmt_inr, load_df  # noqa: E402

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("📊 Dashboard")

df = load_df()
if df.empty:
    st.info("No data yet. Add transactions on the home page or load demo data from Settings.")
    st.stop()

today = date.today()
choice = st.radio("Period", ["This month", "Last month", "Last 30 days", "Last 90 days", "All time", "Custom"],
                  horizontal=True)
first = today.replace(day=1)
if choice == "This month":
    start, end = first, today
elif choice == "Last month":
    end = first - timedelta(days=1)
    start = end.replace(day=1)
elif choice == "Last 30 days":
    start, end = today - timedelta(days=29), today
elif choice == "Last 90 days":
    start, end = today - timedelta(days=89), today
elif choice == "All time":
    start, end = df["transaction_date"].min().date(), max(today, df["transaction_date"].max().date())
else:
    c1, c2 = st.columns(2)
    start = c1.date_input("From", first, max_value=today)
    end = c2.date_input("To", today, max_value=today)
    if start > end:
        st.error("'From' must not be after 'To'.")
        st.stop()
st.caption(f"Showing {start:%d %b %Y} – {end:%d %b %Y} · amounts in ₹ · spending is net of refunds · income never counts as spending")

spent = analytics.total_expenses(df, start, end)
income = analytics.total_income(df, start, end)
avg, n_days = analytics.daily_average(df, start, end)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total spending", fmt_inr(spent))
k2.metric("Daily average", fmt_inr(avg), f"over {n_days} days", delta_color="off")
k3.metric("Income", fmt_inr(income))
k4.metric("Net cash flow", fmt_inr(income - spent))

cats = analytics.category_breakdown(df, start, end)
if cats.empty:
    st.warning("No expenses in this period.")
else:
    a, b = st.columns(2)
    with a:
        st.subheader("Spending by category")
        fig = px.bar(cats.assign(rupees=cats["total_paise"] / 100, label=cats["share_pct"].map("{:.1f}%".format)),
                     x="rupees", y="category", orientation="h", text="label",
                     labels={"rupees": "Amount (₹)", "category": ""})
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=0, r=0, t=10, b=0), height=360)
        st.plotly_chart(fig, width="stretch")
    with b:
        st.subheader("Daily spending")
        daily = analytics.daily_series(df, start, end)
        fig = px.bar(daily.assign(rupees=daily["total_paise"] / 100), x="date", y="rupees",
                     labels={"rupees": "Amount (₹)", "date": ""})
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=360)
        st.plotly_chart(fig, width="stretch")

st.subheader("Month by month")
mt = analytics.monthly_totals(df)
long = pd.melt(mt.assign(Spending=mt["expenses_paise"] / 100, Income=mt["income_paise"] / 100),
               id_vars="month", value_vars=["Spending", "Income"], var_name="kind", value_name="rupees")
fig = px.bar(long, x="month", y="rupees", color="kind", barmode="group", labels={"rupees": "Amount (₹)", "month": "", "kind": ""})
fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320)
st.plotly_chart(fig, width="stretch")

st.subheader("Month-over-month by category")
months = list(mt["month"])[::-1]
sel = st.selectbox("Compare this month with the one before it", months)
y, m = map(int, sel.split("-"))
mom = analytics.month_over_month(df, y, m)
if mom.empty:
    st.caption("No spending in either month.")
else:
    st.dataframe(pd.DataFrame({
        "Category": mom["category"], sel: mom["current_paise"].map(fmt_inr), "Previous month": mom["previous_paise"].map(fmt_inr),
        "Change": mom["change_paise"].map(fmt_inr),
        "Change %": mom["change_pct"].map(lambda v: "new" if pd.isna(v) else f"{v:+.0f}%")}),
        hide_index=True, width="stretch")

l, r = st.columns(2)
with l:
    st.subheader("Largest expenses")
    top = analytics.top_expenses(df, start, end, 5)
    if top.empty:
        st.caption("No expenses in this period.")
    else:
        st.dataframe(pd.DataFrame({"Date": top["transaction_date"].dt.strftime("%d %b"), "Category": top["category"],
                                   "Description": top["description"], "Amount": top["amount_paise"].map(fmt_inr)}),
                     hide_index=True, width="stretch")
with r:
    st.subheader("Wasteful (your own labels)")
    w = analytics.wasteful_summary(df, start, end)
    st.metric("Marked wasteful", fmt_inr(w["total_paise"]),
              f"{w['share_pct']}% of expenses · {w['count']} purchases" if w["share_pct"] is not None else None, delta_color="off")
    st.caption(f"{w['labelled_count']} of {w['expense_count']} expenses carry a wasteful label. "
               "This reflects your judgement only - the app never decides what is wasteful.")
    nw = analytics.need_want_summary(df, start, end)
    if not nw.empty:
        st.dataframe(pd.DataFrame({"Need / want": nw["need_want"], "Total": nw["total_paise"].map(fmt_inr), "Purchases": nw["count"]}),
                     hide_index=True, width="stretch")
