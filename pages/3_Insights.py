"""Explainable insights and unusual-expense detection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.anomaly_detection import detect_isolation_forest, detect_rule_based  # noqa: E402
from src.insights import generate_insights  # noqa: E402
from src.ui import fmt_inr, load_df  # noqa: E402

st.set_page_config(page_title="Insights", page_icon="💡", layout="wide")
st.title("💡 Insights")
st.caption("Every alert states the rule it used. Unusual ≠ wrong, and nothing here is fraud detection.")

df = load_df()
if df.empty:
    st.info("Add some transactions first (or load demo data in Settings).")
    st.stop()

reminder = st.toggle("Show daily-logging reminder", value=True)
for ins in generate_insights(df, reminder=reminder):
    box = st.warning if ins.severity == "notice" else st.info
    box(f"**{ins.title}**\n\n{ins.message}")
    with st.expander("How was this calculated?"):
        st.write(ins.rule)

st.divider()
st.subheader("Unusually large expenses")
method = st.radio("Method", ["Rule-based baseline (recommended)", "Isolation Forest (experimental)"], horizontal=True)
if method.startswith("Rule"):
    res = detect_rule_based(df)
    st.caption("Flags an expense above max(Q3 + 3×IQR, 3×median) of your *earlier* entries in the same category. "
               "Needs 8 earlier entries in the category before it flags anything.")
else:
    res = detect_isolation_forest(df)
    st.caption("Unsupervised model on amount, amount relative to category median and weekday. Needs ≥ 40 expenses; "
               "hard to validate on small personal data - see README for the measured comparison.")

flagged = res[res["is_anomaly"]].merge(df, on="id")
if flagged.empty:
    st.success("Nothing unusual found with this method.")
else:
    st.dataframe(pd.DataFrame({
        "Date": flagged["transaction_date"].dt.strftime("%d %b %Y"), "Category": flagged["category"],
        "Description": flagged["description"], "Amount": flagged["amount_paise"].map(fmt_inr),
        "Why": flagged["reason"]}).sort_values("Date", ascending=False), hide_index=True, width="stretch")
    st.caption("If an alert is a false positive, that's useful to know - alerts are prompts to look, not conclusions.")
