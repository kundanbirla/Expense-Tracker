"""Small helpers shared by the Streamlit pages."""
from __future__ import annotations

import json
import re
from datetime import date

import pandas as pd
import streamlit as st

from . import config, database as db, text_classifier as tc, transactions as tx


def fmt_inr(paise: float) -> str:
    """Indian digit grouping: 12345678 paise*100 -> ₹1,23,456.78 (decimals only when non-zero)."""
    neg = paise < 0
    rupees, ps = divmod(abs(int(round(paise))), 100)
    s = str(rupees)
    if len(s) > 3:
        s = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", s[:-3]) + "," + s[-3:]
    return f"{'-' if neg else ''}₹{s}" + (f".{ps:02d}" if ps else "")


@st.cache_resource(show_spinner="Loading ML model…")
def get_model() -> tc.TransactionTextModel:
    return tc.load_or_train()


def load_df() -> pd.DataFrame:
    with db.connect() as conn:
        return tx.load_dataframe(conn)


def load_metrics() -> dict | None:
    try:
        return json.loads(config.METRICS_PATH.read_text())
    except (OSError, ValueError):
        return None


def user_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Expenses with a description -> (text, category, optional context) for retraining."""
    ex = df[(df["transaction_type"] == "expense") & (df["description"].fillna("").str.strip() != "")]
    out = pd.DataFrame({"text": ex["description"], "category": ex["category"],
                        "context": ex["need_want"].map(tc.NEED_WANT_TO_CONTEXT)})
    return out.reset_index(drop=True)


def month_label(ts: pd.Timestamp | date) -> str:
    return f"{ts:%b %Y}"


def confidence_badge(level: str) -> str:
    return {"high": "🟢 High", "medium": "🟡 Medium", "low": "🔴 Low"}[level]


def render_prediction(pred: tc.Prediction, *, expanded_limits: bool = False) -> None:
    """Shared card used by the entry form and the Smart Categoriser page."""
    if pred.confidence_level == "low":
        st.warning(f"**Not sure.** Best guess is *{pred.category}* "
                   f"({pred.category_confidence:.0%}) - too low to pre-fill. Please choose yourself.")
    else:
        st.success(f"**{pred.category}** · {pred.context_label}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Category confidence", f"{pred.category_confidence:.0%}", confidence_badge(pred.confidence_level),
              delta_color="off")
    c2.metric("Need/want confidence", f"{pred.context_confidence:.0%}")
    c3.metric("Words understood", f"{pred.known_token_ratio:.0%}")
    if pred.top_terms:
        st.caption("Words that pushed the decision: " + ", ".join(f"`{t}`" for t in pred.top_terms))
    alts = ", ".join(f"{c} {p:.0%}" for c, p in pred.category_alternatives)
    st.caption(f"Top candidates: {alts}")
    with st.expander("Limitations of this prediction", expanded=expanded_limits):
        for note in pred.limitations:
            st.write("• " + note)
