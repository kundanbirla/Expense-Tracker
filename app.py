"""Smart Expense Analyzer - home page: quick daily entry with ML-assisted categorisation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datetime import date  # noqa: E402

import streamlit as st  # noqa: E402

from src import analytics, config, database as db, transactions as tx  # noqa: E402
from src.ui import fmt_inr, get_model, load_df, render_prediction  # noqa: E402
from src.validation import ValidationError, parse_amount_to_paise  # noqa: E402

st.set_page_config(page_title="Smart Expense Analyzer", page_icon="💸", layout="wide")

BLANK = "—"
DEFAULTS = {
    "f_type": "expense", "f_date": date.today(), "f_amount": "", "f_desc": "", "f_category": config.CATEGORIES[0],
    "f_income_category": config.INCOME_CATEGORIES[0], "f_wasteful": BLANK, "f_needwant": BLANK,
    "f_payment": BLANK, "f_context": BLANK, "f_notes": "",
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def _opt(value):
    return None if value in (None, BLANK) else value


def apply_suggestion() -> None:
    pred = get_model().predict(st.session_state.f_desc)
    st.session_state.f_category = pred.category
    st.session_state.f_needwant = pred.need_want


def save_entry() -> None:
    s = st.session_state
    pred = None
    if s.f_type == "expense" and s.f_desc.strip():
        pred = get_model().predict(s.f_desc)
    data = {
        "transaction_date": s.f_date, "amount": s.f_amount, "transaction_type": s.f_type,
        "category": s.f_income_category if s.f_type == "income" else s.f_category,
        "description": s.get("f_desc", ""), "wasteful_label": _opt(s.get("f_wasteful", BLANK)),
        "need_want": _opt(s.get("f_needwant", BLANK)),
        "payment_method": _opt(s.get("f_payment", BLANK)), "context": _opt(s.get("f_context", BLANK)),
        "notes": s.get("f_notes", ""),
        # remember what the model suggested so the Settings page can report how often you agreed
        "predicted_category": pred.category if pred and pred.suggest else None,
        "predicted_confidence": pred.category_confidence if pred and pred.suggest else None,
    }
    try:
        with db.connect() as conn:
            tx.add_transaction(conn, data)
        saved = f"Saved {fmt_inr(parse_amount_to_paise(s.f_amount))} to {data['category']}."
    except ValidationError as exc:
        s.flash = ("error", exc.errors)
        return
    for k in ("f_amount", "f_desc", "f_notes"):
        s[k] = ""
    for k in ("f_wasteful", "f_needwant", "f_payment", "f_context"):
        s[k] = BLANK
    s.flash = ("success", saved)


st.title("💸 Smart Expense Analyzer")
st.caption("Log today's spending in under 30 seconds. Totals are exact; the ML only *suggests* - you decide.")

flash = st.session_state.pop("flash", None)
if flash:
    kind, payload = flash
    if kind == "success":
        st.success(payload)
    else:
        for field, msg in payload.items():
            st.error(f"**{field.replace('_', ' ').title()}:** {msg}")

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Add a transaction")
    st.radio("Type", config.TRANSACTION_TYPES, key="f_type", horizontal=True,
             format_func=str.capitalize,
             help="Refunds reduce spending in their category. Income is never counted as an expense.")
    c1, c2 = st.columns(2)
    c1.date_input("Date", key="f_date", max_value=date.today())
    c2.text_input("Amount (₹)", key="f_amount", placeholder="e.g. 250 or 99.50")
    st.text_input("Description / purpose (English or Hinglish)", key="f_desc", max_chars=config.MAX_DESCRIPTION_LEN,
                  placeholder="e.g. Late night food delivery · friend ke saath dinner")

    if st.session_state.f_type == "expense" and len(st.session_state.f_desc.strip()) >= 3:
        pred = get_model().predict(st.session_state.f_desc)
        with st.container(border=True):
            st.markdown("**💡 Smart suggestion**")
            render_prediction(pred)
            st.button("Use this suggestion", on_click=apply_suggestion, disabled=not pred.suggest,
                      help="Fills Category and Need/Want. You can still change them.")

    if st.session_state.f_type == "income":
        st.selectbox("Income category", config.INCOME_CATEGORIES, key="f_income_category")
    else:
        st.selectbox("Category", config.CATEGORIES, key="f_category")

    if st.session_state.f_type == "expense":
        with st.expander("Reflection & details (optional)"):
            r1, r2 = st.columns(2)
            r1.selectbox("Was it wasteful? (your own judgement)", [BLANK] + config.WASTEFUL_OPTIONS, key="f_wasteful")
            r2.selectbox("Need or want?", [BLANK] + config.NEED_WANT_OPTIONS, key="f_needwant")
            r3, r4 = st.columns(2)
            r3.selectbox("Payment method", [BLANK] + config.PAYMENT_METHODS, key="f_payment")
            r4.selectbox("Purchase context", [BLANK] + config.CONTEXT_OPTIONS, key="f_context")
            st.text_area("Notes", key="f_notes", max_chars=config.MAX_NOTES_LEN,
                         help="Avoid entering sensitive details such as card numbers or OTPs.")
    st.button("Save transaction", type="primary", on_click=save_entry)

with right:
    df = load_df()
    today = date.today()
    st.subheader("This month")
    if df.empty:
        st.info("No transactions yet. Add one on the left, or load demo data from **Settings**.")
    else:
        start = today.replace(day=1)
        spent = analytics.total_expenses(df, start, today)
        avg, n_days = analytics.daily_average(df, start, today)
        k1, k2 = st.columns(2)
        k1.metric("Spent so far", fmt_inr(spent))
        k2.metric("Daily average", fmt_inr(avg), f"over {n_days} days", delta_color="off")
        st.subheader("Recent transactions")
        recent = tx.filter_transactions(df).head(8)
        st.dataframe(
            recent.assign(date=recent["transaction_date"].dt.strftime("%d %b"), amount_fmt=recent["amount_paise"].map(fmt_inr))
            [["date", "category", "description", "amount_fmt", "transaction_type"]]
            .rename(columns={"amount_fmt": "amount", "transaction_type": "type"}),
            hide_index=True, width="stretch")
    st.caption("Use the sidebar pages for history, dashboard, insights and the ML Smart Categoriser.")
