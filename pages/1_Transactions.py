"""Transaction history: filter, search, sort, edit, delete (with confirmation) and export."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math  # noqa: E402
from datetime import date  # noqa: E402

import streamlit as st  # noqa: E402

from src import analytics, config, database as db, transactions as tx  # noqa: E402
from src.ui import fmt_inr, load_df  # noqa: E402
from src.validation import ValidationError  # noqa: E402

st.set_page_config(page_title="Transactions", page_icon="🧾", layout="wide")
st.title("🧾 Transactions")

df = load_df()
if df.empty:
    st.info("Nothing recorded yet. Add a transaction on the home page or load demo data in Settings.")
    st.stop()

with st.expander("Filters", expanded=True):
    c1, c2, c3 = st.columns(3)
    dmin, dmax = df["transaction_date"].min().date(), max(df["transaction_date"].max().date(), date.today())
    rng = c1.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    cats = c2.multiselect("Category", sorted(df["category"].unique()))
    types = c3.multiselect("Type", config.TRANSACTION_TYPES)
    c4, c5, c6, c7 = st.columns(4)
    waste = c4.multiselect("Wasteful label", config.WASTEFUL_OPTIONS)
    pay = c5.multiselect("Payment method", config.PAYMENT_METHODS)
    search = c6.text_input("Search description / notes")
    sort = c7.selectbox("Sort by", ["newest", "oldest", "largest", "smallest"], format_func=str.capitalize)

start, end = (rng if isinstance(rng, tuple) and len(rng) == 2 else (dmin, dmax))
view = tx.filter_transactions(df, start, end, cats, types, waste, pay, search, sort)

net = analytics.net_expense_paise(view).sum()
m1, m2, m3 = st.columns(3)
m1.metric("Matching records", len(view))
m2.metric("Net spending (expenses − refunds)", fmt_inr(net))
m3.metric("Income in view", fmt_inr(view.loc[view["transaction_type"] == "income", "amount_paise"].sum()))

if view.empty:
    st.warning("No transactions match these filters.")
    st.stop()

page_size = 20
pages = max(1, math.ceil(len(view) / page_size))
page = st.number_input(f"Page (of {pages})", 1, pages, 1) if pages > 1 else 1
chunk = view.iloc[(page - 1) * page_size: page * page_size]

st.dataframe(
    chunk.assign(date=chunk["transaction_date"].dt.strftime("%Y-%m-%d"), amount_inr=chunk["amount_paise"].map(fmt_inr))
    [["id", "date", "transaction_type", "category", "amount_inr", "description", "wasteful_label", "need_want", "payment_method"]]
    .rename(columns={"transaction_type": "type", "amount_inr": "amount", "wasteful_label": "wasteful"}),
    hide_index=True, width="stretch")

st.download_button("⬇️ Export these records (CSV)", tx.to_export_dataframe(view).to_csv(index=False).encode("utf-8"),
                   file_name="my_transactions.csv", mime="text/csv")

st.divider()
st.subheader("Edit or delete")
labels = {int(r.id): f"#{r.id} · {r.transaction_date:%d %b %Y} · {r.description or r.category} · {fmt_inr(r.amount_paise)}"
          for r in chunk.itertuples()}
tx_id = st.selectbox("Choose a transaction (from the page above)", list(labels), format_func=labels.get)
with db.connect() as conn:
    row = tx.get_transaction(conn, tx_id)

if row:
    BLANK = "—"
    k = f"edit_{tx_id}_"
    with st.form(f"edit_form_{tx_id}"):
        e1, e2, e3 = st.columns(3)
        ttype = row["transaction_type"]  # type is fixed when editing (delete and re-add to change it)
        e1.text_input("Type", ttype.capitalize(), disabled=True, key=k + "t")
        edate = e2.date_input("Date", date.fromisoformat(row["transaction_date"]), max_value=date.today(), key=k + "d")
        amount = e3.text_input("Amount (₹)", f"{row['amount_paise'] / 100:.2f}", key=k + "a")
        cat_opts = config.INCOME_CATEGORIES if ttype == "income" else config.CATEGORIES
        e4, e5 = st.columns(2)
        category = e4.selectbox("Category", cat_opts, index=cat_opts.index(row["category"]) if row["category"] in cat_opts else 0, key=k + "c")
        desc = e5.text_input("Description", row["description"], key=k + "x")
        e6, e7, e8, e9 = st.columns(4)

        def pick(col, label, options, current, key):
            opts = [BLANK] + options
            return col.selectbox(label, opts, index=opts.index(current) if current in opts else 0, key=key)

        waste_v = pick(e6, "Wasteful?", config.WASTEFUL_OPTIONS, row["wasteful_label"], k + "w")
        nw_v = pick(e7, "Need/want", config.NEED_WANT_OPTIONS, row["need_want"], k + "n")
        pay_v = pick(e8, "Payment", config.PAYMENT_METHODS, row["payment_method"], k + "p")
        ctx_v = pick(e9, "Context", config.CONTEXT_OPTIONS, row["context"], k + "q")
        notes = st.text_area("Notes", row["notes"], key=k + "o")
        if st.form_submit_button("Save changes", type="primary"):
            nz = lambda v: None if v == BLANK else v  # noqa: E731
            try:
                with db.connect() as conn:
                    tx.update_transaction(conn, tx_id, {
                        "transaction_date": edate, "amount": amount, "transaction_type": ttype, "category": category,
                        "description": desc, "wasteful_label": nz(waste_v), "need_want": nz(nw_v),
                        "payment_method": nz(pay_v), "context": nz(ctx_v), "notes": notes})
                st.success("Updated. Reports now reflect the change.")
                st.rerun()
            except ValidationError as exc:
                for field, msg in exc.errors.items():
                    st.error(f"**{field.replace('_', ' ').title()}:** {msg}")

    with st.popover("🗑️ Delete this transaction"):
        st.write(f"Delete **{labels[tx_id]}**? This cannot be undone.")
        if st.button("Yes, delete permanently", key=f"del_{tx_id}", type="primary"):
            with db.connect() as conn:
                tx.delete_transaction(conn, tx_id)
            st.rerun()
