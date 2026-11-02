"""Settings: demo data, export, deletion, ML feedback and privacy notes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import date  # noqa: E402

import streamlit as st  # noqa: E402

from src import config, database as db, text_classifier as tc, transactions as tx  # noqa: E402
from src.synthetic_data import generate_transactions  # noqa: E402
from src.ui import get_model, load_df, user_training_rows  # noqa: E402

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")
st.title("⚙️ Settings")

df = load_df()
st.subheader("Your data")
st.write(f"{len(df)} transactions stored locally in `{config.db_path().name}`.")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Load synthetic demo data**")
    st.caption("About 75 days of fake transactions ending today (incl. income, refunds and a few unusual expenses). "
               "Adds to what you already have.")
    if st.button("Add demo data"):
        with db.connect() as conn:
            n = tx.bulk_add(conn, generate_transactions(end=date.today()))
        st.success(f"Added {n} synthetic transactions.")
        st.rerun()
with c2:
    st.markdown("**Export everything**")
    st.caption("Your records as a spreadsheet-safe CSV.")
    if df.empty:
        st.button("Export CSV", disabled=True)
    else:
        st.download_button("⬇️ Download all (CSV)", tx.to_export_dataframe(df).to_csv(index=False).encode("utf-8"),
                           "smart_expense_export.csv", "text/csv")

st.markdown("**Delete all data**")
confirm_key = f"del_confirm_{st.session_state.get('del_confirm_version', 0)}"
confirm = st.text_input("Type DELETE to enable the button", key=confirm_key)
if st.button("Delete all my transactions", type="primary", disabled=confirm != "DELETE"):
    with db.connect() as conn:
        n = tx.delete_all(conn)
    # Use a fresh widget key after deletion instead of mutating the current
    # text_input's session-state value after the widget was instantiated.
    st.session_state.del_confirm_version = st.session_state.get("del_confirm_version", 0) + 1
    st.success(f"Deleted {n} transactions.")
    st.rerun()

st.divider()
st.subheader("ML suggestions: how often did you agree?")
used = df[(df["transaction_type"] == "expense") & df["predicted_category"].notna()]
if used.empty:
    st.caption("No saved entries used a suggestion yet. Once you add expenses with a description, the app records "
               "what was suggested so the model's real-world accuracy for *you* can be measured.")
else:
    agree = (used["predicted_category"] == used["category"]).mean()
    st.metric("Suggestion matched your final category", f"{agree:.0%}", f"{len(used)} entries with a suggestion", delta_color="off")

st.markdown("**Retrain on my confirmed entries**")
rows = user_training_rows(df)
st.caption(f"{len(rows)} of your expenses have a description and category. Retraining adds them to the bundled training "
           "data (category from your final choice; need/want only where you filled it in). It runs locally.")
b1, b2 = st.columns(2)
if b1.button("Retrain model with my data", disabled=len(rows) < 20, help="Needs at least 20 described expenses."):
    with st.spinner("Training…"):
        model = tc.train_default_model(extra=rows)
        model.meta["user_rows"] = len(rows)
        try:
            model.save()
        except OSError:
            st.warning("Could not save the model file; using it only until the app restarts.")
        get_model.clear()
    st.success(f"Retrained with {len(rows)} of your entries.")
if b2.button("Reset to the bundled model"):
    with st.spinner("Training…"):
        model = tc.train_default_model()
        try:
            model.save()
        except OSError:
            pass
        get_model.clear()
    st.success("Model reset to the bundled training data.")
st.caption("Note: retraining on your own labels can only help if those labels are consistent; evaluate before trusting it.")

st.divider()
st.subheader("Privacy & scope")
st.markdown(
    "- Data stays in a local SQLite file; nothing is sent anywhere. The ML model runs on your machine.\n"
    "- Never enter bank passwords, card numbers, PINs or OTPs (not needed, not requested).\n"
    "- This is a personal reflection tool - not a bank, financial adviser or fraud-detection system.\n"
    "- Single-user app: no accounts yet. If you deploy publicly, use synthetic data only (see README).")
