"""ML feature: Intelligent Transaction Understanding (NLP + TF-IDF + Logistic Regression)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.ui import get_model, load_metrics, render_prediction  # noqa: E402

st.set_page_config(page_title="Smart Categoriser", page_icon="🧠", layout="wide")
st.title("🧠 Intelligent Transaction Understanding")
st.caption("Describe a purchase in plain English or Hinglish. The model predicts the **category** and the likely "
           "**spending context**, with its confidence and limitations. TF-IDF + Logistic Regression.")

model = get_model()
EXAMPLES = ["Late night food delivery", "friend ke saath dinner", "college ke liye books kharidi",
            "monthly bus pass", "bought headphones i did not need", "electricity bill"]

st.session_state.setdefault("sc_text", EXAMPLES[0])
tab_one, tab_batch, tab_quality = st.tabs(["Try it", "Batch", "How good is it?"])

with tab_one:
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        col.button(ex, key=f"ex_{ex}", on_click=lambda e=ex: st.session_state.update(sc_text=e), width="stretch")
    text = st.text_input("Transaction description", key="sc_text", max_chars=200)
    if text.strip():
        pred = model.predict(text)
        render_prediction(pred, expanded_limits=True)
        l, r = st.columns(2)
        with l:
            st.markdown("**Category probabilities (top 3)**")
            st.bar_chart(pd.DataFrame(pred.category_alternatives, columns=["category", "probability"]).set_index("category"))
        with r:
            st.markdown("**Spending-context probabilities**")
            st.bar_chart(pd.Series(pred.context_probabilities, name="probability"))
        st.info("Output format for this input → "
                f"**{pred.category}**, {pred.context_label.lower()} · confidence {pred.category_confidence:.0%}. "
                "The context is a suggestion about the *type* of purchase, never a verdict on you.")
    else:
        st.caption("Type a description above.")

with tab_batch:
    st.write("One description per line.")
    raw = st.text_area("Descriptions", "Zomato order at 2am\nmetro card recharge\nbirthday gift for dad\nasdfgh", height=160)
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()][:200]
    if lines:
        preds = model.predict_many(lines)
        table = pd.DataFrame({
            "description": lines, "category": [p.category for p in preds],
            "confidence": [round(p.category_confidence, 3) for p in preds],
            "level": [p.confidence_level for p in preds],
            "context": [p.context_label for p in preds],
            "suggest?": ["yes" if p.suggest else "no - choose manually" for p in preds]})
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button("⬇️ Download CSV", table.to_csv(index=False).encode(), "predictions.csv", "text/csv")

with tab_quality:
    m = load_metrics()
    st.write(f"Model trained on **{model.meta.get('n_category_rows', '?')}** rows "
             f"({model.meta.get('trained_at', 'unknown date')}).")
    if not m:
        st.warning("No evaluation file found. Run `python scripts/train_model.py` to generate one.")
    else:
        st.error("All data is synthetic / hand-written. These numbers show the pipeline works - they do **not** prove "
                 "accuracy on your real descriptions.")
        rows = [
            ("Grouped CV (unseen phrases)", "Category", m["cv_category"]), ("Grouped CV (unseen phrases)", "Context", m["cv_context"]),
            ("Hand-written test set", "Category", m["test_category"]), ("Hand-written test set", "Context", m["test_context"]),
        ]
        st.dataframe(pd.DataFrame([{"Evaluation": a, "Target": b, "Accuracy": v["accuracy"], "Macro-F1": v["macro_f1"],
                                    "Macro-precision": v["macro_precision"], "Macro-recall": v["macro_recall"]} for a, b, v in rows]),
                     hide_index=True, width="stretch")
        st.write("**Does confidence mean anything?** Accuracy by confidence band (category):")
        cal = pd.DataFrame(m["cv_category_calibration"]).rename(columns={"n": "CV n", "accuracy": "CV accuracy"})
        cal2 = pd.DataFrame(m["test_category_calibration"])[["n", "accuracy"]].rename(columns={"n": "test n", "accuracy": "test accuracy"})
        st.dataframe(pd.concat([cal[["band", "CV n", "CV accuracy"]], cal2], axis=1), hide_index=True, width="stretch")
        st.caption("Full per-class report, baselines and confusion matrices: `reports/ml_report.md`. "
                   "If you retrained with your own data in Settings, these figures describe the bundled model only.")
