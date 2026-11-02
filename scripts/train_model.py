"""Train, evaluate and save the transaction-text model.

    python scripts/train_model.py

Outputs
  models/text_classifier.joblib   the model used by the app
  models/metrics.json             numbers shown in the app
  reports/ml_report.md            human-readable evaluation (+ confusion-matrix PNGs)
  data/training_descriptions.csv  the exact (augmented) training rows, for inspection

Evaluation protocol (no number here comes from data the model was trained on):
  A. 5-fold GroupKFold cross-validation on the training rows, grouped by seed phrase, so a phrase
     and its augmented variants are never on both sides of a split.
  B. Independent hand-written test set (data/handwritten_test_set.csv), checked for zero overlap
     with the training seeds.
Baselines: majority class, Complement Naive Bayes (same TF-IDF features).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, confusion_matrix,  # noqa: E402
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.naive_bayes import ComplementNB  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src import config  # noqa: E402
from src.text_classifier import (HIGH_CONF, MEDIUM_CONF, TransactionTextModel, build_pipeline,  # noqa: E402
                                 preprocess)
from src.training_data import (build_training_frame, load_handwritten_test, normalise_for_overlap,  # noqa: E402
                               validate_seeds)

CATS = config.CATEGORIES
CTXS = ["essential", "discretionary", "mixed"]


def scores(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float((np.array(y_true) == np.array(y_pred)).mean()), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def build_nb_pipeline() -> Pipeline:
    return Pipeline([("tfidf", build_pipeline().named_steps["tfidf"]), ("clf", ComplementNB(alpha=0.3))])


def oof_predictions(frame: pd.DataFrame, target: str, make=build_pipeline):
    """Out-of-fold predictions + probabilities with group-aware CV."""
    X, y, groups = frame["text"].to_numpy(), frame[target].to_numpy(), frame["group"].to_numpy()
    classes = sorted(set(y))
    proba = np.zeros((len(frame), len(classes)))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        pipe = make().fit(X[tr], y[tr])
        cols = [classes.index(c) for c in pipe.named_steps["clf"].classes_]
        proba[np.ix_(te, cols)] = pipe.predict_proba(X[te])
    pred = np.array(classes)[proba.argmax(1)]
    return y, pred, proba, classes


def baselines(train: pd.DataFrame, test: pd.DataFrame, target: str) -> dict:
    out = {}
    dummy = DummyClassifier(strategy="most_frequent").fit(train["text"], train[target])
    out["majority_class"] = scores(test[target], dummy.predict(test["text"]))
    nb = build_nb_pipeline()
    nb.fit(train["text"], train[target])
    out["tfidf_complement_nb"] = scores(test[target], nb.predict(test["text"]))
    return out


def calibration(correct: np.ndarray, conf: np.ndarray) -> list[dict]:
    bands = [("low (<%.2f)" % MEDIUM_CONF, 0, MEDIUM_CONF), ("medium", MEDIUM_CONF, HIGH_CONF),
             ("high (>=%.2f)" % HIGH_CONF, HIGH_CONF, 1.01)]
    rows = []
    for name, lo, hi in bands:
        m = (conf >= lo) & (conf < hi)
        rows.append({"band": name, "n": int(m.sum()),
                     "accuracy": round(float(correct[m].mean()), 3) if m.any() else None,
                     "mean_confidence": round(float(conf[m].mean()), 3) if m.any() else None})
    return rows


def save_confusion(y_true, y_pred, labels, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=labels), display_labels=labels).plot(
        ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    validate_seeds()
    frame = build_training_frame()
    hw = load_handwritten_test()
    # --- leakage check: no hand-written test phrase may equal a training row
    train_norm = {normalise_for_overlap(t) for t in frame["text"]}
    overlap = [t for t in hw["text"] if normalise_for_overlap(t) in train_norm]
    assert not overlap, f"test/training overlap: {overlap}"
    frame.to_csv(config.DATA_DIR / "training_descriptions.csv", index=False)
    print(f"training rows: {len(frame)} from {frame['group'].nunique()} seed phrases; test rows: {len(hw)}")

    metrics: dict = {"n_train_rows": len(frame), "n_seed_phrases": int(frame["group"].nunique()),
                     "n_handwritten_test": len(hw)}

    # ---- A. grouped cross-validation
    y_c, p_c, proba_c, cls_c = oof_predictions(frame, "category")
    y_x, p_x, proba_x, _ = oof_predictions(frame, "context")
    metrics["cv_category"] = scores(y_c, p_c)
    metrics["cv_context"] = scores(y_x, p_x)
    conf_oof = proba_c.max(1)
    metrics["cv_category_calibration"] = calibration(y_c == p_c, conf_oof)
    metrics["cv_baseline_nb_category"] = scores(*oof_predictions(frame, "category", build_nb_pipeline)[:2])
    metrics["cv_baseline_nb_context"] = scores(*oof_predictions(frame, "context", build_nb_pipeline)[:2])
    print("CV NB", metrics["cv_baseline_nb_category"], metrics["cv_baseline_nb_context"])
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        m = conf_oof >= t
        print(f"CV conf>={t}: n={m.sum()} acc={(y_c == p_c)[m].mean():.3f}")
    print("CV category", metrics["cv_category"], "\nCV context ", metrics["cv_context"])
    print("CV calibration", metrics["cv_category_calibration"])

    # ---- B. final model + hand-written test set
    model = TransactionTextModel().fit(frame["text"], frame["category"], frame["text"], frame["context"])
    preds = model.predict_many(hw["text"].tolist())
    hw_cat = [p.category for p in preds]
    hw_ctx = [p.context for p in preds]
    metrics["test_category"] = scores(hw["category"], hw_cat)
    metrics["test_context"] = scores(hw["context"], hw_ctx)
    metrics["test_category_calibration"] = calibration(
        np.array(hw_cat) == hw["category"].to_numpy(), np.array([p.category_confidence for p in preds]))
    metrics["baselines_test_category"] = baselines(frame, hw, "category")
    metrics["baselines_test_context"] = baselines(frame, hw, "context")
    levels = pd.Series([p.confidence_level for p in preds]).value_counts().to_dict()
    metrics["test_confidence_level_counts"] = {k: int(v) for k, v in levels.items()}
    print("TEST category", metrics["test_category"], "\nTEST context ", metrics["test_context"])
    print("baselines", metrics["baselines_test_category"])
    print("test calibration", metrics["test_category_calibration"])

    errors = hw.assign(pred_category=hw_cat, pred_context=hw_ctx,
                       confidence=[round(p.category_confidence, 2) for p in preds])
    errors = errors[(errors["category"] != errors["pred_category"]) | (errors["context"] != errors["pred_context"])]
    metrics["n_test_errors"] = int(len(errors))

    # ---- persist
    model.meta.update({"metrics_file": "metrics.json"})
    model.save(config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    save_confusion(hw["category"], hw_cat, CATS, "Category - hand-written test set", ROOT / "reports/confusion_category.png")
    save_confusion(hw["context"], hw_ctx, CTXS, "Spending context - hand-written test set", ROOT / "reports/confusion_context.png")

    # ---- markdown report
    def table(d: dict) -> str:
        return "| " + " | ".join(d) + " |\n|" + "---|" * len(d) + "\n| " + " | ".join(str(v) for v in d.values()) + " |"

    lines = [
        "# ML evaluation report - Intelligent Transaction Understanding", "",
        "Model: TF-IDF (word 1-2-grams + char 2-5-grams) -> Logistic Regression (two heads: category, spending context).", "",
        f"Training rows: **{len(frame)}** (augmented from **{metrics['n_seed_phrases']}** seed phrases). "
        f"Hand-written test rows: **{len(hw)}**.", "",
        "> **Read this first.** All data is synthetic / hand-written by the project author. These numbers show the "
        "pipeline works and generalises to unseen phrasings; they do **not** prove accuracy on real users' descriptions.", "",
        "## A. Grouped 5-fold cross-validation (phrase variants never split across folds)", "",
        "Category:", "", table(metrics["cv_category"]), "", "Context:", "", table(metrics["cv_context"]), "",
        "Complement NB on the same folds - category: "
        f"acc {metrics['cv_baseline_nb_category']['accuracy']}, macro-F1 {metrics['cv_baseline_nb_category']['macro_f1']}; "
        f"context: acc {metrics['cv_baseline_nb_context']['accuracy']}, macro-F1 {metrics['cv_baseline_nb_context']['macro_f1']}.", "",
        "**Why CV is much lower than the test set:** CV holds out whole seed phrases (e.g. every variant of 'mattress'), "
        "so many held-out phrases contain words never seen in training - a harsh, realistic 'new vocabulary' test. The "
        "hand-written test set was written by the same author and re-uses many everyday keywords, so it is the optimistic end. "
        "Real-world accuracy should be expected between the two.", "",
        "## B. Independent hand-written test set", "",
        "Category:", "", table(metrics["test_category"]), "", "Context:", "", table(metrics["test_context"]), "",
        "Baselines (category, same test set):", "",
        "| model | accuracy | macro-F1 |", "|---|---|---|",
    ]
    lines.append(f"| logistic regression (ours) | {metrics['test_category']['accuracy']} | {metrics['test_category']['macro_f1']} |")
    for k, v in metrics["baselines_test_category"].items():
        lines.append(f"| {k} | {v['accuracy']} | {v['macro_f1']} |")
    lines += ["", "Baselines (context, same test set):", "", "| model | accuracy | macro-F1 |", "|---|---|---|"]
    lines.append(f"| logistic regression (ours) | {metrics['test_context']['accuracy']} | {metrics['test_context']['macro_f1']} |")
    for k, v in metrics["baselines_test_context"].items():
        lines.append(f"| {k} | {v['accuracy']} | {v['macro_f1']} |")
    lines += ["", "## Confidence vs. reality (category head)", "",
              "If the confidence is meaningful, accuracy should rise with the band.", "",
              "| band | CV n | CV accuracy | test n | test accuracy |", "|---|---|---|---|---|"]
    for a, b in zip(metrics["cv_category_calibration"], metrics["test_category_calibration"]):
        lines.append(f"| {a['band']} | {a['n']} | {a['accuracy']} | {b['n']} | {b['accuracy']} |")
    lines += ["", "## Per-class report (test set, category)", "", "```",
              classification_report(hw["category"], hw_cat, labels=CATS, zero_division=0), "```", "",
              "## Test-set mistakes", "", "| text | true cat | pred cat | true ctx | pred ctx | conf |", "|---|---|---|---|---|---|"]
    for _, r in errors.iterrows():
        lines.append(f"| {r['text']} | {r['category']} | {r['pred_category']} | {r['context']} | {r['pred_context']} | {r['confidence']} |")
    lines += ["", "![category confusion](confusion_category.png)", "", "![context confusion](confusion_context.png)", "",
              "## Known limitations", "",
              "- Synthetic / hand-written data; real descriptions (merchant codes, abbreviations, other languages) will be harder.",
              "- 'Context' labels are the author's subjective view of typical need/want; the same purchase can be a need for one person and a want for another.",
              "- Romanised Hinglish only; Devanagari script and other languages are not handled.",
              "- Bag-of-words cannot understand negation or sarcasm and ignores amount, time and who paid.",
              "- Next step with real data: collect the user's accepted/corrected suggestions (stored by the app) and retrain; "
              "compare against sentence-embedding features (e.g. multilingual MiniLM) before switching."]
    (ROOT / "reports/ml_report.md").write_text("\n".join(lines))
    print("saved model, metrics and report")


if __name__ == "__main__":
    main()
