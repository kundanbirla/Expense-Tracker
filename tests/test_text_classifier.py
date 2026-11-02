import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src import training_data as td
from src.text_classifier import (TransactionTextModel, load_or_train, preprocess, train_default_model)


def test_seed_data_is_clean():
    td.validate_seeds()                                     # valid labels, no duplicate phrases
    frame = td.build_training_frame()
    assert set(frame["category"]) == set(td.config.CATEGORIES)
    assert set(frame["context"]) == set(td.CONTEXTS)
    assert frame["group"].nunique() == sum(len(v) for v in td.SEEDS.values())
    assert td.build_training_frame().equals(frame)           # deterministic


def test_handwritten_test_set_does_not_leak_into_training():
    train = {td.normalise_for_overlap(t) for t in td.build_training_frame()["text"]}
    hw = td.load_handwritten_test()
    assert len(hw) >= 60
    assert not [t for t in hw["text"] if td.normalise_for_overlap(t) in train]
    assert set(hw["category"]) <= set(td.config.CATEGORIES) and set(hw["context"]) <= set(td.CONTEXTS)


def test_preprocess_drops_amounts_and_punctuation():
    assert preprocess("Rs. 250 for Lunch!!") == "for lunch"
    assert preprocess("₹1,200.50 Uber-ride") == "uber ride"
    assert preprocess("  ") == ""


def test_user_examples_from_the_brief(text_model):
    p = text_model.predict("Late night food delivery")
    assert p.category == "Food" and p.context == "discretionary"
    assert p.confidence_level in ("high", "medium") and p.suggest
    assert p.limitations and p.top_terms                      # limitations + explanation always present
    p = text_model.predict("college ke liye books kharidi")
    assert p.category == "Education" and p.context == "essential"
    p = text_model.predict("friend ke saath dinner")
    assert p.category == "Food" and p.context == "discretionary"


def test_probabilities_are_valid(text_model):
    p = text_model.predict("metro card recharge")
    assert 0 < p.category_confidence <= 1
    assert sum(p.context_probabilities.values()) == pytest.approx(1)
    assert [a[1] for a in p.category_alternatives] == sorted((a[1] for a in p.category_alternatives), reverse=True)
    assert p.category_alternatives[0][0] == p.category


def test_gibberish_empty_and_unknown_words_are_flagged_low(text_model):
    for text in ["", "   ", "asdfgh qwerty", "₹500", "!!!"]:
        p = text_model.predict(text)
        assert p.confidence_level == "low" and not p.suggest, text
        assert any("guess" in n or "empty" in n or "Low confidence" in n for n in p.limitations)
    p = text_model.predict("lunch with zxqvb")
    assert any("never seen" in n and "zxqvb" in n for n in p.limitations)


def test_amount_in_text_does_not_change_prediction(text_model):
    a, b = text_model.predict("petrol"), text_model.predict("petrol Rs 5000")
    assert a.category == b.category
    assert a.category_confidence == pytest.approx(b.category_confidence)


def test_predict_many_matches_predict(text_model):
    texts = ["zomato order", "electricity bill", "gym membership"]
    assert [p.category for p in text_model.predict_many(texts)] == [text_model.predict(t).category for t in texts]
    assert text_model.predict_many([]) == []


def test_quality_on_independent_handwritten_set(text_model):
    """Regression guard. Floors are deliberately below the measured values (see reports/ml_report.md)."""
    hw = td.load_handwritten_test()
    preds = text_model.predict_many(hw["text"].tolist())
    f1_cat = f1_score(hw["category"], [p.category for p in preds], average="macro")
    f1_ctx = f1_score(hw["context"], [p.context for p in preds], average="macro")
    assert f1_cat >= 0.85 and f1_ctx >= 0.75
    # confidence must carry information: high-confidence predictions are more accurate than low ones
    ok = np.array([p.category == t for p, t in zip(preds, hw["category"])])
    hi = np.array([p.confidence_level == "high" for p in preds])
    assert ok[hi].mean() >= 0.9


def test_save_and_load_roundtrip(text_model, tmp_path):
    path = text_model.save(tmp_path / "m.joblib")
    loaded = TransactionTextModel.load(path)
    for t in ["zomato order", "bus pass", "random words here"]:
        a, b = text_model.predict(t), loaded.predict(t)
        assert (a.category, a.context) == (b.category, b.context)
        assert a.category_confidence == pytest.approx(b.category_confidence)


def test_load_or_train_recovers_from_missing_and_corrupt_file(tmp_path):
    path = tmp_path / "m.joblib"
    assert load_or_train(path).predict("electricity bill").category == "Bills"     # trains + saves
    assert path.exists()
    path.write_bytes(b"not a model")
    assert load_or_train(path).predict("electricity bill").category == "Bills"     # corrupt -> retrain


def test_retraining_with_user_rows_learns_new_vocabulary():
    extra = pd.DataFrame({"text": ["chaiwala bhaiya"] * 6 + ["xerox wala"] * 6,
                          "category": ["Food"] * 6 + ["Education"] * 6,
                          "context": ["mixed"] * 6 + [None] * 6})        # None -> category head only
    m = train_default_model(extra=extra)
    assert m.predict("chaiwala bhaiya").category == "Food"
    assert m.predict("xerox wala").category == "Education"
    assert m.meta["n_category_rows"] > m.meta["n_context_rows"]
