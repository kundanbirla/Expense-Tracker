from datetime import date, datetime

import pytest

from src.validation import ValidationError, parse_amount_to_paise, validate_transaction

TODAY = date(2026, 10, 11)


def base(**kw):
    d = {"transaction_date": "2026-10-10", "amount": "250", "category": "Food", "description": "Lunch with friends"}
    d.update(kw)
    return d


@pytest.mark.parametrize("raw,expected", [
    ("250", 25000), (250, 25000), (99.5, 9950), ("₹1,250.50", 125050), ("Rs. 80", 8000), ("0.01", 1),
    ("10.005", 1001),  # more than 2 decimals: rounded half-up
    ("1,00,000", 10_000_000),  # Indian grouping
])
def test_amount_parsing(raw, expected):
    assert parse_amount_to_paise(raw) == expected


@pytest.mark.parametrize("bad", ["", "   ", None, "abc", "0", 0, "-5", -5, "nan", float("nan"), float("inf"),
                                 "1e400", True, "12.3.4", "99999999999"])
def test_bad_amounts_rejected(bad):
    with pytest.raises(ValueError):
        parse_amount_to_paise(bad)


def test_valid_expense_cleaned():
    out = validate_transaction(base(wasteful_label="yes", need_want="WANT"), TODAY)
    assert out["amount_paise"] == 25000
    assert out["transaction_type"] == "expense"
    assert out["wasteful_label"] == "Yes" and out["need_want"] == "Want"
    assert out["transaction_date"] == "2026-10-10"


@pytest.mark.parametrize("field,value", [
    ("transaction_date", "2026-13-40"), ("transaction_date", ""), ("transaction_date", "2026-10-12"),  # future
    ("transaction_date", "1999-01-01"), ("amount", ""), ("amount", "-3"), ("category", "Groceries"), ("category", ""),
    ("wasteful_label", "maybe"), ("payment_method", "Bitcoin"), ("transaction_type", "transfer"),
    ("description", "x" * 201),
])
def test_invalid_fields_rejected(field, value):
    with pytest.raises(ValidationError) as exc:
        validate_transaction(base(**{field: value}), TODAY)
    assert field in exc.value.errors


def test_multiple_errors_reported_together():
    with pytest.raises(ValidationError) as exc:
        validate_transaction({"amount": "", "category": ""}, TODAY)
    assert {"amount", "category", "transaction_date"} <= set(exc.value.errors)


def test_income_uses_income_categories_and_drops_reflection_labels():
    out = validate_transaction(base(transaction_type="income", category="Salary", wasteful_label="Yes", need_want="Want"), TODAY)
    assert out["wasteful_label"] is None and out["need_want"] is None
    with pytest.raises(ValidationError):
        validate_transaction(base(transaction_type="income", category="Food"), TODAY)
    with pytest.raises(ValidationError):  # and expenses cannot use income categories
        validate_transaction(base(category="Salary"), TODAY)


def test_date_objects_and_timestamps_accepted():
    assert validate_transaction(base(transaction_date=date(2026, 10, 1)), TODAY)["transaction_date"] == "2026-10-01"
    assert validate_transaction(base(transaction_date=datetime(2026, 10, 1, 15, 0)), TODAY)["transaction_date"] == "2026-10-01"


def test_text_is_cleaned():
    out = validate_transaction(base(description="  Lunch \n  with\tfriends\x00 "), TODAY)
    assert out["description"] == "Lunch with friends"


def test_prediction_bookkeeping_sanitised():
    out = validate_transaction(base(predicted_category="Nonsense", predicted_confidence=5), TODAY)
    assert out["predicted_category"] is None and out["predicted_confidence"] is None
    out = validate_transaction(base(predicted_category="Food", predicted_confidence=0.8), TODAY)
    assert out["predicted_category"] == "Food" and out["predicted_confidence"] == 0.8
