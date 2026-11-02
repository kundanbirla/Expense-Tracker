"""Input validation. Every write path goes through ``validate_transaction``.

Amounts are parsed with ``Decimal`` and stored as integer paise, so totals are
exact (no floating-point drift).
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Mapping

from . import config


class ValidationError(ValueError):
    """Raised with a ``{field: message}`` dict so the UI can show clear messages."""

    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))


_CURRENCY_PREFIX = re.compile(r"^(?:₹|rs\.?|inr)\s*", re.IGNORECASE)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def parse_amount_to_paise(value: Any) -> int:
    """Parse rupees (``250``, ``"₹1,250.50"``) into integer paise.

    More than two decimals are rounded half-up to the nearest paisa.
    Raises ``ValueError`` for blank, non-numeric, non-finite, zero, negative or huge values.
    """
    if value is None or isinstance(value, bool):
        raise ValueError("Amount is required and must be a number.")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Amount must be a finite number.")
        dec = Decimal(str(value))
    elif isinstance(value, (int, Decimal)):
        dec = Decimal(value)
    elif isinstance(value, str):
        text = _CURRENCY_PREFIX.sub("", value.strip()).replace(",", "").replace(" ", "")
        if not text:
            raise ValueError("Amount is required.")
        try:
            dec = Decimal(text)
        except InvalidOperation:
            raise ValueError("Amount must be a valid number, e.g. 250 or 99.50.") from None
    else:
        # numpy / pandas scalars
        try:
            dec = Decimal(str(value))
        except InvalidOperation:
            raise ValueError("Amount must be a valid number.") from None

    if not dec.is_finite():
        raise ValueError("Amount must be a finite number.")
    if dec <= 0:
        raise ValueError("Amount must be greater than zero.")
    if dec * 100 > config.MAX_AMOUNT_PAISE:
        raise ValueError(f"Amount is too large (maximum is ₹{config.MAX_AMOUNT_PAISE // 100:,}).")
    paise = int((dec * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    if paise <= 0:
        raise ValueError("Amount must be at least ₹0.01.")
    return paise


def parse_date(value: Any) -> date:
    if isinstance(value, datetime):  # also covers pandas.Timestamp
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            pass
    raise ValueError("Date must be a valid date (YYYY-MM-DD).")


def _clean_text(value: Any, max_len: int, field: str, errors: dict[str, str]) -> str:
    if value is None:
        return ""
    text = _CONTROL_CHARS.sub("", str(value))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        errors[field] = f"Must be at most {max_len} characters."
    return text


def _choice(value: Any, options: list[str], field: str, errors: dict[str, str],
            required: bool = False) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()) or _is_nan(value):
        if required:
            errors[field] = "This field is required."
        return None
    lookup = {o.lower(): o for o in options}
    key = str(value).strip().lower()
    if key not in lookup:
        errors[field] = f"Must be one of: {', '.join(options)}."
        return None
    return lookup[key]


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def validate_transaction(data: Mapping[str, Any], today: date | None = None) -> dict[str, Any]:
    """Return a cleaned transaction dict (amount in paise) or raise ValidationError."""
    today = today or date.today()
    errors: dict[str, str] = {}

    ttype = _choice(data.get("transaction_type") or "expense", config.TRANSACTION_TYPES,
                    "transaction_type", errors, required=True)

    # date
    tx_date: date | None = None
    try:
        tx_date = parse_date(data.get("transaction_date"))
        if tx_date > today:
            errors["transaction_date"] = "Date cannot be in the future."
        elif tx_date.year < config.MIN_DATE_YEAR:
            errors["transaction_date"] = f"Date cannot be before {config.MIN_DATE_YEAR}."
    except ValueError as exc:
        errors["transaction_date"] = str(exc)

    # amount
    paise = None
    try:
        paise = parse_amount_to_paise(data.get("amount"))
    except ValueError as exc:
        errors["amount"] = str(exc)

    # category must be consistent with the transaction type
    allowed = config.INCOME_CATEGORIES if ttype == "income" else config.CATEGORIES
    category = _choice(data.get("category"), allowed, "category", errors, required=True)

    description = _clean_text(data.get("description"), config.MAX_DESCRIPTION_LEN, "description", errors)
    notes = _clean_text(data.get("notes"), config.MAX_NOTES_LEN, "notes", errors)

    wasteful = _choice(data.get("wasteful_label"), config.WASTEFUL_OPTIONS, "wasteful_label", errors)
    need_want = _choice(data.get("need_want"), config.NEED_WANT_OPTIONS, "need_want", errors)
    payment = _choice(data.get("payment_method"), config.PAYMENT_METHODS, "payment_method", errors)
    context = _choice(data.get("context"), config.CONTEXT_OPTIONS, "context", errors)

    # ML bookkeeping (optional)
    pred_cat = data.get("predicted_category")
    pred_cat = pred_cat if isinstance(pred_cat, str) and pred_cat in config.CATEGORIES else None
    pred_conf = data.get("predicted_confidence")
    try:
        pred_conf = float(pred_conf) if pred_conf is not None and not _is_nan(pred_conf) else None
        if pred_conf is not None and not (0.0 <= pred_conf <= 1.0):
            pred_conf = None
    except (TypeError, ValueError):
        pred_conf = None

    if errors:
        raise ValidationError(errors)

    # Reflection labels only make sense for money the user spent.
    if ttype != "expense":
        wasteful = need_want = context = None

    return {
        "transaction_date": tx_date.isoformat(),
        "amount_paise": paise,
        "transaction_type": ttype,
        "category": category,
        "description": description,
        "wasteful_label": wasteful,
        "need_want": need_want,
        "payment_method": payment,
        "context": context,
        "notes": notes,
        "predicted_category": pred_cat,
        "predicted_confidence": pred_conf,
    }
