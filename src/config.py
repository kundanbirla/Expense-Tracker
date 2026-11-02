"""Central configuration: controlled vocabularies, limits and paths."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = MODEL_DIR / "text_classifier.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def db_path() -> Path:
    """Database location. Override with SEA_DB_PATH (used by tests / deployment)."""
    return Path(os.environ.get("SEA_DB_PATH", DATA_DIR / "expenses.db"))


# --- controlled vocabularies (avoid spelling variants splitting reports) ---
CATEGORIES = [
    "Food", "Travel", "Shopping", "Education", "Bills",
    "Entertainment", "Health", "Household", "Other",
]
INCOME_CATEGORIES = ["Salary", "Freelance", "Gift", "Other Income"]

# expense  -> counts as spending
# income   -> money received (never counted as an expense)
# refund   -> money returned for an earlier expense; REDUCES net spending of its category
# (transfers between own accounts are intentionally not modelled: do not record them)
TRANSACTION_TYPES = ["expense", "income", "refund"]

WASTEFUL_OPTIONS = ["Yes", "No", "Not sure"]
NEED_WANT_OPTIONS = ["Need", "Want", "Both"]
PAYMENT_METHODS = ["UPI", "Cash", "Card", "Bank transfer", "Other"]
CONTEXT_OPTIONS = ["Planned", "Necessary/urgent", "Impulse", "Other"]

# --- limits ---
MAX_AMOUNT_PAISE = 10_000_000 * 100  # Rs 1 crore per transaction
MAX_DESCRIPTION_LEN = 200
MAX_NOTES_LEN = 500
MIN_DATE_YEAR = 2000

# --- insight thresholds ---
MIN_EXPENSES_FOR_INSIGHTS = 10
