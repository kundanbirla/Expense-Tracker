"""Unusual-expense detection.

Two detectors are provided so the ML option can be compared against a transparent baseline
(see ``scripts/evaluate_anomaly.py``). The word "anomaly" here only means *unusually large for
this category in your own history* - never fraud.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Defaults chosen on separate tuning seeds (100-104) - not the evaluation seeds (0-9). With 1.5 x IQR
# precision was only ~0.3 on mixed-price categories (grocery run vs. tea); 3 x IQR ("far-out" fence) balances it.
DEFAULT_IQR_K = 3.0
DEFAULT_RATIO_FLOOR = 3.0

RESULT_COLUMNS = ["id", "is_anomaly", "threshold_paise", "history_n", "reason"]


def detect_rule_based(df: pd.DataFrame, min_history: int = 8, iqr_k: float = DEFAULT_IQR_K,
                      ratio_floor: float = DEFAULT_RATIO_FLOOR) -> pd.DataFrame:
    """Flag an expense that is above ``max(Q3 + k*IQR, ratio_floor * median)`` of the *earlier*
    expenses in the same category. Only past data is used (no look-ahead), and nothing is
    flagged until the category has ``min_history`` earlier entries ("not enough data yet").
    """
    ex = df[df["transaction_type"] == "expense"].sort_values(["transaction_date", "id"])
    rows = []
    for cat, g in ex.groupby("category"):
        amts = g["amount_paise"].to_numpy(dtype=float)
        ids = g["id"].to_numpy()
        for i in range(len(g)):
            hist = amts[:i]
            if len(hist) < min_history:
                rows.append((int(ids[i]), False, np.nan, len(hist), "not enough history"))
                continue
            q1, q3 = np.percentile(hist, [25, 75])
            thr = max(q3 + iqr_k * (q3 - q1), ratio_floor * float(np.median(hist)))
            flag = bool(amts[i] > thr)
            reason = (f"above usual {cat} range (limit ≈ ₹{thr / 100:,.0f}, "
                      f"from {len(hist)} earlier entries)") if flag else ""
            rows.append((int(ids[i]), flag, thr, len(hist), reason))
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def detect_isolation_forest(df: pd.DataFrame, contamination: float = 0.03, min_rows: int = 40,
                            random_state: int = 42) -> pd.DataFrame:
    """Experimental: IsolationForest on [log amount, amount relative to category median, weekday].

    Fitted on the user's own expenses (unsupervised, in-sample). Only *high* outliers are reported,
    because a tiny expense is not something to alert on.
    """
    ex = df[df["transaction_type"] == "expense"].sort_values(["transaction_date", "id"])
    if len(ex) < min_rows:
        return pd.DataFrame({"id": ex["id"].astype(int), "is_anomaly": False, "threshold_paise": np.nan,
                             "history_n": len(ex), "reason": "not enough data"}, columns=RESULT_COLUMNS)
    amt = ex["amount_paise"].to_numpy(dtype=float)
    med = ex.groupby("category")["amount_paise"].transform("median").to_numpy(dtype=float)
    feats = np.column_stack([np.log1p(amt / 100), np.log(amt / med), ex["transaction_date"].dt.dayofweek])
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
    pred = model.fit_predict(feats)
    flag = (pred == -1) & (amt > med)
    reason = np.where(flag, "isolated by IsolationForest and larger than your typical amount in this category", "")
    return pd.DataFrame({"id": ex["id"].astype(int).to_numpy(), "is_anomaly": flag,
                         "threshold_paise": np.nan, "history_n": len(ex), "reason": reason},
                        columns=RESULT_COLUMNS)
