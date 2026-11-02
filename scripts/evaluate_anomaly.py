"""Compare the transparent rule baseline with IsolationForest on synthetic data with injected anomalies.

    python scripts/evaluate_anomaly.py

Why synthetic: real personal data has no ground truth for "unusual". Here each generated history has ~3% of
expenses multiplied 4-8x (known labels), so precision / recall can be measured over 10 random histories.
Bills are excluded (fixed, heterogeneous recurring payments - same rule as the app's insights).
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.anomaly_detection import detect_isolation_forest, detect_rule_based  # noqa: E402
from src.synthetic_data import generate_transactions  # noqa: E402


def make_frame(seed: int) -> pd.DataFrame:
    rows = generate_transactions(end=date(2026, 10, 11), days=365, seed=seed, anomaly_rate=0.03)
    df = pd.DataFrame(rows)
    df = df[(df["transaction_type"] == "expense") & (df["category"] != "Bills")].reset_index(drop=True)
    df["id"] = df.index + 1
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["amount_paise"] = (df["amount"] * 100).round().astype("int64")
    return df


def score(df: pd.DataFrame, flags: pd.DataFrame, evaluable: pd.Series | None = None) -> tuple[int, int, int]:
    m = df.merge(flags[["id", "is_anomaly"]], on="id", how="left").fillna({"is_anomaly": False})
    if evaluable is not None:
        m = m[m["id"].isin(evaluable)]
    tp = int((m["is_anomaly"] & m["is_injected_anomaly"]).sum())
    fp = int((m["is_anomaly"] & ~m["is_injected_anomaly"]).sum())
    fn = int((~m["is_anomaly"] & m["is_injected_anomaly"]).sum())
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main() -> None:
    tot = {"rule": [0, 0, 0], "rule (rows with enough history)": [0, 0, 0], "isolation_forest": [0, 0, 0]}
    n_rows = n_anom = 0
    for seed in range(10):
        df = make_frame(seed)
        n_rows += len(df)
        n_anom += int(df["is_injected_anomaly"].sum())
        rule = detect_rule_based(df)
        iso = detect_isolation_forest(df, contamination=0.03)
        ok_ids = rule.loc[rule["history_n"] >= 8, "id"]
        for key, args in (("rule", (df, rule)), ("rule (rows with enough history)", (df, rule, ok_ids)),
                          ("isolation_forest", (df, iso))):
            for i, v in enumerate(score(*args)):
                tot[key][i] += v
    lines = [f"Evaluated on 10 synthetic one-year histories: {n_rows} expenses, {n_anom} injected anomalies.", "",
             "| detector | precision | recall | F1 | TP | FP | FN |", "|---|---|---|---|---|---|---|"]
    for k, (tp, fp, fn) in tot.items():
        p, r, f = prf(tp, fp, fn)
        lines.append(f"| {k} | {p:.2f} | {r:.2f} | {f:.2f} | {tp} | {fp} | {fn} |")
    lines += ["", "Caveats: injected anomalies are simple multiples of a normal amount, which favours both detectors; the rule "
              "baseline cannot flag the first 8 entries of a category (so its plain recall is capped), and IsolationForest is fitted "
              "in-sample with the true contamination rate supplied. Real data will behave worse - review alerts manually."]
    text = "\n".join(lines)
    print(text)
    (ROOT / "reports/anomaly_report.md").write_text("# Anomaly detector comparison\n\n" + text + "\n")


if __name__ == "__main__":
    main()
